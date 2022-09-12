from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import fnmatch
import utils
import losses
import itertools
from datasets.CIFAR import get_training_CIFAR100, get_test_CIFAR100
from folder2lmdb import ImageFolderLMDB
import torchvision.transforms as transforms
from torch.autograd import Variable
from network.models.resnet import resnet56
from network.models.resnetv2 import resnet34
from network.models.vgg import vgg13_bn as Vgg13
from torch.utils.tensorboard import SummaryWriter
from utils import load_pretrained_model
from EncoderDecoderConvLSTM import EncoderDecoderConvLSTM
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--tmodel', type=str, default="resnet56", help='resnet56,resnet34, Vgg13')
parser.add_argument('--data_name', type=str, default="CIFAR100", help='CIFAR100,TinyImageNet')
parser.add_argument('--epochs', type=int, default=12, help='number of total epochs to run')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
parser.add_argument('--N_Teacher', default=6, type=int, help='How many epoch states of the teacher network '
                                                              'are used to predict its next action.')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--train_bs', default=256, type=int, help='learning rate')
parser.add_argument('--test_bs', default=256, type=int, help='learning rate')
args = parser.parse_args()

best_Meta_loss = 100
best_Shortcut_loss = 100

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
print('local_rank', local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if args.train_bs == 512:
    LR = 0.4
elif args.train_bs == 256:
    LR = 0.2
elif args.train_bs == 128:
    LR = 0.1
else:
    LR = 0.05
LR_DECAY_STAGES = [3, 6, 9]
LR_DECAY_RATE = 0.1
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

if args.data_name == 'CIFAR100':
    NUM_CLASSES = 100
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    trainloader = get_training_CIFAR100(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, num_workers=args.num_workers,
                                     batch_size=args.train_bs, shuffle=True, distributed=True)
    testloader = get_test_CIFAR100(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, num_workers=args.num_workers,
                                   batch_size=args.test_bs, shuffle=False, distributed=True)
else:
    NUM_CLASSES = 200
    # traindir = os.path.join('datasets/tiny-imagenet-200/', 'train.lmdb')
    # valdir = os.path.join('datasets/tiny-imagenet-200/', 'val.lmdb')
    traindir = os.path.join('/dev/shm/tiny-imagenet-200/', 'train.lmdb')
    valdir = os.path.join('/dev/shm/tiny-imagenet-200/', 'val.lmdb')
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    trainset = ImageFolderLMDB(traindir,
                               transforms.Compose([
                                   transforms.RandomResizedCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    testset = ImageFolderLMDB(valdir,
                              transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))
    trainloader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, batch_size=args.train_bs,
                                              pin_memory=True, sampler=DistributedSampler(trainset))
    testloader = torch.utils.data.DataLoader(testset, num_workers=args.num_workers, batch_size=args.test_bs,
                                             shuffle=False, pin_memory=True, sampler=DistributedSampler(testset))

ShortcutPrediction = losses.Shortcut(args.N_Teacher, nf=64, future_step=4)
ShortcutPrediction.to(device)
ShortcutPrediction = torch.nn.parallel.DistributedDataParallel(ShortcutPrediction, device_ids=[local_rank],
                                                               output_device=local_rank)

if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':

    if args.tmodel == 'resnet56':
        ExperiencePrediction1 = EncoderDecoderConvLSTM(in_chan=16, nf=32, future_step=4)
        ExperiencePrediction2 = EncoderDecoderConvLSTM(in_chan=32, nf=64, future_step=4)
    else:
        ExperiencePrediction1 = EncoderDecoderConvLSTM(in_chan=128, nf=256, future_step=4)
        ExperiencePrediction2 = EncoderDecoderConvLSTM(in_chan=256, nf=512, future_step=4)

    ExperiencePrediction1.to(device)
    ExperiencePrediction2.to(device)
    ExperiencePrediction1 = torch.nn.parallel.DistributedDataParallel(ExperiencePrediction1, device_ids=[local_rank],
                                                                      output_device=local_rank)
    ExperiencePrediction2 = torch.nn.parallel.DistributedDataParallel(ExperiencePrediction2, device_ids=[local_rank],
                                                                      output_device=local_rank)
    MSE_criterion = nn.MSELoss().cuda()
    MetaLearner_optimizer = optim.SGD(itertools.chain(ExperiencePrediction1.parameters(),
                            ExperiencePrediction2.parameters()), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

LogCosh_criterion = losses.LogCoshLoss().cuda()
Shortcut_optimizer = optim.SGD(ShortcutPrediction.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

if torch.distributed.get_rank() == 0:
    path = os.path.join(args.save_root + args.data_name + '_' + args.tmodel + '_Skill')
    if not os.path.isdir(path):
        os.mkdir(path)
    writer = SummaryWriter(log_dir=path)
    f = open(path + '/' + args.data_name + '_' + args.tmodel + '_Skill' + '.txt', 'a')
    f.write('\nThe dataset used for training is:                ' + str(args.data_name))
    f.write('\nThe Model of Teacher Network is:                 ' + str(args.tmodel))
    f.write('\n==> Preparing data..')
    f.close()

# Training
def train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6):

    if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
        ExperiencePrediction1.train()
        ExperiencePrediction2.train()
        train_Meta_loss = utils.AverageMeter()
    ShortcutPrediction.train()
    train_Shortcut_loss = utils.AverageMeter()

    steps = np.sum(epoch > np.asarray(LR_DECAY_STAGES))
    if steps > 0:
        current_lr = LR * (LR_DECAY_RATE ** steps)
        if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
            utils.set_lr(MetaLearner_optimizer, current_lr)
        utils.set_lr(Shortcut_optimizer, current_lr)
    else:
        current_lr = LR

    if torch.distributed.get_rank() == 0:
        f = open(path + '/' + args.data_name + '_' + args.tmodel + '_Skill' + '.txt', 'a')
        f.write('\nlearning_rate: %s' % str(current_lr))
        f.close()

    for batch_idx, (inputs, targets, index) in enumerate(trainloader):
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        with torch.no_grad():

            bes_outputs, bes_feats = bes_tnet(inputs)
            t_outputs1, t_feats1 = tnet1(inputs)
            t_outputs2, t_feats2 = tnet2(inputs)
            t_outputs3, t_feats3 = tnet3(inputs)
            t_outputs4, t_feats4 = tnet4(inputs)
            t_outputs5, t_feats5 = tnet5(inputs)
            t_outputs6, t_feats6 = tnet6(inputs)
            t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                   t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2)), 2)

            if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
                aft_outputs, aft_feats = aft_tnet(inputs)
                aft_rb1_t, aft_rb2_t = aft_feats["feats"][1], aft_feats["feats"][2]
                rb1_t1, rb2_t1 = t_feats1["feats"][1], t_feats1["feats"][2]
                rb1_t2, rb2_t2 = t_feats2["feats"][1], t_feats2["feats"][2]
                rb1_t3, rb2_t3 = t_feats3["feats"][1], t_feats3["feats"][2]
                rb1_t4, rb2_t4 = t_feats4["feats"][1], t_feats4["feats"][2]
                rb1_t5, rb2_t5 = t_feats5["feats"][1], t_feats5["feats"][2]
                rb1_t6, rb2_t6 = t_feats6["feats"][1], t_feats6["feats"][2]

                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                       rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                       rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1)), 1)

                if args.tmodel == 'resnet56':
                    rb1_t = F.normalize(rb1_t+1e-8)
                    rb2_t = F.normalize(rb2_t+1e-8)
                    aft_rb1_t = F.normalize(aft_rb1_t+1e-8)
                    aft_rb2_t = F.normalize(aft_rb2_t+1e-8)

        if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
            MetaLearner_optimizer.zero_grad()
            rb1_t, rb2_t, aft_rb1_t, aft_rb2_t = Variable(rb1_t), Variable(rb2_t), Variable(aft_rb1_t), Variable(aft_rb2_t)
            ExperiencePrediction_rb1_t = ExperiencePrediction1(rb1_t)
            ExperiencePrediction_rb2_t = ExperiencePrediction2(rb2_t)
            Meta_hard_loss_1 = MSE_criterion(ExperiencePrediction_rb1_t, aft_rb1_t)
            Meta_hard_loss_2 = MSE_criterion(ExperiencePrediction_rb2_t, aft_rb2_t)
            Meta_loss = Meta_hard_loss_1 + Meta_hard_loss_2
            Meta_loss.backward()
            utils.clip_gradient(MetaLearner_optimizer, 0.1)
            MetaLearner_optimizer.step()
            train_Meta_loss.update(Meta_loss.cpu().detach().numpy().mean(), inputs.size(0))

        # Here, the short-cut knowledge of the teacher network is acquired by training the meta-learner.
        Shortcut_optimizer.zero_grad()
        t_outputs, bes_outputs = Variable(t_outputs), Variable(bes_outputs)
        Shortcut_t_outputs = ShortcutPrediction(t_outputs)

        Shortcut_loss = LogCosh_criterion(torch.exp(torch.nn.functional.normalize(Shortcut_t_outputs+1e-8, p=2, dim=1)),
                                          torch.exp(torch.nn.functional.normalize(bes_outputs+1e-8, p=2, dim=1)))
        Shortcut_loss.backward()
        utils.clip_gradient(Shortcut_optimizer, 0.1)
        Shortcut_optimizer.step()
        train_Shortcut_loss.update(Shortcut_loss.cpu().detach().numpy().mean(), inputs.size(0))

    if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
        return train_Meta_loss.avg, train_Shortcut_loss.avg
    else:
        return 00.00, train_Shortcut_loss.avg



def test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6):

    if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
        ExperiencePrediction1.eval()
        ExperiencePrediction2.eval()
        test_Meta_loss = utils.AverageMeter()
    ShortcutPrediction.eval()
    test_Shortcut_loss = utils.AverageMeter()
    
    for batch_idx, (inputs, _, index) in enumerate(testloader):
        inputs = inputs.cuda()
        inputs = Variable(inputs)

        with torch.no_grad():

            bes_outputs, bes_feats = bes_tnet(inputs)
            t_outputs1, t_feats1 = tnet1(inputs)
            t_outputs2, t_feats2 = tnet2(inputs)
            t_outputs3, t_feats3 = tnet3(inputs)
            t_outputs4, t_feats4 = tnet4(inputs)
            t_outputs5, t_feats5 = tnet5(inputs)
            t_outputs6, t_feats6 = tnet6(inputs)
            t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                   t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2)), 2)
            Shortcut_t_outputs = ShortcutPrediction(t_outputs)

            if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
                aft_outputs, aft_feats = aft_tnet(inputs)
                aft_rb1_t, aft_rb2_t = aft_feats["feats"][1], aft_feats["feats"][2]
                rb1_t1, rb2_t1 = t_feats1["feats"][1], t_feats1["feats"][2]
                rb1_t2, rb2_t2 = t_feats2["feats"][1], t_feats2["feats"][2]
                rb1_t3, rb2_t3 = t_feats3["feats"][1], t_feats3["feats"][2]
                rb1_t4, rb2_t4 = t_feats4["feats"][1], t_feats4["feats"][2]
                rb1_t5, rb2_t5 = t_feats5["feats"][1], t_feats5["feats"][2]
                rb1_t6, rb2_t6 = t_feats6["feats"][1], t_feats6["feats"][2]

                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1)), 1)

                if args.tmodel == 'resnet56':
                    rb1_t = F.normalize(rb1_t+1e-8)
                    rb2_t = F.normalize(rb2_t+1e-8)
                    aft_rb1_t = F.normalize(aft_rb1_t+1e-8)
                    aft_rb2_t = F.normalize(aft_rb2_t+1e-8)

                ExperiencePrediction_rb1_t = ExperiencePrediction1(rb1_t)
                ExperiencePrediction_rb2_t = ExperiencePrediction2(rb2_t)

        if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
            Meta_hard_loss_1 = MSE_criterion(ExperiencePrediction_rb1_t, aft_rb1_t)
            Meta_hard_loss_2 = MSE_criterion(ExperiencePrediction_rb2_t, aft_rb2_t)
            Meta_loss = Meta_hard_loss_1 + Meta_hard_loss_2
            test_Meta_loss.update(Meta_loss.cpu().detach().numpy().mean(), inputs.size(0))

        Shortcut_loss = LogCosh_criterion(torch.exp(torch.nn.functional.normalize(Shortcut_t_outputs+1e-8, p=2, dim=1)),
                                          torch.exp(torch.nn.functional.normalize(bes_outputs+1e-8, p=2, dim=1)))
        test_Shortcut_loss.update(Shortcut_loss.cpu().detach().numpy().mean(), inputs.size(0))

    if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
        return test_Meta_loss.avg, test_Shortcut_loss.avg
    else:
        return 00.00, test_Shortcut_loss.avg



path_teacher = os.path.join(args.save_root + args.data_name + '_' + args.tmodel)

normal_N = 0
for f_name in os.listdir(path_teacher):
    if fnmatch.fnmatch(f_name, 'Teacher_model_Normal_*.t7'):
        normal_N = normal_N + 1

best_N = np.zeros(shape=0, dtype=np.int8)
for f_name in os.listdir(path_teacher):
    if fnmatch.fnmatch(f_name, 'Teacher_model_Best_*.t7'):
        N = f_name.split('.')[0].split('Best_')[1]
        best_N = np.append(best_N, int(N))
best_N = np.sort(best_N, axis = 0)


if args.tmodel == 'resnet34':
    tnet1 = resnet34(num_classes=NUM_CLASSES).to(device).eval()
    tnet2 = resnet34(num_classes=NUM_CLASSES).to(device).eval()
    tnet3 = resnet34(num_classes=NUM_CLASSES).to(device).eval()
    tnet4 = resnet34(num_classes=NUM_CLASSES).to(device).eval()
    tnet5 = resnet34(num_classes=NUM_CLASSES).to(device).eval()
    tnet6 = resnet34(num_classes=NUM_CLASSES).to(device).eval()
    aft_tnet = resnet34(num_classes=NUM_CLASSES).to(device).eval()
    bes_tnet = resnet34(num_classes=NUM_CLASSES).to(device).eval()
elif args.tmodel == 'resnet56':
    tnet1 = resnet56(num_classes=NUM_CLASSES).to(device).eval()
    tnet2 = resnet56(num_classes=NUM_CLASSES).to(device).eval()
    tnet3 = resnet56(num_classes=NUM_CLASSES).to(device).eval()
    tnet4 = resnet56(num_classes=NUM_CLASSES).to(device).eval()
    tnet5 = resnet56(num_classes=NUM_CLASSES).to(device).eval()
    tnet6 = resnet56(num_classes=NUM_CLASSES).to(device).eval()
    aft_tnet = resnet56(num_classes=NUM_CLASSES).to(device).eval()
    bes_tnet = resnet56(num_classes=NUM_CLASSES).to(device).eval()
elif args.tmodel == 'Vgg13':
    tnet1 = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
    tnet2 = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
    tnet3 = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
    tnet4 = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
    tnet5 = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
    tnet6 = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
    aft_tnet = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
    bes_tnet = Vgg13(num_classes=NUM_CLASSES).to(device).eval()
else:
    raise Exception('Invalid model name...')

tnet1 = torch.nn.parallel.DistributedDataParallel(tnet1, device_ids=[local_rank], output_device=local_rank)
tnet2 = torch.nn.parallel.DistributedDataParallel(tnet2, device_ids=[local_rank], output_device=local_rank)
tnet3 = torch.nn.parallel.DistributedDataParallel(tnet3, device_ids=[local_rank], output_device=local_rank)
tnet4 = torch.nn.parallel.DistributedDataParallel(tnet4, device_ids=[local_rank], output_device=local_rank)
tnet5 = torch.nn.parallel.DistributedDataParallel(tnet5, device_ids=[local_rank], output_device=local_rank)
tnet6 = torch.nn.parallel.DistributedDataParallel(tnet6, device_ids=[local_rank], output_device=local_rank)
aft_tnet = torch.nn.parallel.DistributedDataParallel(aft_tnet, device_ids=[local_rank], output_device=local_rank)
bes_tnet = torch.nn.parallel.DistributedDataParallel(bes_tnet, device_ids=[local_rank], output_device=local_rank)


for epoch in range(0, args.epochs):
    local_N = np.arange(args.N_Teacher, normal_N, 1)
    Train_Meta_loss, Train_Shortcut_loss, Test_Meta_loss, Test_Shortcut_loss = 0,0,0,0
    for i in range(0, normal_N-args.N_Teacher):

        local_best_N = best_N[np.argmax(np.array(best_N) >= local_N[i])]
        bcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Best_' + str(local_best_N) + '.t7'), map_location=device)
        load_pretrained_model(bes_tnet, bcheckpoint['tnet'])
        
        tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 6) + '.t7'), map_location=device)
        load_pretrained_model(tnet1, tcheckpoint1['tnet'])
        tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 5) + '.t7'), map_location=device)
        load_pretrained_model(tnet2, tcheckpoint2['tnet'])
        tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'), map_location=device)
        load_pretrained_model(tnet3, tcheckpoint3['tnet'])
        tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'), map_location=device)
        load_pretrained_model(tnet4, tcheckpoint4['tnet'])
        tcheckpoint5 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'), map_location=device)
        load_pretrained_model(tnet5, tcheckpoint5['tnet'])
        tcheckpoint6 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'), map_location=device)
        load_pretrained_model(tnet6, tcheckpoint6['tnet'])
        tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'), map_location=device)
        load_pretrained_model(aft_tnet, tcheckpoint['tnet'])
        
        train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6)
        test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6)

        if torch.distributed.get_rank() == 0:
            f = open(path + '/' + args.data_name + '_' + args.tmodel + '_Skill' + '.txt', 'a')
            f.write("\nEpoch: %d, local_N: %d, train_Meta_loss:  %0.3f, train_Shortcut_loss:  %0.6f, "
                    "test_Meta_loss:  %0.3f, test_Shortcut_loss:  %0.6f" % (epoch, local_N[i], train_Meta_loss,
                                                                train_Shortcut_loss, test_Meta_loss, test_Shortcut_loss))
            f.close()

        Train_Meta_loss = Train_Meta_loss + train_Meta_loss
        Train_Shortcut_loss = Train_Shortcut_loss + train_Shortcut_loss
        Test_Meta_loss = Test_Meta_loss + test_Meta_loss
        Test_Shortcut_loss = Test_Shortcut_loss + test_Shortcut_loss
        
    Train_Meta_loss = Train_Meta_loss / (normal_N-args.N_Teacher)
    Train_Shortcut_loss = Train_Shortcut_loss / (normal_N-args.N_Teacher)
    Test_Meta_loss = Test_Meta_loss / (normal_N-args.N_Teacher)
    Test_Shortcut_loss = Test_Shortcut_loss / (normal_N-args.N_Teacher)

    if torch.distributed.get_rank() == 0:
        f = open(path + '/' + args.data_name + '_' + args.tmodel + '_Skill' + '.txt', 'a')
        f.write("\n\nEpoch: %d, train_Meta_loss:  %0.3f, train_Shortcut_loss:  %0.6f, test_Meta_loss:  %0.3f, "
                "test_Shortcut_loss:  %0.6f" % (epoch, Train_Meta_loss, Train_Shortcut_loss, Test_Meta_loss, Test_Shortcut_loss))

        if args.tmodel == 'resnet56' or args.tmodel == 'Vgg13':
            if best_Meta_loss > Test_Meta_loss:
                best_Meta_loss = Test_Meta_loss
                f.write('\nSaving ExperiencePrediction......')
                ExperiencePrediction_state = {
                    'ExperiencePrediction1': ExperiencePrediction1.state_dict(),
                    'ExperiencePrediction2': ExperiencePrediction2.state_dict(),
                }
                torch.save(ExperiencePrediction_state, os.path.join(path, 'ExperiencePrediction.t7'))

        if best_Shortcut_loss > Test_Shortcut_loss:
            best_Shortcut_loss = Test_Shortcut_loss
            f.write('\nSaving ShortcutPrediction......')
            ShortcutPrediction_state = {
                'ShortcutPrediction': ShortcutPrediction.state_dict(),
            }
            torch.save(ShortcutPrediction_state, os.path.join(path, 'ShortcutPrediction.t7'))
        f.close()

if torch.distributed.get_rank() == 0:
    writer.close()
