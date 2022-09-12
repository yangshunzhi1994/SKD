import argparse
import os
import numpy as np
import torch
import fnmatch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms as transforms
import utils
import losses
import itertools
from EncoderDecoderConvLSTM import EncoderDecoderConvLSTM
from utils import load_pretrained_model
from datasets.RAF import RAF_teacher
from datasets.ExpW import ExpW_teacher
from datasets.CK_Plus import CK_Plus_teacher
from network.teacherNet import Teacher

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--model', type=str, default="Teacher", help='Teacher')
parser.add_argument('--data_name', type=str, default="RAF", help='RAF, ExpW, CK_Plus')
parser.add_argument('--epochs', type=int, default=15, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='Batch size')
parser.add_argument('--test_bs', default=256, type=int, help='Batch size')
parser.add_argument('--N_Teacher', default=6, type=int, help='How many epoch states of the teacher network '
                                                              'are used to predict its next action.')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()

path = os.path.join(args.save_root + args.data_name + '_' + args.model + '_N' + str(args.N_Teacher) + '_Skill')
if not os.path.isdir(path):
    os.mkdir(path)
writer = SummaryWriter(log_dir=path)

f = open(path + '/' + args.data_name + '_' + args.model + '_N' + str(args.N_Teacher) + '_Skill' + '.txt', 'a')
f.write('The dataset used for training is:                             ' + str(args.data_name))
f.write('\nThe training mode is:                                         ' + str(args.model))
f.write('\nThe number of epochs used to capture teacher skills is:       ' + str(args.N_Teacher))
f.write('\n==> Preparing data..')
f.close()

if args.data_name == 'RAF':
    NUM_CLASSES = 7
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5884594, 0.45767313, 0.40865755),
            (0.25717735, 0.23602168, 0.23505741)),])
    transform_test = transforms.Compose([
        transforms.Resize(92),
        transforms.ToTensor(),
        transforms.Normalize((0.59003043, 0.4573948, 0.40749523),
                             (0.2465465, 0.22635746, 0.22564183)),])
    trainset = RAF_teacher(split='Training', transform=transform_train)
    testset = RAF_teacher(split='Testing', transform=transform_test)


elif args.data_name == 'CK_Plus':
    NUM_CLASSES = 7
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5950821, 0.59496826, 0.5949638),
                             (0.2783952, 0.27837786, 0.27837303)), ])
    transform_test = transforms.Compose([
        transforms.Resize(92),
        transforms.ToTensor(),
        transforms.Normalize((0.5283895, 0.49888685, 0.48856217),
                             (0.22694704, 0.21892785, 0.22059701)), ])
    trainset = CK_Plus_teacher(split='Training', transform=transform_train)
    testset = CK_Plus_teacher(split='Testing', transform=transform_test)

elif args.data_name == 'ExpW':
    NUM_CLASSES = 7
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6199751, 0.46946654, 0.4103778),
                             (0.25622123, 0.22915973, 0.2232292)), ])
    transform_test = transforms.Compose([
        transforms.Resize(92),
        transforms.ToTensor(),
        transforms.Normalize((0.6081647, 0.4579959, 0.3987486),
                             (0.25485262, 0.22496806, 0.21835831)), ])
    trainset = ExpW_teacher(split='Training', transform=transform_train)
    testset = ExpW_teacher(split='Testing', transform=transform_test)

else:
    raise Exception('Invalid ...')


trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, pin_memory=True, num_workers=args.num_workers)

MSE_criterion = nn.MSELoss().cuda()
ExperiencePrediction1 = EncoderDecoderConvLSTM(in_chan=96, nf=256, future_step=16).cuda()
ExperiencePrediction2 = EncoderDecoderConvLSTM(in_chan=160, nf=256, future_step=16).cuda()
MetaLearner_optimizer = optim.SGD(itertools.chain(ExperiencePrediction1.parameters(), ExperiencePrediction2.parameters()),
                                  lr=args.lr, momentum=0.9, weight_decay=5e-4)

LogCosh_criterion = losses.LogCoshLoss().cuda()
ShortcutPrediction = losses.Shortcut(args.N_Teacher, nf=256, future_step=32).cuda()
Shortcut_optimizer = optim.SGD(ShortcutPrediction.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

learning_rate_decay_start = 3
learning_rate_decay_every = 3
learning_rate_decay_rate = 0.9
best_Meta_loss = 100
best_Shortcut_loss = 100

def train(aft_tnet, bes_tnet, tnet1, tnet2=None, tnet3=None, tnet4=None, tnet5=None, tnet6=None,
          tnet7=None, tnet8=None, tnet9=None, tnet10=None):
    ExperiencePrediction1.train()
    ExperiencePrediction2.train()
    ShortcutPrediction.train()
    train_Meta_loss = 0
    train_Shortcut_loss = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = args.lr * decay_factor
        utils.set_lr(MetaLearner_optimizer, current_lr)
        utils.set_lr(Shortcut_optimizer, current_lr)
    else:
        current_lr = args.lr

    f = open(path + '/' + args.data_name + '_' + args.model + '_N' + str(args.N_Teacher) + '_Skill' + '.txt', 'a')
    f.write('\nlearning_rate: %s' % str(current_lr))
    f.close()

    for batch_idx, (inputs, _, _) in enumerate(trainloader):
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        # Here the empirical knowledge of the teacher network is captured by training a meta-learner.
        with torch.no_grad():
            aft_rb1_t, aft_rb2_t, _, _, _, _ = aft_tnet(inputs)
            _, _, _, _, _, bes_outputs = bes_tnet(inputs)
            if args.N_Teacher == 2:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2)), 2)
            elif args.N_Teacher == 3:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2)), 2)
            elif args.N_Teacher == 4:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2), t_outputs4.unsqueeze(2)), 2)
            elif args.N_Teacher == 5:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2)), 2)
            elif args.N_Teacher == 6:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2)), 2)
            elif args.N_Teacher == 7:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2)), 2)
            elif args.N_Teacher == 8:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t8, rb2_t8, _, _, _, t_outputs8 = tnet8(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1), rb1_t8.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1), rb2_t8.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2), t_outputs8.unsqueeze(2)), 2)
            elif args.N_Teacher == 9:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t8, rb2_t8, _, _, _, t_outputs8 = tnet8(inputs)
                rb1_t9, rb2_t9, _, _, _, t_outputs9 = tnet9(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1), rb1_t8.unsqueeze(1),
                                   rb1_t9.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1), rb2_t8.unsqueeze(1),
                                   rb2_t9.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2), t_outputs8.unsqueeze(2), t_outputs9.unsqueeze(2)), 2)
            elif args.N_Teacher == 10:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t8, rb2_t8, _, _, _, t_outputs8 = tnet8(inputs)
                rb1_t9, rb2_t9, _, _, _, t_outputs9 = tnet9(inputs)
                rb1_t10, rb2_t10, _, _, _, t_outputs10 = tnet10(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1), rb1_t8.unsqueeze(1),
                                   rb1_t9.unsqueeze(1), rb1_t10.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1), rb2_t8.unsqueeze(1),
                                   rb2_t9.unsqueeze(1), rb2_t10.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2), t_outputs8.unsqueeze(2), t_outputs9.unsqueeze(2),
                                       t_outputs10.unsqueeze(2)), 2)
            else:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t = rb1_t1.unsqueeze(1)
                rb2_t = rb2_t1.unsqueeze(1)
                t_outputs = t_outputs1.unsqueeze(2)

            rb1_t = torch.nn.functional.interpolate(rb1_t, size=[96, 22, 22], mode='nearest', align_corners=None)
            rb2_t = torch.nn.functional.interpolate(rb2_t, size=[160, 11, 11], mode='nearest', align_corners=None)
            aft_rb1_t = torch.nn.functional.interpolate(aft_rb1_t, size=[22, 22], mode='nearest', align_corners=None)
            aft_rb2_t = torch.nn.functional.interpolate(aft_rb2_t, size=[11, 11], mode='nearest', align_corners=None)

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
        train_Meta_loss += Meta_loss.item()

        # Here, the short-cut knowledge of the teacher network is acquired by training the meta-learner.
        Shortcut_optimizer.zero_grad()
        t_outputs, bes_outputs = Variable(t_outputs), Variable(bes_outputs)
        Shortcut_t_outputs = ShortcutPrediction(t_outputs)

        Shortcut_loss = LogCosh_criterion(torch.exp(torch.nn.functional.normalize(Shortcut_t_outputs+1e-8, p=2, dim=1)),
                                          torch.exp(torch.nn.functional.normalize(bes_outputs+1e-8, p=2, dim=1)))
        Shortcut_loss.backward()

        utils.clip_gradient(Shortcut_optimizer, 0.1)
        Shortcut_optimizer.step()
        train_Shortcut_loss += Shortcut_loss.item()

    return train_Meta_loss / (batch_idx + 1), train_Shortcut_loss / (batch_idx + 1)

def test(aft_tnet, bes_tnet, tnet1, tnet2=None, tnet3=None, tnet4=None, tnet5=None, tnet6=None,
         tnet7=None, tnet8=None, tnet9=None, tnet10=None):
    ExperiencePrediction1.eval()
    ExperiencePrediction2.eval()
    ShortcutPrediction.eval()
    test_Meta_loss = 0
    test_Shortcut_loss = 0
    for batch_idx, (inputs, _, _) in enumerate(testloader):
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        with torch.no_grad():
            aft_rb1_t, aft_rb2_t, _, _, _, _ = aft_tnet(inputs)
            _, _, _, _, _, bes_outputs = bes_tnet(inputs)
            if args.N_Teacher == 2:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2)), 2)
            elif args.N_Teacher == 3:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2)), 2)
            elif args.N_Teacher == 4:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1)),
                                  1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1)),
                                  1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2)), 2)
            elif args.N_Teacher == 5:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2)), 2)
            elif args.N_Teacher == 6:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2)), 2)
            elif args.N_Teacher == 7:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2)), 2)
            elif args.N_Teacher == 8:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t8, rb2_t8, _, _, _, t_outputs8 = tnet8(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1), rb1_t8.unsqueeze(1)),
                                  1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1), rb2_t8.unsqueeze(1)),
                                  1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2), t_outputs8.unsqueeze(2)), 2)
            elif args.N_Teacher == 9:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t8, rb2_t8, _, _, _, t_outputs8 = tnet8(inputs)
                rb1_t9, rb2_t9, _, _, _, t_outputs9 = tnet9(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1), rb1_t8.unsqueeze(1),
                                   rb1_t9.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1), rb2_t8.unsqueeze(1),
                                   rb2_t9.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2), t_outputs8.unsqueeze(2), t_outputs9.unsqueeze(2)), 2)
            elif args.N_Teacher == 10:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t2, rb2_t2, _, _, _, t_outputs2 = tnet2(inputs)
                rb1_t3, rb2_t3, _, _, _, t_outputs3 = tnet3(inputs)
                rb1_t4, rb2_t4, _, _, _, t_outputs4 = tnet4(inputs)
                rb1_t5, rb2_t5, _, _, _, t_outputs5 = tnet5(inputs)
                rb1_t6, rb2_t6, _, _, _, t_outputs6 = tnet6(inputs)
                rb1_t7, rb2_t7, _, _, _, t_outputs7 = tnet7(inputs)
                rb1_t8, rb2_t8, _, _, _, t_outputs8 = tnet8(inputs)
                rb1_t9, rb2_t9, _, _, _, t_outputs9 = tnet9(inputs)
                rb1_t10, rb2_t10, _, _, _, t_outputs10 = tnet10(inputs)
                rb1_t = torch.cat((rb1_t1.unsqueeze(1), rb1_t2.unsqueeze(1), rb1_t3.unsqueeze(1), rb1_t4.unsqueeze(1),
                                   rb1_t5.unsqueeze(1), rb1_t6.unsqueeze(1), rb1_t7.unsqueeze(1), rb1_t8.unsqueeze(1),
                                   rb1_t9.unsqueeze(1), rb1_t10.unsqueeze(1)), 1)
                rb2_t = torch.cat((rb2_t1.unsqueeze(1), rb2_t2.unsqueeze(1), rb2_t3.unsqueeze(1), rb2_t4.unsqueeze(1),
                                   rb2_t5.unsqueeze(1), rb2_t6.unsqueeze(1), rb2_t7.unsqueeze(1), rb2_t8.unsqueeze(1),
                                   rb2_t9.unsqueeze(1), rb2_t10.unsqueeze(1)), 1)
                t_outputs = torch.cat((t_outputs1.unsqueeze(2), t_outputs2.unsqueeze(2), t_outputs3.unsqueeze(2),
                                       t_outputs4.unsqueeze(2), t_outputs5.unsqueeze(2), t_outputs6.unsqueeze(2),
                                       t_outputs7.unsqueeze(2), t_outputs8.unsqueeze(2), t_outputs9.unsqueeze(2),
                                       t_outputs10.unsqueeze(2)), 2)
            else:
                rb1_t1, rb2_t1, _, _, _, t_outputs1 = tnet1(inputs)
                rb1_t = rb1_t1.unsqueeze(1)
                rb2_t = rb2_t1.unsqueeze(1)
                t_outputs = t_outputs1.unsqueeze(2)

            rb1_t = torch.nn.functional.interpolate(rb1_t, size=[96, 22, 22], mode='nearest', align_corners=None)
            rb2_t = torch.nn.functional.interpolate(rb2_t, size=[160, 11, 11], mode='nearest', align_corners=None)
            aft_rb1_t = torch.nn.functional.interpolate(aft_rb1_t, size=[22, 22], mode='nearest', align_corners=None)
            aft_rb2_t = torch.nn.functional.interpolate(aft_rb2_t, size=[11, 11], mode='nearest', align_corners=None)

            ExperiencePrediction_rb1_t = ExperiencePrediction1(rb1_t)
            ExperiencePrediction_rb2_t = ExperiencePrediction2(rb2_t)
            Shortcut_t_outputs = ShortcutPrediction(t_outputs)

        Meta_hard_loss_1 = MSE_criterion(ExperiencePrediction_rb1_t, aft_rb1_t)
        Meta_hard_loss_2 = MSE_criterion(ExperiencePrediction_rb2_t, aft_rb2_t)
        Meta_loss = Meta_hard_loss_1 + Meta_hard_loss_2
        test_Meta_loss += Meta_loss.item()

        Shortcut_loss = LogCosh_criterion(torch.exp(torch.nn.functional.normalize(Shortcut_t_outputs+1e-8, p=2, dim=1)),
                                          torch.exp(torch.nn.functional.normalize(bes_outputs+1e-8, p=2, dim=1)))
        test_Shortcut_loss += Shortcut_loss.item()
        
    return test_Meta_loss/(batch_idx+1), test_Shortcut_loss/(batch_idx+1)

path_teacher = os.path.join(args.save_root + args.data_name + '_' + args.model)

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

if args.N_Teacher == 2:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 3:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 4:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet4 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 5:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet4 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet5 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 6:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet4 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet5 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet6 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 7:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet4 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet5 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet6 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet7 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 8:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet4 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet5 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet6 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet7 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet8 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 9:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet4 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet5 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet6 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet7 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet8 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet9 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
elif args.N_Teacher == 10:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet2 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet3 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet4 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet5 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet6 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet7 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet8 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet9 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
    tnet10 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
else:
    tnet1 = Teacher(num_classes=NUM_CLASSES).cuda().eval()
aft_tnet = Teacher(num_classes=NUM_CLASSES).cuda().eval()
bes_tnet = Teacher(num_classes=NUM_CLASSES).cuda().eval()

for epoch in range(0, args.epochs):
    local_N = np.arange(args.N_Teacher, normal_N, 1)
    Train_Meta_loss, Train_Shortcut_loss, Test_Meta_loss, Test_Shortcut_loss = 0,0,0,0
    for i in range(0, normal_N-args.N_Teacher):

        local_best_N = best_N[np.argmax(np.array(best_N) >= local_N[i])]
        bcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Best_' + str(local_best_N) + '.t7'))
        load_pretrained_model(bes_tnet, bcheckpoint['tnet'])

        if args.N_Teacher == 2:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]-2) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]-1) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2)

        elif args.N_Teacher == 3:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]-3) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]-2) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]-1) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3)

        elif args.N_Teacher == 4:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet4, tcheckpoint4['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4)

        elif args.N_Teacher == 5:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 5) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'))
            load_pretrained_model(tnet4, tcheckpoint4['tnet'])
            tcheckpoint5 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet5, tcheckpoint5['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5)

        elif args.N_Teacher == 6:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 6) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 5) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'))
            load_pretrained_model(tnet4, tcheckpoint4['tnet'])
            tcheckpoint5 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'))
            load_pretrained_model(tnet5, tcheckpoint5['tnet'])
            tcheckpoint6 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet6, tcheckpoint6['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6)

        elif args.N_Teacher == 7:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 7) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 6) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 5) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'))
            load_pretrained_model(tnet4, tcheckpoint4['tnet'])
            tcheckpoint5 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'))
            load_pretrained_model(tnet5, tcheckpoint5['tnet'])
            tcheckpoint6 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'))
            load_pretrained_model(tnet6, tcheckpoint6['tnet'])
            tcheckpoint7 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet7, tcheckpoint7['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7)

        elif args.N_Teacher == 8:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 8) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 7) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 6) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 5) + '.t7'))
            load_pretrained_model(tnet4, tcheckpoint4['tnet'])
            tcheckpoint5 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'))
            load_pretrained_model(tnet5, tcheckpoint5['tnet'])
            tcheckpoint6 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'))
            load_pretrained_model(tnet6, tcheckpoint6['tnet'])
            tcheckpoint7 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'))
            load_pretrained_model(tnet7, tcheckpoint7['tnet'])
            tcheckpoint8 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet8, tcheckpoint8['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7, tnet8)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7, tnet8)

        elif args.N_Teacher == 9:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 9) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 8) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 7) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 6) + '.t7'))
            load_pretrained_model(tnet4, tcheckpoint4['tnet'])
            tcheckpoint5 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 5) + '.t7'))
            load_pretrained_model(tnet5, tcheckpoint5['tnet'])
            tcheckpoint6 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'))
            load_pretrained_model(tnet6, tcheckpoint6['tnet'])
            tcheckpoint7 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'))
            load_pretrained_model(tnet7, tcheckpoint7['tnet'])
            tcheckpoint8 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'))
            load_pretrained_model(tnet8, tcheckpoint8['tnet'])
            tcheckpoint9 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet9, tcheckpoint9['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7, tnet8, tnet9)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7, tnet8, tnet9)

        elif args.N_Teacher == 10:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 10) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint2 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 9) + '.t7'))
            load_pretrained_model(tnet2, tcheckpoint2['tnet'])
            tcheckpoint3 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 8) + '.t7'))
            load_pretrained_model(tnet3, tcheckpoint3['tnet'])
            tcheckpoint4 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 7) + '.t7'))
            load_pretrained_model(tnet4, tcheckpoint4['tnet'])
            tcheckpoint5 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 6) + '.t7'))
            load_pretrained_model(tnet5, tcheckpoint5['tnet'])
            tcheckpoint6 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 5) + '.t7'))
            load_pretrained_model(tnet6, tcheckpoint6['tnet'])
            tcheckpoint7 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 4) + '.t7'))
            load_pretrained_model(tnet7, tcheckpoint7['tnet'])
            tcheckpoint8 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 3) + '.t7'))
            load_pretrained_model(tnet8, tcheckpoint8['tnet'])
            tcheckpoint9 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 2) + '.t7'))
            load_pretrained_model(tnet9, tcheckpoint9['tnet'])
            tcheckpoint10 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet10, tcheckpoint10['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7, tnet8, tnet9, tnet10)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1, tnet2, tnet3, tnet4, tnet5, tnet6, tnet7, tnet8, tnet9, tnet10)

        else:
            tcheckpoint1 = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i] - 1) + '.t7'))
            load_pretrained_model(tnet1, tcheckpoint1['tnet'])
            tcheckpoint = torch.load(os.path.join(path_teacher, 'Teacher_model_Normal_' + str(local_N[i]) + '.t7'))
            load_pretrained_model(aft_tnet, tcheckpoint['tnet'])

            train_Meta_loss, train_Shortcut_loss = train(aft_tnet, bes_tnet, tnet1)
            test_Meta_loss, test_Shortcut_loss = test(aft_tnet, bes_tnet, tnet1)

        f = open(path + '/' + args.data_name + '_' + args.model + '_N' + str(args.N_Teacher) + '_Skill' + '.txt', 'a')
        f.write("\nEpoch: %d, local_N: %d, train_Meta_loss:  %0.3f, train_Shortcut_loss:  %0.6f, "
                "test_Meta_loss:  %0.3f, test_Shortcut_loss:  %0.6f" % (epoch, local_N[i], train_Meta_loss,
                                                                        train_Shortcut_loss, test_Meta_loss,
                                                                        test_Shortcut_loss))
        f.close()

        Train_Meta_loss = Train_Meta_loss + train_Meta_loss
        Train_Shortcut_loss = Train_Shortcut_loss + train_Shortcut_loss
        Test_Meta_loss = Test_Meta_loss + test_Meta_loss
        Test_Shortcut_loss = Test_Shortcut_loss + test_Shortcut_loss
    Train_Meta_loss = Train_Meta_loss / (normal_N-args.N_Teacher)
    Train_Shortcut_loss = Train_Shortcut_loss / (normal_N-args.N_Teacher)
    Test_Meta_loss = Test_Meta_loss / (normal_N-args.N_Teacher)
    Test_Shortcut_loss = Test_Shortcut_loss / (normal_N-args.N_Teacher)

    f = open(path + '/' + args.data_name + '_' + args.model + '_N' + str(args.N_Teacher) + '_Skill' + '.txt', 'a')
    f.write("\n\nEpoch: %d, epoch_lr:  %0.9f, train_Meta_loss:  %0.3f, train_Shortcut_loss:  %0.6f, test_Meta_loss:  %0.3f, "
            "test_Shortcut_loss:  %0.6f" % (
            epoch, Shortcut_optimizer.param_groups[0]['lr'], Train_Meta_loss, Train_Shortcut_loss, Test_Meta_loss, Test_Shortcut_loss))

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
writer.close()
