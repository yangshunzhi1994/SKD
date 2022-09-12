from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import utils
import torch.utils.data.distributed
import torchvision.transforms as transforms
from datasets.RAF import RAF_teacher
from datasets.ExpW import ExpW_teacher
from datasets.CK_Plus import CK_Plus_teacher
from torch.autograd import Variable
from network.teacherNet import Teacher
from torch.utils.tensorboard import SummaryWriter
from utils import ACC_evaluation

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--model', type=str, default="Teacher", help='Teacher')
parser.add_argument('--data_name', type=str, default="ExpW", help='RAF, ExpW,CK_Plus')
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='Batch size')
parser.add_argument('--test_bs', default=8, type=int, help='Batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()

best_acc = 0
best_mAP = 0
best_F1 = 0
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5  # 5
learning_rate_decay_rate = 0.9  # 0.9

path = os.path.join(args.save_root + args.data_name + '_' + args.model)
if not os.path.isdir(path):
    os.makedirs(path)
writer = SummaryWriter(log_dir=path)

f = open(path + '/' + args.data_name + '_' + args.model + '.txt', 'a')
f.write('\nThe dataset used for training is:                ' + str(args.data_name))
f.write('\nThe training mode is:                            ' + str(args.model))
f.write('\n==> Preparing data..')
f.close()

if args.data_name == 'RAF':
    NUM_CLASSES = 7
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5884594, 0.45767313, 0.40865755),
                             (0.25717735, 0.23602168, 0.23505741)),
    ])
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.589667, 0.45717254, 0.40727714], std=[0.25235596, 0.23242524, 0.23155019])
                                                     (transforms.ToTensor()(crop)) for crop in crops])),])
    trainset = RAF_teacher(split='Training', transform=transform_train)
    testset = RAF_teacher(split='Testing', transform=transform_test)

elif args.data_name == 'CK_Plus':
    NUM_CLASSES = 7
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5950821, 0.59496826, 0.5949638),
                             (0.2783952, 0.27837786, 0.27837303)),
    ])
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.5283895, 0.49888685, 0.48856217], std=[0.22694704, 0.21892785, 0.22059701])
                                                     (transforms.ToTensor()(crop)) for crop in crops])),])
    trainset = CK_Plus_teacher(split='Training', transform=transform_train)
    testset = CK_Plus_teacher(split='Testing', transform=transform_test)

elif args.data_name == 'ExpW':
    NUM_CLASSES = 7
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6199751, 0.46946654, 0.4103778),
                             (0.25622123, 0.22915973, 0.2232292)),
    ])
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.6081647, 0.4579959, 0.3987486], std=[0.25485262, 0.22496806, 0.21835831])
                                                     (transforms.ToTensor()(crop)) for crop in crops])),])
    trainset = ExpW_teacher(split='Training', transform=transform_train)
    testset = ExpW_teacher(split='Testing', transform=transform_test)

else:
    raise Exception('Invalid ...')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, pin_memory=True)

# Model
if args.model == 'Teacher':
    net = Teacher(num_classes=NUM_CLASSES).cuda()
else:
    raise Exception('.............Invalid..............')

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    f = open(path + '/' + args.data_name + '_' + args.model + '.txt', 'a')
    f.write('\n\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = args.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = args.lr
    f.write('\nlearning_rate: %s' % str(current_lr))
    f.close()

    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        _, _, _, _, _, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, outputs, targets, NUM_CLASSES)
    return train_loss / (batch_idx + 1), 100. * acc, 100. * mAP, 100 * F1_score


# +
def test(epoch):
    net.eval()
    PrivateTest_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        test_bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, _, _, _, _, outputs = net(inputs)
        outputs_avg = outputs.view(test_bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, outputs_avg, targets, NUM_CLASSES)
    return PrivateTest_loss / (batch_idx + 1), 100. * acc, 100. * mAP, 100 * F1_score





for epoch in range(0, args.epochs):
    # train one epoch
    train_loss, train_acc, train_mAP, train_F1 = train(epoch)
    # evaluate on testing set
    test_loss, test_acc, test_mAP, test_F1 = test(epoch)

    f = open(path + '/' + args.data_name + '_' + args.model + '.txt', 'a')
    f.write("\ntrain_loss:  %0.3f, train_acc:  %0.3f, train_mAP:  %0.3f, train_F1:  %0.3f" % (
    train_loss, train_acc, train_mAP, train_F1))
    f.write("\ntest_loss:   %0.3f, test_acc:   %0.3f, test_mAP:   %0.3f, test_F1:   %0.3f" % (
    test_loss, test_acc, test_mAP, test_F1))

    writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
    writer.add_scalars('epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch)
    writer.add_scalars('epoch/mAP', {'train': train_mAP, 'test': test_mAP}, epoch)
    writer.add_scalars('epoch/F1', {'train': train_F1, 'test': test_F1}, epoch)

    tnet_state = {
        'tnet': net.state_dict(),
        'test_acc': test_acc,
        'test_mAP': test_mAP,
        'test_F1': test_F1,
        'test_epoch': epoch,
    }

    # save model
    if test_acc > best_acc:
        best_acc = test_acc
        best_mAP = test_mAP
        best_F1 = test_F1
        f.write('\nSaving models......')
        f.write("\nbest_PrivateTest_acc: %0.3f" % test_acc)
        f.write("\nbest_PrivateTest_mAP: %0.3f" % test_mAP)
        f.write("\nbest_PrivateTest_F1: %0.3f" % test_F1)
        torch.save(tnet_state, os.path.join(path, 'Best_Teacher_model.t7'))
        torch.save(tnet_state, os.path.join(path, 'Teacher_model_Best_' + str(epoch) + '.t7'))
    torch.save(tnet_state, os.path.join(path, 'Teacher_model_Normal_' + str(epoch) + '.t7'))
    f.close()

f = open(path + '/' + args.data_name + '_' + args.model + '.txt', 'a')
f.write("\n\n\n\nbest_PrivateTest_acc: %0.2f" % best_acc)
f.write("\nbest_PrivateTest_mAP: %0.2f" % best_mAP)
f.write("\nbest_PrivateTest_F1: %0.2f" % best_F1)
f.close()
writer.close()
