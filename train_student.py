#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from datasets.RAF import RAF
from datasets.ExpW import ExpW
from datasets.CK_Plus import CK_Plus
from network.studentNet import CNN_RIS
from network.teacherNet import Teacher
import itertools
from itertools import chain

import losses
import utils
from utils import load_pretrained_model, ACC_evaluation
from torch.utils.tensorboard import SummaryWriter
from EncoderDecoderConvLSTM import EncoderDecoderConvLSTM

parser = argparse.ArgumentParser(description='train kd')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--t_model', type=str, default="Teacher", help='Teacher, ResNet152, ResNet50')
parser.add_argument('--s_model', type=str, default="CNNRIS", help='CNNRIS, resnet50, resnet20')
parser.add_argument('--distillation', type=str, default="Shortcut", help='Shortcut,Block1,Block2,Block12')
parser.add_argument('--data_name', type=str, default='CK_Plus', help='RAF, Imagenet, ExpW, CK_Plus')
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='learning rate')
parser.add_argument('--test_bs', default=256, type=int, help='learning rate')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--noise', type=str, default='none',
                    help='none, GaussianBlur,AverageBlur,MedianBlur,BilateralFilter,Salt-and-pepper')
parser.add_argument('--N_Teacher', default=6, type=int, help='How many epoch states of the teacher network '
                                                              'are used to predict its next action.')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.0, help='0.1,0.4,0.7,0.9')
parser.add_argument('--beta', type=float, default=0.0, help='0.1,0.4,0.7,0.9')
parser.add_argument('--gamma', type=float, default=0.0, help='0.1,0.4,0.7,0.9')
parser.add_argument('--delta', type=float, default=0.0, help='0.1,0.4,0.7,0.9')
parser.add_argument('--v', type=float, default=0.0, help='0.1,0.4,0.7,0.9')
parser.add_argument('--T', type=int, default=1, help='1,2,3...')

args, unparsed = parser.parse_known_args()
path = os.path.join(args.save_root + args.data_name + '_' + args.t_model + '_N' +
                    str(args.N_Teacher) + '_' + args.s_model + '_' + args.distillation)
writer = SummaryWriter(log_dir=path)

# def set_seed(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# set_seed(args.seed)

if args.data_name == 'RAF':
    NUM_CLASSES = 7
    transforms_teacher_Normalize = transforms.Normalize((0.5884594, 0.45767313, 0.40865755),
                                                              (0.25717735, 0.23602168, 0.23505741))
    transforms_student_Normalize = transforms.Normalize((0.58846486, 0.45766878, 0.40865615),
                                                              (0.2516557, 0.23020789, 0.22939532))
    transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.59003043, 0.4573948, 0.40749523], std=[0.2465465, 0.22635746, 0.22564183])
                                         (transforms.ToTensor()(crop)) for crop in crops]))
elif args.data_name == 'ExpW':
    NUM_CLASSES = 7
    transforms_teacher_Normalize = transforms.Normalize((0.6199751, 0.46946654, 0.4103778),
                                                        (0.25622123, 0.22915973, 0.2232292))
    transforms_student_Normalize = transforms.Normalize((0.6202007, 0.46964768, 0.41054007),
                                                         (0.2498027, 0.22279221, 0.21712679))
    transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.60876304, 0.45839235, 0.39910695], std=[0.2478118, 0.2180687, 0.21176754])
                                        (transforms.ToTensor()(crop)) for crop in crops]))
elif args.data_name == 'CK_Plus':
    NUM_CLASSES = 7
    transforms_teacher_Normalize = transforms.Normalize((0.5950821, 0.59496826, 0.5949638),
                                                        (0.2783952, 0.27837786, 0.27837303))
    transforms_student_Normalize = transforms.Normalize((0.59541404, 0.59529984, 0.59529567),
                                                         (0.2707762, 0.27075955, 0.27075458))
    transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.52888066, 0.4993276, 0.48900297], std=[0.21970414, 0.21182147, 0.21353027])
                                       (transforms.ToTensor()(crop)) for crop in crops]))
else:
    raise Exception('.............Invalid..............')

transform_train = transforms.Compose([
    transforms.RandomCrop(92),
    transforms.RandomHorizontalFlip(),
])
teacher_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms_teacher_Normalize,
])
student_norm = transforms.Compose([
    transforms.Resize(44),
    transforms.ToTensor(),
    transforms_student_Normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(48),
    transforms.TenCrop(44),
    transforms_test_Normalize,
])

if args.data_name == 'RAF':
    trainset = RAF(split='Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm, noise=args.noise)
    testset = RAF(split='Testing', transform=None, student_norm=transform_test, teacher_norm=None, noise=args.noise)
elif args.data_name == 'ExpW':
    trainset = ExpW(split='Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    testset = ExpW(split='Testing', transform=None, student_norm=transform_test, teacher_norm=None)
elif args.data_name == 'CK_Plus':
    trainset = CK_Plus(split='Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    testset = CK_Plus(split='Testing', transform=None, student_norm=transform_test, teacher_norm=None)
else:
    raise Exception('Invalid dataset name...')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, pin_memory=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, pin_memory=True, num_workers=args.num_workers)
best_acc = 0
best_mAP = 0
best_F1 = 0
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5  # 5
learning_rate_decay_rate = 0.9  # 0.9

if args.s_model == 'CNNRIS':
    snet = CNN_RIS(num_classes=NUM_CLASSES).cuda()
    bes_tnet = Teacher(num_classes=NUM_CLASSES).cuda().eval()
else:
    raise Exception('.............Invalid..............')

if args.N_Teacher == 2:
    pre_snet1 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 3:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 4:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
    pre_snet3 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 5:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
    pre_snet3 = deepcopy(snet).cuda().eval()
    pre_snet4 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 6:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
    pre_snet3 = deepcopy(snet).cuda().eval()
    pre_snet4 = deepcopy(snet).cuda().eval()
    pre_snet5 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 7:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
    pre_snet3 = deepcopy(snet).cuda().eval()
    pre_snet4 = deepcopy(snet).cuda().eval()
    pre_snet5 = deepcopy(snet).cuda().eval()
    pre_snet6 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 8:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
    pre_snet3 = deepcopy(snet).cuda().eval()
    pre_snet4 = deepcopy(snet).cuda().eval()
    pre_snet5 = deepcopy(snet).cuda().eval()
    pre_snet6 = deepcopy(snet).cuda().eval()
    pre_snet7 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 9:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
    pre_snet3 = deepcopy(snet).cuda().eval()
    pre_snet4 = deepcopy(snet).cuda().eval()
    pre_snet5 = deepcopy(snet).cuda().eval()
    pre_snet6 = deepcopy(snet).cuda().eval()
    pre_snet7 = deepcopy(snet).cuda().eval()
    pre_snet8 = deepcopy(snet).cuda().eval()
elif args.N_Teacher == 10:
    pre_snet1 = deepcopy(snet).cuda().eval()
    pre_snet2 = deepcopy(snet).cuda().eval()
    pre_snet3 = deepcopy(snet).cuda().eval()
    pre_snet4 = deepcopy(snet).cuda().eval()
    pre_snet5 = deepcopy(snet).cuda().eval()
    pre_snet6 = deepcopy(snet).cuda().eval()
    pre_snet7 = deepcopy(snet).cuda().eval()
    pre_snet8 = deepcopy(snet).cuda().eval()
    pre_snet9 = deepcopy(snet).cuda().eval()
else:
    raise Exception('.............Invalid..............')

Experience_checkpoint = torch.load(os.path.join(args.save_root + args.data_name + '_' + args.t_model + '_N' +
                                                str(args.N_Teacher) + '_Skill', 'ExperiencePrediction.t7'))
Shortcut_checkpoint = torch.load(os.path.join(args.save_root + args.data_name + '_' + args.t_model + '_N' +
                                              str(args.N_Teacher) + '_Skill', 'ShortcutPrediction.t7'))

ShortcutPrediction = losses.Shortcut(args.N_Teacher, nf=256, future_step=32).cuda()
load_pretrained_model(ShortcutPrediction, Shortcut_checkpoint['ShortcutPrediction'])
ShortcutPrediction.eval()
if args.distillation == 'Block1':
    ExperiencePrediction1 = EncoderDecoderConvLSTM(in_chan=96, nf=256, future_step=16).cuda()
    load_pretrained_model(ExperiencePrediction1, Experience_checkpoint['ExperiencePrediction1'])
    ExperiencePrediction1.eval()
    decoder = losses.Decoder().cuda()
    decoder.train()
    optimizer = torch.optim.SGD(itertools.chain(snet.parameters(),decoder.parameters()), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
elif args.distillation == 'Block2':
    ExperiencePrediction2 = EncoderDecoderConvLSTM(in_chan=160, nf=256, future_step=16).cuda()
    load_pretrained_model(ExperiencePrediction2, Experience_checkpoint['ExperiencePrediction2'])
    ExperiencePrediction2.eval()
    optimizer = torch.optim.SGD(snet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
elif args.distillation == 'Block12':
    ExperiencePrediction1 = EncoderDecoderConvLSTM(in_chan=96, nf=256, future_step=16).cuda()
    ExperiencePrediction2 = EncoderDecoderConvLSTM(in_chan=160, nf=256, future_step=16).cuda()
    load_pretrained_model(ExperiencePrediction1, Experience_checkpoint['ExperiencePrediction1'])
    load_pretrained_model(ExperiencePrediction2, Experience_checkpoint['ExperiencePrediction2'])
    ExperiencePrediction1.eval()
    ExperiencePrediction2.eval()
    decoder = losses.Decoder().cuda()
    decoder.train()
    optimizer = torch.optim.SGD(itertools.chain(snet.parameters(), decoder.parameters()), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
else:
    optimizer = torch.optim.SGD(snet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)

f = open(path + '/' + args.data_name + '_' + args.t_model + '_N' + str(args.N_Teacher) + '_' + args.s_model + '_'
         + args.distillation + '.txt', 'a')
f.write('\nThe dataset used for training is:   ' + str(args.data_name))
f.write('\nThe distillation method is:       ' + str(args.distillation))
f.write('\nThe Model of Teacher Network is:       ' + str(args.t_model))
f.write('\nThe Model of Student Network is:       ' + str(args.s_model))
f.write('\nThe number of epochs used to capture teacher skills is:       ' + str(args.N_Teacher))
f.write('\nThe type of noise used is:         ' + str(args.noise))
f.close()


MSE_criterion = nn.MSELoss().cuda()  # MSE
criterion = nn.CrossEntropyLoss().cuda()
LogCosh_criterion = losses.LogCoshLoss().cuda()
kl_criterion = losses.KL_divergence(temperature=args.T).cuda()

def train(epoch, bes_tnet):
    f = open(path + '/' + args.data_name + '_' + args.t_model + '_N' + str(args.N_Teacher) + '_' + args.s_model + '_'
             + args.distillation + '.txt', 'a')
    f.write('\n\nEpoch: %d' % epoch)
    if args.N_Teacher == 2:
        global pre_snet1
    elif args.N_Teacher == 3:
        global pre_snet1, pre_snet2
    elif args.N_Teacher == 4:
        global pre_snet1, pre_snet2, pre_snet3
    elif args.N_Teacher == 5:
        global pre_snet1, pre_snet2, pre_snet3, pre_snet4
    elif args.N_Teacher == 6:
        global pre_snet1, pre_snet2, pre_snet3, pre_snet4, pre_snet5
    elif args.N_Teacher == 7:
        global pre_snet1, pre_snet2, pre_snet3, pre_snet4, pre_snet5, pre_snet6
    elif args.N_Teacher == 8:
        global pre_snet1, pre_snet2, pre_snet3, pre_snet4, pre_snet5, pre_snet6, pre_snet7
    elif args.N_Teacher == 9:
        global pre_snet1, pre_snet2, pre_snet3, pre_snet4, pre_snet5, pre_snet6, pre_snet7, pre_snet8
    elif args.N_Teacher == 10:
        global pre_snet1, pre_snet2, pre_snet3, pre_snet4, pre_snet5, pre_snet6, pre_snet7, pre_snet8, pre_snet9
    else:
        raise Exception('.............Invalid..............')
    snet.train()
    train_skill_loss = 0
    train_cls_loss = 0
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

    for batch_idx, (img_teacher, img_student, targets, _) in enumerate(trainloader):

        img_teacher, img_student, targets = img_teacher.cuda(), img_student.cuda(), targets.cuda()
        img_teacher, img_student, targets = Variable(img_teacher), Variable(img_student), Variable(targets)
        optimizer.zero_grad()
        rb1_s, rb2_s, rb3_s, feat_s, mimic_s, out_s = snet(img_student)

        with torch.no_grad():
            rb1_t, rb2_t, rb3_t, feat_t, mimic_t, out_t = bes_tnet(img_teacher)
            if args.N_Teacher == 2:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 3:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 4:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                rb1_s3, rb2_s3, _, _, _, s_outputs3 = pre_snet3(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s3.unsqueeze(1), rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s3.unsqueeze(1), rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), s_outputs3.unsqueeze(2),
                                       out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 5:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                rb1_s3, rb2_s3, _, _, _, s_outputs3 = pre_snet3(img_student)
                rb1_s4, rb2_s4, _, _, _, s_outputs4 = pre_snet4(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s3.unsqueeze(1), rb1_s4.unsqueeze(1),
                                   rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s3.unsqueeze(1), rb2_s4.unsqueeze(1),
                                   rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), s_outputs3.unsqueeze(2),
                                       s_outputs4.unsqueeze(2), out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 6:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                rb1_s3, rb2_s3, _, _, _, s_outputs3 = pre_snet3(img_student)
                rb1_s4, rb2_s4, _, _, _, s_outputs4 = pre_snet4(img_student)
                rb1_s5, rb2_s5, _, _, _, s_outputs5 = pre_snet5(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s3.unsqueeze(1), rb1_s4.unsqueeze(1),
                                   rb1_s5.unsqueeze(1), rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s3.unsqueeze(1), rb2_s4.unsqueeze(1),
                                   rb2_s5.unsqueeze(1), rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), s_outputs3.unsqueeze(2),
                                       s_outputs4.unsqueeze(2), s_outputs5.unsqueeze(2), out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 7:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                rb1_s3, rb2_s3, _, _, _, s_outputs3 = pre_snet3(img_student)
                rb1_s4, rb2_s4, _, _, _, s_outputs4 = pre_snet4(img_student)
                rb1_s5, rb2_s5, _, _, _, s_outputs5 = pre_snet5(img_student)
                rb1_s6, rb2_s6, _, _, _, s_outputs6 = pre_snet6(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s3.unsqueeze(1), rb1_s4.unsqueeze(1),
                                   rb1_s5.unsqueeze(1), rb1_s6.unsqueeze(1), rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s3.unsqueeze(1), rb2_s4.unsqueeze(1),
                                   rb2_s5.unsqueeze(1), rb2_s6.unsqueeze(1), rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), s_outputs3.unsqueeze(2),
                                       s_outputs4.unsqueeze(2), s_outputs5.unsqueeze(2), s_outputs6.unsqueeze(2),
                                       out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 8:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                rb1_s3, rb2_s3, _, _, _, s_outputs3 = pre_snet3(img_student)
                rb1_s4, rb2_s4, _, _, _, s_outputs4 = pre_snet4(img_student)
                rb1_s5, rb2_s5, _, _, _, s_outputs5 = pre_snet5(img_student)
                rb1_s6, rb2_s6, _, _, _, s_outputs6 = pre_snet6(img_student)
                rb1_s7, rb2_s7, _, _, _, s_outputs7 = pre_snet7(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s3.unsqueeze(1), rb1_s4.unsqueeze(1),
                                   rb1_s5.unsqueeze(1), rb1_s6.unsqueeze(1), rb1_s7.unsqueeze(1),
                                   rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s3.unsqueeze(1), rb2_s4.unsqueeze(1),
                                   rb2_s5.unsqueeze(1), rb2_s6.unsqueeze(1), rb2_s7.unsqueeze(1),
                                   rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), s_outputs3.unsqueeze(2),
                                       s_outputs4.unsqueeze(2), s_outputs5.unsqueeze(2), s_outputs6.unsqueeze(2),
                                       s_outputs7.unsqueeze(2), out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 9:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                rb1_s3, rb2_s3, _, _, _, s_outputs3 = pre_snet3(img_student)
                rb1_s4, rb2_s4, _, _, _, s_outputs4 = pre_snet4(img_student)
                rb1_s5, rb2_s5, _, _, _, s_outputs5 = pre_snet5(img_student)
                rb1_s6, rb2_s6, _, _, _, s_outputs6 = pre_snet6(img_student)
                rb1_s7, rb2_s7, _, _, _, s_outputs7 = pre_snet7(img_student)
                rb1_s8, rb2_s8, _, _, _, s_outputs8 = pre_snet8(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s3.unsqueeze(1), rb1_s4.unsqueeze(1),
                                   rb1_s5.unsqueeze(1), rb1_s6.unsqueeze(1), rb1_s7.unsqueeze(1), rb1_s8.unsqueeze(1),
                                   rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s3.unsqueeze(1), rb2_s4.unsqueeze(1),
                                   rb2_s5.unsqueeze(1), rb2_s6.unsqueeze(1), rb2_s7.unsqueeze(1), rb2_s8.unsqueeze(1),
                                   rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), s_outputs3.unsqueeze(2),
                                       s_outputs4.unsqueeze(2), s_outputs5.unsqueeze(2), s_outputs6.unsqueeze(2),
                                       s_outputs7.unsqueeze(2), s_outputs8.unsqueeze(2), out_s.unsqueeze(2)), 2)
            elif args.N_Teacher == 10:
                rb1_s1, rb2_s1, _, _, _, s_outputs1 = pre_snet1(img_student)
                rb1_s2, rb2_s2, _, _, _, s_outputs2 = pre_snet2(img_student)
                rb1_s3, rb2_s3, _, _, _, s_outputs3 = pre_snet3(img_student)
                rb1_s4, rb2_s4, _, _, _, s_outputs4 = pre_snet4(img_student)
                rb1_s5, rb2_s5, _, _, _, s_outputs5 = pre_snet5(img_student)
                rb1_s6, rb2_s6, _, _, _, s_outputs6 = pre_snet6(img_student)
                rb1_s7, rb2_s7, _, _, _, s_outputs7 = pre_snet7(img_student)
                rb1_s8, rb2_s8, _, _, _, s_outputs8 = pre_snet8(img_student)
                rb1_s9, rb2_s9, _, _, _, s_outputs9 = pre_snet9(img_student)
                be_rb1_s = torch.cat((rb1_s1.unsqueeze(1), rb1_s2.unsqueeze(1), rb1_s3.unsqueeze(1), rb1_s4.unsqueeze(1),
                                   rb1_s5.unsqueeze(1), rb1_s6.unsqueeze(1), rb1_s7.unsqueeze(1), rb1_s8.unsqueeze(1),
                                   rb1_s9.unsqueeze(1), rb1_s.detach().unsqueeze(1)), 1)
                be_rb2_s = torch.cat((rb2_s1.unsqueeze(1), rb2_s2.unsqueeze(1), rb2_s3.unsqueeze(1), rb2_s4.unsqueeze(1),
                                   rb2_s5.unsqueeze(1), rb2_s6.unsqueeze(1), rb2_s7.unsqueeze(1), rb2_s8.unsqueeze(1),
                                   rb2_s9.unsqueeze(1), rb2_s.detach().unsqueeze(1)), 1)
                s_outputs = torch.cat((s_outputs1.unsqueeze(2), s_outputs2.unsqueeze(2), s_outputs3.unsqueeze(2),
                                       s_outputs4.unsqueeze(2), s_outputs5.unsqueeze(2), s_outputs6.unsqueeze(2),
                                       s_outputs7.unsqueeze(2), s_outputs8.unsqueeze(2), s_outputs9.unsqueeze(2),
                                       out_s.unsqueeze(2)), 2)
            else:
                be_rb1_s, be_rb2_s, s_outputs = rb1_s.unsqueeze(1), rb2_s.unsqueeze(1), out_s.unsqueeze(2)

        cls_loss = criterion(out_s, targets)
        Meta_t_outputs = ShortcutPrediction(s_outputs)
        ex_loss = LogCosh_criterion(torch.exp(torch.nn.functional.normalize(out_s+1e-8, p=2, dim=1)),
                                    torch.exp(torch.nn.functional.normalize(Meta_t_outputs+1e-8, p=2, dim=1)))
        kd_loss = kl_criterion(out_s, out_t)

        if args.distillation == 'Shortcut':
            loss = args.alpha * cls_loss + args.v * args.beta * ex_loss + args.beta * kd_loss
        elif args.distillation == 'Block1':
            with torch.no_grad():
                teacher_be_rb1_s = ExperiencePrediction1(be_rb1_s)
            be_loss = MSE_criterion(rb1_s, teacher_be_rb1_s)
            new_rb1_s = decoder(rb1_s)
            decoder_loss = losses.styleLoss(img_teacher, new_rb1_s.cuda(), MSE_criterion)
            loss = args.v * args.gamma * be_loss + args.gamma * decoder_loss + args.alpha*cls_loss \
                   + args.v * args.beta * ex_loss + args.beta * kd_loss
        elif args.distillation == 'Block2':
            with torch.no_grad():
                teacher_be_rb2_s = ExperiencePrediction2(be_rb2_s)
            be_loss = MSE_criterion(rb2_s, teacher_be_rb2_s)
            rb2_loss = losses.Absdiff_Similarity(rb2_s, rb2_t).cuda()
            loss = args.v * args.delta * be_loss + args.delta * rb2_loss + args.alpha * cls_loss \
                   + args.v * args.beta * ex_loss + args.beta * kd_loss
        else:
            with torch.no_grad():
                teacher_be_rb1_s = ExperiencePrediction1(be_rb1_s)
                teacher_be_rb2_s = ExperiencePrediction2(be_rb2_s)
            be_loss1 = MSE_criterion(rb1_s, teacher_be_rb1_s)
            be_loss2 = MSE_criterion(rb2_s, teacher_be_rb2_s)

            new_rb1_s = decoder(rb1_s)
            decoder_loss = losses.styleLoss(img_teacher, new_rb1_s.cuda(), MSE_criterion)
            rb2_loss = losses.Absdiff_Similarity(rb2_s, rb2_t).cuda()

            loss = args.v * args.gamma * be_loss1 + args.gamma * decoder_loss + args.v * args.delta * be_loss2 \
                   + args.delta * rb2_loss + args.alpha * cls_loss + args.v * args.beta * ex_loss + args.beta * kd_loss

        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        if args.distillation == 'Shortcut':
            train_skill_loss += ex_loss.item()
        elif args.distillation == 'Block1' or args.distillation == 'Block2':
            train_skill_loss += be_loss.item()
        else:
            train_skill_loss += be_loss1.item() + be_loss2.item()
        train_cls_loss += cls_loss.item()
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, out_s, targets, NUM_CLASSES)

    if args.N_Teacher == 2:
        pre_snet1 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 3:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 4:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(pre_snet3).cuda().eval()
        pre_snet3 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 5:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(pre_snet3).cuda().eval()
        pre_snet3 = deepcopy(pre_snet4).cuda().eval()
        pre_snet4 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 6:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(pre_snet3).cuda().eval()
        pre_snet3 = deepcopy(pre_snet4).cuda().eval()
        pre_snet4 = deepcopy(pre_snet5).cuda().eval()
        pre_snet5 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 7:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(pre_snet3).cuda().eval()
        pre_snet3 = deepcopy(pre_snet4).cuda().eval()
        pre_snet4 = deepcopy(pre_snet5).cuda().eval()
        pre_snet5 = deepcopy(pre_snet6).cuda().eval()
        pre_snet6 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 8:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(pre_snet3).cuda().eval()
        pre_snet3 = deepcopy(pre_snet4).cuda().eval()
        pre_snet4 = deepcopy(pre_snet5).cuda().eval()
        pre_snet5 = deepcopy(pre_snet6).cuda().eval()
        pre_snet6 = deepcopy(pre_snet7).cuda().eval()
        pre_snet7 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 9:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(pre_snet3).cuda().eval()
        pre_snet3 = deepcopy(pre_snet4).cuda().eval()
        pre_snet4 = deepcopy(pre_snet5).cuda().eval()
        pre_snet5 = deepcopy(pre_snet6).cuda().eval()
        pre_snet6 = deepcopy(pre_snet7).cuda().eval()
        pre_snet7 = deepcopy(pre_snet8).cuda().eval()
        pre_snet8 = deepcopy(snet).cuda().eval()
    elif args.N_Teacher == 10:
        pre_snet1 = deepcopy(pre_snet2).cuda().eval()
        pre_snet2 = deepcopy(pre_snet3).cuda().eval()
        pre_snet3 = deepcopy(pre_snet4).cuda().eval()
        pre_snet4 = deepcopy(pre_snet5).cuda().eval()
        pre_snet5 = deepcopy(pre_snet6).cuda().eval()
        pre_snet6 = deepcopy(pre_snet7).cuda().eval()
        pre_snet7 = deepcopy(pre_snet8).cuda().eval()
        pre_snet8 = deepcopy(pre_snet9).cuda().eval()
        pre_snet9 = deepcopy(snet).cuda().eval()
    else:
        raise Exception('.............Invalid..............')

    return train_cls_loss / (batch_idx + 1), train_skill_loss / (batch_idx + 1), 100. * acc, 100. * mAP, 100 * F1_score


def test(epoch):
    snet.eval()
    PrivateTest_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for batch_idx, (img, targets) in enumerate(testloader):
        test_bs, ncrops, c, h, w = np.shape(img)
        img = img.view(-1, c, h, w)
        img, targets = img.cuda(), targets.cuda()
        img, targets = Variable(img), Variable(targets)

        with torch.no_grad():
            rb1_s, rb2_s, rb3_s, feat_s, mimic_s, out_s = snet(img)
        outputs_avg = out_s.view(test_bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, outputs_avg, targets, NUM_CLASSES)

    return PrivateTest_loss / (batch_idx + 1), 100. * acc, 100. * mAP, 100 * F1_score


for epoch in range(0, args.epochs):
    bcheckpoint = torch.load(os.path.join(args.save_root + args.data_name + '_Teacher/', 'Best_Teacher_model.t7'))
    load_pretrained_model(bes_tnet, bcheckpoint['tnet'])

    train_loss, train_skill_loss, train_acc, train_mAP, train_F1 = train(epoch, bes_tnet)
    test_loss, test_acc, test_mAP, test_F1 = test(epoch)

    f = open(path + '/' + args.data_name + '_' + args.t_model + '_N' + str(args.N_Teacher) + '_' + args.s_model + '_'
             + args.distillation + '.txt', 'a')
    f.write("\ntrain_loss:  %0.3f, train_skill_loss:  %0.3f, train_acc:  %0.3f, train_mAP:  %0.3f, train_F1:  %0.3f" % (
    train_loss, train_skill_loss, train_acc, train_mAP, train_F1))
    f.write("\ntest_loss:   %0.3f, test_acc:   %0.3f, test_mAP:   %0.3f, test_F1:   %0.3f" % (
    test_loss, test_acc, test_mAP, test_F1))

    writer.add_scalars('epoch/loss', {'train': train_loss, 'train_skill': train_skill_loss, 'test': test_loss}, epoch)
    writer.add_scalars('epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch)
    writer.add_scalars('epoch/mAP', {'train': train_mAP, 'test': test_mAP}, epoch)
    writer.add_scalars('epoch/F1', {'train': train_F1, 'test': test_F1}, epoch)

    # save model
    if test_acc > best_acc:
        best_acc = test_acc
        best_mAP = test_mAP
        best_F1 = test_F1
        f.write('\nSaving models......')
        f.write("\nbest_PrivateTest_acc: %0.3f" % best_acc)
        f.write("\nbest_PrivateTest_mAP: %0.3f" % best_mAP)
        f.write("\nbest_PrivateTest_F1: %0.3f" % best_F1)
        state = {
            'epoch': epoch,
            'snet': snet.state_dict(),
            'test_acc': test_acc,
            'test_mAP': test_mAP,
            'test_F1': test_F1,
            'test_epoch': epoch,
        }
        torch.save(state, os.path.join(path, 'Student_Test_model.t7'))
    f.close()

f = open(path + '/' + args.data_name + '_' + args.t_model + '_N' + str(args.N_Teacher) + '_' + args.s_model + '_'
         + args.distillation + '.txt', 'a')
f.write("\n\n\n\nbest_PrivateTest_acc: %0.2f" % best_acc)
f.write("\nbest_PrivateTest_mAP: %0.2f" % best_mAP)
f.write("\nbest_PrivateTest_F1: %0.2f" % best_F1)
f.close()
writer.close()
