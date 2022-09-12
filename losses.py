from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def adjust_lr(optimizer, epoch, args_lr):
    scale   = 0.1
    lr_list =  [args_lr] * 100
    lr_list += [args_lr*scale] * 50
    lr_list += [args_lr*scale*scale] * 50

    lr = lr_list[epoch-1]
    print ('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def confusion_matrix(preds, y, NUM_CLASSES=7):
    """ Returns confusion matrix """
    assert preds.shape[0] == y.shape[0], "1 dim of predictions and labels must be equal"
    rounded_preds = torch.argmax(preds,1)
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(rounded_preds.shape[0]):
        predicted_class = rounded_preds[i]
        correct_class = y[i]
        conf_mat[correct_class][predicted_class] += 1
    return conf_mat



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv0 = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True)
        self.ReLU0 = nn.ReLU(inplace=True)
        self.Dconv0 = torch.nn.PixelShuffle(2)
        self.DReLU0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=True)
        self.ReLU1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=True)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.Dconv1 = torch.nn.PixelShuffle(2)
        self.DReLU1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=True)
        self.ReLU3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=True)
        self.ReLU4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=True)
        self.ReLU5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=True)
        self.ReLU6 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ReLU0(self.conv0(x))
        x0 = self.DReLU0(self.Dconv0(x))
        x1 = self.ReLU1(self.conv1(x0)) + x0

        x = self.ReLU2(self.conv2(x1))
        x2 = self.DReLU1(self.Dconv1(x))
        x3 = self.ReLU3(self.conv3(x2)) + x2

        x = self.ReLU4(self.conv4(x3))
        x = torch.nn.functional.interpolate(x, size=[92, 92], mode='nearest', align_corners=None)

        x = self.ReLU5(self.conv5(x)) + x
        x = self.ReLU6(self.conv6(x))

        return x


def styleLoss(teacher_input, student_feature, MSE_crit):
    teacher_input = gram_matrix(teacher_input.cuda())
    student_feature = gram_matrix(student_feature.cuda())
    loss = MSE_crit(teacher_input, student_feature)
    return loss


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(c * d)



class KL_divergence(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence, self).__init__()
        self.T = temperature
    def forward(self, student_logit, teacher_logit):

        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logit/self.T,dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T

        return KD_loss




class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))






def Absdiff_Similarity(student, teacher):
    B, C, student_H, student_W = student.shape

    teacher_norm = teacher.norm(p=2, dim=2)
    student_norm = student.norm(p=2, dim=2)
    teacher_norm = torch.mean(teacher_norm, dim=2, keepdim=False)
    student_norm = torch.mean(student_norm, dim=2, keepdim=False)
    absdiff = torch.abs(teacher_norm - student_norm)

    teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)
    cosineSimilarity = torch.nn.CosineSimilarity(dim=2, eps=1e-6)(teacher, student)
    cosineSimilarity = 1 - cosineSimilarity
    cosineSimilarity = torch.mean(cosineSimilarity, dim=2, keepdim=False)

    total = absdiff + cosineSimilarity
    C = 0.6 * torch.max(total).item()
    loss = torch.mean(torch.where(total < C, total, (total * total + C * C) / (2 * C)))

    return loss














class FcLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, bias=True):
        super(FcLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.fc = nn.Linear(in_features=self.input_dim + self.hidden_dim, out_features=4 * self.hidden_dim, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=2)  # concatenate along channel axis

        combined_fc = self.fc(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_fc, self.hidden_dim, dim=2)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, N_categories):
        return (torch.zeros(batch_size, N_categories, self.hidden_dim, device=self.fc.weight.device),
                torch.zeros(batch_size, N_categories, self.hidden_dim, device=self.fc.weight.device))




class ConvLSTMCell_1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell_1D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, N_categories):
        return (torch.zeros(batch_size, self.hidden_dim, N_categories, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, N_categories, device=self.conv.weight.device))




class Shortcut(nn.Module):
    def __init__(self, N_Teacher, nf, future_step):
        super(Shortcut, self).__init__()
        self.future_step = future_step
        in_chan = int(nf/2)

        self.encoder_1_fclstm = FcLSTMCell(input_dim=N_Teacher, hidden_dim=in_chan, bias=True)
        self.encoder_2_fclstm = FcLSTMCell(input_dim=in_chan, hidden_dim=nf, bias=True)
        self.encoder_1_convlstm = ConvLSTMCell_1D(input_dim=nf, hidden_dim=nf, kernel_size=3, bias=True)
        self.encoder_2_convlstm = ConvLSTMCell_1D(input_dim=nf, hidden_dim=nf, kernel_size=3, bias=True)

        self.decoder_1_convlstm = ConvLSTMCell_1D(input_dim=nf, hidden_dim=nf, kernel_size=3, bias=True)
        self.decoder_2_convlstm = ConvLSTMCell_1D(input_dim=nf, hidden_dim=nf, kernel_size=3, bias=True)

        self.decoder_1_CNN2D = nn.Conv2d(in_channels=self.future_step, out_channels=1, kernel_size=3, padding=1)
        self.decoder_BN = nn.BatchNorm1d(nf)
        self.decoder_Tanh = nn.Tanh()

        self.decoder_fc1 = nn.Linear(nf, in_chan)
        self.decoder_fc2 = nn.Linear(in_chan, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        b, N_categories, seq_len = x.size()
        h_t, c_t = self.encoder_1_fclstm.init_hidden(batch_size=b, N_categories=N_categories)
        h_t2, c_t2 = self.encoder_2_fclstm.init_hidden(batch_size=b, N_categories=N_categories)
        for t in range(self.future_step):
            h_t, c_t = self.encoder_1_fclstm(x, cur_state=[h_t, c_t])
            h_t2, c_t2 = self.encoder_2_fclstm(input_tensor=h_t, cur_state=[h_t2, c_t2])
        encoder_fc_vector = h_t2

        h_t3, c_t3 = self.encoder_1_convlstm.init_hidden(batch_size=b, N_categories=N_categories)
        h_t4, c_t4 = self.encoder_2_convlstm.init_hidden(batch_size=b, N_categories=N_categories)
        encoder_fc_vector = encoder_fc_vector.permute(0, 2, 1)
        for t in range(self.future_step):
            h_t3, c_t3 = self.encoder_1_convlstm(encoder_fc_vector, cur_state=[h_t3, c_t3])
            h_t4, c_t4 = self.encoder_2_convlstm(input_tensor=h_t3, cur_state=[h_t4, c_t4])
        encoder_conv_vector = h_t4

        outputs = []
        h_t5, c_t5 = self.decoder_1_convlstm.init_hidden(batch_size=b, N_categories=N_categories)
        h_t6, c_t6 = self.decoder_2_convlstm.init_hidden(batch_size=b, N_categories=N_categories)
        for t in range(self.future_step):
            h_t5, c_t5 = self.decoder_1_convlstm(input_tensor=encoder_conv_vector, cur_state=[h_t5, c_t5])
            h_t6, c_t6 = self.decoder_2_convlstm(input_tensor=h_t5, cur_state=[h_t6, c_t6])
            decoder_vector = h_t6
            outputs += [decoder_vector]  # predictions
        outputs = torch.stack(outputs, 1)
        outputs = self.decoder_1_CNN2D(outputs).squeeze(1)
        outputs = self.decoder_Tanh(self.decoder_BN(outputs))

        outputs = outputs.permute(0,2,1)
        outputs = self.decoder_fc2(self.decoder_fc1(outputs))
        return outputs.squeeze(2)
