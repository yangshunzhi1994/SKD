from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import torch

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

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

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_ACC_evaluation(conf_mat_a, conf_mat_b, outputs, targets_a, targets_b, lam, NUM_CLASSES=None):
    
    conf_mat_a += confusion_matrix(outputs, targets_a, NUM_CLASSES)
    acc_a = sum([conf_mat_a[i, i] for i in range(conf_mat_a.shape[0])])/conf_mat_a.sum()
    precision_a = np.array([conf_mat_a[i, i]/(conf_mat_a[i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
    recall_a = np.array([conf_mat_a[i, i]/(conf_mat_a[:, i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
    mAP_a = sum(precision_a)/len(precision_a)
    F1_score_a = (2 * precision_a*recall_a/(precision_a+recall_a + 1e-10)).mean()

    conf_mat_b += confusion_matrix(outputs, targets_b, NUM_CLASSES)
    acc_b = sum([conf_mat_b[i, i] for i in range(conf_mat_b.shape[0])])/conf_mat_b.sum()
    precision_b = np.array([conf_mat_b[i, i]/(conf_mat_b[i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
    recall_b = np.array([conf_mat_b[i, i]/(conf_mat_b[:, i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
    mAP_b = sum(precision_b)/len(precision_b)
    F1_score_b = (2 * precision_b*recall_b/(precision_b+recall_b + 1e-10)).mean()

    acc = lam * acc_a  +  (1 - lam) * acc_b
    mAP = lam * mAP_a  +  (1 - lam) * mAP_b
    F1_score = lam * F1_score_a  +  (1 - lam) * F1_score_b
    
    return conf_mat_a, conf_mat_b, acc, mAP, F1_score

def ACC_evaluation(conf_mat, outputs, targets, NUM_CLASSES=None):
    
    conf_mat += confusion_matrix(outputs, targets, NUM_CLASSES)
    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
    precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
    mAP = sum(precision)/len(precision)

    recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = 2 * precision*recall/(precision+recall + 1e-10)
    F1_score = f1.mean()
    
    return conf_mat, acc, mAP, F1_score

def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def load_pretrained_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Top_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('.............Invalid.v............')



