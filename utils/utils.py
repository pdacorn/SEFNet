import numpy as np
from torch import nn
import torch
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable


med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]


class CrossEntropyLoss2d_eval(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d_eval, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        inputs = inputs_scales
        targets = targets_scales
        # for inputs, targets in zip(inputs_scales, targets_scales):
        mask = targets > 0
        targets_m = targets.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, ignore_label=255,weight=None,is_city=False):
        super(CrossEntropyLoss2d, self).__init__()
        self.ignore_label = ignore_label
        if is_city:
            self.weight =None
        else:
            self.weight=torch.tensor(weight).to('cuda:0')
    def bce2d(self, input, target):
        n, c, h, w = input.size()


        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss
    def bceNd(self, input, target,classes=41):
        n, c, h, w = input.size()

        target=torch.squeeze(target)
        weight = torch.Tensor(classes).fill_(0)
        weight = weight.numpy()
        sum_num=torch.sum(target!=255)
        t=torch.unique(target)
        for i in t:
            if i!=255:
                i_sum=torch.sum(target==i)
                weight[i]=1-i_sum * 1.0 / sum_num
        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.cross_entropy(input, target, weight=weight, size_average=True, ignore_index=self.ignore_label,
                                reduction='mean')
        return loss
    def forward(self, predict, target,dege_pred=None,edge_label=None):
        losses = []

        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        # target = torch.LongTensor(target)
        loss1 = F.cross_entropy(predict, target, weight=self.weight, size_average=True, ignore_index=self.ignore_label,
                                reduction='mean')
        # loss2 = self.rce(predict, target.clone())
        losses.append(float(loss1.data))
        if dege_pred!=None:
            loss2=self.bceNd(dege_pred,edge_label)
            # loss2=self.bce2d(dege_pred,edge_label)
            losses.append(float(loss2.data))
            loss1=loss1*5+loss2#loss1+loss2*20.0
        # loss = 0.1 * loss1 + loss2
        return loss1,losses

# hxx add, focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 size_average=self.size_average, reduce=False)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)


class FocalLoss2d(nn.Module):
    def __init__(self, weight=med_frq, gamma=0):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.fl_loss = FocalLoss(gamma=self.gamma, weight=self.weight, size_average=False)

    def forward(self, predict, target, weight=None):
        losses = []
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3

        loss1 = F.cross_entropy(predict, target, weight=weight, size_average=True, ignore_index=self.ignore_label,
                                reduction='mean')
        total_loss = sum(losses)
        return total_loss


def color_label_eval(label):
    # label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    # return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    return colored.transpose([0, 2, 1])

def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train,save_filename=None):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if  save_filename==None:
        ckpt_model_filename = "ckpt_epoch_best.pth"
    else:
        ckpt_model_filename=save_filename+".pth"
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' "
              .format(model_file))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)




def accuracy(preds, label):
    valid = (label < 40) # hxx
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == -1:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard_perclass = []
        for i in range(self.nclass):
            jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m
def get_meaniou(segmentation_result, y, n_classes = 19):
    iou = []
    iou_sum = 0
    segmentation_result = segmentation_result.view(-1)
    y = y.view(-1)
    classes=torch.unique(y)
    #print(classes)

    for cls in range(1, n_classes):
        if cls not in classes:
            n_classes-=1
            continue
        result_inds = segmentation_result == cls
        y_inds = y == cls
        intersection = (result_inds[y_inds]).long().sum().data.cpu().item()
        union = result_inds.long().sum().data.cpu().item() + y_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            iou.append(float('nan'))
        else:
            iou.append(float(intersection) / float(max(union, 1)))
            iou_sum += float(intersection) / float(max(union, 1))
    #print(iou)
    del segmentation_result,y
    return iou_sum/n_classes
def get_iou(data_list, class_num,is_print=False):
    from multiprocessing import Pool

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    if is_print:
        for i, iou in enumerate(j_list):
            print('class {:2d} IU {:.2f}'.format(i, j_list[i]))
    return aveJ

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg