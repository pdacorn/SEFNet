
import argparse
import os
from deeplabv3plus.network import Network
import numpy as np
from Flow.fc_flow import flow_seg_decoder,SegLieaner,flow_model
import datetime
import torchvision
import skimage.transform
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import time
import torch, math
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from utils.init_func import init_weight, group_weight

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
import SEFNet_data_nyuv2_eval as SEFNet_data_eval
from torch import nn
from sklearn.metrics import auc, precision_recall_curve
from skimage.measure import label, regionprops
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import SEFNet_models_V1
import SEFNet_data_nyuv2 as SEFNet_data
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from torch.optim.lr_scheduler import LambdaLR

# nyuv2_frq = [0.04636878, 0.10907704, 0.152566  , 0.28470833, 0.29572534,
#        0.42489686, 0.49606689, 0.49985867, 0.45401091, 0.52183679,
#        0.50204292, 0.74834397, 0.6397011 , 1.00739467, 0.80728748,
#        1.01140891, 1.09866549, 1.25703345, 0.9408835 , 1.56565388,
#        1.19434108, 0.69079067, 1.86669642, 1.908     , 1.80942453,
#        2.72492965, 3.00060817, 2.47616595, 2.44053651, 3.80659652,
#        3.31090131, 3.9340523 , 3.53262803, 4.14408881, 3.71099056,
#        4.61082739, 4.78020462, 0.44061509, 0.53504894, 0.21667766]
nyuv2_frq = []
weight_path = './data/nyuv2_40class_weight.txt'
with open(weight_path,'r',encoding='utf-8') as f:
    context = f.readlines()

for x in context[1:]:
    x = x.strip().strip('\ufeff')
    nyuv2_frq.append(float(x))


parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--last-ckpt', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--Encoder-dir', default='./model/SEFnetEncoder.pth', metavar='DIR',
                    help='pretrain Encoder of SEFNet')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
log_theta = torch.nn.LogSigmoid()


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
def calculate_pro_metric(scores, gt_mask):
    """
    calculate segmentation AUPRO, from https://github.com/YoungGod/DFR
    """
    max_step = 1000
    expect_fpr = 0.3  # default 30%
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = label(gt_mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            if gt_mask[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        # print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        gt_masks_neg = ~gt_mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)
    # best per image iou
    best_miou = ious_mean.max()
    # print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pix_pro_auc = auc(fprs_selected, pros_mean_selected)

    return pix_pro_auc
def convert_to_anomaly_scores(args, logps_list):
    normal_map = [list() for _ in range(args.feature_levels)]
    for l in range(args.feature_levels):
        logps = torch.cat(logps_list[l], dim=0)
        logps -= torch.max(logps)  # normalize log-likelihoods to (-Inf:0] by subtracting a constant
        probs = torch.exp(logps)  # convert to probs in range [0:1]
        # upsample
        normal_map[l] = F.interpolate(probs.unsqueeze(1),
                                      size=args.img_size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()

    # score aggregation
    scores = np.zeros_like(normal_map[0])
    for l in range(args.feature_levels):
        scores += normal_map[l]

    # normality score to anomaly score
    scores = scores.max() - scores

    return scores
class scaleNorm(object):
    def __init__(self,is_city=False):
        self.is_city=is_city
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if self.is_city:
            image_h=image.shape[0]
            image_w=image.shape[1]
        else:
            image_w = 640
            image_h = 480
        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        ind = label == 0
        label = label - 1
        label[ind] = 255
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float32)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).long()}

class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth

        return sample
def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None
def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)

    return P

def adjust_learning_rate(c, optimizer, epoch):
    lr = 2e-4
    lr_decay_rate=0.1
    meta_epochs=25
    eta_min = lr * (lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / meta_epochs)) / 2


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def normal_fl_weighting(logps, gamma=0.5, alpha=11.7, normalizer=10):
    """
    Normal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -0.2
    mask_lower = logps <= -0.2
    probs = torch.exp(logps)
    fl_weights = alpha * (1 - probs).pow(gamma) * torch.abs(logps)
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_larger] = 1.0
    weights[mask_lower] = fl_weights[mask_lower]

    return weights

def abnormal_fl_weighting(logps, gamma=2, alpha=0.53, normalizer=10):
    """
    Abnormal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -1.0
    mask_lower = logps <= -1.0
    probs = torch.exp(logps)
    fl_weights = alpha * (1 + probs).pow(gamma) * (1 / torch.abs(logps))
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_lower] = 1.0
    weights[mask_larger] = fl_weights[mask_larger]

    return weights

def get_logp_boundary(logps, mask, pos_beta=0.05, margin_tau=0.1, normalizer=10):
    """
    Find the equivalent log-likelihood decision boundaries from normal log-likelihood distribution.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
        margin_tau: margin hyperparameter: tau
    """
    normal_logps = logps[mask == 0].detach()
    n_idx = int(((mask == 0).sum() * pos_beta).item())
    sorted_indices = torch.sort(normal_logps)[1]

    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary
    b_n = b_n / normalizer

    b_a = b_n - margin_tau  # abnormal boundary

    return b_n, b_a


def calculate_bg_spp_loss_normal(logps, mask, boundaries, normalizer=10, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights

    loss_n = torch.mean(loss_n)

    return loss_n
def calculate_bg_spp_loss(logps, mask, boundaries, normalizer=10, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    b_a = boundaries[1]
    anomaly_logps = logps[mask == 1]
    anomaly_logps_inter = anomaly_logps[anomaly_logps >= b_a]
    loss_a = anomaly_logps_inter - b_a

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights
        ano_weights = weights[mask == 1][anomaly_logps >= b_a]
        loss_a = loss_a * ano_weights

    loss_n = torch.mean(loss_n)
    loss_a = torch.mean(loss_a)

    return loss_n, loss_a

def evaluate_thresholds(gt_label, gt_mask, img_scores, scores):
    precision, recall, thresholds = precision_recall_curve(gt_label, img_scores)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    image_threshold = thresholds[np.argmax(f1)]

    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    pixel_threshold = thresholds[np.argmax(f1)]

    return image_threshold, pixel_threshold
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
    # else:
    #     for i, iou in enumerate(j_list):
    #         if i==0:
    #             print('class {:2d} IU {:.2f}'.format(i, j_list[i]))
    return aveJ
def save_model_weight(encoder, decoder,head,exp_name):
    filename = '..//Flow_Model_'+exp_name+'.pt'
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': decoder.state_dict(),
             'head_state_dict': head.state_dict()
             }
    torch.save(state, filename)
    print('Saving weights success!')
def load_weights(encoder, decoder,head, filename):
    #path = os.path.join(WEIGHT_DIR, filename)
    state = torch.load(filename)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoder.load_state_dict(state['decoder_state_dict'], strict=False)
    head.load_state_dict(state['head_state_dict'], strict=False)
    print('Loading weights success!')
def validate(args,epoch, data_loader, encoder, decoder,head):
    # print('\nCompute loss and scores on category: {}'.format(args.class_name))

    # decoders = [decoder.eval() for decoder in decoders]
    decoders= decoder.eval()
    # encoder=encoder.eval()
    head=head.eval()
    val_num = len(data_loader)
    data_list = []
    image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            start = datetime.datetime.now()
            args.num_class = 40
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].numpy()
            features,_= encoder(image,depth,is_feature=True)
             # BxCxHxW
            e = features
            bs, dim, h, w = e.size()
            e = e.permute(0, 2, 3, 1).reshape(-1, dim)
            # pred = decoders(e)
            z, log_jac_det = decoders(e)
            pred = head(z, log_jac_det)
            pred=pred.reshape(-1, dim)
            pred = pred.reshape(bs, h, w, 40).permute(0, 3, 1, 2)
            pred = F.interpolate(pred, size=(label.shape[1], label.shape[2]), mode='bilinear',
                                        align_corners=True)
            pred = pred.cpu().data[0].numpy()
            label = np.asarray(label[0], dtype=np.int_)
            pred = pred.transpose(1, 2, 0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.int_)
            data_list.append([label.reshape(-1), pred.reshape(-1)])
            end = datetime.datetime.now()
            running_time = (end - start) * (val_num - batch_idx - 1)
            running_time = str(running_time)[:7]
            print(
                '\rVal [%d/%d] Running time: %s' % (
                    batch_idx + 1, val_num, running_time), end='')
            image = image.to('cpu')
            depth = depth.to('cpu')
            # label = label.to('cpu')
            del image, depth, label, pred
    print()
    iou = get_iou(data_list, class_num=40)
    del data_list
    return iou.mean()
def train():
    args.meta_epochs=8
    args.vis=False
    best=0.0
    flag=False
    train_data = SEFNet_data.SUNRGBD(transform=transforms.Compose([#SEFNet_data.scaleNorm(),
                                                                   # SEFNet_data.RandomScale((1.0, 1.4)),
                                                                   # SEFNet_data.RandomHSV((0.9, 1.1),
                                                                   #                       (0.9, 1.1),
                                                                   #                       (25, 25)),
                                                                   # SEFNet_data.RandomCrop(image_h//2, image_w//2),
                                                                   # SEFNet_data.RandomFlip(),
                                                                   # SEFNet_data.gaussian_blur(),
                                                                   SEFNet_data.ToTensor()
                                                                  ,SEFNet_data.to_edge()
                                                                     # ,SEFNet_data.Normalize()
                                                                  ]
                                                                 ),
                                     phase_train=True,
                                     data_dir=args.data_dir)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    val_data = SEFNet_data_eval.SUNRGBD(transform=torchvision.transforms.Compose([scaleNorm(),
                                                                            ToTensor()
                                                                            # ,Normalize()
                                                                                 ]),
                                  phase_train=False,
                                  data_dir=args.data_dir
                                  )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    num_train = len(train_data)

    pretrained_model=None
    model=Network(40, norm_layer=SynchronizedBatchNorm2d, pretrained_model=pretrained_model)
    if args.Encoder_dir!=None:
        model.encoder.load_state_dict(torch.load(args.Encoder_dir))
    else:
        print("Encoder_dir cannot be null")
        return
    model.to(device)

    base_lr = 1e-5
    params_list_l = []
    encoder=model
    global_step = 0
    in_channels=256
    head=SegLieaner(in_channels, 40)
    decoder= flow_model(args, in_channels)
    head.to(device)
    decoder.to(device)
    params=[]

    params.append(dict(params=decoder.parameters(), lr=2e-4))
    params.append(dict(params=head.parameters(), lr=2e-4))
    optimizer=torch.optim.Adam(params, lr=args.lr)
    CEL_weighted =utils.CrossEntropyLoss2d(weight=nyuv2_frq).to(device)

    args.pos_embed_dim=128
    args.coupling_layers=8
    args.focal_weighting=True
    args.pos_beta=0.05
    args.margin_tau=0.1
    args.normalizer=10
    args.bgspp_lambda=1
    args.img_size=(480,640)
    args.class_name="wall"
    args.pro=True
    N_batch = 120*160
    best=0.0
    best_Miou=0
    Losss=[]
    Mious=[]
    train_meaniou=[]
    encoder.eval()
    for epoch in range(int(args.start_epoch), args.epochs):
        total_loss, loss_count = 0.0, 0
        decoder.train()
        head.train()
        adjust_learning_rate(args, optimizer, epoch)
        for batch_idx, sample in enumerate(train_loader):
            start = datetime.datetime.now()
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            dege_label=sample['dege_pred'].to(device)
            mask=sample['label']
            mask=mask.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feature,edgeout = encoder(image.detach(), depth.detach(),is_feature=True)
                e = feature.detach()
            bs, dim, h, w = e.size()
            e = e.permute(0, 2, 3, 1).reshape(-1, dim)
            perm = torch.randperm(bs * h * w).to(device)
            inv_perm = torch.empty_like(perm).to(device)
            inv_perm[perm] = torch.arange(len(perm)).to(device)
            num_N_batches=bs*h*w // N_batch
            pre_list=[]
            for i in range(num_N_batches):
                idx = torch.arange(i * N_batch, (i + 1) * N_batch)
                e_b = e[perm[idx]]
                z, log_jac_det= decoder(e_b)
                pre = head(z, log_jac_det)
                pre_list.append(pre)
            pre_finally=torch.stack(pre_list)
            pre_finally=pre_finally.reshape(-1,pre_finally.size()[2])[inv_perm]
            pre_finally=pre_finally.reshape(bs, h, w, 40).permute(0, 3, 1, 2)
            pre_finally=F.interpolate(pre_finally, size=(mask.size()[1], mask.size()[2]), mode='bilinear', align_corners=True)
            loss,_=CEL_weighted(pre_finally,mask.detach())
            loss.backward()
            optimizer.step()
            p = (batch_idx * args.batch_size / num_train) * 100
            show_str = ('[%%-%ds]' % 65) % (int(65 * p / 100) * ">")
            end = datetime.datetime.now()
            running_time = (end - start) * (
                    (num_train // args.batch_size) - batch_idx)
            running_time = str(running_time)[:7]
            print(
                '\rEpoch [%d/%d] %d/%d %s %d%% Running time: %s [Training] Loss %.4f' % (
                    epoch + 1, args.epochs, batch_idx * args.batch_size, num_train, show_str, p, running_time,
                    float(loss.data)), end='')
            Losss.append(float(loss.data))
        print()

        if (epoch+1)%5==0 or flag:
            Miou=validate(args,epoch, val_loader, encoder, decoder,head)
            if best_Miou<Miou:
                best_Miou=Miou
                save_model_weight(encoder, decoder, head, 'best')
            print("Val Current MIOU=", Miou)
            print("Val best MIOU=", best_Miou)

    print("Training completed ")

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    train()
