import argparse
import torch
import torch.nn as nn
import imageio
import torch.nn.functional as F
from Flow.fc_flow import flow_seg_decoder,SegLieaner,flow_model
from torch.autograd import Variable

import skimage.transform
from scipy.ndimage import distance_transform_edt
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torchvision
import numpy as np
from deeplabv3plus.network import Network
import os
from torch.utils.data import DataLoader
import datetime
import cv2
import torch.optim
import SEFNet_data_nyuv2_eval as SEFNet_data

import SEFNet_models_V1
from utils import utils
from utils.utils import load_ckpt, get_iou, AverageMeter, accuracy
import os
os.environ["PATH"] += os.pathsep + 'D:\\little app\\Graphviz\\bin'
parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output', default='../visualization', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='.\model\Flow_Model.pt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=40, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=False, action='store_true',#False
                    help='if output image')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
img_mean=[0.485, 0.456, 0.406]
img_std=[0.229, 0.224, 0.225]

# transform
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
class add_gaussian_noise(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        mean=0
        stddev=20
        noise = np.random.normal(mean, stddev, depth.shape).astype(np.float32)
        depth = depth + noise
        #depth = np.expand_dims(depth, 0).astype(np.float32)
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

# def visualize_result(img, label, preds, info, args):
#     # segmentation
#     img = img.squeeze(0).transpose(0, 2, 1)
#     seg_color = utils.color_label_eval(label)
#
#     # prediction
#     pred_color = utils.color_label_eval(preds)
#
#     # aggregate images and save
#     im_vis = np.concatenate((img, seg_color, pred_color),
#                             axis=1).astype(np.uint8)
#     im_vis = im_vis.transpose(2, 1, 0)
#
#     img_name = str(info)
#     # print('write check: ', im_vis.dtype)
#     cv2.imwrite(os.path.join(args.output,
#                 img_name+'.png'), im_vis)
class_colors = {
    0: (66, 131, 233),
    1: (137, 75, 121),
    2: (170, 254, 52),
    3: (80, 78, 228),
    4: (49, 92, 152),
    5: (16, 35, 137),
    6: (77, 22, 222),
    7: (15, 210, 206),
    8: (239, 42, 226),
    9: (230, 9, 248),
    10: (92, 178, 167),
    11: (15, 218, 211),
    12: (104, 205, 42),
    13: (216, 68, 210),
    14: (56, 205, 184),
    15: (165, 203, 97),
    16: (149, 6, 5),
    17: (40, 45, 34),
    18: (181, 42, 27),
    19: (126, 61, 151),
    20: (24, 71, 175),
    21: (37, 14, 47),
    22: (71, 129, 109),
    23: (51, 161, 93),
    24: (40, 0, 145),
    25: (242, 202, 13),
    26: (71, 78, 172),
    27: (23, 214, 200),
    28: (211, 255, 170),
    29: (222, 160, 77),
    30: (29, 175, 13),
    31: (75, 163, 13),
    32: (195, 220, 87),
    33: (56, 25, 31),
    34: (246, 185, 111),
    35: (236, 60, 219),
    36: (135, 48, 223),
    37: (206, 59, 147),
    38: (47, 24, 25),
    39: (170, 5, 60),
    40: (116, 248, 28)
}
def to_edge( mask):
    label=mask
    _edgemap = mask
    num_classes = 40
    _edgemap = mask_to_onehot(_edgemap, num_classes)
    dege_pred=onehot_to_binary_edges(_edgemap, 2, num_classes,label)
    # sample['edge'] = torch.from_numpy(edge).float()
    return dege_pred
def mask_to_onehot(mask, num_classes):
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_binary_edges(mask, radius, num_classes,label):

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist >= radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    mask_zero_indices=label==255
    # edgemap_copy=edgemap.copy()
    edgemap[0][mask_zero_indices]=2
    # edgemap_copy[0][mask_zero_indices]=0
    # edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = np.squeeze(edgemap, axis=0)
    return edgemap
def visualize_result(img, depth, label, preds, info, args,edge_pred=None):
    # segmentation
    img = img.squeeze(0).transpose(1, 2, 0).astype(np.uint8)
    dep = depth.squeeze(0)
    dep = ((dep-dep.min())*255.0/(dep.max()-dep.min())).astype(np.uint8)
    # dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    dep = dep.transpose(1, 2, 0)
    img_name = str(info)
    new_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    new_preds = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    new_edge_pred = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    list = np.unique(label)  # .tolist()
    for i in list:
        ind = i == label
        if i==255:
            new_label[ind] = class_colors[40]
        else:
            new_label[ind] = class_colors[i]
    preds[label==255]=40
    list = np.unique(preds)
    for i in list:
        ind = i == preds
        new_preds[ind] = class_colors[i]
    if type(edge_pred)==np.ndarray:
        new_edge_pred[edge_pred>=0.9]=(0, 0, 0)
        new_edge_pred[edge_pred < 0.9] = (255, 255, 255)
        new_edge_pred[label==255]=(255, 255, 255)
        cv2.imwrite(os.path.join(args.output, img_name + '_pred_edge.png'), new_edge_pred, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    edge=to_edge(label)
    new_edge = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    list = np.unique(edge)  # .tolist()
    for i in list:
        if i == 1:
            ind = i == edge
            new_edge[ind] = (0, 0, 0)
        else:
            ind = i == edge
            new_edge[ind] = (255, 255, 255)
    # cv2.imshow('new_preds', new_preds)
    # cv2.imshow('new_label', new_label)
    # cv2.imshow('ges and save
    #     # im_vis = np.concatenate((img, dep, seg_color, pred_color),
    #     #                         axis=1).astype(np.uint8)
    #     # im_vis = im_vis.transpose(2, 1, 0)image', img)
    # cv2.imshow('depth', dep)
    # seg_color = utils.color_label_eval(label)
    # prediction
    # pred_color = utils.color_label_eval(preds)

    # aggregate ima


    # print('write check: ', im_vis.dtype)
    # cv2.imwrite(os.path.join(args.output,
    #             img_name+'.png'), im_vis)
    # cv2.imwrite(os.path.join(args.output, img_name + '_edge.png'), new_edge, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    # cv2.imwrite(os.path.join(args.output,img_name+'_image.png'), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    # cv2.imwrite(os.path.join(args.output,img_name+'_label.png'), new_label, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    # cv2.imwrite(os.path.join(args.output,img_name+'_our_pred.png'), new_preds, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    # cv2.imwrite(os.path.join(args.output,img_name+'_depth.png'), dep, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def inference():

    model = Network(40, norm_layer=SynchronizedBatchNorm2d, pretrained_model=None)

    #ACNet_models_V1.ACNet(num_class=40, pretrained=False)
    # load_ckpt(model, None, args.last_ckpt, device)

    in_channels = 256
    args.pos_embed_dim = 128
    args.coupling_layers = 8
    args.focal_weighting = True
    args.pos_beta = 0.05
    args.margin_tau = 0.1
    args.normalizer = 10
    args.bgspp_lambda = 1
    args.img_size = (480, 640)
    args.class_name = "wall"
    args.pro = True
    head = SegLieaner(in_channels, 40)
    decoder = flow_model(args, in_channels)

    # from thop import profile
    # # model.to(device)
    # decoder.to(device)
    # input_tensor1 = torch.randn(1, 3, 480, 640).to('cuda:0')
    # input_tensor2 = torch.randn(1,1, 480, 640).to('cuda:0')
    # input_tensor3 = torch.randn(19200, 256).to('cuda:0')
    # # sample=input_tensor1,input_tensor2
    # # flops, params = profile(model, inputs=(input_tensor1,input_tensor2, ))
    # flops, params = profile(decoder, inputs=(input_tensor3, ))
    # print(flops / 1e9, params / 1e6)  # flops单位G，para单位M
    # return

    load_weights(model, decoder,head, args.last_ckpt)
    model.eval()
    model.to(device)
    head.eval()
    head.to(device)
    decoder.eval()

    decoder.to(device)
    # scale_list = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    val_data = SEFNet_data.SUNRGBD(transform=torchvision.transforms.Compose([
                                                                    scaleNorm(),
                                                                    #add_gaussian_noise(),
                                                                   ToTensor(),
                                                                   # Normalize()
                                                                            ]),
                                   phase_train=False,
                                   data_dir=args.data_dir
                                   )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,num_workers=1, pin_memory=True)
    data_list = []
    acc_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            #todo batch=1，这里要查看sample的size，决定怎么填装image depth label，估计要用到for循环

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].numpy()
            origin_image = sample['image'].numpy()
            origin_depth = sample['depth'].numpy()

            with torch.no_grad():
                features,_ = model(image, depth, is_feature=True)
                # BxCxHxW
                e = features
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                z, log_jac_det = decoder(e)
                pred = head(z, log_jac_det)
                # pred = decoder(e)
            pred = pred.reshape(-1, dim)
            pred = pred.reshape(bs, h, w, 40).permute(0, 3, 1, 2)
            pred = F.interpolate(pred, size=(label.shape[1], label.shape[2]), mode='bilinear',
                                 align_corners=True)
            pred = pred.cpu().data[0].numpy()
            label = np.asarray(label[0], dtype=np.int_)
            pred = pred.transpose(1, 2, 0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.int_)
            # pred=pred.cpu().data[0].numpy()
            # label = np.asarray(label[0], dtype=np.int_)
            # pred = pred.transpose(1, 2, 0)
            # pred = np.asarray(np.argmax(pred, axis=2), dtype=np.int_)
            # edge_pred=edge_pred.squeeze(0).squeeze(0).cpu().numpy()
            data_list.append([label.reshape(-1), pred.reshape(-1)])
            acc, pix = accuracy(pred, label)
            # intersection, union = intersectionAndUnion(output, label, args.num_class)
            acc_meter.update(acc, pix)
            # a_m, b_m = macc(output, label, args.num_class)
            # intersection_meter.update(intersection)
            # union_meter.update(union)
            # a_meter.update(a_m)
            # b_meter.update(b_m)
            print('[{}] iter {}, accuracy: {}'
                  .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          batch_idx, acc))

            # img = image.cpu().numpy()
            # print('origin iamge: ', type(origin_image))
            if args.visualize:
                visualize_result(origin_image, origin_depth, label, pred, batch_idx, args,None)
            image = image.to('cpu')
            depth = depth.to('cpu')
            # label = label.to('cpu')
            del image, depth, label, pred

    # iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    # for i, _iou in enumerate(iou):
    #     print('class [{}], IoU: {}'.format(i, _iou))
    iou = get_iou(data_list, class_num=40,is_print=True)
    # mAcc = (a_meter.average() / (b_meter.average()+1e-10))
    # print(mAcc.mean())
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average() * 100))
        # imageio.imsave(args.output, output.cpu().numpy().transpose((1, 2, 0)))

if __name__ == '__main__':
    # filename = os.path.join('./', "test.txt")
    # filename1 = os.path.join('./', "ltest.txt")
    # lines = []
    # with open(filename, 'r') as file:
    #     for line in file:
    #         # 去除行尾的换行符
    #         line = line.strip()
    #         lines.append(line)
    # with open(filename1, 'w') as file:
    #     for line in lines:
    #         line=line.lstrip('0')
    #         file.write(line + '\n')
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    inference()


