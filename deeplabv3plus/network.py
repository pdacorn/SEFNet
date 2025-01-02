# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from functools import partial
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from collections import OrderedDict
#from deeplabv3plus.config import config
from deeplabv3plus.resnet import resnet101,resnet152
from deeplabv3plus.Segformer import SegFormerHead

momentum=0.1





class Network(nn.Module):
    def __init__(self, num_classes, norm_layer, criterion=None, pretrained_model=None):
        super(Network, self).__init__()
        # self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
        #                           bn_eps=1e-5,
        #                           bn_momentum=momentum,
        #                           deep_stem=True, stem_width=64)
        # self.dilate = 2
        # for m in self.backbone.layer4.children():
        #     m.apply(partial(self._nostride_dilate, dilate=self.dilate))
        #     self.dilate *= 2
        self.encoder=encoder_bulid(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=1e-5,
                                  bn_momentum=momentum,
                                  deep_stem=True, stem_width=64)

        # self.head = Head(num_classes, norm_layer, momentum)
        self.head=SegFormerHead(dims=[256,512,1024,2048],embed_dim=256,num_classes=num_classes)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion
        droprate=0.5
        # self.classifier = nn.Sequential(*[nn.Dropout2d(droprate),
        #                                   nn.Conv2d(256, num_classes, kernel_size=1, bias=True)])
        self.classifier =nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        # self.business_layer.append(self.classifier)

    def forward(self, img,depth,is_feature=False,if_edge_f=False):

        with torch.no_grad():#6 3 480 640
            x_size = img.size()
            im_arr = img.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
            canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
            for i in range(x_size[0]):
                canny[i] = cv2.Canny(im_arr[i], 10, 100)
            canny = torch.from_numpy(canny).cuda().float()



        blocks,shape=self.encoder(img,depth)
        b, c, h, w = img.shape
        newshape=F.interpolate(shape, size=(h, w), mode='bilinear', align_corners=True)
        if is_feature:
            seg, edgeout=self.head(blocks, shape, newshape,canny, is_feature)#, canny
            return seg, edgeout
        if if_edge_f:
            pred, edgeout,edge_f = self.head(blocks, shape, newshape,canny, is_feature,if_edge_f)
        else:
            pred,edgeout = self.head(blocks,shape,newshape,canny,is_feature)# (6, 256, 120, 160)
        # out_dict = {}
        # out_dict['feat'] = v3plus_feature
        # v3plus_feature=self.L1(v3plus_feature)
        # pred = self.classifier(v3plus_feature)#2 40 120 160
        # out_dict['out'] = pred
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        #
        # if self.training:
        #     return v3plus_feature, pred

        if if_edge_f:
            return pred, edgeout,edge_f
        else:
            return pred,edgeout
    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
def encoder_bulid(pretrained_model=None, norm_layer=None,bn_eps=1e-5,bn_momentum=0.9,
                                  deep_stem=True, stem_width=64):
    model=Encoder(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=1e-5,
                                  bn_momentum=momentum,
                                  deep_stem=True, stem_width=64)
    return model


class Encoder(nn.Module):
    def __init__(self,pretrained_model=None, norm_layer=None,bn_eps=1e-5,bn_momentum=0.9,
                                  deep_stem=True, stem_width=64):
        super(Encoder, self).__init__()
        self.backbone =resnet101(pretrained_model, norm_layer=norm_layer,
                  bn_eps=bn_eps,
                  bn_momentum=bn_momentum,
                  deep_stem=deep_stem, stem_width=stem_width)
        # self.backbone = resnet152(pretrained_model, norm_layer=norm_layer,
        #                           bn_eps=bn_eps,
        #                           bn_momentum=bn_momentum,
        #                           deep_stem=deep_stem, stem_width=stem_width)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        stem_width = 64
        self.bn1_d = norm_layer(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.atten_rgb = self.channel_attention(64)
        self.atten_depth = self.channel_attention(64)
        self.l0=nn.Conv2d(128, 128, 1)
        self.dsn = nn.Conv2d(128, 64, 1)
        self.dsn0 = nn.Conv2d(256, 1, 1)
        self.dsn1 = nn.Conv2d(512, 1, 1)
        self.dsn2 = nn.Conv2d(1024, 1, 1)
        self.dsn3 = nn.Conv2d(2048, 1, 1)
        self.gate0 = GatedSpatialShapeConv2d(128, 128)
        self.gate1 = GatedSpatialShapeConv2d(128, 128)
        self.gate2 = GatedSpatialShapeConv2d(128, 128)
        self.gate3 = GatedSpatialShapeConv2d(128, 128)
        self.res0 = BasicBlock(128, 128, stride=1, downsample=None)
        self.d0 = nn.Conv2d(128, 128, 1)
        self.res1 = BasicBlock(128, 128, stride=1, downsample=None)
        self.d1 = nn.Conv2d(128, 128, 1)
        self.res2 = BasicBlock(128, 128, stride=1, downsample=None)
        self.d2 = nn.Conv2d(128, 128, 1)
        self.res3 = BasicBlock(128, 128, stride=1, downsample=None)
        self.d3 = nn.Conv2d(128, 128, 1)
    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        # bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid() # todo modify the activation function
        return nn.Sequential(*[pool, conv, activation])
    def forward(self, img,depth):
        shape0, blocks = self.backbone(img)
        shape0 = self.dsn(shape0)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu_d(depth)
        depth = self.maxpool_d(depth)  # 6 64 120 160
        x_size = depth.size()
        m0 = blocks[0]  # 6 256 120 160
        m1 = blocks[1]  # 6 512 60 80
        m2 = blocks[2]  # 6 1024 30 40
        m3 = blocks[3]  # 6 2048 30 40
        s0 = F.interpolate(self.dsn0(m0), x_size[2:],
                           mode='bilinear', align_corners=True)
        s1 = F.interpolate(self.dsn1(m1), x_size[2:],
                           mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.dsn2(m2), x_size[2:],
                           mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.dsn3(m3), x_size[2:],
                           mode='bilinear', align_corners=True)
        atten_rgb = self.atten_rgb(shape0)  # 6 64 1 1
        atten_depth = self.atten_depth(depth)  # 6 64 1 1
        shape1 = self.l0(torch.cat((shape0.mul(atten_rgb), depth.mul(atten_depth)), dim=1))
        shape1 = self.d0(shape1)#
        shape1 = self.gate0(shape1, s0)
        shape1 = self.res0(shape1)
        # cs = F.interpolate(cs, x_size[2:],
        #                    mode='bilinear', align_corners=True)
        shape1 = self.d1(shape1)
        shape1 = self.gate1(shape1, s1)
        shape1 = self.res1(shape1)# 1 128 120 160
        shape2 = self.d2(shape1)
        shape2 = self.gate2(shape2, s2)
        shape2 = self.res2(shape2)# 1 128 120 160
        shape3 = self.d3(shape2)
        shape3 = self.gate3(shape3, s3)
        shape3 = self.res3(shape3)# 1 128 120 160
        return blocks,shape3
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)#2 1024 30 28
        out = self.leak_relu(out)# add activation layer
        out = self.red_conv(out)#2 256 30 28

        # Global pooling
        pool = self._global_pooling(x)#2 2048 1 1
        pool = self.global_pooling_conv(pool)#2 256 1 1
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)#2 256 1 1
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))#2 256 30 40

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            # ????????? raise error??????
            pooling_size = (min((self.pooling_size, 0), x.shape[2]),
                            min((self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304+64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

        self.fuse=nn.Conv2d(64+1, 1, kernel_size=1, padding=0, bias=False)

        self.activation = nn.Sigmoid()
    def forward(self, f_list,shape,canny,newshape):
        f = f_list[-1]#6 2048 30 40
        f = self.aspp(f)#6 256 30 40

        low_level_features = f_list[0]#6 256 120 160
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)#6 48 120 160

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)#6 256 120 160
        f = torch.cat((f, low_level_features,shape), dim=1)#2 304 120 160
        f = self.last_conv(f)#2 256 120 160
        edgeout=self.fuse(torch.cat((canny,newshape), dim=1))
        edgeout=self.activation(edgeout)
        return f,edgeout
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GatedSpatialShapeConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialShapeConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
if __name__ == '__main__':
    pass
