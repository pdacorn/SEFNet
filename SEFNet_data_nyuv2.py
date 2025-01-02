import numpy as np
import scipy.io
import imageio
from skimage import io
import h5py
from scipy.ndimage import distance_transform_edt
import cv2
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
image_h = 480
image_w = 640


train_file = './data/train.txt'
test_file = './data/test.txt'

def make_dataset_fromlst(listfilename):
    """
    NYUlist format:
    imagepath seglabelpath depthpath HHApath
    """
    images = []
    segs = []
    depths = []
    with open(listfilename) as f:
        content = f.readlines()
        for x in content:

            x = x.rstrip('\n')
            # imgname = os.path.join('./data', 'images', x + '.npy')
            # segname = os.path.join('./data', 'labels', x + '.npy')
            # depthname = os.path.join('./data', 'depths', x + '.npy')
            x = "{:0>{width}}".format(x, width=6)
            imgname = os.path.join('./data/new_data', 'image', x + '.PNG')
            segname = os.path.join('./data/new_data', 'label40', x + '.PNG')
            depthname = os.path.join('./data/new_data', 'depth', x + '.PNG')
            images += [imgname]
            segs += [segname]
            depths += [depthname]
    return {'images':images, 'labels':segs, 'depths':depths}

class SUNRGBD(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):

        self.phase_train = phase_train
        self.transform = transform

        result = make_dataset_fromlst(train_file)
        self.img_dir_train = result['images']
        self.depth_dir_train = result['depths']
        self.label_dir_train = result['labels']

        result = make_dataset_fromlst(test_file)
        self.img_dir_test = result['images']
        self.depth_dir_test = result['depths']
        self.label_dir_test = result['labels']

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test


        # image = cv2.imread(img_dir[idx]).astype(np.float32)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label = cv2.imread(label_dir[idx], cv2.IMREAD_GRAYSCALE)
        # # label -= 1
        # depth = cv2.imread(depth_dir[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)

        # label = np.load(label_dir[idx]).astype(np.uint8)
        # depth = np.load(depth_dir[idx]).astype(np.float32)
        # image = np.load(img_dir[idx]).astype(np.float32)
        depth = cv2.imread(depth_dir[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
        div = np.max(depth) - np.min(depth)
        min = np.min(depth)
        scaling_factor = 255.0 / div
        depth = depth - min
        depth = depth * scaling_factor
        image= cv2.imread(img_dir[idx]).astype(np.float32)
        label =cv2.imread(label_dir[idx], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __init__(self,is_city=False):
        self.is_city=is_city
    def __call__(self, sample):

        image, depth, label = sample['image'], sample['depth'], sample['label']
        if self.is_city:
            image_h=image.shape[0]
            image_w=image.shape[1]
        else:
            image_h = 480
            image_w = 640
        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale=scale
        self.scale_list=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if self.scale:
            from random import choice
            target_scale = choice(self.scale_list)
            # (H, W, C)
            target_height = int(round(target_scale * image.shape[0]))
            target_width = int(round(target_scale * image.shape[1]))
            # Bi-linear
            image = skimage.transform.resize(image, (target_height, target_width),
                                             order=1, mode='reflect', preserve_range=True)
            # Nearest-neighbor
            depth = skimage.transform.resize(depth, (target_height, target_width),
                                             order=0, mode='reflect', preserve_range=True)
            label = skimage.transform.resize(label, (target_height, target_width),
                                             order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


def RandomFlip(sample):
    image, depth, label,dege_pred = sample['image'], sample['depth'], sample['label'],sample['dege_pred']
    for i in range(image.shape[0]):
        if random.random() > 0.5:
            image[i] = torch.flip(image[i], dims=[2])
            depth[i] = torch.flip(depth[i], dims=[2])
            label[i] = torch.flip(label[i], dims=[1])
            dege_pred[i] = torch.flip(dege_pred[i], dims=[2])
    for i in range(image.shape[0]):
        if random.random() > 0.5:
            image[i] = torch.flip(image[i], dims=[1])
            depth[i] = torch.flip(depth[i], dims=[1])
            label[i] = torch.flip(label[i], dims=[0])
            dege_pred[i] = torch.flip(dege_pred[i], dims=[1])
    # if random.random() > 0.5:
    #     image = torch.rot90(image, k=1,dims=[2,3])
    #     depth = torch.rot90(depth, k=1,dims=[2,3])
    #     label = torch.rot90(label, k=1,dims=[1,2])
    #     dege_pred = torch.rot90(dege_pred, k=1,dims=[2,3])
    return {'image': image, 'depth': depth, 'label': label,'dege_pred':dege_pred}
def RandomCrop( sample):
    image, depth, label,dege_pred = sample['image'], sample['depth'], sample['label'],sample['dege_pred']
    th=image.shape[2]//2
    tw=image.shape[3]//2
    h = image.shape[2]
    w = image.shape[3]
    t=random.random()*3.0
    if t <1.0:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return {'image': image[:,:,i:i + th, j:j + tw],
                'depth': depth[:,:,i:i + th, j:j + tw],
                'label': label[:,i:i + th, j:j +tw],
                'dege_pred':dege_pred[:,:,i:i + th, j:j +tw]}
    elif (t<2.0 and t>=1.0):
        newth=(th+h)//2
        newtw=(tw+h)//2
        i = random.randint(0, h - newth)
        j = random.randint(0, w - newtw)

        return {'image': image[:,:,i:i + newth, j:j + newtw],
                'depth': depth[:,:,i:i + newth, j:j + newtw],
                'label': label[:,i:i + newth, j:j + newtw],
                'dege_pred':dege_pred[:,:,i:i + newth, j:j + newtw]
                }
    else:
        return {'image': image,
                'depth': depth,
                'label': label,
                'dege_pred':dege_pred}





# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        ind = label == 0
        label = label - 1
        label[ind] = 255
        # scale=1
        # Generate different label scales
        # label2 = skimage.transform.resize(label, (label.shape[0] // 2*scale, label.shape[1] // 2*scale),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label3 = skimage.transform.resize(label, (label.shape[0] // 4*scale, label.shape[1] // 4*scale),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label4 = skimage.transform.resize(label, (label.shape[0] // 8*scale, label.shape[1] // 8*scale),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label5 = skimage.transform.resize(label, (label.shape[0] // 16*scale, label.shape[1] // 16*scale),
        #                                   order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float32)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).long(),
                # 'label2': torch.from_numpy(label2).float(),
                # 'label3': torch.from_numpy(label3).float(),
                # 'label4': torch.from_numpy(label4).float(),
                # 'label5': torch.from_numpy(label5).float()
                }
class gaussian_blur(object):

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        gauss_size = random.choice([1, 3, 5, 7])
        if gauss_size > 1:
            # do the gaussian blur
            image = cv2.GaussianBlur(image, (gauss_size, gauss_size), 0)

        return {'image': image,
                'depth': depth,
                'label': label,
                }
class to_edge(object):
    def __init__(self,is_city=False):
        self.is_city=is_city
    def __call__(self, sample):
        mask = sample['label']
        label=mask.byte().numpy()
        _edgemap = mask.byte().numpy()
        if self.is_city:
            num_classes = 19
        else:
            num_classes = 40
        _edgemap = self.mask_to_onehot(_edgemap, num_classes)
        dege_pred=self.onehot_to_binary_edges(_edgemap, 2, num_classes,label)
        # sample['edge'] = torch.from_numpy(edge).float()

        sample['dege_pred'] = torch.from_numpy(dege_pred).long()
        return sample
    def mask_to_onehot(self,mask, num_classes):
        _mask = [mask == i for i in range(num_classes)]
        return np.array(_mask).astype(np.uint8)

    def onehot_to_binary_edges(self,mask, radius, num_classes,label):

        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

        edgemap = np.zeros(mask.shape[1:])

        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist < radius] = 1
            dist[dist >= radius] = 0
            dist[dist == 1] = i+1
            edgemap=np.maximum(edgemap, dist)
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap).astype(np.uint8)
        mask_zero_indices=label==255
        # edgemap_copy=edgemap.copy()
        edgemap[0][mask_zero_indices]=255
        # edgemap_copy[0][mask_zero_indices]=0
        return edgemap