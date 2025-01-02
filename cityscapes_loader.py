import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data


# from data.augmentations import *
def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {"cityscapes": [73.15835921, 82.90891754, 72.39239876],}

    def __init__(self, transform=None, phase_train=True, data_dir=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = data_dir
        if phase_train:
            self.split = 'train'
        else:
            self.split = 'val'
        split=self.split
        self.transform = transform
        # self.img_norm = img_norm
        self.n_classes = 19
        # self.img_size = (
        #     img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # )
        # self.mean = img_mean
        self.files = {}

        self.images_base = os.path.join(self.root,"leftImg8bit", self.split)
        self.images_depth = os.path.join(self.root, "disparity", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.files["depth_"+split] = recursive_glob(rootdir=self.images_depth, suffix=".png")

        # filename1 = "Cityscapes/data/"+split+".txt"
        # with open(filename1, 'w') as file:
        #     for line in self.files[split]:
        #         file.write(line + '\n')
        # filename1 = "Cityscapes/data/depth_"+split+".txt"
        # with open(filename1, 'w') as file:
        #     for line in self.files["depth_"+split]:
        #         file.write(line + '\n')
        # filename1 = "Cityscapes/data/label_"+split+".txt"
        # with open(filename1, 'w') as file:
        #     for line in self.files[split]:
        #         txt=os.path.join(
        #     self.annotations_base,
        #     line.split(os.sep)[-2], # temporary for cross validation
        #     os.path.basename(line)[:-15] + "gtFine_labelIds.png",
        # )
        #         file.write(txt + '\n')
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 0
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

        # self.return_id = return_id

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        depth_path = self.files["depth_"+self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2], # temporary for cross validation
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        img_dir = img_path
        depth_dir = depth_path
        label_dir = lbl_path
        depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED).astype(np.float32)
        div = np.max(depth) - np.min(depth)
        min = np.min(depth)
        scaling_factor = 255.0 / div
        depth = depth - min
        depth = depth * scaling_factor
        image = cv2.imread(img_dir).astype(np.float32)
        label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        label = self.encode_segmap(label)
        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def transform(self, img, lbl,depth):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        depth=cv2.resize(depth, (self.img_size[1], self.img_size[0]))
        #     m.imresize(
        #     depth, (self.img_size[0], self.img_size[1])
        # )
        depth = depth.astype(np.float64)
        # img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
            depth = depth.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        depth=np.expand_dims(depth, axis=-1)
        depth = depth.transpose(2, 0, 1)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        '''
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        '''

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        img = np.concatenate((img, depth), axis=0)
        img = torch.from_numpy(img).float()
        depth= torch.from_numpy(depth).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl,depth

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]+1
        return mask


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = None#Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip()])

    local_path = "Cityscapes/data/"
    dst = cityscapesLoader(local_path, split="train",is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    for i, data in enumerate(trainloader):
        images, labels = data
        images=images[:,0:3,:,:]
        images = images.numpy()[:, ::-1, :, :]
        images = np.transpose(images, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(images[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()

