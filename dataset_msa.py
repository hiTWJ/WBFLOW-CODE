"""
 Dataloader
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

from os.path import join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageOps,ImageEnhance,ImageFile
import random
import os
from scipy.io import loadmat

class BasicDataset(Dataset):
    def __init__(self, name, patch_size=128, patch_num_per_image=4, type='train'):

        self.patch_size = patch_size
        self.patch_num_per_image = 1
        self.type = type

        "train_all_12000_12_"

        files = np.load('/home/lcx/deep_final/deep_final/folds/' + type + '_' + name + '.npy',
                       allow_pickle=True).item()
        self.input = files['name']
        self.gt = files['gt']
        self.label = files['cam']


        print(f'Loading {name} images information...')


    def __len__(self):
        return len(self.input)

    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op):
        if flip_op == 1:
            pil_img = ImageOps.mirror(pil_img)
        elif flip_op == 2:
            pil_img = ImageOps.flip(pil_img)

        img_nd = np.array(pil_img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255

        return img_trans

    @staticmethod
    def randomCT(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度

        return color_image

    def __getitem__(self, i):
        # list = [0, 1, 2, 3, 4]
        # batch_list = random.sample(list, 4)
        # cam_label = np.zeros((12,1,1))
        cam_label = np.zeros((15, 1, 1))
        # cam_label[self.label[i]-1,0,0] = 1
        cam_label[self.label[i] , 0, 0] = 1


        in_img = Image.open(self.input[i])
        awb_img = Image.open(self.gt[i])


        # label = self.label[i]
        # label_ = self.cam_list[label]+1
        # label = np.array([label])
        # get image size
        w, h = in_img.size
        # get ground truth images

        # get flipping option
        if self.type == 'train':
            flip_op = np.random.randint(3)

            flip_op1 = np.random.randint(3)

            "变量：color jittering / color temperature shifting: 仅针对input"
            # in_img = self.randomColor(in_img)


            # get random patch coord
            patch_x = np.random.randint(0, high=w - self.patch_size)
            patch_y = np.random.randint(0, high=h - self.patch_size)
            in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
            if not(flip_op1 == 0):
                in_img_patches = self.randomCT(in_img_patches)
            awb_img_patches = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)


            return {'image': torch.from_numpy(in_img_patches), 'gt-AWB': torch.from_numpy(awb_img_patches),
                    'label': torch.from_numpy(cam_label)}

        elif self.type == 'val':
            name = []
            name.append(os.path.split(self.input[i])[1].split('.')[0])
            # name = np.array(name)
            in_img_re = np.array(in_img.resize((self.patch_size,self.patch_size)))
            in_img_re = in_img_re.transpose((2, 0, 1))/255
            awb_img_re = np.array(awb_img.resize((self.patch_size, self.patch_size)))
            awb_img_re = awb_img_re.transpose((2, 0, 1)) / 255

            return {'image': torch.from_numpy(in_img_re), 'gt-AWB': torch.from_numpy(awb_img_re),
                    'label': torch.from_numpy(cam_label),'name':name}

