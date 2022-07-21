# -*- coding:utf-8 -*-
"""
Author   : Vandaci 
Date     : 2022/7/21 7:51
E-mail   : cnfendaki@qq.com
Project  : YOLOV1/dataset.py
Software : PyCharm
"""

import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import os
from torch import Tensor


class YOLODataset(Dataset):
    def __init__(self, img_dir, labeltxt, transform=None):
        self.img_dir = img_dir
        self.labeltxt = labeltxt
        self.transform = transform
        with open(labeltxt, 'r') as f:
            self.__labeltxtlines = f.readlines()

    def __len__(self):
        return len(self.__labeltxtlines)

    def __getitem__(self, item):
        labeltxt = self.__labeltxtlines[item].strip('\n').split(' ')
        img_name = labeltxt[0]
        img_height = np.array(labeltxt[1]).astype(float)
        img_width = np.array(labeltxt[2]).astype(float)
        # bndbox转浮点数列表
        bndbox = np.array(labeltxt[3:]).astype(float).reshape(-1, 5)
        label = bndbox[:, 0]
        bndbox = bndbox[:, 1:]
        bndbox[:, (0, 2)] = bndbox[:, (0, 2)] / img_width
        bndbox[:, (1, 3)] = bndbox[:, (1, 3)] / img_height
        label = torch.from_numpy(label)
        bndbox = torch.from_numpy(bndbox)
        target = self.__encoder(bndbox, label)
        img = read_image(os.path.join(self.img_dir, img_name), mode=ImageReadMode.RGB) / 255.
        if self.transform:
            img = self.transform(img)
        return img, target

    def __encoder(self, bndboxes: Tensor, labels: Tensor, gridcell_num: int = 7, class_num: int = 20) -> Tensor:
        '''
        :param boxes: 边界框坐标，输入形状(n,4) ,dim=2，为归一化的值(0-1)
        :param labels: 对应标签，输入形状[1,n],dim=1
        :param gridcell_num: 将图片划分为多少个网格，默认为7
        :return: target（gridcell×gridcell×class_num）
        '''
        target = torch.zeros(gridcell_num, gridcell_num, 10 + class_num)
        cell_size = 1. / gridcell_num
        bndbox_wh = bndboxes[:, 2:] - bndboxes[:, :2]
        center_xy = (bndboxes[:, 2:] + bndboxes[:, :2]) / 2
        row_col = torch.div(center_xy, cell_size, rounding_mode='floor')
        grid_cell_leftxy = row_col * cell_size
        relative_center_xy = (center_xy - grid_cell_leftxy) / cell_size
        for i in range(bndboxes.size(dim=0)):
            target[int(row_col[i, 1]), int(row_col[i, 0]), (4, 9, int(labels[i]) + 10)] = 1
            target[int(row_col[i, 1]), int(row_col[i, 0]), 2:4] = bndbox_wh[i]
            target[int(row_col[i, 1]), int(row_col[i, 0]), 7:9] = bndbox_wh[i]
            target[int(row_col[i, 1]), int(row_col[i, 0]), :2] = relative_center_xy[i]
            target[int(row_col[i, 1]), int(row_col[i, 0]), 5:7] = relative_center_xy[i]
        return target


class VocClassifier(Dataset):
    def __init__(self, img_dir, labeltxt, transform=None):
        with open(labeltxt, 'r') as f:
            self.__lines = f.readlines()
        self.img_dir = img_dir
        self.labeltxt = labeltxt
        self.transform = transform

    def __len__(self):
        return len(self.__lines)

    def __getitem__(self, item):
        labelstr = self.__lines[item].strip('\n').split(' ')
        image = read_image(os.path.join(self.img_dir, labelstr[0]), mode=ImageReadMode.RGB) / 255.
        label = np.array(labelstr[1:]).astype(float)
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # transform = torchvision.transforms.Resize((448, 448))
    # img_dir = r"E:\VOC\2012\Train"
    # trainset = YOLODataset(img_dir, '../data/train.txt', transform=transform)
    # img, label = trainset[20]
    # trainset = VocClassifier(r"E:\VOC\2012\Train", '../data/classifier.txt')
    # image, label = trainset[1]
    pass
