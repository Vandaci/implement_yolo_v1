# Time    : 2022.07.20 上午 09:10
# Author  : Vandaci(cnfendaki@qq.com)
# File    : utils.py
# Project : YOLOV1
import torch
from torch import Tensor
import numpy as np


def encoder(bndboxes: Tensor, labels: Tensor, gridcell_num: int = 7, class_num: int = 20) -> Tensor:
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


def write2numpy(path='train.txt'):
    '''
    后续这里直接写到内存中
    :param path:
    :return:
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip('\n').split(' ')
            pass


if __name__ == '__main__':
    # box = torch.tensor([[120, 120, 240, 360],
    #                     [100, 100, 200, 250]], dtype=torch.float) / 448.0
    # labels = torch.tensor([1, 0], dtype=torch.float)
    # tgt = encoder(box, labels)
    write2numpy()
    pass
