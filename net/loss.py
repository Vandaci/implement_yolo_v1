# -*- coding:utf-8 -*-
"""
Author   : Vandaci 
Date     : 2022/7/28 7:40
E-mail   : cnfendaki@qq.com
Project  : implement_yolo_v1/loss.py
Software : PyCharm
"""
from torch import nn


class YoloV1Loss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, y_pred, y_true):
        pass


if __name__ == "__main__":
    pass
