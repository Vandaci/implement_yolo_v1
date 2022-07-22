# -*- coding:utf-8 -*-
"""
Author   : Vandaci 
Date     : 2022/7/22 11:03
E-mail   : cnfendaki@qq.com
Project  : implement_yolo_v1/metrics.py
Software : PyCharm
"""
from torch import Tensor
import copy


def accuracy(pred: Tensor, target: Tensor) -> Tensor:
    pred_ = copy.deepcopy(pred)
    pred_[pred_ >= 0.5] = 1
    pred_[pred_ < 0.5] = 0
    true_num = target[pred_ == target]
    true_num = len(true_num)
    acc = true_num / len(target.view(-1, 1))
    return acc


if __name__ == "__main__":
    import torch

    # test for average accuracy for multiclass
    pred = torch.rand(20, 20)
    target = torch.ones_like(pred)
    acc = accuracy(pred, target)
    print(acc)
    pass
