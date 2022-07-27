# -*- coding:utf-8 -*-
"""
Author   : Vandaci 
Date     : 2022/7/22 11:03
E-mail   : cnfendaki@qq.com
Project  : implement_yolo_v1/metrics.py
Software : PyCharm
"""
from torch import Tensor


def accuracy(pred: Tensor, target: Tensor) -> Tensor:
    pred_ = pred.clone()
    pred_[pred_ >= 0.5] = 1
    pred_[pred_ < 0.5] = 0
    true_num = len(target[pred_ == target])
    acc = true_num / len(target.view(-1, 1))
    del pred_
    return acc


if __name__ == "__main__":
    import torch

    # test for average accuracy for multiclass
    pred = torch.rand(20, 20)
    target = torch.ones_like(pred)
    acc = accuracy(pred, target)
    print(acc)
    pass
