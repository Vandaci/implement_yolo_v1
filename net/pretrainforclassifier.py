# -*- coding:utf-8 -*-
"""
Author   : Vandaci 
Date     : 2022/7/22 11:16
E-mail   : cnfendaki@qq.com
Project  : implement_yolo_v1/pretrainforclassifier.py
Software : PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import Module
from torchvision.models import resnet50, ResNet50_Weights
from utils.dataset import VocClassifier

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def resnet(class_num: int) -> Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, class_num)
    return model


def train():
    img_dir = r"E:\VOC\2012\Train"
    labeltxt = "../data/classifier_train.txt"
    test_img_dir = r"E:\VOC\2012\Test"
    trainset = VocClassifier(img_dir, labeltxt, )
    net = resnet(20)
    net.to(DEVICE)


if __name__ == "__main__":
    train()
    pass
