# -*- coding:utf-8 -*-
"""
Author   : Vandaci 
Date     : 2022/7/22 11:16
E-mail   : cnfendaki@qq.com
Project  : implement_yolo_v1/pretrainforclassifier.py
Software : PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn import Module
from torchvision.models import resnet50, ResNet50_Weights
from utils.dataset import VocClassifier
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.xml2txt import CLASS_NAME, CLASS_NAME_LIST
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from utils.metrics import accuracy
import time
import gc

CLASS_NAME_LIST = np.array(CLASS_NAME_LIST)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def resnet(class_num: int) -> Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, class_num)
    return model


def imshow(inp: torch.Tensor, title=None):
    inp = inp.numpy().transpose([1, 2, 0])
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


def train(num_epochs=20):
    train_img_dir = r"E:\VOC\2012\Train"
    test_img_dir = r"E:\VOC\2012\Test"
    train_labeltxt = "../data/classifier_train.txt"
    test_labeltxt = "../data/classifier_test.txt"
    data_transforms = {'train': transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAutocontrast(),
    ]), 'test': transforms.Compose([transforms.Resize((448, 448))])}
    image_dataset = {'train': VocClassifier(train_img_dir, train_labeltxt, transform=data_transforms['train']),
                     'test': VocClassifier(test_img_dir, test_labeltxt, transform=data_transforms['test'])}
    dataloaders = {'train': DataLoader(image_dataset['train'], batch_size=64, shuffle=True, num_workers=6),
                   'test': DataLoader(image_dataset['test'], batch_size=64, shuffle=False,
                                      num_workers=6)}
    # data_sizes = {x: len(image_dataset[x]) for x in ['train', 'test']}
    # 加载模型
    model = resnet(20)
    model.to(DEVICE)
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.
    loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(1, num_epochs + 1):
        since = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.
            running_acc = 0.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    acc = accuracy(outputs, labels)
                    print(f"{phase}: Batch Loss:{loss:.4f} Batch Acc:{acc * 100:.2f}%")
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += float(loss)
                running_acc += acc
            if phase == 'train':
                lr_scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_acc / len(dataloaders[phase])
            print(f"{phase} Loss:{epoch_loss:.4f} Acc:{epoch_acc * 100:.2f}%")
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print(f"在{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s训练完成")
    print(f"最佳测试精度:{best_acc * 100:.2f}%")
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), '../data/best_classifier_weight.pt')
    pass


pass

if __name__ == "__main__":
    # test_img_dir = r"E:\VOC\2007\Test"
    # test_labeltxt = "../data/classifier_test.txt"
    # testset = VocClassifier(test_img_dir, test_labeltxt, transform=transforms.Resize((448, 448)))
    # dataloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)
    # img, label = next(iter(dataloader))
    # out = torchvision.utils.make_grid(img)
    # labels = []
    # for i in range(len(label)):
    #     labels.append(CLASS_NAME_LIST[label[i] == 1].tolist())
    # imshow(out, labels)
    train()
    pass
