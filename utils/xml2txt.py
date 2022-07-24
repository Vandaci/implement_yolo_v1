# Time    : 2022.07.19 下午 08:48
# Author  : Vandaci(cnfendaki@qq.com)
# File    : xml2txt.py
# Project : YOLOV1
import xml.etree.ElementTree as ET
import os
import numpy as np

# 数据集所有分类字典
CLASS_NAME = {'person': 0,
              'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6,
              'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13,
              'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19
              }
CLASS_NAME_LIST = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus',
                   'car', 'motorbike', 'train', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']


def xml2txt(xml_path, txt_path):
    '''
    :param xml_path: xml文件存放路径
    :param txt_path: 写入到txt文档的位置，当前比较慢，后续改进方法或者使用多线程能否解决问题
    :return:
    '''
    with open(txt_path, 'w+') as f:
        # 列出xml路径下所有xml文件
        xml_lists = os.listdir(xml_path)
        # 遍历所有xml文件
        for xmlfile in xml_lists:
            # 获取根对象
            xroot = ET.parse(os.path.join(xml_path, xmlfile))
            # 按节点名查找并提取其名称
            labelstr = xroot.find('filename').text
            width = xroot.find('size').find('width').text
            height = xroot.find('size').find('height').text
            # .iter('object')查找所有子节点'object'创建迭代器
            labelstr = labelstr + " " + height + " " + width
            for obj in xroot.iter('object'):
                # 每个类型名转化为CLASS_NAME对应Value,方便后续编码
                labelstr += " " + str(CLASS_NAME[obj.find('name').text])
                bndbox = obj.find('bndbox')
                # 找出所有坐标，因此txt文档每行 jpg_name class_name xmin ymin xmax ymax class_name ...
                for loc in bndbox:
                    labelstr += " " + loc.text
            # 遍历完一个xml文件换行
            labelstr += "\n"
            f.write(labelstr)


def xml2txtforclassifier(xml_path, txt_path):
    with open(txt_path, 'w+') as f:
        xml_lists = os.listdir(xml_path)
        class_list = np.zeros(20, dtype=int)
        for xmlfile in xml_lists:
            xroot = ET.parse(os.path.join(xml_path, xmlfile))
            labelstr = xroot.find('filename').text
            for obj in xroot.iter('object'):
                class_idx = CLASS_NAME[obj.find('name').text]
                class_list[class_idx] = 1
            class_list_str = str(class_list)
            class_list_str = class_list_str.strip('[]')
            labelstr += " " + class_list_str + '\n'
            f.write(labelstr)
            class_list[:] = 0


if __name__ == "__main__":
    # xml2txt(r"E:\VOC\2012\Train_Label", '../data/train.txt')
    # xml2txtforclassifier(r"E:\VOC\2012\Train_Label", '../data/classifier_train.txt')
    # xml2txtforclassifier(r"E:\VOC\2012\Test_Label", "../data/classifier_test.txt")
    xml2txtforclassifier(r"E:\DataSets\PascalVOC\2007\Test_Label", "../data/classifier_test.txt")
