import os
import numpy as np
from xml2txt import CLASS_NAME


def txt2txtforclassify(txt_path, target_path):
    dirlist = os.listdir(txt_path)
    with open(os.path.join(txt_path, dirlist[0]), 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.split(" ")[0]
        lines[i] = line
    pic_nums = len(lines)
    lines = np.array(lines)
    label = np.zeros((pic_nums, 20), dtype=int).astype(str)
    label[:, 0] = lines[:]
    for file in dirlist:
        class_name = file.split("_")[0]
        with open(os.path.join(txt_path, file), 'r') as f:
            line = f.readline()
            line = line.strip('\n').split(' ')
            label[label[:, 0] == line[0], CLASS_NAME[class_name]] = line[-1]
            pass
    pass


if __name__ == "__main__":
    txt2txtforclassify(r"E:\DataSets\PascalVOC\2012\VOC2012test\VOCdevkit\VOC2012\ImageSets\Main",
                       '../data/classifier_test.txt')
    pass
