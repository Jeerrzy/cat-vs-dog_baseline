#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/26
# @file: dataset.py
# @author: jeerrzy


"""
说明: 按照pytorch规则组织数据集
"""


import cv2
import torch
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import warnings


warnings.filterwarnings("ignore")


class CatVSDogDataset(data.Dataset):
    def __init__(self, txt_path, input_shape, cls_num, mode):
        """
        :param txt_path: txt格式的文字简化版数据集路径(由generate_database.py生成)
        :param input_shape: 规定输入神经网络的长宽大小, pytorch的tensor数据格式为[B, C, H, W] (batch, channel, height, width)
                            注意, 数据的输入尺度应该和模型结构相符合
        :param cls_num: 分类类别数目
        :param mode: 数据集格式, 主要是为了区别在train模式下进行数据增强操作, 而验证或测试时不进行
        """
        self.mode = mode
        self.input_shape = input_shape
        with open(txt_path, 'r') as f:
            self.annotation_lines = f.readlines()
        self.cls_num = cls_num
        # 提示: 如果希望补充更多transform, 请在下方列表中添加
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.input_shape)
            ]
        )

    def __len__(self):
        return len(self.annotation_lines)
    
    def __getitem__(self, index):
        src_image_path = self.annotation_lines[index].split(';')[1].split()[0]
        cls_label = int(self.annotation_lines[index].split(';')[0])
        src_image_data = cv2.imread(src_image_path)
        # 提示: 如果希望进行自定义数据增强操作, 请补充完data_augment函数后在此打开注释
        # if self.mode == 'train':
        #     src_image_data = data_augment(src_image_data)
        trans_image_data = self.transform(src_image_data)
        trans_cls_label = torch.tensor(cls_label).to(torch.int64)
        if self.mode == 'train':
            cls_one_hot = F.one_hot(trans_cls_label, num_classes=self.cls_num).float()
            return trans_image_data, cls_one_hot
        else:
            return trans_image_data, cls_label
    

def data_augment(src_image_data):
    """

    提示: 如果希望进行自定义数据增强操作, 请在此补充代码
    注意: 仅在训练数据集中进行增强, 以扩充数据, 提升模型泛化性。而测试集为了保证衡量的客观性, 不需要增强。用参数mode控制。
    
    """
    return src_image_data
    

if __name__ == "__main__":
    dataset = CatVSDogDataset(txt_path='./test.txt', input_shape=[224, 224], mode='test')
    data, label = dataset[1]
    print(label)
