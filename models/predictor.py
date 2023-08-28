#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/28
# @file: predictor.py
# @author: jeerrzy


"""
说明: 封装预测和验证等前向传播过程的类
"""


import torch
import torch.nn as nn
from torchvision import transforms
from models.metrics import calculate_metrics
import warnings


warnings.filterwarnings("ignore")


class Predictor(object):
    """预测器"""
    def __init__(self, cuda, model, parameters_path, input_shape):
        self.device = torch.device("cuda" if cuda else "cpu")  # 是否启用GPU
        self.model = model  # 模型
        self.model = self.model.to(self.device)
        self.input_shape = input_shape  # 输入尺度
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.input_shape)
            ]
        )
        self.model.load_state_dict(torch.load(parameters_path))
        print('Load models down.')

    def __call__(self, image_data):
        trans_image_data = self.transform(image_data)
        trans_image_data = trans_image_data.to(self.device)
        with torch.no_grad():
            result = self.model(trans_image_data).squeeze(dim=0)
            result = nn.functional.softmax(result, dim=-1)
            result = result.argmax(dim=-1).to(torch.device('cpu')).numpy()
        return result

    def eval(self, test_dataloader):
        label_list = []
        pred_list = []
        self.model.eval()
        with torch.no_grad():
            for i, (images, classes) in enumerate(test_dataloader):
                print(f'eval batch: {i}/{len(test_dataloader)}')
                images, classes = images.to(self.device), classes.to(self.device)  # 将数据传送到GPU
                outputs = self.model(images)  # 前向传播
                result_list = nn.functional.softmax(outputs.squeeze(dim=0), dim=-1)
                result_list = result_list.argmax(dim=-1).to(torch.device('cpu')).numpy()
                # 添加到结果, 准备批量计算
                label_list += list(classes)
                pred_list += list(result_list)
        metrics = calculate_metrics(cls_num=2, y=label_list, y_pred=pred_list)
        return metrics

