#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/26
# @file: lenet-5.py
# @author: jeerrzy


"""
说明: 定义lenet-5卷积神经网络用于测试和示例, 做过手写数字识别实验的同学不会陌生
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, output_channels):
        """
        :param output_channels: 输出通道数, 在分类任务使用one_hot编码和softmax激活函数时一般取为分类类别数
        """
        super(LeNet, self).__init__()
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(3, 6, 5)    # 输入通道为3，输出通道为6，卷积核大小5x5
        self.pool1 = nn.MaxPool2d(2, 2)    # 过滤器大小2x2，步长2 
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)   
        # 通过卷积池化后的是四维张量[batch_size,channels,height,width]，要想接入全连接层就必须变为二维张量
        self.fc2 = nn.Linear(120, 84)
        # 注意模型的输出通道数要与任务的信息维度符合
        self.fc3 = nn.Linear(84, self.output_channels)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))		# input_shape=(batch_size,3,32,32) output_shape=(batch_size,6,28,28)
        x = self.pool1(x)				# input_shape=(batch_size,6,28,28) output_shape=(batch_size,6,14,14)
        x = F.relu(self.conv2(x))		# input_shape=(batch_size,6,14,14) output_shape=(batch_size,16,10,10)
        x = self.pool2(x)				# input_shape=(batch_size,16,10,10) output_shape=(batch_size,16,5,5)
        x = x.view(-1,16*5*5)      	    # input_shape=(batch_size,16,5,5) output_shape=(batch_size,400)
        x = F.relu(self.fc1(x))			# input_shape=(batch_size,400) output_shape=(batch_size,120)
        x = F.relu(self.fc2(x))			# input_shape=(batch_size,120) output_shape=(batch_size,84)
        x = self.fc3(x)					# input_shape=(batch_size,84) output_shape=(batch_size,10)
        return x


if __name__ == "__main__":
    # 这里是一个小小的测试用栗，可以用来验证模型没有问题，并且查看输入和输出的维度
    # [batch_size, channel_num, height, width]
    test_input_tensor = torch.Tensor(4, 3, 32, 32)
    model = LeNet(output_channels=2)
    output_tensor = model(test_input_tensor)
    print(output_tensor.size())