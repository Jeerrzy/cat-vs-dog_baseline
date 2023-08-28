#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/27
# @file: config.py
# @author: jeerrzy


"""
说明: 统一调整和保存重要参数
注意: 完成调整后记得运行保存为文件, 否则不会起作用
"""


import json


if __name__ == "__main__":
    cfg = {
        # 是否使用GPU和CUDA运算, 如果有则设置为True, 否则为False
        'cuda': True,
        # 选择使用的模型: [lenet-5, ]
        'model': 'lenet-5',
        # 输入图片张量的尺度
        'input_shape': [32, 32],
        # 输出通道数
        'output_channels': 2,
        # 数据集路径
        'datasets_root': 'D:/datasets',
        # 项目数据库路径
        'database_root': './database',
        # 参数保存路径
        'logs_root': './logs',
        # 分类类别
        'class_names': ['cat', 'dog'],
        # batch数量
        'batch_size': 16,
        # 损失函数: [CE, ]
        'loss_function_type': 'CE',
        # 优化器: [SGD, Adam]
        'optimizer_type': 'SGD',
        # 学习率
        'lr': 1e-3,
        # 训练轮次
        'epoch': 500,
        # 每隔多少轮次保存一次模型参数
        'save_parameters_period': 10,
        # 参数路径
        'parameters_path': './logs/2023-08-28-03_42_49/parameters/model_parameters_200.pkl'
    }
    with open('./database/config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Save config file down.')
