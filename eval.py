#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/7/22
# @file: eval.py
# @author: jerrzzy



"""
说明: 在训练结束后可以选择再次
"""


import os
import json
from torch.utils.data import DataLoader
from models import get_model
from database.dataset import CatVSDogDataset
from models.predictor import Predictor


if __name__ == "__main__":
    # 获取关键参数
    cfg_file_path = './database/config.json'
    with open(cfg_file_path, 'r') as f:
        cfg = json.load(f)
    # 模型
    model = get_model(cfg['model'])(output_channels=cfg['output_channels'])
    predictor = Predictor(
        cuda=cfg['cuda'],
        model=model,
        parameters_path=cfg['parameters_path'],
        input_shape=cfg['input_shape']
    )
    # 测试数据集
    test_file_path = os.path.join(cfg['database_root'], 'test.txt')
    test_dataset = CatVSDogDataset(
        txt_path=test_file_path,
        input_shape=cfg['input_shape'],
        mode='test'
    )
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=cfg['batch_size'])
    # 计算指标
    metrics = predictor.eval(test_dataloader=test_dataloader)
    print('eval down.')
    print(metrics)

