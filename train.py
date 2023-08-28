#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/27
# @file: train.py
# @author: jeerrzy


"""
说明: 训练模型
"""


import os
import json
from torch.utils.data import DataLoader
from database.dataset import CatVSDogDataset
from models import get_model
from models.trainer import CatVSDogTrainer


if __name__ == "__main__":
    # 获取关键参数
    cfg_file_path = './database/config.json'
    with open(cfg_file_path, 'r') as f:
        cfg = json.load(f)
    # 准备数据
    train_file_path = os.path.join(cfg['database_root'], 'train.txt')
    test_file_path = os.path.join(cfg['database_root'], 'test.txt')
    train_dataset = CatVSDogDataset(
        txt_path=train_file_path,
        input_shape=cfg['input_shape'],
        cls_num=len(cfg['class_names']),
        mode='train'
    )
    test_dataset = CatVSDogDataset(
        txt_path=test_file_path,
        input_shape=cfg['input_shape'],
        cls_num=len(cfg['class_names']),
        mode='test'
    )
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=cfg['batch_size'])
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=cfg['batch_size'])
    # 准备计算模型
    model = get_model(cfg['model'])(output_channels=cfg['output_channels'])
    # 将参数输入到训练器
    trainer = CatVSDogTrainer(
        logs_root=cfg['logs_root'],
        cuda=cfg['cuda'],
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        epoch=cfg['epoch'],
        optimizer_type=cfg['optimizer_type'],
        lr=cfg['lr'],
        loss_function_type=cfg['loss_function_type'],
        save_parameters_period=cfg['save_parameters_period']
    )
    trainer.train()

        
