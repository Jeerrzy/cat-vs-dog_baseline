#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/29
# @file: predict.py
# @author: jeerrzy


"""
说明: 针对现有图片进行简单测试
"""


import json
import cv2
from models import get_model
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
    # 测试图片
    image_path = './database/dog.jpg'
    # image_path = './database/dog.jpg'
    test_image = cv2.imread(image_path)
    result = predictor(test_image)
    cls = cfg['class_names'][result]
    print(f'image path:{image_path}, cls:{cls}')

