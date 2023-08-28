#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/26
# @file: generate_database.py
# @author: jeerrzy


"""
说明: 生成txt格式的文字简化版数据集, 便于后续处理
"""


import os
import json


def generate_database(datasets_root, class_names):
    """
    params:
        - datasets_root: 数据集根路径, 数据集应该按照如下规则组织: 
            1.第一层目录下存放train, test等划分集 2.第二层目录下存放每个类别
            |-datasets
                |-train
                    |-cat
                        |-xxx.jpg
                        ...
                    |-dog
                        |-xxx.jpg
                |-test
                    |-cat
                        |-xxx.jpg
                    |-dog
                        |-xxx.jpg
        - class_names: 所有类别组成的有序列表, 在猫狗二分类中可以为['cat', 'dog']
            此时猫对应第一个通道, 狗对应第二个通道
    """
    for set in os.listdir(datasets_root):
        with open(set + '.txt', 'w') as f:
            file_num = 0
            datasets_path = os.path.join(datasets_root, set)
            for folder_name in os.listdir(datasets_path):
                if folder_name in class_names:
                    cls_id = class_names.index(folder_name)
                    cls_path = os.path.join(datasets_path, folder_name)
                    for image_name in os.listdir(cls_path):
                        if image_name.endswith('.jpg'):
                            f.write(str(cls_id) + ";" + os.path.join(cls_path, image_name) + '\n')
                            file_num += 1
            f.close()
            print(f'Generate {set}.txt down. Totol File Num: {file_num}')


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    generate_database(
        datasets_root=cfg['datasets_root'],
        class_names=cfg['class_names']
    )

