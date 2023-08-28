#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/26
# @folder: database
# @author: jeerrzy


"""
说明:
    - database 用来存放数据相关的文件
        - generate_data.py 生成txt格式的数据集信息文件
        - train.txt/test.txt txt格式的数据集信息文件, 具体地, 每一行为<cls_label; src_image_path \n>
        - visualization_demo.py 可视化展示数据集
        - dataset.py 定义了猫狗数据集的格式
        - cat.jpg/dog.jpg 用于简单测试的图片


提示:
    - 在pytorch中定义数据集方法是继承torch.utils.data.Dataset类, 然后重写__getitem__, __len__两个方法


        from torch.utils.data import Dataset, DataLoader
        import torch

        class MyDataset(Dataset):
            def __init__(self):
                super(MyDataset, self).__init__()          # 继承父类
                self.x = torch.linspace(11,20,10)
                self.y = torch.linspace(1,10,10)

            def __getitem__(self, index):
                return self.x[index], self.y[index]        # __getitem__方法定义了根据索引从数据集取得的数据

            def __len__(self):
                return len(self.x)                         # __len__方法定义了数据集的大小(总数)
"""

