#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/26
# @file: visualization_demo.py
# @author: jeerrzy


"""
说明: 可视化展示数据集
"""


import cv2


txt_path = './train.txt'   # txt格式的文字简化版数据集路径(由generate_database.py生成)
visualization_index = 10   # 查看数据的索引(可以随意调整, 但应当小于总数防止溢出)


if __name__ == "__main__":
    with open(txt_path, 'r') as f:
        annotation_lines = f.readlines()
    line = annotation_lines[visualization_index]
    src_image_path = line.split(';')[1].split()[0]
    cls_label = int(line.split(';')[0])
    src_image_data = cv2.imread(src_image_path)
    print(f'path: {src_image_path}, label: {cls_label}')
    cv2.imshow('demo', src_image_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()