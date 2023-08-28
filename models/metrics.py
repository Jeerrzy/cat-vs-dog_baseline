#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/27
# @file: metrics.py
# @author: jeerrzy


"""
说明: 分类问题的指标计算公式: 先计算混淆矩阵, 然后通过混淆矩阵计算各个指标

    - Confuse Matrix: np.array(cls_num, cls_num)

            | cat | dog |
      | cat |     |     |
      | dog |     |     |
"""


import numpy as np


def calculate_metrics(cls_num, y, y_pred):
    """
    :param cls_num: 分类任务的类别数
    :param y: 列表形式的label(ground_truth), 每个元素为预测类别的号码  [cls_id0, cls_id1, ...]
    :param y_pred: 列表形式的预测结果, 每个元素为预测类别的号码  [pred_id0, pred_id1, ...]
    :return: 字典形式的各个衡量指标
    """
    confuse_matrix = np.zeros((cls_num, cls_num))
    for i, j in zip(y, y_pred):
        confuse_matrix[int(i)][int(j)] += 1
    accuracy = np.diagonal(confuse_matrix).sum()/len(y)
    precision_list = []
    recall_list = []
    for i in range(cls_num):
        if confuse_matrix[:, i].sum() == 0:
            precision_list.append(0)
        else:
            precision_list.append(confuse_matrix[i][i]/confuse_matrix[:, i].sum())
        if confuse_matrix[i, :].sum() == 0:
            recall_list.append(0)
        else:
            recall_list.append(confuse_matrix[i][i]/confuse_matrix[i,:].sum())
    precision = np.array(precision_list).mean()
    recall = np.array(recall_list).mean()
    f1_score = 2*precision*recall/(precision + recall)
    return {
        'confuse_matrix': confuse_matrix,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4)
        }


if __name__ == '__main__':
    # 这里是一个小小的测试用栗，可以帮助理解分类任务指标的计算原理
    cls_num = 2
    y      = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1]
    metrics = calculate_metrics(cls_num, y, y_pred)
    print(metrics)