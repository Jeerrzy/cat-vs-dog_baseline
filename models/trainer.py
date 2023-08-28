#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @date: 2023/8/27
# @file: trainer.py
# @author: jeerrzy


"""
说明: 封装完整训练过程和保存关键参数的类
"""


import os
import datetime
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.metrics import calculate_metrics


class CatVSDogTrainer(object):
    """训练器"""
    def __init__(self, logs_root, cuda, model, train_dataloader, val_dataloader, epoch, optimizer_type, lr, loss_function_type, save_parameters_period):
        """
        :param logs_root: 参数根路径
        :param cuda: 是否使用GPU
        :param model: 模型对象
        :param train_dataloader: 训练数据集对象
        :param val_dataloader: 验证数据集对象
        :param epoch: 训练总轮次
        :param optimizer_type: 优化器类型
        :param lr: 学习率
        :param loss_function_type: 损失函数轮次
        :param save_parameters_period: 保存参数的周期
        """
        now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        logs_save_path = os.path.join(logs_root, now_time)  # 当前训练信息缓存路径
        self.parameters_save_path = os.path.join(logs_save_path, 'parameters')  # 模型参数保存路径
        if not os.path.exists(logs_root):
            os.makedirs(logs_root)
        if not os.path.exists(logs_save_path):
            os.makedirs(logs_save_path)
        if not os.path.exists(self.parameters_save_path):
            os.makedirs(self.parameters_save_path)
        self.logger = TrainLogger(log_root=logs_save_path)  # 日志记录
        self.device = torch.device("cuda" if cuda else "cpu")  # 是否启用GPU
        self.model = model  # 模型
        self.model = self.model.to(self.device)
        self.train_dataloader = train_dataloader  # 训练数据集
        self.val_dataloader = val_dataloader  # 验证数据集
        self.epoch = epoch  # 训练总轮次
        self.optimizer_type = optimizer_type  # 优化器种类
        self.learning_rate = lr  # 学习率
        self.loss_function_type = loss_function_type
        self.save_parameters_period = save_parameters_period  # 保存参数的周期
        if self.optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))  # 优化器
        elif self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False)
        if self.loss_function_type == 'CE':
            self.loss_function = nn.CrossEntropyLoss()  # 损失函数
        self.loss_function = self.loss_function.to(self.device)

    def train(self):
        self.logger.add_info(f'Start Train and Validate. Total Epoch: {self.epoch}')
        # epoch循环
        for batch_iter in range(1, self.epoch+1):
            self.logger.add_info(f'————Epoch:{batch_iter} Start')
            # 执行训练
            loss_item = self.train_one_epoch()
            self.logger.add_loss(loss_item, type='train')
            self.logger.add_info(f'Epoch{batch_iter} Train down. Train Loss:{loss_item}')
            # 执行测试
            loss_item, metrics = self.validate_one_epoch()
            # 保存最优参数
            f1_score = metrics['f1_score']
            if f1_score > self.logger.get_best_f1_score():
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.parameters_save_path, f'best_model_parameters_{str(batch_iter).zfill(3)}_f1_{f1_score}.pkl')
                )
            self.logger.add_loss(loss_item, type='validate')
            for key in ['accuracy', 'precision', 'recall', 'f1_score']:
                self.logger.add_metrics(metric=metrics[key], type=key)
            self.logger.add_info(f'Epoch{batch_iter} Validate down. Val Loss:{loss_item}')
            # 到达固定轮次时保存参数
            if batch_iter % self.save_parameters_period == 0:
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(self.parameters_save_path, f'model_parameters_{str(batch_iter).zfill(3)}.pkl')
                )
            self.logger.save_info()
            self.logger.plot()
        self.logger.add_info('Trainning Down.')

    def train_one_epoch(self):
        loss_sum_list = []
        self.model.train()
        for i, (images, classes) in enumerate(self.train_dataloader):
            self.logger.add_info(f'————————training batch: {i}/{len(self.train_dataloader)}')
            images, classes = images.to(self.device), classes.to(self.device)  # 将数据传送到GPU
            outputs = self.model(images)  # 前向传播
            batch_loss = self.loss_function(outputs, classes)  # 计算损失
            self.optimizer.zero_grad()  # 反向传播
            batch_loss.backward()
            self.optimizer.step()
            loss_sum_list.append(batch_loss.item())
        return np.array(loss_sum_list).mean()
    
    def validate_one_epoch(self):
        loss_sum_list = []
        label_list = []
        pred_list = []
        self.model.eval()
        with torch.no_grad():
            for i, (images, classes) in enumerate(self.val_dataloader):
                self.logger.add_info(f'————————validating batch: {i}/{len(self.val_dataloader)}')
                cls_one_hot = F.one_hot(classes, num_classes=2).float()
                images, cls_one_hot = images.to(self.device), cls_one_hot.to(self.device)  # 将数据传送到GPU
                outputs = self.model(images)  # 前向传播
                # 计算损失
                batch_loss = self.loss_function(outputs, cls_one_hot)  # 计算损失
                loss_sum_list.append(batch_loss.item())
                # 转化预测结果为类别
                result_list = nn.functional.softmax(outputs.squeeze(dim=0), dim=-1)
                result_list = result_list.argmax(dim=-1).to(torch.device('cpu')).numpy()
                # 添加到结果, 准备批量计算
                label_list += list(classes)
                pred_list += list(result_list)
        metrics = calculate_metrics(cls_num=2, y=label_list, y_pred=pred_list)
        return np.array(loss_sum_list).mean(), metrics


class TrainLogger(object):
    """训练日志记录"""
    def __init__(self, log_root='./logs'):
        """
        :param log_root: 参数根路径
        """
        self.log_root = log_root
        self.logger = self.make_logger()
        self.result_dict = {}
        self.loss_history = {
            'train': [],
            'validate': []
        }
        self.metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

    def get_best_f1_score(self):
        f1_score_list = self.metrics_history['f1_score']
        if len(f1_score_list) > 0:
            return np.array(f1_score_list).max()
        else:
            return 0

    def make_logger(self):
        # 创建日志记录
        logger = logging.getLogger("logger")
        logger.setLevel(logging.INFO)
        # 创建处理器：sh为控制台处理器，fh为文件处理器
        sh = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(self.log_root, 'train.log'), encoding="UTF-8")
        formator = logging.Formatter(fmt="%(asctime)s %(filename)s %(levelname)s %(message)s", datefmt="%Y/%m/%d %X")
        sh.setFormatter(formator)
        fh.setFormatter(formator)
        logger.addHandler(sh)
        logger.addHandler(fh)
        return logger
    
    def add_info(self, _str):
        self.logger.info(_str)

    def add_loss(self, loss, type='train'):
        self.loss_history[type].append(loss)

    def add_metrics(self, metric, type='accuracy'):
        self.metrics_history[type].append(metric)

    def plot(self):
        fig = plt.figure()
        # 绘制损失变化情况
        ax1 = fig.add_subplot(121)
        period_list = np.arange(1, len(self.loss_history['train'])+1)
        train_loss_list = np.array(self.loss_history['train'])
        val_loss_list = np.array(self.loss_history['validate'])
        ax1.plot(period_list, train_loss_list, color='r', label='train_loss')
        ax1.plot(period_list, val_loss_list, color='b', label='val_loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.set_title('Train/Val Loss - Epoch', fontproperties='SimHei', fontsize=10)
        # 绘制测试指标变化情况
        ax2 = fig.add_subplot(122)
        for key in self.metrics_history.keys():
            metric_list = np.array(self.metrics_history[key])
            ax2.plot(period_list, metric_list, label=key)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metrics')
        ax2.legend(loc='upper left')
        ax2.set_title('Metrics - Epoch', fontproperties='SimHei', fontsize=10)
        # 保存
        plt.savefig(os.path.join(self.log_root, 'history.png'))

    def save_info(self):
        self.result_dict['loss_history'] = self.loss_history
        self.result_dict['metric_history'] = self.metrics_history
        with open(os.path.join(self.log_root, 'result.json'), 'w') as f:
            json.dump(self.result_dict, f, indent=2)

