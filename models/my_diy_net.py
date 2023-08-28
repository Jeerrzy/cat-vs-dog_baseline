"""

pytorch定义神经网络模型的一般写法如下。
更多细节请查阅pytorch官方文档或自行搜索学习


import torch.nn as nn


class MyDIYNet(nn.Module):            # 1.定义类名, 继承nn.Module
  def __init__(self):
    super(MyNet, self).__init__()     # 2.写初始化方法, 继承父类, 并且在初始化部分声明各个计算层
    ...
    ...
    ...

  def forward(self, x):               # 3.必须重写forward方法: 调用类时如何进行计算(即前向传播)
    ...
    ...
    ...


pytorch中常用的计算层包括卷积层、线性层、池化层、BN层、激活函数等。其中, 以卷积层为例, 其函数定义如下。
更多细节请查阅pytorch官方文档或自行搜索学习


torch.nn.Conv2d(
  in_channels(int)
  out_channels(int)
  kernel_size(int or tuple)
  stride(int or tuple, optional)
  padding(int or tuple, optional)
  bias(bool, optional)
  ...
)
"""

