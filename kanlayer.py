"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/6/11 9:20
@File : kanlayer.py
@function :
"""
import torch
import torch.nn as nn
import numpy as np


class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim

        # 初始化傅里叶系数，参数化以便训练
        # 2: 表示两组系数，一组用于余弦（cos）项，另一组用于正弦（sin）项。
        # outdim: 输出维度，这表示每个输出特征都将有一组对应的傅里叶系数。
        # inputdim: 输入维度，对于每个输入特征，都有相应的傅里叶系数。
        # gridsize: 网格大小，对应于傅里叶级数中的项数，这决定了傅里叶级数的复杂度和能够表示的频率分辨率。
        # 除以 np.sqrt(inputdim) * np.sqrt(gridsize) 是为了调整初始化的规模, 防止梯度过大或过小。

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], dim=0), self.fouriercoeffs)

        y = y.view(outshape)
        return y


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        # 初始化输入维度、输出维度和多项式最高次数
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # 初始化切比雪夫多项式系数，使用正态分布初始化
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        # nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        nn.init.xavier_uniform_(self.cheby_coeffs)

        # 创建一个从0到degree的数组，用于计算多项式的次数乘积
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # 使用tanh函数将输入数据归一化到[-1, 1]
        x = torch.tanh(x)
        # 将归一化后的数据扩展到多项式的次数，为后续的计算做准备
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )
        # 计算反余弦值，用于切比雪夫多项式的角度变换
        x = x.acos()
        # 将角度乘以数组中的次数，用于生成切比雪夫多项式的值
        x *= self.arange
        # 计算余弦值，得到多项式的具体值
        x = x.cos()
        # 通过切比雪夫系数对多项式的值进行加权和，得到最终的输出
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )
        # 将输出数据重新调整为正确的维度
        y = y.view(-1, self.outdim)
        return y
