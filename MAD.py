"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/4/22 21:26
@File : MAD.py
@function : 计算平均绝对误差(MAD)
"""
import numpy as np
from sklearn.metrics import pairwise_distances


def mad_value(in_arr, mask_arr, distance_metric='cosine', digt_num=4):
    dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)
    mask_dist = np.multiply(dist_arr, mask_arr)
    divide_arr = (mask_dist != 0).sum(1) + 1e-8
    node_dist = mask_dist.sum(1) / divide_arr
    mad = np.mean(node_dist)
    mad = np.round(mad, digt_num)
    return mad


    # 生成掩码
    # self.mask_rate = 0.05
    # image_mask = torch.rand_like(temp_features) < self.mask_rate
    # temp_features.data.masked_fill_(image_mask, 0)
    # text_mask = torch.rand_like(self.t_feat) < self.mask_rate
    # self.t_feat.data.masked_fill_(text_mask, 0)
