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
