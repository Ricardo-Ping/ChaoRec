"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/5 19:55
@File : metrics.py
@function :
"""
import numpy as np
import torch


def precision_at_k(ranked_list, test_list, k):
    # 计算 precision@k
    # 计算这k个推荐项目与测试集中的实际喜欢项目的交集的大小
    return len(set(ranked_list[:k]) & set(test_list)) / k


def recall_at_k(ranked_list, test_list, k):
    if len(test_list) == 0:
        return 0
    # 推荐列表和测试集的交互列表的交集的大小/测试集中交互列表的大小
    return len(set(ranked_list[:k]) & set(test_list)) / len(test_list)


def ndcg_at_k(ranked_list, test_list, k):
    # 计算 NDCG@k
    if not test_list:
        return 0
    dcg = 0
    idcg = sum([1.0 / np.log(i + 2) for i in range(min(len(test_list), k))])
    for i, item in enumerate(ranked_list[:k]):
        if item in test_list:
            dcg += 1.0 / np.log(i + 2)
    return dcg / idcg


def hit_rate_at_k(ranked_list, test_list, k):
    # 计算 hit rate@k
    return int(bool(set(ranked_list[:k]) & set(test_list)))


def map_at_k(ranked_list, test_list, k):
    # 计算 MAP@k
    if not test_list:
        return 0
    scores = 0
    num_hits = 0
    for i, item in enumerate(ranked_list[:k]):
        if item in test_list:
            num_hits += 1
            scores += num_hits / (i + 1)
    return scores / len(test_list)
