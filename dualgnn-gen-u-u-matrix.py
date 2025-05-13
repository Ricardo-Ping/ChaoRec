"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/25 21:44
@File : dualgnn-gen-u-u-matrix.py
@function :
"""
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from arg_parser import parse_args


# 生成用户-用户矩阵，权重为用户共同交互的项目数量
def gen_user_matrix(all_edge, no_users):
    edge_dict = defaultdict(set)

    # 遍历所有的用户-项目对，将每个项目添加到与其对应的用户的集合中
    for edge in all_edge:
        user, item = edge
        edge_dict[user].add(item)

    # 初始化用户-用户矩阵
    num_user = no_users
    user_graph_matrix = torch.zeros(num_user, num_user)

    # 获取所有与项目交互的用户ID，并将它们排序
    key_list = list(edge_dict.keys())
    key_list.sort()

    bar = tqdm(total=len(key_list))
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head + 1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            inter_len = len(item_head.intersection(item_rear))
            if inter_len > 0:
                user_graph_matrix[head_key][rear_key] = inter_len
                user_graph_matrix[rear_key][head_key] = inter_len
    bar.close()

    return user_graph_matrix


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_path
    print(f'Generating u-u matrix for {dataset} ...\n')
    dir_str = './Data/' + dataset
    train_data = np.load(dir_str + '/train.npy', allow_pickle=True)

    num_user = None
    if dataset == 'netfilx':
        num_user = 14971  # max 9599 min 7
    elif dataset == 'clothing':
        num_user = 18072  # max 979, min 4
    elif dataset == 'baby':
        num_user = 12351
    elif dataset == 'sports':
        num_user = 28940  # max 6356, min 4
    elif dataset == 'beauty':
        num_user = 15482  # max 2279, min 2
    elif dataset == 'microlens':
        num_user = 46420  # max 2038, min 6
    user_graph_matrix = gen_user_matrix(train_data, num_user)

    user_graph = user_graph_matrix
    # 初始化一个零向量来存储每个用户与其他用户的交互数
    user_num = torch.zeros(num_user)

    user_graph_dict = {}
    item_graph_dict = {}
    edge_list_i = []
    edge_list_j = []

    # 对于每个用户，计算他们与其他用户的非零交互数，并打印
    for i in range(num_user):
        user_num[i] = len(torch.nonzero(user_graph[i]))
        print("this is ", i, "num", user_num[i])

    # # 对于每个用户，我们要找出他们与其他用户的最大交互数
    for i in range(num_user):
        # 只关心与其他用户的前200个最大交互数
        if user_num[i] <= 200:
            user_i = torch.topk(user_graph[i], int(user_num[i]))
            edge_list_i = user_i.indices.numpy().tolist()
            edge_list_j = user_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            user_graph_dict[i] = edge_list
        else:
            user_i = torch.topk(user_graph[i], 200)
            edge_list_i = user_i.indices.numpy().tolist()
            edge_list_j = user_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            user_graph_dict[i] = edge_list

    # 保存user_graph_dict
    file_path = os.path.join(dir_str, 'user_graph_dict.npy')
    np.save(file_path, user_graph_dict, allow_pickle=True)

    # 初始化用于存储最大和最小相似用户数的变量
    max_similar_users = 0
    min_similar_users = float('inf')  # 无穷大，方便初始比较

    # 统计每个用户的相似用户数
    for i in range(num_user):
        similar_users_count = len(torch.nonzero(user_graph[i]))  # 统计非零值（相似用户）的数量

        # 更新最大相似用户数
        if similar_users_count > max_similar_users:
            max_similar_users = similar_users_count

        # 更新最小相似用户数
        if similar_users_count < min_similar_users:
            min_similar_users = similar_users_count

    # 打印出所有用户中最大和最小的相似用户数
    print(f'Maximum number of similar users: {max_similar_users}')
    print(f'Minimum number of similar users: {min_similar_users}')
