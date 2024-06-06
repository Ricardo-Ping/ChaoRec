"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/13 17:04
@File : LightGCN.py
@function :
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
from torch_geometric.utils import degree, dropout_adj, add_self_loops

from metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, map_at_k


class LightGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(LightGCNConv, self).__init__(aggr='add', **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        edge_index = edge_index.long()

        # 添加自循环
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        # Compute normalization coefficient
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class LightGCN(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, aggr_mode,
                 device):
        super(LightGCN, self).__init__()
        # 传入的参数
        self.result = None
        self.device = device
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight
        self.dim_embedding = dim_E
        # 转置并设置为无向图
        self.edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)  # [2, 188381]
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)  # [2, 376762]

        # 自定义嵌入和参数
        self.user_embedding = nn.Embedding(num_user, dim_E)
        self.item_embedding = nn.Embedding(num_item, dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 定义图卷积层
        self.conv_layers = nn.ModuleList([LightGCNConv(self.dim_embedding, self.dim_embedding, aggr=self.aggr_mode)
                                          for _ in range(n_layers)])

    def forward(self):
        embs = []
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        embs.append(x)  # 第 0 层

        for conv in self.conv_layers:
            x = conv(x, self.edge_index)
            embs.append(x)

        # 计算权重
        num_layers = len(embs)
        weights = [1.0 / num_layers] * num_layers

        # 对每层的嵌入应用权重并相加
        final_embeddings = torch.zeros_like(embs[0])
        for i in range(num_layers):
            final_embeddings += weights[i] * embs[i]
        self.result = final_embeddings

        return self.result

    def bpr_loss(self, users, pos_items, neg_items, embeddings):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = embeddings[users]
        pos_item_embeddings = embeddings[self.num_user + pos_items]
        neg_item_embeddings = embeddings[self.num_user + neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items, embeddings):
        # 计算正则化损失
        user_embeddings = embeddings[users]
        pos_item_embeddings = embeddings[self.num_user + pos_items]
        neg_item_embeddings = embeddings[self.num_user + neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeddings = self.forward()

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, embeddings)
        total_loss = bpr_loss + reg_loss

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.result[:self.num_user].cpu()
        item_tensor = self.result[self.num_user:self.num_user + self.num_item].cpu()

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 生成评分矩阵
        score_matrix = torch.matmul(user_tensor, item_tensor.t())

        # 将历史交互设置为极小值
        for row, col in self.user_item_dict.items():
            col = torch.LongTensor(list(col)) - self.num_user
            score_matrix[row][col] = 1e-6

        # 选出每个用户的 top-k 个物品
        _, index_of_rank_list_train = torch.topk(score_matrix, topk)
        # 总的top-k列表
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
            dim=0)

        # 返回三个推荐列表
        return all_index_of_rank_list
