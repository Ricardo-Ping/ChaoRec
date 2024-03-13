"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/13 21:40
@File : LayerGCN.py
@function :
"""
import random

import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F


class LayerGCN(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, dropout,
                 device):
        super(LayerGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.n_nodes = num_user + num_item

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        # 定义用户和物品的嵌入层，使用Xavier均匀分布初始化嵌入参数
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_user, self.dim_E)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_item, self.dim_E)))

        # 生成并归一化邻接矩阵
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # 初始化边剪枝后的邻接矩阵和前向传播使用的邻接矩阵为空
        self.masked_adj = None
        self.forward_adj = None
        # 边剪枝方法是否随机选择的标志
        self.pruning_random = False

        # 获取图的边信息（边的索引和值）
        self.edge_indices, self.edge_values = self.get_edge_info()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.num_user + self.num_item,
                           self.num_user + self.num_item), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        # 先将row和col转换为numpy数组
        row_col_array = np.array([row, col])
        # 然后使用转换后的numpy数组创建Tensor
        i = torch.LongTensor(row_col_array)

        data = torch.FloatTensor(L.data)

        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.num_user, self.num_item)))
        return edges, values

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj_matrix
            return
        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        if self.pruning_random:
            # pruning randomly
            keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len))
        else:
            # pruning edges by pro
            keep_idx = torch.multinomial(self.edge_values, keep_len)  # prune high-degree nodes
        self.pruning_random = True ^ self.pruning_random
        keep_indices = self.edge_indices[:, keep_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.num_user, self.num_item)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.num_user
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj_matrix.shape).to(self.device)

    def get_ego_embeddings(self):
        ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], 0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        embeddings_layers = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.forward_adj, all_embeddings)
            # 计算当前层嵌入与初始嵌入的余弦相似度，作为权重
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            # 使用权重调整当前层的嵌入
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        user_all_embeddings, item_all_embeddings = torch.split(ui_all_embeddings, [self.num_user, self.num_item])
        return user_all_embeddings, item_all_embeddings

    def bpr_loss(self, users, pos_items, neg_items, user_all_embeddings, item_all_embeddings):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = user_all_embeddings[users]
        pos_item_embeddings = item_all_embeddings[pos_items]
        neg_item_embeddings = item_all_embeddings[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items):
        # 计算正则化损失
        user_embeddings = self.user_embeddings[users]
        pos_item_embeddings = self.item_embeddings[pos_items]
        neg_item_embeddings = self.item_embeddings[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        self.forward_adj = self.masked_adj
        user_all_embeddings, item_all_embeddings = self.forward()

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, user_all_embeddings, item_all_embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items)
        total_loss = bpr_loss + reg_loss

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入

        self.forward_adj = self.norm_adj_matrix
        restore_user_e, restore_item_e = self.forward()

        user_tensor = restore_user_e.cpu()
        item_tensor = restore_item_e.cpu()

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


