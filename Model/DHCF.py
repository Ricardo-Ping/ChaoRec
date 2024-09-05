"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/9/4 19:09
@File : DHCF.py
@function :
"""

import numpy as np
import torch
from torch import nn


class DJconv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DJconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

    def create_diag(self, values, size, device):
        diag = torch.pow(values + 1e-7, -0.5).to(device)  # 确保 diag 在正确的设备上
        indices = torch.arange(diag.size(0), device=device).unsqueeze(0).repeat(2, 1)
        return torch.sparse_coo_tensor(indices, diag, (size, size), device=device) if torch.is_tensor(
            diag) else torch.diag(diag)

    def compute_matrix(self, H, U, device, use_sparse):
        H_t = H.transpose(0, 1)

        if use_sparse:
            HTH = torch.sparse.mm(H_t, H).to(device)
            Hu = torch.cat((H, torch.sparse.mm(H, HTH)), dim=1).to(torch.float32).to(device)
            row_sum = torch.sparse.sum(Hu, dim=1).to_dense().to(device)
            col_sum = torch.sparse.sum(Hu, dim=0).to_dense().to(device)
        else:
            HTH = torch.mm(H_t, H).to(device)
            Hu = torch.cat((H, torch.mm(H, HTH)), dim=1).to(torch.float32).to(device)
            row_sum = Hu.sum(dim=1).to(device)
            col_sum = Hu.sum(dim=0).to(device)

        # 构造对角矩阵
        Du_v = self.create_diag(row_sum, Hu.size(0), device)
        Du_e = self.create_diag(col_sum, Hu.size(1), device)

        M_u = torch.linalg.multi_dot([Du_v, Hu, Du_e, Du_e, Hu.transpose(0, 1), Du_v, U]) + U

        return M_u

    def forward(self, H, U, I):
        device = self.weight.device
        U, I = U.to(torch.float32).to(device), I.to(torch.float32).to(device)

        # 判断是否使用稀疏矩阵运算
        use_sparse = H.is_sparse

        # 分别计算 M_u 和 M_i
        M_u = self.compute_matrix(H, U, device, use_sparse)
        M_i = self.compute_matrix(H.transpose(0, 1), I, device, use_sparse)

        # 计算最终输出
        U_out = torch.matmul(M_u, self.weight) + self.bias
        I_out = torch.matmul(M_i, self.weight) + self.bias

        return U_out, I_out


class DHCF(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, dropout,
                 device):
        super(DHCF, self).__init__()
        # 传入的参数
        self.device = device
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight
        self.dim_embedding = dim_E

        self.use_sparse = True  # 控制是否使用稀疏矩阵

        self.user_e = None
        self.item_e = None

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        values = torch.ones(len(edge_index))

        # 先将 numpy 数组转换为 torch tensor
        indices = np.vstack((edge_index[:, 0], adjusted_item_ids))
        indices = torch.LongTensor(indices)

        # 根据是否使用稀疏矩阵，创建相应的矩阵
        if self.use_sparse:
            self.interaction_matrix = torch.sparse_coo_tensor(
                indices, values, (self.num_user, self.num_item), dtype=torch.float32, device=self.device
            )
        else:
            # 创建稠密矩阵
            self.interaction_matrix = torch.zeros((self.num_user, self.num_item), dtype=torch.float32)
            self.interaction_matrix[indices[0], indices[1]] = 1

        # 自定义嵌入和参数
        self.user_embedding = nn.Embedding(num_user, dim_E).to(self.device)
        self.item_embedding = nn.Embedding(num_item, dim_E).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.layers = [DJconv(dim_E, dim_E).to(self.device) for _ in range(n_layers)]
        self.dropout = [nn.Dropout(dropout).to(self.device) for _ in range(n_layers)]

    def forward(self):
        U = self.user_embedding.weight.to(self.device)
        I = self.item_embedding.weight.to(self.device)
        H = self.interaction_matrix.to(self.device)
        U_out = U.clone()
        I_out = I.clone()
        for idx, layer in enumerate(self.layers):
            U = self.dropout[idx](U)
            I = self.dropout[idx](I)
            U, I = layer(H, U, I)
            U_out = torch.concat((U_out, U.to(self.device)), dim=1)
            I_out = torch.concat((I_out, I.to(self.device)), dim=1)

        self.user_e = U_out
        self.item_e = I_out

        return U_out, I_out

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        U_out, I_out = self.forward()

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, U_out, I_out)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, U_out, I_out)
        total_loss = bpr_loss + reg_loss

        return total_loss

    def bpr_loss(self, users, pos_items, neg_items, U_out, I_out):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = U_out[users]
        pos_item_embeddings = I_out[pos_items]
        neg_item_embeddings = I_out[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items, U_out, I_out):
        # 计算正则化损失
        user_embeddings = U_out[users]
        pos_item_embeddings = I_out[pos_items]
        neg_item_embeddings = I_out[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.user_e[:self.num_user].cpu()
        item_tensor = self.item_e[:self.num_item].cpu()

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
