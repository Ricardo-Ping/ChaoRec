"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/6/11 9:22
@File : FKAN_GCF.py
@function :
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from kanlayer import NaiveFourierKANLayer


class FourierGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim, grid_size):
        super(FourierGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        # self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = NaiveFourierKANLayer(in_dim, out_dim, self.grid_size)

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        # inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return features + x + inter_part2


class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse_coo_tensor(rc, val, x.shape)


class FKAN_GCF(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers,
                 node_dropout, message_dropout, grid_size, device):
        super(FKAN_GCF, self).__init__()
        self.diag = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.grid_size = grid_size
        self.device = device

        self.hidden_size_list = [self.dim_E] * self.n_layers

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)
        self.sparse_dropout = SparseDropout(self.node_dropout)

        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])):
            self.GNNlayers.append(FourierGNNLayer(input_size, output_size, self.grid_size))

        self.user_embedding = nn.Embedding(num_embeddings=self.num_user, embedding_dim=self.dim_E)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_item, embedding_dim=self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        # i = torch.LongTensor([row, col])
        indices = np.vstack((row, col)).astype(np.int64)
        i = torch.tensor(indices, dtype=torch.long)
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.
        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.num_item + self.num_user  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse_coo_tensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        A_hat = self.sparse_dropout(self.norm_adj_matrix) if self.node_dropout != 0 else self.norm_adj_matrix
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.num_user, self.num_item])

        return user_all_embeddings, item_all_embeddings

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = user_emb[users]
        pos_item_embeddings = item_emb[pos_items]
        neg_item_embeddings = item_emb[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items):
        # 计算正则化损失
        user_embeddings = self.user_embedding.weight[users]
        pos_item_embeddings = self.item_embedding.weight[pos_items]
        neg_item_embeddings = self.item_embedding.weight[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        user_emb, item_emb = self.forward()
        self.user_emb_final = user_emb
        self.item_emb_final = item_emb

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
        # 正则化损失
        reg_loss = self.regularization_loss(users, pos_items, neg_items)

        loss = bpr_loss + reg_loss

        return loss

    def gene_ranklist(self, topk=50):
        # 用户嵌入和项目嵌入
        user_tensor = self.user_emb_final[:self.num_user].cpu()
        item_tensor = self.item_emb_final[:self.num_item].cpu()

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
