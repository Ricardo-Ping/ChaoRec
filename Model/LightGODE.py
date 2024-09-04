"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/9/3 9:22
@File : LightGODE.py
@function :
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, num_user, num_item, adj, k_hops=1):
        super(ODEFunc, self).__init__()
        self.g = adj
        self.num_user = num_user
        self.num_item = num_item

    def update_e(self, emb):
        self.e = emb

    def forward(self, t, x):
        ax = torch.spmm(self.g, x)
        f = ax + self.e  # 相当于加入自循环
        return f


class LightGODE(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, gamma, t, device):
        super(LightGODE, self).__init__()
        self.dim_E = dim_E
        self.num_user = num_user
        self.num_item = num_item
        self.gamma = gamma  # uniformity weight
        self.train_stage = 'pretrain'
        self.train_strategy = 'MF_init'
        if self.train_strategy == 'MF':
            self.use_mf = True
        elif self.train_strategy == 'GODE':
            self.use_mf = False
        else:  # MF_init
            self.use_mf = None
        self.t = torch.tensor([0, t])  # step
        self.solver = 'euler'  # 表示使用欧拉方法来近似解微分方程
        self.user_item_dict = user_item_dict
        self.device = device

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.user_embedding = torch.nn.Embedding(num_user, dim_E)
        self.item_embedding = torch.nn.Embedding(num_item, dim_E)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.odefunc1hop = ODEFunc(self.num_user, self.num_item, self.norm_adj, k_hops=1)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        indices = np.vstack((row, col))  # 将 row 和 col 堆叠成一个 2xN 的数组
        i = torch.LongTensor(indices)
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, user, item):
        if self.train_strategy == 'MF_init' and self.train_stage == 'pretrain':
            self.use_mf = self.training
        user_embeddings, item_embeddings = self.get_all_embeddings()
        user_e = user_embeddings[user]
        item_e = item_embeddings[item]
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    # MF init
    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()

        # not train -> ode for testing
        if not self.use_mf:
            t = self.t.type_as(all_embeddings)
            self.odefunc1hop.update_e(all_embeddings)
            z1 = odeint(self.odefunc1hop, all_embeddings, t, method=self.solver)[1]
            all_embeddings = z1

        # trian -> MF for training
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item])
        return user_all_embeddings, item_all_embeddings

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_e, item_e = self.forward(users, pos_items)
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        return align + uniform

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        if self.restore_user_e is None or self.restore_item_e is None:
            if self.train_strategy == 'MF_init':
                self.use_mf = self.training
            self.restore_user_e, self.restore_item_e = self.get_all_embeddings()
        user_tensor = self.restore_user_e.cpu()
        item_tensor = self.restore_item_e.cpu()

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
