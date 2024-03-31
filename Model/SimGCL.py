"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/31 19:18
@File : SimGCL.py
@function :
"""
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


class SimGCL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, ssl_temp,
                 ssl_reg, device):
        super(SimGCL, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.ssl_temp = ssl_temp  # 0.2
        self.ssl_reg = ssl_reg
        self.device = device

        self.eps = 0.1  # 扰动强度，用于对比学习中引入的扰动

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        self.user_embedding = nn.Embedding(num_embeddings=self.num_user, embedding_dim=self.dim_E)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_item, embedding_dim=self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

    def get_norm_adj_mat(self):
        # 创建一个空的稀疏矩阵A，大小为(n_users+n_items) x (n_users+n_items)，数据类型为float32
        A = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        # 加载交互矩阵
        inter_M = self.interaction_matrix
        # 将交互矩阵转置，以便获取物品到用户的关系
        inter_M_t = self.interaction_matrix.transpose()
        # 将用户到物品的交互和物品到用户的交互合并到邻接矩阵A中
        # 注意：物品的索引需要加上用户的数量，因为矩阵的前半部分是用户，后半部分是物品
        # nnz 属性表示稀疏矩阵中的非零元素的数量
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        # 更新稀疏矩阵A的数据
        A._update(data_dict)

        # 归一化邻接矩阵
        # 计算A中每个节点的度（即每行的非零元素个数）
        sumArr = (A > 0).sum(axis=1)
        # 为了防止除以0，给度数加上一个很小的数epsilon
        diag = np.array(sumArr.flatten())[0] + 1e-7
        # 度数的-0.5次幂，用于拉普拉斯矩阵的归一化
        diag = np.power(diag, -0.5)
        # 将numpy数组转换为torch张量，并移动到模型的设备上（CPU或GPU）
        self.diag = torch.from_numpy(diag).to(self.device)
        # 创建对角矩阵D
        D = sp.diags(diag)
        # 使用D对A进行归一化：L = D^-0.5 * A * D^-0.5
        L = D @ A @ D
        # 将归一化后的L转换为COO格式的稀疏矩阵，以便后续转换为torch稀疏张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        # 创建torch张量来表示稀疏矩阵的坐标(indices)和值(values)
        rows_and_cols = np.array([row, col])  # 将行和列的列表转换成numpy数组
        i = torch.tensor(rows_and_cols, dtype=torch.long)  # 从numpy数组创建张量
        data = torch.FloatTensor(L.data)
        # 创建torch的稀疏张量来表示归一化的邻接矩阵
        # SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        return SparseL

    def forward(self, perturbed=False):
        # 将用户和物品的嵌入向量拼接成一个大的嵌入矩阵
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        all_embeddings = []  # 用于存储每一层的嵌入
        for k in range(self.n_layers):
            # 使用归一化的邻接矩阵进行图卷积，更新嵌入
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                # 如果启用扰动，向嵌入中添加随机噪声
                random_noise = torch.rand_like(ego_embeddings).cuda()
                # 扰动是基于嵌入的符号和规范化的随机噪声的乘积
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        # 将所有层的嵌入堆叠起来，然后计算平均值得到最终的嵌入
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # 将合并的嵌入分割为用户嵌入和物品嵌入
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item])
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

    def regularization_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        # 计算正则化损失
        user_embeddings = user_emb[users]
        pos_item_embeddings = item_emb[pos_items]

        reg_loss = self.reg_weight * (torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2))

        return reg_loss

    def cal_cl_loss(self, users, pos_items):
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[users], user_view_2[users], self.ssl_temp)
        item_cl_loss = InfoNCE(item_view_1[pos_items], item_view_2[pos_items], self.ssl_temp)

        return user_cl_loss + item_cl_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        self.user_emb, self.item_emb = self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, self.user_emb, self.item_emb)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, self.user_emb, self.item_emb)
        cl_loss = self.ssl_reg * self.cal_cl_loss(users, pos_items)

        loss = bpr_loss + reg_loss + cl_loss

        return loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入

        user_tensor = self.user_emb[:self.num_user].cpu()
        item_tensor = self.item_emb[:self.num_item].cpu()

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

