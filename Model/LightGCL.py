"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/8 19:20
@File : LightGCL.py
@function :
"""
import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F


class LightGCL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, aggr_mode,
                 ssl_alpha, ssl_temp, device):
        super(LightGCL, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.edge_index = edge_index
        self.n_layers = n_layers
        self.aggr_mode = aggr_mode
        self.device = device
        self.q = 5  # SVD近似的秩
        self.dropout = 0.0  # 丢失率
        self.temp = ssl_temp  # 温度系数
        self.lambda_1 = ssl_alpha  # 自监督学习损失的系数
        self.lambda_2 = reg_weight  # 正则化损失的系数
        self.act = nn.LeakyReLU(0.5)  # 激活函数

        # 获取用户和项目id序列
        self._user = self.edge_index[:, 0]
        self._item = self.edge_index[:, 1] - self.num_user

        # 获取正则化的邻接矩阵
        self.adj_norm = self.coo2tensor(self.create_adjust_matrix())

        # 执行SVD重构
        svd_u, s, svd_v = torch.svd_lowrank(self.adj_norm, q=self.q)
        self.u_mul_s = svd_u @ (torch.diag(s))
        self.v_mul_s = svd_v @ (torch.diag(s))
        del s  # 删除中间变量以节省内存
        self.ut = svd_u.T
        self.vt = svd_v.T

        # 初始化用户和物品的嵌入
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_user, self.dim_E)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_item, self.dim_E)))

        # 初始化层级列表
        # 存储每一图卷积层之后用户和物品的嵌入表示
        self.E_u_list = [None] * (self.n_layers + 1)
        self.E_i_list = [None] * (self.n_layers + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        # 存储每一图卷积层之后用户和物品嵌入的临时表示
        self.Z_u_list = [None] * (self.n_layers + 1)
        self.Z_i_list = [None] * (self.n_layers + 1)
        # 存储经过SVD重建后的图表示中的用户和物品嵌入
        self.G_u_list = [None] * (self.n_layers + 1)
        self.G_i_list = [None] * (self.n_layers + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        self.E_u = None  # 用于存储最终的用户嵌入
        self.E_i = None  # 用于存储最终的物品嵌入
        self.restore_user_e = None  # 用于恢复用户嵌入的辅助变量
        self.restore_item_e = None  # 用于恢复物品嵌入的辅助变量

    def create_adjust_matrix(self):
        # 获取用户和物品之间的正则化交互矩阵。

        # 创建一个所有元素都为1的数组，大小与用户交互的数量相同，表示每个交互的初始权重。
        ratings = np.ones_like(self._user, dtype=np.float32)
        # 使用用户ID和物品ID创建一个稀疏矩阵，形状为用户数量x物品数量。
        matrix = sp.csr_matrix(
            (ratings, (self._user, self._item)),
            shape=(self.num_user, self.num_item),
        ).tocoo()  # 将CSR格式的稀疏矩阵转换为COO格式，便于之后的处理。

        # 计算矩阵的每一行和每一列的和，用于后续的归一化处理。
        rowD = np.squeeze(np.array(matrix.sum(1)), axis=1)
        colD = np.squeeze(np.array(matrix.sum(0)), axis=0)

        # 遍历矩阵中的所有非零元素，按照特定的归一化公式修改它们的值。
        # 这里使用的是归一化公式: matrix[i,j] / sqrt(rowD[i] * colD[j])，
        # 目的是减少流行项对模型的影响，平衡用户和物品的交互频率。
        for i in range(len(matrix.data)):
            matrix.data[i] = matrix.data[i] / pow(rowD[matrix.row[i]] * colD[matrix.col[i]], 0.5)
        return matrix

    def coo2tensor(self, matrix: sp.coo_matrix):
        # 将COO格式的稀疏矩阵转换为PyTorch的稀疏张量。

        # 将矩阵的行索引和列索引堆叠起来，转换为PyTorch张量，用于表示稀疏张量中非零元素的位置。
        indices = torch.from_numpy(
            np.vstack((matrix.row, matrix.col)).astype(np.int64))
        # 将矩阵的数据部分转换为PyTorch张量，表示稀疏张量中非零元素的值。
        values = torch.from_numpy(matrix.data)
        # 定义稀疏张量的形状。
        shape = torch.Size(matrix.shape)
        # 使用上述组件创建PyTorch的稀疏张量，并确保它在正确的设备上（如CPU或GPU）。
        x = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(self.device)
        return x

    def sparse_dropout(self, matrix, dropout):
        # 如果dropout率为0，则不进行任何操作，直接返回原始矩阵。
        if dropout == 0.0:
            return matrix

        # 获取稀疏张量的索引。这些索引表示非零值在张量中的位置。
        indices = matrix.indices()

        # 对稀疏张量的值执行dropout操作。`F.dropout`随机将一些元素设置为0，根据概率`p=dropout`。
        # 注意，这里只对非零元素进行操作，因为稀疏张量的零值不存储在`values`中。
        values = F.dropout(matrix.values(), p=dropout)

        # 获取原始稀疏张量的大小，以便创建新的稀疏张量时使用。
        size = matrix.size()

        # 使用处理过的索引和值，以及原始的大小，创建一个新的稀疏张量。
        # 这里使用的是`torch.sparse.FloatTensor`来创建张量。
        return torch.sparse_coo_tensor(indices, values, size)

    def forward(self):
        # 遍历每一层图卷积网络
        for layer in range(1, self.n_layers + 1):
            # GNN传播过程
            # 对归一化的邻接矩阵应用稀疏dropout，然后与上一层的物品嵌入进行稀疏矩阵乘法（spmm），
            # 得到当前层的用户嵌入中间表示。
            self.Z_u_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout),
                                              self.E_i_list[layer - 1])

            # 得到当前层的物品嵌入中间表示。
            self.Z_i_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
                                              self.E_u_list[layer - 1])

            # 聚合操作：直接将计算得到的中间表示赋值给对应层的嵌入列表。
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        # 跨层聚合
        # 将所有层的用户和物品嵌入向量进行累加，得到最终的用户和物品嵌入向量。
        # 这一步骤实现了跨层特征的融合，增强了模型捕捉用户和物品特征的能力。
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        # 返回最终的用户和物品嵌入向量
        return self.E_u, self.E_i

    def bpr_loss(self, E_u_norm, E_i_norm, user, pos_item, neg_item):
        # 计算贝叶斯个性化排序（BPR）损失和参数正则化损失。

        # 从用户和物品的嵌入中索引出对应用户、正样本物品和负样本物品的嵌入向量。
        u_e = E_u_norm[user]  # 用户嵌入
        pi_e = E_i_norm[pos_item]  # 正样本物品嵌入
        ni_e = E_i_norm[neg_item]  # 负样本物品嵌入

        # 计算用户与正样本物品和负样本物品之间的得分（内积）。
        pos_scores = torch.mul(u_e, pi_e).sum(dim=1)
        neg_scores = torch.mul(u_e, ni_e).sum(dim=1)

        # 计算BPR损失，即用户对正样本的偏好程度高于对负样本的偏好程度的概率的对数似然。
        loss1 = -(pos_scores - neg_scores).sigmoid().log().mean()

        # 计算正则化损失，以避免过拟合。
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= self.lambda_2
        return loss1 + loss_reg

    def ssl_loss(self, E_u_norm, E_i_norm, user, pos_item):
        # 计算自监督学习（SSL）损失。

        # 通过SVD近似的图传播得到用户和物品的全局嵌入。
        for layer in range(1, self.n_layers + 1):
            vt_ei = self.vt @ self.E_i_list[layer - 1]  # 物品嵌入通过V矩阵传播
            self.G_u_list[layer] = self.u_mul_s @ vt_ei  # 更新用户全局嵌入
            ut_eu = self.ut @ self.E_u_list[layer - 1]  # 用户嵌入通过U矩阵传播
            self.G_i_list[layer] = self.v_mul_s @ ut_eu  # 更新物品全局嵌入

        # 跨层聚合全局嵌入。
        G_u_norm = sum(self.G_u_list)
        G_i_norm = sum(self.G_i_list)

        # 计算自监督损失。它由负样本的得分（通过softmax归一化）和正样本得分的负对数似然组成。
        neg_score = torch.log(torch.exp(G_u_norm[user] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[pos_item] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[user] * E_u_norm[user]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                    (torch.clamp((G_i_norm[pos_item] * E_i_norm[pos_item]).sum(1) / self.temp, -5.0,
                                 5.0)).mean()
        ssl_loss = -pos_score + neg_score  # 最终的SSL损失

        return self.lambda_1 * ssl_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        E_u_norm, E_i_norm = self.forward()
        self.restore_user_e = E_u_norm
        self.restore_item_e = E_i_norm
        bpr_loss = self.bpr_loss(E_u_norm, E_i_norm, users, pos_items, neg_items)
        ssl_loss = self.ssl_loss(E_u_norm, E_i_norm, users, pos_items)
        total_loss = bpr_loss + ssl_loss
        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.restore_user_e[:self.num_user].cpu()
        item_tensor = self.restore_item_e[:self.num_item].cpu()

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





