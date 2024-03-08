"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/7 20:28
@File : MultVAE.py
@function :
"""
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
import torch.nn.functional as F


class MultVAE(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E,  reg_weight, device):
        super(MultVAE, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.edge_index = edge_index
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.device = device
        self.keep_prob = 0.5  # dropout保留率
        self.total_anneal_steps = 200000  # KL散度权重的退火步数
        self.anneal_cap = 0.2  # KL散度权重的上限
        self.update_count = 0.0  # KL散度已完成的更新步数

        # 创建一个空的稀疏矩阵
        data = np.ones(edge_index.shape[0])
        row_indices = edge_index[:, 0]  # 用户编号作为行索引
        col_indices = edge_index[:, 1] - self.num_user  # 调整项目编号为从0开始的索引
        self.train_csr_mat = csr_matrix((data, (row_indices, col_indices)), shape=(self.num_user, self.num_item))

        # 设置解码器（p网络）的维度
        p_dims = [64]
        self.p_dims = p_dims + [self.num_item]

        # 设置编码器（q网络）的维度。如果未指定q_dims，则使用p_dims的反向作为q_dims
        self.q_dims = self.p_dims[::-1]

        # 用户和项目的嵌入表示的定义
        self.layers_q = nn.ModuleList()  # 用于分别存储编码器和解码器的网络层
        # 构建编码器网络层
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            # 如果是最后一层，输出维度翻倍，为潜在表示的均值和方差各预留空间
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2
            layer = nn.Linear(d_in, d_out, bias=True)
            # 对权重进行均匀分布初始化
            nn.init.uniform_(layer.weight)
            # 对偏置进行均匀分布初始化
            if layer.bias is not None:
                nn.init.uniform_(layer.bias)
            self.layers_q.append(layer)

        # 解码器网络层的定义
        self.layers_p = nn.ModuleList()
        # 构建解码器网络层
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            layer = nn.Linear(d_in, d_out, bias=True)
            # 对权重进行均匀分布初始化
            nn.init.uniform_(layer.weight)
            # 对偏置进行均匀分布初始化
            if layer.bias is not None:
                nn.init.uniform_(layer.bias)
            self.layers_p.append(layer)

        # Dropout层，用于防止过拟合，1-keep_prob计算丢弃概率
        self.dropout = nn.Dropout(1 - self.keep_prob)

    # 编码器处理输入数据，生成潜在表示的均值、方差，并计算KL散度
    def q_graph(self, input_x):
        # 编码器逻辑
        mu_q, std_q, kl_dist = None, None, None
        h = F.normalize(input_x, p=2, dim=1)  # 对输入数据进行L2归一化
        h = self.dropout(h)  # 应用dropout

        for i, layer in enumerate(self.layers_q):
            h = layer(h)  # 通过每一层
            if i != len(self.layers_q) - 1:
                h = F.tanh(h)  # 使用tanh激活函数，除了最后一层
            else:
                # 最后一层分割均值和对数方差
                size = int(h.shape[1] / 2)
                mu_q, logvar_q = torch.split(h, size, dim=1)
                std_q = torch.exp(0.5 * logvar_q)  # 计算标准差
                # 计算KL散度
                kl_dist = torch.sum(0.5 * (-logvar_q + logvar_q.exp() + mu_q.pow(2) - 1), dim=1).mean()

        return mu_q, std_q, kl_dist

    # 从潜在空间中的表示重构原始输入
    def p_graph(self, z):
        # 解码器逻辑
        h = z  # 输入潜在表示

        for i, layer in enumerate(self.layers_p):
            h = layer(h)  # 通过每一层
            if i != len(self.layers_p) - 1:
                h = F.tanh(h)  # 最后一层之前使用tanh激活函数

        return h

    def forward(self, input_x):
        # input_x 代表用户的交互
        # 编码器：计算潜在表示的均值、标准差和KL散度
        mu_q, std_q, kl_dist = self.q_graph(input_x)

        # 重参数化技巧：从正态分布中采样，以便使用梯度下降
        epsilon = torch.randn_like(std_q).to(self.device)  # 生成与std_q形状相同的随机噪声
        sampled_z = mu_q + epsilon * std_q  # 采样潜在表示

        # 解码器：重构输入
        reconstructed_x = self.p_graph(sampled_z)

        # 返回重构的输入和KL散度，用于计算损失函数
        return reconstructed_x, kl_dist

    def predict(self, input_x):
        ratings, _ = self.forward(input_x)

        return ratings

    def l2_regularization(self):
        # 初始化正则化损失为0
        reg_loss = 0.0
        # 遍历模型中所有的权重参数
        for param in self.parameters():
            # 累加参数的L2范数（即所有元素的平方和）
            reg_loss += torch.norm(param, p=2) ** 2
        # 将累加的L2范数乘以正则化系数的一半
        reg_loss = self.reg_weight * 0.5 * reg_loss
        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        users_cpu = users.cpu()
        bat_input = self.train_csr_mat[users_cpu.numpy()].toarray()
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        bat_input = torch.from_numpy(bat_input).float().to(self.device)

        # 前向传播，计算重构的输入和KL散度
        logits, kl_dist = self.forward(bat_input)
        # 计算重构误差，这里使用负对数似然作为示例
        log_softmax_var = F.log_softmax(logits, dim=-1)
        neg_ll = -torch.mul(log_softmax_var, bat_input).sum(dim=-1).mean()

        reg_loss = self.reg_weight * self.l2_regularization()

        loss = neg_ll + anneal * kl_dist + 2 * reg_loss

        self.update_count += 1

        return loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        bat_input = self.train_csr_mat.toarray()
        bat_input = torch.from_numpy(bat_input).float().to(self.device)
        ratings = self.predict(bat_input).cpu().detach().numpy()

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 生成评分矩阵
        # score_matrix = torch.matmul(user_tensor, item_tensor.t())

        # 将历史交互设置为极小值
        for row, col in self.user_item_dict.items():
            col = torch.LongTensor(list(col)) - self.num_user
            ratings[row][col] = 1e-6

        # 首先将ratings NumPy数组转换为PyTorch张量
        ratings_tensor = torch.from_numpy(ratings).to(self.device)
        # 然后在转换后的张量上应用torch.topk
        _, index_of_rank_list_train = torch.topk(ratings_tensor, topk)

        # 总的top-k列表
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
            dim=0)

        # 返回三个推荐列表
        return all_index_of_rank_list

