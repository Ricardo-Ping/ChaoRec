"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/13 14:08
@File : MVGAE.py
@function :
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.inits import uniform
from torch.autograd import Variable

EPS = 1e-15
MAX_LOGVAR = 10


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index.long(), num_nodes=x.size(0))
            edge_index = edge_index.long()
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return F.dropout(aggr_out, p=0.1, training=self.training)

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class ProductOfExperts(torch.nn.Module):
    def __init__(self):
        super(ProductOfExperts, self).__init__()

    def forward(self, mu, logvar, eps=1e-8):
        """
        前向传播方法，计算多个高斯专家输出乘积的参数。根据专家乘积原理融合特定于模态的高斯节点嵌入

        参数:
        - mu (Tensor): 形状为 M x D 的张量，表示 M 个专家的均值。
        - logvar (Tensor): 形状为 M x D 的张量，表示 M 个专家的对数方差。
        - eps (float): 用于数值稳定性的小正数，默认值为 1e-8。

        返回:
        - pd_mu (Tensor): 乘积高斯分布的均值。
        - pd_logvar (Tensor): 乘积高斯分布的对数方差。
        - pd_var (Tensor): 乘积高斯分布的方差。
        """
        # 计算方差，并加上一个小的正数 eps 以避免除以零
        var = torch.exp(logvar) + eps
        # 计算每个高斯专家在某点 x 的精度（方差的倒数）
        T = 1. / var
        # 计算乘积高斯分布的均值  μf
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        # 计算乘积高斯分布的方差
        pd_var = 1. / torch.sum(T, dim=0)
        # 计算乘积高斯分布的对数方差
        pd_logvar = torch.log(pd_var)

        return pd_mu, pd_logvar, pd_var


class GCN(torch.nn.Module):
    def __init__(self, device, features, edge_index, num_user, num_item, dim_id, aggr_mode, concate,
                 num_layer, dim_latent=None):
        super(GCN, self).__init__()
        # 初始化成员变量
        self.device = device
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)  # 特征的维度
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer

        # 根据是否指定dim_latent使用不同的初始化策略
        if self.dim_latent:
            # 初始化用户偏好
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(
                self.device)
            # 特征到潜在空间的映射
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            nn.init.xavier_normal_(self.MLP.weight)
            # 定义图卷积层和线性层
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)
        else:
            # 如果没有指定dim_latent，则直接使用原始特征
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(
                self.device)
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        # 更多图卷积和线性层的定义，用于构建多层GCN
        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_4 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_4.weight)
        self.linear_layer4 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer4.weight)
        self.g_layer4 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer4.weight)

        self.conv_embed_5 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_5.weight)
        self.linear_layer5 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer5.weight)
        self.g_layer5 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer5.weight)

    def forward(self):
        # 首先根据是否有dim_latent，将特征通过一个全连接层（如果有）
        temp_features = self.MLP(self.features) if self.dim_latent else self.features
        # 将用户偏好和特征拼接
        x = torch.cat((self.preference, temp_features), dim=0)
        # 对拼接后的特征进行归一化
        x = F.normalize(x).to(self.device)

        # 通过图卷积层和线性层进行特征转换
        if self.num_layer > 0:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))  # Eq(4)
            x_hat = F.leaky_relu(self.linear_layer1(x))  # Eq(5)
            x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer1(h))  # Eq(6)
            del x_hat
            del h

        if self.num_layer > 1:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer2(x))
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer2(h))
            del x_hat
            del h

        if self.num_layer > 2:
            h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer3(x))
            x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer3(h))
            del h
            del x_hat

        # 计算均值和对数方差，用于后续的重参数化技巧
        # 使用两个独立的基于 GCN 的层来学习 μm和 log(σ^2)
        # 删除了激活函数 LeakyReLU来学习μm和 log(σ^2)的正负数值
        mu = F.leaky_relu(self.conv_embed_4(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer4(x))
        mu = self.g_layer4(torch.cat((mu, x_hat), dim=1)) if self.concate else self.g_layer4(mu) + x_hat
        del x_hat

        logvar = F.leaky_relu(self.conv_embed_5(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer5(x))
        logvar = self.g_layer5(torch.cat((logvar, x_hat), dim=1)) if self.concate else self.g_layer5(logvar) + x_hat
        del x_hat

        return mu, logvar


class MVGAE(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight,
                 n_layers, device):
        super(MVGAE, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.kl_weight = reg_weight
        self.device = device
        self.n_layers = n_layers
        # 聚合模式和是否拼接的标志
        self.aggr_mode = 'add'
        self.concate = False
        self.v_feat = v_feat
        self.t_feat = t_feat

        # 初始化协同过滤的项目嵌入向量
        self.collaborative = nn.init.xavier_normal_(torch.rand((self.num_item, self.dim_E), requires_grad=True)).to(
            self.device)

        # 转置并设置为无向图
        self.edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 初始化专家系统
        self.experts = ProductOfExperts()

        # 初始化多模态图卷积
        self.v_gcn = GCN(self.device, self.v_feat, self.edge_index, self.num_user, self.num_item,
                         self.dim_E, self.aggr_mode, self.concate, num_layer=self.n_layers, dim_latent=128)
        self.t_gcn = GCN(self.device, self.t_feat, self.edge_index, self.num_user, self.num_item, self.dim_E,
                         self.aggr_mode, self.concate, num_layer=self.n_layers, dim_latent=128)

        # 初始化协同过滤的GCN
        self.c_gcn = GCN(self.device, self.collaborative, self.edge_index, self.num_user, self.num_item,
                         self.dim_E, self.aggr_mode, self.concate, num_layer=self.n_layers, dim_latent=128)

        # 初始化结果嵌入向量
        self.result_embed = nn.init.xavier_normal_(torch.rand((self.num_user + self.num_item, self.dim_E))).to(
            self.device)

    def reparametrize(self, mu, logvar):
        """
        重参数化技巧，用于变分自编码器。
        :param mu: 均值向量
        :param logvar: 对数方差向量
        :return: 重参数化后的样本
        """
        # 对对数方差进行裁剪以提高数值稳定性
        logvar = logvar.clamp(max=MAX_LOGVAR)
        if self.training:
            # 如果处于训练状态，根据均值和方差生成样本
            return mu + torch.randn_like(logvar) * 0.1 * torch.exp(logvar.mul(0.5))
        else:
            # 如果不是训练状态，直接返回均值
            return mu

    def dot_product_decode_neg(self, z, user, neg_items, sigmoid=True):
        """
        对负样本进行解码，计算用户和负样本物品之间的交互概率。
        :param z: 所有节点的嵌入向量。
        :param user: 用户节点的索引。
        :param neg_items: 负样本物品节点的索引列表。
        :param sigmoid: 是否应用sigmoid函数，默认为True。
        :return: 用户和每个负样本物品之间交互的概率。
        """
        # 将用户索引扩展为与负样本数量相同的维度
        users = torch.unsqueeze(user, 1)
        re_users = users.repeat(1, neg_items.size(0))

        # 计算用户和负样本物品之间的点积，进而计算交互概率
        neg_values = torch.sum(z[re_users] * z[neg_items + self.num_user], -1)
        max_neg_value = torch.max(neg_values, dim=-1).values
        return torch.sigmoid(max_neg_value) if sigmoid else max_neg_value

    def dot_product_decode(self, z, edge_index, sigmoid=True):
        """
        对正样本或所有样本进行解码，计算节点间的交互概率。
        :param z: 所有节点的嵌入向量。
        :param edge_index: 边的索引，表示节点间的交互。
        :param sigmoid: 是否应用sigmoid函数，默认为True。
        :return: 节点间交互的概率。
        """
        value = torch.sum(z[edge_index[0]] * z[edge_index[1] + self.num_user], dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self):
        # 分别从视觉、文本和协同过滤GCN中获取均值和方差
        v_mu, v_logvar = self.v_gcn()  # [U+I, D]
        t_mu, t_logvar = self.t_gcn()
        c_mu, c_logvar = self.c_gcn()

        # 结合视觉和文本的信息,从专家系统中输出
        mu = torch.stack([v_mu, t_mu], dim=0)  # [2, U+I, D]
        logvar = torch.stack([v_logvar, t_logvar], dim=0)
        pd_mu, pd_logvar, _ = self.experts(mu, logvar)  # [U+I, D]
        del mu
        del logvar

        # 进一步与协同过滤信息结合
        mu = torch.stack([pd_mu, c_mu], dim=0)
        logvar = torch.stack([pd_logvar, c_logvar], dim=0)
        pd_mu, pd_logvar, _ = self.experts(mu, logvar)
        del mu
        del logvar

        # 应用重参数化技巧，来生成z
        z = self.reparametrize(pd_mu, pd_logvar)  # [U+I, D]

        # 根据数据集的稀疏性调整最终嵌入的表示
        # self.result_embed = torch.sigmoid(pd_mu)  # [U+I, D]
        self.result_embed = pd_mu

        return pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar

    def recon_loss(self, z, pos_edge_index, user, neg_items):
        """
        计算重构损失，对正样本边和负样本边的二元交叉熵损失。
        :param z: 潜在空间张量 Z。
        :param pos_edge_index: 正样本边的索引。
        :param user: 用户节点索引。
        :param neg_items: 负样本物品节点的索引列表。
        :return: 重构损失值。
        """
        # 对于稀疏数据集如amazon，使用sigmoid函数进行正则化以获得更好的结果
        # z = torch.sigmoid(z)

        # 计算正样本的得分
        pos_scores = self.dot_product_decode(z, pos_edge_index, sigmoid=True)
        # 计算负样本的得分
        neg_scores = self.dot_product_decode_neg(z, user, neg_items, sigmoid=True)
        # 计算重构损失，即正样本得分与负样本得分之差的二元交叉熵
        loss = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def kl_loss(self, mu, logvar):
        """
        计算KL散度损失，用于正则化潜在空间。
        :param mu: 潜在空间的均值向量 mu。
        :param logvar: 潜在空间的对数方差向量 log(sigma^2)。
        :return: KL散度损失值。
        """
        # 将对数方差限制在一个最大值以内，增加数值稳定性
        logvar = logvar.clamp(max=MAX_LOGVAR)
        # 计算KL散度损失
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    def bpr_loss(self, users, pos_items, neg_items, z):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = z[users]
        pos_item_embeddings = z[pos_items + self.num_user]
        neg_item_embeddings = z[neg_items + self.num_user]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
        pos_edge_index = torch.stack([users, pos_items], dim=0)

        pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar = self.forward()
        z_v = self.reparametrize(v_mu, v_logvar)
        z_t = self.reparametrize(t_mu, t_logvar)
        z_c = self.reparametrize(c_mu, c_logvar)

        # recon_loss = self.recon_loss(z, pos_edge_index, users, neg_items)
        recon_loss = self.bpr_loss(users, pos_items, neg_items, z)
        kl_loss = self.kl_loss(pd_mu, pd_logvar)
        loss_multi = recon_loss + self.kl_weight * kl_loss
        # loss_v = self.recon_loss(z_v, pos_edge_index, users, neg_items)+self.kl_weight * self.kl_loss(v_mu, v_logvar)
        # loss_t = self.recon_loss(z_t, pos_edge_index, users, neg_items)+self.kl_weight * self.kl_loss(t_mu, t_logvar)
        # loss_c = self.recon_loss(z_c, pos_edge_index, users, neg_items)+self.kl_weight * self.kl_loss(c_mu, c_logvar)
        loss_v = self.bpr_loss(users, pos_items, neg_items, z_v) + self.kl_weight * self.kl_loss(v_mu, v_logvar)
        loss_t = self.bpr_loss(users, pos_items, neg_items, z_t) + self.kl_weight * self.kl_loss(t_mu, t_logvar)
        loss_c = self.bpr_loss(users, pos_items, neg_items, z_c) + self.kl_weight * self.kl_loss(c_mu, c_logvar)

        total_loss = loss_multi + loss_v + loss_t + loss_c

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:].cpu()

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



