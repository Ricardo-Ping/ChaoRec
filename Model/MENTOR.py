"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/4/8 16:09
@File : MENTOR.py
@function :
"""
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, degree


class GCN(torch.nn.Module):
    # GCN类的初始化函数
    def __init__(self, num_user, num_item, dim_E, aggr_mode, device=None, features=None):
        super(GCN, self).__init__()
        # 初始化基本属性
        self.num_user = num_user  # 用户数量
        self.num_item = num_item  # 物品数量
        self.dim_feat = features.size(1)  # 特征维度
        self.dim_E = dim_E  # 潜在特征维度
        self.aggr_mode = aggr_mode  # 聚合模式
        self.device = device  # 设备

        # 初始化网络参数
        if self.dim_E:
            # 如果指定了潜在维度，则初始化偏好向量和两层多层感知机（MLP）
            self.preference = nn.Parameter(nn.init.xavier_normal_(
                torch.tensor(np.random.randn(num_user, self.dim_E), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_E)  # 第一层MLP
            self.MLP_1 = nn.Linear(4 * self.dim_E, self.dim_E)  # 第二层MLP
            # 初始化图卷积层
            self.conv_embed_1 = Base_gcn(self.dim_E, self.dim_E, aggr=self.aggr_mode)
        else:
            # 如果未指定潜在维度，则直接初始化偏好向量和图卷积层
            self.preference = nn.Parameter(nn.init.xavier_normal_(
                torch.tensor(np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)

    # GCN的前向传播函数
    def forward(self, edge_index, features, perturbed=False):
        # 计算特征的潜在表示
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_E else features
        # 将用户的偏好向量和物品的潜在特征拼接
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        # 对特征向量进行归一化
        x = F.normalize(x).to(self.device)

        # 通过图卷积层进行特征传播
        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            # 如果启用扰动模式，向特征中加入随机噪声
            random_noise = torch.rand_like(h).cuda()
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        # 进行第二次图卷积传播
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            # 如果启用扰动模式，再次向特征中加入随机噪声
            random_noise = torch.rand_like(h).cuda()
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1

        # 计算最终的特征表示，为原始特征向量和两次图卷积传播后的特征向量之和(这里只有三层)
        x_hat = x + h + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    # 构造函数初始化图卷积层
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)  # 调用父类构造函数，设置聚合方式
        self.aggr = aggr  # 聚合方式，例如：'add'表示邻居特征的加和
        self.in_channels = in_channels  # 输入特征的维度
        self.out_channels = out_channels  # 输出特征的维度

    # 前向传播函数，执行图卷积操作
    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)  # 移除自循环的边
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # 可以选择添加自循环
        x = x.unsqueeze(-1) if x.dim() == 1 else x  # 确保x是二维的
        # 调用propagate方法进行消息传递
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    # 消息函数，定义每条边上传递的消息
    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index  # 获取边的源节点和目标节点索引
            deg = degree(row, size[0], dtype=x_j.dtype)  # 计算每个节点的度
            deg_inv_sqrt = deg.pow(-0.5)  # 计算度的-1/2次方，用于归一化
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # 计算归一化系数
            return norm.view(-1, 1) * x_j  # 返回归一化后的消息
        return x_j  # 如果不是'add'聚合方式，则不进行归一化

    # 更新函数，对聚合后的消息进行更新（这里直接返回聚合结果，不做额外处理）
    def update(self, aggr_out):
        return aggr_out

    # 字符串表示方法
    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class MENTOR(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, mm_layers,
                 reg_weight, ssl_temp, dropout, align_weight, mask_weight_g, mask_weight_f, device):
        super(MENTOR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.dropout = dropout  # 0.5
        self.temp = ssl_temp  # 0.2 0.4 0.6
        self.user_item_dict = user_item_dict
        self.device = device

        self.knn_k = 10
        self.mm_layers = mm_layers  # 1
        self.mm_image_weight = 0.5
        self.aggr_mode = 'add'
        self.align_weight = align_weight  # 0.1
        self.mask_weight_g = mask_weight_g  # 0.001 0.0001 图掩码权重
        self.mask_weight_f = mask_weight_f  # 特征掩码权重 1.5

        self.mlp = nn.Linear(2 * self.dim_E, 2 * self.dim_E)  # 一个多层感知机网络
        self.v_feat = v_feat
        self.t_feat = t_feat

        # 多模态特征
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        # self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        # self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)

        # 多模态项目-项目图
        indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        del text_adj
        del image_adj

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 初始化用户和物品的权重参数
        # 对每一行的两个元素进行了 softmax 计算，使得每一行的两个值相加等于1，代表了概率分布
        # [num-user, 2, 1]
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.v_gcn = GCN(num_user, num_item, self.dim_E, self.aggr_mode, device=self.device, features=v_feat)
        self.v_gcn_n1 = GCN(num_user, num_item, self.dim_E, self.aggr_mode, device=self.device, features=v_feat)
        self.v_gcn_n2 = GCN(num_user, num_item, self.dim_E, self.aggr_mode, device=self.device, features=v_feat)

        self.t_gcn = GCN(num_user, num_item, self.dim_E, self.aggr_mode, device=self.device, features=t_feat)
        self.t_gcn_n1 = GCN(num_user, num_item, self.dim_E, self.aggr_mode, device=self.device, features=t_feat)
        self.t_gcn_n2 = GCN(num_user, num_item, self.dim_E, self.aggr_mode, device=self.device, features=t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, self.dim_E), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(num_user, num_item, self.dim_E, self.aggr_mode, device=self.device, features=self.id_feat)

        self.result_embed = None
        self.result_embed_guide = None
        self.result_embed_v = None
        self.result_embed_t = None
        self.result_embed_n1 = None
        self.result_embed_n2 = None

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def buildItemGraph(self, h):
        for i in range(self.mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def get_knn_adj_mat(self, mm_embeddings):
        # 归一化特征向量，使得每个向量的L2范数为1
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        # 计算归一化后的特征向量之间的相似度
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        # 获取每个向量的最近k个邻居的索引
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        # 记录相似度矩阵的大小
        adj_size = sim.size()
        # 释放相似度矩阵的内存
        del sim
        # 构造邻接矩阵的索引
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # 返回邻接矩阵的索引和通过`compute_normalized_laplacian`计算的归一化Laplacian矩阵
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        # 使用邻接矩阵的索引和全为1的值构造稀疏矩阵
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        # 计算每行的和，即每个节点的度数，并加上一个非常小的数避免除以0
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        # 计算度数的-1/2次方，用于归一化
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        # 计算归一化因子
        values = rows_inv_sqrt * cols_inv_sqrt
        # 返回归一化Laplacian矩阵
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def fit_Gaussian_dis(self):
        # 计算result_embed的方差和均值，代表整个结果嵌入的分布情况
        r_var = torch.var(self.result_embed)  # 计算方差
        r_mean = torch.mean(self.result_embed)  # 计算均值

        # 计算result_embed_guide的方差和均值，代表引导结果嵌入的分布情况
        g_var = torch.var(self.result_embed_guide)  # 计算方差
        g_mean = torch.mean(self.result_embed_guide)  # 计算均值

        # 计算result_embed_v的方差和均值，代表视觉结果嵌入的分布情况
        v_var = torch.var(self.result_embed_v)  # 计算方差
        v_mean = torch.mean(self.result_embed_v)  # 计算均值

        # 计算result_embed_t的方差和均值，代表文本结果嵌入的分布情况
        t_var = torch.var(self.result_embed_t)  # 计算方差
        t_mean = torch.mean(self.result_embed_t)  # 计算均值

        # 返回所有计算得到的方差和均值
        return r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean

    def forward(self):
        # 使用GCN处理不同的模态（身份信息、视觉特征、文本特征）
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, self.id_preference = self.id_gcn(self.edge_index, self.id_feat)

        # 对视觉和文本模态引入随机噪声进行数据增强
        self.v_rep_n1, _ = self.v_gcn_n1(self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n1, _ = self.t_gcn_n1(self.edge_index, self.t_feat, perturbed=True)
        self.v_rep_n2, _ = self.v_gcn_n2(self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n2, _ = self.t_gcn_n2(self.edge_index, self.t_feat, perturbed=True)

        # 对不同模态的嵌入进行拼接 v, t, id, and vt modalities
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)  # 视觉-文本表示 [num-user + num-item, 2 * dim_E]
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)  # id表示
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)  # 视觉表示
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)  # 文本表示

        # 对加入随机噪声的嵌入进行拼接
        representation_n1 = torch.cat((self.v_rep_n1, self.t_rep_n1), dim=1)  # 视觉-文本噪声表示
        representation_n2 = torch.cat((self.v_rep_n2, self.t_rep_n2), dim=1)

        # 调整嵌入的形状以匹配不同的表示需求
        self.v_rep = torch.unsqueeze(self.v_rep, 2)  # [num-user + num-item, dim_E, 1]
        self.t_rep = torch.unsqueeze(self.t_rep, 2)
        self.id_rep = torch.unsqueeze(self.id_rep, 2)

        # 结合用户的视觉偏好和文本偏好
        # [num-user, dim_E, 2]
        user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep  # 加权不同模态，表明了不同模态对用户偏好的贡献可能不同
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        # 以下步骤重复上述过程，生成引导用户嵌入、视觉用户嵌入和文本用户嵌入
        guide_user_rep = torch.cat((self.id_rep[:self.num_user], self.id_rep[:self.num_user]), dim=2)
        guide_user_rep = torch.cat((guide_user_rep[:, :, 0], guide_user_rep[:, :, 1]), dim=1)

        v_user_rep = torch.cat((self.v_rep[:self.num_user], self.v_rep[:self.num_user]), dim=2)
        v_user_rep = torch.cat((v_user_rep[:, :, 0], v_user_rep[:, :, 1]), dim=1)

        t_user_rep = torch.cat((self.t_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
        t_user_rep = torch.cat((t_user_rep[:, :, 0], t_user_rep[:, :, 1]), dim=1)

        # 对带有噪声的嵌入进行用户表示的生成
        self.v_rep_n1 = torch.unsqueeze(self.v_rep_n1, 2)  # [num-user + num-item, dim_E, 1]
        self.t_rep_n1 = torch.unsqueeze(self.t_rep_n1, 2)
        user_rep_n1 = torch.cat((self.v_rep_n1[:self.num_user], self.t_rep_n1[:self.num_user]), dim=2)
        user_rep_n1 = self.weight_u.transpose(1, 2) * user_rep_n1
        user_rep_n1 = torch.cat((user_rep_n1[:, :, 0], user_rep_n1[:, :, 1]), dim=1)

        self.v_rep_n2 = torch.unsqueeze(self.v_rep_n2, 2)
        self.t_rep_n2 = torch.unsqueeze(self.t_rep_n2, 2)
        user_rep_n2 = torch.cat((self.v_rep_n2[:self.num_user], self.t_rep_n2[:self.num_user]), dim=2)
        user_rep_n2 = self.weight_u.transpose(1, 2) * user_rep_n2
        user_rep_n2 = torch.cat((user_rep_n2[:, :, 0], user_rep_n2[:, :, 1]), dim=1)

        # 生成物品嵌入
        item_rep = representation[self.num_user:]  # 融合视觉-文本表示的项目嵌入
        item_rep_n1 = representation_n1[self.num_user:]
        item_rep_n2 = representation_n2[self.num_user:]

        guide_item_rep = guide_representation[self.num_user:]
        v_item_rep = v_representation[self.num_user:]
        t_item_rep = t_representation[self.num_user:]

        # 构建物品-物品图，并更新物品嵌入
        # 这里与FREEDOM不同，使用GNN之后的嵌入再次放入项目-项目图中进行学习
        h = self.buildItemGraph(item_rep)
        h_guide = self.buildItemGraph(guide_item_rep)
        h_v = self.buildItemGraph(v_item_rep)
        h_t = self.buildItemGraph(t_item_rep)
        h_n1 = self.buildItemGraph(item_rep_n1)
        h_n2 = self.buildItemGraph(item_rep_n2)

        user_rep = user_rep
        item_rep = item_rep + h

        item_rep_n1 = item_rep_n1 + h_n1
        item_rep_n2 = item_rep_n2 + h_n2

        guide_item_rep = guide_item_rep + h_guide
        v_item_rep = v_item_rep + h_v
        t_item_rep = t_item_rep + h_t

        self.user_rep = user_rep
        self.item_rep = item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        self.guide_user_rep = guide_user_rep
        self.guide_item_rep = guide_item_rep
        self.result_embed_guide = torch.cat((guide_user_rep, guide_item_rep), dim=0)

        self.v_user_rep = v_user_rep
        self.v_item_rep = v_item_rep
        self.result_embed_v = torch.cat((v_user_rep, v_item_rep), dim=0)

        self.t_user_rep = t_user_rep
        self.t_item_rep = t_item_rep
        self.result_embed_t = torch.cat((t_user_rep, t_item_rep), dim=0)

        self.user_rep_n1 = user_rep_n1
        self.item_rep_n1 = item_rep_n1
        self.result_embed_n1 = torch.cat((user_rep_n1, item_rep_n1), dim=0)

        self.user_rep_n2 = user_rep_n2
        self.item_rep_n2 = item_rep_n2
        self.result_embed_n2 = torch.cat((user_rep_n2, item_rep_n2), dim=0)

    def bpr_loss(self, users, pos_items, neg_items):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = self.result_embed[users]
        pos_item_embeddings = self.result_embed[self.num_user + pos_items]
        neg_item_embeddings = self.result_embed[self.num_user + neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users):
        # 计算正则化损失
        reg_embedding_loss_v = (self.v_preference[users] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[users] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items)
        reg_loss = self.regularization_loss(users)

        # 特征掩码
        with torch.no_grad():
            u_temp, i_temp = self.user_rep.clone(), self.item_rep.clone()
            u_temp2, i_temp2 = self.user_rep.clone(), self.item_rep.clone()
            u_temp.detach()
            i_temp.detach()
            u_temp2.detach()
            i_temp2.detach()
            u_temp2 = self.mlp(u_temp2)
            i_temp2 = self.mlp(i_temp2)
            u_temp = F.dropout(u_temp, self.dropout)
            i_temp = F.dropout(i_temp, self.dropout)
        mask_loss_u = 1 - F.cosine_similarity(u_temp, u_temp2).mean()
        mask_loss_i = 1 - F.cosine_similarity(i_temp, i_temp2).mean()
        mask_f_loss = self.mask_weight_f * (mask_loss_i + mask_loss_u)

        # 对齐损失
        r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean = self.fit_Gaussian_dis()
        align_loss = ((torch.abs(g_var - r_var) +
                       torch.abs(g_mean - r_mean)).mean() +  # id直接指导
                      (torch.abs(g_var - v_var) +
                       torch.abs(g_mean - v_mean)).mean() +  # id间接指导
                      (torch.abs(g_var - t_var) +
                       torch.abs(g_mean - t_mean)).mean() +  # id间接指导
                      (torch.abs(r_var - v_var) +
                       torch.abs(r_mean - v_mean)).mean() +  # 模态直接对齐
                      (torch.abs(r_var - t_var) +
                       torch.abs(r_mean - t_mean)).mean() +  # 模态直接对齐
                      (torch.abs(v_var - t_var) +
                       torch.abs(v_mean - t_mean)).mean())  # 模态间接对齐
        align_loss = align_loss * self.align_weight

        # 图噪声掩码
        mask_g_loss = (self.InfoNCE(self.result_embed_n1[:self.num_user], self.result_embed_n2[:self.num_user], self.temp)
                       + self.InfoNCE(self.result_embed_n1[self.num_user:], self.result_embed_n2[self.num_user:],
                                      self.temp))

        mask_g_loss = mask_g_loss * self.mask_weight_g

        total_loss = bpr_loss + reg_loss + align_loss + mask_f_loss + mask_g_loss

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




