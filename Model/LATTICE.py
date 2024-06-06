"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/22 21:59
@File : LATTICE.py
@function :
"""
import random

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LATTICEGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(LATTICEGCNConv, self).__init__(aggr='add', **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index):
        edge_index = edge_index.long()

        row, col = edge_index

        # Compute normalization coefficient
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Apply normalization
        out = norm.view(-1, 1) * x_j

        return out

    def update(self, aggr_out):
        return aggr_out


# 构建 K-最近邻（K-NN）的邻居关系，用于创建加权邻接矩阵。 等式2
def build_knn_neighbourhood(adj, topk):
    # 找到每行（每个节点）的 top-k 最大值和对应的索引
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    # 创建一个新的加权邻接矩阵，只保留每个节点的 top-k 邻居的权重，其他权重设为 0
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    # # shape(num_item, num_item)，每一行只有 topk个非零元素，表示对应节点的 topk 个最近邻节点的连接权重
    return weighted_adjacency_matrix


# 计算归一化的 Laplacian 矩阵 等式3
def compute_normalized_laplacian(adj):
    # 计算每个节点的度（与其他节点的连接权重之和）
    rowsum = torch.sum(adj, -1)
    # 归一化
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


# 计算特征的相似性矩阵 等式1
def build_sim(context):
    # 传入的context是多模态嵌入
    # 计算每个节点特征的 L2 范数，并将每个特征向量除以其 L2 范数，得到单位向量
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    # 计算余弦相似度
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    # shape(num_item, num_item)  相当于邻接矩阵
    return sim


class LATTICE(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, feat_embed_dim,
                 reg_weight, n_layers, mm_layers, ii_topk, aggr_mode, lambda_coeff, device):
        super(LATTICE, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.weight_size = [64, 64]  # [64,64]
        # self.n_ui_layers = len(self.weight_size)  # 2
        self.weight_size = [self.dim_E] + self.weight_size
        self.topk = ii_topk  # 多模态图的topk
        self.device = device
        self.feat_embed_dim = feat_embed_dim
        self.lambda_coeff = lambda_coeff  # 跳跃连接系数
        self.mm_layers = mm_layers  # 多模态图卷积层数
        self.n_layers = n_layers  # 交互图卷积层数
        dropout_list = [0.1] * n_layers
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight

        # 初始化用户和项目嵌入
        self.user_embedding = nn.Embedding(num_user, self.dim_E)
        self.item_embedding = nn.Embedding(num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.gcn_layers = nn.ModuleList([LATTICEGCNConv(self.dim_E, self.dim_E, aggr_mode)
                                         for _ in range(self.n_layers)])

        # 多模态嵌入
        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)

        # 转置为无向图
        self.edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 处理图像和文本邻接矩阵，基于初始的图像和文本嵌入权重
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_neighbourhood(image_adj, topk=self.topk)
        image_adj = compute_normalized_laplacian(image_adj)

        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_neighbourhood(text_adj, topk=self.topk)
        text_adj = compute_normalized_laplacian(text_adj)

        self.text_original_adj = text_adj.to(device)
        self.image_original_adj = image_adj.to(device)

        # 初始化用于转换图像和文本特征的线性层。
        self.image_trs = nn.Linear(v_feat.shape[1], self.feat_embed_dim)
        self.text_trs = nn.Linear(t_feat.shape[1], self.feat_embed_dim)

        # 可学习的模态权重
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, build_item_graph=False):
        # 多模态降维
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        if build_item_graph:
            # 根据当前的特征重新构建邻接矩阵 等式4
            weight = self.softmax(self.modal_weight)
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.topk)

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.topk)

            # 多模态融合构建多模态项目-项目图(学习图)
            learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            # 初始图
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            # 跳跃连接 等式5和6 最终维度是(num_item, num_item)  聚合多模态潜在图
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()

        h = self.item_embedding.weight
        # 对项目-项目图的图卷积操作 等式7
        for i in range(self.mm_layers):
            h = torch.mm(self.item_adj, h)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = self.gcn_layers[i](ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)
        i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
        self.result = torch.cat((u_g_embeddings, i_g_embeddings), dim=0)
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

    def loss(self, users, pos_items, neg_items, build_item_graph):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeddings = self.forward(build_item_graph)

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
