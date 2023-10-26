"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/26 19:53
@File : DRAGON.py
@function :
"""
import os

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
import torch.nn.functional as F
from arg_parser import parse_args
from BasicGCN import GCNConv

args = parse_args()


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):

        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


# 单模态图卷积模块
class GCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_latent, feat_embed_dim, aggr_mode,
                 device, feat_size):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_latent = dim_latent  # id嵌入的维度
        self.aggr_mode = aggr_mode
        self.device = device
        self.dim_feat = feat_embed_dim  # 特征嵌入维度

        # 初始化用户偏好
        self.preference = nn.Parameter(
            torch.empty(num_user, self.dim_latent, dtype=torch.float32, device=self.device, requires_grad=True))
        nn.init.xavier_normal_(self.preference)

        self.MLP = nn.Linear(feat_size, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

        # 图卷积
        self.conv_embed_1 = GCNConv(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index, features):
        # 将特征进行降维
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        # temp_features = F.normalize(temp_features)
        # 拼接用户偏好和项目多模态特征
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        # 单模态卷积(这里的两层图卷积使用的是同一个)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = x + h + h_1  # 等式2

        return x_hat, self.preference


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph  # 用户索引
        u_features = features[index]  # 用户的特征
        user_matrix = user_matrix.unsqueeze(1)  # 用户权重矩阵

        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()

        return u_pre


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


class DRAGON(torch.nn.Module):
    def __init__(self, v_feat, t_feat, edge_index, user_item_dict, num_user, num_item, aggr_mode, n_layers,
                 dim_E, feature_embedding, reg_weight, device):
        super(DRAGON, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.item_topk = 10  # topk
        self.user_topk = 40  # topk
        self.aggr_mode = aggr_mode
        self.construction = 'cat'  # 多模态信息聚合方法
        self.reg_weight = reg_weight
        self.user_item_dict = user_item_dict
        self.v_rep = None
        self.a_rep = None
        self.t_rep = None
        self.device = device
        self.v_preference = None
        self.a_preference = None
        self.t_preference = None
        self.dim_latent = dim_E
        self.dim_feat = feature_embedding
        self.user_aggr_mode = 'softmax'
        self.n_layers = 2
        self.mm_image_weight = 0.5

        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)

        # 转换成无向图
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 初始化用户对于多模态偏好的权重
        self.weight_u = self.init_weight(self.num_user)
        self.weight_i = self.init_weight(self.num_item)

        # 这个是多模态用户偏好建模时候选择串联的时候的MLP
        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        # 初始化多模态特征
        self.v_feat = v_feat.clone().detach().to(self.device)
        self.t_feat = t_feat.clone().detach().to(self.device)

        # ======================异构图部分==============================
        # 创建单模态图卷积
        self.v_gcn = GCN(num_user, num_item, self.dim_latent, self.dim_feat, self.aggr_mode,
                         self.device, v_feat.size(1))
        self.t_gcn = GCN(num_user, num_item, self.dim_latent, self.dim_feat, self.aggr_mode,
                         self.device, t_feat.size(1))

        # ========================用户同构图============================
        # 初始化用户图采样
        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)

        # 加载用户-用户图数据
        dataset = args.data_path
        dir_str = './Data/' + dataset
        self.user_graph_dict = np.load(os.path.join(dir_str, 'user_graph_dict.npy'),
                                       allow_pickle=True).item()

        # =======================项目同构图===============================
        # 多模态嵌入
        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)
        # 初始化用于转换图像和文本特征的线性层。
        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_feat)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_feat)

        # 处理图像和文本邻接矩阵，基于初始的图像和文本嵌入权重
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_neighbourhood(image_adj, topk=self.item_topk)
        image_adj = compute_normalized_laplacian(image_adj)

        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_neighbourhood(text_adj, topk=self.item_topk)
        text_adj = compute_normalized_laplacian(text_adj)

        self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj

        # 最终的嵌入
        self.result = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, self.dim_latent)))).to(self.device)

    # 初始化权重
    def init_weight(self, num_entities):
        weight = nn.Parameter(torch.randn(num_entities, 2, 1), requires_grad=True)
        nn.init.xavier_normal_(weight)
        weight.data = F.softmax(weight.data, dim=1)
        return weight

    def forward(self):

        # ==========================单模态表示学习模块==============================
        # self.v_rep的维度是(N, 64) N代表节点数量
        # self.v_preference 应该代表对单模态的用户偏好
        user_rep = None
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)

        # ============================多模态信息构建模块============================
        if self.construction == 'cat':
            representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        else:
            representation = self.v_rep + self.t_rep

        # 串联 等式3
        if self.construction == 'cat':
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1, 2) * user_rep

            user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        # 项目嵌入
        item_rep = representation[self.num_user:]

        # ===========================多模态信息聚合模块=================================
        # 用户表示增强
        h_u = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)
        user_rep = user_rep + h_u

        # 项目表示增强
        h_i = item_rep
        # 对项目-项目图的图卷积操作 等式7
        for i in range(self.n_layers):
            h_i = torch.sparse.mm(self.mm_adj, h_i)
        item_rep = item_rep + h_i

        # 最终嵌入
        self.result = torch.cat((user_rep, item_rep), dim=0)

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

    def regularization_loss(self, users):

        reg_embedding_loss_v = (self.v_preference[users] ** 2).mean()
        reg_embedding_loss_t = (self.t_preference[users] ** 2).mean()

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)

        if self.construction == 'cat':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeddings = self.forward()

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, embeddings)
        reg_loss = self.regularization_loss(users)

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

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.user_topk)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def topk_sample(self, k):
        # 保存每个用户的最多k个相邻用户的索引
        user_graph_index = []
        count_num = 0
        # 保存与每个用户的邻居相关的权重
        # user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        user_weight_matrix = torch.zeros(self.num_user, k)
        # 如果某个用户没有足够的邻居，这个列表将被用作占位符
        tasike = [0] * k

        for i in range(self.num_user):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean

            user_graph_index.append(user_graph_sample)

        return user_graph_index, user_weight_matrix