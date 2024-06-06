"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/9 19:43
@File : MGCL.py
@function :
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicGCN import GCNConv
from MAD import mad_value


class MGCL(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight,
                 n_layers, aggr_mode, ssl_temp, ssl_alpha, device):
        super(MGCL, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.n_layers = n_layers
        self.ssl_temp = ssl_temp
        self.ssl_alpha = ssl_alpha
        self.device = device
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.reg_weight = reg_weight
        self.aggr_mode = aggr_mode

        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.user_embedding_v = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding_v.weight)
        self.user_embedding_t = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding_t.weight)

        # 多模态特征线性转换
        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)
        nn.init.xavier_uniform_(self.image_trs.weight)
        nn.init.xavier_uniform_(self.text_trs.weight)
        self.lambda_m = nn.Parameter(torch.tensor(0.1))  # 多模态融合系数

        # 转置并设置为无向图
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        # 定义图卷积层
        self.conv_layers = nn.ModuleList([GCNConv(self.dim_E, self.dim_E, aggr=self.aggr_mode)
                                          for _ in range(n_layers)])

    def forward(self):
        v_embedding = self.image_trs(self.v_feat)
        t_embedding = self.text_trs(self.t_feat)

        # =====================id模态卷积===================
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g, i_g = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)
        self.result = all_embeddings

        # =====================视觉模态卷积===================
        ego_embeddings = torch.cat((self.user_embedding_v.weight, v_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_v, i_v = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)

        # =====================文本模态卷积===================
        ego_embeddings = torch.cat((self.user_embedding_t.weight, t_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_t, i_t = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)

        return u_g, i_g, u_v, i_v, u_t, i_t

    def bpr_loss(self, users, pos_items, neg_items, u_g, i_g):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items, u_g, i_g):
        # 计算正则化损失
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def cl_loss(self, id, emb, visual, textual):
        emb = F.normalize(emb[id], p=2, dim=1)
        visual = F.normalize(visual[id], p=2, dim=1)
        text = F.normalize(textual[id], p=2, dim=1)

        logits_1 = torch.mm(emb, visual.T)
        logits_1 /= self.ssl_temp
        labels = torch.tensor(list(range(emb.shape[0]))).to(self.device)
        v_cl_loss = nn.CrossEntropyLoss()(logits_1, labels)

        logits_2 = torch.mm(emb, text.T)
        logits_2 /= self.ssl_temp
        labels = torch.tensor(list(range(emb.shape[0]))).to(self.device)
        t_cl_loss = nn.CrossEntropyLoss()(logits_2, labels)

        cl_loss = self.ssl_alpha * (v_cl_loss + t_cl_loss)

        return cl_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        u_g, i_g, u_v, i_v, u_t, i_t = self.forward()

        # 计算 BPR 损失和正则化损失
        bpr_loss_g = self.bpr_loss(users, pos_items, neg_items, u_g, i_g)
        bpr_loss_v = self.bpr_loss(users, pos_items, neg_items, u_v, i_v)
        bpr_loss_t = self.bpr_loss(users, pos_items, neg_items, u_t, i_t)
        bpr_loss = bpr_loss_g + bpr_loss_v + bpr_loss_t
        reg_loss_g = self.regularization_loss(users, pos_items, neg_items, u_g, i_g)
        reg_loss_v = self.regularization_loss(users, pos_items, neg_items, u_v, i_v)
        reg_loss_t = self.regularization_loss(users, pos_items, neg_items, u_t, i_t)
        reg_loss = reg_loss_g + reg_loss_v + reg_loss_t
        # 计算对比损失
        cl_loss_u = self.cl_loss(users, u_g, u_v, u_t)  # 用户
        cl_loss_t = self.cl_loss(pos_items, i_g, i_v, i_t)  # 项目
        cl_loss = cl_loss_u + cl_loss_t
        total_loss = bpr_loss + reg_loss + cl_loss

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
