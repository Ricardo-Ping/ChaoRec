"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/8 20:21
@File : VBPR.py
@function :
"""
import torch
import torch.nn as nn
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class VBPR(nn.Module):
    def __init__(self, num_user, num_item, user_item_dict, v_feat, embedding_dim, feature_embedding, reg_weight,
                 device):
        super(VBPR, self).__init__()
        self.result = None
        self.user_item_dict = user_item_dict
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.visual_embedding = 64
        # =======================gamma==========================
        # 用户隐式向量 gamma_u
        self.user_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, embedding_dim + self.visual_embedding)))

        # 物品隐式向量 gamma_i
        self.item_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, embedding_dim)))

        # 视觉特征
        # 读入多模态特征
        self.v_feat = nn.Embedding.from_pretrained(v_feat, freeze=False)
        self.item_linear = nn.Linear(v_feat.shape[1], self.visual_embedding)
        nn.init.xavier_uniform_(self.item_linear.weight)

        self.reg_weight = reg_weight  # 正则化系数

    def forward(self):
        visual_embeddings = self.item_linear(self.v_feat.weight)
        item_embeddings = torch.cat((self.item_embedding, visual_embeddings), dim=-1)

        self.result = torch.cat((self.user_embedding, item_embeddings), dim=0)

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

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeddings = self.forward()

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
