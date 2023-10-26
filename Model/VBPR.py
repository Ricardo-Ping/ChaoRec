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
    def __init__(self, num_user, num_item, v_feat, embedding_dim, user_item_dict, device, weight_decay, feature_embedding):
        super(VBPR, self).__init__()
        self.user_item_dict = user_item_dict
        self.num_user = num_user
        self.device = device
        self.visual_embedding = feature_embedding
        # =======================beta============================
        self.item_bias = nn.Embedding(num_item, 1)  # 物品偏置项 b_i
        nn.init.zeros_(self.item_bias.weight)

        # =======================gamma==========================
        # 用户隐式向量 gamma_u
        self.user_embedding = nn.Embedding(num_user, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)

        # 物品隐式向量 gamma_i
        self.item_embedding = nn.Embedding(num_item, embedding_dim)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # =======================beta撇==========================
        # 视觉偏置项beta撇
        self.visual_bias = nn.Embedding(v_feat.size(1), 1)
        nn.init.zeros_(self.visual_bias.weight)

        # 用户视觉嵌入
        self.user_visual_embedding = nn.Embedding(num_user, self.visual_embedding)
        nn.init.xavier_normal_(self.user_visual_embedding.weight)

        # 嵌入矩阵
        self.E = nn.Linear(v_feat.size(1), self.visual_embedding, bias=False)
        nn.init.xavier_normal_(self.E.weight)

        # 视觉特征
        self.v_feat = nn.Embedding.from_pretrained(v_feat, freeze=True)

        self.weight_decay = weight_decay  # 正则化系数

    def forward(self, users, pos_items, neg_items):
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
        # 嵌入
        user_embedding = self.user_embedding(users).to(self.device)  # (batchsize, 64)
        positive_item_embedding = self.item_embedding(pos_items).to(self.device)  # (batchsize, 64)
        negative_item_embedding = self.item_embedding(neg_items).to(self.device)  # (batchsize, 64)
        # 用户视觉嵌入
        user_visual_embedding = self.user_visual_embedding(users).to(self.device)  # (batchsize, 256)

        # 偏置
        positive_item_bias = self.item_bias(pos_items).squeeze().to(self.device)
        negative_item_bias = self.item_bias(neg_items).squeeze().to(self.device)

        # 从 v_feat 中获取正样本和负样本的视觉特征
        positive_item_features = self.v_feat(pos_items).to(self.device) # (batchsize, feature_dim)
        negative_item_features = self.v_feat(neg_items).to(self.device)  # (batchsize, feature_dim)

        positive_E_features = self.E(positive_item_features).to(self.device)  # (batchsize, 256)
        negative_E_features = self.E(negative_item_features).to(self.device)  # (batchsize, 256)

        positive_scores = ( positive_item_bias +
                           torch.sum(user_embedding * positive_item_embedding, dim=1) +  # 逐元素求和
                           torch.sum(user_visual_embedding * positive_E_features, dim=1) +  # 逐元素求和
                           torch.matmul(positive_item_features , self.visual_bias.weight))

        negative_scores = (negative_item_bias +
                           torch.sum(user_embedding * negative_item_embedding, dim=1) +
                           torch.sum(user_visual_embedding * negative_E_features, dim=1) +  # 修正这里
                           torch.matmul(negative_item_features , self.visual_bias.weight))

        return positive_scores, negative_scores

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
        # 预测用户对正样本物品和负样本物品的评分得分
        positive_scores, negative_scores = self.forward(users, pos_items, neg_items)
        # 计算BPR Loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores) + 1e-5))

        # 计算BPR嵌入向量的L2正则化损失
        reg_loss = (self.user_embedding(users) ** 2).mean() + \
                   (self.item_embedding(pos_items) ** 2).mean() + \
                   (self.item_embedding(neg_items) ** 2).mean()

        # 添加VBPR参数到正则化项中
        reg_loss += (self.user_visual_embedding(users) ** 2).mean() + \
                    (self.E.weight ** 2).mean() + \
                    (self.visual_bias.weight ** 2).mean()

        # 应用正则化权重
        reg_loss = reg_loss * self.weight_decay

        # 总体损失为BPR Loss和L2正则化损失的加权和
        total_loss = bpr_loss + reg_loss

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.user_embedding.weight
        item_tensor = self.item_embedding.weight

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

