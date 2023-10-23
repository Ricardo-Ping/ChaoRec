"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/8 20:27
@File : BPR.py
@function :
"""
import torch
import torch.nn as nn

from metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, map_at_k


class BPRMF(nn.Module):
    def __init__(self, num_user, num_item, embedding_dim, user_item_dict, device, weight_decay):
        super(BPRMF, self).__init__()
        self.user_item_dict = user_item_dict
        self.num_user = num_user
        self.device = device
        self.item_bias = nn.Embedding(num_item, 1)  # 物品偏置项 b_i
        # 用户隐式向量 U_u
        self.user_embedding = nn.Embedding(num_user, embedding_dim)
        # 物品隐式向量 V_i
        self.item_embedding = nn.Embedding(num_item, embedding_dim)
        self.reg_weight = weight_decay  # 正则化系数

        # 初始化权重
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, users, pos_items, neg_items):
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
        # 嵌入
        user_embedding = self.user_embedding(users).to(self.device)
        positive_item_embedding = self.item_embedding(pos_items).to(self.device)
        negative_item_embedding = self.item_embedding(neg_items).to(self.device)

        # 偏置
        positive_item_bias = self.item_bias(pos_items).squeeze().to(self.device)
        negative_item_bias = self.item_bias(neg_items).squeeze().to(self.device)

        positive_scores = torch.sum(user_embedding * positive_item_embedding,
                                    dim=1) + positive_item_bias
        negative_scores = torch.sum(user_embedding * negative_item_embedding,
                                    dim=1) + negative_item_bias

        return positive_scores, negative_scores

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
        # 预测用户对正样本物品和负样本物品的评分得分
        positive_scores, negative_scores = self.forward(users, pos_items, neg_items)
        # 计算BPR Loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))

        reg_loss = (self.user_embedding(users) ** 2).mean() + (self.item_embedding(pos_items) ** 2).mean() + (self.item_embedding(neg_items)).mean()
        reg_loss = reg_loss * self.reg_weight

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
