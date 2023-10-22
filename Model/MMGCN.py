"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/2 22:32
@File : MMGCN.py
@function :
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BasicGCN import BasicGCN
from torch_geometric.utils import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform

from metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, map_at_k
from utils import gpu


class GCN(torch.nn.Module):
    def __init__(self, edge_index, num_user, num_item, dim_feat, dim_id, aggr_mode, concate,
                 has_id, dim_latent=None):
        super(GCN, self).__init__()
        # self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        # self.num_layer = num_layer
        self.has_id = has_id
        self.device = gpu()

        if self.dim_latent:
            # 用户偏好嵌入
            self.preference = nn.init.xavier_normal_(
                torch.rand((self.num_user, self.dim_latent), requires_grad=True)).to(self.device)
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent).to(self.device)

            self.conv_embed_1 = BasicGCN(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.lin.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(
                self.device)
            self.conv_embed_1 = BasicGCN(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.lin.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BasicGCN(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.lin.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BasicGCN(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.lin.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

    def forward(self, features, id_embedding):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x).to(self.device)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))  # equation 1
        u_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))  # equation 5
        x = F.leaky_relu(self.g_layer1(torch.cat((h, u_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + u_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))  # equation 1
        u_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))  # equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, u_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + u_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))  # equation 1
        u_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))  # equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, u_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + u_hat)

        return x


class MMGCN(torch.nn.Module):
    def __init__(self, v_feat, t_feat, edge_index, num_user, num_item, aggr_mode,
                 concate, has_id, user_item_dict, reg_weight, dim_x, device):
        super(MMGCN, self).__init__()
        self.device = device
        # self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.user_item_dict = user_item_dict
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = reg_weight

        # 转置并设置为无向图
        self.edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # self.v_feat = torch.tensor(v_feat, dtype=torch.float).to(self.device)
        self.v_feat = v_feat.clone().detach().requires_grad_(True).to(self.device)
        self.v_gcn = GCN(self.edge_index, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode,
                         self.concate, has_id=has_id, dim_latent=256)

        # self.t_feat = torch.tensor(t_feat, dtype=torch.float).to(self.device)
        self.t_feat = t_feat.clone().detach().requires_grad_(True).to(self.device)
        self.t_gcn = GCN(self.edge_index, num_user, num_item, self.t_feat.size(1), dim_x, self.aggr_mode,
                         self.concate, has_id=has_id)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x), requires_grad=True)).to(
            self.device)
        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))).to(self.device)

    def forward(self):
        # 多模态图卷积
        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        t_rep = self.t_gcn(self.t_feat, self.id_embedding)

        # 最终表征
        representation = (v_rep + t_rep) / 2

        self.result = representation  # (num_user + num_item, 64)
        return representation

    def loss(self, user_tensor, item_tensor):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        # 预测分数
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        # BPR
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        # 正则化
        reg_embedding_loss = (self.id_embedding[user_tensor] ** 2 + self.id_embedding[item_tensor] ** 2).mean() + (
                self.v_gcn.preference ** 2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        return loss + reg_loss

    def gene_ranklist(self, step=200, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.result[:self.num_user].cpu()
        item_tensor = self.result[self.num_user:self.num_user + self.num_item].cpu()

        # 分批处理数据
        start_index = 0
        end_index = self.num_user if step is None else step

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        while self.num_user >= end_index > start_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            # 生成评分矩阵
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            # 将历史交互设置为极小值
            for row, col in self.user_item_dict.items():
                if start_index <= row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))- self.num_user
                    score_matrix[row][col] = 1e-5

            # 选出每个用户的 top-k 个物品
            _, index_of_rank_list_train = torch.topk(score_matrix, topk)
            # 总的top-k列表
            all_index_of_rank_list = torch.cat(
                (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
                dim=0)

            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        # 返回三个推荐列表
        return all_index_of_rank_list

    # def gene_ranklist(self, step=200, topk=50):
    #     # step需要小于用户数量才能达到分批的效果不然会报错
    #     # 用户嵌入和项目嵌入
    #     user_tensor = self.result[:self.num_user].cpu()  # (num_user , 64)
    #     item_tensor = self.result[self.num_user:self.num_user + self.num_item].cpu() # (num_item , 64)
    #
    #     # 不同阶段的评估（例如训练、验证和测试）
    #     all_index_of_rank_list = torch.LongTensor([])
    #
    #     score_matrix = torch.matmul(user_tensor, item_tensor.t())  # (num_user,num_item)
    #
    #     # 将历史交互设置为极小值
    #     for row, col in self.user_item_dict.items():
    #         col = torch.LongTensor(list(col)) - self.num_user
    #         score_matrix[row][col] = 1e-5
    #
    #     # 选出每个用户的 top-k 个物品
    #     _, index_of_rank_list_train = torch.topk(score_matrix, topk)
    #     # 总的top-k列表
    #     all_index_of_rank_list = torch.cat(
    #         (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
    #         dim=0)
    #
    #     # 返回推荐列表
    #     return all_index_of_rank_list

    def gene_metrics(self, val_data, rank_list, k_list):
        # 初始化存储评估指标的字典
        metrics = {k: {'precision': 0, 'recall': 0, 'ndcg': 0, 'hit_rate': 0, 'map': 0} for k in k_list}

        for data in val_data:
            user = data[0]
            pos_items = data[1:]
            ranked_items = rank_list[user].tolist()

            # 对每个 k 值计算评估指标
            for k in k_list:
                metrics[k]['precision'] += precision_at_k(ranked_items, pos_items, k)
                metrics[k]['recall'] += recall_at_k(ranked_items, pos_items, k)
                metrics[k]['ndcg'] += ndcg_at_k(ranked_items, pos_items, k)
                metrics[k]['hit_rate'] += hit_rate_at_k(ranked_items, pos_items, k)
                metrics[k]['map'] += map_at_k(ranked_items, pos_items, k)

        num_users = len(val_data)

        # 计算评估指标的平均值
        for k in k_list:
            metrics[k]['precision'] /= num_users
            metrics[k]['recall'] /= num_users
            metrics[k]['ndcg'] /= num_users
            metrics[k]['hit_rate'] /= num_users
            metrics[k]['map'] /= num_users

        return metrics
