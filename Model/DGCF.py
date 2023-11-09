"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/18 17:35
@File : DGCF.py
@function :
"""
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, dropout_adj, degree
import torch.nn as nn
import torch.nn.functional as F

from metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, map_at_k
from utils import distance_correlation


class DGCFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(DGCFConv, self).__init__(aggr=aggr)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_index_intents):
        edge_index_intents = torch.cat((edge_index_intents, edge_index_intents), dim=0)

        edge_index = edge_index.long()

        # 计算归一化权重  等式10
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_index_intents

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # 等式9
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class DGCF(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, reg_weight, corDecay,
                 n_factors, n_iterations, n_layers, dim_E, device, aggr_mode):
        super(DGCF, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight  # 正则化权重
        self.corDecay = corDecay  # 距离相关性权重
        self.dim_E = dim_E  # 嵌入维度
        self.n_factors = n_factors  # 解耦因素数k
        self.n_iterations = n_iterations  # 邻域路由迭代数
        self.n_layers = n_layers  # GCN层数
        self.device = device
        self.aggr_mode = aggr_mode

        # 确保嵌入维度是因素数的倍数
        assert self.dim_E % self.n_factors == 0

        # 转置
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        # 初始化得分矩阵 S 等式6 反映k个意图的分数
        self.S = torch.ones((self.n_factors, self.edge_index_clone.shape[1]), dtype=torch.float32).to(self.device)

        # 初始化用户和项目的意图感知嵌入
        # 自定义嵌入和参数
        self.user_intent_embeddings = nn.Embedding(num_user, dim_E)
        self.item_intent_embeddings = nn.Embedding(num_item, dim_E)
        nn.init.xavier_uniform_(self.user_intent_embeddings.weight)
        nn.init.xavier_uniform_(self.item_intent_embeddings.weight)

        # 定义图卷积层
        self.conv_embed = DGCFConv(self.dim_E, self.dim_E, aggr=self.aggr_mode)

    def forward(self):
        # 获取初始用户和项目的意图感知嵌入
        users_embs = self.user_intent_embeddings.weight
        items_embs = self.item_intent_embeddings.weight

        # (num_user + num_item, dim_E)
        ego_embs = torch.cat((users_embs, items_embs), dim=0)
        # 所有层的嵌入
        all_embs = [ego_embs]

        for l in range(self.n_layers):
            # n_factors个维度是(num_user + num_item, dim_E // n_factors)的向量
            ego_layer_embs = torch.split(ego_embs, self.dim_E // self.n_factors, 1)
            # 保存每一层的嵌入
            layer_embeddings = []
            for t in range(self.n_iterations):
                # 迭代的嵌入
                iter_embeddings = []
                # 每次迭代更新的意图分数
                s_iter_value = []

                # 意图分数归一化 等式8
                self.S = torch.softmax(self.S, dim=0)

                for k in range(self.n_factors):
                    # 获取第k个意图的嵌入 (num_user + num_item, dim_E // n_factors)
                    x_k = ego_layer_embs[k]

                    # 获取第k个意图的得分
                    s_k = self.S[k]

                    # 通过图卷积层更新嵌入
                    x_k = self.conv_embed(x_k, self.edge_index, s_k)
                    iter_embeddings.append(x_k)
                    # 最后一次迭代
                    if t == self.n_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # 等式11
                    u_embedding_layer = nn.Embedding.from_pretrained(x_k)
                    user_k_embs = u_embedding_layer(self.edge_index_clone[0])
                    i_embedding_layer = nn.Embedding.from_pretrained(ego_layer_embs[k])
                    item_k_embs = i_embedding_layer(self.edge_index_clone[1])

                    user_k_embs = F.normalize(user_k_embs, dim=1)
                    item_k_embs = F.normalize(item_k_embs, dim=1)

                    # (len(self.edge_index_clone[0]), 1)
                    s_k_value = torch.sum(torch.mul(user_k_embs, torch.tanh(item_k_embs)), dim=1)
                    s_iter_value.append(s_k_value)

                s_iter_value = torch.stack(s_iter_value, 0)

                self.S += s_iter_value

            ego_embs = torch.concat(layer_embeddings, 1)
            all_embs.append(ego_embs)

        all_embs = torch.sum(torch.stack(all_embs, dim=0), dim=0)
        self.result = all_embs
        # 从 all_embs 中分割用户和项目的嵌入
        final_user_embs, final_item_embs = torch.split(all_embs, [self.num_user, self.num_item], dim=0)

        return final_user_embs, final_item_embs, self.S

    def bpr_loss(self, users, pos_items, neg_items, final_user_embs, final_item_embs):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = final_user_embs[users]
        pos_item_embeddings = final_item_embs[pos_items]
        neg_item_embeddings = final_item_embs[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items, final_user_embs, final_item_embs):
        # 计算正则化损失
        user_embeddings = final_user_embs[users]
        pos_item_embeddings = final_item_embs[pos_items]
        neg_item_embeddings = final_item_embs[neg_items]

        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def cor_loss(self, user, item):
        user_tensor = self.result[user]
        item_tensor = self.result[self.num_user + item]

        ui_embeddings = torch.cat([user_tensor, item_tensor], dim=0)
        ui_factor_embeddings = torch.split(ui_embeddings, self.dim_E // self.n_factors, 1)

        cor_loss = torch.zeros(1).to(self.device)

        for k in range(self.n_factors - 1):
            x = ui_factor_embeddings[k]
            y = ui_factor_embeddings[k+1]
            cor_loss += distance_correlation(x, y, self.device)

        cor_loss /= ((self.n_factors + 1) * self.n_factors / 2)

        return cor_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        final_user_embs, final_item_embs, self.S = self.forward()

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, final_user_embs, final_item_embs)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, final_user_embs, final_item_embs)
        # 计算 Cor损失
        cor_loss = self.corDecay * self.cor_loss(users, pos_items)
        total_loss = bpr_loss + reg_loss + cor_loss

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

