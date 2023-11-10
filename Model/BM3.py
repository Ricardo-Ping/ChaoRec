"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/26 16:25
@File : BM3.py
@function :
"""
import torch
from torch import nn
from BasicGCN import GCNConv
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity


class BM3(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, feat_E,
                 reg_weight, dropout, n_layers, cl_weight, aggr_mode, device):
        super(BM3, self).__init__()
        # 传入的参数
        self.result = None
        self.device = device
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight
        self.dim_E = dim_E  # id嵌入维度
        self.feat_E = feat_E  # 特征嵌入维度
        self.cl_weight = cl_weight  # 对比损失权重
        self.dropout = dropout
        self.n_layers = n_layers

        # 转置并设置为无向图
        self.edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)  # [2, 188381]
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)  # [2, 376762]

        # 自定义嵌入和参数
        self.user_embedding = nn.Embedding(num_user, dim_E)
        self.item_embedding = nn.Embedding(num_item, dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.predictor = nn.Linear(self.dim_E, self.dim_E)

        # 特征处理
        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
        self.image_trs = nn.Linear(v_feat.shape[1], self.feat_E)
        nn.init.xavier_normal_(self.image_trs.weight)
        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)
        self.text_trs = nn.Linear(t_feat.shape[1], self.feat_E)
        nn.init.xavier_normal_(self.text_trs.weight)

        # 定义图卷积层
        self.conv_layers = nn.ModuleList([GCNConv(self.dim_E, self.dim_E, aggr=self.aggr_mode)
                                          for _ in range(n_layers)])

    def forward(self):
        # 等式2-4
        h = self.item_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)
        # 项目进行了残差
        i_g_embeddings = i_g_embeddings + h
        self.result = torch.cat((u_g_embeddings, i_g_embeddings), dim=0)

        return u_g_embeddings, i_g_embeddings

    def loss(self, users, items, _):
        items = items - self.num_user
        # 在线网络
        u_online_ori, i_online_ori = self.forward()

        # 等式1
        t_feat_online = self.text_trs(self.text_embedding.weight)
        v_feat_online = self.image_trs(self.image_embedding.weight)

        # 停止梯度
        with torch.no_grad():
            # 目标网络：通过对在线网络的表示进行克隆并应用 dropout 获得的 等式5
            u_target = F.dropout(u_online_ori, self.dropout)
            i_target = F.dropout(i_online_ori, self.dropout)

            # 多模态特征的目标网络
            t_feat_target = F.dropout(t_feat_online, self.dropout)
            v_feat_target = F.dropout(v_feat_online, self.dropout)

        # 等式6
        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        # 在线
        u_online = u_online[users, :]  # h_hat_u
        i_online = i_online[items, :]  # h_hat_i
        # 目标
        u_target = u_target[users, :]  # h_dian_u
        i_target = i_target[items, :]  # h_dian_i

        t_feat_online = self.predictor(t_feat_online)
        t_feat_online = t_feat_online[items, :]
        t_feat_target = t_feat_target[items, :]

        # 对齐损失 等式10
        loss_t = 1 - cosine_similarity(t_feat_online, i_target, dim=-1).mean()
        # 掩码损失 等式11
        loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target, dim=-1).mean()

        v_feat_online = self.predictor(v_feat_online)
        v_feat_online = v_feat_online[items, :]
        v_feat_target = v_feat_target[items, :]

        # 对齐损失 等式10
        loss_v = 1 - cosine_similarity(v_feat_online, i_target, dim=-1).mean()
        # 掩码损失 等式11
        loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target, dim=-1).mean()

        # 图重建损失 等式7
        loss_ui = 1 - cosine_similarity(u_online, i_target, dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target, dim=-1).mean()

        # 正则化损失
        reg_loss = self.reg_weight * (
                torch.mean(u_online_ori ** 2) + torch.mean(i_online_ori ** 2))

        total_loss = (loss_ui + loss_iu).mean() + reg_loss + self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:self.num_user + self.num_item]

        # 使用了预测器的效果没有那么好
        user_tensor = self.predictor(user_tensor)
        item_tensor = self.predictor(item_tensor)

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
