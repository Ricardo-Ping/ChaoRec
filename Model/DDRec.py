"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/11/14 19:01
@File : MMGCL.py
@function :
"""
import torch
from torch import nn
from BasicGCN import GCNConv
import torch.nn.functional as F
import numpy as np


class DDRec(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, feat_E,
                 reg_weight, n_layers, mm_layers, mm_image_weight, threshold, aggr_mode, device):
        super(DDRec, self).__init__()
        self.final_i_g_embeddings = None
        self.i_g_embeddings = None
        self.t_embedding = None
        self.v_embedding = None
        self.result = None
        self.text_result = None
        self.visual_result = None
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.feat_E = feat_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.aggr_mode = aggr_mode
        self.device = device
        self.mm_layers = mm_layers
        self.knn_k = 10
        self.mm_image_weight = mm_image_weight
        self.threshold = threshold

        # 转置并设置为无向图
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        # 初始化user_id
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_embedding.weight)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # 读入多模态特征
        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=True)

        self.image_trs = nn.Linear(v_feat.shape[1], self.feat_E)
        nn.init.xavier_normal_(self.image_trs.weight)
        self.text_trs = nn.Linear(t_feat.shape[1], self.feat_E)
        nn.init.xavier_normal_(self.text_trs.weight)

        # 行为引导多模态
        self.guide_image_trs = nn.Sequential(
            nn.Linear(self.feat_E, self.feat_E),
            nn.Sigmoid()
        )
        self.guide_text_trs = nn.Sequential(
            nn.Linear(self.feat_E, self.feat_E),
            nn.Sigmoid()
        )

        # 定义图卷积层
        self.conv_layers = nn.ModuleList([GCNConv(self.dim_E, self.dim_E, aggr=self.aggr_mode)
                                          for _ in range(n_layers)])
        self.conv_layers_m = nn.ModuleList([GCNConv(self.dim_E, self.dim_E, aggr=self.aggr_mode)
                                            for _ in range(mm_layers)])

        # --------------冻结项目图-------------------
        indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        self.image_adj = image_adj
        self.text_adj = text_adj

    def get_knn_adj_mat(self, mm_embeddings):
        # mm_embeddings: 项目的多模态嵌入
        # 计算l2范数
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        # 计算相似性矩阵
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        # 找到k个最近邻 knn_ind:(num_item, 10)
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # 创建项目的索引
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)  # 扩展  (num_item, 10)
        # 相当于创建了knn_k之后的edges_index (2, num_item * 10)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        # 权重全1
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        # 计算邻接矩阵的每一行的和
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        # 计算归一化值
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def forward(self):
        self.v_embedding = self.image_trs(self.image_embedding.weight)
        self.t_embedding = self.text_trs(self.text_embedding.weight)
        visual_tensor = self.v_embedding
        text_tensor = self.t_embedding

        # visual_tensor_cpu = visual_tensor.detach().cpu().numpy()
        # text_tensor_cpu = text_tensor.detach().cpu().numpy()
        # np.save('noise_visual.npy', visual_tensor_cpu)
        # np.save('noise_text.npy', text_tensor_cpu)

        if self.final_i_g_embeddings is not None:
            item_embedding = self.final_i_g_embeddings.detach()
            visual_tensor = item_embedding * self.guide_image_trs(self.v_embedding)
            text_tensor = item_embedding * self.guide_text_trs(self.t_embedding)

        # =============项目-项目图========================
        # 对项目-项目图更新项目表示
        h = self.item_embedding.weight
        for i in range(self.mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        h = F.normalize(h, p=2, dim=1)

        h1 = self.v_embedding
        for i in range(self.mm_layers):
            h1 = torch.sparse.mm(self.image_adj, h1)
        h1 = F.normalize(h1, p=2, dim=1)

        h2 = self.t_embedding
        for i in range(self.mm_layers):
            h2 = torch.sparse.mm(self.text_adj, h2)
        h2 = F.normalize(h2, p=2, dim=1)

        # =====================视觉模态卷积去噪===================
        ego_embeddings = torch.cat((self.user_embedding.weight, visual_tensor), dim=0)
        all_embeddings_d = [ego_embeddings]
        for conv in self.conv_layers:
            # 计算余弦相似度
            user_embeddings = ego_embeddings[:self.num_user]
            item_embeddings = ego_embeddings[self.num_user:]
            # sim_matrix = self.cosine_similarity(user_embeddings, item_embeddings)
            # sim_matrix = 1 / (1 + torch.cdist(user_embeddings, item_embeddings, p=2))
            sim_matrix = torch.mm(user_embeddings, item_embeddings.t())
            # 过滤边
            filtered_edge_index = self.filter_edges(self.edge_index_clone, sim_matrix, threshold=self.threshold)
            filtered_edge_index = torch.cat((filtered_edge_index, filtered_edge_index[[1, 0]]), dim=1)
            ego_embeddings = conv(ego_embeddings, filtered_edge_index)
            # ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings_d += [ego_embeddings]
        all_embeddings_d = torch.stack(all_embeddings_d, dim=1)
        # 均值处理
        all_embeddings_d = all_embeddings_d.mean(dim=1, keepdim=False)
        self.u_v_embeddings, self.i_v_embeddings = torch.split(all_embeddings_d, [self.num_user, self.num_item], dim=0)
        self.i_v_embeddings = self.i_v_embeddings + h1

        # =====================文本模态卷积去噪===================
        ego_embeddings = torch.cat((self.user_embedding.weight, text_tensor), dim=0)
        all_embeddings_d = [ego_embeddings]
        for conv in self.conv_layers:
            # # 计算余弦相似度
            user_embeddings = ego_embeddings[:self.num_user]
            item_embeddings = ego_embeddings[self.num_user:]
            # sim_matrix = self.cosine_similarity(user_embeddings, item_embeddings)
            # sim_matrix = 1 / (1 + torch.cdist(user_embeddings, item_embeddings, p=2))
            sim_matrix = torch.mm(user_embeddings, item_embeddings.t())
            # 过滤边
            filtered_edge_index = self.filter_edges(self.edge_index_clone, sim_matrix, threshold=self.threshold)
            filtered_edge_index = torch.cat((filtered_edge_index, filtered_edge_index[[1, 0]]), dim=1)
            ego_embeddings = conv(ego_embeddings, filtered_edge_index)
            # ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings_d += [ego_embeddings]
        all_embeddings_d = torch.stack(all_embeddings_d, dim=1)
        # 均值处理
        all_embeddings_d = all_embeddings_d.mean(dim=1, keepdim=False)
        self.u_t_embeddings, self.i_t_embeddings = torch.split(all_embeddings_d, [self.num_user, self.num_item], dim=0)
        self.i_t_embeddings = self.i_t_embeddings + h2

        # =====================id模态卷积===================
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        self.u_g_embeddings, self.i_g_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)
        self.final_i_g_embeddings = self.i_g_embeddings + h
        # self.final_i_g_embeddings = self.i_g_embeddings

        # 最终的用户和项目嵌入
        self.i = torch.cat((self.final_i_g_embeddings, self.i_v_embeddings, self.i_t_embeddings), dim=1)
        self.u = torch.cat((self.u_g_embeddings, self.u_v_embeddings, self.u_t_embeddings), dim=1)

        self.result = torch.cat((self.u, self.i), dim=0)

        return self.result

    def cosine_similarity(self, user_embeddings, item_embeddings):
        # 计算余弦相似度
        user_embeddings_norm = F.normalize(user_embeddings, p=2, dim=1)
        item_embeddings_norm = F.normalize(item_embeddings, p=2, dim=1)
        return torch.mm(user_embeddings_norm, item_embeddings_norm.t())

    def filter_edges(self, edge_index, sim_matrix, threshold):
        # 获得用户和项目的索引
        user_indices, item_indices = edge_index[0].long(), (edge_index[1] - self.num_user).long()
        # 创建掩码
        mask = sim_matrix[user_indices, item_indices] >= threshold
        # 应用掩码
        filtered_edges = edge_index[:, mask]

        return filtered_edges

    def bpr_loss(self, users, pos_items, neg_items, embedding):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = embedding[users]
        pos_item_embeddings = embedding[self.num_user + pos_items]
        neg_item_embeddings = embedding[self.num_user + neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return bpr_loss

    def adaptive_bpr_loss(self, users, pos_items, neg_items, embedding, xi=0.99):
        # 获取用户、正样本和负样本的嵌入
        user_embeddings = embedding[users]
        pos_item_embeddings = embedding[self.num_user + pos_items]
        neg_item_embeddings = embedding[self.num_user + neg_items]

        # 计算正样本和负样本的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算自适应参数 delta
        delta = 1 - torch.log(1 - torch.min(torch.sigmoid(neg_scores), torch.tensor(xi)))

        # 计算自适应 BPR 损失
        adaptive_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - delta * neg_scores) + 1e-5))

        return adaptive_loss

    def regularization_loss(self, users, pos_items, neg_items, embedding):
        # 计算正则化损失
        user_embeddings = embedding[users]
        pos_item_embeddings = embedding[self.num_user + pos_items]
        neg_item_embeddings = embedding[self.num_user + neg_items]

        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embedding = self.forward()

        # bpr损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, embedding)

        # 正则化损失
        reg_loss = self.regularization_loss(users, pos_items, neg_items, embedding)

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
