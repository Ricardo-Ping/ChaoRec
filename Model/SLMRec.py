"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/30 20:01
@File : SLMRec.py
@function :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SLMRec(torch.nn.Module):
    def __init__(self, v_feat, t_feat, edge_index, num_user, num_item, n_layers, user_item_dict, dim_E, device):
        super(SLMRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.n_layers = n_layers
        self.ssl_temp = 0.1  # [0.1, 0.2, 0.5, 1.0]
        self.temp = 0.2
        self.ssl_alpha = 0.01  # [0.01, 0.05, 0.1, 0.5, 1.0]
        self.num_nodes = num_user + num_item
        self.ssl_task = "FAC"
        self.infonce_criterion = nn.CrossEntropyLoss()
        self.device = device
        self.user_item_dict = user_item_dict

        self.v_feat = v_feat
        self.t_feat = t_feat

        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        mul_modal_cnt = 0
        if self.v_feat is not None:
            self.v_feat = F.normalize(self.v_feat, dim=1)
            self.v_dense = nn.Linear(self.v_feat.shape[1], self.dim_E)
            nn.init.xavier_uniform_(self.v_dense.weight)
            mul_modal_cnt += 1
        if self.t_feat is not None:
            self.t_feat = F.normalize(self.t_feat, dim=1)
            self.t_dense = nn.Linear(self.t_feat.shape[1], self.dim_E)
            nn.init.xavier_uniform_(self.t_dense.weight)
            mul_modal_cnt += 1

        self.item_feat_dim = self.dim_E * (mul_modal_cnt + 1)

        self.embedding_item_after_GCN = nn.Linear(self.item_feat_dim, self.dim_E)
        self.embedding_user_after_GCN = nn.Linear(self.item_feat_dim, self.dim_E)
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)

        # 转置并设置为无向图
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        edge_index_norm, normed_weight = self.normalize_edge_index(self.edge_index)
        self.norm_adj = self.edge_index_to_sparse_tensor(edge_index_norm, normed_weight, self.num_nodes)

        if self.ssl_task == "FAC":
            self.g_i_iv = nn.Linear(self.dim_E, self.dim_E)
            self.g_v_iv = nn.Linear(self.dim_E, self.dim_E)
            self.g_iv_iva = nn.Linear(self.dim_E, self.dim_E)
            self.g_a_iva = nn.Linear(self.dim_E, self.dim_E)
            self.g_iva_ivat = nn.Linear(self.dim_E, self.dim_E // 2)
            self.g_t_ivat = nn.Linear(self.dim_E, self.dim_E // 2)
            nn.init.xavier_uniform_(self.g_i_iv.weight)
            nn.init.xavier_uniform_(self.g_v_iv.weight)
            nn.init.xavier_uniform_(self.g_iv_iva.weight)
            nn.init.xavier_uniform_(self.g_a_iva.weight)
            nn.init.xavier_uniform_(self.g_iva_ivat.weight)
            nn.init.xavier_uniform_(self.g_t_ivat.weight)

    def normalize_edge_index(self, edge_index):
        edge_index = edge_index.long()
        # 计算每个节点的度
        row, col = edge_index
        deg = torch.bincount(torch.cat([row, col]))

        # 计算归一化值
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, norm

    def edge_index_to_sparse_tensor(self, edge_index, edge_weight, num_nodes):
        # 创建稀疏张量
        indices = edge_index.to(torch.int64).cpu()
        values = edge_weight.cpu()
        shape = (num_nodes, num_nodes)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).to(edge_index.device)

        return sparse_tensor

    def compute_graph(self, u_emb, i_emb):
        all_emb = torch.cat([u_emb, i_emb])
        embs = [all_emb]
        g_droped = self.norm_adj
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out

    def forward(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight

        if self.v_feat is not None:
            self.v_dense_emb = self.v_dense(self.v_feat)  # v=>id
        if self.t_feat is not None:
            self.t_dense_emb = self.t_dense(self.t_feat)  # t=>id

        self.i_emb = self.compute_graph(users_emb, items_emb)
        self.i_emb_u, self.i_emb_i = torch.split(self.i_emb, [self.num_user, self.num_item])
        if self.v_feat is not None:
            self.v_emb = self.compute_graph(users_emb, self.v_dense_emb)
            self.v_emb_u, self.v_emb_i = torch.split(self.v_emb, [self.num_user, self.num_item])
        if self.t_feat is not None:
            self.t_emb = self.compute_graph(users_emb, self.t_dense_emb)
            self.t_emb_u, self.t_emb_i = torch.split(self.t_emb, [self.num_user, self.num_item])

        # multi - modal features fusion
        # 这一步和别人不一样
        user = self.embedding_user_after_GCN(torch.cat([self.i_emb_u, self.v_emb_u, self.t_emb_u], dim=1))
        item = self.embedding_item_after_GCN(torch.cat([self.i_emb_i, self.v_emb_i, self.t_emb_i], dim=1))
        self.result = torch.cat((user, item), dim=0)

        return user, item

    def fac(self, idx):
        x_i_iv = self.g_i_iv(self.i_emb_i[idx])
        x_v_iv = self.g_v_iv(self.v_emb_i[idx])
        v_logits = torch.mm(x_i_iv, x_v_iv.T)

        v_logits /= self.ssl_temp
        v_labels = torch.tensor(list(range(x_i_iv.shape[0]))).to(self.device)
        v_loss = self.infonce_criterion(v_logits, v_labels)

        x_iv_iva = self.g_iv_iva(x_i_iv)

        x_iva_ivat = self.g_iva_ivat(x_iv_iva)
        x_t_ivat = self.g_t_ivat(self.t_emb_i[idx])

        t_logits = torch.mm(x_iva_ivat, x_t_ivat.T)
        t_logits /= self.ssl_temp
        t_labels = torch.tensor(list(range(x_iva_ivat.shape[0]))).to(self.device)
        t_loss = self.infonce_criterion(t_logits, t_labels)

        return v_loss + t_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        user_tensor, item_tensor = self.forward()
        # =============main_loss====================
        users_emb = user_tensor[users]
        pos_emb = item_tensor[pos_items]
        users_emb = torch.nn.functional.normalize(users_emb, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, dim=1)
        logits = torch.mm(users_emb, pos_emb.T)
        logits /= self.temp
        labels = torch.tensor(list(range(users_emb.shape[0]))).to(self.device)
        main_loss = self.infonce_criterion(logits, labels)
        # ==============ssl_loss=======================
        ssl_loss = self.ssl_alpha * self.fac(pos_items)

        total_loss = main_loss + ssl_loss

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