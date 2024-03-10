"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/10 8:51
@File : POWERec.py
@function :
"""
import random
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F


class LayerGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, user_fea, item_fea, emb_size, prompt_embedding):
        super(LayerGCN, self).__init__()
        self.n_layers = 4
        self.num_user = num_user
        self.num_item = num_item
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.emb_size = emb_size
        self.prompt_embedding = prompt_embedding  # 提示嵌入
        # 用MLP将物品特征映射到嵌入空间
        self.mlp = nn.Sequential(nn.Linear(self.item_fea.shape[1], self.emb_size), nn.Tanh())

    def forward(self, adj):
        # 对所有提示嵌入求和，得到一个集合的提示嵌入向量
        prompt_embd = torch.sum(self.prompt_embedding, 0)  # [emb]
        # 将提示嵌入添加到每个用户嵌入中
        user_embd = self.user_fea + prompt_embd[None, :]  # [user, emb_size]
        # 使用MLP处理物品特征，得到物品嵌入
        item_embd = self.mlp(self.item_fea)  # [item_num, emb_size]

        # 合并用户和物品嵌入
        ego_embeddings = torch.cat((user_embd, item_embd), dim=0)
        all_embeddings = ego_embeddings
        embeddings_layers = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            # 计算新嵌入与初始嵌入的相似度，用于调整嵌入权重
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            # 对于 all_embeddings 中的每一行，它都会被对应的 _weights 中的权重所乘
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        u_embd, i_embd = torch.split(ui_all_embeddings, [self.num_user, self.num_item])

        return u_embd, i_embd


class POWERec(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight,
                 n_layers, prompt_num, neg_weight, dropout, device):
        super(POWERec, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.reg_weight = reg_weight
        self.n_nodes = self.num_user + self.num_item
        self.prompt_num = prompt_num
        self.neg_weight = neg_weight

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        self.num_modal = 3  # 模态数量

        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_user, self.dim_E)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_item, self.dim_E)))

        # 初始化归一化的邻接矩阵
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None  # 用于dropout后的邻接矩阵
        self.forward_adj = None
        self.pruning_random = False

        # 定义提示嵌入
        self.id_prompt = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.prompt_num, self.dim_E)))
        self.v_prompt = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.prompt_num, self.dim_E)))
        self.t_prompt = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.prompt_num, self.dim_E)))

        # 初始化三个模态的GCN模型
        self.id_model = LayerGCN(self.num_user, self.num_item, self.user_embeddings, self.item_embeddings,
                                 self.dim_E, self.id_prompt)
        self.v_model = LayerGCN(self.num_user, self.num_item, self.user_embeddings, v_feat, self.dim_E,
                                self.v_prompt)
        self.t_model = LayerGCN(self.num_user, self.num_item, self.user_embeddings, t_feat, self.dim_E,
                                self.t_prompt)

        # 获取边缘信息
        self.edge_indices, self.edge_values = self.get_edge_info()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.num_user + self.num_item,
                           self.num_user + self.num_item), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        # i = torch.LongTensor([row, col])
        indices = np.vstack((row, col))  # np.vstack堆叠数组
        i = torch.tensor(indices, dtype=torch.long)  # 直接转换为long类型的Tensor
        data = torch.FloatTensor(L.data)

        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.num_user, self.num_item)))
        return edges, values

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj_matrix  # 如果dropout率为0，则不进行dropout
            return
        # 计算保留边的数量
        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        if self.pruning_random:
            # 随机剪枝
            keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len)).to(self.device)
        else:
            # 基于权重的剪枝
            keep_idx = torch.multinomial(self.edge_values, keep_len)
        self.pruning_random = not self.pruning_random  # 切换剪枝模式
        keep_indices = self.edge_indices[:, keep_idx]
        # 归一化保留的边
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.num_user, self.num_item)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.num_user  # 更新物品节点的索引
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        # 构建新的稀疏邻接矩阵
        self.masked_adj = torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj_matrix.shape).to(self.device)

    def forward(self, adj):
        user_id, item_id = self.id_model(adj)
        user_v, item_v = self.v_model(adj)
        user_t, item_t = self.t_model(adj)

        u_embeddings = torch.cat([user_id, user_v, user_t], 1)
        i_embeddings = torch.cat([item_id, item_v, item_t], 1)

        return u_embeddings, i_embeddings

    def bpr_loss(self, users, pos_items, neg_items, u_g, i_g):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        # 弱模态特征处理
        weak_modality, modality_indicator = self.find_weak_modality(user_embeddings, pos_item_embeddings, neg_item_embeddings)
        # 根据弱模态特征调整负样本
        fake_neg_pos_e = (1 - weak_modality) * pos_item_embeddings
        fake_neg_neg_e = weak_modality * neg_item_embeddings
        fake_neg_e = fake_neg_pos_e + fake_neg_neg_e  # [bzs, num_model * dim]
        fake_neg_scores = torch.mul(user_embeddings, fake_neg_e).sum(1)
        weak_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - fake_neg_scores) + 1e-5))
        loss = bpr_loss + self.neg_weight * weak_loss

        return loss

    def find_weak_modality(self, user_e, pos_e, neg_e):
        # user_e = F.normalize(user_e, p=2, dim=1)
        # pos_e = F.normalize(pos_e, p=2, dim=1)
        # neg_e = F.normalize(neg_e, p=2, dim=1)
        # 计算用户与正负样本的得分，重新分配至各模态
        pos_score_ = torch.mul(user_e, pos_e).view(-1, self.num_modal, self.dim_E).sum(dim=-1)
        neg_score_ = torch.mul(user_e, neg_e).view(-1, self.num_modal, self.dim_E).sum(dim=-1)
        # 使用softmax计算模态指标，标识每种模态的重要性
        modality_indicator = (pos_score_ - neg_score_).softmax(-1).detach()

        # 识别并标记最弱的模态
        weak_modality = (modality_indicator == modality_indicator.min(dim=-1, keepdim=True)[0]).to(dtype=torch.float32)
        # 将弱模态标记扩展至整个嵌入维度
        weak_modality = torch.tile(weak_modality.view(-1, self.num_modal, 1), [1, 1, self.dim_E])
        weak_modality = weak_modality.view(-1, self.num_modal * self.dim_E)

        return weak_modality, modality_indicator

    def regularization_loss(self, users, pos_items, neg_items, u_g, i_g):
        # 计算正则化损失
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        user_all_embeddings, item_all_embeddings = self.forward(self.masked_adj)

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, user_all_embeddings, item_all_embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, user_all_embeddings, item_all_embeddings)

        total_loss = bpr_loss + reg_loss
        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        restore_user_e, restore_item_e = self.forward(self.norm_adj_matrix)
        user_tensor = restore_user_e.cpu()
        item_tensor = restore_item_e.cpu()

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
