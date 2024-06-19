"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/6/18 11:18
@File : MCLN.py
@function :
"""
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F


class MCLN(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight, n_layers,
                 n_mca, device):
        super(MCLN, self).__init__()
        self.diag = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.device = device

        self.n_mca = n_mca  # 反事实层数

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # 读入多模态特征
        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=True)

        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        nn.init.xavier_normal_(self.image_trs.weight)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)
        nn.init.xavier_normal_(self.text_trs.weight)

        # 初始化user_id
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_embedding.weight)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_normal_(self.item_embedding.weight)

        self.user_embedding_v = nn.Embedding(self.num_user, self.dim_E)
        self.user_embedding_t = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_embedding_v.weight)
        nn.init.xavier_normal_(self.user_embedding_t.weight)

        self.fc_pos = nn.Linear(3 * self.dim_E, self.dim_E)
        self.fc_neg = nn.Linear(3 * self.dim_E, self.dim_E)
        self.relu = nn.ReLU()

        # 反事实层1
        self.V1 = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.K1 = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.Q1 = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.K_int = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.Q_int = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.cfl1 = nn.Linear(3 * self.dim_E, 3 * self.dim_E, bias=False).to(device)
        self.ln1 = nn.LayerNorm(3 * self.dim_E).to(device)
        # 反事实层2
        self.V2 = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.K2 = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.Q2 = nn.Linear(3 * dim_E, 3 * dim_E, bias=False).to(device)
        self.cfl2 = nn.Linear(3 * self.dim_E, 3 * self.dim_E, bias=False).to(device)
        self.ln2 = nn.LayerNorm(3 * self.dim_E).to(device)
        # Feed forward layers
        self.inner_layer = nn.Linear(3 * dim_E, 12 * dim_E).to(device)
        self.output_layer = nn.Linear(12 * dim_E, 3 * dim_E).to(device)
        self.layer_norm = nn.LayerNorm(3 * dim_E).to(device)

    def get_norm_adj_mat(self):
        # 创建一个空的稀疏矩阵A，大小为(n_users+n_items) x (n_users+n_items)，数据类型为float32
        A = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        # 加载交互矩阵
        inter_M = self.interaction_matrix
        # 将交互矩阵转置，以便获取物品到用户的关系
        inter_M_t = self.interaction_matrix.transpose()
        # 将用户到物品的交互和物品到用户的交互合并到邻接矩阵A中
        # 注意：物品的索引需要加上用户的数量，因为矩阵的前半部分是用户，后半部分是物品
        # nnz 属性表示稀疏矩阵中的非零元素的数量
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        # 更新稀疏矩阵A的数据
        A._update(data_dict)

        # 归一化邻接矩阵
        # 计算A中每个节点的度（即每行的非零元素个数）
        sumArr = (A > 0).sum(axis=1)
        # 为了防止除以0，给度数加上一个很小的数epsilon
        diag = np.array(sumArr.flatten())[0] + 1e-7
        # 度数的-0.5次幂，用于拉普拉斯矩阵的归一化
        diag = np.power(diag, -0.5)
        # 将numpy数组转换为torch张量，并移动到模型的设备上（CPU或GPU）
        self.diag = torch.from_numpy(diag).to(self.device)
        # 创建对角矩阵D
        D = sp.diags(diag)
        # 使用D对A进行归一化：L = D^-0.5 * A * D^-0.5
        L = D @ A @ D
        # 将归一化后的L转换为COO格式的稀疏矩阵，以便后续转换为torch稀疏张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        # 创建torch张量来表示稀疏矩阵的坐标(indices)和值(values)
        rows_and_cols = np.array([row, col])  # 将行和列的列表转换成numpy数组
        i = torch.tensor(rows_and_cols, dtype=torch.long)  # 从numpy数组创建张量
        data = torch.FloatTensor(L.data)
        # 创建torch的稀疏张量来表示归一化的邻接矩阵
        # SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        return SparseL

    def _create_norm_embed(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            # 使用 self.norm_adj_mat 进行图卷积操作
            side_embeddings = torch.sparse.mm(self.norm_adj_mat, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)
        return u_g_embeddings, i_g_embeddings

    def causal_difference_1(self, cd_inputs_embedding, cd_inputs_embedding_int):
        cd_outputs = cd_inputs_embedding
        cd_outputs_int = cd_inputs_embedding_int

        for i in range(self.n_mca):
            cl_outputs = self.counterfactual_learning_layer_1(
                query=cd_outputs,
                key_value=cd_outputs,
                query_int=cd_outputs_int,
                key_value_int=cd_outputs_int,
            )
            cd_outputs = self.feed_forward_layer(
                cl_outputs,
                activation=F.relu,
            )
        return cd_outputs

    def causal_difference_2(self, cd_inputs_embedding):
        cd_outputs = cd_inputs_embedding

        for i in range(self.n_mca):
            cl_outputs = self.counterfactual_learning_layer_2(
                query=cd_outputs,
                key_value=cd_outputs,
            )
            cd_outputs = self.feed_forward_layer(
                cl_outputs,
                activation=F.relu,
            )
        return cd_outputs

    def counterfactual_learning_layer_1(self, query, key_value, query_int, key_value_int):
        V_k = self.V1(key_value)
        K_k = self.K1(key_value)
        Q_q = self.Q1(query)
        K_int_k = self.K_int(key_value_int)
        Q_int_q = self.Q_int(query_int)

        score = torch.matmul(Q_q, K_k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(3 * self.dim_E, dtype=torch.float32))
        score_int = torch.matmul(Q_int_q, K_int_k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(3 * self.dim_E, dtype=torch.float32))

        score -= score_int
        softmax = F.softmax(score, dim=-1)
        attention = torch.matmul(softmax, V_k)

        counterfactual_learning = self.cfl1(attention)
        counterfactual_learning += query
        counterfactual_learning = self.ln1(counterfactual_learning)
        return counterfactual_learning

    def counterfactual_learning_layer_2(self, query, key_value):
        V_k = self.V2(key_value)
        K_k = self.K2(key_value)
        Q_q = self.Q2(query)

        score = torch.matmul(Q_q, K_k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(3 * self.dim_E, dtype=torch.float32))
        softmax = F.softmax(score, dim=-1)
        attention = torch.matmul(softmax, V_k)

        counterfactual_learning = self.cfl2(attention)
        counterfactual_learning += query
        counterfactual_learning = self.ln2(counterfactual_learning)
        return counterfactual_learning

    def feed_forward_layer(self, inputs, activation):
        x = activation(self.inner_layer(inputs))
        x = self.output_layer(x)
        x += inputs
        return self.layer_norm(x)

    def forward(self, users, pos_items, neg_items, int_items):
        self.visual = self.image_trs(self.image_embedding.weight)
        self.textual = self.text_trs(self.text_embedding.weight)

        self.ua_embeddings, self.ia_embeddings = self._create_norm_embed()

        self.u_g_embeddings = self.ua_embeddings[users]
        self.pos_i_g_embeddings = self.ia_embeddings[pos_items]
        self.neg_i_g_embeddings = self.ia_embeddings[neg_items]
        self.int_i_g_embeddings = self.ia_embeddings[int_items]

        self.u_g_embeddings_pre = self.user_embedding(users)
        self.pos_i_g_embeddings_pre = self.item_embedding(pos_items)
        self.neg_i_g_embeddings_pre = self.item_embedding(neg_items)
        self.int_i_g_embeddings_pre = self.item_embedding(int_items)

        # 多模态嵌入
        self.u_g_embeddings_v = self.user_embedding_v(users)
        self.u_g_embeddings_t = self.user_embedding_t(users)

        self.pos_i_g_embeddings_v = self.visual[pos_items]
        self.pos_i_g_embeddings_t = self.textual[pos_items]
        self.neg_i_g_embeddings_v = self.visual[neg_items]
        self.neg_i_g_embeddings_t = self.textual[neg_items]
        self.int_i_g_embeddings_v = self.visual[int_items]
        self.int_i_g_embeddings_t = self.textual[int_items]

        # 合并输入
        self.pos_inputs_embeddings = torch.cat(
            [self.pos_i_g_embeddings, self.pos_i_g_embeddings_v, self.pos_i_g_embeddings_t], dim=1)
        self.neg_inputs_embeddings = torch.cat(
            [self.neg_i_g_embeddings, self.neg_i_g_embeddings_v, self.neg_i_g_embeddings_t], dim=1)
        self.int_inputs_embeddings = torch.cat(
            [self.int_i_g_embeddings, self.int_i_g_embeddings_v, self.int_i_g_embeddings_t], dim=1)

        # 因果差异
        self.pos_outputs_embeddings = self.causal_difference_1(self.pos_inputs_embeddings, self.int_inputs_embeddings)
        self.neg_outputs_embeddings = self.causal_difference_2(self.neg_inputs_embeddings)

        # 聚合所有模式的嵌入
        self.pos_i_g_embeddings_m = self.relu(self.fc_pos(self.pos_outputs_embeddings))
        self.neg_i_g_embeddings_m = self.relu(self.fc_neg(self.neg_outputs_embeddings))

        # 计算最终的相似度得分
        # self.multiply = (self.u_g_embeddings * self.pos_i_g_embeddings).sum(dim=1) + \
        #                 (self.u_g_embeddings_v * self.pos_i_g_embeddings_v).sum(dim=1) + \
        #                 (self.u_g_embeddings_t * self.pos_i_g_embeddings_t).sum(dim=1) + \
        #                 (self.u_g_embeddings * self.pos_i_g_embeddings_m).sum(dim=1)
        total_scores = torch.matmul(self.u_g_embeddings, self.pos_i_g_embeddings.t()) + \
                       torch.matmul(self.u_g_embeddings_v, self.pos_i_g_embeddings_v.t()) + \
                       torch.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t.t()) + \
                       torch.matmul(self.u_g_embeddings, self.pos_i_g_embeddings_m.t())

        return total_scores

    def loss(self, users, pos_items, neg_items, int_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        int_items = int_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
        int_items = int_items.to(self.device)

        _ = self.forward(users, pos_items, neg_items, int_items)

        pos_scores = torch.sum(self.u_g_embeddings * self.pos_i_g_embeddings, dim=1)
        neg_scores = torch.sum(self.u_g_embeddings * self.neg_i_g_embeddings, dim=1)

        pos_scores_v = torch.sum(self.u_g_embeddings * self.pos_i_g_embeddings_v, dim=1)
        neg_scores_v = torch.sum(self.u_g_embeddings * self.neg_i_g_embeddings_v, dim=1)

        pos_scores_t = torch.sum(self.u_g_embeddings * self.pos_i_g_embeddings_t, dim=1)
        neg_scores_t = torch.sum(self.u_g_embeddings * self.neg_i_g_embeddings_t, dim=1)

        pos_scores_m = torch.sum(self.u_g_embeddings * self.pos_i_g_embeddings_m, dim=1)
        neg_scores_m = torch.sum(self.u_g_embeddings * self.neg_i_g_embeddings_m, dim=1)

        # L2 Regularization
        regularizer_mf = torch.sum(self.u_g_embeddings_pre.pow(2)) + \
                         torch.sum(self.pos_i_g_embeddings_pre.pow(2)) + \
                         torch.sum(self.neg_i_g_embeddings_pre.pow(2))
        regularizer_mf_v = torch.sum(self.pos_i_g_embeddings_v.pow(2)) + \
                           torch.sum(self.neg_i_g_embeddings_v.pow(2))
        regularizer_mf_t = torch.sum(self.pos_i_g_embeddings_t.pow(2)) + \
                           torch.sum(self.neg_i_g_embeddings_t.pow(2))
        regularizer_mf_m = torch.sum(self.pos_i_g_embeddings_m.pow(2)) + \
                           torch.sum(self.neg_i_g_embeddings_m.pow(2))

        # BPR Loss
        mf_loss = torch.mean(F.softplus(-(pos_scores - neg_scores))) + \
                  torch.mean(F.softplus(-(pos_scores_v - neg_scores_v))) + \
                  torch.mean(F.softplus(-(pos_scores_t - neg_scores_t))) + \
                  torch.mean(F.softplus(-(pos_scores_m - neg_scores_m)))

        # Embedding Loss
        emb_loss = self.reg_weight * (
                regularizer_mf + regularizer_mf_t + regularizer_mf_v + regularizer_mf_m)

        # Total Loss
        loss = mf_loss + emb_loss

        # Normalize Embeddings for stability
        self.user_embed = F.normalize(self.u_g_embeddings_pre, p=2, dim=1)
        self.item_embed = F.normalize(self.pos_i_g_embeddings_pre, p=2, dim=1)
        self.item_embed_v = F.normalize(self.pos_i_g_embeddings_v, p=2, dim=1)
        self.item_embed_t = F.normalize(self.pos_i_g_embeddings_t, p=2, dim=1)

        return loss

    def gene_ranklist(self, topk=50):
        # 用户嵌入和项目嵌入
        user_tensor = self.ua_embeddings[:self.num_user].cpu()
        item_tensor = self.ia_embeddings[:self.num_item].cpu()
        visual_user_tensor = self.user_embedding_v.weight[:self.num_user].cpu()
        textual_user_tensor = self.user_embedding_t.weight[:self.num_user].cpu()
        visual = self.visual[:self.num_item].cpu()
        textual = self.textual[:self.num_item].cpu()

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 生成评分矩阵
        score_matrix = torch.matmul(user_tensor, item_tensor.t()) + torch.matmul(visual_user_tensor, visual.t()) + \
                       torch.matmul(textual_user_tensor, textual.t())

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
