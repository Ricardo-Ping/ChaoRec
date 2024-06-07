"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/6/6 16:15
@File : LightGT.py
@function :
"""
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import scipy.sparse as sp


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, nheads=1, dropout=0.):
        super(MultiheadAttention, self).__init__()
        # 嵌入维度、头数、dropout率
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.dropout = dropout

        self.head_dim = embed_dim // 1  # 计算每个头的维度
        assert self.head_dim * 1 == embed_dim  # 确保维度匹配

        # 线性变换层，用于对Q、K、V进行投影
        self.q_in_proj = nn.Linear(embed_dim, embed_dim)
        self.k_in_proj = nn.Linear(embed_dim, embed_dim)
        self.v_in_proj = nn.Linear(embed_dim, embed_dim)

        # 输出的线性变换层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # 序列长度、批量大小和嵌入维度
        tgt_len, batch_size, embed_dim = query.size()
        nheads = self.nheads
        assert embed_dim == self.embed_dim
        head_dim = embed_dim // nheads
        assert head_dim * nheads == embed_dim

        scaling = float(head_dim) ** -0.5  # 缩放因子，避免计算出现数值问题

        # 分别对查询（Q）、键（K）和值（V）进行线性变换
        q = self.q_in_proj(query)
        k = self.k_in_proj(key)
        v = self.v_in_proj(value)

        # 将查询向量进行缩放
        q = (q * scaling) / 100

        # 从原始的(tgt_len, batch_size, embed_dim)变为(tgt_len, batch_size * nheads, head_dim)
        q = q.contiguous().view(tgt_len, batch_size * nheads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * nheads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * nheads, head_dim).transpose(0, 1)

        # 计算源序列长度
        src_len = k.size(1)

        # 执行批量矩阵乘法，计算查询和键的点积，然后转置以获得注意力得分
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [batch_size * nheads, tgt_len, src_len]

        attn_output_weights = attn_output_weights.view(batch_size, nheads, tgt_len, src_len)
        # 使用键填充掩码更新注意力得分，将被掩码的位置设为负无穷
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        # 重新将注意力得分展平为三维张量，准备进行softmax操作
        attn_output_weights = attn_output_weights.view(batch_size * nheads, tgt_len, src_len)
        # 应用softmax函数，获得最终的注意力权重
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        # 应用dropout，防止过拟合
        attn_output_weights = torch.dropout(attn_output_weights, p=self.dropout, train=self.training)

        # 使用注意力权重对值向量进行加权和
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [batch_size * nheads, tgt_len, head_dim]

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.nhead = nhead
        # 创建多头注意力模块列表
        self.self_attn = nn.ModuleList([MultiheadAttention(d_model, dropout=dropout) for _ in range(self.nhead)])

        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        # self.activation = torch.relu

    def forward(self, query, key, value, src_mask=None, src_key_padding_mask=None):
        # 根据头数执行不同的注意力处理
        if self.nhead != 1:
            attn_output = []
            for mod in self.self_attn:
                attn_output.append(mod(query, key, value,
                                       attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask))
            src2 = torch.sum(torch.stack(attn_output, dim=-1), dim=-1)
        else:
            src2 = self.self_attn[0](query, key, value,
                                     attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)

        # src = value + self.dropout1(src2)
        # 应用归一化
        src = self.norm1(src2)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        # 复制多个编码层
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, input, src, mask=None, src_key_padding_mask=None):
        output = input
        for i in range(self.num_layers):
            # 每个编码层的输入是上一个层的输出
            output = self.layers[i](output + src[i], output + src[i], output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LightGCN(nn.Module):
    def __init__(self, user_num, item_num, graph, transformer_layers, latent_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.graph = graph
        self.transformer_layers = transformer_layers
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(user_num, latent_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        self.item_emb = nn.Embedding(item_num, latent_dim)
        nn.init.xavier_normal_(self.item_emb.weight)

    def cal_mean(self, embs):
        if len(embs) > 1:
            embs = torch.stack(embs, dim=1)  # 如果有多个嵌入，将它们堆叠
            embs = torch.mean(embs, dim=1)  # 计算堆叠嵌入的平均值
        else:
            embs = embs[0]
        users_emb, items_emb = torch.split(embs, [self.user_num, self.item_num])

        return users_emb, items_emb

    def forward(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb]

        embs_mean = []
        for i in range(self.n_layers):
            embs_mean.append([all_emb])

        for layer in range(self.transformer_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            if layer < self.n_layers:
                embs.append(all_emb)

            for i in range(layer, self.transformer_layers):
                embs_mean[i].append(all_emb)
            # embs_mean[layer].append(all_emb)

        # 计算所有嵌入的平均并分割为用户和物品嵌入
        users, items = self.cal_mean(embs)

        users_mean, items_mean = [], []
        for i in range(self.transformer_layers):
            a, b = self.cal_mean(embs_mean[i])
            users_mean.append(a)
            items_mean.append(b)

        return users, items, users_mean, items_mean


class LightGT(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E,
                 reg_weight, n_layers, device):
        super(LightGT, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.lightgcn_layers = n_layers
        self.transformer_layers = n_layers
        self.device = device

        self.score_weight1 = 0.05
        self.score_weight2 = 1 - self.score_weight1
        self.src_len = 20
        self.nhead = 1

        self.weight = torch.tensor([[1.], [-1.]]).cuda()

        self.v_feat = F.normalize(v_feat) if v_feat is not None else None
        self.t_feat = F.normalize(t_feat) if t_feat is not None else None

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        self.lightgcn = LightGCN(self.num_user, self.num_item, self.norm_adj_mat, self.transformer_layers, self.dim_E,
                                 self.lightgcn_layers)

        self.user_exp = nn.Parameter(torch.rand(self.num_user, self.dim_E))
        nn.init.xavier_normal_(self.user_exp)

        if self.v_feat is not None:
            self.v_mlp = nn.Linear(self.dim_E, self.dim_E)
            self.v_linear = nn.Linear(self.v_feat.size(1), self.dim_E)
            # 单层实现
            self.v_encoder_layer = TransformerEncoderLayer(d_model=self.dim_E, nhead=self.nhead)
            # 多层实现
            self.v_encoder = TransformerEncoder(self.v_encoder_layer, num_layers=self.transformer_layers)
            self.v_dense = nn.Linear(self.dim_E, self.dim_E)

        if self.t_feat is not None:
            self.t_mlp = nn.Linear(self.dim_E, self.dim_E)
            self.t_linear = nn.Linear(self.t_feat.size(1), self.dim_E)
            self.t_encoder_layer = TransformerEncoderLayer(d_model=self.dim_E, nhead=self.nhead)
            self.t_encoder = TransformerEncoder(self.t_encoder_layer, num_layers=self.transformer_layers)
            self.t_dense = nn.Linear(self.dim_E, self.dim_E)

    # 构建邻接矩阵
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

    def forward(self, users, user_item, mask):
        user_emb, item_emb, users_mean, items_mean = self.lightgcn()

        v_src, t_src = [], []
        for i in range(self.transformer_layers):
            temp = items_mean[i][user_item].detach()  # 获取特定物品的嵌入  停止梯度操作
            temp[:, 0] = users_mean[i][users].detach()  # 将对应用户的嵌入设置为第一个位置
            if self.v_feat is not None:
                v_src.append(torch.sigmoid(self.v_mlp(temp).transpose(0, 1)))
            if self.t_feat is not None:
                t_src.append(torch.sigmoid(self.t_mlp(temp).transpose(0, 1)))

        v, t, v_out, t_out = None, None, None, None

        if self.v_feat is not None:
            v = self.v_linear(self.v_feat)
            v_in = v[user_item]  # 获取对应用户物品的视觉特征
            v_in[:, 0] = self.user_exp[users]  # 将用户特定表达设置在第一个位置
            # 使用视觉编码器处理输入数据，使用mask作为键填充掩码
            v_out = self.v_encoder(v_in.transpose(0, 1), v_src, src_key_padding_mask=mask).transpose(0, 1)[:, 0]
            v_out = F.leaky_relu(self.v_dense(v_out))  # 应用激活函数和线性变换

        if self.t_feat is not None:
            t = self.t_linear(self.t_feat)
            t_in = t[user_item]
            t_in[:, 0] = self.user_exp[users]
            t_out = self.t_encoder(t_in.transpose(0, 1), t_src, src_key_padding_mask=mask).transpose(0, 1)[:, 0]
            t_out = F.leaky_relu(self.t_dense(t_out))

        return user_emb, item_emb, v, t, v_out, t_out

    def loss(self, users, items, mask, user_item):
        user_emb, item_emb, v, t, v_out, t_out = self.forward(users[:, 0], user_item, mask.to(self.device))

        users = users.view(-1)
        items = items - self.num_user

        pos_items = items[:, 0].view(-1)
        neg_items = items[:, 1].view(-1)
        items = items.view(-1)

        # id
        score1 = torch.sum(user_emb[users] * item_emb[items], dim=1).view(-1, 2)
        # v and t
        score2_1 = torch.sum(v_out * v[pos_items], dim=1).view(-1, 1) + torch.sum(t_out * t[pos_items], dim=1).view(-1,
                                                                                                                    1)
        score2_2 = torch.sum(v_out * v[neg_items], dim=1).view(-1, 1) + torch.sum(t_out * t[neg_items], dim=1).view(-1,
                                                                                                                    1)

        score = self.score_weight1 * score1 + self.score_weight2 * torch.cat((score2_1, score2_2), dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight)))).cuda()

        reg_embedding_loss = (user_emb ** 2).mean() + (item_emb ** 2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss

        total_loss = loss + reg_loss

        if torch.isnan(loss):
            print('Loss is Nan.')
            exit()

        return total_loss

    def get_score_matrix(self, users, user_item, mask):
        user_emb, item_emb, v, t, v_out, t_out = self.forward(users, user_item, mask.cuda())

        score1 = torch.matmul(user_emb[users], item_emb.T)
        score2 = torch.matmul(v_out, v.T) + torch.matmul(t_out, t.T)

        score_matrix = self.score_weight1 * score1 + self.score_weight2 * score2

        return score_matrix

    def gene_ranklist(self, eval_dataloader, step=2000, topk=50):
        start_index = 0
        end_index = self.num_user if step is None else step

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        for users, user_item, mask in eval_dataloader:
            score_matrix = self.get_score_matrix(users.view(-1), user_item, mask)

            # 将历史交互设置为极小值
            for row, col in self.user_item_dict.items():
                if start_index <= row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - self.num_user
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
