"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/9 15:48
@File : HCCF.py
@function :
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp


class HCCF(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, aggr_mode,
                 ssl_alpha, ssl_temp, keepRate, leaky, mult, device):
        super(HCCF, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.edge_index = edge_index
        self.gnn_layer = n_layers  # 2
        self.aggr_mode = aggr_mode
        self.device = device
        self.reg_weight = reg_weight
        self.ssl_alpha = ssl_alpha
        self.ssl_temp = ssl_temp
        self.hyperNum = 128
        self.leaky = leaky  # LeakyReLU
        self.keepRate = keepRate
        self.mult = mult  # 当数据集稀疏时

        # 初始化用户和项目嵌入
        self.uEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_user, self.dim_E)))
        self.iEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_item, self.dim_E)))

        # 用户和物品的超图可学习嵌入矩阵w
        # self.uHyper = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dim_E, self.hyperNum)))  # (64,128)
        # self.iHyper = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dim_E, self.hyperNum)))

        # self.act = nn.LeakyReLU(negative_slope=self.leaky)

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.adj = self.get_norm_adj_mat().to(self.device)  # 归一化的邻接矩阵

    def gcn_layer(self, adj, embeds):
        return torch.spmm(adj, embeds)  # (num_user+num_item, 64)

    def hgnn_layer(self, adj, embeds):
        # HHTE
        lat = (adj.T @ embeds)
        ret = (adj @ lat)
        return ret

    def sp_adj_drop_edge(self):
        # 如果保持率为1.0，即不丢弃任何边，直接返回原始邻接矩阵
        if self.keepRate == 1.0:
            return self.adj

        # 获取稀疏邻接矩阵的值和索引
        vals = self.adj._values()  # 邻接矩阵的非零值
        idxs = self.adj._indices()  # 邻接矩阵非零值的坐标索引

        # 计算邻接矩阵非零元素的数量
        edgeNum = vals.size(0)

        # 生成一个随机数组，并根据保持率计算掩码（mask），用于决定哪些边被保留
        # torch.rand(edgeNum) 生成一个与邻接矩阵非零元素数量相同长度的随机数组，每个元素范围从0到1
        # 加上keepRate后，小于1的部分表示这些边将被丢弃，大于等于1的表示保留
        # 使用.floor()方法将所有小数部分去除，保留整数部分，生成一个0和1组成的数组
        # 最后，将其转换为布尔类型，用作丢弃或保留边的掩码
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        # 使用掩码过滤邻接矩阵的值和索引，仅保留被选中（即保留）的边
        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]

        return torch.sparse_coo_tensor(newIdxs, newVals, self.adj.shape)

    def get_norm_adj_mat(self):
        # 创建一个空的稀疏矩阵A，大小为(n_users+n_items) x (n_users+n_items)，数据类型为float32
        A = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        # 加载交互矩阵
        inter_M = self.interaction_matrix
        # 将交互矩阵转置，以便获取物品到用户的关系
        inter_M_t = self.interaction_matrix.transpose()
        # 将用户到物品的交互和物品到用户的交互合并到邻接矩阵A中
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        # 更新稀疏矩阵A的数据
        A._update(data_dict)

        # 归一化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        rows_and_cols = np.array([row, col])  # 将行和列的列表转换成numpy数组
        i = torch.tensor(rows_and_cols, dtype=torch.long)  # 从numpy数组创建张量
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        return SparseL

    def forward(self):
        # 将用户和物品嵌入向量合并，作为图的节点表示。
        embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]  # 第0层
        gnnLats = [embeds]
        hyperLats = [embeds]

        # 计算超参数嵌入的线性变换。
        # uuHyper = self.uEmbeds @ self.uHyper  # (num_user, 128)
        # iiHyper = self.iEmbeds @ self.iHyper  # (num_item, 128)
        # 如果数据集系数的化
        uuHyper = self.uEmbeds * self.mult
        iiHyper = self.iEmbeds * self.mult

        # 通过GCN和HGNN层进行多层图信息传播和处理。
        for i in range(self.gnn_layer):
            temEmbeds = self.gcn_layer(self.sp_adj_drop_edge(), lats[-1])
            hyperULat = self.hgnn_layer(F.dropout(uuHyper, p=1 - self.keepRate), lats[-1][:self.num_user])
            hyperILat = self.hgnn_layer(F.dropout(iiHyper, p=1 - self.keepRate), lats[-1][self.num_user:])
            gnnLats.append(temEmbeds)
            hyperLats.append(torch.concat([hyperULat, hyperILat], dim=0))
            lats.append(temEmbeds + hyperLats[-1])
        embeds = sum(lats)  # (num_user+num_item, 64)
        self.result = embeds
        return embeds, gnnLats, hyperLats

    def bpr_loss(self, users, pos_items, neg_items, embeddings):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = embeddings[users]
        pos_item_embeddings = embeddings[self.num_user + pos_items]
        neg_item_embeddings = embeddings[self.num_user + neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def ssl_loss(self, embeds1, embeds2, nodes):
        embeds1 = F.normalize(embeds1 + 1e-8, p=2)
        embeds2 = F.normalize(embeds2 + 1e-8, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / self.ssl_temp)
        deno = torch.exp(pckEmbeds1 @ pckEmbeds2.T / self.ssl_temp).sum(-1) + 1e-8
        return -torch.log(nume / deno).mean()

    def regularization_loss(self, users, pos_items, neg_items):
        # 计算正则化损失
        user_embeddings = self.result[users]
        pos_item_embeddings = self.result[self.num_user + pos_items]
        neg_item_embeddings = self.result[self.num_user + neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, embeds)
        reg_loss = self.regularization_loss(users, pos_items, neg_items)

        sslLoss = 0
        for i in range(self.gnn_layer):
            # 图卷积嵌入
            embeds1 = gcnEmbedsLst[i].detach()
            # 超图卷积嵌入
            embeds2 = hyperEmbedsLst[i]
            sslLoss += self.ssl_loss(embeds1[:self.num_user], embeds2[:self.num_user], users) + self.ssl_loss(
                embeds1[self.num_user:], embeds2[self.num_user:], pos_items)

        total_loss = bpr_loss + self.ssl_alpha * sslLoss + reg_loss

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
