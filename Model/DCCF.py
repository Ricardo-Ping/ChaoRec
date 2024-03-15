"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/14 22:45
@File : DCCF.py
@function :
"""
import numpy as np
import torch
import torch_sparse
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F


class DCCF(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, ssl_temp,
                 ssl_alpha, n_intents, cen_reg, device):
        super(DCCF, self).__init__()
        self.ua_embedding = None
        self.ia_embedding = None
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.cen_reg = cen_reg  # 意图嵌入正则化
        self.n_layers = n_layers
        self.device = device
        self.ssl_temp = ssl_temp  # 温度系数
        self.ssl_alpha = ssl_alpha  # 对比系数
        self.n_intents = n_intents  # 意图数量

        self.all_h_list = edge_index[:, 0].tolist()
        self.all_t_list = edge_index[:, 1].tolist()

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        self.all_h_list = torch.LongTensor(self.all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(self.all_t_list).to(self.device)

        # 为用户和物品创建嵌入层
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # 初始化用户和物品的意图嵌入，并设为可训练参数
        _user_intent = torch.empty(self.dim_E, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)
        _item_intent = torch.empty(self.dim_E, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

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

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        # 计算形状
        A_in_shape = (self.num_user + self.num_item, self.num_user + self.num_item)

        # 对头节点和尾节点的嵌入进行归一化处理
        head_embeddings = F.normalize(head_embeddings)
        tail_embeddings = F.normalize(tail_embeddings)
        # 计算边的权重
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2

        # 使用计算出的边权重创建稀疏张量
        i = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        SparseA = torch.sparse_coo_tensor(i, edge_alpha, A_in_shape, dtype=torch.float32).to(self.device)

        return SparseA

    def forward(self):
        # 初始化所有嵌入的列表
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]

        # 初始化各种嵌入的列表
        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = [], [], [], []

        for i in range(0, self.n_layers):
            # 基于图的消息传递
            gnn_layer_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings[i])

            # 意图感知的信息聚合
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.num_user, self.num_item], 0)

            # 第一步：用户和物品在不同意图下的表示
            # u_embeddings @ self.user_intent: [num_users, dim_E] @ [dim_E, n_intents] -> [num_users, n_intents]
            # 第二步：softmax归一化  [num_users, n_intents]
            # 第三步：获取加权的意图嵌入
            # [num_users, n_intents] @ [n_intents, dim_E] -> [num_users, dim_E]
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            # 第四步：聚合用户和物品的意图嵌入
            int_layer_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)

            # 自适应增强
            # gnn的用户和项目嵌入
            gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list)
            gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list)
            # 意图嵌入的用户和项目嵌入
            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list)
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list)
            # 基于节点间的相互作用动态地调整边的权重
            SparseA_gaa = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
            SparseA_iaa = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)

            # 使用自适应增强后的图进行消息传递(扰乱图结构生成了局部和全局的增强表示)
            gaa_layer_embeddings = torch.sparse.mm(SparseA_gaa, all_embeddings[i])
            iaa_layer_embeddings = torch.sparse.mm(SparseA_iaa, all_embeddings[i])

            # 收集各层的嵌入
            gnn_embeddings.append(gnn_layer_embeddings)  # 局部协作视图
            int_embeddings.append(int_layer_embeddings)  # 解耦的全局协作视图
            gaa_embeddings.append(gaa_layer_embeddings)  # 自适应增强的局部协作视图
            iaa_embeddings.append(iaa_layer_embeddings)  # 自适应增强的全局协作视图

            # 聚合前面所有层的嵌入
            all_embeddings.append(
                gnn_layer_embeddings + int_layer_embeddings + gaa_layer_embeddings + iaa_layer_embeddings +
                all_embeddings[i])

        # 堆叠所有层的嵌入，并求和
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)

        # 分割用户和物品的嵌入
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.num_user, self.num_item], 0)

        return gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings

    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), dim=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.num_user, self.num_item], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.num_user, self.num_item], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.num_user, self.num_item], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.num_user, self.num_item], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(u_gnn_embs, u_gaa_embs)
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

        return cl_loss

    def bpr_loss(self, users, pos_items, neg_items):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = self.ua_embedding[users]
        pos_item_embeddings = self.ia_embedding[pos_items]
        neg_item_embeddings = self.ia_embedding[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items):
        # 计算正则化损失
        user_embeddings = self.user_embedding.weight[users]
        pos_item_embeddings = self.item_embedding.weight[pos_items]
        neg_item_embeddings = self.item_embedding.weight[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items)
        reg_loss = self.regularization_loss(users, pos_items, neg_items)
        # intent prototypes
        cen_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
        cen_loss = self.cen_reg * cen_loss
        ssl_loss = self.ssl_alpha * self.cal_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings, gaa_embeddings,
                                                      iaa_embeddings)

        total_loss = bpr_loss + reg_loss + ssl_loss + cen_loss

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.ua_embedding.cpu()
        item_tensor = self.ia_embedding.cpu()

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




