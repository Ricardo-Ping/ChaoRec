"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/7 21:39
@File : NCL.py
@function :
"""
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import faiss
import torch.nn.functional as F


class NCL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, aggr_mode, ssl_temp,
                 ssl_reg, device):
        super(NCL, self).__init__()
        self.diag = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.aggr_mode = aggr_mode
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.device = device
        self.hyper_layers = 1  # 超参数层的数量
        self.alpha = 1  # 控制某些损失项重要性的系数
        self.proto_reg = 1e-7
        self.k = 200

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                            (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                           shape=(self.num_user, self.num_item), dtype=np.float32)

        self.user_embedding = nn.Embedding(num_embeddings=self.num_user, embedding_dim=self.dim_E)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_item, embedding_dim=self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 用于加速全排序评估的存储变量
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)  # 获取并转移归一化邻接矩阵到设备上

        # 用户和物品的聚类中心及其映射，初始为空
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """
        使用K-means算法对输入的张量x进行聚类。

        参数:
        - x: 输入的张量，例如用户或物品的嵌入向量

        返回:
        - centroids: 聚类中心的张量
        - node2cluster: 每个节点（用户或物品）所属的聚类索引
        """
        # 初始化faiss库的Kmeans对象，设置维度、聚类数和是否使用GPU
        kmeans = faiss.Kmeans(d=self.dim_E, k=self.k, gpu=False)
        # 使用K-means算法对输入x进行训练
        kmeans.train(x)
        # 获取聚类中心
        cluster_cents = kmeans.centroids

        # 对输入x进行聚类索引搜索，即找到每个点最近的聚类中心
        _, I = kmeans.index.search(x, 1)

        # 将聚类中心转换为CUDA张量，并进行L2归一化，以便在后续的计算中使用
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        # 将聚类索引转换为CUDA张量
        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

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

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.num_user, self.num_item])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):

        if self.user_2cluster is None or self.item_2cluster is None:
            raise RuntimeError(
                "user_2cluster or item_2cluster is None. Please ensure e_step is called before ProtoNCE_loss.")

        # 分割用户和项目嵌入
        user_embeddings_all, item_embeddings_all = torch.split(
            node_embedding, [self.num_user, self.num_item]
        )

        # 获取批次用户的嵌入向量，并进行L2归一化
        user_embeddings = user_embeddings_all[user]  # [批次大小, 嵌入维度]
        norm_user_embeddings = F.normalize(user_embeddings)

        # 根据用户到聚类的映射，获取对应的聚类中心
        user2cluster = self.user_2cluster[user]  # [批次大小,]
        user2centroids = self.user_centroids[user2cluster]  # [批次大小, 嵌入维度]

        # 计算用户嵌入向量与其聚类中心的正向得分，并通过softmax温度参数进行缩放
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        # 计算用户嵌入向量与所有聚类中心的总得分
        ttl_score_user = torch.matmul(
            norm_user_embeddings, self.user_centroids.transpose(0, 1)
        )
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        # 计算用户部分的ProtoNCE损失
        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # 对物品嵌入进行相同的处理
        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(
            norm_item_embeddings, self.item_centroids.transpose(0, 1)
        )
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        # 将当前和先前的嵌入向量分别按用户和物品分割
        current_user_embeddings, current_item_embeddings = torch.split(
            current_embedding, [self.num_user, self.num_item]
        )
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(
            previous_embedding, [self.num_user, self.num_item]
        )

        # 获取特定用户的当前和先前嵌入向量，并进行归一化处理
        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        # 计算当前用户嵌入向量与先前所有用户嵌入向量的归一化内积，用于后续计算正、负样本得分
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        # 通过softmax函数（使用温度参数ssl_temp）计算得分，进而计算用户部分的SSL损失
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # 对物品嵌入向量进行相同的处理，以计算物品部分的SSL损失
        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        # 通过softmax函数（使用温度参数ssl_temp）计算得分，进而计算物品部分的SSL损失
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        # 将用户和物品的SSL损失按照配置的正则化权重和alpha系数加权求和，得到最终的SSL损失
        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        self.restore_user_e, self.restore_item_e, embedding_list = self.forward()

        # 获取中心嵌入和上下文嵌入向量
        center_embedding = embedding_list[0]  # 图卷积第0层
        context_embedding = embedding_list[self.hyper_layers * 2]  # 图卷积第2层

        # 计算自监督学习层的损失
        ssl_loss = self.ssl_layer_loss(
            context_embedding, center_embedding, users, pos_items
        )

        # 计算原型对比学习损失
        proto_loss = self.ProtoNCE_loss(center_embedding, users, pos_items)

        # 获取当前批次用户和物品的嵌入向量
        u_embeddings = self.restore_user_e[users]
        pos_embeddings = self.restore_item_e[pos_items]
        neg_embeddings = self.restore_item_e[neg_items]

        # 计算BPR损失
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        # 获取用户和物品的原始嵌入向量，用于正则化损失的计算
        u_ego_embeddings = self.user_embedding(users)
        pos_ego_embeddings = self.item_embedding(pos_items)
        neg_ego_embeddings = self.item_embedding(neg_items)
        reg_loss = self.reg_weight * (
                torch.mean(u_ego_embeddings ** 2) + torch.mean(pos_ego_embeddings ** 2) + torch.mean(
            neg_ego_embeddings ** 2))

        total_loss = bpr_loss + reg_loss + ssl_loss + proto_loss

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.restore_user_e[:self.num_user].cpu()
        item_tensor = self.restore_item_e[:self.num_item].cpu()

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
