"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/30 19:52
@File : VGCL.py
@function :
"""
import faiss
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F


class VGCL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, ssl_temp,
                 ssl_alpha, device):
        super(VGCL, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.device = device

        self.alpha = ssl_alpha  # α参数，可能用于控制对比学习损失的权重
        self.beta = 1  # β参数，可能用于控制KL散度损失的权重
        # self.gamma = gamma  # γ参数，可能用于调节某种损失的权重
        self.temp_node = ssl_temp  # 节点级对比学习的温度参数 0.2
        self.temp_cluster = 0.7 * ssl_temp  # 簇级对比学习的温度参数 0.13
        self.num_user_cluster = 50
        self.num_item_cluster = 50

        # 初始化用户和项目嵌入
        self.user_embedding = nn.Embedding(num_embeddings=self.num_user, embedding_dim=self.dim_E)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_item, embedding_dim=self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        self.adj_matrix = self.get_norm_adj_mat().to(self.device)

        # 初始化嵌入参数的权重和偏置，用于图编码器中生成节点嵌入的高斯分布参数
        self.eps_weight = nn.Parameter(torch.randn(self.dim_E, self.dim_E))
        nn.init.xavier_uniform_(self.eps_weight)
        self.eps_bias = nn.Parameter(torch.zeros(self.dim_E))  # 偏置初始化为0

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
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        return SparseL

    def e_step(self):
        # user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        # item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        user_embeddings = self.user_emb.detach().cpu().numpy()
        item_embeddings = self.item_emb.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings, self.num_user_cluster)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings, self.num_item_cluster)

    def run_kmeans(self, x, num_cluster):
        """
        使用K-means算法对输入的张量x进行聚类。

        参数:
        - x: 输入的张量，例如用户或物品的嵌入向量

        返回:
        - centroids: 聚类中心的张量
        - node2cluster: 每个节点（用户或物品）所属的聚类索引
        """
        # 初始化faiss库的Kmeans对象，设置维度、聚类数和是否使用GPU
        kmeans = faiss.Kmeans(d=self.dim_E, k=num_cluster, gpu=False)
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
        node2cluster = torch.LongTensor(I).to(self.device)  # 这里的I是二维的向量，区别于NCL
        return centroids, node2cluster

    def graph_encoder(self):
        # 将用户和物品的潜在嵌入向量合并为一整个嵌入矩阵
        ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_emb = []
        for _ in range(self.n_layers):
            ego_emb = torch.sparse.mm(self.adj_matrix, ego_emb)
            all_emb.append(ego_emb)

        # 计算所有层的嵌入的平均值，作为节点嵌入的均值
        mean = torch.mean(torch.stack(all_emb), dim=0)
        # 使用均值与预定义的权重进行矩阵乘法，并加上偏置，以此计算对数标准差 (MLP)
        logstd = torch.matmul(mean, self.eps_weight) + self.eps_bias
        # 通过指数函数将对数标准差转换为标准差
        std = torch.exp(logstd)
        # 重参数化技巧：生成两组噪声
        noise1 = torch.randn_like(std)
        noise2 = torch.randn_like(std)
        # 通过均值、标准差和噪声生成两组噪声嵌入
        noised_emb1 = mean + 0.01 * std * noise1
        noised_emb2 = mean + 0.01 * std * noise2
        return noised_emb1, noised_emb2, mean, std

    def forward(self):
        noised_emb1, noised_emb2, self.mean, self.std = self.graph_encoder()
        self.user_emb, self.item_emb = torch.split(noised_emb1, [self.num_user, self.num_item], dim=0)

        self.user_emb_sub1, self.item_emb_sub1 = torch.split(noised_emb1, [self.num_user, self.num_item], dim=0)
        self.user_emb_sub2, self.item_emb_sub2 = torch.split(noised_emb2, [self.num_user, self.num_item], dim=0)

    def bpr_loss(self, users, pos_items, neg_items):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = self.user_emb[users]
        pos_item_embeddings = self.item_emb[pos_items]
        neg_item_embeddings = self.item_emb[neg_items]

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

        # reg_loss = self.reg_weight * (torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2)
        #                               + torch.mean(neg_item_embeddings ** 2) + torch.mean(self.eps_weight ** 2)
        #                               + torch.mean(self.eps_bias ** 2))
        reg_loss = self.reg_weight * (torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2)
                                      + torch.mean(neg_item_embeddings ** 2))

        return reg_loss

    def compute_cl_loss_node(self, users, pos_items):
        """
            节点级对比学习损失
        """

        # 用户部分
        user_emb1 = self.user_emb_sub1[users]
        user_emb2 = self.user_emb_sub2[users]
        normalize_user_emb1 = F.normalize(user_emb1, p=2, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, p=2, dim=1)
        pos_score_user = (normalize_user_emb1 * normalize_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(normalize_user_emb1, normalize_user_emb2.T)
        pos_score_user = torch.exp(pos_score_user / self.temp_node)
        ttl_score_user = torch.exp(ttl_score_user / self.temp_node).sum(dim=1)
        cl_loss_user = -torch.mean(torch.log(pos_score_user / ttl_score_user))

        # 项目部分
        item_emb1 = self.item_emb_sub1[pos_items]
        item_emb2 = self.item_emb_sub2[pos_items]
        normalize_item_emb1 = F.normalize(item_emb1, p=2, dim=1)
        normalize_item_emb2 = F.normalize(item_emb2, p=2, dim=1)
        pos_score_item = (normalize_item_emb1 * normalize_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(normalize_item_emb1, normalize_item_emb2.T)
        pos_score_item = torch.exp(pos_score_item / self.temp_node)
        ttl_score_item = torch.exp(ttl_score_item / self.temp_node).sum(dim=1)
        cl_loss_item = -torch.mean(torch.log(pos_score_item / ttl_score_item))

        # 综合用户和物品的对比学习损失
        cl_loss = self.alpha * (cl_loss_user + cl_loss_item)
        return cl_loss

    def compute_cl_loss_cluster(self, users, pos_items):
        """
        簇级对比学习损失计算
        (1) 使用K-means聚类的结果作为原型分布，分布是onehot的
        (2) 对于每个锚点节点，选择与其有相同聚类原型的用户/物品作为正样本
        (3) 相比节点级的对比学习损失，簇级对比学习的温度参数可以设置得更小
        """
        # 选择正样本
        user_cluster_id = F.embedding(users, self.user_2cluster)  # 查找用户的簇ID
        user_mask = (user_cluster_id == user_cluster_id.transpose(0, 1)).float()  # 创建用户的掩码矩阵
        num_pos_per_cow = user_mask.sum(dim=1)
        item_cluster_id = F.embedding(pos_items, self.item_2cluster)  # 查找物品的簇ID
        item_mask = (item_cluster_id == item_cluster_id.transpose(0, 1)).float()  # 创建物品的掩码矩阵
        num_item_pos_per_cow = item_mask.sum(dim=1)

        # 用户的对比学习
        user_emb1 = F.embedding(users, self.user_emb_sub1)
        user_emb2 = F.embedding(users, self.user_emb_sub2)
        normalize_user_emb1 = F.normalize(user_emb1, p=2, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, p=2, dim=1)
        logit = torch.matmul(normalize_user_emb1, normalize_user_emb2.transpose(0, 1)) / self.temp_cluster
        logit = logit - logit.max(dim=1, keepdim=True).values  # 从每行中减去最大值(提高数值稳定性并防止梯度爆炸)
        exp_logit = torch.exp(logit)
        denominator = exp_logit.sum(dim=1, keepdim=True)
        log_probs = exp_logit / denominator * user_mask  # 仅保留属于同一簇的用户对的概率
        log_probs = log_probs.sum(dim=1)  # 得到每个用户与其簇内其他用户的总相似度
        log_probs = log_probs / num_pos_per_cow  # 进行归一化，得到平均相似度
        cl_loss_user = -torch.mean(torch.log(log_probs))

        # 对物品进行相同的处理
        item_emb1 = F.embedding(pos_items, self.item_emb_sub1)
        item_emb2 = F.embedding(pos_items, self.item_emb_sub2)
        normalize_item_emb1 = F.normalize(item_emb1, p=2, dim=1)
        normalize_item_emb2 = F.normalize(item_emb2, p=2, dim=1)
        logit_item = torch.matmul(normalize_item_emb1, normalize_item_emb2.transpose(0, 1)) / self.temp_cluster
        logit_item = logit_item - logit_item.max(dim=1, keepdim=True).values
        exp_logit_item = torch.exp(logit_item)
        denominator_item = exp_logit_item.sum(dim=1, keepdim=True)
        log_probs_item = exp_logit_item / denominator_item * item_mask
        log_probs_item = log_probs_item.sum(dim=1)
        log_probs_item = log_probs_item / num_item_pos_per_cow
        cl_loss_item = -torch.mean(torch.log(log_probs_item))

        cl_loss = self.alpha * (cl_loss_user + cl_loss_item)  # 组合用户和物品的对比学习损失
        return cl_loss

    def kl_regularizer(self, mean, std):
        """
        计算KL散度正则项，用于ELBO损失中
        旨在将近似后验分布拉近至先验分布
        """
        batch_size = 1024
        # 计算KL散度正则项的公式
        regu_loss = -0.5 * (1 + 2 * std - mean.pow(2) - std.exp().pow(2))
        # 对所有维度求和，然后取平均，最后除以批次大小以获得正则化损失
        return regu_loss.sum(1).mean() / batch_size

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        bpr_loss = self.bpr_loss(users, pos_items, neg_items)
        reg_loss = self.regularization_loss(users, pos_items, neg_items)
        cl_loss_node = self.compute_cl_loss_node(users, pos_items)
        cl_loss_cluster = self.compute_cl_loss_cluster(users, pos_items)
        kl_loss = self.kl_regularizer(self.mean, self.std) * self.beta

        loss = bpr_loss + reg_loss + cl_loss_node + cl_loss_cluster + kl_loss

        return loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.user_emb[:self.num_user].cpu()
        item_tensor = self.item_emb[:self.num_item].cpu()

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
