"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/7 15:43
@File : SGL.py
@function :
"""
import scipy.sparse as sp
from torch import nn
import numpy as np
import torch
import torch.sparse as torch_sp
import torch.nn.functional as F


def sp_mat_to_sp_tensor(sp_mat):
    # 将输入的Scipy稀疏矩阵转换为COO格式，并确保数据类型为float32
    # COO格式是一种存储稀疏矩阵的格式，它通过三个数组（行索引、列索引和元素值）来表示非零元素
    coo = sp_mat.tocoo().astype(np.float32)

    # 从COO格式的矩阵中提取行索引和列索引，并将它们组合成一个2D NumPy数组
    # 这个2D数组的形状是(2, N)，其中N是非零元素的数量
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))

    # 使用提取的索引和非零元素值创建一个PyTorch的稀疏张量
    # torch.sparse_coo_tensor需要索引数组、值数组和稀疏张量的形状作为输入
    # indices指定了每个非零元素在稀疏张量中的位置，coo.data包含了这些非零元素的值
    # coo.shape指定了稀疏张量的总体形状
    # 最后，调用.coalesce()方法来确保稀疏张量以一种优化的方式存储，合并任何重复的索引
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


class SGL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, aggr_mode, ssl_temp,
                 ssl_reg, device):
        super(SGL, self).__init__()
        self.item_emb_final = None
        self.user_emb_final = None
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.aggr_mode = aggr_mode
        self.device = device
        self.ssl_aug_type = 'ed'  # 'nd','ed', 'rw'
        self.ssl_temp = ssl_temp  # 温度系数 0.2
        self.ssl_reg = ssl_reg  # 正则化系数
        self.ssl_ratio = 0.1  # 数据增强中节点或边删除的比例
        self.edge_index = edge_index

        # 初始化邻接矩阵(用于GCF)
        adj_matrix = self.create_adj_mat()
        self.norm_adj = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.user_embeddings = nn.Embedding(self.num_user, self.dim_E)
        self.item_embeddings = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_user + self.num_item
        users_np, items_np = self.edge_index[:, 0], self.edge_index[:, 1]
        items_np = items_np - self.num_user  # 确保项目编号和用户编号都从0开始
        tmp_adj = None

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                # 选出丢弃节点的id
                drop_user_idx = np.random.choice(self.num_user, size=int(self.num_user * self.ssl_ratio), replace=False)
                drop_item_idx = np.random.choice(self.num_item, size=int(self.num_item * self.ssl_ratio), replace=False)
                # 用于标记哪些节点（用户或物品）会在数据增强步骤中被保留
                indicator_user = np.ones(self.num_user, dtype=np.float32)
                indicator_item = np.ones(self.num_item, dtype=np.float32)
                # 节点删除
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                # 转换为对角矩阵
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                # 创建稀疏矩阵 sp.csr_matrix
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
                    shape=(self.num_user, self.num_item))
                # 实现了节点删除
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                # 保留下来的用户-物品交互
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                # 保留下来的交互的权重
                ratings_keep = R_prime.data
                # 构建新的邻接矩阵
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.num_user)),
                                        shape=(n_nodes, n_nodes))
            if aug_type in ['ed', 'rw']:
                # 保留的用户-项目对
                keep_idx = np.random.choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)),
                                            replace=False)
                # 只包含了被随机保留下来的用户和项目编号
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)

                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_user)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # 归一化邻接矩阵
        rowsum = np.array(adj_mat.sum(1)).flatten()
        # 避免除以零的情况，将0值替换为非常小的正数
        rowsum[rowsum == 0] = 1e-10
        d_inv = np.power(rowsum, -0.5)
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            # 如果norm_adj是列表，表示每一层有不同的规范化邻接矩阵
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                # 如果norm_adj不是列表，表示所有层使用相同的规范化邻接矩阵
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)

        return user_embeddings, item_embeddings

    def forward(self):
        # 创建不同的视图
        if self.ssl_aug_type in ['nd', 'ed']:
            sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
            sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
            sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
            sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
        else:
            sub_graph1, sub_graph2 = [], []
            for _ in range(0, self.n_layers):
                tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
                tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))

        # 进行图卷积
        user_emb, item_emb = self.gcn(self.norm_adj)
        user_s1, item_s1 = self.gcn(sub_graph1)
        user_s2, item_s2 = self.gcn(sub_graph2)

        return user_emb, item_emb, user_s1, item_s1, user_s2, item_s2

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = user_emb[users]
        pos_item_embeddings = item_emb[pos_items]
        neg_item_embeddings = item_emb[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def inner_product(self, a, b):
        return torch.sum(a * b, dim=-1)

    def ssl_loss(self, users, items, user_s1, item_s1, user_s2, item_s2):
        # 归一化
        user_embeddings1 = F.normalize(user_s1, dim=1)
        item_embeddings1 = F.normalize(item_s1, dim=1)
        user_embeddings2 = F.normalize(user_s2, dim=1)
        item_embeddings2 = F.normalize(item_s2, dim=1)

        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        pos_ratings_user = self.inner_product(user_embs1, user_embs2)  # [batch_size]
        pos_ratings_item = self.inner_product(item_embs1, item_embs2)  # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1,
                                        torch.transpose(user_embeddings2, 0, 1))  # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))  # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]

        # InfoNCE Loss
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
        infonce_loss = torch.sum(clogits_user + clogits_item)

        return infonce_loss

    def regularization_loss(self, users, pos_items, neg_items):
        # 计算正则化损失
        user_embeddings = self.user_embeddings.weight[users]
        pos_item_embeddings = self.item_embeddings.weight[pos_items]
        neg_item_embeddings = self.item_embeddings.weight[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        user_emb, item_emb, user_s1, item_s1, user_s2, item_s2 = self.forward()
        self.user_emb_final = user_emb
        self.item_emb_final = item_emb

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
        # 正则化损失
        reg_loss = self.regularization_loss(users, pos_items, neg_items)
        # 对比损失
        ssl_loss = self.ssl_loss(users, pos_items, user_s1, item_s1, user_s2, item_s2)

        loss = bpr_loss + reg_loss + self.ssl_reg * ssl_loss

        return loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.user_emb_final[:self.num_user].cpu()
        item_tensor = self.item_emb_final[:self.num_item].cpu()

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
