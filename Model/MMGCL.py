"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/11/14 19:01
@File : MMGCL.py
@function :
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
import random



class MMGCL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E,
                 reg_weight, n_layers, ssl_alpha, ssl_temp, dropout, device):
        super(MMGCL, self).__init__()
        self.result_item = None
        self.result_user = None
        self.t_dense_emb = None
        self.v_dense_emb = None
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.edge_index = edge_index
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.ssl_alpha = ssl_alpha
        self.device = device
        self.ssl_temp = ssl_temp
        self.ssl_task = "ED+MM+CN"  # "ED+MM+CN", "ED+MM", "ED", "MM"
        self.dropout_rate = dropout  # 0.05
        self.dropout = nn.Dropout(p=self.dropout_rate)  # 创建Dropout层
        self.ssl_criterion = nn.CrossEntropyLoss()
        self.p_vat = [0.5, 0.5]  # 掩码概率

        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数

        self.user_embeddings = nn.Embedding(self.num_user, self.dim_E)
        self.item_embeddings = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                            (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                           shape=(self.num_user, self.num_item), dtype=np.float32)

        # 将COO格式转换为CSR格式
        self.ui_interaction = interaction_matrix.tocsr()

        # 初始化多模态特征
        self.v_feat = F.normalize(v_feat, dim=1)  # 标准化视觉特征
        self.v_dense = nn.Linear(self.v_feat.shape[1], self.dim_E)  # 创建视觉特征到潜在维度的线性映射
        nn.init.xavier_uniform_(self.v_dense.weight)

        self.t_feat = F.normalize(t_feat, dim=1)  # 标准化文本特征
        self.t_dense = nn.Linear(self.t_feat.shape[1], self.dim_E)  # 创建文本特征到潜在维度的线性映射
        nn.init.xavier_uniform_(self.t_dense.weight)

        self.item_feat_dim = self.dim_E * 3  # 计算物品特征的总维度
        self.read_user = nn.Linear(self.item_feat_dim, self.dim_E)
        self.read_item = nn.Linear(self.item_feat_dim, self.dim_E)
        nn.init.xavier_uniform_(self.read_user.weight)
        nn.init.xavier_uniform_(self.read_item.weight)

        # 将csr格式的邻接矩阵转化为拉普拉斯矩阵并标准化
        sp_adj = self.convert_to_laplacian_mat(self.ui_interaction)
        self.norm_adj = self.convert_sparse_mat_to_tensor(sp_adj).to(self.device)  # 将拉普拉斯矩阵转换为tensor

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()  # 获取邻接矩阵的形状
        n_nodes = adj_shape[0] + adj_shape[1]  # 计算图中节点的总数，即用户数和物品数之和
        # 获取非零元素的位置，即存在交互的用户ID和物品ID
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data  # 获取非零元素的值，这里假设所有存在的交互都用1表示
        # 创建一个新的CSR格式稀疏矩阵，这个矩阵的形状是(n_nodes, n_nodes)，即将用户和物品视为图中的节点
        # 这里的关键操作是将物品ID调整（加上adj_shape[0]）以保证不与用户ID重叠
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T  # 使得矩阵对称，因为用户-物品交互是无向的，用户i到物品j的路径和物品j到用户i的路径应该是等价的

        # 调用normalize_graph_mat方法对矩阵进行标准化，返回标准化后的拉普拉斯矩阵
        return self.normalize_graph_mat(tmp_adj)

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1)).flatten()
        # 使用np.where将rowsum中的0替换为一个很小的正数，避免除以零
        rowsum = np.where(rowsum == 0, 1e-10, rowsum)
        if shape[0] == shape[1]:
            # 计算D^(-1/2)
            d_inv_sqrt = np.power(rowsum, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 额外的安全措施，尽管不太可能发生
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        else:
            # 计算D^(-1) 对于非方阵情况
            d_inv = np.power(rowsum, -1.0)
            d_inv[np.isinf(d_inv)] = 0.  # 同样的安全措施
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_sparse_mat_to_tensor(self, mat):
        coo = mat.tocoo()
        # 先将coo格式的行索引和列索引转换为一个NumPy数组
        indices = np.vstack((coo.row, coo.col))
        i = torch.tensor(indices, dtype=torch.long)
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def node_dropout(self, sp_adj, drop_rate):
        adj_shape = sp_adj.get_shape()
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    def edge_dropout(self, sp_adj, drop_rate):
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj

    def sgl_encoder(self, user_emb, item_emb, perturbed_adj=None):
        ego_embeddings = torch.cat([user_emb, item_emb], 0)
        all_embeddings = [ego_embeddings]

        # 对每一层图卷积进行迭代
        for k in range(self.n_layers):
            # 如果提供了扰动的邻接矩阵
            if perturbed_adj is not None:
                # 如果扰动的邻接矩阵是列表形式（可能对应于多层的不同扰动）
                if isinstance(perturbed_adj, list):
                    # 使用对应层的扰动邻接矩阵进行图卷积
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    # 如果扰动的邻接矩阵不是列表，那么对所有层使用相同的扰动邻接矩阵进行图卷积
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item])

        return user_all_embeddings, item_all_embeddings

    # 正常的图卷积
    def forward(self):
        # 获取用户和物品的嵌入向量
        users_emb = self.user_embeddings.weight
        items_emb = self.item_embeddings.weight

        self.v_dense_emb = self.v_dense(self.v_feat)  # v=>id
        self.t_dense_emb = self.t_dense(self.t_feat)  # t=>id

        # 使用图编码器对用户和物品嵌入进行编码
        i_emb_u, i_emb_i = self.sgl_encoder(users_emb, items_emb)
        # 模态嵌入
        v_emb_u, v_emb_i = self.sgl_encoder(users_emb, self.v_dense_emb)
        t_emb_u, t_emb_i = self.sgl_encoder(users_emb, self.t_dense_emb)

        # 融合嵌入
        user = self.read_user(torch.cat([i_emb_u, v_emb_u, t_emb_u], dim=1))
        item = self.read_item(torch.cat([i_emb_i, v_emb_i, t_emb_i], dim=1))

        return user, item

    def random_graph_augment(self, aug_type):
        dropped_mat = None
        # 根据增强类型，选择不同的图扰动方式
        if aug_type == 1:
            # 节点丢弃：随机丢弃图中的一部分节点
            dropped_mat = self.node_dropout(self.ui_interaction, self.dropout_rate)
        elif aug_type == 0:
            # 边丢弃：随机丢弃图中的一部分边
            dropped_mat = self.edge_dropout(self.ui_interaction, self.dropout_rate)
        # 将经过扰动的图转换为拉普拉斯矩阵
        dropped_mat = self.convert_to_laplacian_mat(dropped_mat)
        # 将拉普拉斯矩阵转换为张量，并移动到CUDA设备上
        return self.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def graph_reconstruction(self, aug_type):

        dropped_adj = self.random_graph_augment(aug_type)

        return dropped_adj

    # 边缘丢弃方法
    def modality_edge_dropout_emb(self, user, pos_item, neg_item):
        # 获取用户和物品的原始嵌入向量
        users_emb = self.user_embeddings.weight
        items_emb = self.item_embeddings.weight
        v_dense = self.v_dense_emb
        t_dense = self.t_dense_emb

        # 使用边缘丢弃策略生成扰动的邻接矩阵 aug_type=0
        perturbed_adj = self.graph_reconstruction(aug_type=0)

        # 使用扰动的邻接矩阵和sgl_encoder函数编码用户和物品嵌入
        i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb, perturbed_adj)
        v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
        t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense, perturbed_adj)

        # 根据给定的用户ID、正例和负例物品ID选取相应的嵌入
        i_emb_u_sub, i_emb_i_sub, i_emb_neg_i_sub = i_emb_u_sub[user], i_emb_i_sub[pos_item], i_emb_i_sub[neg_item]
        v_emb_u_sub, v_emb_i_sub, v_emb_neg_i_sub = v_emb_u_sub[user], v_emb_i_sub[pos_item], v_emb_i_sub[neg_item]
        t_emb_u_sub, t_emb_i_sub, t_emb_neg_i_sub = t_emb_u_sub[user], t_emb_i_sub[pos_item], t_emb_i_sub[neg_item]

        users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub, t_emb_u_sub], dim=1))
        items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, t_emb_i_sub], dim=1))
        neg_items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, t_emb_neg_i_sub], dim=1))

        # 对融合后的用户和物品嵌入进行标准化
        users_sub = F.normalize(users_sub, dim=1)
        items_sub = F.normalize(items_sub, dim=1)
        neg_items_sub = F.normalize(neg_items_sub, dim=1)

        # 返回融合并标准化后的用户、物品和负例物品嵌入
        return users_sub, items_sub, neg_items_sub

    # 模态掩码方法
    def modality_masking_emb(self, user, pos_item, neg_item):
        # 获取用户和物品的原始嵌入向量
        users_emb = self.user_embeddings.weight
        items_emb = self.item_embeddings.weight
        v_dense = self.v_dense_emb
        t_dense = self.t_dense_emb

        # 使用模态掩蔽策略生成扰动的邻接矩阵
        perturbed_adj = self.graph_reconstruction(aug_type=1)

        # 随机选择视觉或文本模态进行掩蔽
        modalities = ["image", "text"]
        modality_index = np.random.choice(len(modalities), p=self.p_vat)
        modality = modalities[modality_index]

        i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb)

        v_emb_u_sub, v_emb_i_sub, t_emb_u_sub, t_emb_i_sub = None, None, None, None
        if modality == "image":
            v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
            t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense)
        elif modality == "text":
            t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense, perturbed_adj)
            v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, v_dense)

        # 根据用户ID、正例ID和负例ID获取对应的嵌入向量
        i_emb_u_sub, i_emb_i_sub, i_emb_neg_i_sub = i_emb_u_sub[user], i_emb_i_sub[pos_item], i_emb_i_sub[neg_item]
        v_emb_u_sub, v_emb_i_sub, v_emb_neg_i_sub = v_emb_u_sub[user], v_emb_i_sub[pos_item], v_emb_i_sub[neg_item]
        t_emb_u_sub, t_emb_i_sub, t_emb_neg_i_sub = t_emb_u_sub[user], t_emb_i_sub[pos_item], t_emb_i_sub[neg_item]

        # 融合不同模态的嵌入向量，并对融合后的嵌入进行标准化
        users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub, t_emb_u_sub], dim=1))
        items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, t_emb_i_sub], dim=1))
        neg_items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_neg_i_sub, t_emb_neg_i_sub], dim=1))

        users_sub = F.normalize(users_sub, dim=1)
        items_sub = F.normalize(items_sub, dim=1)
        neg_items_sub = F.normalize(neg_items_sub, dim=1)

        return users_sub, items_sub, neg_items_sub

    # 计算ssl损失
    def cal_multiview_MM_ED_CN(self, users, pos_items, neg_items):
        # 根据配置的自监督学习任务类型执行不同的逻辑
        if self.ssl_task == "ED+MM":
            # 对于"ED+MM"任务，使用边缘丢弃和模态掩蔽两种数据增强策略
            users_sub_1, items_sub_1, _ = self.modality_edge_dropout_emb(users, pos_items, neg_items)
            users_sub_2, items_sub_2, _ = self.modality_masking_emb(users, pos_items, neg_items)

            # 计算两组增强后嵌入的相似度矩阵（logits），并基于这些相似度计算交叉熵损失
            logits_1 = torch.mm(users_sub_1, items_sub_1.T) / self.ssl_temp
            labels_1 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_1 = F.cross_entropy(logits_1, labels_1)

            logits_2 = torch.mm(items_sub_1, items_sub_1.T) / self.ssl_temp
            labels_2 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_2 = F.cross_entropy(logits_2, labels_2)
            ssl_loss = ssl_loss_1 + ssl_loss_2

            return ssl_loss

        elif self.ssl_task == "ED+MM+CN":
            # 对于"ED+MM+CN"任务，除了边缘丢弃和模态掩蔽，还考虑负样本
            users_sub_1, items_sub_1, neg_items_sub_1 = self.modality_edge_dropout_emb(users, pos_items, neg_items)
            users_sub_2, items_sub_2, neg_items_sub_2 = self.modality_masking_emb(users, pos_items, neg_items)

            # 计算正样本和负样本的损失，并将它们相加
            logits_1 = torch.mm(users_sub_1, items_sub_1.T) / self.ssl_temp
            labels_1 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_1 = F.cross_entropy(logits_1, labels_1)

            logits_2 = torch.mm(users_sub_1, items_sub_2.T) / self.ssl_temp
            labels_2 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_2 = F.cross_entropy(logits_2, labels_2)

            logits_3 = torch.mm(users_sub_1, neg_items_sub_1.T)
            logits_3 /= self.ssl_temp
            labels_3 = torch.tensor(list(range(neg_items_sub_1.shape[0]))).to(self.device)
            ssl_loss_3 = - F.cross_entropy(logits_3, labels_3)
            ssl_loss = ssl_loss_1 + ssl_loss_2

            return ssl_loss

        elif self.config["ssl_task"] == "ED":
            # 对于"ED"任务，只使用边缘丢弃策略
            users_sub_1, items_sub_1, _ = self.modality_edge_dropout_emb(users, pos_items, neg_items)
            users_sub_2, items_sub_2, _ = self.modality_edge_dropout_emb(users, pos_items, neg_items)

            logits_1 = torch.mm(users_sub_1, items_sub_1.T) / self.ssl_temp
            labels_1 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_1 = F.cross_entropy(logits_1, labels_1)

            logits_2 = torch.mm(users_sub_2, items_sub_2.T) / self.ssl_temp
            labels_2 = torch.tensor(list(range(items_sub_2.shape[0]))).to(self.device)
            ssl_loss_2 = F.cross_entropy(logits_2, labels_2)
            ssl_loss = ssl_loss_1 + ssl_loss_2

            return ssl_loss

    def extract_ui_embeddings(self, users, pos_items, neg_items=None):
        # 通过compute函数计算得到当前的用户和物品嵌入
        user_embeddings, item_embeddings = self.compute()
        # 提取指定用户和物品的嵌入向量
        user_emb = user_embeddings[users]
        positive_emb = item_embeddings[pos_items]
        # 如果存在负样本，则同样提取负样本的嵌入向量，否则为None
        negative_emb = None if neg_items is None else item_embeddings[neg_items]

        return user_emb, positive_emb, negative_emb

    def bpr_loss(self, users, pos_items, neg_items, u_g, i_g):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items, u_g, i_g):
        # 计算正则化损失
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]
        reg_loss = self.reg_weight * (torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2)
                                      + torch.mean(neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        user, item = self.forward()
        self.result_user = user
        self.result_item = item

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, user, item)
        # reg_loss = self.regularization_loss(users, pos_items, neg_items, user, item)

        ssl_loss = self.cal_multiview_MM_ED_CN(users, pos_items, neg_items)

        total_loss = bpr_loss + self.ssl_alpha * ssl_loss
        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.result_user.cpu()
        item_tensor = self.result_item.cpu()

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
