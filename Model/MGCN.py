"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/9 21:50
@File : MGCN.py
@function :
"""
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


class MGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight,
                 n_layers, aggr_mode, ssl_temp, ssl_alpha, device):
        super(MGCN, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.n_layers = 1
        self.n_ui_layers = 2
        self.ssl_temp = ssl_temp
        self.ssl_alpha = ssl_alpha
        self.device = device
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.reg_weight = reg_weight
        self.aggr_mode = aggr_mode

        # 指示是否使用稀疏矩阵表示
        self.sparse = True

        # 最近邻算法的k值
        self.knn_k = 10

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        # 初始化用户和项目的嵌入层，使用Xavier方法初始化权重
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 获取归一化的邻接矩阵
        self.norm_adj = self.get_adj_mat()
        # 将R（用户-项目的归一化邻接矩阵）转换为PyTorch的稀疏张量
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        # 将归一化的邻接矩阵转换为PyTorch的稀疏张量
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # 加载多模态特征和创建邻接矩阵
        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
        # 动态构建图像邻接矩阵
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                               norm_type='sym')
        self.image_original_adj = image_adj.to(self.device)

        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)
        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        self.text_original_adj = text_adj.to(self.device)

        # 转换矩阵
        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)

        self.softmax = nn.Softmax(dim=-1)

        # 定义一个查询共通特征的序列模型
        self.query_common = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Tanh(),
            nn.Linear(self.dim_E, 1, bias=False)
        )

        # 定义门控单元，用于控制视觉特征和文本特征在最终嵌入中的贡献
        self.gate_v = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Sigmoid()
        )

    def get_adj_mat(self):
        # 创建一个空的邻接矩阵,使用DOK（Dictionary Of Keys）格式
        adj_mat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        # 将用户-项目的交互矩阵也转换为LIL格式
        R = self.interaction_matrix.tolil()

        # 在邻接矩阵的左上角和右下角填充用户对项目的交互信息和其转置，构建无向图
        adj_mat[:self.num_user, self.num_user:] = R
        adj_mat[self.num_user:, :self.num_user] = R.T
        # 再次将LIL格式的邻接矩阵转换回DOK格式
        adj_mat = adj_mat.todok()

        # 归一化邻接矩阵
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            # 用非常小的正数替换零值，避免除以零
            rowsum[rowsum == 0.] = 1e-16
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)

            return norm_adj.tocoo()

        # 调用上述函数归一化邻接矩阵
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.num_user, self.num_user:]
        # 返回归一化的邻接矩阵，转换为CSR格式
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        # 将SciPy稀疏矩阵转换为COO格式，并确保数据类型为float32
        sparse_mx = sparse_mx.tocoo().astype(np.float32)

        # 从稀疏矩阵中提取行索引和列索引，并将它们堆叠成一个2D数组，
        # 这个数组的shape为[2, 非零元素个数]，之后将数据类型转换为int64
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

        # 从稀疏矩阵中提取非零元素的值
        values = torch.from_numpy(sparse_mx.data)

        # 获取稀疏矩阵的形状，并转换为torch.Size对象
        shape = torch.Size(sparse_mx.shape)

        # 使用提取的索引、值和形状信息构建一个PyTorch的稀疏张量（FloatTensor）
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        # Behavior-Guided Purifier: 通过门控机制调整项目嵌入，使其包含视觉或文本信息
        image_item_embeds = torch.multiply(self.item_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_embedding.weight, self.gate_t(text_feats))

        # User-Item View: 合并用户和项目嵌入，并通过图卷积网络处理用户-项目交互
        item_embeds = self.item_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View: 分别对视觉和文本特征进行图卷积处理，以模拟项目之间的相似性
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Behavior-Aware Fuser: 使用注意力机制融合视觉和文本特征
        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        # 模态特定
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds

        # 使用门控机制调整个性化嵌入
        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3

        # 综合内容嵌入和个性化嵌入
        all_embeds = content_embeds + side_embeds
        self.result = all_embeds
        # 将嵌入分为用户嵌入和项目嵌入
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.num_user, self.num_item], dim=0)

        return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

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
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def InfoNCE(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.ssl_temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, ua_embeddings, ia_embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, ua_embeddings, ia_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.num_user, self.num_item], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.num_user, self.num_item], dim=0)
        ssl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items]) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users])

        total_loss = bpr_loss + self.ssl_alpha * ssl_loss + reg_loss
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


