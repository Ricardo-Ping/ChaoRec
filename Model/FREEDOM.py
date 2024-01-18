"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/27 21:14
@File : FREEDOM.py
@function :
"""
import torch
from torch import nn
import torch.nn.functional as F


class FREEDOM(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, dim_feat, reg_weight,
                 dropout, n_layers, mm_layers, ii_topk, mm_image_weight, device):
        super(FREEDOM, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.n_nodes = self.num_user + self.num_item  # 节点总数
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.dim_feat = dim_feat
        self.reg_weight = reg_weight
        self.n_layers = n_layers  # 用户-项目交互图层数
        self.mm_layers = mm_layers  # 项目-项目多模态交互图层数
        self.mm_image_weight = mm_image_weight
        self.dropout = dropout  # Freedom的dropout率很高 0.9
        self.knn_k = ii_topk  # 项目-项目图的k个最近邻
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.device = device

        # 转置并设置为无向图
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        # 不剪枝的邻接矩阵
        self.norm_adj = self.get_norm_adj_mat(self.edge_index).to(self.device)
        # 定义剪枝的邻接矩阵，和项目多模态的邻接矩阵
        self.masked_adj, self.mm_adj = None, None

        # 得到用户-项目交互图的边和权重
        self.edge_indices, self.edge_values = self.get_edge_info(self.edge_index_clone)
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)

        # 创建用户和项目嵌入
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # ==================项目-项目多模态图==================
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.dim_feat)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.dim_feat)

        if self.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.mm_adj = image_adj
        if self.t_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj
            del image_adj

    def get_norm_adj_mat(self, edge_index):
        edge_index = edge_index.long()
        row, col = edge_index
        deg = torch.bincount(torch.cat([row, col]))

        # 计算归一化值
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 返回归一化的邻接矩阵，以稀疏张量的形式
        return torch.sparse_coo_tensor(edge_index, norm, torch.Size([self.n_nodes, self.n_nodes]))

    def _normalize_adj_m(self, indices, adj_size):
        # adj_size:(num_user, num_item)
        # 创建一个稀疏的邻接矩阵，其权重都为1
        # torch.sparse_coo_tensor
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        # 计算adj每一行和每一列的和==计算每个节点的度数
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        # 计算了每行和每列和的平方根的逆
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        # 计算了每条边的归一化权值
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self, edge_index):
        rows, cols = edge_index  # 从edge_index中直接提取行和列  这里是有向图
        cols = cols - self.num_user
        edges = torch.stack([rows, cols]).type(torch.LongTensor)  # edges_index
        # edge normalized values
        # 归一化值
        values = self._normalize_adj_m(edges, torch.Size((self.num_user, self.num_item)))
        return edges, values

    def get_knn_adj_mat(self, mm_embeddings):
        # mm_embeddings: 项目的多模态嵌入
        # 计算l2范数
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        # 计算相似性矩阵
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        # 找到k个最近邻 knn_ind:(num_item, 10)
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # 创建项目的索引
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)  # 扩展  (num_item, 10)
        # 相当于创建了knn_k之后的edges_index (2, num_item * 10)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        # 权重全1
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        # 计算邻接矩阵的每一行的和
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        # 计算归一化值
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    # 用于在每个epoch开始之前进行剪枝
    def pre_epoch_processing(self):
        # 如果dropout系数为0或小于0，直接将标准化的邻接矩阵赋值给掩码邻接矩阵，并结束此方法
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # 根据dropout系数确定要保留的边的数量
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        # 随机选择degree_len个边进行采样，使用多项式分布进行随机采样，这里的self.edge_values是邻接矩阵中每条边的权重
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # 获取这些随机选择的边的索引
        keep_indices = self.edge_indices[:, degree_idx]
        # 归一化处理
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.num_user, self.num_item)))
        # 无向图
        all_values = torch.cat((keep_values, keep_values))
        # 更新边的索引，使其包括用户和项目
        keep_indices[1] += self.num_user
        # 无向图
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def forward(self, adj):
        # 对项目-项目图更新项目表示
        h = self.item_embedding.weight
        for i in range(self.mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # 用户-项目图进行卷积
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)
        i_g_embeddings = i_g_embeddings + h
        self.result = torch.cat((u_g_embeddings, i_g_embeddings), dim=0)

        return u_g_embeddings, i_g_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                      neg_i_g_embeddings)
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])

        total_loss = batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

        return total_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        # _, _ = self.forward(self.norm_adj)
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