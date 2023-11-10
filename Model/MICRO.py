"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/11/1 21:45
@File : MICRO.py
@function :
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


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
    L_norm = None
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


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index):
        edge_index = edge_index.long()

        row, col = edge_index

        # Compute normalization coefficient
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Apply normalization
        out = norm.view(-1, 1) * x_j

        return out

    def update(self, aggr_out):
        return aggr_out


class MICRO(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E,  n_layer, reg_weight,
                  ii_topk, mm_layers, ssl_temp, lambda_coeff, ssl_alpha, aggr_mode, device):
        super().__init__()
        self.text_adj = None
        self.image_adj = None
        self.text_item_embeds = None
        self.image_item_embeds = None
        self.h = None
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.n_ui_layers = n_layer
        self.topk = ii_topk
        self.sparse = True
        self.norm_type = 'sym'  # 对称归一化
        self.tau = ssl_temp  # 温度系数
        self.lambda_coeff = lambda_coeff
        self.n_ii_layer = mm_layers
        self.device = device
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight
        self.beta = ssl_alpha

        self.user_embedding = nn.Embedding(num_user, self.dim_E)
        self.item_embedding = nn.Embedding(num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)

        self.gcn_layers = nn.ModuleList([GCNConv(self.dim_E, self.dim_E, aggr_mode)
                                         for _ in range(self.n_ui_layers)])

        # 转置为无向图
        self.edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 对多模态计算项目-项目图
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=self.topk, is_sparse=self.sparse,
                                               norm_type=self.norm_type)
        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=self.topk, is_sparse=self.sparse, norm_type=self.norm_type)

        # 保存原始语义图
        self.text_original_adj = text_adj.cuda()
        self.image_original_adj = image_adj.cuda()

        # 特征转换
        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)

        self.softmax = nn.Softmax(dim=-1)

        # 生成注意力向量
        self.query = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Tanh(),
            nn.Linear(self.dim_E, 1, bias=False)
        )

    # 矩阵乘法
    def mm(self, x, y):
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    # 相似性方法
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # 对比辅助任务
    def batched_contrastive_loss(self, z1, z2, batch_size=1024):
        device = z1.device
        num_nodes = z1.size(0)  # 项目数量
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            # 使用mask来选择当前批次
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def forward(self, build_item_graph=False):

        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        if build_item_graph:
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_normalized_graph(self.image_adj, topk=self.topk, is_sparse=self.sparse,
                                                        norm_type=self.norm_type)
            # 跳跃连接
            self.image_adj = (1 - self.lambda_coeff) * self.image_adj + self.lambda_coeff * self.image_original_adj

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_normalized_graph(self.text_adj, topk=self.topk, is_sparse=self.sparse,
                                                       norm_type=self.norm_type)
            # 跳跃连接
            self.text_adj = (1 - self.lambda_coeff) * self.text_adj + self.lambda_coeff * self.text_original_adj

        else:
            self.image_adj = self.image_adj.detach()
            self.text_adj = self.text_adj.detach()

        image_item_embeds = self.item_embedding.weight
        text_item_embeds = self.item_embedding.weight

        # 多模态项目图卷积
        for i in range(self.n_ii_layer):
            self.image_item_embeds = self.mm(self.image_adj, image_item_embeds)
        for i in range(self.n_ii_layer):
            self.text_item_embeds = self.mm(self.text_adj, text_item_embeds)

        # 利用注意力进行多模态融合
        att = torch.cat([self.query(self.image_item_embeds), self.query(self.text_item_embeds)], dim=-1)
        weight = self.softmax(att)
        # h是项目的多模态融合嵌入
        self.h = weight[:, 0].unsqueeze(dim=1) * self.image_item_embeds + weight[:, 1].unsqueeze(dim=1) * self.text_item_embeds

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            ego_embeddings = self.gcn_layers[i](ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)
        i_g_embeddings = i_g_embeddings + F.normalize(self.h, p=2, dim=1)
        self.result = torch.cat((u_g_embeddings, i_g_embeddings), dim=0)
        return self.result

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

    def regularization_loss(self, users, pos_items, neg_items, embeddings):
        # 计算正则化损失
        user_embeddings = embeddings[users]
        pos_item_embeddings = embeddings[self.num_user + pos_items]
        neg_item_embeddings = embeddings[self.num_user + neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items, build_item_graph):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeddings = self.forward(build_item_graph)

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, embeddings)
        image_contrastive_loss = self.batched_contrastive_loss(self.image_item_embeds, self.h)
        text_contrastive_loss = self.batched_contrastive_loss(self.text_item_embeds, self.h)
        contrastive_loss = self.beta * (image_contrastive_loss + text_contrastive_loss)
        total_loss = bpr_loss + reg_loss + contrastive_loss

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