"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/31 9:57
@File : MGAT.py
@function :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, degree, softmax
from torch_geometric.nn.inits import uniform


class GraphGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(GraphGAT, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.dropout = 0.1

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, size=None):
        edge_index = edge_index.long()
        # 移除自循环
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.matmul(x, self.weight)

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_index, size):
        # 计算内积来获得注意力权重.
        x_i = x_i.view(-1, self.out_channels)
        x_j = x_j.view(-1, self.out_channels)
        inner_product = torch.mul(x_i, F.leaky_relu(x_j)).sum(dim=-1)

        # 门控注意力机制
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_i.dtype)
        deg_inv_sqrt = deg[row].pow(-0.5)
        tmp = torch.mul(deg_inv_sqrt, inner_product)
        gate_w = torch.sigmoid(tmp)
        # gate_w = F.dropout(gate_w, p=self.dropout)

        # attention
        tmp = torch.mul(inner_product, gate_w)
        attention_w = softmax(tmp, index=edge_index_i)
        # attention_w = F.dropout(attention_w, p=self.dropout)
        return torch.mul(x_j, attention_w.view(-1, 1))

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GNN(torch.nn.Module):
    def __init__(self, features, edge_index, num_user, num_item, dim_E, dim_latent=None):
        super(GNN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.dim_feat = features.size(1)
        self.edge_index = edge_index
        self.features = features
        self.dim_latent = dim_latent

        # 初始化用户偏好
        self.preference = nn.Embedding(num_user, self.dim_latent)
        nn.init.xavier_normal_(self.preference.weight).cuda()

        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)

            # 第一层的图注意力网络、线性层和门控层
            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, aggr='add')
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_E)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_E)

            # 初始化权重
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            nn.init.xavier_normal_(self.g_layer1.weight)

        # 第二层的图注意力网络、线性层和门控层
        self.conv_embed_2 = GraphGAT(self.dim_E, self.dim_E, aggr='add')
        self.linear_layer2 = nn.Linear(self.dim_E, self.dim_E)
        self.g_layer2 = nn.Linear(self.dim_E, self.dim_E)

        # 初始化权重
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        nn.init.xavier_normal_(self.g_layer2.weight)

        # 第三层的图注意力网络、线性层和门控层
        self.conv_embed_3 = GraphGAT(self.dim_E, self.dim_E, aggr='add')
        self.linear_layer3 = nn.Linear(self.dim_E, self.dim_E)
        self.g_layer3 = nn.Linear(self.dim_E, self.dim_E)

        # 初始化权重
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        nn.init.xavier_normal_(self.g_layer3.weight)

    def forward(self, id_embedding):
        temp_features = torch.tanh(self.MLP(self.features)) if self.dim_latent else self.features
        x = torch.cat((self.preference.weight, temp_features), dim=0)
        x = F.normalize(x).cuda()

        # layer-1
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
        x_1 = F.leaky_relu(self.g_layer1(h) + x_hat)

        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
        x_2 = F.leaky_relu(self.g_layer2(h) + x_hat)

        # layer-3
        h = F.leaky_relu(self.conv_embed_3(x_2, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer3(x_2)) + id_embedding.weight
        x_3 = F.leaky_relu(self.g_layer3(h) + x_hat)

        x = torch.cat((x_1, x_2, x_3), dim=1)

        return x


class MGAT(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight, device):
        super(MGAT, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        # self.v_feat = v_feat
        # self.t_feat = t_feat
        self.dim_E = dim_E
        self.device = device
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight

        # 转置并设置为无向图
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        self.v_feat = v_feat.clone().detach().requires_grad_(True).to(self.device)
        self.t_feat = t_feat.clone().detach().requires_grad_(True).to(self.device)
        self.v_gnn = GNN(self.v_feat, self.edge_index, num_user, num_item, dim_E, dim_latent=256)
        self.t_gnn = GNN(self.t_feat, self.edge_index, num_user, num_item, dim_E, dim_latent=100)

        self.id_embedding = nn.Embedding(num_user + num_item, dim_E)
        nn.init.xavier_normal_(self.id_embedding.weight)

        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))).to(self.device)

    def forward(self):
        v_rep = self.v_gnn(self.id_embedding)
        t_rep = self.t_gnn(self.id_embedding)
        representation = (v_rep + t_rep) / 2
        self.result = representation

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

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeddings = self.forward()

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, embeddings)
        total_loss = bpr_loss + reg_loss

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
