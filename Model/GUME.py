"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2025/6/13 20:08
@File : GUME.py
@function :
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
from scipy.sparse import lil_matrix
import random
import json


def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


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


class GUME(nn.Module):
    def __init__(self, n_users, n_items, edge_index, user_item_dict, v_feat, t_feat, dim_E,
                 n_layers, n_ui_layers, um_loss, vt_loss, dataset_path, device):
        super(GUME, self).__init__()
        self.result = None
        self.n_users = n_users
        self.n_items = n_items
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.sparse = True
        self.bm_loss = 0.01
        self.um_loss = um_loss  # 0.1, 0.2
        self.vt_loss = vt_loss  # 0.1, 0.01, 0.001
        self.reg_weight_1 = float(1e-05)
        self.reg_weight_2 = 0.1
        self.bm_temp = 0.2  #
        self.um_temp = 0.2
        self.n_ui_layers = n_ui_layers
        self.embedding_dim = dim_E
        self.knn_k = 10
        self.n_layers = n_layers
        self.dataset_path = dataset_path
        self.device = device
        self.batch_size = 1024

        adjusted_item_ids = edge_index[:, 1] - self.n_users
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.n_users, self.n_items), dtype=np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.extended_image_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_image_user.weight)

        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        # self.dataset_path = os.path.abspath(os.getcwd() + config['data_path'] + config['dataset'])
        # self.data_name = config['dataset']
        #
        # image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        # text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            # if os.path.exists(image_adj_file):
            #     image_adj = torch.load(image_adj_file)
            # else:
            #     image_adj = build_sim(self.image_embedding.weight.detach())
            #     image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
            #                                            norm_type='sym')
            #     torch.save(image_adj, image_adj_file)
            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                   norm_type='sym')
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            # if os.path.exists(text_adj_file):
            #     text_adj = torch.load(text_adj_file)
            # else:
            #     text_adj = build_sim(self.text_embedding.weight.detach())
            #     text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            #     torch.save(text_adj, text_adj_file)
            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            self.text_original_adj = text_adj.cuda()

        #  Enhancing User-Item Graph
        self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter)
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )

        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )

        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

    def find_inter(self, image_adj, text_adj):
        inter_file = os.path.join('Data', self.dataset_path, 'gume_inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim = []
            txt_sim = []
            for i in range(0, len(image_adj._indices()[0])):
                img_id = image_adj._indices()[0][i]
                txt_id = text_adj._indices()[0][i]
                assert img_id == txt_id
                id = img_id.item()
                img_sim.append(image_adj._indices()[1][j].item())
                txt_sim.append(text_adj._indices()[1][j].item())

                if len(img_sim) == 10 and len(txt_sim) == 10:
                    it_inter = list(set(img_sim) & set(txt_sim))
                    inter[id] = [v for v in it_inter if v != id]
                    img_sim = []
                    txt_sim = []

                j += 1

            with open(inter_file, "w") as f:
                json.dump(inter, f)

        return inter

    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)

        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1] * len(sim_rows)

        # item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items, self.n_items), dtype=np.int)
        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items, self.n_items), dtype=int)

        return item_adj

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj

        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()

        self.R = norm_adj_mat[:self.n_users, self.n_users:]

        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def forward(self, adj, train=False):
        #  Encoding Multiple Modalities

        image_item_embeds = torch.multiply(self.item_id_embedding.weight,
                                           self.image_space_trans(self.image_embedding.weight))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight,
                                          self.text_space_trans(self.text_embedding.weight))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)

        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)

        extended_image_embeds = self.conv_ui(adj, self.extended_image_user.weight, explicit_image_item)

        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)

        extended_text_embeds = self.conv_ui(adj, self.extended_text_user.weight, explicit_text_item)

        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2

        # Attributes Separation for Better Integration
        image_weights, text_weights = torch.split(
            self.softmax(
                torch.cat([
                    self.separate_coarse(explicit_image_embeds),
                    self.separate_coarse(explicit_text_embeds)
                ], dim=-1)
            ),
            1,
            dim=-1
        )
        coarse_grained_embeds = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds

        fine_grained_image = torch.multiply(self.image_behavior(extended_id_embeds),
                                            (explicit_image_embeds - coarse_grained_embeds))
        fine_grained_text = torch.multiply(self.text_behavior(extended_id_embeds),
                                           (explicit_text_embeds - coarse_grained_embeds))
        integration_embeds = (fine_grained_image + fine_grained_text + coarse_grained_embeds) / 3

        all_embeds = extended_id_embeds + integration_embeds

        self.result = all_embeds

        if train:
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (
                explicit_image_embeds, explicit_text_embeds)

        return all_embeds

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.n_users
        neg_items = neg_items - self.n_users
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)

        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        explicit_image_embeds, explicit_text_embeds = embeds_3

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(explicit_image_embeds, explicit_text_embeds)

        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        bm_loss = self.bm_loss * (
                self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp) + self.InfoNCE(
            integration_items[pos_items], extended_id_items[pos_items], self.bm_temp))

        al_loss = vt_loss + bm_loss

        extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)

        # Enhancing User Modality Representation
        c_loss = self.InfoNCE(extended_it_user[users], integration_users[users], self.um_temp)
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        reg_loss_2 = self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2

        return bpr_loss + al_loss + um_loss + reg_loss

    def cal_noise_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss

    def align_vt(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)

        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()

        return vt_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.result[:self.n_users].cpu()
        item_tensor = self.result[self.n_users:self.n_users + self.n_items].cpu()

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 生成评分矩阵
        score_matrix = torch.matmul(user_tensor, item_tensor.t())

        # 将历史交互设置为极小值
        for row, col in self.user_item_dict.items():
            col = torch.LongTensor(list(col)) - self.n_users
            score_matrix[row][col] = 1e-6

        # 选出每个用户的 top-k 个物品
        _, index_of_rank_list_train = torch.topk(score_matrix, topk)
        # 总的top-k列表
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.n_users),
            dim=0)

        # 返回三个推荐列表
        return all_index_of_rank_list
