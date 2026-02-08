import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torch_sparse
from arg_parser import parse_args

args = parse_args()


class GCNLayer(torch.nn.Module):
    def __init__(self, num_user, num_item, num_layer, dim_latent=None, device=None, features=None):
        super(GCNLayer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features, id_embd, adj):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        temp_features = torch.abs(
            ((torch.mul(id_embd, id_embd) + torch.mul(temp_features, temp_features)) / 2) + 1e-8).sqrt()
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        ego_embeddings = x
        all_embeddings = ego_embeddings.to(self.device)
        embeddings_layers = [all_embeddings]

        for layer_idx in range(self.num_layer):
            all_embeddings = torch.sparse.mm(adj, all_embeddings).to(self.device)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1).to(self.device)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings).to(self.device)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0).to(self.device)

        return ui_all_embeddings, self.preference


class User_Graph_sample(torch.nn.Module):
    """
        user-user graph
    """

    def __init__(self, num_user, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class COHESION(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight,
                 dropout, n_layers, mm_layers, ii_topk, mm_image_weight, device):
        super(COHESION, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.n_nodes = self.num_user + self.num_item  # 节点总数
        self.dim = dim_E
        self.dim_feat = dim_E
        self.n_layers = mm_layers
        self.knn_k = ii_topk
        self.mm_image_weight = mm_image_weight
        self.dropout = dropout
        self.k = 40
        self.num_layer = n_layers
        self.drop_rate = 0.1
        self.reg_weight = reg_weight
        self.device = device
        self.user_item_dict = user_item_dict
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.v_rep, self.t_rep, self.id_rep = None, None, None
        self.v_preference, self.t_preference, self.id_preference = None, None, None
        self.dim_latent = 64
        self.mm_adj = None

        # 加载用户-用户图数据
        dataset = args.data_path
        dir_str = './Data/' + dataset
        self.user_graph_dict = np.load(os.path.join(dir_str, 'user_graph_dict.npy'),
                                       allow_pickle=True).item()

        # 项目语义图和多模态特征
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

        self.weight_u = nn.Parameter(
            nn.init.xavier_normal_(
                torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True))
        )
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        # 转置并设置为无向图
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        # 不剪枝的邻接矩阵
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = self.norm_adj

        # 得到用户-项目交互图的边和权重
        self.edge_indices, self.edge_values = self.get_edge_info(self.edge_index_clone)
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)

        # Create GCN layers for different modalities
        self.create_gcn_layers()

        # Create user graph and result embeddings
        self.user_graph = User_Graph_sample(self.num_user, self.dim_latent)
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_nodes, self.dim)))
        ).to(self.device)

    def create_gcn_layers(self):
        if self.v_feat is not None:
            self.v_gcn = GCNLayer(self.num_user, self.num_item, num_layer=self.num_layer, dim_latent=64,
                                  device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCNLayer(self.num_user, self.num_item, num_layer=self.num_layer, dim_latent=64,
                                  device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCNLayer(self.num_user, self.num_item, num_layer=self.num_layer, dim_latent=64,
                               device=self.device, features=self.id_feat)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.num_user + self.num_item,
                           self.num_user + self.num_item), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

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
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.num_user, self.num_item)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.num_user
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.num_user
        return np.column_stack((rows, cols))

    def forward(self, user, pos_items, neg_items):
        user_nodes, pos_item_nodes, neg_item_nodes = user, pos_items, neg_items
        pos_item_nodes += self.num_user
        neg_item_nodes += self.num_user

        # get representation and id_rep_data
        representation, id_rep_data = self.build_representation()

        # get user and item representation
        user_rep, item_rep = self.process_user_item_representation(representation, id_rep_data)

        # get user and item tensor
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]

        # Adaptively optimize the weight of the three modalities
        adaptive_weight = self.adaptive_optimization(user_tensor, pos_item_tensor, neg_item_tensor)
        pos_scores = torch.sum(user_tensor * pos_item_tensor * adaptive_weight, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor * adaptive_weight, dim=1)
        return pos_scores, neg_scores

    def build_representation(self):
        id_rep, id_preference = self.id_gcn(self.id_feat, self.id_feat, self.masked_adj)
        id_rep_data = id_rep.data

        representation = id_rep_data

        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.v_feat, self.id_feat, self.masked_adj)
            representation = torch.cat((id_rep_data, self.v_rep), dim=1)

        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.t_feat, self.id_feat, self.masked_adj)
            representation = torch.cat((id_rep_data, self.t_rep) if representation is None
                                       else (id_rep_data, self.v_rep, self.t_rep), dim=1)

        self.v_rep = torch.unsqueeze(self.v_rep, 2)
        self.t_rep = torch.unsqueeze(self.t_rep, 2)
        id_rep_data = torch.unsqueeze(id_rep_data, 2)

        return representation, id_rep_data

    def process_user_item_representation(self, representation, id_rep_data):
        user_rep, item_rep = None, None

        if self.v_rep is not None and self.t_rep is not None:
            user_rep = torch.cat((id_rep_data[:self.num_user], self.v_rep[:self.num_user], self.t_rep[:self.num_user]),
                                 dim=2)
            user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)

        item_rep = representation[self.num_user:]
        h_i = item_rep
        for i in range(self.n_layers):
            h_i = torch.sparse.mm(self.mm_adj, h_i)
        h_u = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)

        user_rep = user_rep + h_u
        item_rep = item_rep + h_i

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        return user_rep, item_rep

    def adaptive_optimization(self, user_e, pos_e, neg_e):
        pos_score_ = torch.mul(user_e, pos_e).view(-1, 3, self.dim_latent).sum(dim=-1)
        neg_score_ = torch.mul(user_e, neg_e).view(-1, 3, self.dim_latent).sum(dim=-1)
        modality_indicator = 1 - (pos_score_ - neg_score_).softmax(-1).detach()

        adaptive_weight = torch.tile(modality_indicator.view(-1, 3, 1), [1, 1, self.dim_latent])
        adaptive_weight = adaptive_weight.view(-1, 3 * self.dim_latent)

        return adaptive_weight

    def loss(self, users, pos_items, neg_items):

        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        user, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        pos_scores, neg_scores = self.forward(user, pos_items, neg_items)

        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        return loss_value + reg_loss

    def gene_ranklist(self, topk=50):
        # 用户嵌入和项目嵌入
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:self.num_user + self.num_item].cpu()

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

