"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/4/4 19:57
@File : LGMRec.py
@function :
"""
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()  # 调用父类的构造函数进行初始化

        self.h_layer = n_hyper_layer  # 超图层的数量

    def forward(self, i_hyper, u_hyper, embeds):
        u_ret = None
        i_ret = embeds  # 初始嵌入向量
        for _ in range(self.h_layer):  # 对每一层进行迭代
            lat = torch.mm(i_hyper.T, i_ret)  # 第一步：使用物品到超边的转置矩阵与当前嵌入相乘，计算中间表示
            i_ret = torch.mm(i_hyper, lat)  # 第二步：使用物品到超边的矩阵与上一步的结果相乘，更新物品的嵌入
            u_ret = torch.mm(u_hyper, lat)  # 第三步：使用用户到超边的矩阵与中间表示相乘，计算用户的嵌入
        return u_ret, i_ret  # 返回更新后的用户和物品嵌入


class LGMRec(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E,
                 reg_weight, n_layers, ssl_alpha, device):
        super(LGMRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_mm_layer = 2  # 模态图层数
        self.n_ui_layers = n_layers  # 用户项目图层数
        self.n_hyper_layer = 1  # 超图层数
        self.hyper_num = 4  # 超边数
        self.keep_rate = 0.3  # 保留率，用于Dropout
        self.tau = 0.2  # 用于Gumbel-Softmax的温度参数
        self.ssl_reg = ssl_alpha
        self.device = device
        self.alpha = 0.2

        self.cf_model = 'lightgcn'

        self.n_nodes = self.num_user + self.num_item  # 节点总数（用户数+物品数）

        # 初始化超图神经网络层
        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        # 将交互矩阵转换为稀疏张量格式
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix,
                                                      torch.Size((self.num_user, self.num_item)))
        # 获取规范化的邻接矩阵
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()

        # 计算每个用户的交互次数的倒数，用于调整用户特征
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        # 初始化用户和物品ID的嵌入
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        # 使用Xavier初始化嵌入权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.drop = nn.Dropout(p=1 - self.keep_rate)  # 初始化Dropout层

        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=True)
        self.item_image_trs = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(v_feat.shape[1], self.dim_E)))
        self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(v_feat.shape[1], self.hyper_num)))

        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=True)
        self.item_text_trs = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(t_feat.shape[1], self.dim_E)))
        self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(t_feat.shape[1], self.hyper_num)))

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
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
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))

    # 用户-项目图卷积
    def cge(self):
        cge_embs = None
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs

    # 多模态图卷积
    def mge(self, str='v'):
        item_feats = None
        # 线性转换
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)

        # self.num_inters[:self.num_user]，包含了每个用户交互次数的逆
        # 根据物品特征（item_feats）计算用户特征（user_feats）
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.num_user]
        # user_feats = self.user_embedding.weight
        mge_feats = torch.concat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats

    def forward(self):
        # 构建超边依赖
        # 计算图像模态的超边嵌入
        iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)  # 物品图像特征与超边嵌入的映射
        uv_hyper = torch.mm(self.adj, iv_hyper)  # 用户到物品的图像超边依赖
        # 使用Gumbel-Softmax进行离散化处理，使其更适合梯度下降优化
        iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
        uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)

        # 计算文本模态的超边嵌入
        it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)  # 物品文本特征与超边嵌入的映射
        ut_hyper = torch.mm(self.adj, it_hyper)  # 用户到物品的文本超边依赖
        # 使用Gumbel-Softmax进行离散化处理
        it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
        ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        # 协同图嵌入（CGE）
        cge_embs = self.cge()  # 生成用户和物品的协同图嵌入

        # 模态图嵌入（MGE）
        v_feats = self.mge('v')  # 视觉模态的图嵌入
        t_feats = self.mge('t')  # 文本模态的图嵌入
        # 局部嵌入 = 协同相关嵌入 + 模态相关嵌入
        mge_embs = F.normalize(v_feats) + F.normalize(t_feats)
        lge_embs = cge_embs + mge_embs

        # 全局超图嵌入（GHE）
        uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper),
                                                      cge_embs[self.num_user:])
        ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper),
                                                      cge_embs[self.num_user:])
        # 将视觉和文本的全局嵌入合并
        av_hyper_embs = torch.concat([uv_hyper_embs, iv_hyper_embs], dim=0)
        at_hyper_embs = torch.concat([ut_hyper_embs, it_hyper_embs], dim=0)
        ghe_embs = av_hyper_embs + at_hyper_embs
        # 局部嵌入 + alpha * 全局嵌入
        all_embs = lge_embs + self.alpha * F.normalize(ghe_embs)

        # 将所有嵌入分割为用户嵌入和物品嵌入
        u_embs, i_embs = torch.split(all_embs, [self.num_user, self.num_item], dim=0)

        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs]

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

    def regularization_loss(self, users, pos_items, neg_items, u_g, i_g):
        # 计算正则化损失
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]
        reg_loss = self.reg_weight * (torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2)
                                      + torch.mean(neg_item_embeddings ** 2))

        return reg_loss

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        ua_embeddings, ia_embeddings, hyper_embeddings = self.forward()
        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, ua_embeddings, ia_embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, ua_embeddings, ia_embeddings)

        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + \
                         self.ssl_triple_loss(iv_embs[pos_items], it_embs[pos_items], it_embs)

        loss = bpr_loss + self.ssl_reg * batch_hcl_loss + reg_loss

        return loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        with torch.no_grad():
            user_embs, item_embs, _ = self.forward()
        user_tensor = user_embs[:self.num_user].cpu()
        item_tensor = item_embs[:self.num_item].cpu()

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
