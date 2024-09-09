"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/9/6 10:54
@File : DiffMM.py
@function :
"""
import torch
from scipy.sparse import coo_matrix
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import math
import scipy.sparse as sp


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate

    def forward(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class DiffMM(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E,
                 reg_weight, n_layers, ssl_alpha, ssl_temp, ris_lambda, e_loss, rebuild_k, device):
        super(DiffMM, self).__init__()
        self.restore_itmEmbeds = None
        self.restore_usrEmbeds = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.device = device
        self.ris_adj_lambda = 0.2
        self.ris_lambda = ris_lambda  # eq23中的𝜔
        self.steps = 5
        self.noise_scale = 0.1
        self.noise_min = 0.0001
        self.noise_max = 0.02
        self.trans = 1
        self.ssl_temp = ssl_temp  # 温度系数
        self.ssl_alpha = ssl_alpha
        self.cl_method = 0  # 0:m vs m ; 1:m vs main
        self.n_layers = n_layers
        self.e_loss = e_loss
        self.rebuild_k = rebuild_k

        # 初始化用户和项目嵌入
        self.uEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, dim_E)))
        self.iEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, dim_E)))
        # 多层 GCN 图卷积层
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(n_layers)])

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.adj = self.get_norm_adj_mat().to(self.device)

        # 稀疏图边的随机丢弃，用于防止过拟合
        keepRate = 0.5
        self.edgeDropper = SpAdjDropEdge(keepRate)

        # 根据trans 的值，初始化不同的多模态特征变换方式
        if self.trans == 1:
            self.image_trans_l = nn.Linear(v_feat.shape[1], self.dim_E)
            self.text_trans_l = nn.Linear(t_feat.shape[1], self.dim_E)
            nn.init.xavier_uniform_(self.image_trans_l.weight)
            nn.init.xavier_uniform_(self.text_trans_l.weight)
        elif self.trans == 0:
            self.image_trans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(v_feat.shape[1], dim_E))))
            self.text_trans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(t_feat.shape[1], dim_E))))

        self.image_embedding = v_feat
        self.text_embedding = t_feat
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))  # 两个模态的权重均分

        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

        dims = '[1000]'
        out_dims = eval(dims) + [num_item]  # [1000, num_item]
        in_dims = out_dims[::-1]  # [num_item, 1000]
        norm = False
        d_emb_size = 10
        self.denoise_model_image = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)
        self.denoise_model_text = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)

        self.diffusion_model = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps).to(self.device)

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
        # SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        return SparseL

    # 创建模态邻接矩阵
    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.num_user, self.num_item), dtype=np.float32)

        a = sp.csr_matrix((self.num_user, self.num_user))
        b = sp.csr_matrix((self.num_item, self.num_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def getItemEmbeds(self):
        return self.iEmbeds

    def getUserEmbeds(self):
        return self.uEmbeds

    def getImageFeats(self):
        if self.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            return image_feats
        elif self.trans == 1:
            image_feats = self.image_trans_l(self.image_embedding)
            return image_feats

    def getTextFeats(self):
        if self.trans == 0:
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
            return text_feats
        elif self.trans == 1:
            text_feats = self.text_trans_l(self.text_embedding)
            return text_feats

    def forward_MM(self, image_adj, text_adj):
        # 如果 args.trans == 0（不使用线性层进行转换），则使用 leakyrelu 激活函数对图像和文本特征进行变换
        if self.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))  # 图像特征通过矩阵乘法进行变换
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))  # 文本特征通过矩阵乘法进行变换
        # 如果 args.trans == 1，使用线性层进行特征变换
        elif self.trans == 1:
            image_feats = self.image_trans_l(self.image_embedding)  # 通过线性层变换图像特征
            text_feats = self.text_trans_l(self.text_embedding)  # 通过线性层变换文本特征

        # 通过 softmax 对模态权重进行归一化
        weight = self.softmax(self.modal_weight)

        # 视觉邻接矩阵处理：拼接用户和项目嵌入，进行图卷积
        embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

        # 处理图像特征：拼接用户嵌入和标准化后的图像特征，并进行图卷积
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(self.adj, embedsImage)

        # 再次更新图像特征，使用从用户嵌入和项目嵌入的组合进行图卷积
        embedsImage_ = torch.concat([embedsImage[:self.num_user], self.iEmbeds])
        embedsImage_ = torch.spmm(self.adj, embedsImage_)
        embedsImage += embedsImage_  # eq20

        # 文本邻接矩阵处理：拼接用户和项目嵌入，进行图卷积
        embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

        # 处理文本特征：拼接用户嵌入和标准化后的文本特征，并进行图卷积
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(self.adj, embedsText)

        # 再次更新文本特征，使用从用户嵌入和项目嵌入的组合进行图卷积
        embedsText_ = torch.concat([embedsText[:self.num_user], self.iEmbeds])
        embedsText_ = torch.spmm(self.adj, embedsText_)
        embedsText += embedsText_

        # 加入 RIS（Residual Information Smoothing）正则化项，对图像、文本的特征进行额外的邻接矩阵处理 eq21
        embedsImage += self.ris_adj_lambda * embedsImageAdj
        embedsText += self.ris_adj_lambda * embedsTextAdj

        # 加权多模态特征的融合  eq21
        embedsModal = weight[0] * embedsImage + weight[1] * embedsText

        # 将多模态融合后的嵌入输入到 GCN 层中，进行多层图卷积  eq22
        embeds = embedsModal
        embedsLst = [embeds]  # 保存每一层的嵌入
        for gcn in self.gcnLayers:
            embeds = gcn(self.adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)  # 将每一层的嵌入结果相加

        # 加入 RIS 正则化项，对最终的嵌入结果进行归一化处理 eq23
        embeds = embeds + self.ris_lambda * F.normalize(embedsModal)

        # 返回用户嵌入和项目嵌入
        return embeds[:self.num_user], embeds[self.num_user:]

    def forward_cl_MM(self, image_adj, text_adj):
        if self.trans == 0:
            # 使用 leakyrelu 激活函数和矩阵乘法转换图像和文本特征
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        elif self.trans == 1:
            # 使用线性层转换图像和文本特征
            image_feats = self.image_trans_l(self.image_embedding)
            text_feats = self.text_trans_l(self.text_embedding)

        # 将用户嵌入和标准化后的图像特征拼接，使用图像邻接矩阵进行图卷积
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(image_adj, embedsImage)

        # 将用户嵌入和标准化后的文本特征拼接，使用文本邻接矩阵进行图卷积
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(text_adj, embedsText)

        # 对图像特征进行多层图卷积处理
        embeds1 = embedsImage
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:  # 遍历 GCN 层，进行图卷积
            embeds1 = gcn(self.adj, embedsLst1[-1])
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)  # 将每一层的图卷积结果相加，形成最终的图像嵌入

        # 对文本特征进行多层图卷积处理
        embeds2 = embedsText
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:  # 遍历 GCN 层，进行图卷积
            embeds2 = gcn(self.adj, embedsLst2[-1])
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)  # 将每一层的图卷积结果相加，形成最终的文本嵌入

        return embeds1[:self.num_user], embeds1[self.num_user:], embeds2[:self.num_user], embeds2[self.num_user:]

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

    def loss(self, users, pos_items, neg_items, image_adj, text_adj):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        usrEmbeds, itmEmbeds = self.forward_MM(image_adj, text_adj)
        self.restore_usrEmbeds = usrEmbeds
        self.restore_itmEmbeds = itmEmbeds
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, usrEmbeds, itmEmbeds)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, usrEmbeds, itmEmbeds)

        usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward_cl_MM(image_adj, text_adj)

        clLoss = (self.contrastLoss(usrEmbeds1, usrEmbeds2, users, self.ssl_temp) +
                  self.contrastLoss(itmEmbeds1, itmEmbeds2, pos_items, self.ssl_temp)) * self.ssl_alpha

        clLoss1 = (self.contrastLoss(usrEmbeds, usrEmbeds1, users, self.ssl_temp) + self.contrastLoss(itmEmbeds, itmEmbeds1, pos_items,
                                                                                       self.ssl_temp)) * self.ssl_alpha
        clLoss2 = (self.contrastLoss(usrEmbeds, usrEmbeds2, users, self.ssl_temp) + self.contrastLoss(itmEmbeds, itmEmbeds2, pos_items,
                                                                                       self.ssl_temp)) * self.ssl_alpha
        clLoss_ = clLoss1 + clLoss2

        if self.cl_method == 1:
            clLoss = clLoss_

        loss = bpr_loss + reg_loss + clLoss

        return loss

    def contrastLoss(self,embeds1, embeds2, nodes, temp):
        embeds1 = F.normalize(embeds1, p=2)
        embeds2 = F.normalize(embeds2, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
        return -torch.log(nume / deno).mean()

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.restore_usrEmbeds[:self.num_user].cpu()
        item_tensor = self.restore_itmEmbeds[:self.num_item].cpu()

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


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        # 输入和输出维度的列表，以及时间嵌入维度
        self.in_dims = in_dims  # [num_item, 1000]
        self.out_dims = out_dims  # [1000, num_item]
        self.time_emb_dim = emb_size  # 64
        self.norm = norm

        # 定义时间嵌入的线性层
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 输入层的维度：将时间嵌入与原始输入数据的第一个维度（如特征维度）相加  [num_item + 64, 1000]
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims  # [1000, num_item] 输出层重新变回原来的维度
        # 定义输入层的多层线性变换（使用 ModuleList 保存多层的 nn.Linear）
        # num_item + 64 >> 1000
        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        # 1000 >> num_item
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))  # Xavier 初始化
            layer.weight.data.normal_(0.0, std)  # 使用正态分布初始化权重
            layer.bias.data.normal_(0.0, 0.001)  # 偏置初始化为一个较小的随机值

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        # 初始化时间嵌入层的权重
        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        # 计算时间嵌入，使用正弦和余弦位置编码
        # torch.arange 生成一个从 0 到 time_emb_dim // 2 的张量用于时间编码
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (
                    self.time_emb_dim // 2)).cuda()

        # 将 timesteps 扩展到相应的维度，并与 freqs 相乘以得到时间嵌入
        temp = timesteps[:, None].float() * freqs[None]

        # 使用 cos 和 sin 函数构造时间嵌入
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)

        # 如果时间嵌入维度是奇数，补齐为偶数维度
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

        # 将时间嵌入通过线性层进行处理  [batchsize, 64]
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)

        # 将输入 x 和时间嵌入 emb 进行拼接，作为输入层的输入  [batchsize, num_item + 64]
        h = torch.cat([x, emb], dim=-1)
        # 依次通过每一层输入层的线性变换，并使用 tanh 激活函数
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        # 依次通过每一层输出层的线性变换，除了最后一层不使用激活函数
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)  # [batchsize, num_item]

        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        # 扩散过程中的噪声相关参数
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps  # 扩散的步数

        # 如果噪声比例不为0，计算每一步的噪声系数 beta
        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    # 计算扩散过程中的 beta 系数，用于在每一步添加噪声
    def get_betas(self):
        start = self.noise_scale * self.noise_min  # 噪声的起始值
        end = self.noise_scale * self.noise_max  # 噪声的结束值
        # 在扩散步数范围内线性插值，得到每一步的方差
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance  # 计算 alpha_bar，用于表示去噪过程中的保持率
        betas = []
        betas.append(1 - alpha_bar[0])  # 初始 beta 值
        # 逐步计算每一步的 beta 值
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
        return np.array(betas)  # 返回 beta 的数组

    # 计算扩散和去噪过程中需要的参数
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas  # alpha 用于表示每一步中去噪后保留的数据比例
        # 计算 alpha 的累积乘积，即 alpha 的逐步积累过程
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()
        # 前一步的 alpha 累积乘积，初始时假设为 1
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        # 下一步的 alpha 累积乘积，最后一步假设为 0
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

        # 计算 alpha 累积乘积的平方根，用于去噪过程中保留的比例
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # 计算 1 - alpha 累积乘积的平方根，用于去噪过程中噪声的比例
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # 计算 log(1 - alpha 累积乘积)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        # 计算 alpha 累积乘积的倒数平方根
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        # 计算 1/alpha 累积乘积 - 1 的平方根，用于后续采样的方差调整
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 计算后验分布的方差，公式来源于扩散模型中后验的推导：
        # betas * (1 - 前一步 alpha 累积乘积) / (1 - 当前步的 alpha 累积乘积)
        #  eq8中的方差
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 计算后验方差的对数，并将第一个元素固定为后续计算方便
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))

        # 计算后验均值的两个系数，分别用于表示在去噪过程中均值的线性组合
        # 系数 1：betas * sqrt(前一步 alpha 累积乘积) / (1 - 当前步 alpha 累积乘积)  eq8和eq10中后面一项的系数
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # 系数 2：(1 - 前一步 alpha 累积乘积) * sqrt(alpha) / (1 - 当前步 alpha 累积乘积)  eq8和eq10中前面一项的系数
        self.posterior_mean_coef2 = (
                    (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    # 从给定的初始状态 x_start 中逐步采样，恢复出原始数据
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        # 如果步数是 0，直接使用初始用户-项目交互序列
        if steps == 0:
            x_t = x_start
        else:
            # 构造一个长度为 x_start 的 t 张量，值为 steps - 1，用于从扩散过程中提取样本
            t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
            # 调用 q_sample 函数，生成带噪声的 x_t
            x_t = self.q_sample(x_start, t)

        # 创建一个索引列表，表示反向采样步骤的顺序，从 steps-1 到 0
        indices = list(range(self.steps))[::-1]

        # 逐步执行从 t = steps-1 到 t = 0 的采样过程
        for i in indices:
            # 为每一个步数创建一个 t 张量
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            # 通过模型计算后验均值和对数方差
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)

            # 如果开启了采样噪声，则加入噪声
            if sampling_noise:
                # 生成与 x_t 形状相同的标准正态噪声
                noise = torch.randn_like(x_t)
                # 确保在时间步t=0时不会加噪声
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                # 更新 x_t，基于模型的均值和噪声
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean  # 如果不加噪声，直接使用均值作为下一步的at-1
        return x_t

    # 执行扩散模型中的前向过程，它在每一步中向数据中加入噪声 eq2
    def q_sample(self, x_start, t, noise=None):
        # x_start代表论文中的a0，表示原始用户项目交互序列
        if noise is None:
            noise = torch.randn_like(x_start)

        # 提取 alpha 的平方根并对 x_start 加权
        alpha_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start

        # 提取 (1 - alpha) 的平方根并对噪声加权
        one_minus_alpha_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        # 返回加权后的结果
        return alpha_t + one_minus_alpha_t

    # 从给定的数组 arr 中提取与时间步 t 对应的值，并扩展维度以适应 broadcast_shape
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.cuda()
        # 根据时间步 t 提取数组中对应的值，并将其转换为浮点数
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    # 根据模型输出和扩散过程中的时间步 t，计算模型的均值和方差  eq4的均值和方差
    def p_mean_variance(self, model, x, t):
        # 使用模型输出，假设模型根据输入 x(at) 和时间步 t 返回结果
        model_output = model(x, t, False)  # 相当于预测初始状态a0
        # 后验分布的方差和对数方差，已经预先计算好
        # model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        # 根据时间步 t 从方差和对数方差中提取对应的值，并扩展到输入 x 的形状
        # model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        # 计算后验均值。通过 posterior_mean_coef1 和 posterior_mean_coef2 加权模型输出和输入 x
        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
                                                x.shape) * model_output + self._extract_into_tensor(
            self.posterior_mean_coef2, t, x.shape) * x)

        # 返回模型均值和对数方差
        return model_mean, model_log_variance

    # ELBO 损失
    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        batch_size = x_start.size(0)
        # 随机选择时间步 ts，范围为 0 到 self.steps
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        # 生成与 x_start 形状相同的随机噪声
        noise = torch.randn_like(x_start)
        # 如果噪声比例不为 0，执行前向扩散过程生成 x_t
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)  # 生成带噪声的用户项目交互序列
        else:
            x_t = x_start

        # 通过模型生成预测输出(去噪过程p)
        model_output = model(x_t, ts)

        # 计算均方误差 MSE，L0部分(eq12)
        mse = self.mean_flat((x_start - model_output) ** 2)

        # 计算 ts-1 和 ts 之间的 SNR 差异，用于权重调节
        # weight 计算了时间步 $t$ 上的 SNR 差异，这反映了不同时间步 KL 散度的加权
        weight = self.SNR(ts - 1) - self.SNR(ts)
        # 如果时间步 ts 为 0，则将权重设置为 1.0（即不衰减）
        weight = torch.where((ts == 0), 1.0, weight)

        # diff_loss 是加权后的 ELBO 损失
        diff_loss = weight * mse

        # ==============模态感知信号注入===================
        # 计算用户模型嵌入与模型特征之间的点积
        usr_model_embeds = torch.mm(model_output, model_feats)
        # 计算用户 ID 嵌入与物品嵌入之间的点积
        usr_id_embeds = torch.mm(x_start, itmEmbeds)

        # gc_loss，衡量用户模型嵌入和用户 ID 嵌入之间的差异(eq14中的msi损失)
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

        return diff_loss, gc_loss

    def mean_flat(self, tensor):
        # 计算张量 tensor 除了第一维度外的均值
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    # 计算扩散过程中的信噪比
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        # SNR = alpha_t / (1 - alpha_t)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])