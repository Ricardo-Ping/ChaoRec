"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/10/8 8:55
@File : Grade.py
@function :
"""
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
import torch_sparse
import scipy.sparse as sp


class vgae_encoder(nn.Module):
    # 类的初始化方法
    def __init__(self, dim_E, device, forward_graphcl):  # 直接传入所需参数
        super(vgae_encoder, self).__init__()
        hidden = dim_E  # 从参数中获取隐藏层的维度
        self.device = device
        self.forward_graphcl = forward_graphcl

        # 定义编码器的均值网络部分，由两个线性层和一个ReLU激活层组成
        # 这个网络负责产生节点嵌入的均值
        self.encoder_mean = nn.Sequential(
            nn.Linear(hidden, hidden),  # 第一个线性变换层
            nn.ReLU(inplace=True),  # ReLU激活函数，inplace参数为True意味着直接在原地修改数据，节省内存
            nn.Linear(hidden, hidden)  # 第二个线性变换层
        )

        # 定义编码器的标准差网络部分，多了一个Softplus激活函数
        # 这个网络负责产生节点嵌入的标准差，Softplus确保标准差是正值
        self.encoder_std = nn.Sequential(
            nn.Linear(hidden, hidden),  # 第一个线性变换层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(hidden, hidden),  # 第二个线性变换层
            nn.Softplus()  # Softplus激活函数，确保输出为正
        )

    # 定义前向传播方法
    def forward(self, adj):
        # 使用Model类中定义的forward_graphcl方法处理邻接矩阵，获得图的初始嵌入
        x = self.forward_graphcl(adj)

        # 计算均值和标准差
        x_mean = self.encoder_mean(x)  # 通过均值网络得到均值
        x_std = self.encoder_std(x)  # 通过标准差网络得到标准差

        # 生成高斯噪声，形状与x_mean相同，然后将其传输到当前设备（CUDA）
        gaussian_noise = torch.randn(x_mean.shape).to(self.device)

        # 根据均值和标准差以及生成的高斯噪声，采样得到隐含表示
        x = gaussian_noise * x_std + x_mean

        # 返回隐含表示、均值和标准差
        return x, x_mean, x_std


class vgae_decoder(nn.Module):
    # 初始化方法
    def __init__(self, dim_E, device, num_user, num_item, reg_weight):  # 直接传入所需参数
        super(vgae_decoder, self).__init__()  # 调用父类的初始化方法

        hidden = dim_E
        self.device = device
        self.num_user = num_user
        self.num_item = num_item
        self.reg_weight = reg_weight

        # 定义解码器网络，用于从隐含表示重构图的边信息
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),  # ReLU激活函数，inplace为True表示原地操作，减少内存消耗
            nn.Linear(hidden, hidden),  # 线性变换层
            nn.ReLU(inplace=True),  # 又一个ReLU激活函数
            nn.Linear(hidden, 1)  # 输出层，输出一个标量值
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，将输出压缩到(0, 1)区间，表示边的存在概率
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')

    # 前向传播方法
    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        # 将输入的隐含表示分为用户和物品两部分
        x_user, x_item = torch.split(x, [self.num_user, self.num_item], dim=0)

        # 预测正样本边和负样本边的存在概率
        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.mse_loss(edge_pos_pred, torch.ones(edge_pos_pred.shape).to(self.device))
        loss_edge_neg = self.mse_loss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(self.device))
        loss_rec = loss_edge_pos + loss_edge_neg  # 总的重构损失

        # 计算KL散度损失
        kl_divergence = -0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        # 综合各部分损失，计算总损失
        beta = 1  # KL散度损失的权重
        loss = (beta * kl_divergence.mean() + loss_rec).mean()

        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder  # 编码器，用于学习图中节点的表示
        self.decoder = decoder  # 解码器，用于重构图或进行其他任务

    def forward(self, data, users, items, neg_items):
        # 前向传播方法，计算模型的损失
        x, x_mean, x_std = self.encoder(data)  # 使用编码器处理输入数据
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)  # 计算解码器的损失
        return loss  # 返回损失值

    def generate(self, data, edge_index, adj):
        # 生成方法，用于根据输入数据生成边的预测
        x, _, _ = self.encoder(data)  # 使用编码器获取节点的嵌入

        # 使用解码器的Sigmoid激活函数预测边的存在概率
        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))
        edge_pred = edge_pred[:, 0]  # 调整边预测的形状

        # 根据预测值确定哪些边被保留
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)
        retained_edge_indices = edge_index[:, mask]  # 保留的索引
        retained_edge_values = edge_pred[mask]  # 保留边的预测值作为权重

        # 构造新的稀疏邻接矩阵
        new_adj = torch.sparse_coo_tensor(retained_edge_indices, retained_edge_values, adj.shape)

        # 归一化新的邻接矩阵
        return self.normalize_adjacency_matrix(new_adj)

    def normalize_adjacency_matrix(self, adj):
        # 转换为SciPy稀疏矩阵并进行归一化处理的步骤与之前相同
        adj = adj.coalesce()
        adj_np = sp.coo_matrix((adj.values().cpu().numpy(), adj.indices().cpu().numpy()), shape=adj.shape)
        # adj_np = adj_np + sp.eye(adj_np.shape[0])
        rowsum = np.array(adj_np.sum(1))
        d_inv_sqrt = np.power(rowsum + 1e-7, -0.5).flatten()  # 使用1e-7避免除以0
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_adj_np = d_mat_inv_sqrt.dot(adj_np).dot(d_mat_inv_sqrt).tocoo()
        indices = torch.from_numpy(np.vstack((normalized_adj_np.row, normalized_adj_np.col)).astype(np.int64))
        values = torch.from_numpy(normalized_adj_np.data.astype(np.float32))
        normalized_adj_torch = torch.sparse_coo_tensor(indices, values, torch.Size(normalized_adj_np.shape))
        return normalized_adj_torch


class GCNLayer(nn.Module):
    # 初始化方法
    def __init__(self, device):
        super(GCNLayer, self).__init__()  # 调用父类的初始化方法
        self.device = device

    # 定义前向传播方法
    def forward(self, adj, embeds, flag=True):
        # adj: 邻接矩阵，表示图的结构
        # embeds: 节点的特征矩阵（或嵌入）
        # flag: 一个标志位，用于控制使用稠密还是稀疏的矩阵乘法
        adj = adj.to(self.device)
        embeds = embeds.to(self.device)
        if flag:
            # 如果flag为True，使用torch.spmm执行稠密矩阵和稀疏矩阵的乘法
            # 这适用于邻接矩阵是稀疏的，而嵌入是稠密的场景
            return torch.spmm(adj, embeds)
        else:
            # 如果flag为False，使用torch_sparse.spmm执行稀疏矩阵乘法
            # 这需要从邻接矩阵中提取索引和值，以及矩阵的形状，适用于更高效的稀疏矩阵运算
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class Grade(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight,
                 n_layers, ssl_temp, ssl_alpha, ssl_temp2, noise_alpha, device):
        super(Grade, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.device = device
        self.ssl_temp = ssl_temp
        self.ssl_alpha = ssl_alpha
        self.ssl_temp2 = ssl_temp2
        self.knn_k = 10
        self.mm_image_weight = 0.5
        self.mm_layers = 1
        self.noise_alpha = noise_alpha

        self.uEmbeds = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_uniform_(self.uEmbeds.weight)
        self.utEmbeds = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_uniform_(self.utEmbeds.weight)
        self.uvEmbeds = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_uniform_(self.uvEmbeds.weight)
        self.iEmbeds = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.iEmbeds.weight)

        # 加载多模态特征
        self.v_feat = nn.Embedding.from_pretrained(v_feat, freeze=True)
        self.t_feat = nn.Embedding.from_pretrained(t_feat, freeze=True)

        # 转化到64维度
        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)

        # 多模态项目-项目图
        indices, image_adj = self.get_knn_adj_mat(self.v_feat.weight.detach())
        indices, text_adj = self.get_knn_adj_mat(self.t_feat.weight.detach())
        self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        del text_adj
        del image_adj

        # 初始化图卷积层数
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.device) for i in range(self.n_layers)])

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # id
        encoder = vgae_encoder(self.dim_E, self.device, self.forward_graphcl).to(self.device)
        decoder = vgae_decoder(self.dim_E, self.device, num_user, num_item, reg_weight).to(self.device)
        self.generator_1 = vgae(encoder, decoder).to(self.device)

        # visual
        visual_encoder = vgae_encoder(self.dim_E, self.device, self.visual_forward_graphcl).to(self.device)
        visual_decoder = vgae_decoder(self.dim_E, self.device, num_user, num_item, reg_weight).to(self.device)
        self.generator_2 = vgae(visual_encoder, visual_decoder).to(self.device)

        # textual
        textual_encoder = vgae_encoder(self.dim_E, self.device, self.textual_forward_graphcl).to(self.device)
        textual_decoder = vgae_decoder(self.dim_E, self.device, num_user, num_item, reg_weight).to(self.device)
        self.generator_3 = vgae(textual_encoder, textual_decoder).to(self.device)

    # 构建邻接矩阵
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
        # 将归一化后的L转换为COO格式的稀疏矩阵，以便后续转换为torch稀疏张量   超哥超哥，晚上去吃饭吗
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

    def buildItemGraph(self, h):
        for i in range(self.mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def get_knn_adj_mat(self, mm_embeddings):
        # 归一化特征向量，使得每个向量的L2范数为1
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        # 计算归一化后的特征向量之间的相似度
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        # 获取每个向量的最近k个邻居的索引
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        # 记录相似度矩阵的大小
        adj_size = sim.size()
        # 释放相似度矩阵的内存
        del sim
        # 构造邻接矩阵的索引
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # 返回邻接矩阵的索引和通过`compute_normalized_laplacian`计算的归一化Laplacian矩阵
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        # 使用邻接矩阵的索引和全为1的值构造稀疏矩阵
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        # 计算每行的和，即每个节点的度数，并加上一个非常小的数避免除以0
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        # 计算度数的-1/2次方，用于归一化
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        # 计算归一化因子
        values = rows_inv_sqrt * cols_inv_sqrt
        # 返回归一化Laplacian矩阵
        return torch.sparse_coo_tensor(indices, values, adj_size)

    # 图卷积
    def forward_gcn(self, adj):
        # 使用GCN处理图数据
        h = self.buildItemGraph(self.iEmbeds.weight)
        iniEmbeds = torch.concat([self.uEmbeds.weight, self.iEmbeds.weight], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds[:self.num_user], mainEmbeds[self.num_user:] + h

    # 图对比
    def forward_graphcl(self, adj):
        h = self.buildItemGraph(self.iEmbeds.weight)
        iEmbeds = self.iEmbeds.weight + h
        iniEmbeds = torch.concat([self.uEmbeds.weight, iEmbeds], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def visual_forward_graphcl(self, adj):
        visual = self.image_trs(self.v_feat.weight)

        h = self.buildItemGraph(visual)
        visual = visual + h
        iniEmbeds = torch.concat([self.uvEmbeds.weight, visual], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def textual_forward_graphcl(self, adj):
        textual = self.text_trs(self.t_feat.weight)

        h = self.buildItemGraph(textual)
        textual = textual + h
        iniEmbeds = torch.concat([self.utEmbeds.weight, textual], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def noise_visual_forward_graphcl(self, adj):
        visual = self.image_trs(self.v_feat.weight)
        iniEmbeds = torch.concat([self.uvEmbeds.weight, visual], dim=0)

        embedsLst = [iniEmbeds]
        delta_noise = torch.rand(embedsLst[-1].shape, device=self.device)
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            noised_embeds = embeds + delta_noise / torch.sqrt(
                torch.tensor(embedsLst[-1].shape[1], dtype=torch.float32))
            embedsLst.append(noised_embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def noise_textual_forward_graphcl(self, adj):
        textual = self.text_trs(self.t_feat.weight)
        iniEmbeds = torch.concat([self.utEmbeds.weight, textual], dim=0)

        embedsLst = [iniEmbeds]
        delta_noise = torch.rand(embedsLst[-1].shape, device=self.device)
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            noised_embeds = embeds + delta_noise / torch.sqrt(
                torch.tensor(embedsLst[-1].shape[1], dtype=torch.float32))
            embedsLst.append(noised_embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def loss_graphcl(self, x1, x2, users, items, ssl_temp):
        T = ssl_temp  # 温度参数，用于调整相似度计算的尺度

        # 分别处理两组输入嵌入，分割用户和物品嵌入
        user_embeddings1, item_embeddings1 = torch.split(x1, [self.num_user, self.num_item], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [self.num_user, self.num_item], dim=0)

        # 归一化嵌入向量，使其在单位球上
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        # 使用F.embedding根据索引提取对应的用户和物品嵌入
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        # 将用户和物品嵌入拼接，形成完整的嵌入向量
        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        # 计算嵌入向量的范数
        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        # 计算两组嵌入之间的相似度矩阵
        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        # 提取对角线上的正样本相似度
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]

        # 计算图对比学习的损失
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)

        return loss

    def generator_generate(self, generator):
        edge_index = [[], []]  # 初始化边索引列表
        adj = deepcopy(self.norm_adj_mat)  # 深拷贝原始的邻接矩阵
        idxs = adj._indices()  # 获取邻接矩阵的索引，即边的起点和终点索引

        with torch.no_grad():  # 禁用梯度计算
            view = generator.generate(self.norm_adj_mat, idxs, adj)  # 调用generator的generate方法生成新的视图

        return view  # 返回新生成的视图

    def loss_1(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        data1 = self.generator_generate(self.generator_1)
        data2 = self.generator_generate(self.generator_2)
        data3 = self.generator_generate(self.generator_3)

        out1 = self.forward_graphcl(data1)
        out2 = self.visual_forward_graphcl(data2)
        out3 = self.textual_forward_graphcl(data3)
        loss = (self.loss_graphcl(out1, out2, users, pos_items, self.ssl_temp).mean() +
                self.loss_graphcl(out1, out3, users, pos_items, self.ssl_temp).mean()) * self.ssl_alpha

        noise_viusal = self.noise_visual_forward_graphcl(data1)
        loss += (self.loss_graphcl(out2, noise_viusal, users, pos_items, self.ssl_temp2).mean()) * self.noise_alpha

        noise_textual = self.noise_textual_forward_graphcl(data1)
        loss += (self.loss_graphcl(out3, noise_textual, users, pos_items, self.ssl_temp2).mean()) * self.noise_alpha

        return loss

    def bpr_reg_loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        data = deepcopy(self.norm_adj_mat)
        self.usrEmbeds, self.itmEmbeds = self.forward_gcn(data)
        ancEmbeds = self.usrEmbeds[users]
        posEmbeds = self.itmEmbeds[pos_items]
        negEmbeds = self.itmEmbeds[neg_items]
        # 计算BPR损失
        pos_scores = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
        neg_scores = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        u_ego_embeddings = self.uEmbeds(users)
        ut_emb = self.utEmbeds(users)
        uv_emb = self.uvEmbeds(users)
        pos_ego_embeddings = self.iEmbeds(pos_items)
        neg_ego_embeddings = self.iEmbeds(neg_items)
        reg_loss = self.reg_weight * (torch.mean(u_ego_embeddings ** 2) + torch.mean(pos_ego_embeddings ** 2)
                                      + torch.mean(neg_ego_embeddings ** 2) + torch.mean(ut_emb ** 2)
                                      + torch.mean(uv_emb ** 2))

        return bpr_loss + reg_loss

    def gen_loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        loss_1 = self.generator_1(deepcopy(self.norm_adj_mat).to(self.device), users, pos_items, neg_items)
        loss_2 = self.generator_2(deepcopy(self.norm_adj_mat).to(self.device), users, pos_items, neg_items)
        loss_3 = self.generator_3(deepcopy(self.norm_adj_mat).to(self.device), users, pos_items, neg_items)

        return loss_1 + loss_3 + loss_2

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        data = deepcopy(self.norm_adj_mat)
        self.usrEmbeds, self.itmEmbeds = self.forward_gcn(data)
        # 用户嵌入和项目嵌入
        user_tensor = self.usrEmbeds.cpu()
        item_tensor = self.itmEmbeds.cpu()

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
