"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/16 20:25
@File : AdaGCL.py
@function :
"""
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
import torch_sparse
import scipy.sparse as sp


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


class AdaGCL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, ssl_temp,
                 ssl_alpha, device):
        super(AdaGCL, self).__init__()
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
        self.ib_reg = 0.01
        # self.gamma = -0.45
        # self.zeta = 1.05

        # 初始化用户和项目嵌入
        self.uEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, dim_E)))
        self.iEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, dim_E)))
        # 初始化图卷积层数
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.device) for i in range(self.n_layers)])

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        encoder = vgae_encoder(self.dim_E, self.device, self.forward_graphcl).to(self.device)
        decoder = vgae_decoder(self.dim_E, self.device, num_user, num_item, reg_weight).to(self.device)
        self.generator_1 = vgae(encoder, decoder).to(self.device)
        self.generator_2 = DenoisingNet(self.getGCN(), self.getEmbeds(), self.dim_E, self.device, num_user, num_item,
                                        reg_weight).to(self.device)
        self.generator_2.set_fea_adj(num_user + num_item, deepcopy(self.norm_adj_mat))

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

    # 图卷积
    def forward_gcn(self, adj):
        # 使用GCN处理图数据
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds[:self.num_user], mainEmbeds[self.num_user:]

    # 图对比
    def forward_graphcl(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    # 使用图生成的邻接矩阵
    def forward_graphcl_(self, generator):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)

        embedsLst = [iniEmbeds]
        count = 0
        for gcn in self.gcnLayers:
            with torch.no_grad():
                adj = generator.generate(x=embedsLst[-1], layer=count)
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
            count += 1
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def loss_graphcl(self, x1, x2, users, items):
        T = self.ssl_temp  # 温度参数，用于调整相似度计算的尺度

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

    def getEmbeds(self):
        self.unfreeze(self.gcnLayers)  # 解冻GCN层，允许其参数在训练中更新
        return torch.concat([self.uEmbeds, self.iEmbeds], dim=0)  # 返回拼接后的用户和物品嵌入

    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True  # 设置参数为可训练

    def getGCN(self):
        return self.gcnLayers  # 返回模型中的GCN层序列

    def generator_generate(self, generator):
        edge_index = [[], []]  # 初始化边索引列表
        adj = deepcopy(self.norm_adj_mat)  # 深拷贝原始的邻接矩阵
        idxs = adj._indices()  # 获取邻接矩阵的索引，即边的起点和终点索引

        with torch.no_grad():  # 禁用梯度计算
            view = generator.generate(self.norm_adj_mat, idxs, adj)  # 调用generator的generate方法生成新的视图

        return view  # 返回新生成的视图

    # def loss(self, users, pos_items, neg_items):
    #     pos_items = pos_items - self.num_user
    #     neg_items = neg_items - self.num_user
    #     users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
    #
    #     data = deepcopy(self.norm_adj_mat)
    #     data1 = self.generator_generate(self.generator_1)
    #
    #     out1 = self.forward_graphcl(data1)
    #     out2 = self.forward_graphcl_(self.generator_2)
    #     loss = self.loss_graphcl(out1, out2, users, pos_items).mean() * self.ssl_alpha
    #
    #     _out1 = self.forward_graphcl(data1)
    #     _out2 = self.forward_graphcl_(self.generator_2)
    #     loss_ib = self.loss_graphcl(_out1, out1.detach(), users, pos_items) + self.loss_graphcl(_out2,
    #                                                                                             out2.detach(),
    #                                                                                             users, pos_items)
    #     loss_ib = loss_ib.mean() * self.ib_reg
    #
    #     usrEmbeds, itmEmbeds = self.forward_gcn(data)
    #     ancEmbeds = usrEmbeds[users]
    #     posEmbeds = itmEmbeds[pos_items]
    #     negEmbeds = itmEmbeds[neg_items]
    #     # 计算BPR损失
    #     pos_scores = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
    #     neg_scores = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
    #     bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))
    #
    #     u_ego_embeddings = self.uEmbeds[users]
    #     pos_ego_embeddings = self.iEmbeds[pos_items]
    #     neg_ego_embeddings = self.iEmbeds[neg_items]
    #     reg_loss = self.reg_weight * (
    #             torch.mean(u_ego_embeddings ** 2) + torch.mean(pos_ego_embeddings ** 2) + torch.mean(
    #         neg_ego_embeddings ** 2))
    #
    #     loss_1 = self.generator_1(deepcopy(self.norm_adj_mat).to(self.device), users, pos_items, neg_items)
    #     loss_2 = self.generator_2(users, pos_items, neg_items, self.ssl_temp)
    #
    #     total_loss = loss + loss_ib + bpr_loss + reg_loss + loss_1 + loss_2
    #
    #     return total_loss

    def loss_1(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        data1 = self.generator_generate(self.generator_1)

        out1 = self.forward_graphcl(data1)
        out2 = self.forward_graphcl_(self.generator_2)
        loss = self.loss_graphcl(out1, out2, users, pos_items).mean() * self.ssl_alpha

        return loss, out1, out2

    def loss_2(self, users, pos_items, neg_items, out1, out2):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        data1 = self.generator_generate(self.generator_1)

        _out1 = self.forward_graphcl(data1)
        _out2 = self.forward_graphcl_(self.generator_2)

        loss_ib = self.loss_graphcl(_out1, out1.detach(), users, pos_items) + self.loss_graphcl(_out2,
                                                                                                out2.detach(),
                                                                                                users, pos_items)
        loss_ib = loss_ib.mean() * self.ib_reg

        return loss_ib

    def bpr_reg_loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        data = deepcopy(self.norm_adj_mat)
        usrEmbeds, itmEmbeds = self.forward_gcn(data)
        ancEmbeds = usrEmbeds[users]
        posEmbeds = itmEmbeds[pos_items]
        negEmbeds = itmEmbeds[neg_items]
        # 计算BPR损失
        pos_scores = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
        neg_scores = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        u_ego_embeddings = self.uEmbeds[users]
        pos_ego_embeddings = self.iEmbeds[pos_items]
        neg_ego_embeddings = self.iEmbeds[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(u_ego_embeddings ** 2) + torch.mean(pos_ego_embeddings ** 2) + torch.mean(
            neg_ego_embeddings ** 2))

        return bpr_loss + reg_loss

    def gen_loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        loss_1 = self.generator_1(deepcopy(self.norm_adj_mat).to(self.device), users, pos_items, neg_items)
        loss_2 = self.generator_2(users, pos_items, neg_items, self.ssl_temp)

        return loss_1 + loss_2

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错

        usrEmbeds, itmEmbeds = self.forward_gcn(self.norm_adj_mat)
        # 用户嵌入和项目嵌入
        user_tensor = usrEmbeds.cpu()
        item_tensor = itmEmbeds.cpu()

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
        self.bceloss = nn.BCELoss(reduction='none')  # 二元交叉熵损失函数，不进行损失的平均或求和

    # 前向传播方法
    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        # 将输入的隐含表示分为用户和物品两部分
        x_user, x_item = torch.split(x, [self.num_user, self.num_item], dim=0)

        # 预测正样本边和负样本边的存在概率
        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        # 计算正样本和负样本边的重构损失
        # 用于比较每条边存在的预测概率与实际存在（标签为1）或不存在（标签为0）之间的差异
        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).to(self.device))
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(self.device))
        loss_rec = loss_edge_pos + loss_edge_neg  # 总的重构损失

        # 计算KL散度损失
        kl_divergence = -0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        # 计算BPR损失
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        pos_scores = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
        neg_scores = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        # 计算正则化损失
        reg_loss = calcRegLoss(encoder) * self.reg_weight

        # 综合各部分损失，计算总损失
        beta = 0.1  # KL散度损失的权重
        loss = (loss_rec + beta * kl_divergence.mean() + bpr_loss + reg_loss).mean()

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

        vals = adj._values()  # 获取邻接矩阵中的边的值
        idxs = adj._indices()  # 获取邻接矩阵中边的索引
        edgeNum = vals.size()  # 计算邻接矩阵中边的总数
        edge_pred = edge_pred[:, 0]  # 调整边预测的形状
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)  # 根据预测值确定哪些边被保留

        newVals = vals[mask]  # 根据mask筛选出保留下来的边的值
        newVals = newVals / (newVals.shape[0] / edgeNum[0])  # 根据保留下来的边的数量调整边的值
        newIdxs = idxs[:, mask]  # 根据mask筛选出保留下来的边的索引

        # 返回一个新的稀疏邻接矩阵，该矩阵仅包含预测存在的边
        return torch.sparse_coo_tensor(newIdxs, newVals, adj.shape)


class DenoisingNet(nn.Module):
    def __init__(self, gcnLayers, features, dim_E, device, num_user, num_item, reg_weight):
        super(DenoisingNet, self).__init__()  # 调用父类构造函数初始化模块

        self.device = device
        self.features = features.to(self.device)  # 输入的特征向量

        self.gcnLayers = gcnLayers.to(self.device)  # 图卷积网络层
        self.num_user = num_user
        self.num_item = num_item
        self.reg_weight = reg_weight
        self.lambda0 = 0.0001

        self.edge_weights = []  # 用于存储边的权重
        self.nblayers = []  # 邻居层列表（未使用）
        self.selflayers = []  # 自注意力层列表（未使用）

        self.attentions = []  # 注意力权重列表
        self.attentions.append([])  # 初始化注意力权重列表
        self.attentions.append([])

        hidden = dim_E  # 隐藏层的维度

        # 定义两个邻居层，每层包含一个线性变换和ReLU激活函数
        self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        # 定义两个自注意力层，每层包含一个线性变换和ReLU激活函数
        self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        # 定义两个注意力机制，用于计算注意力权重
        self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden, 1))
        self.attentions_1 = nn.Sequential(nn.Linear(2 * hidden, 1))

    def freeze(self, layer):
        # 冻结指定层的参数，防止在训练过程中更新
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False  # 设置参数不需要梯度

    def get_attention(self, input1, input2, layer=0):
        # 计算输入特征之间的注意力权重
        if layer == 0:
            nb_layer = self.nblayers_0  # 第一层邻居层
            selflayer = self.selflayers_0  # 第一层自注意力层
        if layer == 1:
            nb_layer = self.nblayers_1  # 第二层邻居层
            selflayer = self.selflayers_1  # 第二层自注意力层

        input1 = input1.to(self.device)
        input2 = input2.to(self.device)
        input1 = nb_layer(input1).to(self.device)  # 对第一个输入应用邻居层变换
        input2 = selflayer(input2).to(self.device)  # 对第二个输入应用自注意力层变换

        input10 = torch.concat([input1, input2], dim=1)  # 将两个输入拼接

        # 根据层次选择对应的注意力机制计算权重
        if layer == 0:
            weight10 = self.attentions_0(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)

        return weight10  # 返回注意力权重

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        # 采样方法，用于实现参数的稀疏化
        self.gamma = -0.45
        self.zeta = 1.05
        gamma = self.gamma  # 拉伸参数的下界
        zeta = self.zeta  # 拉伸参数的上界

        if training:
            # 训练模式下，引入随机噪声以模拟分布
            debug_var = 1e-7  # 避免除以0的小常数
            bias = 0.0  # 偏置项，这里未使用
            np_random = np.random.uniform(low=debug_var, high=1.0 - debug_var,
                                          size=np.shape(log_alpha.cpu().detach().numpy()))
            random_noise = bias + torch.tensor(np_random)  # 生成随机噪声
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)  # 将噪声变换到logit空间
            gate_inputs = (gate_inputs.to(self.device) + log_alpha) / beta  # 加上log_alpha并按beta缩放
            gate_inputs = torch.sigmoid(gate_inputs)  # 应用sigmoid函数，得到(0, 1)之间的值
        else:
            # 非训练模式下直接使用sigmoid函数处理log_alpha
            gate_inputs = torch.sigmoid(log_alpha)

        # 将sigmoid输出拉伸到(gamma, zeta)区间并裁剪到[0, 1]
        stretched_values = gate_inputs * (zeta - gamma) + gamma
        clipped = torch.clamp(stretched_values, 0.0, 1.0)
        return clipped.float()

    def generate(self, x, layer=0):
        # 根据注意力权重生成稀疏的邻接矩阵
        f1_features = x[self.row, :]  # 提取行节点的特征
        f2_features = x[self.col, :]  # 提取列节点的特征

        weight = self.get_attention(f1_features, f2_features, layer)  # 计算注意力权重

        mask = self.hard_concrete_sample(weight, training=False)  # 通过采样得到稀疏化掩码

        mask = torch.squeeze(mask)  # 移除多余的维度
        adj = torch.sparse_coo_tensor(self.adj_mat._indices(), mask, self.adj_mat.shape)  # 创建新的稀疏邻接矩阵

        ind = deepcopy(adj._indices())  # 深拷贝邻接矩阵的索引
        row = ind[0, :]  # 行索引
        col = ind[1, :]  # 列索引

        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()  # 计算每行的度数
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])  # 计算度数的逆平方根
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)  # 对逆平方根进行裁剪以防止数值问题
        row_inv_sqrt = d_inv_sqrt[row]  # 应用到行索引
        col_inv_sqrt = d_inv_sqrt[col]  # 应用到列索引
        values = torch.mul(adj._values(), row_inv_sqrt)  # 调整边的权重
        values = torch.mul(values, col_inv_sqrt)  # 继续调整边的权重

        support = torch.sparse_coo_tensor(adj._indices(), values, adj.shape)  # 创建调整后的稀疏邻接矩阵

        return support  # 返回调整后的稀疏邻接矩阵

    def l0_norm(self, log_alpha, beta):
        # 计算L0范数，用于实现参数的稀疏化
        gamma = self.gamma
        zeta = self.zeta
        gamma = torch.tensor(gamma)
        zeta = torch.tensor(zeta)
        # 使用sigmoid函数和log_alpha计算每个权重的正则化项
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma / zeta))

        return torch.mean(reg_per_weight)  # 返回所有权重的正则化项的均值

    def set_fea_adj(self, nodes, adj):
        # 设置特征和邻接矩阵
        self.node_size = nodes  # 节点的总数
        self.adj_mat = adj  # 邻接矩阵

        ind = deepcopy(adj._indices())  # 深拷贝邻接矩阵的索引

        self.row = ind[0, :]  # 行索引
        self.col = ind[1, :]  # 列索引

    def call(self, inputs, training=None):
        # 根据输入特征生成去噪后的特征表示
        if training:
            temperature = inputs  # 训练模式下，温度由输入给定
        else:
            temperature = 1.0  # 非训练模式下，温度默认为1.0

        self.masks = []  # 初始化掩码列表

        x = self.features.detach()  # 从输入特征开始
        layer_index = 0  # 当前层索引
        embedsLst = [self.features.detach()]  # 初始化嵌入列表

        for layer in self.gcnLayers:
            f1_features = x[self.row, :]  # 获取行节点特征
            f2_features = x[self.col, :]  # 获取列节点特征

            weight = self.get_attention(f1_features, f2_features, layer=layer_index).to(self.device)  # 计算注意力权重
            mask = self.hard_concrete_sample(weight, temperature, training).to(self.device)  # 通过硬混凝土采样获得掩码

            self.edge_weights.append(weight)  # 存储边的权重
            self.masks.append(mask)  # 存储掩码
            mask = torch.squeeze(mask)  # 移除掩码的多余维度

            # 根据掩码和原始邻接矩阵创建新的稀疏邻接矩阵
            adj = torch.sparse_coo_tensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()

            # 计算归一化系数
            rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
            row_inv_sqrt = d_inv_sqrt[self.row]
            col_inv_sqrt = d_inv_sqrt[self.col]
            values = torch.mul(adj.values(), row_inv_sqrt)
            values = torch.mul(values, col_inv_sqrt)

            support = torch.sparse_coo_tensor(adj._indices(), values, adj.shape).coalesce()

            nextx = layer(support, x, False)  # 应用当前GCN层
            x = nextx  # 更新特征表示
            embedsLst.append(x)  # 将新的特征表示添加到列表中
            layer_index += 1  # 更新层索引
        return sum(embedsLst)  # 返回所有层输出的和

    def lossl0(self, temperature):
        # 计算L0正则化损失
        l0_loss = torch.zeros([]).to(self.device)  # 初始化L0损失
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)  # 累加每个权重的L0正则化损失
        self.edge_weights = []  # 重置边权重列表
        return l0_loss  # 返回总的L0损失

    def forward(self, users, items, neg_items, temperature):
        # 前向传播方法，计算模型的总损失
        self.freeze(self.gcnLayers)  # 冻结GCN层的参数
        x = self.call(temperature, True)  # 生成去噪后的特征表示
        x_user, x_item = torch.split(x, [self.num_user, self.num_item], dim=0)  # 分离用户和物品的特征表示

        # 计算基于特征表示的用户对物品的偏好差异
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        pos_scores = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
        neg_scores = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        reg_loss = calcRegLoss(self) * self.reg_weight  # 计算正则化损失

        lossl0 = self.lossl0(temperature) * self.lambda0  # 计算L0损失

        return bpr_loss + reg_loss + lossl0  # 返回总损失