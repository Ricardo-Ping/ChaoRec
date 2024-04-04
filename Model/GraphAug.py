"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/4/3 21:44
@File : GraphAug.py
@function :
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_sparse import coalesce
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import scipy.sparse as sp
import numpy as np
from torch.utils.data.dataloader import default_collate


class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        # 初始化函数，keepRate定义了边缘保留的概率。
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        # 前向传播函数，负责执行边缘丢弃操作。
        # adj: 输入的稀疏邻接矩阵。
        # keepRate: 边缘保留的概率。

        vals = adj._values()  # 获取邻接矩阵中所有边的权重。
        idxs = adj._indices()  # 获取边缘对应的索引，即哪些节点之间存在边。
        edgeNum = vals.size()  # 计算总边数。

        # 生成一个随机数组，并根据keepRate计算每条边是否保留。
        mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
        # 使用floor函数确保生成的mask为布尔值，即根据keepRate决定每条边是否保留。

        newVals = vals[mask] / keepRate  # 对保留下来的边的权重进行调整，以保持权重的总和不变。
        newIdxs = idxs[:, mask]  # 根据mask过滤掉不保留的边，只保留选中的边的索引。

        # 返回经过边缘丢弃操作后的新的稀疏邻接矩阵。
        return torch.sparse_coo_tensor(newIdxs, newVals, adj.shape)


class GCNLayer(nn.Module):
    def __init__(self):
        # 构造函数初始化模型层
        super(GCNLayer, self).__init__()
        # 激活函数采用LeakyReLU，其负斜率由args.leaky提供
        self.leaky = 0.5  # slope of leaky relu
        self.act = nn.LeakyReLU(negative_slope=self.leaky)

    def forward(self, adj, embeds):
        # 前向传播函数，执行图卷积操作
        # adj: 输入的邻接矩阵，表示图的结构
        # embeds: 输入的节点特征矩阵

        # 获取邻接矩阵的索引和值
        idxs = adj._indices()
        vals = adj._values()

        # 使用coalesce函数对索引和值进行处理，以确保它们是合并并排序的，
        # 这是进行稀疏矩阵乘法操作前的必要步骤。
        index, value = coalesce(idxs, vals, m=adj.size(0), n=adj.size(1))

        # 使用spmm（稀疏矩阵乘法）函数执行图卷积操作，
        # 将处理过的索引和值与节点特征矩阵相乘，然后通过LeakyReLU激活函数。
        # spmm函数的参数分别是索引、值、邻接矩阵的形状和节点特征矩阵。
        return self.act(spmm(index, value, adj.size(0), adj.size(1), embeds))


class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:  # 遍历所有传入的模块
            self.add_module(str(idx), module)  # 将每个模块添加到当前模块（即ListModule实例）中，并以其索引作为名称
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):  # 检查索引是否在合法范围内
            raise IndexError('index {} is out of range'.format(idx))  # 如果不在合法范围内，则抛出异常
        it = iter(self._modules.values())  # 获取一个迭代器，用于遍历所有添加的模块
        for i in range(idx):  # 遍历直到达到目标索引位置
            next(it)  # 跳过当前模块
        return next(it)  # 返回目标索引处的模块

    def __iter__(self):
        # 返回一个迭代器，用于遍历所有添加的模块
        return iter(self._modules.values())

    def __len__(self):
        # 返回添加的模块数量
        return len(self._modules)


class SparseNGCNLayer(torch.nn.Module):
    """
    多尺度稀疏特征矩阵的GCN层。
    :param in_channels: 输入特征的数量。
    :param out_channels: 输出滤波器的数量。
    :param iterations: 邻接矩阵幂次的迭代次数。
    :param dropout_rate: Dropout比率。
    """

    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels  # 输入特征维度
        self.out_channels = out_channels  # 输出特征维度
        self.iterations = iterations  # 邻接矩阵的幂次迭代次数
        self.dropout_rate = dropout_rate  # dropout比率
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix).cuda()
        torch.nn.init.xavier_uniform_(self.bias).cuda()

    def forward(self, normalized_adjacency_matrix, features):
        # 获取规范化邻接矩阵的索引和值
        adj_index, adj_values = normalized_adjacency_matrix._indices(), normalized_adjacency_matrix._values()

        # 计算特征和权重矩阵的乘积
        base_features = torch.matmul(features, self.weight_matrix.cuda()).cuda()
        base_features = base_features + self.bias.cuda()
        # 应用Dropout
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training).cuda()
        base_features = torch.nn.functional.relu(base_features).cuda()
        # 迭代应用图卷积
        for _ in range(self.iterations - 1):
            base_features = spmm(adj_index,
                                 adj_values,
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features).cuda()
        return base_features


class DenseNGCNLayer(torch.nn.Module):
    """
    多尺度密集特征矩阵的GCN层。
    :param in_channels: 输入特征的数量。
    :param out_channels: 过滤器（输出特征）的数量。
    :param iterations: 邻接矩阵幂次方的次数。
    :param dropout_rate: Dropout率。
    """

    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels  # 输入特征维度
        self.out_channels = out_channels  # 输出特征维度
        self.iterations = iterations  # 邻接矩阵迭代次数
        self.dropout_rate = dropout_rate  # Dropout率
        self.define_parameters()  # 定义权重和偏置
        self.init_parameters()  # 初始化权重和偏置

    def define_parameters(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).cuda()
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).cuda()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        adj_index, adj_values = normalized_adjacency_matrix._indices(), normalized_adjacency_matrix._values()
        base_features = torch.mm(features, self.weight_matrix).cuda()
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)
        for _ in range(self.iterations - 1):
            base_features = spmm(adj_index,
                                 adj_values,
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        return base_features


class MixHopNetwork(torch.nn.Module):
    """
    MixHop网络：通过稀疏化邻域混合来实现高阶图卷积架构。
    :param args: 参数对象。
    :param feature_number: 特征输入数量。
    :param class_number: 目标类别数量。
    """

    def __init__(self, feature_number, class_number):
        super(MixHopNetwork, self).__init__()
        self.feature_number = feature_number  # 输入特征的数量
        self.class_number = class_number  # 目标类别的数量
        self.layers_1 = [200, 200, 200]
        self.layers_2 = [200, 200, 200]
        self.dropout = 0.5
        # 计算每一层的抽象特征数量
        self.abstract_feature_number_1 = sum(self.layers_1)
        self.abstract_feature_number_2 = sum(self.layers_2)
        # 计算每一层的顺序
        self.order_1 = len(self.layers_1)
        self.order_2 = len(self.layers_2)

        # 创建层结构（3个上层卷积层，3个下层卷积层）和密集的最终层。
        # 上层使用SparseNGCNLayer，针对稀疏特征矩阵
        self.upper_layers = [
            SparseNGCNLayer(in_channels=self.feature_number, out_channels=self.layers_1[i - 1], iterations=i,
                            dropout_rate=self.dropout) for i in range(1, self.order_1 + 1)]
        self.upper_layers = ListModule(*self.upper_layers)
        # 下层使用DenseNGCNLayer，针对密集特征矩阵
        self.bottom_layers = [
            DenseNGCNLayer(in_channels=self.abstract_feature_number_1, out_channels=self.layers_2[i - 1],
                           iterations=i, dropout_rate=self.dropout) for i in range(1, self.order_2 + 1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        # 定义全连接层，用于输出最终的类别预测
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number).cuda()

    def forward(self, normalized_adjacency_matrix, features):
        """
        前向传播过程。
        :param normalized_adjacency_matrix: 规范化邻接矩阵。
        :param features: 特征矩阵。
        :return node_emb, predictions: 节点嵌入和标签预测。
        """
        # 将上层的特征进行连接
        abstract_features_1 = torch.cat(
            [self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        # 将下层的特征进行连接
        abstract_features_2 = torch.cat(
            [self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)],
            dim=1)
        # 通过全连接层得到节点嵌入
        node_emb = self.fully_connected(abstract_features_2)
        # 使用log_softmax进行多分类的概率预测
        predictions = torch.nn.functional.log_softmax(node_emb, dim=1).cuda()

        return node_emb, predictions


class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, mlp_edge_model_dim=32):
        super(ViewLearner, self).__init__()
        self.encoder = encoder  # 编码器，用于生成节点嵌入
        self.input_dim = mlp_edge_model_dim  # MLP的输入维度
        # 定义一个MLP，用于计算边的权重
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(self.input_dim * 2, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        ).cuda()
        self.init_emb()

    def init_emb(self):
        # 初始化权重和偏置的函数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def build_prob_neighbourhood(self, edge_wight, temperature=0.1):
        # 生成概率邻接矩阵的函数
        attention = torch.clamp(edge_wight, 0.01, 0.99)

        # 使用RelaxedBernoulli分布进行重参数化采样。RelaxedBernoulli允许使用连续的概率值进行反向传播，
        # 是伯努利分布的一种"软化"形式。采样得到的概率邻接矩阵在[0,1]之间。
        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
                                                     probs=attention).rsample()

        eps = 0.2  # 设置阈值为0.0
        # 通过阈值过滤得到的采样结果，生成掩码。如果采样值大于eps，则掩码为1，否则为0。
        # 这一步相当于将采样得到的概率邻接矩阵二值化，但因为eps为0.0，实际上并没有进行阈值过滤。
        mask = (weighted_adjacency_matrix > eps).detach().float()
        # 使用掩码更新概率邻接矩阵。乘以mask实际上保持了weighted_adjacency_matrix的原始值，
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
        return weighted_adjacency_matrix

    def forward(self, x, edge_index, norm_adjacent_matrix):

        # 通过编码器生成节点嵌入
        node_emb, _ = self.encoder(norm_adjacent_matrix, x)

        src, dst = edge_index[0], edge_index[1]  # 提取边的源节点和目标节点索引
        emb_src = node_emb[src]  # 源节点嵌入
        emb_dst = node_emb[dst]  # 目标节点嵌入

        edge_emb = torch.cat([emb_src, emb_dst], 1)  # 拼接源节点和目标节点嵌入作为边嵌入
        edge_logits = self.mlp_edge_model(edge_emb)  # 通过MLP计算边的权重

        bias = 0.0001  # 添加小的偏置以避免计算问题
        # 对eps进行逻辑变换
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()

        # 将变换后的eps与edge_logits相加并规范化，这个过程类似于加入噪声，以增加模型的鲁棒性和探索能力。
        gate_inputs = (gate_inputs + edge_logits) / 1.0
        # # 通过sigmoid函数将gate_inputs映射到(0,1)范围内，得到每条边存在的概率。
        edge_wight = torch.sigmoid(gate_inputs).squeeze().detach()
        # 调用build_prob_neighbourhood方法生成概率邻接矩阵。
        adj = self.build_prob_neighbourhood(edge_wight, temperature=0.9)
        return node_emb, adj


class GraphAug(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, ssl_temp,
                 ssl_reg, device):
        super(GraphAug, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.ssl_temp = ssl_temp  # 0.05
        self.ssl_reg = ssl_reg  # 0.01
        self.device = device

        self.keepRate = 1.0  # 边缘丢弃率
        self.aug_data = 'ed'  # 数据增强策略
        self.backbone = "mixhop"
        self.IB_size = 32
        self.gen = 2

        # 初始化用户和物品的嵌入，这些嵌入是模型学习的参数
        self.uEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_user, self.dim_E)))
        self.iEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_item, self.dim_E)))

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # 初始化多层GCN网络
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.n_layers)])

        # 初始化边缘丢弃模块，用于图数据增强
        self.edgeDropper = SpAdjDropEdge(self.keepRate)

        if self.backbone == "mixhop":
            self.backbone_gnn = MixHopNetwork(feature_number=self.dim_E, class_number=self.IB_size * 2)
        self.view_learner = ViewLearner(self.backbone_gnn, mlp_edge_model_dim=self.dim_E)

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

    def getEgoEmbeds(self, adjlist, l):
        # 根据给定的邻接矩阵列表和层次获取用户和物品的嵌入
        uEmbeds, iEmbeds = self.forward(adjlist, l)
        return torch.concat([uEmbeds, iEmbeds], dim=0)

    def transsparse(self, mat, edge_index, s):

        idxs = edge_index
        vals = mat

        shape = torch.Size((s, s))
        new_adj = torch.sparse_coo_tensor(idxs, vals, shape).cuda()

        return new_adj

    def forward(self, adjlist, keepRate=1.0, l=1):
        # 使用邻接矩阵列表中的第一个邻接矩阵
        adj = adjlist[0]
        # 初始化嵌入：将用户嵌入和项目嵌入拼接起来
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)
        embedsView1 = None
        embedsView2 = None

        # 用于存储经过每层GCN处理后的嵌入
        embedsLst = [iniEmbeds]
        # 遍历定义的GCN层，逐层更新嵌入
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        # 对所有层的输出嵌入求和，得到最终的嵌入结果
        mainEmbeds = sum(embedsLst)

        # 根据不同的数据增强策略，处理邻接矩阵并更新嵌入
        if keepRate == 1.0 and l == 1:
            # 如果没有特殊的数据增强需求，直接返回当前嵌入
            return mainEmbeds[:self.num_user], mainEmbeds[self.num_user:]

        if keepRate == 1.0 and l == 3:
            # 如果有额外的邻接矩阵，处理额外的视图
            adj1, adj2 = adjlist[1], adjlist[2]
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adj1, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)

            embedsLst = [iniEmbeds]  # 重置嵌入列表
            for gcn in self.gcnLayers:
                embeds = gcn(adj2, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)
            return mainEmbeds[:self.num_user], mainEmbeds[self.num_user:], embedsView1[:self.num_user], \
                embedsView1[self.num_user:], embedsView2[:self.num_user], embedsView2[self.num_user:]

        # 应用边缘丢弃策略
        if self.aug_data == 'ed' or self.aug_data == 'ED':
            adjView1 = self.edgeDropper(adj, keepRate)
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adjView1, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)

            adjView2 = self.edgeDropper(adj, keepRate)
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adjView2, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)

        # 应用随机游走策略
        elif self.aug_data == 'rw' or self.aug_data == 'RW':
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                temadj = self.edgeDropper(adj, keepRate)
                embeds = gcn(temadj, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)

            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                temadj = self.edgeDropper(adj, keepRate)
                embeds = gcn(temadj, embedsLst[-1])
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)

        # 应用节点丢弃策略
        elif self.aug_data == 'nd' or self.aug_data == 'ND':
            # 随机数小于keepRate的节点将被保留，其他节点将被丢弃
            rdmMask = (torch.rand(iniEmbeds.shape[0]) < keepRate) * 1.0
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                # 使用节点掩码与上一层的嵌入相乘，实现节点丢弃
                embeds = gcn(adj, embedsLst[-1] * rdmMask)
                embedsLst.append(embeds)
            embedsView1 = sum(embedsLst)

            rdmMask = (torch.rand(iniEmbeds.shape[0]) < keepRate) * 1.0
            embedsLst = [iniEmbeds]
            for gcn in self.gcnLayers:
                embeds = gcn(adj, embedsLst[-1] * rdmMask)
                embedsLst.append(embeds)
            embedsView2 = sum(embedsLst)

        # 返回处理后的嵌入，具体返回内容根据增强策略和层次不同而有所差异
        return mainEmbeds[:self.num_user], mainEmbeds[self.num_user:], embedsView1[:self.num_user], \
            embedsView1[self.num_user:], embedsView2[:self.num_user], embedsView2[self.num_user:]

    def bpr_loss(self, users, pos_items, neg_items, usrEmbeds, itmEmbeds):
        # 获取用户、正向和负向项目的嵌入
        user_embeddings = usrEmbeds[users]
        pos_item_embeddings = itmEmbeds[pos_items]
        neg_item_embeddings = itmEmbeds[neg_items]

        # 计算正向和负向项目的分数
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items):
        # 计算正则化损失
        user_embeddings = self.uEmbeds[users]
        pos_item_embeddings = self.iEmbeds[pos_items]
        neg_item_embeddings = self.iEmbeds[neg_items]

        reg_loss = self.reg_weight * (torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2)
                                      + torch.mean(neg_item_embeddings ** 2))

        return reg_loss

    def reparametrize_n(self, mu, std):
        eps = 0.01
        return mu + eps * std

    def contrastLoss(self, embeds1, embeds2, nodes, temp):
        embeds1 = F.normalize(embeds1, p=2)
        embeds2 = F.normalize(embeds2, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
        return -torch.log(nume / deno).mean()

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        adjlist = {}
        adjlist[0] = self.norm_adj_mat
        shape = self.norm_adj_mat.size()
        o_edge_index, new_edge_attr = self.norm_adj_mat._indices(), self.norm_adj_mat._values()
        ofea = self.getEgoEmbeds(adjlist, 1)

        number = int(100000)  # 随机选择的节点数
        rdmUsrs = torch.randint(self.num_user, [number])  # 随机选择用户节点
        rdmItms1 = torch.randint_like(rdmUsrs, self.num_item)  # 随机选择物品节点
        new_idxs = default_collate([rdmUsrs, rdmItms1])  # 组合用户和物品节点索引
        new_vals = torch.tensor([0.05] * number)  # 为新增的边赋予初始权重

        node_embs = []  # 存储节点嵌入
        for j in range(self.gen):  # 通过视图学习器生成多个图视图
            add_new = torch.sparse_coo_tensor(new_idxs, new_vals, shape).cuda()  # 创建新的稀疏张量
            ant_node, ant_adj = self.view_learner(ofea, o_edge_index, self.norm_adj_mat)  # 通过视图学习器获取节点嵌入和邻接矩阵
            new_adjs = self.transsparse(ant_adj, o_edge_index, ofea.size()[0])  # 将邻接矩阵转换为稀疏格式
            com_adj_ant = new_adjs + add_new  # 将新的边添加到邻接矩阵中
            adjlist[j + 1] = com_adj_ant  # 更新邻接矩阵列表
            node_embs.append(ant_node)  # 存储节点嵌入
        node_embs = torch.mean(torch.stack(node_embs, 0), dim=0)  # 计算节点嵌入的平均值

        # 从节点嵌入中分离出均值和标准差
        mu = node_embs[:, :self.IB_size]
        std = F.softplus(node_embs[:, self.IB_size:] - self.IB_size, beta=1)
        # new_node_embs = self.reparametrize_n(mu, std)  # 重参数化以获取新的节点嵌入(没用到)

        kl_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))  # 计算KL散度损失

        usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward(adjlist, self.keepRate, 3)

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, usrEmbeds, itmEmbeds)
        reg_loss = self.regularization_loss(users, pos_items, neg_items)
        cl_loss = (self.contrastLoss(usrEmbeds1, usrEmbeds2, users, self.ssl_temp) +
                   self.contrastLoss(itmEmbeds1, itmEmbeds2, pos_items, self.ssl_temp)) * self.ssl_reg

        loss = bpr_loss + reg_loss + cl_loss + 0.00001 * kl_loss

        return loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        adjlist = {}
        adjlist[0] = self.norm_adj_mat
        usrEmbeds, itmEmbeds = self.forward(adjlist, 1.0)
        user_tensor = usrEmbeds[:self.num_user].cpu()
        item_tensor = itmEmbeds[:self.num_item].cpu()

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
