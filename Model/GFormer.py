"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/9/11 9:50
@File : GFormer.py
@function :
"""
import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
import networkx as nx
import multiprocessing as mp
import random

import sys

sys.setrecursionlimit(10000)  # 设置更大的递归深度限制


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


class LocalGraph(nn.Module):

    # 初始化函数，传入图卷积层gtLayer
    def __init__(self, gtLayer, num_user, num_item, dim_E, anchor_set_num, anchorset_id, dists_array):
        super(LocalGraph, self).__init__()
        self.gt_layer = gtLayer  # transformer
        self.sft = torch.nn.Softmax(0)  # 定义softmax操作
        self.device = "cuda:0"  # 设备为cuda
        self.num_users = num_user  # 用户数量
        self.num_items = num_item  # 物品数量
        self.pnn = PNNLayer(dim_E, anchor_set_num).cuda()  # PNN层，用于处理个性化信息
        self.anchorset_id = anchorset_id
        self.dists_array = dists_array

        self.addRate = 0.01  # ratio of nodes to keep

    # 生成噪声的方法，Gumbel噪声用于加到分数上
    def makeNoise(self, scores):
        noise = torch.rand(scores.shape).cuda()  # 生成与分数相同形状的随机噪声
        noise = -torch.log(-torch.log(noise))  # Gumbel噪声公式
        return scores + noise  # 将噪声加到分数上

    # 将稀疏矩阵sp_mat转换为PyTorch稀疏张量
    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)  # 将稀疏矩阵转换为COO格式
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))  # 获取行和列的索引
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()  # 返回PyTorch稀疏张量

    # 合并多个字典为一个字典
    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)  # 更新字典，将所有字典合并为一个
        return result

    # 计算单源最短路径长度，输入为图结构、节点范围和路径截断距离
    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):
        dists_dict = {}
        for node in node_range:
            # 使用NetworkX计算每个节点的最短路径
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
        return dists_dict

    # 并行计算所有节点对的最短路径长度
    def all_pairs_shortest_path_length_parallel(self, graph, cutoff=None, num_workers=1):
        nodes = list(graph.nodes)  # 获取图中的所有节点
        random.shuffle(nodes)  # 随机打乱节点顺序
        if len(nodes) < 50:
            num_workers = int(num_workers / 4)  # 小图减少并行线程数
        elif len(nodes) < 400:
            num_workers = int(num_workers / 2)  # 中等大小的图减少并行线程数
        # num_workers = 1  # 强制在Windows系统上使用单线程

        # 使用单线程计算单源最短路径
        pool = mp.Pool(processes=num_workers)
        results = self.single_source_shortest_path_length_range(graph, nodes, cutoff)
        output = [p.get() for p in results]  # 获取所有结果
        dists_dict = self.merge_dicts(output)  # 合并所有字典
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有进程完成
        return dists_dict

    # 预计算距离数据，输入边的索引和节点数量
    def precompute_dist_data(self, edge_index, num_nodes, approximate=0):
        '''
        这里的距离是1/实际距离，越高表示距离越近，0表示不连通
        '''
        graph = nx.Graph()  # 创建一个NetworkX图
        graph.add_edges_from(edge_index)  # 将边添加到图中

        n = num_nodes  # 节点数量
        # 计算所有节点对的最短路径长度，使用并行加速
        dists_dict = self.all_pairs_shortest_path_length_parallel(graph,
                                                                  cutoff=approximate if approximate > 0 else None)
        dists_array = np.zeros((n, n), dtype=np.int8)  # 初始化距离矩阵

        # 将最短路径长度转换为距离矩阵
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]  # 获取从node_i出发的最短路径
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)  # 获取到node_j的距离
                if dist != -1:
                    dists_array[node_i, node_j] = 1 / (dist + 1)  # 距离越小值越大
        return dists_array  # 返回距离矩阵

    # 前向传播函数
    def forward(self, adj, embeds):
        embeds = self.pnn(self.anchorset_id, self.dists_array, embeds)  # 使用PNN层处理嵌入

        # 获取邻接矩阵的行和列索引
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        # 随机选择部分行和列索引
        tmp_rows = np.random.choice(rows.cpu(), size=[int(len(rows) * self.addRate)])
        tmp_cols = np.random.choice(cols.cpu(), size=[int(len(cols) * self.addRate)])

        # 将随机选择的索引转换为PyTorch张量，并移到指定设备上
        add_cols = torch.tensor(tmp_cols).to(self.device)
        add_rows = torch.tensor(tmp_rows).to(self.device)

        # 将新添加的行列索引与原始的行列索引合并
        newRows = torch.cat([add_rows, add_cols, torch.arange(self.num_users + self.num_items).cuda(), rows])
        newCols = torch.cat([add_cols, add_rows, torch.arange(self.num_users + self.num_items).cuda(), cols])

        # 为新的邻接矩阵创建一个评分矩阵，所有评分为1
        ratings_keep = np.array(torch.ones_like(newRows.cpu().clone().detach()))
        adj_mat = sp.csr_matrix((ratings_keep, (newRows.cpu(), newCols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        # 将稀疏矩阵转换为PyTorch稀疏张量
        add_adj = self.sp_mat_to_sp_tensor(adj_mat).to(self.device)

        # 使用图卷积层处理新的邻接矩阵和嵌入，得到新的嵌入和注意力权重
        embeds_l2, atten = self.gt_layer(add_adj, embeds)

        # 计算边的注意力权重
        att_edge = torch.sum(atten, dim=-1)

        return att_edge, add_adj  # 返回注意力权重和新生成的邻接矩阵


class PNNLayer(nn.Module):
    def __init__(self, dim_E, anchor_set_num):
        super(PNNLayer, self).__init__()
        self.dim_E = dim_E
        self.anchor_set_num = anchor_set_num  # 锚点集数量
        # self.linear_out_position = nn.Linear(self.dim_E, 1)
        # self.linear_out = nn.Linear(self.dim_E, self.dim_E)
        self.linear_hidden = nn.Linear(2 * self.dim_E, self.dim_E)
        # self.act = nn.ReLU()

    def forward(self, anchor_set_id, dists_array, embeds):
        torch.cuda.empty_cache()
        dists_array = torch.tensor(dists_array, dtype=torch.float32).to("cuda:0")
        set_ids_emb = embeds[anchor_set_id]  # 获取锚点集的嵌入向量
        # 通过重复锚点嵌入，使其与距离矩阵形状匹配，然后重塑为三维张量
        # 重复后的形状为 (节点数 * 锚点集大小, 锚点集大小, 潜在维度)
        set_ids_reshape = set_ids_emb.repeat(dists_array.shape[1], 1).reshape(-1, len(set_ids_emb), self.dim_E)
        # 将距离矩阵转置并扩展一个维度，以便与嵌入相乘，扩展后的形状为 (节点数, 锚点集大小, 1)
        dists_array_emb = dists_array.T.unsqueeze(2)
        # 计算信息传播（messages），通过锚点嵌入和距离矩阵的乘积生成
        messages = set_ids_reshape * dists_array_emb  # 形状为 (节点数 * 锚点集大小, 锚点集大小, 潜在维度)

        # 将节点本身的嵌入特征重复多次，匹配锚点集的数量，并调整为三维张量
        self_feature = embeds.repeat(self.anchor_set_num, 1).reshape(-1, self.anchor_set_num, self.dim_E)
        messages = torch.cat((messages, self_feature), dim=-1)
        messages = self.linear_hidden(messages).squeeze()

        outposition1 = torch.mean(messages, dim=1)

        return outposition1


class GTLayer(nn.Module):
    def __init__(self, dim_E, head):
        super(GTLayer, self).__init__()
        self.dim_E = dim_E
        self.head = head
        # 定义 Query、Key 和 Value 的变换矩阵，使用参数化的方式，进行维度转换
        self.qTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dim_E, self.dim_E)))
        self.kTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dim_E, self.dim_E)))
        self.vTrans = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dim_E, self.dim_E)))

    # 生成噪声的方法，用于给分数添加随机噪声
    def makeNoise(self, scores):
        noise = torch.rand(scores.shape).cuda()
        noise = -torch.log(-torch.log(noise))  # Gumbel 噪声的公式
        return scores + 0.01 * noise

    def forward(self, adj, embeds):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        # 获取行和列对应的嵌入向量，分别代表每条边的起始节点和终止节点
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        # 通过线性变换计算 Query、Key 和 Value 的嵌入表示，并调整为多头注意力的形状
        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head, self.dim_E // self.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head, self.dim_E // self.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head, self.dim_E // self.head])

        # 计算注意力分数，通过点积计算 Query 和 Key 的相似度
        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)

        # 创建一个全 0 的张量，用于存储归一化的注意力分数
        tem = torch.zeros([adj.shape[0], self.head]).cuda()
        # 将每个节点的注意力分数累加，用于后续的归一化
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        # 将注意力分数归一化
        att = expAtt / (attNorm + 1e-8)

        # 计算加权后的 Value 嵌入，通过注意力分数对 Value 进行加权
        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.dim_E])
        tem = torch.zeros([adj.shape[0], self.dim_E]).cuda()
        # 将结果嵌入累加到对应的行（节点）
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds, att


class RandomMaskSubgraphs(nn.Module):
    def __init__(self, num_users, num_items, reRate, sub, ext, keepRate):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.num_users = num_users
        self.num_items = num_items
        self.device = "cuda:0"
        self.sft = torch.nn.Softmax(1)
        self.reRate = reRate
        self.sub = sub
        self.keepRate = keepRate
        self.ext = ext

    def normalizeAdj(self, adj):
        degree = torch.pow(torch.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return torch.sparse_coo_tensor(adj._indices(), newVals, adj.shape)

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    # 创建子图邻接矩阵
    def create_sub_adj(self, adj, att_edge, flag):
        # 获取邻接矩阵中的用户和物品索引
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]
        # 如果 flag 为 True，则调整 att_edge 中的值，增加一个小的偏移
        if flag:
            att_edge = (np.array(att_edge.detach().cpu() + 0.001))
        else:
            att_f = att_edge
            att_f[att_f > 3] = 3
            att_edge = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))
        att_f = att_edge / att_edge.sum()  # eq11
        # 根据注意力分数，随机选择要保留的边（用户-物品对），保留的比例由 args.sub 控制
        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * self.sub),
                                      replace=False, p=att_f)

        keep_index.sort()
        # 初始化用于标记是否删除边的列表
        drop_edges = []
        i = 0
        j = 0
        # 遍历所有边，标记要删除的边
        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1
        # 获取保留下来的行和列索引
        rows = users_up[keep_index]
        cols = items_up[keep_index]
        # 添加自环到邻接矩阵（对角线元素）
        rows = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), rows])
        cols = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), cols])
        # 创建评分矩阵，所有评分为 1
        ratings_keep = np.array(torch.ones_like(rows.cpu().clone().detach()))
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        return encoderAdj

    def forward(self, adj, att_edge):
        # 获取邻接矩阵中的用户和物品索引
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]

        att_f = att_edge
        att_f[att_f > 3] = 3
        att_f = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))
        att_f1 = att_f / att_f.sum()  # eq11

        # 随机选择一部分用户-物品边，保留比例由 keepRate 控制
        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * self.keepRate),
                                      replace=False, p=att_f1)
        keep_index.sort()
        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), rows])
        cols = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), cols])
        drop_edges = []
        i, j = 0, 0

        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        ratings_keep = np.array(torch.ones_like(rows.cpu().clone().detach()))
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)  # 基本原理

        # 获取被删除的行和列索引
        drop_row_ids = users_up[drop_edges]
        drop_col_ids = items_up[drop_edges]
        # 扩展额外的行列索引（额外选择的边）
        ext_rows = np.random.choice(rows.cpu(), size=[int(len(drop_row_ids) * self.ext)])
        ext_cols = np.random.choice(cols.cpu(), size=[int(len(drop_col_ids) * self.ext)])

        ext_cols = torch.tensor(ext_cols).to(self.device)
        ext_rows = torch.tensor(ext_rows).to(self.device)
        # 将扩展的行和列索引与删除的索引合并
        tmp_rows = torch.cat([ext_rows, drop_row_ids])
        tmp_cols = torch.cat([ext_cols, drop_col_ids])
        # 重新随机选择部分行和列索引
        new_rows = np.random.choice(tmp_rows.cpu(), size=[int(adj._values().shape[0] * self.reRate)])
        new_cols = np.random.choice(tmp_cols.cpu(), size=[int(adj._values().shape[0] * self.reRate)])

        new_rows = torch.tensor(new_rows).to(self.device)
        new_cols = torch.tensor(new_cols).to(self.device)
        # 生成新的行列索引
        newRows = torch.cat([new_rows, new_cols, torch.arange(self.num_users + self.num_items).cuda(), rows])
        newCols = torch.cat([new_cols, new_rows, torch.arange(self.num_users + self.num_items).cuda(), cols])
        # 对行列索引进行哈希，以去除重复
        hashVal = newRows * (self.num_users + self.num_items) + newCols
        hashVal = torch.unique(hashVal)
        newCols = hashVal % (self.num_users + self.num_items)
        newRows = ((hashVal - newCols) / (self.num_users + self.num_items)).long()

        decoderAdj = torch.sparse_coo_tensor(torch.stack([newRows, newCols], dim=0),
                                             torch.ones_like(newRows).cuda().float(),
                                             adj.shape)  # 基本原理补集
        # 创建两个子图的邻接矩阵（sub 和 cmp）
        sub = self.create_sub_adj(adj, att_edge, True)
        cmp = self.create_sub_adj(adj, att_edge, False)

        return encoderAdj, decoderAdj, sub, cmp


class GFormer(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, reg_weight, n_layers, pnn_layer,
                 ssl_reg, b2, ctra, device):
        super(GFormer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.n_layers = n_layers
        self.ssl_reg = ssl_reg
        self.device = device
        self.num_nodes = self.num_user + self.num_item

        self.gtw = 0.1
        self.gcn_layer = n_layers
        self.pnn_layer = pnn_layer
        self.anchor_set_num = 32  # 锚节点数量
        self.head = 4  # number of heads in attention
        self.reRate = 0.8  # ratio of nodes to keep
        self.sub = 0.1  # sub maxtrix
        self.ext = 0.5
        self.keepRate = 0.9
        self.b2 = b2  # lambda1
        self.ssl_reg = ssl_reg  # 1
        self.ctra = ctra

        self.uEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_user, self.dim_E)))
        self.iEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_item, self.dim_E)))

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)
        self.adj = self.get_norm_adj_mat().to(self.device)
        self.allOneAdj = self.makeAllOne()
        self.preSelect_anchor_set()

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gcn_layer)])
        self.gcnLayer = GCNLayer()
        self.gtLayer = GTLayer(self.dim_E, self.head)  # transformer
        self.pnnLayers = nn.Sequential(*[PNNLayer(dim_E, self.anchor_set_num) for i in range(self.pnn_layer)])

        self.masker = RandomMaskSubgraphs(self.num_user, self.num_item, self.reRate, self.sub, self.ext, self.keepRate)
        self.sampler = LocalGraph(self.gtLayer, self.num_user, self.num_item, dim_E, self.anchor_set_num,
                                  self.anchorset_id, self.dists_array)

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

    def makeAllOne(self):
        idxs = self.adj._indices()
        vals = torch.ones_like(self.adj._values())
        shape = self.adj.shape
        return torch.sparse_coo_tensor(idxs, vals, shape).cuda()

    def preSelect_anchor_set(self):
        n = self.num_nodes
        # 随机从所有节点中选择锚点集，大小为 anchor_set_num，且不重复选择
        annchorset_id = np.random.choice(n, size=self.anchor_set_num, replace=False)
        graph = nx.Graph()  # 创建一个新的无向图
        # 添加所有用户和物品节点到图中，节点范围是 0 到 (用户数 + 物品数)
        graph.add_nodes_from(np.arange(self.num_user + self.num_item))

        # 获取稀疏矩阵 self.allOneAdj 的行和列索引，表示图中的边
        rows = self.allOneAdj._indices()[0, :]
        cols = self.allOneAdj._indices()[1, :]

        rows = np.array(rows.cpu())
        cols = np.array(cols.cpu())

        edge_pair = list(zip(rows, cols))  # 将行和列索引配对为边对 (node_i, node_j)
        graph.add_edges_from(edge_pair)
        # 初始化距离矩阵，形状为 (锚点集大小, 节点总数)，用于存储锚点到其他节点的距离
        dists_array = np.zeros((len(annchorset_id), self.num_nodes))
        # 计算每个锚点到所有其他节点的最短路径
        dicts_dict = self.single_source_shortest_path_length_range(graph, annchorset_id)
        for i, node_i in enumerate(annchorset_id):
            shortest_dist = dicts_dict[node_i]  # 获取从锚点 node_i 出发的最短路径距离字典
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)  # 获取从 node_i 到 node_j 的距离，若无路径则返回 -1
                if dist != -1:
                    dists_array[i, j] = 1 / (dist + 1)  # 将距离转换为 1 / (dist + 1)，越近的节点值越大，避免除零

        # 保存距离矩阵和锚点集的 ID
        self.dists_array = dists_array
        self.anchorset_id = annchorset_id  #

    def single_source_shortest_path_length_range(self, graph, node_range):  # 最短路径算法
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff=None)
        return dists_dict

    def getEgoEmbeds(self):
        return torch.cat([self.uEmbeds, self.iEmbeds], dim=0)

    def forward(self, is_test, sub, cmp, encoderAdj, decoderAdj=None):
        # 将用户和物品的嵌入拼接成一个整体的嵌入向量
        embeds = torch.cat([self.uEmbeds, self.iEmbeds], dim=0)
        embedsLst = [embeds]  # 用列表存储不同层次的嵌入表示

        # 通过图卷积层计算子图嵌入（cmp图）
        emb, _ = self.gtLayer(cmp, embeds)
        cList = [embeds, self.gtw * emb]  # 计算cmp嵌入的加权列表

        # 通过图卷积层计算sub图的嵌入
        emb, _ = self.gtLayer(sub, embeds)
        subList = [embeds, self.gtw * emb]  # 计算sub嵌入的加权列表

        # 遍历所有的GCN层，逐层更新嵌入表示
        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoderAdj, embedsLst[-1])  # 通过encoderAdj更新嵌入
            embeds2 = gcn(sub, embedsLst[-1])  # 通过sub图更新嵌入
            embeds3 = gcn(cmp, embedsLst[-1])  # 通过cmp图更新嵌入
            subList.append(embeds2)  # 将sub图的嵌入加入到subList
            embedsLst.append(embeds)  # 将encoder图的嵌入加入到嵌入列表
            cList.append(embeds3)  # 将cmp图的嵌入加入到cList

        # 如果不是测试模式，则通过PNN层进一步处理嵌入
        if is_test is False:
            for i, pnn in enumerate(self.pnnLayers):
                embeds = pnn(self.anchorset_id, self.dists_array, embedsLst[-1])  # 通过PNN层处理
                embedsLst.append(embeds)  # 更新嵌入列表

        # 如果提供了decoderAdj（解码器的邻接矩阵），则继续通过图卷积层处理
        if decoderAdj is not None:
            embeds, _ = self.gtLayer(decoderAdj, embedsLst[-1])  # 通过decoderAdj更新嵌入
            embedsLst.append(embeds)  # 更新嵌入列表

        # 最终嵌入通过各层的嵌入列表相加得到
        embeds = sum(embedsLst)

        # cmp图的嵌入加权和
        cList = sum(cList)

        # sub图的嵌入加权和
        subList = sum(subList)

        # 返回用户嵌入、物品嵌入、cmp图嵌入和sub图嵌入
        return embeds[:self.num_user], embeds[self.num_user:], cList, subList

    def bpr_loss(self, ancEmbeds, posEmbeds, negEmbeds):
        # 计算正向和负向项目的分数
        pos_scores = torch.sum(ancEmbeds * posEmbeds, dim=1)
        neg_scores = torch.sum(ancEmbeds * negEmbeds, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, ancEmbeds, posEmbeds, negEmbeds, usrEmbeds2, itmEmbeds2,
                            ancEmbeds2, posEmbeds2):
        # 计算正则化损失
        reg_loss = self.reg_weight * (torch.mean(ancEmbeds ** 2) + torch.mean(posEmbeds ** 2)
                                      + torch.mean(negEmbeds ** 2) + torch.mean(ancEmbeds2 ** 2)
                                      + torch.mean(posEmbeds2 ** 2))

        return reg_loss

    def contrast(self, nodes, allEmbeds, allEmbeds2=None):
        if allEmbeds2 is not None:
            pckEmbeds = allEmbeds[nodes]
            scores = torch.log(torch.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
        else:
            uniqNodes = torch.unique(nodes)
            pckEmbeds = allEmbeds[uniqNodes]
            scores = torch.log(torch.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
        return scores

    def contrastNCE(self, nodes, allEmbeds, allEmbeds2=None):
        if allEmbeds2 is not None:
            pckEmbeds = allEmbeds[nodes]
            pckEmbeds2 = allEmbeds2[nodes]
            # posScore = t.sum(pckEmbeds * pckEmbeds2)
            scores = torch.log(torch.exp(pckEmbeds * pckEmbeds2).sum(-1)).mean()
            # ssl_score = scores - posScore
        return scores

    def loss(self, users, pos_items, neg_items, encoderAdj, decoderAdj, sub, cmp):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        usrEmbeds, itmEmbeds, cList, subLst = self.forward(False, sub, cmp, encoderAdj, decoderAdj)

        ancEmbeds = usrEmbeds[users]
        posEmbeds = itmEmbeds[pos_items]
        negEmbeds = itmEmbeds[neg_items]

        usrEmbeds2 = subLst[:self.num_user]
        itmEmbeds2 = subLst[self.num_user:]
        ancEmbeds2 = usrEmbeds2[users]
        posEmbeds2 = itmEmbeds2[pos_items]

        bprLoss = (-torch.sum(ancEmbeds * posEmbeds, dim=-1)).mean()  # eq13
        bprLoss2 = self.bpr_loss(ancEmbeds2, posEmbeds2, negEmbeds) / 1024

        regLoss = self.regularization_loss(ancEmbeds, posEmbeds, negEmbeds, usrEmbeds2, itmEmbeds2,
                                           ancEmbeds2, posEmbeds2)

        contrastLoss = (self.contrast(users, usrEmbeds) + self.contrast(pos_items, itmEmbeds)) * self.ssl_reg \
                       + self.contrast(users, usrEmbeds, itmEmbeds) + self.ctra * self.contrastNCE(users, subLst, cList)

        loss = bprLoss + regLoss + contrastLoss + self.b2 * bprLoss2

        return loss

    def gene_ranklist(self, topk=50, batch_size=100):
        # step需要小于用户数量才能达到分批的效果不然会报错
        usrEmbeds, itmEmbeds, _, _ = self.forward(True, self.adj, self.adj, self.adj)

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        num_users = usrEmbeds.shape[0]
        # 按照批次大小分块处理用户
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_usrEmbeds = usrEmbeds[start:end]

            # 生成评分矩阵，仅针对当前批次的用户
            score_matrix = torch.matmul(batch_usrEmbeds, itmEmbeds.t())

            # 将历史交互设置为极小值，确保这些物品不会出现在推荐结果中
            for row, col in self.user_item_dict.items():
                if start <= row < end:
                    col = torch.LongTensor(list(col)) - self.num_user
                    score_matrix[row - start][col] = 1e-6

            # 选出当前批次用户的 top-k 个物品
            _, index_of_rank_list_batch = torch.topk(score_matrix, topk)

            # 总的top-k列表
            all_index_of_rank_list = torch.cat(
                (all_index_of_rank_list, index_of_rank_list_batch.cpu() + self.num_user),
                dim=0)

        return all_index_of_rank_list
