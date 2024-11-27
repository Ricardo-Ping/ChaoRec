"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/10/15 15:04
@File : MHRec.py
@function :
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import math
import scipy.sparse as sp
from arg_parser import parse_args
import os
import torch_sparse

args = parse_args()


class GCNLayer(nn.Module):
    def __init__(self, device):
        super(GCNLayer, self).__init__()
        self.device = device

    def forward(self, adj, embeds, flag=True):
        adj = adj.to(self.device)
        embeds = embeds.to(self.device)
        if flag:
            return torch.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HypergraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 定义线性变换
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # 注意力参数
        self.a = nn.Parameter(torch.Tensor(2 * out_dim, 1))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, H, X):
        # H: 超图的关联矩阵 (num_nodes x num_hyperedges)，为稀疏张量
        # X: 节点特征 (num_nodes x in_dim)
        # X = self.W(X)  # 线性变换，得到节点的新的特征表示

        # 计算超边的嵌入
        E = torch.sparse.mm(H.transpose(0, 1), X)  # 超边的特征表示

        # 获取非零元素的索引
        H = H.coalesce()
        indices = H.indices()  # 大小为 (2, 非零元素数量)
        values = H.values()  # 非零元素的值

        node_indices = indices[0]  # 节点索引
        hyperedge_indices = indices[1]  # 超边索引

        X_i = X[node_indices]  # 节点的特征
        E_j = E[hyperedge_indices]  # 超边的特征

        # 计算注意力系数
        concat = torch.cat([X_i, E_j], dim=1)  # 连接节点和超边的特征
        # e = self.leakyrelu(torch.matmul(concat, self.a)).squeeze()  # 注意力能量
        e = torch.matmul(concat, self.a).squeeze()

        # 对每个节点的注意力系数进行归一化
        e_exp = torch.exp(e)
        node_attention_sums = torch.zeros(X.size(0), device=X.device)
        node_attention_sums = node_attention_sums.index_add(0, node_indices, e_exp)
        node_attention_sums_nz = node_attention_sums[node_indices] + 1e-16
        alpha = e_exp / node_attention_sums_nz  # 归一化的注意力系数

        # 计算消息传递
        m = alpha.unsqueeze(-1) * E_j  # 消息
        X_out = torch.zeros_like(X)
        X_out = X_out.index_add(0, node_indices, m)  # 聚合消息

        return X_out


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        # 输入和输出维度的列表，以及时间嵌入维度
        self.in_dims = in_dims  # [num_user + num_item, 1000]
        self.out_dims = out_dims  # [1000, num_user + num_item]
        self.time_emb_dim = emb_size  # 64
        self.norm = norm

        # 定义时间嵌入的线性层
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 输入层的维度：将时间嵌入与原始输入数据的第一个维度（如特征维度）相加  [num_user + num_item + 64, 1000]
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims  # [1000, num_user + num_item] 输出层重新变回原来的维度
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
    def training_losses(self, model, x_start, itmEmbeds, model_feats):
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

        # # ==============模态感知信号注入===================
        # # 计算模型嵌入与模型特征之间的点积
        # model_embeds = torch.mm(model_output, model_feats)
        # # 计算ID 嵌入与物品嵌入之间的点积
        # id_embeds = torch.mm(x_start, itmEmbeds)
        #
        # # gc_loss，衡量模型嵌入和ID 嵌入之间的差异(eq14中的msi损失)
        # gc_loss = self.mean_flat((model_embeds - id_embeds) ** 2)

        return diff_loss

    def mean_flat(self, tensor):
        # 计算张量 tensor 除了第一维度外的均值
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    # 计算扩散过程中的信噪比
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        # SNR = alpha_t / (1 - alpha_t)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


class MHRec(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight, ii_topk,
                 uu_topk, num_hypernodes, n_layers, h_layers, ssl_temp, ssl_alpha, beta1, beta2, device):
        super(MHRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.device = device
        self.item_topk = ii_topk  # 项目-项目图topk
        self.user_topk = uu_topk  # 用户-用户图topk
        self.hyperedges_visual = None
        self.hyperedges_textual = None
        self.n_layers = n_layers  # 图卷积层数
        self.h_layers = h_layers  # 超图卷积层数
        self.num_hypernodes = num_hypernodes  # 重建超图之后每条超边包含节点的数量
        self.beta1 = beta1  # 超图嵌入+图卷积嵌入的系数
        self.beta2 = beta2  # 多模态嵌入融合的系数
        # 扩散模型参数
        self.steps = 5
        self.noise_scale = 0.1
        self.noise_min = 0.0001
        self.noise_max = 0.02
        # 对比学习参数
        self.ssl_temp = ssl_temp  # 温度系数
        self.ssl_alpha = ssl_alpha

        # 初始化多模态特征
        self.v_feat = v_feat.clone().detach().to(self.device)
        self.t_feat = t_feat.clone().detach().to(self.device)

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.dim_E)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.dim_E)
        nn.init.xavier_uniform_(self.image_trs.weight)
        nn.init.xavier_uniform_(self.text_trs.weight)

        # 模态权重分配
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))  # 两个模态的权重均分
        self.softmax = nn.Softmax(dim=0)

        # 初始化user嵌入
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_embedding.weight)
        # 初始化用户的视觉和文本嵌入
        self.user_visual_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_visual_embedding.weight)
        self.user_textual_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_textual_embedding.weight)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # 初始化图卷积层数
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.device) for i in range(self.n_layers)])
        self.hypergraphLayersVisual = nn.Sequential(
            *[HypergraphAttentionLayer(self.dim_E, self.dim_E) for _ in range(self.h_layers)]
        )
        self.hypergraphLayersTextual = nn.Sequential(
            *[HypergraphAttentionLayer(self.dim_E, self.dim_E) for _ in range(self.h_layers)]
        )

        # ========================用户-项目图(U-I)============================
        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.user_item_graph = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                              (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                             shape=(self.num_user, self.num_item), dtype=np.float32)
        self.adj = self.get_norm_adj_mat().to(self.device)  # 归一化的邻接矩阵

        # 数据集路径
        dataset = args.data_path
        self.dir_str = './Data/' + dataset

        self.pre_processing()

        # 扩散模型网络部分
        dims = '[1000]'
        out_dims = eval(dims) + [num_user + num_item]  # [1000, num_user + num_item]
        in_dims = out_dims[::-1]  # [num_user + num_item, 1000]
        norm = False
        d_emb_size = 10
        self.denoise_model_image = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)
        self.denoise_model_text = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)

        self.diffusion_model = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps).to(
            self.device)

    def get_norm_adj_mat(self):
        # 创建一个空的稀疏矩阵A，大小为(n_users+n_items) x (n_users+n_items)，数据类型为float32
        A = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        # 加载交互矩阵
        inter_M = self.user_item_graph
        # 将交互矩阵转置，以便获取物品到用户的关系
        inter_M_t = self.user_item_graph.transpose()
        # 将用户到物品的交互和物品到用户的交互合并到邻接矩阵A中
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        # 更新稀疏矩阵A的数据
        A._update(data_dict)

        # 归一化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        rows_and_cols = np.array([row, col])  # 将行和列的列表转换成numpy数组
        i = torch.tensor(rows_and_cols, dtype=torch.long)  # 从numpy数组创建张量
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        return SparseL

    def construct_hypergraph(self):
        # 总的节点数量（用户数 + 项目数）
        num_nodes = self.num_user + self.num_item

        # 超边的数量
        num_visual_hyperedges = len(self.hyperedges_visual)
        num_textual_hyperedges = len(self.hyperedges_textual)

        # ======================== 构建视觉模态的超图 =============================
        rows_visual = []
        cols_visual = []
        data_visual = []

        for i, hyperedge in enumerate(self.hyperedges_visual):
            for node in hyperedge:
                rows_visual.append(node)
                cols_visual.append(i)
                data_visual.append(1)

        # 创建稀疏矩阵
        H_visual = sp.coo_matrix((data_visual, (rows_visual, cols_visual)),
                                 shape=(num_nodes, num_visual_hyperedges), dtype=np.float32)
        self.H_visual = H_visual  # (num_nodes, num_hyperedges)

        # ======================== 构建文本模态的超图 =============================
        rows_textual = []
        cols_textual = []
        data_textual = []

        for i, hyperedge in enumerate(self.hyperedges_textual):
            for node in hyperedge:
                rows_textual.append(node)
                cols_textual.append(i)
                data_textual.append(1)

        # 创建稀疏矩阵
        H_textual = sp.coo_matrix((data_textual, (rows_textual, cols_textual)),
                                  shape=(num_nodes, num_textual_hyperedges), dtype=np.float32)
        self.H_textual = H_textual

    def generate_G_from_H(self, H):
        print("生成超图的拉普拉斯矩阵 G...")
        # 顶点度矩阵 D_v
        DV = np.array(H.sum(axis=1)).squeeze()  # 形状: (num_nodes,)
        DV_inv_sqrt = np.power(DV, -0.5, where=DV != 0)
        DV_inv_sqrt[DV == 0] = 0.

        # 超边度矩阵 D_e
        DE = np.array(H.sum(axis=0)).squeeze()  # 形状: (num_hyperedges,)
        DE_inv = np.power(DE, -1.0, where=DE != 0)
        DE_inv[DE == 0] = 0.

        # 超边权重矩阵 W
        W = np.ones(H.shape[1])  # 可以根据需求调整每条超边的权重
        W = sp.diags(W)  # 将其转换为稀疏对角矩阵

        # D_v^{-1/2} * H
        Dv_inv_sqrt = sp.diags(DV_inv_sqrt)
        H_normalized = Dv_inv_sqrt @ H @ W  # 引入 W

        # H_normalized * D_e^{-1}
        De_inv = sp.diags(DE_inv)
        H_normalized = H_normalized @ De_inv

        # G = H_normalized * H_normalized^T
        G = H_normalized @ H_normalized.transpose().tocsr()

        # 将稀疏矩阵 G 转换为 torch.sparse.FloatTensor
        G = G.tocoo()
        indices = torch.from_numpy(np.vstack((G.row, G.col))).long()
        values = torch.from_numpy(G.data).float()
        shape = G.shape
        G = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(self.device)
        print("拉普拉斯矩阵 G 生成完成。")
        return G

    def pre_processing(self):
        # ======================== 用户同构图 (U-U) =============================
        # 加载用户-用户图数据
        self.user_graph_dict = np.load(os.path.join(self.dir_str, 'user_graph_dict.npy'),
                                       allow_pickle=True).item()

        # [num_user, user_topk]
        self.user_user_k_graph = self.topk_sample(self.user_topk)

        # ======================== 项目语义图 (I-I) =============================
        # 处理邻接矩阵文件
        visual_adj_file = os.path.join(self.dir_str, 'ii_visual_{}.pt'.format(self.item_topk))
        textual_adj_file = os.path.join(self.dir_str, 'ii_textual_{}.pt'.format(self.item_topk))

        if os.path.exists(visual_adj_file) and os.path.exists(textual_adj_file):
            # [num_item, item_topk]
            self.item_item_k_visual_graph = torch.load(visual_adj_file)
            self.item_item_k_textual_graph = torch.load(textual_adj_file)
        else:
            image_graph = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.item_item_k_visual_graph = image_graph
            text_graph = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.item_item_k_textual_graph = text_graph
            del image_graph
            del text_graph
            torch.save(self.item_item_k_visual_graph, visual_adj_file)
            torch.save(self.item_item_k_textual_graph, textual_adj_file)

        # ======================== 构建超图序列 =============================
        visual_file_name = 'hyperedges_visual_u{}_i{}.npy'.format(self.user_topk, self.item_topk)
        textual_file_name = 'hyperedges_textual_u{}_i{}.npy'.format(self.user_topk, self.item_topk)

        visual_file_path = os.path.join(self.dir_str, visual_file_name)
        textual_file_path = os.path.join(self.dir_str, textual_file_name)

        if os.path.exists(visual_file_path) and os.path.exists(textual_file_path):
            # 如果超边文件存在，直接加载
            self.hyperedges_visual = np.load(visual_file_path, allow_pickle=True).tolist()
            self.hyperedges_textual = np.load(textual_file_path, allow_pickle=True).tolist()
        else:
            # 如果超边文件不存在，重新构建并保存
            self.hyperedges_visual = []  # [len(edge_index), 2 + self.user_topk + self.item_topk]
            self.hyperedges_textual = []

            for u_i in self.edge_index:
                u = u_i[0]  # 用户索引
                i = u_i[1]  # 原始项目索引（从 num_user 到 num_user + num_item -1）
                adjusted_item_index = i - self.num_user  # 调整为从 0 开始的项目索引

                # 获取用户 u 的相似用户
                similar_users = self.user_user_k_graph[u]  # 相似用户列表

                # 获取项目 i 的相似项目
                similar_items_visual = self.item_item_k_visual_graph[adjusted_item_index]  # Tensor of indices
                similar_items_textual = self.item_item_k_textual_graph[adjusted_item_index]

                # 调整项目索引，加上 self.num_user 以区分用户和项目
                hyperedge_visual = [u] + similar_users + [i] + (similar_items_visual + self.num_user).tolist()
                hyperedge_textual = [u] + similar_users + [i] + (similar_items_textual + self.num_user).tolist()

                self.hyperedges_visual.append(hyperedge_visual)
                self.hyperedges_textual.append(hyperedge_textual)

            # 保存超边文件
            np.save(visual_file_path, self.hyperedges_visual)
            np.save(textual_file_path, self.hyperedges_textual)

        # 构建超图
        # self.construct_hypergraph()

    def topk_sample(self, k):
        # 保存每个用户的最多k个相邻用户的索引
        user_graph_index = []
        count_num = 0
        # 如果某个用户没有足够的邻居，这个列表将被用作占位符
        tasike = [0] * k

        for i in range(self.num_user):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                user_graph_index.append(user_graph_sample)
                continue

            # 如果邻居数大于等于k，直接取前k个邻居
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_index.append(user_graph_sample)

        return user_graph_index

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        # 将对角线上的值设为负无穷，防止选择自己为最近邻
        sim.fill_diagonal_(-float('inf'))
        _, knn_ind = torch.topk(sim, self.item_topk, dim=-1)

        return knn_ind

    def getItemEmbeds(self):
        return self.item_embedding.weight

    def getUserEmbeds(self):
        return self.user_embedding.weight

    def getUserEmbeds_visual(self):
        return self.user_visual_embedding.weight

    def getUserEmbeds_textual(self):
        return self.user_textual_embedding.weight

    def getImageFeats(self):
        v_embedding = self.image_trs(self.image_embedding.weight)
        return v_embedding

    def getTextFeats(self):
        t_embedding = self.text_trs(self.text_embedding.weight)
        return t_embedding

    def forward(self):
        # 获取项目的模态嵌入
        v_embedding = self.image_trs(self.image_embedding.weight)  # (num_item, dim_E)
        t_embedding = self.text_trs(self.text_embedding.weight)  # (num_item, dim_E)

        # 通过 softmax 对模态权重进行归一化
        weight = self.softmax(self.modal_weight)

        # 获取用户的嵌入
        user_v_embedding = self.user_visual_embedding.weight  # (num_user, dim_E)
        user_t_embedding = self.user_textual_embedding.weight  # (num_user, dim_E)

        # =================== 视觉超图卷积多层处理 ====================
        embedsImageAdj = torch.cat([user_v_embedding, F.normalize(v_embedding)], dim=0)
        embedsImageAdjLst = [embedsImageAdj]
        for gcn in self.hypergraphLayersVisual:
            embedsImageAdj = gcn(self.G_visual, embedsImageAdjLst[-1])
            embedsImageAdj += embedsImageAdjLst[-1]
            embedsImageAdj = F.dropout(embedsImageAdj, 0.5)
            embedsImageAdjLst.append(embedsImageAdj)
        embedsImage = torch.mean(torch.stack(embedsImageAdjLst), dim=0)

        # 二次卷积
        embedsImage_ = torch.cat([user_v_embedding, F.normalize(v_embedding)], dim=0)
        embedsImage_Lst = [embedsImage_]
        for gcn in self.gcnLayers:
            embedsImage_ = gcn(self.adj, embedsImage_Lst[-1])
            embedsImage_Lst.append(embedsImage_)
        # embedsImage_ = sum(embedsImage_Lst)
        embedsImage_ = torch.mean(torch.stack(embedsImage_Lst), dim=0)
        embedsImage += self.beta1 * embedsImage_  # eq20

        # =================== 文本超图卷积多层处理 ====================
        embedsTextAdj = torch.cat([user_t_embedding, F.normalize(t_embedding)], dim=0)
        embedsTextAdjLst = [embedsTextAdj]
        for gcn in self.hypergraphLayersTextual:
            embedsTextAdj = gcn(self.G_textual, embedsTextAdjLst[-1])
            embedsTextAdj += embedsTextAdjLst[-1]
            embedsTextAdj = F.dropout(embedsTextAdj, 0.5)
            embedsTextAdjLst.append(embedsTextAdj)
        embedsText = torch.mean(torch.stack(embedsTextAdjLst), dim=0)

        # 二次卷积
        embedsText_ = torch.cat([user_t_embedding, F.normalize(t_embedding)], dim=0)
        embedsText_Lst = [embedsText_]
        for gcn in self.gcnLayers:
            embedsText_ = gcn(self.adj, embedsText_Lst[-1])
            embedsText_Lst.append(embedsText_)
        # embedsText_ = sum(embedsText_Lst)
        embedsText_ = torch.mean(torch.stack(embedsText_Lst), dim=0)
        embedsText += self.beta1 * embedsText_

        # 加权多模态特征的融合  eq21
        embedsModal = weight[0] * embedsImage + weight[1] * embedsText

        # 将多模态融合后的嵌入输入到 GCN 层中，进行多层图卷积  eq22
        embeds = torch.concat([self.user_embedding.weight, self.item_embedding.weight])
        embedsLst = [embeds]  # 保存每一层的嵌入
        for gcn in self.gcnLayers:
            embeds = gcn(self.adj, embedsLst[-1])
            embedsLst.append(embeds)
        # embeds = sum(embedsLst)  # 将每一层的嵌入结果相加
        embeds = torch.mean(torch.stack(embedsLst), dim=0)

        # 加入 RIS 正则化项，对最终的嵌入结果进行归一化处理 eq23
        all_embs = embeds + self.beta2 * F.normalize(embedsModal)

        self.result = all_embs

        # 返回用户和项目的最终嵌入
        # return self.u_embeddings, self.i_embeddings
        return all_embs[:self.num_user], all_embs[self.num_user:], embedsImage, embedsText, embeds

    def contrastLoss(self, embeds1, embeds2, nodes):
        embeds1 = F.normalize(embeds1, p=2)
        embeds2 = F.normalize(embeds2, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / self.ssl_temp)
        deno = torch.exp(pckEmbeds1 @ embeds2.T / self.ssl_temp).sum(-1) + 1e-8
        return -torch.log(nume / deno).mean()

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
        # 获取最终的用户和项目嵌入
        user_embeddings = u_g[users]  # (batch_size, dim_E)
        pos_item_embeddings = i_g[pos_items]  # (batch_size, dim_E)
        neg_item_embeddings = i_g[neg_items]  # (batch_size, dim_E)

        # 获取初始化的用户嵌入
        user_initial_embeddings = torch.cat([
            self.user_embedding.weight[users],
            self.user_visual_embedding.weight[users],
            self.user_textual_embedding.weight[users]
        ], dim=1)  # (batch_size, dim_E * 3)

        # 获取初始化的正向项目嵌入
        pos_item_initial_embeddings = torch.cat([
            self.item_embedding.weight[pos_items],
            self.image_trs(self.image_embedding.weight[pos_items]),
            self.text_trs(self.text_embedding.weight[pos_items])
        ], dim=1)  # (batch_size, dim_E * 3)

        # 获取初始化的负向项目嵌入
        neg_item_initial_embeddings = torch.cat([
            self.item_embedding.weight[neg_items],
            self.image_trs(self.image_embedding.weight[neg_items]),
            self.text_trs(self.text_embedding.weight[neg_items])
        ], dim=1)  # (batch_size, dim_E * 3)

        # 计算正则化损失
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2) +
                torch.mean(user_initial_embeddings ** 2) + torch.mean(pos_item_initial_embeddings ** 2) + torch.mean(
            neg_item_initial_embeddings ** 2)
        )

        return reg_loss

    def loss(self, users, pos_items, neg_items, G_visual, G_textual):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        # 生成超图的拉普拉斯矩阵 G
        self.G_visual = G_visual
        self.G_textual = G_textual

        u_embeddings, i_embeddings, embeds_v, embeds_t, embeds_g = self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, u_embeddings, i_embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, u_embeddings, i_embeddings)

        ssl_loss_1 = self.contrastLoss(embeds_g[:self.num_user], embeds_t[:self.num_user], users) * self.ssl_alpha
        ssl_loss_2 = self.contrastLoss(embeds_g[self.num_user:], embeds_v[self.num_user:], pos_items) * self.ssl_alpha
        ssl_loss_3 = self.contrastLoss(embeds_g[:self.num_user], embeds_v[:self.num_user], users) * self.ssl_alpha
        ssl_loss_4 = self.contrastLoss(embeds_g[self.num_user:], embeds_t[self.num_user:], pos_items) * self.ssl_alpha
        ssl_loss = ssl_loss_3 + ssl_loss_4 + ssl_loss_1 + ssl_loss_2

        loss = bpr_loss + reg_loss + ssl_loss

        return loss

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
