"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/18 21:12
@File : DualVAE.py
@function :
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp

EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class DualVAE(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E, kl_weight, ssl_reg, device):
        super(DualVAE, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.device = device
        self.kl_weight = kl_weight  # (KL) 散度损失的权重因子
        self.ssl_reg = ssl_reg  # 对比损失（Contrastive Loss）的权重因子

        self.k = 25  # 隐变量维度
        self.a = 5  # 方面的数量
        act_fn = "tanh"  # 激活函数的类型  tanh
        self.act_fn = ACT.get(act_fn, None)
        self.likelihood = "pois"  # 似然函数的类型

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        encoder_structure = [20]
        decoder_structure = [20]
        self.user_encoder_structure = [num_item] + encoder_structure
        self.item_encoder_structure = [num_user] + encoder_structure
        self.user_decoder_structure = [self.k] + decoder_structure
        self.item_decoder_structure = [self.k] + decoder_structure

        # 初始化用户和物品的mu参数，代表隐变量的均值
        self.mu_theta = torch.zeros((self.item_encoder_structure[0], self.a, self.k))  # 物品隐变量均值，尺寸为n_users*t*k
        self.mu_beta = torch.zeros((self.user_encoder_structure[0], self.a, self.k))  # 用户隐变量均值，尺寸为n_items*t*k

        # 使用Kaiming初始化方法初始化用户偏好和物品主题的参数  [a,k] 有a个“方面”，每个“方面”有k维表示
        self.user_preferences = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(self.a, self.k), a=np.sqrt(5)))
        self.item_topics = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(self.a, self.k), a=np.sqrt(5)))

        # 初始化用户和物品的theta和beta参数，代表隐变量的实际取值
        # [num_item, a, k]
        self.theta = torch.randn(self.item_encoder_structure[0], self.a, self.k) * 0.01
        # [num_user, a, k]
        self.beta = torch.randn(self.user_encoder_structure[0], self.a, self.k) * 0.01

        # 构建用户编码器网络
        self.user_encoder = nn.Sequential()
        for i in range(len(self.user_encoder_structure) - 1):
            self.user_encoder.add_module(
                "fc{}".format(i),  # 线性层
                nn.Linear(self.user_encoder_structure[i], self.user_encoder_structure[i + 1]),
            )
            self.user_encoder.add_module("act{}".format(i), self.act_fn)  # 激活层
        # 用户隐变量的均值和标准差层
        self.user_mu = nn.Linear(self.user_encoder_structure[-1], self.k)
        self.user_std = nn.Linear(self.user_encoder_structure[-1], self.k)

        # 构建物品编码器网络
        self.item_encoder = nn.Sequential()
        for i in range(len(self.item_encoder_structure) - 1):
            self.item_encoder.add_module(
                "fc{}".format(i),  # 线性层
                nn.Linear(self.item_encoder_structure[i], self.item_encoder_structure[i + 1]),
            )
            self.item_encoder.add_module("act{}".format(i), self.act_fn)  # 激活层
        # 物品隐变量的均值和标准差层
        self.item_mu = nn.Linear(self.item_encoder_structure[-1], self.k)
        self.item_std = nn.Linear(self.item_encoder_structure[-1], self.k)

        # 构建用户解码器网络
        self.user_decoder = nn.Sequential()
        for i in range(len(self.user_decoder_structure) - 1):
            self.user_decoder.add_module(
                "fc_out{}".format(i),  # 线性层
                nn.Linear(self.user_decoder_structure[i], self.user_decoder_structure[i + 1])
            )
            self.user_decoder.add_module("act_out{}".format(i), self.act_fn)  # 激活层

        # 构建物品解码器网络
        self.item_decoder = nn.Sequential()
        for i in range(len(self.item_decoder_structure) - 1):
            self.item_decoder.add_module(
                "fc_out{}".format(i),
                nn.Linear(self.item_decoder_structure[i], self.item_decoder_structure[i + 1])
            )
            self.item_decoder.add_module("act_out{}".format(i), self.act_fn)

        self.aspect_probability = None

    def to(self, device):
        self.beta = self.beta.to(device=self.device)
        self.theta = self.theta.to(device=self.device)
        self.mu_beta = self.mu_beta.to(device=self.device)
        self.mu_theta = self.mu_theta.to(device=self.device)
        return super(DualVAE, self).to(self.device)  # 调用父类的to方法，确保模型的其他部分也移动到指定设备

    def encode_user(self, x):
        h = self.user_encoder(x)  # 通过用户编码器网络处理输入x
        return self.user_mu(h), torch.sigmoid(self.user_std(h))  # 计算隐空间的均值和标准差，标准差通过sigmoid函数确保其为正

    def encode_item(self, x):
        h = self.item_encoder(x)  # 通过物品编码器网络处理输入x
        return self.item_mu(h), torch.sigmoid(self.item_std(h))  # 计算隐空间的均值和标准差，标准差通过sigmoid函数确保其为正

    def decode_user(self, theta, beta):
        theta_hidden = self.user_decoder(theta)  # 通过用户解码器网络处理用户的隐表示
        beta_hidden = self.item_decoder(beta)  # 通过物品解码器网络处理物品的隐表示
        h_hidden = theta_hidden.mm(beta_hidden.t())  # 计算解码后的用户和物品隐表示的乘积
        h_hidden = nn.Tanh()(h_hidden)
        h = theta.mm(beta.t())  # 计算原始用户和物品隐表示的乘积
        if self.likelihood == 'mult':  # 如果似然函数为多项式，则直接返回h加上h_hidden
            return h + h_hidden
        return torch.sigmoid(h + h_hidden)  # 否则，通过sigmoid函数将输出转换为概率

    def decode_item(self, theta, beta):
        theta_hidden = self.user_decoder(theta)
        beta_hidden = self.item_decoder(beta)
        h_hidden = beta_hidden.mm(theta_hidden.t())  # 计算物品和用户解码后隐表示的乘积
        h_hidden = nn.Tanh()(h_hidden)
        h = beta.mm(theta.t())  # 计算原始物品和用户隐表示的乘积
        if self.likelihood == 'mult':  # 如果似然函数为多项式，则直接返回h加上h_hidden
            return h + h_hidden
        return torch.sigmoid(h + h_hidden)  # 否则，通过sigmoid函数将输出转换为概率

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)  # 重新参数化
        return mu + eps * std

    def contrast_loss(self, x, x_):
        # 对输入的x和x_进行L2范数归一化，保证它们在同一尺度上进行比较
        x = F.normalize(x, p=2, dim=-1)
        x_ = F.normalize(x_, p=2, dim=-1)

        # 计算正样本得分，即x和x_之间的点乘，表示它们之间的相似度
        pos_score = torch.sum(torch.mul(x_, x), dim=-1)
        # 对正样本得分进行缩放并应用指数函数，增强模型对正样本相似度的敏感度
        pos_score = torch.exp(pos_score / 0.2)

        # 计算不同方面（aspects）对于一个用户的得分，即通过批量矩阵乘法（bmm）计算x_与x转置之间的相似度
        acl_score = torch.bmm(x_, x.transpose(1, 2))
        acl_score = torch.sum(torch.exp(acl_score / 0.2), dim=-1)

        # 计算不同用户对于一个方面的得分，即x_和x的转置之间的相似度
        ncl_score = torch.bmm(x_.transpose(0, 1), x.transpose(0, 1).transpose(1, 2))
        ncl_score = torch.sum(torch.exp(ncl_score.transpose(0, 1) / 0.2), dim=-1)

        # 计算负样本得分，即不同方面对用户和不同用户对方面的得分之和
        neg_score = acl_score + ncl_score

        # 计算对比损失，即正样本得分与负样本得分的比值的对数
        info_nec_loss = torch.log(pos_score / neg_score)
        # 对损失取平均并取反，最小化这个损失值即最大化正样本得分与负样本得分的比值
        info_nec_loss = -torch.mean(torch.sum(info_nec_loss, dim=-1))
        return info_nec_loss

    def forward(self, x, user=True, beta=None, theta=None):
        if user:
            # 使用软注意力机制来计算各个方面（aspect）对用户的重要性
            aspect_prob = torch.sum(torch.mul(beta, self.item_topics), dim=-1)
            aspect_prob = torch.softmax(aspect_prob, dim=1)

            # 初始化列表来存储各个方面的编码结果、相邻表示和隐变量均值
            z_u_list = []  # 存储用户的隐表示
            nei_u_list = []  # 存储基于相邻的用户表示
            z_u_mu_list = []  # 存储隐变量的均值
            probs = None
            kl = None

            for a in range(self.a):  # 遍历所有方面
                aspect_a = aspect_prob[:, a].reshape((1, -1))
                # 编码过程，得到每个方面的隐变量均值和标准差
                mu, std = self.encode_user(x * aspect_a)
                # 计算KL散度作为正则项
                kl_a = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
                kl_a = torch.mean(torch.sum(kl_a, dim=-1))
                kl = (kl_a if (kl is None) else (kl + kl_a))
                # 重参数化步骤，生成隐表示
                theta = self.reparameterize(mu, std)
                # 解码过程，基于隐表示重构输入
                probs_a = self.decode_user(theta, beta[:, a, :].squeeze())
                probs_a = probs_a * aspect_a
                probs = probs_a if probs is None else (probs + probs_a)
                z_u_list.append(theta)
                z_u_mu_list.append(mu)
                # 计算邻域表示
                nei_u_list.append(torch.mm(probs_a, beta[:, a, :].squeeze()))

            # 将列表转换为张量
            z_u_list = torch.stack(z_u_list).transpose(0, 1)
            z_u_mu_list = torch.stack(z_u_mu_list).transpose(0, 1)
            nei_u_list = torch.stack(nei_u_list).transpose(0, 1)
            # 平均KL散度
            kl = kl / self.a
            # 计算对比损失
            cl = self.contrast_loss(z_u_list, nei_u_list)
            # 根据似然函数选择最终的概率分布
            if self.likelihood == 'mult':
                probs = torch.softmax(probs, dim=1)
            return z_u_list, z_u_mu_list, probs, kl, cl
        # Item
        else:
            prefer_prob = torch.sum(torch.mul(theta, self.user_preferences), dim=-1)
            prefer_prob = torch.softmax(prefer_prob, dim=1)

            z_i_list = []
            nei_i_list = []
            z_i_mu_list = []
            probs = None
            kl = None
            for a in range(self.a):
                prefer_a = prefer_prob[:, a].reshape((1, -1))
                # encoder
                mu, std = self.encode_item(x * prefer_a)
                # KL term
                kl_a = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
                kl_a = torch.mean(torch.sum(kl_a, dim=-1))
                kl = (kl_a if (kl is None) else (kl + kl_a))
                beta = self.reparameterize(mu, std)
                # decoder
                probs_a = self.decode_item(theta[:, a, :].squeeze(), beta)
                probs_a = probs_a * prefer_a
                probs = probs_a if probs is None else (probs + probs_a)
                z_i_list.append(beta)
                z_i_mu_list.append(mu)
                # neighborhood-based representation
                nei_i_list.append(torch.mm(probs_a, theta[:, a, :].squeeze()))
            z_i_list = torch.stack(z_i_list).transpose(0, 1)
            z_i_mu_list = torch.stack(z_i_mu_list).transpose(0, 1)
            nei_i_list = torch.stack(nei_i_list).transpose(0, 1)
            # KL
            kl = kl / self.a
            # CL
            cl = self.contrast_loss(z_i_list, nei_i_list)
            if self.likelihood == 'mult':
                probs = torch.softmax(probs, dim=1)
            return z_i_list, z_i_mu_list, probs, kl, cl

    def _loss(self, x, x_, kl, kl_beta, cl, cl_gama):
        # 根据不同的似然函数计算重构损失
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),  # 伯努利分布
            "gaus": -(x - x_) ** 2,  # 高斯分布
            "pois": x * torch.log(x_ + EPS) - x_,  # 泊松分布
            "mult": torch.log(x_ + EPS) * x  # 多项分布
        }
        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))
        ll = torch.mean(torch.sum(ll, dim=-1))
        # 计算总损失，包括KL散度、对比损失和重构损失
        return kl_beta * kl - ll + cl_gama * cl

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        # 项目
        tx = self.interaction_matrix.transpose()
        tx_csr = tx.tocsr()
        pos_items_cpu = pos_items.cpu().numpy()
        i_batch_dense = tx_csr[pos_items_cpu, :].toarray()
        i_batch_tensor = torch.tensor(i_batch_dense, dtype=torch.float32, device=self.device)
        z_i_list, z_i_mu_list, probs, kl, cl = self.forward(i_batch_tensor, user=False, theta=self.theta)
        item_loss = self._loss(i_batch_tensor, probs, kl, self.kl_weight, cl, self.ssl_reg)
        self.beta.data[pos_items] = z_i_list.data
        self.mu_beta.data[pos_items] = z_i_mu_list.data

        # 用户
        x = self.interaction_matrix
        x_csr = x.tocsr()
        users_cpu = users.cpu().numpy()
        u_batch_dense = x_csr[users_cpu, :].toarray()
        u_batch_tensor = torch.tensor(u_batch_dense, dtype=torch.float32, device=self.device)
        z_u_list, z_u_mu_list, probs, kl, cl = self.forward(u_batch_tensor, user=True, beta=self.beta)
        user_loss = self._loss(u_batch_tensor, probs, kl, self.kl_weight, cl, self.ssl_reg)
        self.theta.data[users] = z_u_list.data
        self.mu_theta.data[users] = z_u_mu_list.data

        return item_loss + user_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 初始化预测评分变量
        known_item_scores = None
        theta = self.mu_theta  # [15482, 5, 20]
        beta = self.mu_beta  # [8643, 5, 20]
        aspect_prob = torch.sum(torch.mul(self.mu_beta, self.item_topics), dim=-1)  # [8643, 5]
        self.aspect_probability = torch.softmax(aspect_prob, dim=1)  # [8643, 5]
        # 遍历所有方面，计算评分
        for a in range(self.a):
            theta_a = theta[:, a, :]  # 获取当前方面的用户隐变量
            beta_a = beta[:, a, :].squeeze()  # 获取当前方面的所有物品隐变量  [8643, 20]
            aspect_a = aspect_prob[:, a].reshape((1, -1))  # 获取当前方面的概率
            scores_a = self.decode_user(theta_a, beta_a)  # 解码得到评分
            scores_a = scores_a * aspect_a  # 将评分与方面概率相乘，加权评分
            known_item_scores = scores_a if known_item_scores is None else (known_item_scores + scores_a)  # 累加各方面的评分
        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 将历史交互设置为极小值
        for row, col in self.user_item_dict.items():
            col = torch.LongTensor(list(col)) - self.num_user
            known_item_scores[row][col] = 1e-6

        # 选出每个用户的 top-k 个物品
        _, index_of_rank_list_train = torch.topk(known_item_scores, topk)
        # 总的top-k列表
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
            dim=0)

        # 返回三个推荐列表
        return all_index_of_rank_list
