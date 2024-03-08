"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/8 17:39
@File : MacridVAE.py
@function :
"""
import torch
from torch import nn
import torch.nn.functional as F


class MacridVAE(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, dim_E,  reg_weight, device):
        super(MacridVAE, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.edge_index = edge_index
        self.dim_E = dim_E
        # self.reg_weight = reg_weight
        self.device = device
        self.training = True

        self.drop_out = 0.5  # dropout保留率
        self.kfac = 10  # 概念因子的数量
        self.layers = [600]  # 编码器隐藏层的大小
        self.tau = 0.1  # 温度参数tau，用于softmax函数
        self.nogb = False  # 是否使用无梯度背传（No Gradient Backpropagation）
        self.regs = [0.0, 0.0]  # 正则化权重
        self.std = 0.01  # 标准差,用于重参数化技巧中的随机采样
        self.total_anneal_steps = 200000  # KL散度权重的退火步数
        self.anneal_cap = 0.2  # KL散度渐变上限（Anneal Cap）
        self.update = 0  # KL散度已完成的更新步数

        # 设置编码器层的维度，包括输入层（项目数量），隐藏层（从配置中读取），输出层（嵌入大小的两倍，对应均值和方差）
        self.encode_layer_dims = (
                [self.num_item] + self.layers + [self.dim_E * 2]
        )
        # 使用多层感知机（MLP）构建编码器
        self.encoder = self.mlp_layers(self.encode_layer_dims)

        # 初始化项目嵌入层，用于将项目ID转换为稠密向量
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_normal_(self.item_embedding.weight)
        # 初始化概念嵌入层，用于将概念ID转换为稠密向量
        self.k_embedding = nn.Embedding(self.kfac, self.dim_E)
        nn.init.xavier_normal_(self.k_embedding.weight)

        # 创建稀疏矩阵
        indices = torch.LongTensor(edge_index.T)
        indices[1] -= self.num_user  # 调整项目ID
        values = torch.FloatTensor([1] * len(edge_index))
        self.rating_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([self.num_user, self.num_item]))

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        # 遍历layer_dims列表中每一对相邻元素，这里的d_in是输入维度，d_out是输出维度
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))  # 创建一个线性层，并添加到mlp_modules列表中
            if i != len(layer_dims[:-1]) - 1:  # 检查当前层是否是最后一层
                mlp_modules.append(nn.Tanh())  # 如果不是最后一层，添加一个Tanh激活函数
        return nn.Sequential(*mlp_modules)  # 使用Sequential来整合所有的层，并返回这个序列化的模型

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, rating_matrix):
        # 对概念（k_embedding）和项目（item_embedding）的权重进行L2范数归一化
        cores = F.normalize(self.k_embedding.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)

        # 使用self.rating_matrix替换rating_matrix参数

        # 对输入的评分矩阵进行L2范数归一化，并应用dropout
        rating_matrix = F.normalize(rating_matrix)
        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        # 计算项目和概念之间的相似度，并通过温度参数tau进行缩放
        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        # 根据nogb标志使用softmax或gumbel_softmax来处理相似度，得到每个项目属于各个概念的概率
        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode

        # 初始化概率列表、均值列表和对数方差列表
        probs = None
        mulist = []
        logvarlist = []
        for k in range(self.kfac):  # 遍历所有概念
            cates_k = cates[:, k].reshape(1, -1)  # 获取当前概念的概率分布
            # 使用稀疏矩阵转换后的密集表示
            # 编码器部分：使用当前概念的概率加权评分矩阵，然后通过编码器
            x_k = rating_matrix.to(self.device) * cates_k
            h = self.encoder(x_k)
            mu = h[:, :self.dim_E]  # 前一半是均值
            mu = F.normalize(mu, dim=1)
            logvar = h[:, self.dim_E:]  # 后一半是对数方差

            # 保存均值和对数方差
            mulist.append(mu)
            logvarlist.append(logvar)

            # 使用重参数化技巧得到潜在表示z
            z = self.reparameterize(mu, logvar)

            # 解码器部分：将潜在表示与项目嵌入相乘，得到重构的评分
            z_k = F.normalize(z, dim=1)
            logits_k = torch.matmul(z_k, items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            # 累加各概念的重构评分
            probs = probs_k if (probs is None) else (probs + probs_k)

        # 对累加的重构评分取对数
        logits = torch.log(probs)

        return logits, mulist, logvarlist

    def loss(self, users, pos_items, neg_items):

        rating_matrix_dense = self.rating_matrix.to_dense().to(self.device)
        rating_matrix = rating_matrix_dense[users]

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        # 直接调用forward方法，不需要额外的参数
        z, mu, logvar = self.forward(rating_matrix)
        kl_loss = None
        for i in range(self.kfac):
            # 计算KL散度损失，用于测量潜在表示的分布与先验分布的差异
            kl_ = -0.5 * torch.mean(torch.sum(1 + logvar[i] - logvar[i].exp(), dim=1))
            kl_loss = kl_ if (kl_loss is None) else (kl_loss + kl_)

        # 计算交叉熵损失（CE loss），用于测量重构评分与实际评分之间的差异
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        # 如果正则化权重不为0，加上正则化损失
        if self.regs[0] != 0 or self.regs[1] != 0:
            return ce_loss + kl_loss * anneal + self.reg_loss()

        return ce_loss + kl_loss * anneal

    def reg_loss(self):
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.item_embedding.weight.norm(2)
        loss_2 = reg_1 * self.k_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.encoder.named_parameters():
            if name.endswith("weight"):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        self.training = False
        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 生成评分矩阵
        rating_matrix_dense = self.rating_matrix.to_dense()
        score_matrix, _, _ = self.forward(rating_matrix_dense)
        score_matrix = score_matrix.detach()  # 确保不需要梯度

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


