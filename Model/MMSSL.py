"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/23 22:02
@File : MMSSL.py
@function :
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import autograd
from scipy.sparse import csr_matrix

from MAD import mad_value


# 鉴别器
class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.G_drop1 = 0.31
        self.G_drop2 = 0.5
        # Eq(4)
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim / 4)),  # 第一层是线性变换层，将输入的维度从dim降低到dim / 4
            nn.LeakyReLU(True),  # 使用LeakyReLU激活函数，避免梯度消失问题
            nn.BatchNorm1d(int(dim / 4)),  # 批量归一化
            nn.Dropout(self.G_drop1),  # Dropout层

            nn.Linear(int(dim / 4), int(dim / 8)),  # 第二层继续降维，从dim / 4到dim / 8
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 8)),
            nn.Dropout(self.G_drop2),

            nn.Linear(int(dim / 8), 1),  # 最后一层线性变换到1个神经元，输出判别结果
            nn.Sigmoid()  # 通过Sigmoid激活函数将输出压缩到0和1之间，作为判别的概率输出
        )

    def forward(self, x):
        output = 100 * self.net(x.float())
        return output.view(-1)


class MMSSL(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E,
                 reg_weight, ssl_alpha, ssl_temp, G_rate, mmlayer, device):
        super(MMSSL, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.weight_size = [64] * mmlayer # 权重尺寸列表，用于定义多层网络的每层大小
        self.n_ui_layers = len(self.weight_size)
        # 在权重尺寸列表前面加上嵌入维度，用于网络的第一层输入
        self.weight_size = [self.dim_E] + self.weight_size
        self.device = device
        self.mmlayer = mmlayer
        self.user_item_dict = user_item_dict
        self.reg_weight = reg_weight  # [1e-5,1e-3,1e-2]
        self.tau = ssl_temp  # 0.5
        self.feat_reg_decay = 1e-5  # 特征正则化系数
        self.gene_u, self.gene_real, self.gene_fake = None, None, {}
        self.log_log_scale = 0.00001
        self.real_data_tau = 0.005
        self.ui_pre_scale = 100
        self.gp_rate = 1
        self.T = 1
        self.m_topk_rate = 0.0001
        self.cl_rate = ssl_alpha  # 0.003
        self.G_rate = G_rate  # 0.0001

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                            (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                           shape=(self.num_user, self.num_item), dtype=np.float32)
        interaction_matrix = interaction_matrix.tocsr()
        self.ui_graph = self.ui_graph_raw = interaction_matrix
        self.iu_graph = self.ui_graph.T
        # COO格式的稀疏矩阵转换为Numpy数组的稠密表示
        dense_matrix = interaction_matrix.todense()
        self.image_ui_graph_tmp = self.text_ui_graph_tmp = torch.tensor(dense_matrix).to(self.device)
        self.image_iu_graph_tmp = self.text_iu_graph_tmp = torch.tensor(dense_matrix.T).to(self.device)
        # 初始化图像和文本交互的索引
        self.image_ui_index = {'x': [], 'y': []}
        self.text_ui_index = {'x': [], 'y': []}
        # 归一化
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        # 鉴别器
        self.D = Discriminator(self.num_item).to(self.device)
        self.D.apply(self.weights_init)

        # 定义将图像特征映射到嵌入空间的线性变换
        self.image_trans = nn.Linear(v_feat.shape[1], self.dim_E)
        # 定义将文本特征映射到嵌入空间的线性变换
        self.text_trans = nn.Linear(t_feat.shape[1], self.dim_E)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)

        # 使用ModuleDict存储两个编码器（图像和文本）
        self.encoder = nn.ModuleDict()
        self.encoder['image_encoder'] = self.image_trans
        self.encoder['text_encoder'] = self.text_trans

        # 定义一个通用的线性变换层，用于进一步处理嵌入空间的特征
        self.common_trans = nn.Linear(self.dim_E, self.dim_E)
        nn.init.xavier_uniform_(self.common_trans.weight)
        # 使用ModuleDict存储该通用变换层
        self.align = nn.ModuleDict()
        self.align['common_trans'] = self.common_trans

        # 为用户和物品各创建一个嵌入层
        self.user_id_embedding = nn.Embedding(num_user, self.dim_E)
        self.item_id_embedding = nn.Embedding(num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 将图像和文本特征转换为CUDA张量
        self.image_feats = v_feat
        self.text_feats = t_feat

        # 定义softmax激活函数
        self.softmax = nn.Softmax(dim=-1)
        # 定义sigmoid激活函数，用于激活函数和最终的二分类任务
        self.act = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=0.2)
        # 定义批量归一化层，用于加速训练过程并减少过拟合
        self.batch_norm = nn.BatchNorm1d(self.dim_E)
        # 定义温度参数，用于控制对比损失中的尺度
        # self.tau = 0.5

        # 使用Xavier初始化方法初始化自注意力机制所需的参数
        self.head_num = 4
        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_k': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_v': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_self_attention_cat': nn.Parameter(
                initializer(torch.empty([self.head_num * self.dim_E, self.dim_E]))),
        })

        # 初始化用于存储用户和物品嵌入的字典
        self.embedding_dict = {'user': {}, 'item': {}}

        self.sparse = 1
        self.model_cat_rate = 0.55
        self.id_cat_rate = 0.36

    def mm(self, x, y):
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    # 计算梯度惩罚
    def gradient_penalty(self, D, xr, xf):
        LAMBDA = 0.3  # 惩罚系数

        xf = xf.detach()  # 从计算图中分离生成的样本，避免影响生成器的梯度
        xr = xr.detach()  # 从计算图中分离真实样本

        alpha = torch.rand(xr.size(0), 1).cuda()  # 随机生成混合系数
        alpha = alpha.expand_as(xr)  # 扩展alpha以匹配样本的形状

        interpolates = alpha * xr + ((1 - alpha) * xf)  # 计算真实样本和生成样本的加权和
        interpolates.requires_grad_()  # 为加权和的变量添加梯度计算

        disc_interpolates = D(interpolates)  # 通过判别器计算加权和的输出

        # 计算加权和的梯度
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        # 计算梯度的L2范数，并应用惩罚
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    def matrix_to_tensor(self, cur_matrix):
        # 如果当前矩阵不是COO（Coordinate）格式，则转换为COO格式
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()
        # 获取COO格式矩阵的行索引、列索引，并堆叠成二维数组，再转换为长整型张量
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        # 将COO格式矩阵的数据值转换为张量
        values = torch.from_numpy(cur_matrix.data)
        # 获取矩阵的形状，并创建相应的大小对象
        shape = torch.Size(cur_matrix.shape)

        # 利用上述行列索引、值和形状创建PyTorch的稀疏张量，数据类型设置为float32，并移动到GPU上
        return torch.sparse_coo_tensor(indices, values, shape).to(torch.float32).to(self.device)

    def para_dict_to_tenser(self, para_dict):
        """
        将nn.ParameterDict()中的参数堆叠成一个张量。
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        # 遍历参数字典，将每个参数添加到tensors列表中
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        # 将列表中的所有参数张量堆叠成一个新的张量，每个参数张量成为堆叠张量的一个切片
        tensors = torch.stack(tensors, dim=0)

        return tensors

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):
        # trans_w  传入的权重字典，包括用于查询（Q）、键（K）和值（V）的转换权重，以及其他自注意力相关权重
        # 将输入的两组嵌入转换为张量
        q = self.para_dict_to_tenser(embedding_t)  # 查询  (2, num, dim_E)
        v = k = self.para_dict_to_tenser(embedding_t_1)  # 键和值  (2, num, dim_E)

        # 计算每个头的维度
        beh, N, d_h = q.shape[0], q.shape[1], self.dim_E / self.head_num

        # 计算查询、键和值
        Q = torch.matmul(q, trans_w['w_q'])  # 查询与其权重的矩阵乘法
        K = torch.matmul(k, trans_w['w_k'])  # 键与其权重的矩阵乘法
        V = v  # 值  (2, num, dim_E)

        # 重塑查询、键，以适应多头的要求  [4,2,N,16]
        Q = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)
        K = K.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)

        # 增加一个维度以便进行广播操作
        Q = torch.unsqueeze(Q, 2)  # (self.head_num, 2, num, 1, d_h)
        K = torch.unsqueeze(K, 1)  # (self.head_num, 2, 1, num, d_h)
        V = torch.unsqueeze(V, 1)  # (2, 1, num, dim_E)

        # 计算注意力权重
        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  # 点积注意力  (self.head_num, 2, num, num)
        att = torch.sum(att, dim=-1)
        att = torch.unsqueeze(att, dim=-1)
        att = F.softmax(att, dim=2)  # 使用softmax获取注意力分布

        # 应用注意力权重到值上
        Z = torch.mul(att, V)  # (self.head_num, 2, num, d_h)
        Z = torch.sum(Z, dim=2)

        # 将来自不同头的结果连接起来
        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])  # (2, num, dim_E)

        # 应用归一化和比例因子
        self.model_cat_rate * F.normalize(Z, p=2, dim=2)
        return Z, att.detach()

    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):
        # 对图像和文本特征应用dropout和线性变换
        image_feats = image_item_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = text_item_feats = self.dropout(self.text_trans(self.text_feats))

        image_user_id = None
        text_user_id = None
        image_item_id = None
        text_item_id = None
        image_user_feats = None
        text_user_feats = None

        # 使用图结构信息更新图像和文本特征
        for i in range(self.mmlayer):
            # 图像特征的用户和物品嵌入更新  Eq(2)
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)
            # 图像模态感知嵌入  eq(8)
            image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)
            image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

            # 文本特征的用户和物品嵌入更新  Eq(2)
            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)
            # 文本模态感知嵌入  eq(8)
            text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)
            text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)

        # 将最终的图像和文本用户、物品嵌入存储到字典中
        self.embedding_dict['user']['image'] = image_user_id
        self.embedding_dict['user']['text'] = text_user_id
        self.embedding_dict['item']['image'] = image_item_id
        self.embedding_dict['item']['text'] = text_item_id

        # 应用多头自注意力机制，获取用户和物品的综合表示  eq(9)
        user_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['user'],
                                                   self.embedding_dict['user'])
        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'],
                                                   self.embedding_dict['item'])

        # 生成多模态用户/项目表示
        user_emb = user_z.mean(0)
        item_emb = item_z.mean(0)
        # 结合用户和物品ID嵌入与通过自注意力获得的嵌入
        u_g_embeddings = self.user_id_embedding.weight + self.id_cat_rate * F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + self.id_cat_rate * F.normalize(item_emb, p=2, dim=1)

        # 初始化用户和物品嵌入列表  Eq(10)
        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]

        # 根据交云图更新用户和物品嵌入  Eq(10)
        for i in range(self.n_ui_layers):
            if i == (self.n_ui_layers - 1):  # 最后一层使用softmax进行归一化
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        # 计算所有层嵌入的平均值，得到最终嵌入  Eq(10)
        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        # 将图像和文本特征的影响融入最终的用户和物品嵌入中    Eq(12)
        u_g_embeddings = u_g_embeddings + self.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1) + \
                         self.model_cat_rate * F.normalize(
            text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + self.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1) + \
                         self.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)

        # 返回最终的用户嵌入、物品嵌入以及其他相关的特征嵌入
        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, \
            text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):
        # 获取设备信息，确保计算在同一设备上进行
        device = z1.device
        # 计算总的节点数量
        num_nodes = z1.size(0)
        # 计算需要的批次数量
        num_batches = (num_nodes - 1) // batch_size + 1
        # 定义一个函数f，用于计算经过温度参数tau调整后的指数函数
        f = lambda x: torch.exp(x / self.tau)

        # 生成一个从0到num_nodes的索引，用于后续批处理
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        # 对每个批次进行处理
        for i in range(num_batches):
            # 获取当前批次的索引
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            # 对于每个批次，与其他所有批次进行对比
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                # 计算当前批次内部的相似度（自反相似度）
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))
                # 计算当前批次与其他批次之间的相似度（交叉相似度）
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            # 将计算得到的所有相似度拼接起来
            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            # 计算对比损失，这里使用了对数损失形式
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                    refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:,
                                                           i * batch_size:(i + 1) * batch_size].diag()) + 1e-8))

            # 删除临时变量以释放内存
            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        # 将所有批次的损失拼接起来，并计算平均损失
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    # 特征正则化损失
    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        # 计算图像、文本特征以及用户图像、用户文本特征的L2正则化损失
        feat_reg = 1. / 2 * (g_item_image ** 2).sum() + 1. / 2 * (g_item_text ** 2).sum() \
                   + 1. / 2 * (g_user_image ** 2).sum() + 1. / 2 * (g_user_text ** 2).sum()
        # 将正则化损失除以物品数量，以平衡不同大小数据集的影响
        feat_reg = feat_reg / self.num_item
        # 应用一个正则化系数（feat_reg_decay）来调整正则化损失的比重
        feat_emb_loss = self.feat_reg_decay * feat_reg
        return feat_emb_loss

    # 生成器损失  Eq(7)
    def fake_gene_loss_calculation(self, u_emb, i_emb, emb_type=None):
        if self.gene_u is not None:
            # 计算真实样本的损失，使用负的对数sigmoid函数
            gene_real_loss = (-F.logsigmoid((u_emb[self.gene_u] * i_emb[self.gene_real]).sum(-1) + 1e-8)).mean()
            # 计算生成（假）样本的损失，同样使用负的对数sigmoid函数，但是表达式不同
            gene_fake_loss = (1 - (
                -F.logsigmoid((u_emb[self.gene_u] * i_emb[self.gene_fake[emb_type]]).sum(-1) + 1e-8))).mean()

            # 真实样本损失和生成样本损失的和作为总损失
            gene_loss = gene_real_loss + gene_fake_loss
        else:
            # 如果没有指定用于计算的用户向量，损失为0
            gene_loss = 0

        return gene_loss

    def u_sim_calculation(self, users, user_final, item_final):
        # 根据用户索引获取用户最终表示
        topk_u = user_final[users]
        # 从稀疏矩阵中获取用户-物品交互信息，并转移到CUDA上
        u_ui = torch.tensor(self.ui_graph_raw[users.cpu().numpy()].todense()).to(self.device)

        # 初始化变量，用于分批处理物品并计算相似度
        num_batches = (self.num_item - 1) // 1024 + 1
        indices = torch.arange(0, self.num_item).to(self.device)
        u_sim_list = []

        # 分批计算用户与物品的相似度
        for i_b in range(num_batches):
            index = indices[i_b * 1024:(i_b + 1) * 1024]
            # 计算用户表示和物品表示的点积，得到相似度
            sim = torch.mm(topk_u, item_final[index].T)
            # 将相似度乘以一个指示矩阵（1-u_ui），用于过滤已有交互的物品
            sim_gt = torch.multiply(sim, (1 - u_ui[:, index]))
            u_sim_list.append(sim_gt)

        # 将所有批次的相似度结果拼接，并进行L2归一化
        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
        return u_sim

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / 1024

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def loss_D(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        with torch.no_grad():
            ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds \
                , _, _, _, _, _, _ = self.forward(self.ui_graph, self.iu_graph, self.image_ui_graph,
                                                  self.image_iu_graph,
                                                  self.text_ui_graph, self.text_iu_graph)

        # 分别计算用户与物品的相似度，考虑了多模态信息
        ui_u_sim_detach = self.u_sim_calculation(users, ua_embeddings, ia_embeddings).detach()
        image_u_sim_detach = self.u_sim_calculation(users, image_user_embeds, image_item_embeds).detach()
        text_u_sim_detach = self.u_sim_calculation(users, text_user_embeds, text_item_embeds).detach()

        # 将多模态相似度向量合并，作为判别器的输入
        inputf = torch.cat((image_u_sim_detach, text_u_sim_detach), dim=0)
        predf = self.D(inputf)  # 判别器对输入的评分
        lossf = predf.mean()  # 计算生成数据的损失

        # 对用户-物品交互进行处理，包括归一化和应用softmax函数
        u_ui = torch.tensor(self.ui_graph_raw[users.cpu().numpy()].todense()).to(self.device)
        u_ui = F.softmax(u_ui - self.log_log_scale * torch.log(-torch.log(
            torch.empty((u_ui.shape[0], u_ui.shape[1]), dtype=torch.float32).uniform_(0, 1).to(
                self.device) + 1e-8) + 1e-8) /
                         self.real_data_tau, dim=1)
        u_ui += ui_u_sim_detach * self.ui_pre_scale
        u_ui = F.normalize(u_ui, dim=1)

        # 计算真实数据的损失并应用梯度惩罚
        inputr = torch.cat((u_ui, u_ui), dim=0)
        predr = self.D(inputr)
        lossr = -(predr.mean())
        gp = self.gradient_penalty(self.D, inputr, inputf.detach())
        loss_D = lossr + lossf + self.gp_rate * gp

        return loss_D

    def loss(self, users, pos_items, neg_items, idx):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        G_ua_embeddings, G_ia_embeddings, G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, \
            G_text_user_embeds, G_user_emb, _, G_image_user_id, G_text_user_id, _, _ \
            = self.forward(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph,
                           self.text_ui_graph, self.text_iu_graph)

        # 根据用户ID索引获取对应的用户嵌入向量
        G_u_g_embeddings = G_ua_embeddings[users]
        # 根据正样本物品ID索引获取对应的物品嵌入向量
        G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
        # 根据负样本物品ID索引获取对应的物品嵌入向量
        G_neg_i_g_embeddings = G_ia_embeddings[neg_items]

        # 计算基于二部图（Bipartite Graph）推荐损失（BPR Loss）
        G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss(G_u_g_embeddings, G_pos_i_g_embeddings,
                                                                            G_neg_i_g_embeddings)

        # 根据用户嵌入向量和物品嵌入向量计算用户与物品的相似度，这里考虑了图像和文本两种模态
        G_image_u_sim = self.u_sim_calculation(users, G_image_user_embeds, G_image_item_embeds)
        G_text_u_sim = self.u_sim_calculation(users, G_text_user_embeds, G_text_item_embeds)

        # 将计算得到的相似度从计算图中分离，以防止在后续操作中计算这部分的梯度
        G_image_u_sim_detach = G_image_u_sim.detach()
        G_text_u_sim_detach = G_text_u_sim.detach()

        # 每隔args.T个批次，根据收集的交互信息更新用户-物品交互图
        if idx % self.T == 0 and idx != 0:
            # 根据收集的图像模态用户-物品交互信息构建临时稀疏矩阵
            self.image_ui_graph_tmp = csr_matrix((torch.ones(len(self.image_ui_index['x'])),
                                                  (self.image_ui_index['x'], self.image_ui_index['y'])),
                                                 shape=(self.num_user, self.num_item))
            # 根据收集的文本模态用户-物品交互信息构建临时稀疏矩阵
            self.text_ui_graph_tmp = csr_matrix(
                (torch.ones(len(self.text_ui_index['x'])), (self.text_ui_index['x'], self.text_ui_index['y'])),
                shape=(self.num_user, self.num_item))
            # 生成物品-用户交互图，即用户-物品交互图的转置
            self.image_iu_graph_tmp = self.image_ui_graph_tmp.T
            self.text_iu_graph_tmp = self.text_ui_graph_tmp.T
            # 将临时稀疏矩阵进行归一化处理并转换为PyTorch稀疏张量，用于模型的输入
            self.image_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                self.csr_norm(self.image_ui_graph_tmp, mean_flag=True)
            ).cuda()
            self.text_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                self.csr_norm(self.text_ui_graph_tmp, mean_flag=True)
            ).cuda()
            self.image_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                self.csr_norm(self.image_iu_graph_tmp, mean_flag=True)
            ).cuda()
            self.text_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                self.csr_norm(self.text_iu_graph_tmp, mean_flag=True)
            ).cuda()

            # 清空用于收集交互信息的索引列表，准备下一轮的信息收集
            self.image_ui_index = {'x': [], 'y': []}
            self.text_ui_index = {'x': [], 'y': []}
        else:
            # 在非更新图结构的批次中，根据当前模型预测结果收集需要更新的用户-物品交互信息
            # 使用top-k操作选择每个用户与物品的相似度最高的物品作为候选更新项
            _, image_ui_id = torch.topk(G_image_u_sim_detach, int(self.num_item * self.m_topk_rate), dim=-1)
            # 将选中的用户和物品索引添加到图像模态的更新列表中
            self.image_ui_index['x'] += np.array(
                torch.tensor(users.cpu().numpy()).repeat(1, int(self.num_item * self.m_topk_rate)).view(-1)).tolist()
            self.image_ui_index['y'] += np.array(image_ui_id.cpu().view(-1)).tolist()
            _, text_ui_id = torch.topk(G_text_u_sim_detach, int(self.num_item * self.m_topk_rate), dim=-1)
            # 将选中的用户和物品索引添加到文本模态的更新列表中
            self.text_ui_index['x'] += np.array(
                torch.tensor(users.cpu().numpy()).repeat(1, int(self.num_item * self.m_topk_rate)).view(-1)).tolist()
            self.text_ui_index['y'] += np.array(text_ui_id.cpu().view(-1)).tolist()

        # 计算特征嵌入正则化损失，以防止过拟合，并鼓励模型学习更通用的特征表示
        feat_emb_loss = self.feat_reg_loss_calculation(G_image_item_embeds, G_text_item_embeds, G_image_user_embeds,
                                                       G_text_user_embeds)

        # 计算基于图像用户ID和通用用户嵌入向量的对比损失
        batch_contrastive_loss1 = self.batched_contrastive_loss(G_image_user_id[users], G_user_emb[users])
        # 计算基于文本用户ID和通用用户嵌入向量的对比损失
        batch_contrastive_loss2 = self.batched_contrastive_loss(G_text_user_id[users], G_user_emb[users])
        # 将两种对比损失相加得到总的对比损失
        batch_contrastive_loss = batch_contrastive_loss1 + batch_contrastive_loss2

        # 将图像和文本的用户相似度向量合并，作为判别器的输入
        G_inputf = torch.cat((G_image_u_sim, G_text_u_sim), dim=0)
        # 使用判别器对合并的相似度向量进行预测，并计算损失
        G_predf = self.D(G_inputf)
        # 计算生成器的损失
        G_lossf = -(G_predf.mean())

        # 将基础损失、特征嵌入正则化损失、对比损失以及生成器损失相加，得到总的批次损失
        batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + feat_emb_loss + \
                     self.cl_rate * batch_contrastive_loss + self.G_rate * G_lossf

        return batch_loss

    def gene_ranklist(self, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入

        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.forward(self.ui_graph, self.iu_graph, self.image_ui_graph,
                                                             self.image_iu_graph, self.text_ui_graph,
                                                             self.text_iu_graph)

        user_tensor = ua_embeddings.cpu()
        item_tensor = ia_embeddings.cpu()

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
