"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/3/3 15:20
@File : DCMF.py
@function :
"""
import torch
from torch import nn
import torch.nn.functional as F
from BasicGCN import GCNConv


class DCMF(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E,
                 reg_weight, corDecay, ssl_temp, n_factors, device):
        super(DCMF, self).__init__()
        self.user_textual_factor_embedding = None
        self.user_visual_factor_embedding = None
        self.u_t_embeddings = None
        self.i_t_embeddings = None
        self.u_v_embeddings = None
        self.i_v_embeddings = None
        self.u_g_embeddings = None
        self.i_g_embeddings = None
        self.textual_factor_embedding = None
        self.visual_factor_embedding = None
        self.item_factor_embedding = None
        self.user_factor_embedding = None
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E  # 64
        self.corDecay = corDecay
        self.reg_weight = reg_weight
        self.ssl_temp = ssl_temp
        self.n_factors = n_factors
        self.ssl_weight_1 = 0.01
        self.ssl_weight_2 = 0.01
        self.n_layers = 3
        self.m_layers = 3
        self.aggr_mode = 'add'
        self.device = device

        # 初始化用户和项目嵌入
        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 加载多模态特征
        self.v_feat = nn.Embedding.from_pretrained(v_feat, freeze=False)
        self.t_feat = nn.Embedding.from_pretrained(t_feat, freeze=False)
        # 转化到64维度
        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)

        # 转置并设置为无向图
        self.edge_index_clone = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_clone, self.edge_index_clone[[1, 0]]), dim=1)

        # 定义图卷积层
        self.conv_layers = nn.ModuleList([GCNConv(self.dim_E, self.dim_E, aggr=self.aggr_mode)
                                          for _ in range(self.n_layers)])

        self.conv_layers_m = nn.ModuleList([GCNConv(self.dim_E, self.dim_E, aggr=self.aggr_mode)
                                            for _ in range(self.m_layers)])

        # 随机嵌入的初始化
        # self.random_emb_visual = torch.rand(num_item, n_factors, dim_E // n_factors, device=device)
        # self.random_emb_textual = torch.rand(num_item, n_factors, dim_E // n_factors, device=device)
        # self.random_emb_item = torch.rand(num_item, n_factors, dim_E // n_factors, device=device)

        # 模态选择的概率分布
        # self.mode_probs = torch.tensor([0.33, 0.33, 0.34], device=device)

    # 距离相关性
    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            r = torch.sum(torch.square(X), 1, keepdim=True)
            D = torch.sqrt(torch.maximum(r - 2 * torch.matmul(X, X.transpose(1, 0)) + r.transpose(1, 0),
                                         torch.tensor([0.0]).to(self.device)) + 1e-8)
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            n_samples = torch.tensor(D1.shape[0], dtype=torch.float32).to(self.device)
            dcov = torch.sqrt(
                torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor([0.0]).to(self.device)) + 1e-8)
            return dcov

        X1 = X1.to(self.device)
        X2 = X2.to(self.device)
        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        dcor = dcov_12 / (torch.sqrt(torch.max(dcov_11 * dcov_22, torch.tensor([0.0]).to(self.device))) + 1e-10)
        dcor = dcor.squeeze()  # 将形状为 [1] 的张量转换为标量
        return dcor

    def disentangled(self, user_embedding, item_embedding, user_visual_embedding, visual_embedding,
                     user_textual_embedding, textual_embedding):
        item_batch_size, _ = item_embedding.size()
        user_batch_size, _ = user_embedding.size()

        user_factor_embedding = user_embedding.view(user_batch_size, self.n_factors,
                                                    self.dim_E // self.n_factors)
        user_visual_factor_embedding = user_visual_embedding.view(user_batch_size, self.n_factors,
                                                                  self.dim_E // self.n_factors)
        user_textual_factor_embedding = user_textual_embedding.view(user_batch_size, self.n_factors,
                                                                    self.dim_E // self.n_factors)
        item_factor_embedding = item_embedding.view(item_batch_size, self.n_factors,
                                                    self.dim_E // self.n_factors)
        visual_factor_embedding = visual_embedding.view(item_batch_size, self.n_factors, self.dim_E // self.n_factors)
        textual_factor_embedding = textual_embedding.view(item_batch_size, self.n_factors, self.dim_E // self.n_factors)

        return user_factor_embedding, item_factor_embedding, user_visual_factor_embedding, visual_factor_embedding, user_textual_factor_embedding, textual_factor_embedding

    def factor_contrastive(self, users, items):
        def cal_loss(emb1, emb2):
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            logits = torch.matmul(emb1, emb2.transpose(-2, -1)) / self.ssl_temp
            labels = torch.arange(emb1.size(0), device=self.device)
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss

        # 项目
        total_loss = 0.0

        # 对每个因素计算对比损失
        for factor_index in range(self.n_factors):
            # mode = torch.multinomial(self.mode_probs, 1).item()

            item_emb_factor = self.item_factor_embedding[:, factor_index, :]
            visual_emb_factor = self.visual_factor_embedding[:, factor_index, :]
            textual_emb_factor = self.textual_factor_embedding[:, factor_index, :]
            user_emb_factor = self.user_factor_embedding[:, factor_index, :]
            user_visual_emb_factor = self.user_visual_factor_embedding[:, factor_index, :]
            user_textual_emb_factor = self.user_textual_factor_embedding[:, factor_index, :]

            # 根据模态选择使用随机嵌入
            # if mode == 0:
            #     visual_emb_factor = self.random_emb_visual[:, factor_index, :]
            # elif mode == 1:
            #     textual_emb_factor = self.random_emb_textual[:, factor_index, :]
            # else:
            #     item_emb_factor = self.random_emb_item[:, factor_index, :]

            item = item_emb_factor[items]
            visual = visual_emb_factor[items]
            textual = textual_emb_factor[items]
            user = user_emb_factor[users]
            user_visual = user_visual_emb_factor[users]
            user_textual = user_textual_emb_factor[users]

            total_loss += cal_loss(item, visual)
            total_loss += cal_loss(item, textual)
            # total_loss += cal_loss(visual, textual)

            total_loss += cal_loss(user, user_visual)
            total_loss += cal_loss(user, user_textual)
            # total_loss += cal_loss(user_visual, user_textual)

            # total_loss += cal_loss(user, item)

        return total_loss

    def feature_contrastive(self, users, items):
        def cal_loss(emb1, emb2):
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            # pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
            # neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), dim=1)
            # loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            # loss /= pos_score.shape[0]
            logits = torch.mm(emb1, emb2.T) / self.ssl_temp
            labels = torch.tensor(list(range(emb1.shape[0]))).to(self.device)
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss

        # 项目上
        item_embedding = self.item_factor_embedding[items]
        visual_embedding = self.visual_factor_embedding[items]
        textual_embedding = self.textual_factor_embedding[items]
        user_embedding = self.user_factor_embedding[users]
        user_visual_emb = self.user_visual_factor_embedding[users]
        user_textual_emb = self.user_textual_factor_embedding[users]

        batch_size, n_factors, _ = item_embedding.size()
        user_batch_size, _, _ = user_embedding.size()
        # 还原成 [item/v_feat/t_feat, dim_E]
        item = item_embedding.view(batch_size, self.dim_E)
        visual = visual_embedding.view(batch_size, self.dim_E)
        textual = textual_embedding.view(batch_size, self.dim_E)
        user = user_embedding.view(user_batch_size, self.dim_E)
        user_visual = user_visual_emb.view(user_batch_size, self.dim_E)
        user_textual = user_textual_emb.view(user_batch_size, self.dim_E)

        total_loss = 0.0
        total_loss += cal_loss(item, visual)  # 项目与视觉特征对比
        total_loss += cal_loss(item, textual)  # 项目与文本特征对比
        total_loss += cal_loss(visual, textual)

        total_loss += cal_loss(user, user_visual)
        total_loss += cal_loss(user, user_textual)
        total_loss += cal_loss(user_visual, user_textual)

        total_loss += cal_loss(user, item)

        return total_loss

    def forward(self):
        # 转换到64维度
        visual_embedding = self.image_trs(self.v_feat.weight)
        textual_embedding = self.text_trs(self.t_feat.weight)

        # =====================id模态卷积===================
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        self.u_g_embeddings, self.i_g_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)

        # =====================视觉模态卷积===================
        ego_embeddings = torch.cat((self.user_embedding.weight, visual_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers_m:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        self.u_v_embeddings, self.i_v_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)

        # =====================文本模态卷积===================
        ego_embeddings = torch.cat((self.user_embedding.weight, textual_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for conv in self.conv_layers_m:
            ego_embeddings = conv(ego_embeddings, self.edge_index)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 均值处理
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        self.u_t_embeddings, self.i_t_embeddings = torch.split(all_embeddings, [self.num_user, self.num_item], dim=0)

        # 应用解耦表征学习 shape: [num_user/item/v_feat/t_feat, n_factors, dim_E / n_factors]
        user_factor_embedding, item_factor_embedding, user_visual_factor_embedding, \
            visual_factor_embedding, user_textual_factor_embedding, textual_factor_embedding = \
            self.disentangled(self.u_g_embeddings,
                              self.i_g_embeddings,
                              self.u_v_embeddings,
                              self.i_v_embeddings,
                              self.u_t_embeddings,
                              self.i_t_embeddings)

        self.user_factor_embedding = user_factor_embedding
        self.item_factor_embedding = item_factor_embedding
        self.visual_factor_embedding = visual_factor_embedding
        self.textual_factor_embedding = textual_factor_embedding
        self.user_visual_factor_embedding = user_visual_factor_embedding
        self.user_textual_factor_embedding = user_textual_factor_embedding

    def cor_loss(self, users, pos_items):
        user_tensor = self.user_factor_embedding[users]
        user_visual_tensor = self.user_visual_factor_embedding[users]
        user_textual_tensor = self.user_textual_factor_embedding[users]
        item_tensor = self.item_factor_embedding[pos_items]
        visual_tensor = self.visual_factor_embedding[pos_items]
        textual_tensor = self.textual_factor_embedding[pos_items]
        # 距离相关性损失
        cor_loss = 0.0

        # 转换维度以适应 _create_distance_correlation 函数
        # 从 [batch_size, n_factors, dim_E / n_factors] 到 [n_factors, batch_size, dim_E / n_factors]
        user_factor_embedding = user_tensor.transpose(0, 1)
        user_visual_factor_embedding = user_visual_tensor.transpose(0, 1)
        user_textual_factor_embedding = user_textual_tensor.transpose(0, 1)
        item_factor_embedding = item_tensor.transpose(0, 1)
        visual_factor_embedding = visual_tensor.transpose(0, 1)
        textual_factor_embedding = textual_tensor.transpose(0, 1)

        for i in range(0, self.n_factors - 1):
            # 注意：这里 x, y 的维度现在是 [batch_size, dim_E / n_factors]
            x_user = user_factor_embedding[i]
            y_user = user_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x_user, y_user)

            x_visual_user = user_visual_factor_embedding[i]
            y_visual_user = user_visual_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x_visual_user, y_visual_user)

            x_textual_user = user_textual_factor_embedding[i]
            y_textual_user = user_textual_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x_textual_user, y_textual_user)

            x_item = item_factor_embedding[i]
            y_item = item_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x_item, y_item)

            x_visual = visual_factor_embedding[i]
            y_visual = visual_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x_visual, y_visual)

            x_textual = textual_factor_embedding[i]
            y_textual = textual_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x_textual, y_textual)

        # 每个样本在所有因子上的平均距离相关性
        cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)

        return cor_loss

    def bpr_loss(self, users, pos_items, neg_items):
        # 初始化总BPR损失
        total_bpr_loss = 0.0

        # 遍历每个因素
        for factor_index in range(self.n_factors):
            # 获取每个因素的用户、正向和负向项目的嵌入
            user_embeddings = self.user_factor_embedding[:, factor_index, :]  # [batch_size, dim_E / n_factors]
            item_embeddings = self.item_factor_embedding[:, factor_index, :]  # [batch_size, dim_E / n_factors]

            # user_visual_embeddings = self.user_visual_factor_embedding[:, factor_index, :]
            # user_textual_embeddings = self.user_textual_factor_embedding[:, factor_index, :]
            # visual_item_embeddings = self.visual_factor_embedding[:, factor_index, :]
            # textual_item_embeddings = self.textual_factor_embedding[:, factor_index, :]

            # 从索引映射到嵌入
            user_embeddings = user_embeddings[users]
            pos_item_embeddings = item_embeddings[pos_items]
            neg_item_embeddings = item_embeddings[neg_items]

            # user_visual_embeddings = user_visual_embeddings[users]
            # pos_visual_item_embeddings = visual_item_embeddings[pos_items]
            # neg_visual_item_embeddings = visual_item_embeddings[neg_items]
            #
            # user_textual_embeddings = user_textual_embeddings[users]
            # pos_textual_item_embeddings = textual_item_embeddings[pos_items]
            # neg_textual_item_embeddings = textual_item_embeddings[neg_items]

            # 计算正向和负向项目的分数
            pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
            neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

            # pos_visual_scores = torch.sum(user_visual_embeddings * pos_visual_item_embeddings, dim=1)
            # neg_visual_scores = torch.sum(user_visual_embeddings * neg_visual_item_embeddings, dim=1)
            #
            # pos_textual_scores = torch.sum(user_textual_embeddings * pos_textual_item_embeddings, dim=1)
            # neg_textual_scores = torch.sum(user_textual_embeddings * neg_textual_item_embeddings, dim=1)

            # 计算当前因素的BPR损失
            bpr_loss_1 = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            # bpr_loss_2 = -torch.mean(F.logsigmoid(pos_visual_scores - neg_visual_scores))
            # bpr_loss_3 = -torch.mean(F.logsigmoid(pos_textual_scores - neg_textual_scores))

            # 累加到总损失
            total_bpr_loss = total_bpr_loss + bpr_loss_1

        return total_bpr_loss

    def regularization_loss(self, users, pos_items, neg_items):
        final_user = self.user_embedding.weight
        final_item = self.item_embedding.weight
        # 计算正则化损失
        user_embeddings = final_user[users]
        pos_item_embeddings = final_item[pos_items]
        neg_item_embeddings = final_item[neg_items]

        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        self.forward()

        # 计算cor损失
        cor_loss = self.cor_loss(users, pos_items)
        # 计算factor_contrastive损失
        factor_contrastive_loss = self.factor_contrastive(users, pos_items)
        # 计算feature_contrastive损失
        # feature_contrastive_loss = self.feature_contrastive(users, pos_items)

        # 计算 BPR 损失和正则化损失
        bpr_loss = self.bpr_loss(users, pos_items, neg_items)
        reg_loss = self.regularization_loss(users, pos_items, neg_items)

        # total_loss = bpr_loss + reg_loss + self.corDecay * cor_loss + self.ssl_weight_1 * factor_contrastive_loss \
        #              + self.ssl_weight_2 * feature_contrastive_loss
        total_loss = bpr_loss + reg_loss + self.corDecay * cor_loss + self.ssl_weight_1 * factor_contrastive_loss

        return total_loss

    def gene_ranklist(self, topk=50):
        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 初始化一个全零的评分矩阵，大小为 [num_user, num_item]
        combined_score_matrix = torch.zeros((self.num_user, self.num_item))

        # 遍历每个因素
        for factor_index in range(self.n_factors):
            # 获取每个因素的用户和项目嵌入
            user_embeddings = self.user_factor_embedding[:, factor_index, :]  # [num_user, dim_E / n_factors]
            item_embeddings = self.item_factor_embedding[:, factor_index, :]  # [num_item, dim_E / n_factors]
            # 视觉和文本
            # user_visual_embeddings = self.user_visual_factor_embedding[:, factor_index, :]
            # user_textual_embeddings = self.user_textual_factor_embedding[:, factor_index, :]
            # visual_embeddings = self.visual_factor_embedding[:, factor_index, :]
            # textual_embeddings = self.textual_factor_embedding[:, factor_index, :]

            _user_embeddings = user_embeddings[:self.num_user].cpu()
            _item_embeddings = item_embeddings[:self.num_item].cpu()
            # _user_visual_embeddings = user_visual_embeddings[:self.num_user].cpu()
            # _user_textual_embeddings = user_textual_embeddings[:self.num_user].cpu()
            # _visual_embeddings = visual_embeddings[:self.num_item].cpu()
            # _textual_embeddings = textual_embeddings[:self.num_item].cpu()

            # 生成当前因素的评分矩阵
            score_matrix_1 = torch.matmul(_user_embeddings, _item_embeddings.t())
            # score_matrix_2 = torch.matmul(_user_visual_embeddings, _visual_embeddings.t())
            # score_matrix_3 = torch.matmul(_user_textual_embeddings, _textual_embeddings.t())

            # 累加当前因素的评分矩阵到综合评分矩阵中
            combined_score_matrix = combined_score_matrix + score_matrix_1

        # 将历史交互设置为极小值
        for row, col in self.user_item_dict.items():
            col = torch.LongTensor(list(col)) - self.num_user
            combined_score_matrix[row][col] = 1e-6

        # 选出每个用户的 top-k 个物品
        _, index_of_rank_list_train = torch.topk(combined_score_matrix, topk)
        # 总的top-k列表
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
            dim=0)

        # 返回三个推荐列表
        return all_index_of_rank_list
