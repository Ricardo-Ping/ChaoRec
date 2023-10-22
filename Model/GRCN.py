"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/17 21:13
@File : GRCN.py
@function :
"""
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, dropout_adj
import torch.nn as nn
import torch.nn.functional as F

from metrics import precision_at_k, ndcg_at_k, recall_at_k, hit_rate_at_k, map_at_k


#  图注意力网络 用于多模态嵌入
class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super(GATConv, self).__init__(aggr='add')  # , **kwargs)
        self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # 移除自循环
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            # 增加自循环
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # 计算图中每条边的注意力系数  GRCN中的等式2
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        self.alpha = softmax(self.alpha, edge_index_i, num_nodes=size_i)
        return x_j * self.alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


# GraphSAGE  用于id嵌入
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='mean', **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        # 权重向量，用于调整节点特征
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        # 初始化了用户和项目节点的嵌入
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))
        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)
        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding
        # 使图变成无向的
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        #  L2 归一化
        x = F.normalize(x)

        # 两次 GraphSAGE 卷积
        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)
        x_hat_1 = F.leaky_relu(x_hat_1)

        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
        x_hat_2 = F.leaky_relu(x_hat_2)

        return x + x_hat_1 + x_hat_2


class CGCN(torch.nn.Module):
    def __init__(self, features, num_user, num_item, dim_C, aggr_mode, num_routing):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing  # 路由次数，控制图卷积层的迭代次数
        self.dim_C = dim_C
        # 用户偏好
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C)

        self.dim_feat = features.size(1)
        self.features = features
        self.MLP = nn.Linear(self.dim_feat, self.dim_C)

    def forward(self, edge_index):
        # 等式1
        features = F.leaky_relu(self.MLP(self.features))

        preference = F.normalize(self.preference)
        features = F.normalize(features)

        # 路由循环
        for i in range(self.num_routing):
            # 这里x的维度应该是(num_user + num_item , dim_C)
            x = torch.cat((preference, features), dim=0)
            # 经过一层图注意力网络
            x_hat_1 = self.conv_embed_1(x, edge_index)
            # 更新用户偏好  等式3
            preference = preference + x_hat_1[:self.num_user]
            preference = F.normalize(preference)

        # 更新x
        x = torch.cat((preference, features), dim=0)
        # 无向图
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        x_hat_1 = self.conv_embed_1(x, edge_index)
        x_hat_1 = F.leaky_relu(x_hat_1)

        # 返回嵌入和注意力系数
        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)


class GRCN(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, reg_weight,
                 v_feat, t_feat,
                 dim_E, dim_C, dropout, device,
                 aggr_mode, weight_mode='confid', fusion_mode='concat',
                 num_routing=3,pruning='True'):
        super(GRCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        # 详细说明了多模态相关积分的类型 （mean：实现没有置信向量的平均积分，max：实现无置信向量的最大积分，confid：（默认情况下）实现与置信向量的最大积分）
        self.weight_mode = weight_mode  # confid
        # 指定了预测图层中的用户和项目表示类型（concat,mean,id:仅id嵌入）
        self.fusion_mode = fusion_mode  # concat
        self.weight = torch.tensor([[1.0], [-1.0]]).cuda()
        self.reg_weight = reg_weight
        self.dropout = dropout
        self.device = device

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().cuda()
        # 初始化了一个ID的图卷积网络（EGCN）
        self.id_gcn = EGCN(num_user, num_item, dim_E, aggr_mode)
        self.v_feat = v_feat
        self.t_feat = t_feat
        # 修剪操作
        self.pruning = pruning  # True 剪纸

        # 针对不同模态的内容图卷积网络（CGCN）
        num_model = 0
        if v_feat is not None:
            self.v_gcn = CGCN(self.v_feat, num_user, num_item, dim_C, aggr_mode, num_routing)
            num_model += 1

        if t_feat is not None:
            self.t_gcn = CGCN(self.t_feat, num_user, num_item, dim_C, aggr_mode, num_routing)
            num_model += 1

        # 存储每个用户和项目对于每个模型的特定配置或权重的
        self.model_specific_conf = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, num_model))))

        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))).cuda()

    def forward(self):
        representation = None
        t_rep = None
        v_rep = None
        weight = None
        content_rep = None
        num_modal = 0
        # 边丢弃
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)

        if self.v_feat is not None:
            num_modal += 1
            # 返回嵌入和注意力系数
            v_rep, weight_v = self.v_gcn(edge_index)
            weight = weight_v
            content_rep = v_rep
        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight = weight_t
                content_rep = t_rep
            else:
                content_rep = torch.cat((content_rep, t_rep), dim=1)
                if self.weight_mode == 'mean':
                    weight = weight + weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)

        # 决定权重模式
        if self.weight_mode == 'mean':
            weight = weight / num_modal
        elif self.weight_mode == 'max':
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)
        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]),
                                   dim=0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)

        # 剪枝
        if self.pruning:
            weight = torch.relu(weight)

        # id嵌入图卷积
        id_rep = self.id_gcn(edge_index, weight)

        # 决定融合模式
        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
        elif self.fusion_mode == 'id':
            representation = id_rep
        elif self.fusion_mode == 'mean':
            representation = (id_rep + v_rep + t_rep) / 3

        self.result = representation
        return representation

    def loss(self, user_tensor, item_tensor):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))

        reg_embedding_loss = (
                    self.id_gcn.id_embedding[user_tensor] ** 2 + self.id_gcn.id_embedding[item_tensor] ** 2).mean()
        reg_content_loss = torch.zeros(1).cuda()
        if self.v_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor] ** 2).mean()
        if self.t_feat is not None:
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor] ** 2).mean()

        # reg_confid_loss = (self.model_specific_conf ** 2).mean()

        reg_loss = reg_embedding_loss + reg_content_loss

        reg_loss = self.reg_weight * reg_loss

        total_loss = loss + reg_loss
        return total_loss

    def gene_ranklist(self, step=200, topk=50):
        # step需要小于用户数量才能达到分批的效果不然会报错
        # 用户嵌入和项目嵌入
        user_tensor = self.result[:self.num_user].cpu()
        item_tensor = self.result[self.num_user:self.num_user + self.num_item].cpu()

        # 分批处理数据
        start_index = 0
        end_index = self.num_user if step is None else step

        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        while self.num_user >= end_index > start_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            # 生成评分矩阵
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            # 将历史交互设置为极小值
            for row, col in self.user_item_dict.items():
                if start_index <= row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))- self.num_user
                    score_matrix[row][col] = 1e-5

            # 选出每个用户的 top-k 个物品
            _, index_of_rank_list_train = torch.topk(score_matrix, topk)
            # 总的top-k列表
            all_index_of_rank_list = torch.cat(
                (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
                dim=0)

            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        # 返回三个推荐列表
        return all_index_of_rank_list

    def gene_metrics(self, val_data, rank_list, k_list):
        # 初始化存储评估指标的字典
        metrics = {k: {'precision': 0, 'recall': 0, 'ndcg': 0, 'hit_rate': 0, 'map': 0} for k in k_list}

        for data in val_data:
            user = data[0]
            pos_items = data[1:]
            ranked_items = rank_list[user].tolist()

            # 对每个 k 值计算评估指标
            for k in k_list:
                metrics[k]['precision'] += precision_at_k(ranked_items, pos_items, k)
                metrics[k]['recall'] += recall_at_k(ranked_items, pos_items, k)
                metrics[k]['ndcg'] += ndcg_at_k(ranked_items, pos_items, k)
                metrics[k]['hit_rate'] += hit_rate_at_k(ranked_items, pos_items, k)
                metrics[k]['map'] += map_at_k(ranked_items, pos_items, k)

        num_users = len(val_data)

        # 计算评估指标的平均值
        for k in k_list:
            metrics[k]['precision'] /= num_users
            metrics[k]['recall'] /= num_users
            metrics[k]['ndcg'] /= num_users
            metrics[k]['hit_rate'] /= num_users
            metrics[k]['map'] /= num_users

        return metrics
