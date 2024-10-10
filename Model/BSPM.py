"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/10/8 9:48
@File : BSPM.py
@function :
"""
import numpy as np
import torch
import torch_sparse
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F
from sparsesvd import sparsesvd
from torchdiffeq import odeint


class BSPM(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, K_s, T_s, K_b, T_b, idl_beta, device):
        super(BSPM, self).__init__()
        self.ua_embedding = None
        self.ia_embedding = None
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.device = device

        self.solver_idl = 'euler'  # 低通滤波器的求解方法，默认欧拉法
        self.solver_blr = 'euler'  # 模糊操作的求解方程的方法
        self.solver_shr = 'euler'  # 锐化操作的求解方程的方法

        self.K_idl = 1  # IDL操作中的时间步长 T_idl / \tau
        self.T_idl = 1  # IDL操作的总时间
        self.K_b = K_b  # 模糊操作的时间步长 T_b / \tau
        self.T_b = T_b  # 模糊操作的总时间
        self.K_s = K_s  # 锐化操作的时间步长  T_s / \tau
        self.T_s = T_s  # 锐化操作的总时间

        self.final_sharpening = True
        self.sharpening_off = False
        self.t_point_combination = False

        self.factor_dim = 256  # 分解的因子维度
        self.idl_beta = idl_beta  # IDL操作中的超参数β

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  # 填充1表示存在交互
                                                 (edge_index[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                                shape=(self.num_user, self.num_item), dtype=np.float32)

        self.norm_adj, self.linear_Filter = self.get_norm_adj_mat()

        # 生成时间序列
        self.idl_times = torch.linspace(0, self.T_idl, self.K_idl + 1).float().to(self.device)
        self.blurring_times = torch.linspace(0, self.T_b, self.K_b + 1).float().to(self.device)
        self.sharpening_times = torch.linspace(0, self.T_s, self.K_s + 1).float().to(self.device)

        ut, s, self.vt = sparsesvd(self.norm_adj, self.factor_dim)
        del ut
        del s

        left_mat = self.d_mat_i @ self.vt.T
        right_mat = self.vt @ self.d_mat_i_inv
        self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(
            right_mat).to(self.device)

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

        self.d_mat_i = D
        self.d_mat_i_inv = sp.diags(1 / diag)

        # 使用D对A进行归一化：L = D^-0.5 * A * D^-0.5
        L = D @ A @ D
        # 将归一化后的L转换为COO格式的稀疏矩阵，以便后续转换为torch稀疏张量
        L = sp.coo_matrix(L)

        # 使用SciPy进行矩阵乘法计算二阶邻接矩阵 linear_Filter
        linear_Filter = L.T @ L  # 这里是SciPy的稀疏矩阵乘法
        linear_Filter = linear_Filter.tocoo()

        row = linear_Filter.row
        col = linear_Filter.col
        # 创建torch张量来表示稀疏矩阵的坐标(indices)和值(values)
        rows_and_cols = np.array([row, col])  # 将行和列的列表转换成numpy数组
        i = torch.tensor(rows_and_cols, dtype=torch.long)  # 从numpy数组创建张量
        data = torch.FloatTensor(linear_Filter.data)
        # 创建torch的稀疏张量来表示归一化的邻接矩阵
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32).to(self.device)

        return L.tocsc(), SparseL

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def sharpenFunction(self, t, r):
        # out = r @ self.linear_Filter
        out = torch.sparse.mm(self.linear_Filter, r.t()).t()  # 稀疏矩阵乘法
        return -out

    def getUsersRating(self, batch_users):
        if not torch.is_tensor(self.norm_adj):
            adj_mat = self.convert_sp_mat_to_sp_tensor(self.norm_adj).to_dense()

        batch_test = adj_mat[batch_users, :].to(self.device)
        self.left_mat = self.left_mat.to(self.device)
        self.right_mat = self.right_mat.to(self.device)
        # 低通滤波
        idl_out = torch.mm(batch_test, self.left_mat @ self.right_mat)

        # 模糊
        blurred_out = torch.sparse.mm(self.linear_Filter, batch_test.t()).t()

        del batch_test

        # 锐化
        if self.sharpening_off == False:
            if self.final_sharpening == True:
                sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta * idl_out + blurred_out,
                                       t=self.sharpening_times, method=self.solver_shr)

            else:
                sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times,
                                       method=self.solver_shr)

        if self.t_point_combination == True:
            if self.sharpening_off == False:
                U_2 = torch.mean(torch.cat([blurred_out.unsqueeze(0), sharpened_out[1:, ...]], dim=0), dim=0)

            else:
                U_2 = blurred_out
                del blurred_out
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
                del sharpened_out
            else:
                U_2 = blurred_out
                del blurred_out

        if self.final_sharpening == True:
            if self.sharpening_off == False:
                ret = U_2
            elif self.sharpening_off == True:
                ret = self.idl_beta * idl_out + U_2
        else:
            ret = self.idl_beta * idl_out + U_2

        return ret

    def gene_ranklist(self, all_ratings, topk=50):
        # 不同阶段的评估（例如训练、验证和测试）
        all_index_of_rank_list = torch.LongTensor([])

        # 生成评分矩阵
        score_matrix = all_ratings[:self.num_user, self.num_user:]

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
