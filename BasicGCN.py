"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/2 19:32
@File : BasicGCN.py
@function :
"""
import torch
import pdb
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, dropout_adj
from torch_geometric.nn.inits import uniform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ==============一般的BasicGCN===============================
class BasicGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(BasicGCN, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        # aggr可以选择 "mean"  "max"  "add"
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # ==============1.自循环===============
        edge_index = edge_index.long()
        # 移除自循环
        # edge_index, _ = remove_self_loops(edge_index)
        # 添加自循环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # ==============2.对节点特征矩阵做线性变换=================
        x = self.lin(x)
        # ==============3.计算节点的归一化系数================
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 步骤4-6
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # 步骤4
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
