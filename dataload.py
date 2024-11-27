"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/9/28 20:37
@File : dataload.py
@function :
"""
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from arg_parser import parse_args
import logging
import scipy.sparse as sp

args = parse_args()


def data_load(dataset, has_v=True, has_t=True):
    num_user = None
    num_item = None
    dir_str = './Data/' + dataset

    train_data = np.load(dir_str + '/train.npy', allow_pickle=True)  # (len of train, 2)
    # 源代码中验证集和测试集是一个list
    val_data = np.load(dir_str + '/val.npy', allow_pickle=True)
    test_data = np.load(dir_str + '/test.npy', allow_pickle=True)
    user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
    v_feat = np.load(dir_str + '/v_feat.npy', allow_pickle=True) if has_v else None
    t_feat = np.load(dir_str + '/t_feat.npy', allow_pickle=True) if has_t else None
    v_feat = torch.tensor(v_feat, dtype=torch.float).cuda() if has_v else None
    t_feat = torch.tensor(t_feat, dtype=torch.float).cuda() if has_t else None

    if dataset == 'yelp':
        num_user = 11123
        num_item = 8128
    if dataset == 'clothing':
        num_user = 18072
        num_item = 11384
    if dataset == 'baby':
        num_user = 12351
        num_item = 4794
    if dataset == 'sports':
        num_user = 28940
        num_item = 15207
    if dataset == 'beauty':
        num_user = 15482
        num_item = 8643
    if dataset == 'electronics':
        num_user = 150179
        num_item = 51901
    if dataset == 'microlens':
        num_user = 46420
        num_item = 14079

    return train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_user, num_user + num_item))
        self.model_name = args.Model
        self.src_len = 50

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        # --------------------MCLN-------------------
        while True:
            int_items = random.sample(self.all_set, 1)[0]
            if int_items not in self.user_item_dict[user]:
                break

        # (tensor([0, 0]), tensor([769, 328]))
        if self.model_name in ["MMGCN", "GRCN"]:
            return torch.LongTensor([user, user]), torch.LongTensor([pos_item, neg_item])
        elif self.model_name in ["LightGT"]:
            temp = list(self.user_item_dict[user])
            random.shuffle(temp)
            if len(temp) > self.src_len:
                mask = torch.ones(self.src_len + 1) == 0
                temp = temp[:self.src_len]
            else:
                mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
                temp.extend([self.num_user for i in range(self.src_len - len(temp))])

            user_item = torch.tensor(temp) - self.num_user
            user_item = torch.cat((torch.tensor([-1]), user_item))

            return [torch.LongTensor([user, user]), torch.LongTensor([pos_item, neg_item]), mask, user_item]
        elif self.model_name in ["MCLN"]:
            return [int(user), int(pos_item), int(neg_item), int(int_items)]
        else:
            return [int(user), int(pos_item), int(neg_item)]


# --------------LightGT---------------------
class EvalDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict):
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_user, num_user + num_item))
        self.model_name = args.Model
        self.src_len = 20

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = index
        temp = list(self.user_item_dict[user])
        random.shuffle(temp)

        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
            # --------------------MCLN-------------------
        while True:
            int_items = random.sample(self.all_set, 1)[0]
            if int_items not in self.user_item_dict[user]:
                break

        if len(temp) > self.src_len:
            mask = torch.ones(self.src_len + 1) == 0
            temp = temp[:self.src_len]
        else:
            mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
            temp.extend([self.num_user for i in range(self.src_len - len(temp))])

        user_item = torch.tensor(temp) - self.num_user
        user_item = torch.cat((torch.tensor([-1]), user_item))

        return torch.LongTensor([user]), user_item, mask


# --------------DiffMM/DiffRec---------------------
class DiffusionData(Dataset):
    def __init__(self, num_user, num_item, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index

        adjusted_item_ids = self.edge_index[:, 1] - self.num_user

        # 创建COO格式的稀疏矩阵
        self.interaction_matrix = sp.coo_matrix(
            (np.ones(len(self.edge_index)),
             (self.edge_index[:, 0], adjusted_item_ids)),
            shape=(self.num_user, self.num_item), dtype=np.float32
        )
        # 将稀疏矩阵转换为稠密矩阵
        self.data = torch.FloatTensor(self.interaction_matrix.A)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item, index


class HyperDiffusionData(Dataset):
    def __init__(self, num_user, num_item, hypergraph_seq):
        self.hypergraph_seq = hypergraph_seq
        self.num_user = num_user
        self.num_item = num_item

        # 创建超边与节点的稀疏矩阵
        # 行表示超边，列表示节点，矩阵中的值为1表示该节点属于该超边
        row_indices = []
        col_indices = []
        data_values = []

        for hyperedge_idx, hyperedge in enumerate(hypergraph_seq):
            for node in hyperedge:
                row_indices.append(hyperedge_idx)
                col_indices.append(node)
                data_values.append(1)

        # 创建COO格式的稀疏矩阵，shape是 (num_hyperedges, num_user + num_item)
        self.hyperedges_matrix = sp.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(len(hypergraph_seq), num_user + num_item),
            dtype=np.float32
        )

        # 将稀疏矩阵转换为稠密矩阵
        self.data = torch.FloatTensor(self.hyperedges_matrix.toarray())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 返回指定超边对应的节点交互信息
        item = self.data[index]
        return item, index


# --------------CF-Diff---------------------
class DiffusionData_sec_hop(Dataset):
    def __init__(self, num_user, num_item, edge_index):
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index

        adjusted_item_ids = self.edge_index[:, 1] - self.num_user

        self.interaction_matrix = sp.csr_matrix(
            (np.ones(len(self.edge_index)),
             (self.edge_index[:, 0], adjusted_item_ids)),
            shape=(self.num_user, self.num_item), dtype=np.float32
        )

        # 将稀疏矩阵转换为密集矩阵形式
        data = self.interaction_matrix.todense()

        # 计算每个用户的二跳信息
        hop2 = self.get_2hop_item_based(torch.tensor(data, dtype=torch.float32))

        # 将 hop2 信息转换为 FloatTensor
        self.data = torch.FloatTensor(hop2)

    def get_2hop_item_based(self, data):
        # 初始化空张量
        sec_hop_infos = torch.empty(len(data), len(data[0]))  # [n_user, n_item]

        # 对所有用户的物品交互信息按列求和，得到一个物品的交互总数向量，然后除以用户数 n_user
        sec_hop_inters = torch.sum(data, dim=0) / self.num_user

        for i, row in enumerate(data):
            # 找到当前用户未交互的物品索引（交互数接近0）
            zero_indices = torch.nonzero(row < 0.000001).squeeze()
            if i % 1000 == 0:
                print(f"Processing user {i}")

            # 将二跳交互信息赋给当前用户
            sec_hop_infos[i] = sec_hop_inters
            # 将用户未交互过的物品信息置为 0
            sec_hop_infos[i][zero_indices] = 0

        return sec_hop_infos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 返回用户的二跳交互信息
        item = self.data[index]
        return item, index
