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
        num_user = 28974
        num_item = 1922
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


# --------------DiffMM---------------------
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