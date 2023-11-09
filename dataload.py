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

args = parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%a %d %b %Y %H:%M:%S')


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

    logging.info('==============加载数据集================')
    logging.info('The number of users: %d', num_user)
    logging.info('The number of items: %d', num_item)

    return train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_user, num_user + num_item))
        self.model_name = args.Model

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        # (tensor([0, 0]), tensor([769, 328]))
        if self.model_name in ["MMGCN", "GRCN"]:
            return torch.LongTensor([user, user]), torch.LongTensor([pos_item, neg_item])
        else:
            return [int(user), int(pos_item), int(neg_item)]
