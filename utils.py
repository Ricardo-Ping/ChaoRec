"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/9/28 17:12
@File : utils.py
@function :
"""
import datetime

import torch
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置是否使用GPU
def gpu():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # logging.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logging.info("CUDA is not available, using CPU.")
    return device


# 获取当前时间
def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')
    return cur


# 早停
class EarlyStopping:
    def __init__(self, patience=50, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metrics = None

    def __call__(self, score, metrics):  # 参数列表中增加了 metrics
        if self.best_score is None:
            self.best_score = score
            self.best_metrics = metrics
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics = metrics
            self.counter = 0
