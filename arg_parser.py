"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/9/28 17:08
@File : arg_parser.py
@function :
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run ChaoRec.")
    # 模型和数据集选择
    parser.add_argument('--Model', nargs='?', default='BM3', help='Model name')
    parser.add_argument('--data_path', nargs='?', default='yelp', help='Input data path.')
    # 超参数选择
    parser.add_argument('--lr', type=float, nargs='+', default=1e-3, help='Learning rates')
    parser.add_argument('--feature_embed', type=int, default=64, help='Feature Embedding size')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation mode.')
    parser.add_argument('--weight_decay', type=float, nargs='+', default=1e-6, help='Weight decay.')
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout.')
    parser.add_argument('--layer', type=int, default=1, help='conv_layers.')
    parser.add_argument('--corDecay', type=float, default=0.001, help='CorDecay.')
    parser.add_argument('--n_factors', type=int, default=4, help='the number of hidden factor k.')
    parser.add_argument('--n_iterations', type=int, default=3, help='the number of iteration.')
    # 一些默认参数
    parser.add_argument('--seed', type=int, default=42, help='Number of seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--topk', type=float, nargs='+', default=[5, 10, 20], help='topK')

    return parser.parse_args()
