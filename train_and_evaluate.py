"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/4 19:23
@File : train_and_evaluate.py
@function :
"""
import time
import logging
import torch
from tqdm import tqdm
from utils import EarlyStopping, gene_metrics

from arg_parser import parse_args

args = parse_args()
topk = args.topk


def train(model, train_loader, optimizer):
    model.train()
    sum_loss = 0.0
    if args.Model in ["MMGCN", "GRCN"]:
        for user_tensor, item_tensor in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(user_tensor, item_tensor)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ["BPR", "VBPR", "NGCF", "LightGCN", "DGCF", "DualGNN", "BM3", "DRAGON", "FREEDOM", "SLMRec",
                        "MGAT",  'MMGCL', 'DDRec', 'DCMF', 'SGL', 'MultVAE']:
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ["LATTICE", "MICRO", "LATTICE_AE"]:
        build_item_graph = True
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items, build_item_graph=build_item_graph)
            build_item_graph = False
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ['NCL']:
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            # 执行聚类
            model.e_step()
            loss = model.loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    return sum_loss


def evaluate(model, data, ranklist, topk):
    model.eval()
    with torch.no_grad():
        metrics = gene_metrics(data, ranklist, topk)
    return metrics


def train_and_evaluate(model, train_loader, val_data, test_data, optimizer, epochs):
    # 早停
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(epochs):
        if args.Model in ["DualGNN", "DRAGON", "FREEDOM"]:
            # 在每个epoch开始时，调用pre_epoch_processing方法
            model.pre_epoch_processing()
        loss = train(model, train_loader, optimizer)
        logging.info("Epoch {}, Loss: {:.5f}".format(epoch + 1, loss))
        rank_list = model.gene_ranklist()
        val_metrics = evaluate(model, val_data, rank_list, topk)
        test_metrics = evaluate(model, test_data, rank_list, topk)

        # 输出验证集的评价指标
        logging.info('Validation Metrics:')
        for k, metrics in val_metrics.items():
            metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
            logging.info(f"{k}: {' | '.join(metrics_strs)}")

        # 输出测试集的评价指标
        logging.info('Test Metrics:')
        for k, metrics in test_metrics.items():
            metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
            logging.info(f"{k}: {' | '.join(metrics_strs)}")

        recall = test_metrics[max(topk)]['recall']
        early_stopping(recall, test_metrics)  # 确保将 metrics 传递给 early_stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 输出最佳的评价指标
    best_metrics = early_stopping.best_metrics
    logging.info('Best Test Metrics:')
    for k, metrics in best_metrics.items():
        metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
        logging.info(f"{k}: {' | '.join(metrics_strs)}")

    return early_stopping.best_metrics
