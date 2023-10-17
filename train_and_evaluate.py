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
from utils import EarlyStopping

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger()
from arg_parser import parse_args

args = parse_args()
topk = args.topk


def train(model, train_loader, optimizer):
    model.train()
    sum_loss = 0.0
    if args.Model == "MMGCN" or "GRCN":
        for user_tensor, item_tensor in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(user_tensor, item_tensor)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model == "BPR" or "VBPR" or "NGCF" or "LightGCN":
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    return sum_loss


def evaluate(model, data, ranklist, topk):
    model.eval()
    with torch.no_grad():
        metrics = model.gene_metrics(data, ranklist, topk)
    return metrics


def train_and_evaluate(model, train_loader, val_data, test_data, optimizer, epochs):
    # 早停
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer)
        logger.info("Epoch {}, Loss: {:.5f}".format(epoch + 1, loss))
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
