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
                        "MGAT", 'MMGCL', 'DDRec', 'SGL', 'MultVAE', 'MacridVAE', 'LightGCL', 'HCCF', 'MGCL',
                        'MGCN', 'POWERec', 'MVGAE', 'LayerGCN', 'DCCF', 'DualVAE', 'SimGCL', 'XSimGCL', 'GraphAug',
                        'LGMRec']:
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ['MMSSL']:
        # 鉴别器D的参数
        optim_D = torch.optim.Adam(model.D.parameters(), lr=3e-4, betas=(0.5, 0.9))
        # 模型参数
        optimizer_D = torch.optim.AdamW(
            [
                {'params': model.parameters()},
            ],
            lr=args.learning_rate)  # 0.00055
        # 使用enumerate获取批次索引idx和数据
        for idx, (users, pos_items, neg_items) in enumerate(tqdm(train_loader, desc="Training")):
            optim_D.zero_grad()
            loss_D = model.loss_D(users, pos_items, neg_items)
            loss_D.backward()
            optim_D.step()

            optimizer_D.zero_grad()
            batch_loss = model.loss(users, pos_items, neg_items, idx)
            batch_loss.backward(retain_graph=False)
            optimizer_D.step()

            loss = loss_D + batch_loss
            sum_loss += loss.item()
    elif args.Model in ['AdaGCL']:
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        opt_gen_1 = torch.optim.Adam(model.generator_1.parameters(), lr=args.learning_rate, weight_decay=0)
        opt_gen_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.generator_2.parameters()),
                                     lr=args.learning_rate, weight_decay=0, eps=0.001)
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            opt.zero_grad()
            opt_gen_1.zero_grad()
            opt_gen_2.zero_grad()
            loss_1, out1, out2 = model.loss_1(users, pos_items, neg_items)
            loss_1.backward()
            opt.step()
            opt.zero_grad()
            loss_2 = model.loss_2(users, pos_items, neg_items, out1, out2)
            loss_2.backward()
            opt.step()
            opt.zero_grad()
            bpr_reg_loss = model.bpr_reg_loss(users, pos_items, neg_items)
            bpr_reg_loss.backward()
            gen_loss = model.gen_loss(users, pos_items, neg_items)
            gen_loss.backward()
            opt.step()
            opt_gen_1.step()
            opt_gen_2.step()
            loss = loss_1 + loss_2 + bpr_reg_loss + gen_loss
            sum_loss += loss.item()
    elif args.Model in ["LATTICE", "MICRO"]:
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
    elif args.Model in ['VGCL']:
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            # 执行聚类
            model.forward()
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
        if args.Model in ["DualGNN", "DRAGON", "FREEDOM", 'POWERec', 'LayerGCN']:
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
