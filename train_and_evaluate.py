"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/10/4 19:23
@File : train_and_evaluate.py
@function :
"""
import time
import logging
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
from utils import EarlyStopping, gene_metrics
import scipy.sparse as sp
from arg_parser import parse_args

args = parse_args()
topk = args.topk


def train(model, train_loader, optimizer, diffusionLoader=None, train_loader_sec_hop=None, diffusionLoader_visual=None,
          diffusionLoader_textual=None):
    model.train()
    sum_loss = 0.0
    all_ratings = None  # 用于BSPM模型的预测结果

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
                        'LGMRec', 'SelfCF', 'MENTOR', "FKAN_GCF", 'LightGODE', 'DHCF']:
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
    elif args.Model in ["LightGT"]:
        for users, items, mask, user_item in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, items, mask, user_item)
            loss.backward(retain_graph=True)
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ['MCLN']:
        for users, pos_items, neg_items, int_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items, int_items)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ["DiffMM"]:
        epDiLoss_image, epDiLoss_text = 0, 0
        denoise_opt_image = torch.optim.Adam(model.denoise_model_image.parameters(), lr=args.learning_rate,
                                             weight_decay=0)
        denoise_opt_text = torch.optim.Adam(model.denoise_model_text.parameters(), lr=args.learning_rate,
                                            weight_decay=0)

        for i, batch in enumerate(diffusionLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            iEmbeds = model.getItemEmbeds().detach()
            # uEmbeds = model.getUserEmbeds().detach()

            image_feats = model.getImageFeats().detach()
            text_feats = model.getTextFeats().detach()

            denoise_opt_image.zero_grad()
            denoise_opt_text.zero_grad()

            diff_loss_image, gc_loss_image = model.diffusion_model.training_losses(model.denoise_model_image,
                                                                                   batch_item,
                                                                                   iEmbeds, batch_index, image_feats)
            diff_loss_text, gc_loss_text = model.diffusion_model.training_losses(model.denoise_model_text, batch_item,
                                                                                 iEmbeds, batch_index, text_feats)

            loss_image = diff_loss_image.mean() + gc_loss_image.mean() * model.e_loss
            loss_text = diff_loss_text.mean() + gc_loss_text.mean() * model.e_loss

            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()

            loss = loss_image + loss_text
            loss.backward()
            denoise_opt_image.step()
            denoise_opt_text.step()

            logging.info('Diffusion Step %d/%d; Diffusion Loss %.6f' % (
                i, diffusionLoader.dataset.__len__() // args.batch_size, loss.item()))

        logging.info('')  # 空行
        logging.info('Start to re-build UI matrix')

        with torch.no_grad():
            u_list_image = []
            i_list_image = []
            edge_list_image = []

            u_list_text = []
            i_list_text = []
            edge_list_text = []

            sampling_noise = False
            sampling_steps = 0

            for _, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                # image
                denoised_batch = model.diffusion_model.p_sample(model.denoise_model_image, batch_item,
                                                                sampling_steps, sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=model.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)

                # text
                denoised_batch = model.diffusion_model.p_sample(model.denoise_model_text, batch_item,
                                                                sampling_steps, sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=model.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)

            # image
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            image_UI_matrix = model.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            image_UI_matrix = model.edgeDropper(image_UI_matrix)

            # text
            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            text_UI_matrix = model.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            text_UI_matrix = model.edgeDropper(text_UI_matrix)

        logging.info('UI matrix built!')

        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items, image_UI_matrix, text_UI_matrix)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ["GFormer"]:
        fixSteps = 10  # steps to train on the same sampled graph
        encoderAdj, decoderAdj, sub, cmp = None, None, None, None
        for i, (users, pos_items, neg_items) in enumerate(tqdm(train_loader, desc="Training")):
            if i % fixSteps == 0:
                # 每 fixSteps 次进行一次操作
                att_edge, add_adj = model.sampler(model.adj, model.getEgoEmbeds())
                encoderAdj, decoderAdj, sub, cmp = model.masker(add_adj, att_edge)
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items, encoderAdj, decoderAdj, sub, cmp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)  # 梯度裁剪
            optimizer.step()
            sum_loss += loss.item()
    elif args.Model in ['Grade']:
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        opt_gen_1 = torch.optim.Adam(model.generator_1.parameters(), lr=args.learning_rate, weight_decay=0)
        opt_gen_2 = torch.optim.Adam(model.generator_2.parameters(), lr=args.learning_rate, weight_decay=0)
        opt_gen_3 = torch.optim.Adam(model.generator_3.parameters(), lr=args.learning_rate, weight_decay=0)
        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            opt.zero_grad()
            loss_1 = model.loss_1(users, pos_items, neg_items)
            loss_1.backward()
            opt.step()

            opt.zero_grad()
            bpr_reg_loss = model.bpr_reg_loss(users, pos_items, neg_items)
            bpr_reg_loss.backward()
            opt.step()

            opt_gen_1.zero_grad()
            opt_gen_2.zero_grad()
            opt_gen_3.zero_grad()
            gen_loss = model.gen_loss(users, pos_items, neg_items)
            gen_loss.backward()
            opt_gen_1.step()
            opt_gen_2.step()
            opt_gen_3.step()
            loss = loss_1 + bpr_reg_loss + gen_loss
            sum_loss += loss.item()
    elif args.Model in ["BSPM"]:
        all_ratings = []
        # 生成从 0 到 num_user - 1 的用户序列
        user_ids = torch.arange(model.num_user).long().to(model.device)

        # 将用户序列划分为多个小批次，避免显存不足
        batch_size = 1024  # 可以根据实际情况调整批次大小
        num_batches = (model.num_user + batch_size - 1) // batch_size  # 计算批次数量

        for batch_id in tqdm(range(num_batches), desc="Training"):
            # 获取当前批次的用户 ID
            batch_users = user_ids[batch_id * batch_size: (batch_id + 1) * batch_size]
            # 获取当前批次用户的评分预测
            ret = model.getUsersRating(batch_users)  # 获取当前批次用户的评分预测
            # 将当前批次的预测结果添加到列表中
            all_ratings.append(ret)
        # 将所有批次的预测结果拼接成一个完整的评分矩阵
        all_ratings = torch.cat(all_ratings, dim=0)
        return all_ratings
    elif args.Model in ["DiffRec"]:
        optimizer_dnn = torch.optim.AdamW(model.dnn.parameters(), lr=model.learning_rate, weight_decay=0)
        # param_num = sum([param.nelement() for param in model.parameters()])  # 扩散模型的参数数量
        # print("Number of all parameters:", param_num)
        for i, batch in enumerate(diffusionLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
            optimizer_dnn.zero_grad()
            loss = model.training_losses(batch_item)  # 计算损失
            loss.backward()
            optimizer_dnn.step()
            sum_loss += loss.item()
    elif args.Model in ["CF_Diff"]:
        optimizer_CAM_AE = torch.optim.AdamW(model.CAM_AE.parameters(), lr=model.learning_rate, weight_decay=0)
        # param_num = sum([param.nelement() for param in model.parameters()])  # 扩散模型的参数数量
        # print("Number of all parameters:", param_num)
        for (batch_idx, batch), (batch_idx_2, batch_2) in zip(enumerate(diffusionLoader),
                                                              enumerate(train_loader_sec_hop)):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
            batch_item_2, batch_index_2 = batch_2
            batch_item_2, batch_index_2 = batch_item_2.cuda(), batch_index_2.cuda()
            optimizer_CAM_AE.zero_grad()
            loss = model.training_losses(batch_item, batch_item_2)  # 计算损失
            loss.backward()
            optimizer_CAM_AE.step()
            sum_loss += loss.item()
    elif args.Model in ["MHRec"]:
        epDiLoss_image, epDiLoss_text = 0, 0
        denoise_opt_image = torch.optim.Adam(model.denoise_model_image.parameters(), lr=args.learning_rate,
                                             weight_decay=0)
        denoise_opt_text = torch.optim.Adam(model.denoise_model_text.parameters(), lr=args.learning_rate,
                                            weight_decay=0)
        logging.info('Start to visual hyperedges diffusion')
        for i, batch in enumerate(diffusionLoader_visual):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            iEmbeds = model.getItemEmbeds().detach()
            uEmbeds = model.getUserEmbeds().detach()
            uEmbeds_visual = model.getUserEmbeds_visual().detach()
            image_feats = model.getImageFeats().detach()

            combined_node_embeds = torch.cat([uEmbeds, iEmbeds], dim=0)  # [num_users + num_items, embedding_dim]
            combined_visual_embeds = torch.cat([uEmbeds_visual, image_feats],
                                               dim=0)  # [num_users + num_items, embedding_dim]

            denoise_opt_image.zero_grad()
            diff_loss_image = model.diffusion_model.training_losses(model.denoise_model_image, batch_item, combined_node_embeds, combined_visual_embeds)
            # loss_image = diff_loss_image.mean() + gc_loss_image.mean() * model.e_loss
            loss_image = diff_loss_image.mean()
            epDiLoss_image += loss_image.item()
            loss_image.backward()
            denoise_opt_image.step()

            logging.info('Diffusion Step %d/%d; Diffusion Loss %.6f' % (
                i, diffusionLoader_visual.dataset.__len__() // args.batch_size, loss_image.item()))

        logging.info('Start to textual hyperedges diffusion')
        for i, batch in enumerate(diffusionLoader_textual):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            iEmbeds = model.getItemEmbeds().detach()
            uEmbeds = model.getUserEmbeds().detach()
            uEmbeds_textual = model.getUserEmbeds_textual().detach()
            textual_feats = model.getTextFeats().detach()

            combined_node_embeds = torch.cat([uEmbeds, iEmbeds], dim=0)  # [num_users + num_items, embedding_dim]
            combined_textual_embeds = torch.cat([uEmbeds_textual, textual_feats],
                                               dim=0)  # [num_users + num_items, embedding_dim]

            denoise_opt_text.zero_grad()
            diff_loss_text = model.diffusion_model.training_losses(model.denoise_model_text, batch_item, combined_node_embeds, combined_textual_embeds)
            # loss_text = diff_loss_text.mean() + gc_loss_text.mean() * model.e_loss
            loss_text = diff_loss_text.mean()
            epDiLoss_text += loss_text.item()
            loss_text.backward()
            denoise_opt_text.step()

            logging.info('Diffusion Step %d/%d; Diffusion Loss %.6f' % (
                i, diffusionLoader_textual.dataset.__len__() // args.batch_size, loss_text.item()))

        logging.info('')  # 空行
        logging.info('Start to re-build hypergraph matrix')

        with torch.no_grad():
            sampling_noise = False
            sampling_steps = 0

            # 初始化超图构建所需的列表
            rows_visual = []
            cols_visual = []
            data_visual = []
            hyperedge_counter = 0  # 超边计数器

            for _, batch in enumerate(diffusionLoader_visual):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                # visual
                denoised_batch_visual = model.diffusion_model.p_sample(
                    model.denoise_model_image, batch_item,
                    sampling_steps, sampling_noise
                )
                _, indices_visual = torch.topk(denoised_batch_visual, k=model.num_hypernodes)

                batch_size = batch_index.size(0)
                # 为每个样本分配一个唯一的超边索引
                hyperedge_indices = np.arange(hyperedge_counter, hyperedge_counter + batch_size)
                hyperedge_counter += batch_size

                # 将 indices_visual 和 hyperedge_indices 展平，并转换为 numpy 数组
                nodes = indices_visual.cpu().numpy().reshape(-1)
                hyperedges = np.repeat(hyperedge_indices, model.num_hypernodes)

                data = np.ones_like(nodes, dtype=np.float32)

                # 累积所有批次的数据
                rows_visual.append(nodes)
                cols_visual.append(hyperedges)
                data_visual.append(data)

            # 将累积的数据转换为 numpy 数组
            rows_visual = np.concatenate(rows_visual)
            cols_visual = np.concatenate(cols_visual)
            data_visual = np.concatenate(data_visual)

            num_nodes = model.num_user + model.num_item
            num_hyperedges_visual = hyperedge_counter

            H_visual = sp.coo_matrix(
                (data_visual, (rows_visual, cols_visual)),
                shape=(num_nodes, num_hyperedges_visual),
                dtype=np.float32
            )
            H_visual = H_visual.tocoo()
            indices = torch.from_numpy(np.vstack((H_visual.row, H_visual.col))).long()
            values = torch.from_numpy(H_visual.data).float()
            shape = H_visual.shape
            H_visual = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(model.device)

            # 对 textual 模态进行同样的处理
            rows_textual = []
            cols_textual = []
            data_textual = []
            hyperedge_counter = 0  # 超边计数器重置

            for _, batch in enumerate(diffusionLoader_textual):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                # textual
                denoised_batch_textual = model.diffusion_model.p_sample(
                    model.denoise_model_text, batch_item,
                    sampling_steps, sampling_noise
                )
                _, indices_textual = torch.topk(denoised_batch_textual, k=model.num_hypernodes)

                batch_size = batch_index.size(0)
                hyperedge_indices = np.arange(hyperedge_counter, hyperedge_counter + batch_size)
                hyperedge_counter += batch_size

                nodes = indices_textual.cpu().numpy().reshape(-1)
                hyperedges = np.repeat(hyperedge_indices, model.num_hypernodes)

                data = np.ones_like(nodes, dtype=np.float32)

                rows_textual.append(nodes)
                cols_textual.append(hyperedges)
                data_textual.append(data)

            rows_textual = np.concatenate(rows_textual)
            cols_textual = np.concatenate(cols_textual)
            data_textual = np.concatenate(data_textual)

            num_hyperedges_textual = hyperedge_counter

            H_textual = sp.coo_matrix(
                (data_textual, (rows_textual, cols_textual)),
                shape=(num_nodes, num_hyperedges_textual),
                dtype=np.float32
            )
            H_textual = H_textual.tocoo()
            indices = torch.from_numpy(np.vstack((H_textual.row, H_textual.col))).long()
            values = torch.from_numpy(H_textual.data).float()
            shape = H_textual.shape
            H_textual = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(model.device)


        logging.info('hypergraph matrix built!')

        # 生成超图的拉普拉斯矩阵 G
        # G_visual = model.generate_G_from_H(H_visual)
        # G_textual = model.generate_G_from_H(H_textual)

        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items, H_visual, H_textual)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    return sum_loss


def evaluate(model, data, ranklist, topk):
    model.eval()
    with torch.no_grad():
        metrics = gene_metrics(data, ranklist, topk)
    return metrics


def train_and_evaluate(model, train_loader, val_data, test_data, optimizer, epochs, eval_dataloader=None,
                       diffusionLoader=None, test_diffusionLoader=None, train_loader_sec_hop=None,
                       test_loader_sec_hop=None, diffusionLoader_visual=None, diffusionLoader_textual=None):
    model.train()
    # 早停
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # 如果模型是 BSPM，只跑一次 epoch
    if args.Model == "BSPM":
        all_ratings = train(model, train_loader, optimizer)
        rank_list = model.gene_ranklist(all_ratings)
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

        # BSPM 不需要多次循环，可以直接返回
        best_metrics = test_metrics
        return best_metrics

    for epoch in range(epochs):
        if args.Model in ["DualGNN", "DRAGON", "FREEDOM", 'POWERec', 'LayerGCN']:
            # 在每个epoch开始时，调用pre_epoch_processing方法
            model.pre_epoch_processing()
        if args.Model in ["DiffMM", "DiffRec"]:
            loss = train(model, train_loader, optimizer, diffusionLoader=diffusionLoader)
        elif args.Model in ["CF_Diff"]:
            loss = train(model, train_loader, optimizer, diffusionLoader=diffusionLoader,
                         train_loader_sec_hop=train_loader_sec_hop)
        elif args.Model in ["MHRec"]:
            loss = train(model, train_loader, optimizer, diffusionLoader_visual=diffusionLoader_visual,
                         diffusionLoader_textual=diffusionLoader_textual)
        else:
            loss = train(model, train_loader, optimizer)
        logging.info("Epoch {}, Loss: {:.5f}".format(epoch + 1, loss))

        if args.Model in ["LightGT"]:
            model.eval()  # 设置为评估模式
            rank_list = model.gene_ranklist(eval_dataloader)
            val_metrics = evaluate(model, val_data, rank_list, topk)
            test_metrics = evaluate(model, test_data, rank_list, topk)
        elif args.Model in ["DiffRec"]:
            model.eval()  # 设置为评估模式
            predict_items = []
            with torch.no_grad():
                for i, batch in enumerate(test_diffusionLoader):
                    batch_item, batch_index = batch  # [batchsize, num_item],[batchsize]
                    batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
                    prediction = model.p_sample(batch_item)  # [batchsize, num_item]

                    # **批量处理用户的交互历史，将交互物品设为极小值**
                    user_ids = batch_index.cpu().numpy()  # 获取当前批次的用户 ID
                    mask = torch.zeros_like(prediction, dtype=torch.bool)  # 初始化掩码张量为 False

                    # 为每个用户构建掩码
                    for user_idx, user_id in enumerate(user_ids):
                        interacted_items = model.user_item_dict.get(user_id, [])  # 获取用户的交互物品
                        if len(interacted_items) > 0:
                            # 将交互物品索引转为张量
                            interacted_items_tensor = torch.tensor(interacted_items, dtype=torch.long).to(
                                batch_item.device)
                            interacted_items_tensor = interacted_items_tensor - model.num_user  # 调整物品 ID 到索引范围

                            # 设置该用户的掩码为 True 表示这些物品应设为极小值
                            mask[user_idx, interacted_items_tensor] = True

                    # 将掩码应用到 prediction，将掩码为 True 的位置设为极小值
                    prediction.masked_fill_(mask, -np.inf)

                    # 使用 torch.topk 获取前50个物品
                    _, indices = torch.topk(prediction, 50, dim=1)
                    indices = (indices + model.num_user).cpu().tolist()  # 将张量转换为列表
                    predict_items.extend(indices)  # 添加结果

            predict_items = np.array(predict_items)
            val_metrics = evaluate(model, val_data, predict_items, topk)
            test_metrics = evaluate(model, test_data, predict_items, topk)
        elif args.Model in ["CF_Diff"]:
            model.eval()  # 设置为评估模式
            predict_items = []
            # 预先加载所有数据到内存中
            all_test_batches = list(test_diffusionLoader)
            all_test_sec_hop_batches = list(test_loader_sec_hop)
            with torch.no_grad():
                for (batch, batch_2) in zip(all_test_batches, all_test_sec_hop_batches):
                    batch_item, batch_index = batch
                    batch_item_2, batch_index_2 = batch_2
                    batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
                    batch_item_2, batch_index_2 = batch_item_2.cuda(), batch_index_2.cuda()
                    prediction = model.p_sample(batch_item, batch_item_2)  # [batchsize, num_item]

                    # **批量处理用户的交互历史，将交互物品设为极小值**
                    user_ids = batch_index.cpu().numpy()  # 获取当前批次的用户 ID
                    mask = torch.zeros_like(prediction, dtype=torch.bool)  # 初始化掩码张量为 False

                    # 为每个用户构建掩码
                    for user_idx, user_id in enumerate(user_ids):
                        interacted_items = model.user_item_dict.get(user_id, [])  # 获取用户的交互物品
                        if len(interacted_items) > 0:
                            # 将交互物品索引转为张量
                            interacted_items_tensor = torch.tensor(interacted_items, dtype=torch.long).to(
                                batch_item.device)
                            interacted_items_tensor = interacted_items_tensor - model.num_user  # 调整物品 ID 到索引范围

                            # 设置该用户的掩码为 True 表示这些物品应设为极小值
                            mask[user_idx, interacted_items_tensor] = True

                    # 将掩码应用到 prediction，将掩码为 True 的位置设为极小值
                    prediction.masked_fill_(mask, -np.inf)

                    # 使用 torch.topk 获取前50个物品
                    _, indices = torch.topk(prediction, 50, dim=1)
                    indices = (indices + model.num_user).cpu().tolist()  # 将张量转换为列表
                    predict_items.extend(indices)  # 添加结果

            predict_items = np.array(predict_items)
            val_metrics = evaluate(model, val_data, predict_items, topk)
            test_metrics = evaluate(model, test_data, predict_items, topk)
        else:
            model.eval()  # 设置为评估模式
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
