import os
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch
import random

import dataload
from arg_parser import parse_args

random.seed(42)


def topk_sample(k, user_graph_dict, num_user):
    # 保存每个用户的最多k个相邻用户的索引
    user_graph_index = []
    count_num = 0
    # 如果某个用户没有足够的邻居，这个列表将被用作占位符
    tasike = [0] * k

    for i in range(num_user):
        if len(user_graph_dict[i][0]) < k:
            count_num += 1
            if len(user_graph_dict[i][0]) == 0:
                user_graph_index.append(tasike)
                continue
            user_graph_sample = user_graph_dict[i][0][:k]
            while len(user_graph_sample) < k:
                rand_index = np.random.randint(0, len(user_graph_sample))
                user_graph_sample.append(user_graph_sample[rand_index])
            user_graph_index.append(user_graph_sample)
            continue

        # 如果邻居数大于等于k，直接取前k个邻居
        user_graph_sample = user_graph_dict[i][0][:k]
        user_graph_index.append(user_graph_sample)

    return user_graph_index


def get_knn_adj_mat(mm_embeddings, item_topk):
    context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    # 将对角线上的值设为负无穷，防止选择自己为最近邻
    sim.fill_diagonal_(-float('inf'))
    _, knn_ind = torch.topk(sim, item_topk, dim=-1)

    return knn_ind


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_path
    print(f'Generating u-u matrix for {dataset} ...\n')
    dir_str = './Data/' + dataset
    # 加载训练数据
    train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat = dataload.data_load(
        args.data_path)

    image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
    text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)

    print(f'uu_topk: {args.uu_topk}, ii_topk: {args.ii_topk}')  # 输出uu_topk和ii_topk

    # ========================用户-项目图(U-I)============================
    adjusted_item_ids = train_data[:, 1] - num_user
    # 创建COO格式的稀疏矩阵
    user_item_graph = sp.coo_matrix((np.ones(len(train_data)),  # 填充1表示存在交互
                                     (train_data[:, 0], adjusted_item_ids)),  # 用户ID和调整后的项目ID
                                    shape=(num_user, num_item), dtype=np.float32)

    # ======================== 用户同构图 (U-U) =============================
    print('处理用户同构图 (U-U)')
    # 加载用户-用户图数据
    user_graph_dict = np.load(os.path.join(dir_str, 'user_graph_dict.npy'),
                              allow_pickle=True).item()

    # [num_user, user_topk]
    user_user_k_graph = topk_sample(args.uu_topk, user_graph_dict, num_user)

    # ======================== 项目语义图 (I-I) =============================
    print('处理项目语义图 (I-I)')
    # 处理邻接矩阵文件
    visual_adj_file = os.path.join(dir_str, 'ii_visual_{}.pt'.format(args.ii_topk))
    textual_adj_file = os.path.join(dir_str, 'ii_textual_{}.pt'.format(args.ii_topk))

    if os.path.exists(visual_adj_file) and os.path.exists(textual_adj_file):
        # [num_item, item_topk]
        item_item_k_visual_graph = torch.load(visual_adj_file)
        item_item_k_textual_graph = torch.load(textual_adj_file)
    else:
        image_graph = get_knn_adj_mat(image_embedding.weight.detach(), args.ii_topk)
        item_item_k_visual_graph = image_graph
        text_graph = get_knn_adj_mat(text_embedding.weight.detach(), args.ii_topk)
        item_item_k_textual_graph = text_graph
        del image_graph
        del text_graph
        torch.save(item_item_k_visual_graph, visual_adj_file)
        torch.save(item_item_k_textual_graph, textual_adj_file)

    # ======================== 构建超图序列 =============================
    print('构建超图序列')
    visual_file_name = 'hyperedges_visual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk)
    textual_file_name = 'hyperedges_textual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk)

    visual_file_path = os.path.join(dir_str, visual_file_name)
    textual_file_path = os.path.join(dir_str, textual_file_name)

    if os.path.exists(visual_file_path) and os.path.exists(textual_file_path):
        print('超边文件存在，直接加载')
        # 如果超边文件存在，直接加载
        hyperedges_visual = np.load(visual_file_path, allow_pickle=True).tolist()
        hyperedges_textual = np.load(textual_file_path, allow_pickle=True).tolist()
    else:
        # 设置相似用户和项目的最小和最大数量
        min_similar_users = 1
        max_similar_users = args.uu_topk

        min_similar_items = 1
        max_similar_items = args.ii_topk

        # 如果超边文件不存在，重新构建并保存
        hyperedges_visual = set()
        hyperedges_textual = set()

        for u_i in train_data:
            u = u_i[0]  # 用户索引
            i = u_i[1]  # 原始项目索引（从 num_user 到 num_user + num_item -1）
            adjusted_item_index = i - num_user  # 调整为从 0 开始的项目索引

            # 随机选择相似用户的数量
            num_similar_users = random.randint(min_similar_users, max_similar_users)
            # 随机采样相似用户
            similar_users = user_user_k_graph[u]
            similar_users = random.sample(similar_users, min(len(similar_users), num_similar_users))

            # 随机选择相似项目的数量
            num_similar_items = random.randint(min_similar_items, max_similar_items)
            # 对于视觉模态，随机采样相似项目
            similar_items_visual = item_item_k_visual_graph[adjusted_item_index]
            similar_items_visual = similar_items_visual[:num_similar_items]

            # 对于文本模态，同样处理
            similar_items_textual = item_item_k_textual_graph[adjusted_item_index]
            similar_items_textual = similar_items_textual[:num_similar_items]

            # 调整项目索引，加上 num_user 以区分用户和项目
            hyperedge_visual = [u] + similar_users + [i] + (similar_items_visual + num_user).tolist()
            hyperedge_textual = [u] + similar_users + [i] + (similar_items_textual + num_user).tolist()

            # 对超边中的节点进行排序，便于去重
            hyperedge_visual = tuple(sorted(hyperedge_visual))
            hyperedge_textual = tuple(sorted(hyperedge_textual))

            hyperedges_visual.add(hyperedge_visual)
            hyperedges_textual.add(hyperedge_textual)

        # 转换为列表
        hyperedges_visual = list(hyperedges_visual)
        hyperedges_textual = list(hyperedges_textual)

        # 将超边列表转换为 dtype=object 的 numpy 数组
        hyperedges_visual_array = np.array(hyperedges_visual, dtype=object)
        hyperedges_textual_array = np.array(hyperedges_textual, dtype=object)

        # 保存超边文件
        print('保存超边文件')
        np.save(visual_file_path, hyperedges_visual_array, allow_pickle=True)
        np.save(textual_file_path, hyperedges_textual_array, allow_pickle=True)


