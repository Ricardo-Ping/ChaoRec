# ChaoRec（超级推荐）

ChaoRec 是一个基于 Python + PyTorch 的统一推荐算法研究框架，覆盖协同过滤、图推荐、多模态推荐、扩散推荐等方向。  
当前仓库已集成 **54 个模型**，并支持通过 YAML 进行网格搜索式超参组合训练。

## 1. 项目结构

```text
ChaoRec/
├─ Data/                 # 数据集目录（baby/beauty/clothing/...）
├─ Model/                # 模型实现
├─ Model_YAML/           # 各模型超参搜索空间
├─ main.py               # 训练入口
├─ dataload.py           # 数据读取与采样
├─ train_and_evaluate.py # 训练与评估
└─ arg_parser.py         # 通用参数
```

## 2. 环境要求

- Python 3.8（项目依赖按 `requirements.txt` 固定）
- PyTorch 1.11.0
- torch-geometric 2.1.0 及相关扩展
- 建议使用 CUDA 环境（代码中存在 `.cuda()` 调用）

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 数据准备

每个数据集目录（如 `Data/microlens`）至少需要：

- `train.npy`
- `val.npy`
- `test.npy`
- `user_item_dict.npy`
- `v_feat.npy`
- `t_feat.npy`

默认已支持的数据集名（`--data_path`）：

- `baby`
- `beauty`
- `clothing`
- `electronics`
- `microlens`
- `netfilx`
- `sports`

说明：

- 交互数据中 item id 采用“全局偏移”形式（与代码实现一致，即 item 通常从 `num_user` 开始编码）。
- `num_user / num_item` 在 `dataload.py` 中按数据集名固定映射；新增数据集时需同步补充映射。

## 4. 快速开始

示例 1：使用默认配置训练（默认 `COHESION + microlens`）

```bash
python main.py
```

示例 2：指定模型与数据集

```bash
python main.py --Model LightGCN --data_path sports
```

示例 3：覆盖部分通用参数

```bash
python main.py --Model BM3 --data_path clothing --num_epoch 200 --batch_size 2048
```

训练日志会写入：

```text
log/{Model}_{data_path}.log
```

## 5. YAML 超参搜索机制

每个模型在 `Model_YAML/{Model}.yaml` 中定义超参搜索空间。  
`main.py` 会读取 `hyper_parameters` 指定的键，并做笛卡尔积遍历。

示例（`Model_YAML/LightGCN.yaml`）：

```yaml
n_layers: [1, 2, 3]
learning_rate: [0.001]
reg_weight: [0.001]
hyper_parameters: ["n_layers", "learning_rate", "reg_weight"]
```

你可以通过修改 YAML 控制搜索范围；命令行参数会先加载，随后被 YAML 组合覆盖。

## 6. 额外预处理脚本

部分模型需要额外图结构文件，可按需执行：

- `dualgnn-gen-u-u-matrix.py`
  - 生成 `user_graph_dict.npy`
- `gen_hypergraph_u_i.py`
  - 生成超边文件（如 `hyperedges_visual_u10_i10.npy`、`hyperedges_textual_u10_i10.npy`）

示例：

```bash
python dualgnn-gen-u-u-matrix.py --data_path microlens
python gen_hypergraph_u_i.py --data_path microlens --uu_topk 10 --ii_topk 10
```

## 7. 原始模型论文表

以下两个表格对应你原 README 里列出的“一般推荐模型”和“多模态推荐模型”，并补充了论文检索链接。

### 7.1 一般推荐模型（原列表）

| 模型 | 年份 | 论文 | 链接 |
|---|---:|---|---|
| BPR | 2016 | Bayesian Personalized Ranking with Multi-Channel User Feedback | [论文链接](https://doi.org/10.1145/2959100.2959163) |
| DGCF | 2020 | Disentangled Graph Collaborative Filtering | [论文链接](https://doi.org/10.1145/3397271.3401137) |
| NGCF | 2019 | Neural Graph Collaborative Filtering | [论文链接](http://arxiv.org/abs/1905.08108v2) |
| LightGCN | 2020 | Simplifying and Powering Graph Convolution Network for Recommendation | [论文链接](http://arxiv.org/abs/2002.02126v4) |
| MacridVAE | 2019 | Learning Disentangled Representations for Recommendation | [论文链接](https://doi.org/10.1145/3477495.3532012) |
| MultVAE | 2018 | Variational Autoencoders for Collaborative Filtering | [论文链接](https://doi.org/10.1145/3289600.3291007) |
| SGL | 2021 | Self-supervised Graph Learning for Recommendation | [论文链接](https://doi.org/10.1109/eebda60612.2024.10486045) |
| NCL | 2022 | Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning | [论文链接](https://doi.org/10.1145/3485447.3512104) |
| LightGCL | 2023 | Simple Yet Effective Graph Contrastive Learning for Recommendation | [论文链接](https://doi.org/10.1109/tce.2025.3570293) |
| LayerGCN | 2022 | Layer-refined Graph Convolutional Networks for Recommendation | [论文链接](https://doi.org/10.1109/icde55515.2023.00100) |
| HCCF | 2022 | Hypergraph Contrastive Collaborative Filtering | [论文链接](https://doi.org/10.1145/3477495.3532058) |
| DCCF | 2023 | Disentangled Contrastive Collaborative Filtering | [论文链接](https://doi.org/10.1145/3539618.3591665) |
| AdaGCL | 2023 | Adaptive Graph Contrastive Learning for Recommendation | [论文链接](https://doi.org/10.1109/cbd69312.2025.00045) |
| VGCL | 2023 | Generative-Contrastive Graph Learning for Recommendation | [论文链接](http://arxiv.org/abs/2307.05100v1) |
| SimGCL | 2022 | Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation | [论文链接](http://arxiv.org/abs/2112.08679v4) |
| XSimGCL | 2023 | Towards Extremely Simple Graph Contrastive Learning for Recommendation | [论文链接](https://doi.org/10.1109/tkde.2023.3288135) |
| GraphAug | 2024 | Graph Augmentation for Recommendation | [论文链接](https://doi.org/10.1109/iccwamtip64812.2024.10873787) |
| SelfCF | 2023 | A Simple Framework for Self-supervised Collaborative Filtering | [论文链接](https://doi.org/10.1145/3591469) |
| DHCF | 2020 | Dual Channel Hypergraph Collaborative Filtering | [论文链接](https://doi.org/10.1145/3394486.3403253) |
| LightGODE | 2024 | Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation | [论文链接](https://doi.org/10.1145/3627673.3679773) |
| FKAN-GCF | 2024 | FourierKAN-GCF: Fourier Kolmogorov-Arnold Network - An Effective and Efficient Feature Transformation for Graph Collaborative Filtering | [论文链接](https://arxiv.org/abs/2406.01034) |
| DualVAE | 2024 | Dual Disentangled Variational AutoEncoder for Recommendation | [论文链接](https://doi.org/10.1137/1.9781611978032.66) |
| GFormer | 2023 | Graph Transformer for Recommendation | [论文链接](http://arxiv.org/abs/2306.02330v1) |
| LightGODE（原README重复） | 2024 | Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation | [论文链接](https://doi.org/10.1145/3627673.3679773) |
| BSPM | 2023 | Blurring-Sharpening Process Models for Collaborative Filtering | [论文链接](https://doi.org/10.1145/3539618.3591645) |
| DiffRec | 2023 | Diffusion Recommender Model | [论文链接](https://doi.org/10.1145/3696410.3714873) |
| CF-Diff | 2024 | Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity | [论文链接](https://doi.org/10.1145/3626772.3657742) |

### 7.2 多模态推荐模型（原列表）

| 模型 | 年份 | 论文 | 链接 |
|---|---:|---|---|
| VBPR | 2016 | Visual Bayesian Personalized Ranking from Implicit Feedback | [论文链接](https://doi.org/10.1609/aaai.v30i1.9973) |
| MMGCN | 2019 | Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video | [论文链接](https://weiyinwei.github.io/papers/mmgcn.pdf) |
| GRCN | 2020 | Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback | [论文链接](https://doi.org/10.1145/3394171.3413556) |
| MGAT | 2020 | Multimodal Graph Attention Network for Recommendation | [论文链接](https://www.sciencedirect.com/science/article/pii/S0306457320300182) |
| LATTICE | 2021 | Mining Latent Structures for Multimedia Recommendation | [论文链接](https://doi.org/10.1145/3474085.3475259) |
| MICRO | 2022 | Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation | [论文链接](https://doi.org/10.1109/tkde.2022.3221949) |
| FREEDOM | 2023 | A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation | [论文链接](https://doi.org/10.1145/3581783.3611943) |
| DualGNN | 2023 | Dual Graph Neural Network for Multimedia Recommendation | [论文链接](https://doi.org/10.1109/tmm.2021.3138298) |
| DRAGON | 2023 | Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation | [论文链接](https://doi.org/10.3233/faia230631) |
| BM3 | 2023 | Bootstrap Latent Representations for Multi-modal Recommendation | [论文链接](https://doi.org/10.1145/3543507.3583251) |
| SLMRec | 2022 | Self-supervised Learning for Multimedia Recommendation | [论文链接](https://doi.org/10.1109/tmm.2022.3187556) |
| MGCL | 2023 | Multimodal Graph Contrastive Learning for Multimedia-Based Recommendation | [论文链接](https://doi.org/10.1109/tmm.2023.3251108) |
| MGCN | 2023 | Multi-View Graph Convolutional Network for Multimedia Recommendation | [论文链接](https://doi.org/10.1145/3581783.3613915) |
| POWERec | 2024 | Prompt-based and weak-modality enhanced multimodal recommendation | [论文链接](https://doi.org/10.1016/j.inffus.2023.101989) |
| MMGCL | 2022 | Multi-modal Graph Contrastive Learning for Micro-video Recommendation | [论文链接](https://doi.org/10.1145/3477495.3532027) |
| MVGAE | 2022 | Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems | [论文链接](https://doi.org/10.1109/tmm.2021.3111487) |
| MMSSL | 2023 | Multi-Modal Self-Supervised Learning for Recommendation | [论文链接](https://doi.org/10.1145/3543507.3583206) |
| LGMRec | 2024 | Local and Global Graph Learning for Multimodal Recommendation | [论文链接](https://doi.org/10.1609/aaai.v38i8.28688) |
| MENTOR | 2024 | Multi-level Self-supervised Learning for Multimodal Recommendation | [论文链接](https://doi.org/10.1609/aaai.v39i12.33408) |
| MCLN | 2023 | Multimodal Counterfactual Learning Network for Multimedia-based Recommendation | [论文链接](https://doi.org/10.1145/3539618.3591739) |
| DiffMM | 2024 | Multi-Modal Diffusion Model for Recommendation | [论文链接](https://doi.org/10.1145/3664647.3681498) |
| LightGT | 2023 | A Light Graph Transformer for Multimedia Recommendation | [论文链接](https://doi.org/10.1145/3539618.3591716) |
| SMORE | 2025 | Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation | [论文链接](https://doi.org/10.1145/3701551.3703561) |

## 8. 常见问题

1. 运行时报找不到 YAML 文件  
请确认 `--Model` 名称与 `Model_YAML/*.yaml` 文件名严格一致。

2. 显存不足或训练过慢  
先调小 `--batch_size`，再减少 YAML 搜索组合数（尤其是多参数多取值时）。

3. 新增数据集无法运行  
除放置 `.npy` 文件外，还需要在 `dataload.py` 中补充该数据集的 `num_user` 与 `num_item`。

---

如果你希望，我可以继续帮你补一版「按模型分类（CF/多模态/扩散）」的对照表，并附上每个模型的推荐启动参数模板。
