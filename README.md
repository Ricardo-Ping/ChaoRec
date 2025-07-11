## ChaoRec（超级推荐）

ChaoRec 是基于 Python 和 PyTorch 开发的，用于在统一、全面和高效的框架内复制和开发推荐算法，以达到研究目的。

主要包括一般推荐和多模态推荐。

目前的一般推荐模型有27个：

- BPR(2016): Bayesian Personalized Ranking with Multi-Channel User Feedback
- DGCF(2020): Disentangled Graph Collaborative Filtering
- NGCF(2019): Neural Graph Collaborative Filtering
- LightGCN(2020): Simplifying and Powering Graph Convolution Network for Recommendation
- MacridVAE(2019): Learning Disentangled Representations for Recommendation
- MultVAE(2018): Variational Autoencoders for Collaborative Filtering
- SGL(2021): Self-supervised Graph Learning for Recommendation
- NCL(2022): Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning
- LightGCL(2023): Simple Yet Effective Graph Contrastive Learning for Recommendation
- LayerGCN(2022): Layer-refined Graph Convolutional Networks for Recommendation
- HCCF(2022): Hypergraph Contrastive Collaborative Filtering
- DCCF(2023): Disentangled Contrastive Collaborative Filtering
- AdaGCL(2023): Adaptive Graph Contrastive Learning for Recommendation
- VGCL(2023): Generative-Contrastive Graph Learning for Recommendation
- SimGCL(2022): Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation
- XSimGCL(2023): Towards Extremely Simple Graph Contrastive Learning for Recommendation
- GraphAug(2024): Graph Augmentation for Recommendation
- SelfCF(2023): A Simple Framework for Self-supervised Collaborative Filtering
- DHCF(2020): Dual Channel Hypergraph Collaborative Filtering
- LightGODE(2024): Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation
- FKAN-GCF(2024): FourierKAN-GCF: Fourier Kolmogorov-Arnold Network - An Effective and Efficient Feature Transformation for Graph Collaborative Filtering
- DualVAE(2024): Dual Disentangled Variational AutoEncoder for Recommendation
- GFormer(2023): Graph Transformer for Recommendation
- LightGODE(2024): Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation
- BSPM(2023)：Blurring-Sharpening Process Models for Collaborative Filtering (源代码使用的不是留一法预测,并且直接预测交互，所以在我们框架内性能较差)
- DiffRec(2023)：Diffusion Recommender Model
- CF-Diff(2024)：Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity

目前的多模态推荐模型有23个：

- VBPR(2016): Visual Bayesian Personalized Ranking from Implicit Feedback
- MMGCN(2019): Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video
- GRCN(2020): Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback
- MGAT(2020): Multimodal Graph Attention Network for Recommendation
- LATTICE(2021): Mining Latent Structures for Multimedia Recommendation
- MICRO(2022): Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation
- FREEDOM(2023): A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
- DualGNN(2023): Dual Graph Neural Network for Multimedia Recommendation
- DRAGON(2023): Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation
- BM3(2023): Bootstrap Latent Representations for Multi-modal Recommendation
- SLMRec(2022): Self-supervised Learning for Multimedia Recommendation
- MGCL(2023): Multimodal Graph Contrastive Learning for Multimedia-Based Recommendation
- MGCN(2023): Multi-View Graph Convolutional Network for Multimedia Recommendation
- POWERec(2024): Prompt-based and weak-modality enhanced multimodal recommendation
- MMGCL(2022): Multi-modal Graph Contrastive Learning for Micro-video Recommendation
- MVGAE(2022): Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems
- MMSSL(2023): Multi-Modal Self-Supervised Learning for Recommendation
- LGMRec(2024): Local and Global Graph Learning for Multimodal Recommendation
- MENTOR(2024): Multi-level Self-supervised Learning for Multimodal Recommendation
- MCLN(2023): Multimodal Counterfactual Learning Network for Multimedia-based Recommendation
- DiffMM(2024): Multi-Modal Diffusion Model for Recommendation
- LightGT(2023): A Light Graph Transformer for Multimedia Recommendation (受限于数据集中的交互项目长度，性能也不足)
- SMORE(2025): Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation

现有模型大部分按照原作者代码进行改写，如果发现有错误欢迎指正！
