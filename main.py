import os
from itertools import product

import numpy as np

from Model.AdaGCL import AdaGCL
from Model.BM3 import BM3
from Model.BPR import BPRMF
from Model.BSPM import BSPM
from Model.CF_Diff import CF_Diff
from Model.DCCF import DCCF
from Model.DDRec import DDRec
from Model.DGCF import DGCF
from Model.DHCF import DHCF
from Model.DRAGON import DRAGON
from Model.DiffMM import DiffMM
from Model.DiffRec import DiffRec
from Model.DualGNN import DualGNN
from Model.DualVAE import DualVAE
from Model.FKAN_GCF import FKAN_GCF
from Model.FREEDOM import FREEDOM
from Model.GFormer import GFormer
from Model.GRCN import GRCN
from Model.Grade import Grade
from Model.GraphAug import GraphAug
from Model.HCCF import HCCF
from Model.LATTICE import LATTICE
from Model.LGMRec import LGMRec
from Model.LayerGCN import LayerGCN
from Model.LightGCL import LightGCL
from Model.LightGCN import LightGCN
from Model.LightGODE import LightGODE
from Model.LightGT import LightGT
from Model.MCLN import MCLN
from Model.MENTOR import MENTOR
from Model.MGAT import MGAT
from Model.MGCL import MGCL
from Model.MGCN import MGCN
from Model.MHRec import MHRec
from Model.MICRO import MICRO
from Model.MMGCL import MMGCL
from Model.MMSSL import MMSSL
from Model.MVGAE import MVGAE
from Model.MacridVAE import MacridVAE
from Model.MultVAE import MultVAE
from Model.NCL import NCL
from Model.NGCF import NGCF
from Model.POWERec import POWERec
from Model.SGL import SGL
from Model.SLMRec import SLMRec
from Model.SelfCF import SelfCF
from Model.SimGCL import SimGCL
from Model.VBPR import VBPR
from Model.VGCL import VGCL
from Model.XSimGCL import XSimGCL
from arg_parser import parse_args, load_yaml_config
from utils import setup_seed, gpu, get_local_time, convert_to_dict
import torch
import logging
import dataload
from torch.utils.data import DataLoader
from Model.MMGCN import MMGCN
from train_and_evaluate import train_and_evaluate

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # 输出参数信息
    args = parse_args()
    # 创建日志文件夹
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志文件名
    log_filename = os.path.join(log_dir, f"{args.Model}_{args.data_path}").replace("\\", "/") + ".log"

    # 配置日志格式
    log_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%a %d %b %Y %H:%M:%S'

    # 创建一个日志处理器，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)

    # 设置一个文件处理器，用于写入文件
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)

    # 获取 root 记录器并配置处理器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info('============Arguments==============')
    for arg, value in vars(args).items():
        logging.info('%s: %s', arg, value)
    logging.info('local time：%s', get_local_time())
    # 读取设置的不需要修改的参数
    setup_seed(args.seed)
    device = gpu()
    batch_size = args.batch_size
    num_workers = args.num_workers
    dim_E = args.dim_E
    epochs = args.num_epoch
    feature_embedding = args.feature_embed  # 特征嵌入
    model_name = args.Model
    aggr_mode = args.aggr_mode
    # 需要从yaml中读取的参数
    config = load_yaml_config(model_name)
    reg_weight = args.reg_weight
    learning_rate = args.learning_rate
    dropout = args.dropout
    n_layers = args.n_layers
    corDecay = args.corDecay
    n_factors = args.n_factors
    n_iterations = args.n_iterations
    mm_layers = args.mm_layers  # 多模态卷积层数
    ii_topk = args.ii_topk  # 项目-项目图的topk选择
    uu_topk = args.uu_topk  # 用户-用户图的topk选择
    lambda_coeff = args.lambda_coeff  # 跳跃连接系数
    ssl_temp = args.ssl_temp  # ssl的温度系数
    ssl_alpha = args.ssl_alpha  # ssl任务损失的系数
    ae_weight = args.ae_weight  # 自动编码器损失的系数
    threshold = args.threshold  # 去噪门控
    prompt_num = args.prompt_num
    neg_weight = args.neg_weight
    cen_reg = args.cen_reg  # DCCF的意图嵌入正则化
    n_intents = args.n_intents  # DCCF的意图嵌入数量
    G_rate = args.G_rate  # MMSSL的生成器损失权重
    align_weight = args.align_weight  # MENTOR的对齐损失权重
    mask_weight_f = args.mask_weight_f  # MENTOR的特征掩码损失权重
    mask_weight_g = args.mask_weight_g  # MENTOR的图掩码损失权重
    leaky = args.leaky  # HCCf
    keepRate = args.keepRate  # HCCf
    mult = args.mult  # HCCF
    # FKAN_GCF
    grid_size = args.grid_size
    node_dropout = args.node_dropout
    message_dropout = args.message_dropout
    # MCLN
    n_mca = args.n_mca
    # LightGode
    gamma = args.gamma
    t = args.t
    # DiffMM
    e_loss = args.e_loss
    ris_lambda = args.ris_lambda
    rebuild_k = args.rebuild_k
    # GFormer
    pnn_layer = args.pnn_layer
    b2 = args.b2
    ctra = args.ctra
    # Grade
    noise_alpha = args.noise_alpha
    ssl_temp2 = args.ssl_temp2
    # BSPM
    K_s = args.K_s
    T_s = args.T_s
    K_b = args.K_b
    T_b = args.T_b
    idl_beta = args.idl_beta
    # DiffRec
    noise_scale = args.noise_scale
    noise_min = args.noise_min
    noise_max = args.noise_max
    steps = args.steps
    dims = args.dims
    # MHRec
    h_layers = args.h_layers
    num_hypernodes = args.num_hypernodes
    beta1 = args.beta1
    beta2 = args.beta2

    # 加载训练数据
    train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat = dataload.data_load(
        args.data_path)
    train_dataset = dataload.TrainingDataset(num_user, num_item, user_item_dict, train_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    # ------------LightGT模型需要--------------------
    eval_dataset = dataload.EvalDataset(num_user, num_item, user_item_dict)
    eval_dataloader = DataLoader(eval_dataset, 2000, shuffle=False)

    # ------------DiffMM模型需要--------------------
    diffusionData = dataload.DiffusionData(num_user, num_item, train_data)
    diffusionLoader = DataLoader(diffusionData, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    args.num_user = num_user
    args.num_item = num_item
    # ------------DiffRec模型需要--------------------
    if args.Model in ["DiffRec"]:
        diffusionLoader = DataLoader(diffusionData, batch_size=batch_size, pin_memory=True, shuffle=True,
                                     num_workers=num_workers)
    test_diffusionLoader = DataLoader(diffusionData, batch_size=batch_size, shuffle=False)
    # ------------DiffRec模型需要--------------------
    if args.Model in ["CF_Diff"]:
        multi_hop = dataload.DiffusionData_sec_hop(num_user, num_item, train_data)
        train_loader_sec_hop = DataLoader(multi_hop, batch_size=batch_size, pin_memory=True, shuffle=True,
                                          num_workers=num_workers)
        test_diffusionLoader = DataLoader(diffusionData, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        test_loader_sec_hop = DataLoader(multi_hop, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    # ------------MHRec模型需要--------------------
    if args.Model in ["MHRec"]:
        # 加载保存的超边序列文件
        dir_str = './Data/' + args.data_path
        # 这里的uu_topk和ii_topk和arg_parser.py中的一样，而不是yaml文件
        visual_file_path = os.path.join(dir_str, 'hyperedges_visual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk))
        textual_file_path = os.path.join(dir_str, 'hyperedges_textual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk))
        # 加载超边序列
        hyperedges_visual = np.load(visual_file_path, allow_pickle=True).tolist()
        hyperedges_textual = np.load(visual_file_path, allow_pickle=True).tolist()
        diffusion_hyperedges_visual = dataload.HyperDiffusionData(num_user, num_item, hyperedges_visual)
        diffusion_hyperedges_textual = dataload.HyperDiffusionData(num_user, num_item, hyperedges_textual)
        diffusionLoader_visual = DataLoader(diffusion_hyperedges_visual, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        diffusionLoader_textual = DataLoader(diffusion_hyperedges_textual, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # ----------------------------------------------

    # 网格搜索
    hyper_ls = []
    for param in config['hyper_parameters']:
        hyper_ls.append(config[param])
    # 生成所有可能的超参数组合
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)  # 总组合数量

    # 最佳参数
    best_performance = None
    best_params = None
    best_metrics = None

    for idx, hyper_param_combo in enumerate(combinators):
        hyper_param_dict = dict(zip(config['hyper_parameters'], hyper_param_combo))

        # 输出当前网格搜索序号和总的参数搜索次数
        logging.info('========={}/{}: Parameters:{}========='.format(
            idx + 1, total_loops, hyper_param_dict))

        # 覆盖args中的参数
        for key, value in hyper_param_dict.items():
            setattr(args, key, value)

        # 定义模型
        model_constructors = {
            'MMGCN': lambda: MMGCN(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight,
                                   aggr_mode, 'False', True, device),
            'BPR': lambda: BPRMF(num_user, num_item, user_item_dict, dim_E, args.reg_weight, device),
            'VBPR': lambda: VBPR(num_user, num_item, user_item_dict, v_feat, dim_E, feature_embedding, args.reg_weight,
                                 device),
            'NGCF': lambda: NGCF(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight, args.dropout,
                                 args.n_layers, aggr_mode, device),
            'LightGCN': lambda: LightGCN(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                         args.n_layers, aggr_mode, device),
            'GRCN': lambda: GRCN(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                 feature_embedding,
                                 args.reg_weight, args.dropout, args.n_iterations, aggr_mode, device),
            'DGCF': lambda: DGCF(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight, args.corDecay,
                                 args.n_factors, args.n_iterations, args.n_layers, aggr_mode, device),
            'LATTICE': lambda: LATTICE(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                       feature_embedding,
                                       args.reg_weight, args.n_layers, args.mm_layers, args.ii_topk, aggr_mode,
                                       args.lambda_coeff, device),
            'DualGNN': lambda: DualGNN(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                       feature_embedding, args.reg_weight, args.uu_topk, aggr_mode, device),
            'BM3': lambda: BM3(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E, feature_embedding,
                               args.reg_weight, args.dropout, args.n_layers, args.cl_weight, aggr_mode, device),
            'DRAGON': lambda: DRAGON(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                     feature_embedding, args.reg_weight, args.n_layers, args.ii_topk, args.uu_topk,
                                     args.lambda_coeff, aggr_mode, device),
            'FREEDOM': lambda: FREEDOM(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                       feature_embedding, args.reg_weight, args.dropout, args.n_layers, args.mm_layers,
                                       args.ii_topk, args.lambda_coeff, device),
            'SLMRec': lambda: SLMRec(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                     args.n_layers, args.ssl_temp, args.ssl_alpha, device),
            'MGAT': lambda: MGAT(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E, args.reg_weight,
                                 device),
            'MICRO': lambda: MICRO(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E, args.n_layers,
                                   args.reg_weight, args.ii_topk, args.mm_layers, args.ssl_temp, args.lambda_coeff,
                                   args.ssl_alpha, aggr_mode, device),
            'MMGCL': lambda: MMGCL(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight, args.n_layers, args.ssl_alpha, args.ssl_temp, args.dropout, device),
            'DDRec': lambda: DDRec(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   feature_embedding, args.reg_weight, args.n_layers, args.ssl_temp, args.ssl_alpha,
                                   args.threshold, aggr_mode, device),
            'SGL': lambda: SGL(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                               args.n_layers, aggr_mode, args.ssl_temp, args.ssl_alpha, device),
            'MultVAE': lambda: MultVAE(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight, device),
            'NCL': lambda: NCL(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight, args.n_layers,
                               aggr_mode, args.ssl_temp, args.ssl_alpha, device),
            'MacridVAE': lambda: MacridVAE(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                           device),
            'LightGCL': lambda: LightGCL(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                         args.n_layers, aggr_mode, args.ssl_alpha, args.ssl_temp, device),
            'HCCF': lambda: HCCF(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                 args.n_layers, aggr_mode, args.ssl_alpha, args.ssl_temp, args.keepRate,
                                 args.leaky, args.mult, device),
            'MGCL': lambda: MGCL(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E, args.reg_weight,
                                 args.n_layers, aggr_mode, args.ssl_temp, args.ssl_alpha, device),
            'MGCN': lambda: MGCN(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E, args.reg_weight,
                                 args.n_layers, aggr_mode, args.ssl_temp, args.ssl_alpha, device),
            'POWERec': lambda: POWERec(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                       args.reg_weight, args.n_layers, args.prompt_num, args.neg_weight, args.dropout,
                                       device),
            'MVGAE': lambda: MVGAE(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight, args.n_layers, device),
            'LayerGCN': lambda: LayerGCN(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                         args.n_layers, args.dropout, device),
            'DCCF': lambda: DCCF(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight, args.n_layers,
                                 args.ssl_temp, args.ssl_alpha, args.n_intents, args.cen_reg, device),
            'AdaGCL': lambda: AdaGCL(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                     args.n_layers, args.ssl_temp, args.ssl_alpha, device),
            'DualVAE': lambda: DualVAE(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                       args.ssl_alpha, device),
            'MMSSL': lambda: MMSSL(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight, args.ssl_alpha, args.ssl_temp, args.G_rate, args.mm_layers, device),
            'VGCL': lambda: VGCL(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight, args.n_layers,
                                 args.ssl_temp, args.ssl_alpha, device),
            'SimGCL': lambda: SimGCL(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                     args.n_layers, args.ssl_temp, args.ssl_alpha, device),
            'XSimGCL': lambda: XSimGCL(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                       args.n_layers, args.ssl_temp, args.ssl_alpha, device),
            'GraphAug': lambda: GraphAug(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                         args.n_layers,
                                         args.ssl_temp, args.ssl_alpha, device),
            'LGMRec': lambda: LGMRec(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                     args.reg_weight, args.n_layers, args.ssl_alpha, device),
            'SelfCF': lambda: SelfCF(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                     args.n_layers, args.dropout, device),
            'MENTOR': lambda: MENTOR(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                     args.mm_layers, args.reg_weight, args.ssl_temp, args.dropout, args.align_weight,
                                     args.mask_weight_g, args.mask_weight_f, device),
            'LightGT': lambda: LightGT(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                       args.reg_weight, args.n_layers, device),
            'FKAN_GCF': lambda: FKAN_GCF(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                         args.n_layers, args.node_dropout, args.message_dropout, args.grid_size,
                                         device),
            'MCLN': lambda: MCLN(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                 args.reg_weight, args.n_layers, args.n_mca, device),
            'LightGODE': lambda: LightGODE(num_user, num_item, train_data, user_item_dict, dim_E, args.gamma, args.t,
                                           device),
            'DHCF': lambda: DHCF(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight, args.n_layers,
                                 args.dropout, device),
            'DiffMM': lambda: DiffMM(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                     args.reg_weight, args.n_layers, args.ssl_alpha, args.ssl_temp, args.ris_lambda,
                                     args.e_loss, args.rebuild_k, device),
            'GFormer': lambda: GFormer(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                       args.n_layers, args.pnn_layer, args.ssl_alpha, args.b2, args.ctra, device),
            'Grade': lambda: Grade(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight, args.n_layers, args.ssl_temp, args.ssl_alpha, args.ssl_temp2,
                                   args.noise_alpha, device),
            'BSPM': lambda: BSPM(num_user, num_item, train_data, user_item_dict, args.K_s, args.T_s, args.K_b, args.K_s,
                                 args.idl_beta, device),
            'DiffRec': lambda: DiffRec(num_user, num_item, user_item_dict, args.noise_scale, args.noise_min,
                                       args.noise_max, args.steps, args.dims, args.learning_rate, device),
            'CF_Diff': lambda: CF_Diff(num_user, num_item, user_item_dict, args.noise_scale, args.noise_min,
                                       args.noise_max, args.steps, args.learning_rate, device),
            'MHRec': lambda: MHRec(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight, args.ii_topk, args.uu_topk, args.num_hypernodes, args.n_layers,
                                   args.h_layers, args.ssl_temp, args.ssl_alpha, args.beta1, args.beta2, device),
            # ... 其他模型构造函数 ...
        }
        # 实例化模型
        model = model_constructors.get(model_name, lambda: None)()
        model.to(device)
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(f"Parameter requires_grad: {param.requires_grad}")
            # print(f"Parameter data:\n{param.data}")
            print("=" * 30)

        # 定义优化器
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.learning_rate}])
        # 训练和评估
        if args.Model in ["LightGT"]:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs,
                                                      eval_dataloader=eval_dataloader, diffusionLoader=None,
                                                      test_diffusionLoader=None)
        elif args.Model in ["DiffMM"]:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs,
                                                      diffusionLoader=diffusionLoader, test_diffusionLoader=None)
        elif args.Model in ["DiffRec"]:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs,
                                                      diffusionLoader=diffusionLoader,
                                                      test_diffusionLoader=test_diffusionLoader)
        elif args.Model in ["CF_Diff"]:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs,
                                                      diffusionLoader=diffusionLoader,
                                                      test_diffusionLoader=test_diffusionLoader,
                                                      train_loader_sec_hop=train_loader_sec_hop,
                                                      test_loader_sec_hop=test_loader_sec_hop
                                                      )
        elif args.Model in ["MHRec"]:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs,
                                                      diffusionLoader_visual=diffusionLoader_visual,
                                                      diffusionLoader_textual=diffusionLoader_textual)
        else:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs)

        current_best_recall = current_best_metrics[20]['recall']
        if best_performance is None or current_best_recall > best_performance:
            best_performance = current_best_recall
            best_params = hyper_param_dict.copy()
            best_metrics = current_best_metrics

    # 输出最佳性能和对应的超参数
    logging.info("Best performance: {:.5f}".format(best_performance))
    logging.info("Best parameters: {}".format(best_params))

    # 输出最佳指标
    logging.info("Best metrics:")
    for k, metrics in best_metrics.items():
        metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
        logging.info(f"{k}: {' | '.join(metrics_strs)}")
