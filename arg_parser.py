"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2023/9/28 17:08
@File : arg_parser.py
@function :
"""
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run ChaoRec.")
    # 模型和数据集选择
    parser.add_argument('--Model', nargs='?', default='Grade', help='Model name')
    # 数据集：baby,clothing,sports,beauty, microlens, netfilx
    parser.add_argument('--data_path', nargs='?', default='sports', help='Input data path.')
    # 超参数选择(具体模型参数需要到yaml文件中进行调整)
    parser.add_argument('--learning_rate', type=float, nargs='+', default=1e-3, help='Learning rates')
    parser.add_argument('--feature_embed', type=int, default=64, help='Feature Embedding size')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation mode.')
    parser.add_argument('--reg_weight', type=float, nargs='+', default=1e-3, help='Weight decay.')
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout.')
    parser.add_argument('--n_layers', type=int, default=2, help='conv_layers.')
    parser.add_argument('--corDecay', type=float, default=0.001, help='CorDecay.')
    parser.add_argument('--n_factors', type=int, default=4, help='the number of hidden factor k.')
    parser.add_argument('--n_iterations', type=int, default=3, help='the number of iteration.')
    parser.add_argument('--cl_weight', type=float, default=2.0, help='the number of cl_loss_weight.')
    parser.add_argument('--mm_layers', type=int, default=2, help='the number of multimodal layer.')
    parser.add_argument('--ii_topk', type=int, default=10, help='the number of item-item graph topk.')
    parser.add_argument('--uu_topk', type=int, default=10, help='the number of user-user graph topk.')
    parser.add_argument('--lambda_coeff', type=float, default=0.9, help='the number of jump connection factor.')
    parser.add_argument('--ssl_temp', type=float, default=0.9, help='temperature coefficient.')
    parser.add_argument('--ssl_alpha', type=float, default=0.9, help='ssl coefficient.')
    parser.add_argument('--ae_weight', type=float, default=0.1, help='the number of auto encoder loss_weight.')
    parser.add_argument('--threshold', type=float, default=0.1, help='the number of threshold.')
    parser.add_argument('--prompt_num', type=float, default=0.1, help='prompt modal numbers.')
    parser.add_argument('--neg_weight', type=float, default=0.1, help='weak modal weight.')
    parser.add_argument('--cen_reg', type=float, default=5e-3, help='intent regularization')
    parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
    parser.add_argument('--G_rate', type=float, default=0.0001, help='MMSSL')
    parser.add_argument('--align_weight', type=float, default=0.1, help='MENTOR align_weight')
    parser.add_argument('--mask_weight_f', type=float, default=1.5, help='MENTOR mask_weight_f')
    parser.add_argument('--mask_weight_g', type=float, default=0.001, help='MENTOR mask_weight_g')
    parser.add_argument('--leaky', type=float, default=0.5, help='HCCF leaky')
    parser.add_argument('--keepRate', type=float, default=1.0, help='HCCF leaky')
    parser.add_argument('--mult', type=float, default=0.1, help='HCCF leaky')
    parser.add_argument('--grid_size', type=int, default=1, help='FKAN_GCF grid_size.')
    parser.add_argument('--node_dropout', type=float, default=0.1, help='FKAN_GCF node_dropout')
    parser.add_argument('--message_dropout', type=float, default=0.1, help='FKAN_GCF message_dropout')
    parser.add_argument('--n_mca', type=int, default=2, help='MCLN counterfactual layer.')
    parser.add_argument('--gamma', type=float, default=0.5, help='LightGODE uniformity weight.')
    parser.add_argument('--t', type=float, default=1.8, help='LightGODE time step.')
    parser.add_argument('--e_loss', type=float, default=0.1, help='DiffMM e_loss(lamba0).')
    parser.add_argument('--ris_lambda', type=float, default=0.5, help='DiffMM (eq23-𝜔).')
    parser.add_argument('--rebuild_k', type=int, default=1, help='DiffMM rebuild top-k.')
    parser.add_argument('--pnn_layer', type=int, default=1, help='GFormer pnn_layer.')
    parser.add_argument('--b2', type=float, default=1, help='GFormer b2.')
    parser.add_argument('--ctra', type=float, default=0.001, help='GFormer ctra.')
    parser.add_argument('--noise_alpha', type=int, default=0.3, help='Grade noise weight')
    parser.add_argument('--ssl_temp2', type=float, default=0.2, help='Grade temperature coefficient.')
    parser.add_argument('--K_s', type=int, default=1, help='BSPM T_s / \tau')
    parser.add_argument('--T_s', type=float, default=1, help='BSPM T_s')
    parser.add_argument('--K_b', type=int, default=1, help='BSPM T_b / \tau')
    parser.add_argument('--T_b', type=float, default=1, help='BSPM T_b')
    parser.add_argument('--idl_beta', type=float, default=1, help='BSPM idl_beta')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='DiffRec sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='DiffRec steps of the forward process during '
                                                                      'inference')
    parser.add_argument('--steps', type=int, default=100, help='DiffRec diffusion steps')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='DiffRec noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='DiffRec noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.02, help='DiffRec noise upper bound for noise generating')
    parser.add_argument('--dims', type=str, default='[1000]', help='DiffRec the dims for the DNN')
    # MHRec
    parser.add_argument('--h_layers', type=int, default=2, help='hypergraph layers.')
    parser.add_argument('--num_hypernodes', type=int, default=10, help='hypergraph num_hypernodes.')
    parser.add_argument('--beta1', type=float, default=0.5, help='MHRec beta1')
    parser.add_argument('--beta2', type=float, default=0.5, help='MHRec beta2')
    # 一些默认参数
    parser.add_argument('--seed', type=int, default=42, help='Number of seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--topk', type=float, nargs='+', default=[5, 10, 20], help='topK')

    return parser.parse_args()


def load_yaml_config(model_name):
    yaml_file = f"Model_YAML/{model_name}.yaml"
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)
