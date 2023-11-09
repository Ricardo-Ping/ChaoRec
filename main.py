from Model.BM3 import BM3
from Model.BPR import BPRMF
from Model.DGCF import DGCF
from Model.DRAGON import DRAGON
from Model.DualGNN import DualGNN
from Model.FREEDOM import FREEDOM
from Model.GRCN import GRCN
from Model.LATTICE import LATTICE
from Model.LightGCN import LightGCN
from Model.MGAT import MGAT
from Model.MICRO import MICRO
from Model.NGCF import NGCF
from Model.SLMRec import SLMRec
from Model.VBPR import VBPR
from arg_parser import parse_args
from utils import setup_seed, gpu, get_local_time
import torch
import logging
import dataload
from torch.utils.data import DataLoader
from Model.MMGCN import MMGCN
from train_and_evaluate import train_and_evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%a %d %b %Y %H:%M:%S')

if __name__ == '__main__':
    # 输出参数信息
    args = parse_args()
    logging.info('============Arguments==============')
    for arg, value in vars(args).items():
        logging.info('%s: %s', arg, value)
    logging.info('local time：%s', get_local_time())
    # 读取设置的参数
    setup_seed(args.seed)
    device = gpu()
    batch_size = args.batch_size
    num_workers = args.num_workers
    aggr_mode = args.aggr_mode
    weight_decay = args.weight_decay
    dim_E = args.dim_E
    learning_rate = args.lr
    epochs = args.num_epoch
    feature_embedding = args.feature_embed  # 特征嵌入
    dropout = args.dropout
    layer = args.layer
    corDecay = args.corDecay
    n_factors = args.n_factors
    n_iterations = args.n_iterations
    # 加载训练数据
    train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat = dataload.data_load(
        args.data_path)
    train_dataset = dataload.TrainingDataset(num_user, num_item, user_item_dict, train_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    # 定义模型
    model = None
    if args.Model == 'MMGCN':
        model = MMGCN(v_feat, t_feat, train_data, num_user, num_item, aggr_mode, 'False', True, user_item_dict,
                      weight_decay, dim_E, device)
    elif args.Model == 'BPR':
        model = BPRMF(num_user, num_item, dim_E, user_item_dict, device, weight_decay)
    elif args.Model == 'VBPR':
        model = VBPR(num_user, num_item, v_feat, dim_E, user_item_dict, device, weight_decay, feature_embedding)
    elif args.Model == 'NGCF':
        model = NGCF(train_data, num_user, num_item, aggr_mode,
                     user_item_dict, weight_decay, dim_E, device, dropout, layer)
    elif args.Model == 'LightGCN':
        model = LightGCN(train_data, num_user, num_item, aggr_mode,
                         user_item_dict, weight_decay, dim_E, device, dropout, layer)
    elif args.Model == 'GRCN':
        model = GRCN(num_user, num_item, train_data, user_item_dict, weight_decay, v_feat, t_feat,
                     dim_E, feature_embedding, dropout, device, aggr_mode)
    elif args.Model == 'DGCF':
        model = DGCF(num_user, num_item, train_data, user_item_dict, weight_decay, corDecay, n_factors, n_iterations,
                     layer, dim_E, device, aggr_mode)
    elif args.Model == 'LATTICE':
        model = LATTICE(num_user, num_item, train_data, user_item_dict, dim_E, feature_embedding, weight_decay, v_feat,
                        t_feat,
                        layer, device, aggr_mode)
    elif args.Model == 'DualGNN':
        model = DualGNN(v_feat, t_feat, train_data, user_item_dict, num_user, num_item, aggr_mode, dim_E,
                        feature_embedding,
                        weight_decay, device)
    elif args.Model == 'BM3':
        model = BM3(num_user, num_item, train_data, user_item_dict, aggr_mode, v_feat, t_feat, weight_decay, dim_E,
                    feature_embedding, dropout, layer, device)
    elif args.Model == 'DRAGON':
        model = DRAGON(v_feat, t_feat, train_data, user_item_dict, num_user, num_item, aggr_mode, layer, dim_E,
                       feature_embedding,
                       weight_decay, device)
    elif args.Model == 'FREEDOM':
        model = FREEDOM(num_user, num_item, train_data, user_item_dict, dim_E, feature_embedding, weight_decay, dropout,
                        v_feat, t_feat, layer, device)
    elif args.Model == 'SLMRec':
        model = SLMRec(v_feat, t_feat, train_data, num_user, num_item, layer, user_item_dict, dim_E, device)
    elif args.Model == 'MGAT':
        model = MGAT(v_feat, t_feat, train_data, num_user, num_item, user_item_dict, dim_E, weight_decay, device)
    elif args.Model == 'MICRO':
        model = MICRO(num_user, num_item, train_data, user_item_dict, dim_E, v_feat, t_feat, layer, aggr_mode,
                      weight_decay, device)

    model.to(device)
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter requires_grad: {param.requires_grad}")
        # print(f"Parameter data:\n{param.data}")
        print("=" * 30)

    # 定义优化器
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])

    # 训练和评估
    train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs)
