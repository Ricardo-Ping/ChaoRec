from BPR import BPRMF
from NGCF import NGCF
from VBPR import VBPR
from arg_parser import parse_args
from utils import setup_seed, gpu, get_local_time
import torch
import logging
import Dataset
from torch.utils.data import DataLoader
from MMGCN import MMGCN
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
    feature_embedding = args.feature_embed
    dropout = args.dropout
    # 加载训练数据
    train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat = Dataset.data_load(
        args.data_path)
    train_dataset = Dataset.TrainingDataset(num_user, num_item, user_item_dict, train_data)
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
                     user_item_dict, weight_decay, dim_E, device, dropout)
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])

    # 训练和评估
    train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs)
