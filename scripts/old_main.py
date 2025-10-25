# TORCH_DISTRIBUTED_DEBUG=INFO
# TORCH_DISTRIBUTED_DEBUG=DETAIL
from ast import arg
from matplotlib.dates import SU
import builtins, json, warnings, pdb, os, time, sys, torch, torch.optim, torch.utils.data, torch.utils.data.distributed, random
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from log import create_folder_path
path = os.getcwd()
os.chdir(path)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from timeit import default_timer
from utils import init_distributed_mode, HiddenPrints
from eval_scripts.eval_l63 import eval_l63
from tqdm import tqdm
from configuration import args
from scripts.dataloader_init import init_dataloader
from scripts.train_utils import LpLoss_, adjust_learning_rate_cos, visualiztion, plot_loss, save_operator, compute_gradient_norms, plot_attractor_statistics, plot_training_metrics
from models.mlp_timestepper import TimeStepperMLP
from models.mlp import MLP
from scripts.summary_statistics import create_summary_statistic_model
from configs.summary_configs import SUMMARY_CONFIGS
import logging

logger = logging.getLogger(__name__)

def main(args):
    print(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.world_size == -1:
        args.world_size = 1

    args.distributed = args.world_size > 1
    ngpus_per_node = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 0)

    print('start')
    args.dist_backend = 'gloo'

    if args.distributed:
        if args.local_rank != -1:
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.gpu = 0
        args.rank = 0

    print('world_size', args.world_size)
    print('rank', args.rank)

    if args.rank > 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.is_master = args.rank % ngpus_per_node == 0 and args.gpu == 0
    train_dataset, train_dataset_operator, train_loader_metric, val_loader_metric, \
        train_loader_operator, train_sampler, train_sampler_operator =  init_dataloader(args)
    prefix_for_CL, contra_models_path, operators_path, output_path = create_folder_path(args)
    device = args.gpu if torch.cuda.is_available() else 'cpu'
    activation = torch.nn.GELU()
    hidden_layers = [128, 128, 128, 128]

    operator = TimeStepperMLP(
        input_dim=6,
        hidden_layers=hidden_layers,
        activation=activation,
        output_dim=3
    ).to(device)
    activation = torch.nn.GELU()
    layers = [3, 128, 128, 128, 128]

    # Create summary statistic model based on configuration
    summary_config = SUMMARY_CONFIGS.get(args.summary_config, SUMMARY_CONFIGS['learned_mlp'])
    f = create_summary_statistic_model(summary_config, args).to(device)
    
    learning_rate, epochs = args.learning_rate, args.epochs
    optimizer_operator = torch.optim.AdamW(operator.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    optimizer_learnable_ot = None
    if f.is_learnable:
        optimizer_learnable_ot = torch.optim.AdamW(f.parameters(), lr=args.lr_metric, weight_decay=1e-4)
    
    train_epoch_ = 0
    if args.distributed:
        operator = torch.nn.parallel.DistributedDataParallel(operator, device_ids=[args.gpu], find_unused_parameters = False, broadcast_buffers = False)
    operator = nn.SyncBatchNorm.convert_sync_batchnorm(operator)
    from scripts.OT_utils import OT_measure
    OT_measure = OT_measure(args.with_geomloss, args.blur)
    from train_utils import long_length_predict_with_yinit

    if args.train_operator:
        loss_list, ep_loss = [], []
        for ep in tqdm(range(epochs)):
            if args.distributed:
                train_sampler_operator.set_epoch(ep)

            for param, y in train_loader_operator:
                operator.train()
                l2, loss_OT = torch.tensor([0]).to(device).float(), torch.tensor([0]).to(device).float()
                lr_ = adjust_learning_rate_cos(args.learning_rate, optimizer_operator, ep, epochs, args)
                param, y = param.to(device), y.to(device).squeeze()
                assert args.x_len <= y.shape[1]
                assert y.shape[0] == args.batch_size

                optimizer_operator.zero_grad()
                y_predict = long_length_predict_with_yinit(operator, y, param, args.x_len, args.len_to_operator)
                
                l2 += LpLoss_(2).rel(y_predict, y)

                if args.with_geomloss > 0 and ep >= 20:

                    if args.l63:
                        with torch.no_grad():
                            anchor_stats_operator = f(y)
                        out_stats_operator = f(y_predict)

                        if args.with_geomloss_kd != 0:
                            anchor_stats_operator, out_stats_operator = anchor_stats_operator[:,:,np.array([args.with_geomloss_kd-1])]
                            out_stats_operator[:,:,np.array([args.with_geomloss_kd-1])]
                    assert anchor_stats_operator.shape[0] == args.batch_size
                    loss_OT = OT_measure.loss(anchor_stats_operator, out_stats_operator)

                loss_operator =  l2 + args.lambda_geomloss * loss_OT
                loss_operator.backward()
                operator_grad_norm = compute_gradient_norms(operator)
                optimizer_operator.step()

                f.train()

                f_grad_norm = 0.0

                if args.with_geomloss > 0 and ep >= 20:

                    if f.is_learnable:
                        optimizer_learnable_ot.zero_grad()
                        
                        with torch.no_grad():
                            y_predict_detached = long_length_predict_with_yinit(operator, y, param, args.x_len, args.len_to_operator)

                        y_for_f = y.detach().requires_grad_(True)
                        anchor_stats = f(y_for_f)
                        out_stats = f(y_predict_detached)
                    
                        if args.with_geomloss_kd != 0:
                            anchor_stats = anchor_stats[:,:,np.array([args.with_geomloss_kd-1])], 
                            out_stats = out_stats[:,:,np.array([args.with_geomloss_kd-1])]
                        assert anchor_stats.shape[0] == args.batch_size

                        loss_learnable_ot = OT_measure.loss(anchor_stats, out_stats)
                        loss_learnable_ot.backward()
                        f_grad_norm = compute_gradient_norms(f)

                        optimizer_learnable_ot.step()
                    else:
                        with torch.no_grad():
                            anchor_stats = f(y)
                            out_stats = f(y_predict)

                ep_loss.append([l2.item(), \
                                args.lambda_geomloss * loss_OT.item()])
                loss_list.append([np.array(ep_loss).mean(axis = 0)[0], np.array(ep_loss).mean(axis = 0)[1], lr_, operator_grad_norm,
                f_grad_norm])

                # print(f'Epoch {ep}, L2 Loss: {l2.item():.6f}, OT Loss: {loss_OT.item():.6f}, LR: {lr_:.6f}, Operator Grad Norm: {operator_grad_norm:.4E}, f Grad Norm: {f_grad_norm:.4E}')

            if ep% 50 == 0 and ep > 0:
                visualiztion(train_dataset_operator, operator, args, img_pth=f'{output_path}/training_vis', ep=ep)
                plot_loss(loss_list, img_pth = f'{output_path}/training_loss_operator')
                plot_training_metrics(loss_list, img_pth=f'{output_path}/training_metrics')
                # plot_attractor_statistics(train_dataset_operator, operator, f, args, 
                #                         img_pth=f'{output_path}/attractor_stats_{ep:03d}', ep=ep)

        if ep == epochs - 1:
            if args.is_master:
                save_operator(operator, optimizer_operator, saved_pth=f'{operators_path}/{args.prefix}/{ep:03d}', ep=ep)

    ###########################################################################
    #################### load the model and evaluation ########################
    ep = args.epochs - 1
    from eval_scripts.eval_utils import load_operator
    operator  = load_operator(operator, saved_pth = f'{operators_path}/{args.prefix}/{ep:03d}')
    visualiztion(train_dataset_operator, operator, args, img_pth = f'{output_path}/training_vis', ep=ep)
    ############### evaluate the statistics and save them ######################
    if args.l63:
        eval_len_list = [1500]
    for eval_len in eval_len_list:
        x_len = args.x_len
        eval_l63(operator, args, args.noisy_scale, x_len = eval_len, calculate_l2 = True, output_path = output_path)
        eval_l63(operator, args, 0, x_len = eval_len, calculate_l2 = True, output_path = output_path)
    if args.eval_LE:
        from eval_scripts.eval_LE import cal_LE
        LE_results = cal_LE(operator, args)

if __name__ == '__main__':
    main(args)
