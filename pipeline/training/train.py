import torch
import wandb
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pipeline.dataloader import dynamical_system_dataset as ds, transforms as tf
from models.operator import initialize_operator
from models.optimization import initialize_optimizer
from models.optimization import initialize_loss_function
from models.summary import initialize_summary_stats
from .runtime_resolve import resolve_runtime_config
from .train_loop_helper import train_step, optimize_ot, log_trajectory_visualizations
from .get_experiment_info import get_problem_name, get_exp_path, get_exp_name

logger = logging.getLogger(__file__)

def train(train_config: dict, use_wandb: bool):
    exp_cfg = train_config['exp_config']
    model_cfg = train_config['operator_config']
    loss_cfg = train_config['loss_config']
    problem_name = get_problem_name(exp_cfg.train_data_path)
    ot_type = loss_cfg.summary_config['type'] if loss_cfg.ot_penalty > 0 else 'no_ot'
    clean_or_noisy = 'noisy' if exp_cfg.noise_level > 0 else 'clean'
    exp_name = get_exp_name(problem_name, ot_type, clean_or_noisy)
    output_path = get_exp_path(problem_name, exp_name)

    device = torch.device(exp_cfg.device)
    dtype = getattr(torch, exp_cfg.dtype)

    if use_wandb: 
        wandb.init(
            project="OTdynSys",
            name=f"{exp_name[:-9]}",
            config={
                'experiment': str(output_path),
                'device': exp_cfg.device,
                'dtype': exp_cfg.dtype,
                'epochs': exp_cfg.epochs,
                'batch_size': exp_cfg.batch_size,
                'model_lr': exp_cfg.model_lr,
                'summary_lr': exp_cfg.summary_lr,
                'short_term_loss': loss_cfg.short_term_loss,
                'dist_loss': loss_cfg.dist_loss,
                'ot_penalty': loss_cfg.ot_penalty,
                'noise_level': exp_cfg.noise_level,
                'rollout_steps': exp_cfg.rollout_steps,
                'crop_window_size': exp_cfg.crop_window_size,
            }
        )

    dataset = ds.DynamicalSystemDataset(
        npz_path=exp_cfg.train_data_path,
        device=device,
        dtype=dtype,
        transforms=[
            # tf.NormalizeTransform(stats_path="data/lorenz63/66f92019/normalization_stats.npy"),
            tf.AddNoiseTransform(noise_level=exp_cfg.noise_level),
            tf.CropTransform(length=exp_cfg.crop_window_size)
        ]
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=exp_cfg.batch_size,
        collate_fn=ds.collate_fn,
        num_workers=0,
        pin_memory=(device.type == 'cpu')
    )

    runtime = resolve_runtime_config(exp_cfg, loss_cfg)

    model = initialize_operator(operator_config=model_cfg, device=device, dtype=dtype, runtime=runtime)
    Lp = initialize_loss_function(loss_config=loss_cfg.short_term_loss)
    ot_c_φ = initialize_loss_function(loss_config=loss_cfg.dist_loss)
    λ_ot = loss_cfg.ot_penalty
    model_optimizer = initialize_optimizer(model=model, learning_rate=exp_cfg.model_lr, optimizer_type=model_cfg.optimizer_type)
    f_φ = initialize_summary_stats(summary_config=loss_cfg.summary_config, device=device, dtype=dtype, runtime=runtime)
    
    if f_φ.is_learnable:
        summary_optimizer = initialize_optimizer(model=f_φ, learning_rate=exp_cfg.summary_lr, optimizer_type=loss_cfg.optimizer_type)
    else:
        summary_optimizer = None

    if use_wandb:
        wandb.config.update({'model_architecture': str(model)})
        wandb.config.update({'summary_architecture': str(f_φ)})
        wandb.watch(model, log='all', log_freq=100)
        wandb.watch(f_φ, log='all', log_freq=100)

    global_step = 0

    for epoch in range(exp_cfg.epochs):
        epoch_metrics = {
            'train/epoch': epoch,
            'train/total_loss': 0.0,
            'train/lp_loss': 0.0,
            'train/ot_loss': 0.0,
        }
        
        num_batches = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{exp_cfg.epochs}')

        for batch_idx, (_, u, param) in enumerate(pbar):
            use_ot = (epoch > exp_cfg.ot_delay)

            optimize = optimize_ot(epoch=epoch, ot_delay=exp_cfg.ot_delay, summary_step_freq=loss_cfg.summary_step_freq)
            
            print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(dataloader)} - Use OT: {use_ot}, loss_cfg.summary_step_freq: {loss_cfg.summary_step_freq}, Optimize ot: {optimize}")

            u_hat, s, s_hat, total_loss, Lp_val, ot_c_φ_val = train_step(
                model=model,
                f_φ=f_φ,
                u=u,
                param=param,
                Lp=Lp,
                ot_c_φ=ot_c_φ,
                model_optimizer=model_optimizer,
                summary_optimizer=summary_optimizer,
                λ_ot=λ_ot,
                rollout_steps=exp_cfg.rollout_steps,
                use_ot=use_ot,
                step_f_φ=optimize_ot(epoch=epoch, ot_delay=exp_cfg.ot_delay, summary_step_freq=loss_cfg.summary_step_freq)
            )


            epoch_metrics['train/total_loss'] += total_loss
            epoch_metrics['train/lp_loss'] += Lp_val
            epoch_metrics['train/ot_loss'] += ot_c_φ_val
            num_batches += 1

            if use_wandb and global_step % 10 == 0:
                wandb.log({
                    'batch/total_loss': total_loss,
                    'batch/lp_loss': Lp_val,
                    'batch/ot_loss': ot_c_φ_val,
                    'batch/step': global_step,
                    'batch/epoch': epoch,
                })
            u_sample = u.detach().cpu().numpy()
            u_hat_sample = u_hat.detach().cpu().numpy()
            s_sample = s.detach().cpu().numpy()
            s_hat_sample = s_hat.detach().cpu().numpy()
            from pipeline.visualization.lorentz63_plots import (plot_summary_stats_comparison)
            fig_summary_stats = plot_summary_stats_comparison(
                    s_true=s_sample[0], s_pred=s_hat_sample[0],
                    title=f"Summary Statistics Comparison (epoch {epoch})",
                )
            if use_wandb and global_step % 10 == 0:
                u_sample = u.detach().cpu().numpy()
                u_hat_sample = u_hat.detach().cpu().numpy()
                s_sample = s.detach().cpu().numpy()
                s_hat_sample = s_hat.detach().cpu().numpy()
                from pipeline.visualization.lorentz63_plots import (plot_summary_stats_comparison)
                fig_summary_stats = plot_summary_stats_comparison(
                        s_true=s_sample[0], s_pred=s_hat_sample[0],
                        title=f"Summary Statistics Comparison (epoch {epoch})",
                    )
                log_trajectory_visualizations(
                    u_sample=u_sample, 
                    u_hat_sample=u_hat_sample, 
                    s_sample=s_sample, 
                    s_hat_sample=s_hat_sample, 
                    epoch=epoch, 
                    global_step=global_step,
                    train_data_path=exp_cfg.train_data_path,
                )

            pbar.set_postfix({
                'loss': f'{total_loss:.4f}',
                'Lp': f'{Lp_val:.4f}',
                'OT': f'{ot_c_φ_val:.4f}'
            })

            global_step += 1

        for key in ['train/total_loss', 'train/lp_loss', 'train/ot_loss']:
            epoch_metrics[key] /= num_batches

        if use_wandb:
            wandb.log(epoch_metrics)

        # logger.info(f"\nEpoch {epoch+1} Summary:")
        # logger.info(f"  Avg Loss: {epoch_metrics['train/total_loss']:.4f}")
        # logger.info(f"  Avg Lp:   {epoch_metrics['train/lp_loss']:.4f}")
        # logger.info(f"  Avg OT:   {epoch_metrics['train/ot_loss']:.4f}")
        

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'summary_state_dict': f_φ.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'summary_optimizer_state_dict': summary_optimizer.state_dict() if f_φ.is_learnable else None,
                'loss': epoch_metrics['train/total_loss'],
            }
            
            if f_φ.is_learnable:
                checkpoint['summary_state_dict'] = f_φ.state_dict()
                checkpoint['summary_optimizer_state_dict'] = summary_optimizer.state_dict()
            
            checkpoint_path = output_path / 'checkpoints' / f'_epoch_{epoch+1}.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(checkpoint, checkpoint_path)
            
            if use_wandb:
                wandb.save(checkpoint_path)

    if use_wandb:
        wandb.finish()
    
    logger.info(f"checkpoint -> {checkpoint_path}")
    logger.info("Training complete!")