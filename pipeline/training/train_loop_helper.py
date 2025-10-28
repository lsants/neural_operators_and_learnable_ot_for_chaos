import wandb
import torch
import numpy as np
from typing import Callable
from pipeline.visualization import plot_1d_components_comparison, plot_3d_trajectories_comparison, plot_summary_stats_comparison, plot_error_evolution
from matplotlib import pyplot as plt
from .get_experiment_info import get_dataset_config
from pipeline.metric_bookkeping.compute_metrics import compute_trajectory_errors, compute_summary_errors


def optimize_ot(epoch: int, ot_delay: int, summary_step_freq: int) -> bool:
    optimize = (epoch > ot_delay) and (epoch % summary_step_freq == 0) and epoch < 27
    return optimize

def normalize(s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = s.mean(dim=1, keepdim=True)
    std = s.std(dim=1, keepdim=True) + eps
    s_normalized = (s - mean) / std
    s_mean = s.mean(dim=1, keepdim=True)
    return s_mean / std # Testing because OT loss was negative when using full normalized features

def rollout(
    emulator: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    rollout_steps: int,
) -> torch.Tensor:
    
    batch_size, total_time_steps, spatial_dim = u.shape
    ic_indices = torch.arange(0, total_time_steps, rollout_steps, device=u.device)
    add_final_step = (ic_indices[-1] != total_time_steps - 1)
    if not add_final_step and len(ic_indices) > 1:
        ic_indices = ic_indices[:-1]

    num_ics = len(ic_indices)

    u_ics = u[:, ic_indices]

    current_state = u_ics.reshape(batch_size * num_ics, spatial_dim)
    params_expanded = param.repeat_interleave(num_ics, dim=0)
    
    prediction_list = [u_ics]

    for step in range(rollout_steps - 1):
        next_state = emulator(current_state, params_expanded)
        prediction_list.append(next_state.reshape(batch_size, num_ics, spatial_dim))
        current_state = next_state

    predictions = torch.stack(prediction_list, dim=2)
    predictions = predictions.reshape(batch_size, -1, spatial_dim)

    if add_final_step:
        predictions = torch.cat([predictions, u[:, total_time_steps - 1 : total_time_steps, :]], dim=1)

    predictions = predictions[:, :total_time_steps, :]
    assert predictions.shape == u.shape, f"Shape mismatch: {predictions.shape} vs {u.shape}"

    return predictions

def rollout_with_short_horizon(
    emulator: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    loss_fn: Callable,
    short_horizon: int # must be at least 2
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_size, total_time_steps, spatial_dim = u.shape
    ic_indices = torch.arange(0, total_time_steps, short_horizon, device=u.device)
    last_ic = ic_indices[-1].item()
    last_prediction = last_ic + short_horizon - 1
    add_final_step = (last_prediction < total_time_steps - 1)

    num_ics = len(ic_indices)
    u_ics = u[:, ic_indices]

    current_state = u_ics.reshape(batch_size * num_ics, spatial_dim)
    params_expanded = param.repeat_interleave(num_ics, dim=0)
    
    all_predictions = [u_ics]
    loss_predictions = []
    loss_targets = []

    for step in range(short_horizon - 1):
        next_state = emulator(current_state, params_expanded)
        next_state_reshaped = next_state.reshape(batch_size, num_ics, spatial_dim)
        all_predictions.append(next_state_reshaped)

        target_time_indices = ic_indices + step + 1

        valid_mask = target_time_indices < total_time_steps
        if valid_mask.any():
            valid_targets = target_time_indices[valid_mask]
            u_target = u[:, valid_targets]
            u_pred = next_state_reshaped[:, valid_mask, :]
            loss_predictions.append(u_pred)
            loss_targets.append(u_target)
        current_state = next_state

    if loss_predictions:
        all_preds = torch.cat(loss_predictions, dim=1)
        all_targets = torch.cat(loss_targets, dim=1)

        loss = loss_fn(all_targets, all_preds)
    else:
        loss = torch.tensor(0.0, device=u.device)

    predictions = torch.stack(all_predictions, dim=2)
    predictions = predictions.reshape(batch_size, -1, spatial_dim)

    if add_final_step:
        predictions = torch.cat(
            [predictions, u[:, total_time_steps-1:total_time_steps, :]], 
            dim=1
        )
    predictions = predictions[:, :total_time_steps, :]

    assert predictions.shape == u.shape
    return predictions, loss
    


def train_step(
    model: torch.nn.Module,
    f_φ: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    Lp: Callable,
    ot_c_φ: Callable,
    model_optimizer: torch.optim.Optimizer,
    summary_optimizer: torch.optim.Optimizer,
    λ_ot: float,
    rollout_steps: int,
    use_ot: bool,
    step_f_φ: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:

    model.train()
    if f_φ.is_learnable:
        f_φ.train()

    model_optimizer.zero_grad()

    u_hat, Lp_batch = rollout_with_short_horizon(
        emulator=model, u=u, param=param, loss_fn=Lp, short_horizon=rollout_steps)

    if use_ot:
        with torch.no_grad():
            s = f_φ(u)
            s_hat = f_φ(u_hat)

        if s.dim() == 2:
            s = s.unsqueeze(1)
            s_hat = s_hat.unsqueeze(1)
            
        ot_c_φ_batch = ot_c_φ(normalize(s), normalize(s_hat))

        model_loss = Lp_batch + λ_ot * ot_c_φ_batch
        
        # print(Lp_batch, ot_c_φ_batch * λ_ot)
    else:
        model_loss = Lp_batch
        ot_c_φ_batch = torch.tensor(0.0)

    model_loss.backward()
    model_optimizer.step()

    if use_ot:
        if f_φ.is_learnable:
            summary_optimizer.zero_grad()
        s = f_φ(u)
        u_hat_detached = u_hat.detach()
        s_hat = f_φ(u_hat_detached)
        ot_c_φ_batch = ot_c_φ(normalize(s), normalize(s_hat))
        summary_loss = -ot_c_φ_batch

        if step_f_φ and f_φ.is_learnable:
            summary_loss.backward()
            summary_optimizer.step()
    else:
        with torch.no_grad():
            s = f_φ(u)
            u_hat_detached = u_hat.detach()
            s_hat = f_φ(u_hat_detached)
            ot_c_φ_batch = torch.tensor(0.0)

    return (
        u_hat,
        s,
        s_hat,
        model_loss.item(),
        Lp_batch.item(),
        ot_c_φ_batch.item()
    )

def log_trajectory_visualizations(u_sample: np.ndarray, 
                                  u_hat_sample: np.ndarray, 
                                  s_sample: np.ndarray, 
                                  s_hat_sample: np.ndarray, 
                                  epoch: int, 
                                  global_step: int,
                                  **kwargs):
    dataset_config = get_dataset_config(kwargs.get("train_data_path", {}).replace('npz', 'json'))

    traj_errors = compute_trajectory_errors(u_sample, u_hat_sample)
    summary_errors = compute_summary_errors(s_sample, s_hat_sample)

    wandb.log({
        'errors/trajectory_mse': traj_errors['mse'],
        'errors/trajectory_rmse': traj_errors['rmse'],
        'errors/trajectory_mae': traj_errors['mae'],
        'errors/trajectory_l2': traj_errors['l2_error'],
        'errors/trajectory_max': traj_errors['max_error'],
        'errors/trajectory_mse_x': traj_errors['mse_per_component'][0],
        'errors/trajectory_mse_y': traj_errors['mse_per_component'][1],
        'errors/trajectory_mse_z': traj_errors['mse_per_component'][2],
        'errors/summary_mse': summary_errors['summary_mse'],
        'errors/summary_mae': summary_errors['summary_mae'],
        'epoch': epoch,
        'step': global_step
    })

    u_true_np = u_sample[0]
    u_hat_np = u_hat_sample[0]
    s_true_np = s_sample[0]
    s_hat_np = s_hat_sample[0]

    fig_3d = plot_3d_trajectories_comparison(
        u_true=u_true_np, u_pred=u_hat_np,
        title=f"3D Trajectory (epoch {epoch})",
    )
    fig_1d = plot_1d_components_comparison(
        u_true=u_true_np, u_pred=u_hat_np, dt=dataset_config['dt'],
        title=f"Component-wise comparison (epoch {epoch})",
    )
    fig_summary_stats = plot_summary_stats_comparison(
        s_true=s_true_np, s_pred=s_hat_np,
        title=f"Summary Statistics Comparison (epoch {epoch})",
    )
    fig_error = plot_error_evolution(traj_errors['mse_per_time'],
                                     traj_errors['spectral_distance'],
                                      title=f"Error Evolution (Epoch {epoch})")

    wandb.log({
        'visualizations/3d_trajectory': wandb.Image(fig_3d),
        'visualizations/1d_trajectory': wandb.Image(fig_1d),
        'visualizations/summary_stats': wandb.Image(fig_summary_stats),
        'visualizations/error_evolution': wandb.Image(fig_error),
        'batch/step': global_step,
        'batch/epoch': epoch,
    })
    plt.close(fig_3d)
    plt.close(fig_1d)
    plt.close(fig_summary_stats)
    plt.close(fig_error)