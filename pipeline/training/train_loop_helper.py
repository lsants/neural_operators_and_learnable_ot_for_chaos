import torch
from typing import Callable


def optimize_ot(epoch: int, ot_delay: int, summary_step_freq: int) -> bool:
    optimize = (epoch > ot_delay) and (epoch % summary_step_freq == 0) and epoch < 25
    return optimize

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
    short_horizon: int #must be at least 2
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
) -> tuple[float, float, float]:

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
            
        ot_c_φ_batch = ot_c_φ(s, s_hat)

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
        ot_c_φ_batch = ot_c_φ(s, s_hat)

        summary_loss = -ot_c_φ_batch
        
        if step_f_φ:
            summary_loss.backward()
            summary_optimizer.step()

    return (
         model_loss.item(),
         Lp_batch.item(),
         ot_c_φ_batch.item() if use_ot else 0
    )