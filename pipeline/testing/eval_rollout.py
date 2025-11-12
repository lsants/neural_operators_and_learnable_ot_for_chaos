import torch
from typing import Callable

@torch.no_grad()
def eval_rollout(
    emulator: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    rollout_steps: int|None = None,
) -> torch.Tensor:
    """Full rollout of the emulator.

    Args:
        emulator (torch.nn.Module): Emulator of the dynamics.
        u (torch.Tensor): Data tensor of shape (B, T, d).
        param (torch.Tensor): Parameters for the emulator (problem dependent).

    Returns:
        torch.Tensor: Rollout predictions of shape (B, T, d).
    """
    emulator.eval()
    B, T, U = u.shape
    u_hat = torch.zeros((B, T, U), dtype=u.dtype, device=u.device)
    u_hat[:, 0, :] = u[:, 0, :]
    steps = T if rollout_steps is None else rollout_steps
    for t in range(1, steps):
        u_hat[:, t, :] = emulator(u_hat[:, t-1, :], param)

    return u_hat

@torch.no_grad()
def eval_anchor_rollout(
    emulator: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    rollout_steps: int,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    emulator.eval()
    pred = u.unsqueeze(1) # [B, 1, D]
    current = u

    predictions = [pred]
    for _ in range(rollout_steps - 1):
        next_state = emulator(current, param)
        predictions.append(next_state.unsqueeze(1))
        current = next_state
    predictions = torch.cat(predictions, dim=1)  # [B, T, D]


    return predictions
    
def summary_rollout(
    f_φ: torch.nn.Module,
    u: torch.Tensor,
    rollout_steps: int
) -> torch.Tensor:
    f_φ.eval()
    with torch.no_grad():
        s = f_φ(u)  # [B, T, S]
        s_rollout = s[:, :rollout_steps, :]  # [B, R, S]
    return s_rollout