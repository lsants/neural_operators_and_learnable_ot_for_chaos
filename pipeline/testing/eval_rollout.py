import torch
from typing import Callable

@torch.no_grad()
def eval_rollout(
    emulator: torch.nn.Module,
    u: torch.Tensor,
    param: torch.Tensor,
    rollout_steps: int,
    loss_fn: Callable | None = None
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

    if loss_fn is not None:
        loss = loss_fn(u, predictions)
        return predictions, loss
    else:
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