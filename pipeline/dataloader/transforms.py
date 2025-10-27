from typing import Protocol
import torch
import numpy as np

class Transform(Protocol):
    def __call__(self, traj_id: str, traj: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        ...

class AddNoiseTransform(Transform):
    def __init__(self, noise_level: float, seed: int | None = None):
        self.noise_level = noise_level
        self.rng = torch.Generator() if seed is not None else None
        if seed is not None:
            self.rng.manual_seed(seed) # pyright: ignore[reportOptionalMemberAccess]

    def __call__(self, traj_id: str, traj: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        noise = self.noise_level * traj.std(dim=0, keepdim=True) * torch.randn_like(traj)
        return traj + noise
    
class CropTransform(Transform):
    def __init__(self, length: int):
        self.length = length

    def __call__(self, traj_id: str, traj: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return traj[:self.length]

class NormalizeTransform(Transform):
    def __init__(self, stats_path: str = None, mean: torch.Tensor = None, std: torch.Tensor = None):
        """
        Normalize trajectories using provided mean and std, or load from stats file.
        Args:
            stats_path: Path to .npy file containing normalization stats
            mean: Manual mean values 
            std: Manual std values
        """
        if stats_path is not None:
            stats = np.load(stats_path, allow_pickle=True).item()
            self.mean = torch.tensor(stats['mean'], dtype=torch.float32)
            self.std = torch.tensor(stats['std'], dtype=torch.float32)
        elif mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

    def __call__(self, traj_id: str, traj: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            # Move to same device as trajectory
            mean = self.mean.to(traj.device)
            std = self.std.to(traj.device)
            return (traj - mean) / (std + 1e-8)  # avoid division by zero
        else:
            # Compute per-trajectory normalization (less ideal but fallback)
            mean = traj.mean(dim=0, keepdim=True)
            std = traj.std(dim=0, keepdim=True) + 1e-8  # avoid division by zero
            return (traj - mean) / std