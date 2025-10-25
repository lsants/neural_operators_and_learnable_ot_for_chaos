from typing import Protocol, Any
import torch

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