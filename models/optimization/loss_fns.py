import torch
from geomloss import SamplesLoss

class SinkhornDivergence:
    def __init__(self, blur: float, p: int, backend: str):
        self.loss = SamplesLoss(loss='sinkhorn', p=p, blur=blur, debias=True, backend=backend)

    def __call__(self, u_i: torch.Tensor, u_hat_i: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.loss(u_i, u_hat_i)) + 1e-12

class LpNorm:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, u: torch.Tensor, u_hat: torch.Tensor):
        return (torch.mean((u - u_hat)**self.p))