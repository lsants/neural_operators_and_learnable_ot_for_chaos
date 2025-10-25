import torch
from .base_summary import SummaryStatistic
from models.architectures import MLP
from typing import Callable

class MLPSummaryStats(SummaryStatistic):
    def __init__(self, input_dim: int, hidden_layers: list[int], activation: Callable, output_dim: int):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_layers, output_dim, activation)
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.mlp(x_flat)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim