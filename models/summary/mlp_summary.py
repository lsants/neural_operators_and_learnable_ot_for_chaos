import torch
from typing import Callable
from .base_summary import SummaryStatistic
from models.architectures import MLP
from models.architectures.activation_fns import ACTIVATION_MAP

class MLPSummaryStats(SummaryStatistic):
    def __init__(self, input_dim: int, hidden_layers: list[int], activation: str, output_dim: int):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_layers, output_dim, ACTIVATION_MAP[activation])
        self._summary_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x_flat = x.reshape(-1, d)
        embedded_x = self.mlp(x_flat)
        learned_summary = embedded_x.reshape(B, T, self._summary_dim)
        return learned_summary

    @property
    def summary_dim(self) -> int:
        return self._summary_dim