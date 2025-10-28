import torch
from .base_summary import SummaryStatistic
from models.architectures import MLP
from typing import Callable

class MLPSummaryStats(SummaryStatistic):
    def __init__(self, input_dim: int, hidden_layers: list[int], activation: Callable, embedding_dim: int, output_dim: int):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_layers, embedding_dim, activation)
        self._summary_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        embedded_x = self.mlp(x_flat)
        learned_summary = embedded_x.reshape(batch_size, -1, self._summary_dim)

        return learned_summary

    @property
    def summary_dim(self) -> int:
        return self._summary_dim