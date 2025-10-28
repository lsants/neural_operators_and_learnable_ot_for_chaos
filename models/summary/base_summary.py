import torch
from abc import ABC, abstractmethod

class SummaryStatistic(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def summary_dim(self) -> int:
        pass

    @property
    def is_learnable(self):
        return len(list(self.parameters())) > 0