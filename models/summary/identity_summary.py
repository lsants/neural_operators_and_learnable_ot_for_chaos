import torch
from .base_summary import SummaryStatistic

class IdentitySummaryStats(SummaryStatistic):
    def __init__(self,):
        super().__init__()
        self.identity = torch.nn.Identity()

    @property
    def summary_dim(self) -> None:
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x)