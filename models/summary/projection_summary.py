import torch
from .base_summary import SummaryStatistic

class ProjectionSummaryStats(SummaryStatistic):
    def __init__(self, state: int):
        super().__init__()
        self.state = state
        self.identity = torch.nn.Identity()

    @property
    def summary_dim(self) -> None:
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)
        projection = identity[:, :, self.state][..., None]
        return projection