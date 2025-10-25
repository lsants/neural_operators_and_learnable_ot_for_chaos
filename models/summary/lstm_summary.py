import torch
from .base_summary import SummaryStatistic
from models.architectures import LSTM
from typing import Callable

torch.nn.LSTM
class LSTMSummaryStats(SummaryStatistic):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int):
        super().__init__()