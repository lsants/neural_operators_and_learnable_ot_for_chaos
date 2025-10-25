from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

@dataclass(frozen=True)
class DataGenConfig:
    experiment: str
    n_samples: int
    t_start: float
    t_end: float
    dt: float
    trajectory_length: int
    param_ranges: dict[str, Any]
    n_dim: int
    noise_level: float
    output_dir: Path

    @classmethod
    def from_json(cls, json_path: Path | str) -> "DataGenConfig":
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def __post_init__(self):
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be > t_start")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.trajectory_length <= 0:
            raise ValueError("trajectory_length must be positive")
        if not self.param_ranges:
            raise ValueError("param_ranges cannot be empty")

