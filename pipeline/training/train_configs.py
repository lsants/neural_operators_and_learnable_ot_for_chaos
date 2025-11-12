import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class TrainConfig:
    #------------------------
    # Experiment settings
    train_data_path: str
    val_data_path: str
    device: str
    dtype: str
    epochs: int
    batch_size: int
    ot_delay: int 
    rollout_steps: int
    noise_level: float
    crop_window_size: int    
    summary_step_freq: int
    clip_summary_grad_norm: float
    #------------------------
    # Model settings
    emulator_lr: float
    emulator_config: dict[str, Any]
    emulator_optimizer_type: str
    #------------------------
    # Summary settings
    summary_config: dict[str, Any]
    summary_optimizer_type: str
    summary_lr: float
    #------------------------
    # Loss settings
    distribution_loss: str
    short_term_loss: str
    ot_penalty: float
    ot_penalty_increase: float
    feature_penalty: float
    geom_loss_p: int
    blur: float
    p_norm: float
    #------------------------

    def post_init(self):
        pass

def get_train_configs(config_paths: list[Path]) -> dict[str, TrainConfig]:
    with open(config_paths[0], 'r') as f:
        train_config_dict = json.load(f)
        train_config_obj = TrainConfig(**train_config_dict)
        train_config_obj.post_init()

    train_config = {
        "exp_config": train_config_obj,
    }
    return train_config