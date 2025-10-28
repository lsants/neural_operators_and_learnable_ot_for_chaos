import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class TrainConfig:
    train_data_path: str
    val_data_path: str
    device: str
    dtype: str
    epochs: int
    batch_size: int
    model_lr: float
    summary_lr: float
    ot_delay: int 
    rollout_steps: int
    noise_level: float
    crop_window_size: int    

    def post_init(self):
        pass

@dataclass
class OperatorConfig:
    operator_type: str
    optimizer_type: str

    def post_init(self):
        pass

@dataclass
class LossConfig:
    summary_config: dict[str, str]
    p_val: float
    blur: float
    geom_loss_p: int
    short_term_loss: str
    dist_loss: str
    optimizer_type: str
    ot_penalty: float
    num_summary_stats: int
    feature_penalty: float
    summary_step_freq: int
    
    def post_init(self):
        pass

def get_train_configs(config_paths: list[Path]):
    with open(config_paths[0], 'r') as f:
        train_config_dict = json.load(f)
        train_config_obj = TrainConfig(**train_config_dict)
        train_config_obj.post_init()
    with open(config_paths[1], 'r') as f:
        operator_config_dict = json.load(f)
        operator_config_obj = OperatorConfig(**operator_config_dict)
        operator_config_obj.post_init()
    with open(config_paths[2], 'r') as f:
        loss_config_dict = json.load(f)
        loss_config_obj = LossConfig(**loss_config_dict)
        loss_config_obj.post_init()

    train_config = {
        "exp_config": train_config_obj,
        "operator_config": operator_config_obj,
        "loss_config": loss_config_obj,
    }
    return train_config