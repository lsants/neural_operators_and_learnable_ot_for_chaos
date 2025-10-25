import torch
import json
from pathlib import Path
from typing import Any
from .mlp_timestepper import TimeStepperMLP
from models.architectures.activation_fns import ACTIVATION_MAP

OPERATOR_REGISTRY = {
    'fno': {
        'class': None,
        'config_path': 'configs/operators/fno.json'
    },
    'deeponet': {
        'class': None,
        'config_path': 'configs/operators/deeponet.json'
    },
    'mlp': {
        'class': TimeStepperMLP,
        'config_path': 'configs/operators/mlp_timestepper_config.json'
    },
}

def load_operator_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    print(path)
    if not path.exists():
        raise FileNotFoundError(f"Operator config not found: {config_path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def process_arch_json(config: dict) -> dict[str, Any]:
    if 'activation' in config:
        config['activation'] = ACTIVATION_MAP[config['activation']]
    return config

def initialize_operator(operator_config, device: torch.device, dtype: torch.dtype, **kwargs):
    operator_type = operator_config.operator_type.lower()

    if operator_type not in OPERATOR_REGISTRY:
        raise ValueError(
            f"Unknown operator type: {operator_type}. "
            f"Available: {list(OPERATOR_REGISTRY.keys())}"
        )
    
    operator_info = OPERATOR_REGISTRY[operator_type]
    operator_class = operator_info['class']
    config_path = operator_info['config_path']

    if operator_class is None:
        raise NotImplementedError(
            f"Operator class for '{operator_type}' not yet implemented"
        )
    
    arch_config = load_operator_config(config_path)
    arch_config['input_dim'] = kwargs['runtime']['emulator_input_dim']
    arch_config['output_dim'] = kwargs['runtime']['emulator_output_dim']
    processed_arch_config = process_arch_json(arch_config)
    
    print(processed_arch_config)

    model = operator_class(**processed_arch_config)
    model = model.to(device=device, dtype=dtype)

    return model