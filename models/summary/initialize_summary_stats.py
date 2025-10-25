import torch
import json
from pathlib import Path
from typing import Any
from .mlp_summary import MLPSummaryStats
from .fixed_summary import FixedSummaryStats
from models.architectures.activation_fns import ACTIVATION_MAP

SUMMARY_REGISTRY = {
    'fixed': {
        'class': FixedSummaryStats,
        'config_path': 'configs/summary_statistics/fixed.json'
    },
    'mlp': {
        'class': MLPSummaryStats,
        'config_path': 'configs/summary_statistics/mlp.json'
    },
}

def load_summary_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary config not found: {config_path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def process_arch_json(config: dict) -> dict[str, Any]:
    if 'activation' in config:
        config['activation'] = ACTIVATION_MAP[config['activation']]
    return config

def initialize_summary_stats(
        summary_config,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs
):

    summary_type = summary_config['type']
    if summary_type not in SUMMARY_REGISTRY:
        raise ValueError(
            f"Unknown summary type: {summary_type}. "
            f"Available: {list(SUMMARY_REGISTRY.keys())}"
        )
    
    summary_info = SUMMARY_REGISTRY[summary_type]
    summary_class = summary_info['class']
    config_path = summary_info['config_path']

    arch_config = load_summary_config(config_path)
    arch_config['input_dim'] = kwargs['runtime']['summary_input_dim']
    arch_config['output_dim'] = kwargs['runtime']['summary_output_dim']
    processed_arch_config = process_arch_json(arch_config)

    module = summary_class(**processed_arch_config)
    module = module.to(device=device, dtype=dtype)

    return module