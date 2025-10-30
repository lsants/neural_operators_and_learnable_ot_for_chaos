import torch
import json
from pathlib import Path
from typing import Any
from .mlp_summary import MLPSummaryStats
from .fixed_summary import FixedSummaryStats
from .identity_summary import IdentitySummaryStats
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
    'identity': {
        'class': IdentitySummaryStats,
        'config_path': None
    },
}

def load_summary_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary config not found: {config_path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def initialize_summary_stats(
        summary_config,
        device: torch.device,
        dtype: torch.dtype,
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
    if config_path is None:
        module = summary_class()
        module = module.to(device=device, dtype=dtype)
        summary_config.clear()
        return module
    else:
        arch_config = load_summary_config(config_path)
        arch_config['input_dim'] = summary_config['input_dim']
        arch_config['output_dim'] = summary_config['output_dim']
        summary_config.pop('type')
        summary_config.update(**arch_config)

    module = summary_class(**summary_config)
    module = module.to(device=device, dtype=dtype)

    return module

if __name__ == "__main__":
    # Example usage
    device = torch.device('cpu')
    dtype = torch.float32
    runtime = {
        'summary_input_dim': 3,
        'summary_output_dim': 1
    }
    summary_config = {
        'type': 'identity'
    }
    summary_stats = initialize_summary_stats(
        summary_config,
        device=device,
        dtype=dtype,
        runtime=runtime
    )
    print(summary_stats)