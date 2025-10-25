from .loss_fns import SinkhornDivergence, LpNorm
from typing import Any

LOSS_REGISTRY = {
    'l2': {
        'class': LpNorm,
        'params': {'p': 2}
    },
    
    'sinkhorn': {
        'class': SinkhornDivergence,
        'params': {'blur': 0.05, 'p': 2, 'backend': 'tensorized'}
    },
    
}

def initialize_loss_function(loss_config: str | dict[str, Any], **kwargs):
    if isinstance(loss_config, str):
        loss_type = loss_config.lower()
        loss_params = {}
    elif isinstance(loss_config, dict):
        loss_type = loss_config.get('type', '').lower()
        loss_params = {k:v for k,v in loss_config.items() if k != 'type'}
    else:
        raise TypeError(f"loss_config must be str or dict, got {type(loss_config)}")
    
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )
    
    loss_info = LOSS_REGISTRY[loss_type]
    loss_class = loss_info['class']
    default_params = loss_info['params'].copy()

    default_params.update(loss_params)
    default_params.update(kwargs)

    loss_fn = loss_class(**default_params)

    return loss_fn