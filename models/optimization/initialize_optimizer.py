import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any

OPTIMIZER_REGISTRY = {
    'adam': torch.optim.Adam,
}

SCHEDULER_REGISTRY = {
    'step': torch.optim.lr_scheduler.StepLR,
}

def initialize_optimizer(
        model: torch.nn.Module,
        learning_rate: float,
        optimizer_type: str,
        scheduler_config: dict[str, Any] | None = None,
        **optimizer_kwargs
) -> Optimizer | tuple[Optimizer, _LRScheduler]:
    optimizer_type = optimizer_type.lower()

    if optimizer_type not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    
    optimizer_class = OPTIMIZER_REGISTRY[optimizer_type]
    
    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        **optimizer_kwargs
    )

    if scheduler_config is not None:
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type not in SCHEDULER_REGISTRY:
            raise ValueError(
                f"Unknown scheduler type: {scheduler_type}"
                f"Available: {list(SCHEDULER_REGISTRY.keys())}"
            )
        
        scheduler_class = SCHEDULER_REGISTRY[scheduler_type]
        scheduler_params = {k:v for k, v in scheduler_config.items() if k != 'type'}
        scheduler = scheduler_class(optimizer, **scheduler_params)
        
        return optimizer, scheduler

    return optimizer