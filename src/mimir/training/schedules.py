from __future__ import annotations
from typing import Literal, Optional
import math
from mimir.model.module_learning_rates import configured_module_rates, module_lr_scales, module_lr_metrics, update_auto_module_rates, windowed_cosine_lr
from mimir.model.lr_rewarm import resolve_rewarm, rewarm_lr, rewarm_metadata
from mimir.model.lr_piecewise import piecewise_lr, validate_lr_points
from mimir.training.config import PretrainConfig
from mimir.training.state import TrainState


def update_lr(config: PretrainConfig, train_state: TrainState, cooldown_ratio: Optional[float] = None) -> float:
    # Linear warmup cosine schedule
    if getattr(config, 'lr_piecewise_points', None) is not None:
        lr = piecewise_lr(config.lr_piecewise_points, train_state.step)
    elif cooldown_ratio is not None:
        lr = config.lr * cooldown_ratio
    elif config.lr_rewarm_steps:
        lr = rewarm_lr(config, train_state.step)
    elif config.lr_decay_start_step is not None or config.lr_decay_end_step is not None:
        lr = windowed_cosine_lr(config.lr, config.lr_min_ratio, train_state.step,
                               config.lr_decay_start_step, config.lr_decay_end_step)
    elif train_state.step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, train_state.step / config.lr_warmup_steps)
    else:
        progress = (train_state.step - config.lr_warmup_steps) / (train_state.total_steps - config.lr_warmup_steps)
        lr = config.lr * (config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))

    for param_group in train_state.optim.param_groups:
        param_group["lr"] = lr

    return lr

