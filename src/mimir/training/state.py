from __future__ import annotations
from typing import Literal, Optional
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from mimir.model.layers import Carry
from mimir.model.adam_atan2 import AdamATan2


@dataclass
class TrainState:
    model: nn.Module
    carry: Optional[Carry]
    
    optim: AdamATan2

    step: int
    total_steps: int
    fwd_bwd_dtype: torch.dtype
    use_cuda_autocast: bool



@dataclass
class ResumeState:
    tag: str
    step: int
    start_epoch: int
    skip_batches: int
    start_row_cursor: Optional[int] = None
    resume_mode: str = "batch"

