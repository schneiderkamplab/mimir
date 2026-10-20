from __future__ import annotations
from typing import Literal, Optional
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch import Tensor, nn
from mimir.model.transformer import Transformer, TransformerBlock
from mimir.training.config import PretrainConfig


def apply_fsdp(
    module: nn.Module,
    param_dtype: torch.dtype,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: bool = False,
):
    fully_shard(module,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype,
                                               reduce_dtype=torch.get_default_dtype()),  # Use master dtype for reduction
                reshard_after_forward=reshard_after_forward)
    
    assert isinstance(module, FSDPModule)
    # Disable gradient division. Adams is scale invariant.
    # The FSDP2 API for this was renamed across torch versions; support both.
    if hasattr(module, "set_gradient_divide_factor"):
        module.set_gradient_divide_factor(1.0)
    else:
        module.set_reduce_scatter_divide_factor(1.0)
    if hasattr(module, "set_force_sum_reduction_for_comms"):
        module.set_force_sum_reduction_for_comms(True)



def create_fsdp_mesh(config: PretrainConfig) -> Optional[DeviceMesh]:
    """Create a native FSDP2 HSDP mesh, preserving the default 1D path."""
    degree = config.fsdp_shard_degree
    if degree is None:
        return None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("fsdp_shard_degree requires initialized distributed training")

    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    if degree > local_world_size:
        raise ValueError(
            f"fsdp_shard_degree={degree} exceeds LOCAL_WORLD_SIZE={local_world_size}"
        )
    if world_size % degree != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by fsdp_shard_degree={degree}"
        )
    if degree == world_size:
        # mesh=None is the established full-world FSDP path.
        return None

    return init_device_mesh(
        "cuda",
        (world_size // degree, degree),
        mesh_dim_names=("replicate", "shard"),
    )



def fsdp_reshard_after_forward(config: PretrainConfig, *, checkpointed: bool) -> bool:
    if config.fsdp_reshard_after_forward is not None:
        return config.fsdp_reshard_after_forward
    # Preserve the existing activation-checkpointing integration by default.
    return checkpointed



def fsdp_wrap_modules(model: nn.Module, policy: str) -> list[nn.Module]:
    """Select bottom-up FSDP units while leaving the root wrapper unchanged."""
    if policy == "transformer_block":
        return [module for module in model.modules() if isinstance(module, TransformerBlock)]
    if policy == "recurrent_level":
        return [module for module in model.modules() if isinstance(module, Transformer)]
    raise ValueError(f"Unsupported fsdp_wrap_policy: {policy}")



def contains_checkpointed_block(
    module: nn.Module,
    checkpointed_blocks: set[nn.Module],
) -> bool:
    return any(child in checkpointed_blocks for child in module.modules())



def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model



def fsdp_model_dtype(config: PretrainConfig) -> torch.dtype:
    if config.fsdp_params_precision == "fp32":
        return torch.float32
    if config.fsdp_params_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported fsdp_params_precision: {config.fsdp_params_precision}")



def ddp_model_dtype(config: PretrainConfig, fwd_bwd_dtype: torch.dtype) -> torch.dtype:
    if config.ddp_params_precision == "fp32":
        return torch.float32
    if config.ddp_params_precision == "bf16":
        return fwd_bwd_dtype
    raise ValueError(f"Unsupported ddp_params_precision: {config.ddp_params_precision}")



def optimizer_ema_dtype(config: PretrainConfig) -> Optional[torch.dtype]:
    if config.distributed_strategy == "ddp" and config.ddp_params_precision == "bf16":
        return torch.float32
    if config.distributed_strategy == "fsdp" and config.fsdp_params_precision == "bf16":
        return torch.float32
    return None

