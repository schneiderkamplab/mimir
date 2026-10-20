from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch import Tensor, nn
from torch.utils.data import DataLoader
from mimir.model.layers import Carry
from mimir.model.activation_checkpointing import apply_activation_checkpointing
from mimir.model.accelerator import (
    AcceleratorType,
    empty_accelerator_cache,
    memory_stats_for_device,
    set_accelerator_type,
    synchronize_device,
    torch_device_for_accelerator,
)
from mimir.model.adam_atan2 import AdamATan2
from mimir.model.module_learning_rates import configured_module_rates, module_lr_scales, module_lr_metrics, update_auto_module_rates, windowed_cosine_lr
from mimir.model import Mimir2
from mimir.model.lm_head import LMHead
from mimir.corpus.sampled import V1Dataset, V1DatasetConfig, V1DatasetMeta
from mimir.training.config import PretrainConfig
from mimir.training.state import TrainState


def create_dataloader(
    config: PretrainConfig,
    local_batch_size: int,
    drop_last_batch: bool,
    rank: int,
    world_size: int,
    dataset_path: Optional[str] = None,
):
    dataset = V1Dataset(V1DatasetConfig(
        seed=config.seed,

        dataset_path=dataset_path or config.data.path,
        drop_last_batch=drop_last_batch,

        target_only=config.data.target_only,

        batch_max_length=local_batch_size,
        rank=rank,
        num_replicas=world_size,
        homogeneous_categories=config.data.homogeneous_categories if dataset_path is None else False,
        routing_categories=config.data.routing_categories if dataset_path is None else False,
    ))
    num_workers = 0 if config.accelerator_type in ("mps", "cpu", "none") else 1
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "generator": torch.Generator().manual_seed(config.seed + rank),
        "pin_memory": config.accelerator_type in ("sm90", "sm100", "rocm"),
    }
    if num_workers > 0:
        dataloader_kwargs |= {
            "prefetch_factor": config.dataloader_prefetch_factor,
            "persistent_workers": True,  # NOTE: Required for correct epoch handling
        }
    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader, dataset.metadata



def compute_train_extra_args(model: nn.Module, train_state: TrainState) -> dict:
    from mimir.training.distributed import unwrap_model
    return unwrap_model(model).compute_train_extra_args(train_state)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]



def initial_model_carry(model: nn.Module, local_batch_size: int, dtype: torch.dtype) -> Optional[Carry]:
    from mimir.training.distributed import unwrap_model
    module = unwrap_model(model)
    if hasattr(module, "initial_carry"):
        return module.initial_carry(local_batch_size, dtype=dtype)  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(module, "model") and hasattr(module.model, "initial_carry"):
        return module.model.initial_carry(local_batch_size, dtype=dtype)  # pyright: ignore[reportAttributeAccessIssue]
    raise AttributeError(f"{type(module).__name__} does not expose initial_carry")



def create_model_and_carry(config: PretrainConfig, train_metadata: V1DatasetMeta, local_batch_size: int, device: torch.device):
    from mimir.training.distributed import apply_fsdp
    from mimir.training.distributed import contains_checkpointed_block
    from mimir.training.distributed import create_fsdp_mesh
    from mimir.training.distributed import ddp_model_dtype
    from mimir.training.distributed import fsdp_model_dtype
    from mimir.training.distributed import fsdp_reshard_after_forward
    from mimir.training.distributed import fsdp_wrap_modules
    from mimir.training.distributed import optimizer_ema_dtype
    model_cfg = (
        config.arch.model_dump()
        | train_metadata.model_dump()
        | config.data.model_dump()
        | {
            "fwd_bwd_dtype": config.fwd_bwd_dtype,
            "activation_checkpointing": config.activation_checkpointing,
        }
    )
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)
    is_v2 = config.arch.name == 'mimir2'
    if (config.data.homogeneous_categories or config.data.routing_categories) and not is_v2:
        raise ValueError('Metadata-guided routing currently requires HRM v2')
    if is_v2 and config.fsdp_wrap_policy == 'recurrent_level' and (
        model_cfg.get('gates_enabled', False) or model_cfg.get('mos_num_experts', 1) > 1
        or model_cfg.get('h_memory_branches', 1) > 1 or model_cfg.get('l_memory_branches', 1) > 1
    ):
        raise ValueError('V2 direct level parameter access requires fsdp_wrap_policy=transformer_block')

    # Instantiate model with head
    model_cls = Mimir2
    head_cls = LMHead

    with torch.device(device):
        model: nn.Module = model_cls(model_cfg)
        carry = model.initial_carry(local_batch_size, dtype=fwd_bwd_dtype)  # pyright: ignore[reportCallIssue]
        # Attach loss head
        model = head_cls(model, model_cfg)

    if config.initialize_from:
        from mimir.checkpoints.importer import initialize_weights
        initialize_weights(model, config.initialize_from, upcycle=config.arch.upcycle_from_dense)

    checkpointed_blocks: set[nn.Module] = set()
    if config.activation_checkpointing != "none":
        checkpointed_blocks = apply_activation_checkpointing(model, config.activation_checkpointing)
        if not checkpointed_blocks:
            raise ValueError(
                f"activation_checkpointing={config.activation_checkpointing} found no matching TransformerBlock modules"
            )

    if dist.is_available() and dist.is_initialized():
        if config.distributed_strategy == "fsdp":
            fsdp_mesh = create_fsdp_mesh(config)
            model_dtype = fsdp_model_dtype(config)
            if model_dtype != torch.float32:
                model = model.to(dtype=model_dtype)

            # Broadcast buffers
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

            for module in fsdp_wrap_modules(model, config.fsdp_wrap_policy):
                apply_fsdp(
                    module,
                    fwd_bwd_dtype,
                    mesh=fsdp_mesh,
                    reshard_after_forward=fsdp_reshard_after_forward(
                        config,
                        checkpointed=contains_checkpointed_block(
                            module, checkpointed_blocks
                        ),
                    ),
                )

            apply_fsdp(
                model,
                fwd_bwd_dtype,
                mesh=fsdp_mesh,
                reshard_after_forward=fsdp_reshard_after_forward(
                    config,
                    checkpointed=bool(checkpointed_blocks),
                ),
            )
        elif config.distributed_strategy == "ddp":
            if device.type != "cuda":
                raise RuntimeError("distributed_strategy=ddp is currently only supported for CUDA torchrun jobs")
            model_dtype = ddp_model_dtype(config, fwd_bwd_dtype)
            if model_dtype != torch.float32:
                model = model.to(dtype=model_dtype)
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=config.ddp_find_unused_parameters,
            )
        elif config.distributed_strategy == "none":
            pass
        else:
            raise ValueError(f"Unsupported distributed_strategy: {config.distributed_strategy}")
    elif config.distributed_strategy not in ("fsdp", "none"):
        raise RuntimeError(f"distributed_strategy={config.distributed_strategy} requires torchrun/distributed training")

    # ----Create optimizer----
    optimizer_cls, optimizer_params, optimizer_options = AdamATan2, model.parameters(), {}
    if config.optimizer == 'muon':
        from mimir.model.muon import MuonAdamW
        optimizer_cls, optimizer_params = MuonAdamW, model.named_parameters()
        optimizer_options = dict(momentum=config.muon_momentum, ns_steps=config.muon_ns_steps,
                                 muon_lr_scale=config.muon_lr_scale)
    optim = optimizer_cls(optimizer_params, **optimizer_options,
                      lr=0.0,
                      betas=(config.beta1, config.beta2),
                      weight_decay=config.weight_decay,
                      ema=config.ema,
                      ema_dtype=optimizer_ema_dtype(config),
                      parameter_lr_scales=module_lr_scales(model, config.lr, configured_module_rates(config)))

    return model, carry, optim



def init_train(config: PretrainConfig, rank: int, world_size: int, device: Optional[torch.device] = None):
    set_accelerator_type(config.accelerator_type)
    if device is None:
        device = torch_device_for_accelerator(config.accelerator_type, local_rank=rank)
    physical_batch_divisor = world_size * config.gradient_accumulation_steps
    assert config.global_batch_size % physical_batch_divisor == 0, (
        f"Global batch size {config.global_batch_size} must be divisible by "
        f"world_size * gradient_accumulation_steps ({world_size} * {config.gradient_accumulation_steps})."
    )
    local_batch_size = config.global_batch_size // physical_batch_divisor

    # Dataset
    train_loader, train_metadata = create_dataloader(config, local_batch_size, drop_last_batch=True,  rank=rank, world_size=world_size)

    # Model
    model, carry, optim = create_model_and_carry(config, train_metadata, local_batch_size, device)
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)

    # Train state
    # Estimated optimizer steps. Each epoch is iterated separately and drops its
    # own incomplete final effective batch. A global override is needed when a
    # run resumes at a nonzero step on a different dataset.
    estimated_total_steps = config.epochs * int(train_metadata.total_length // config.global_batch_size)
    total_steps = config.training_total_steps or estimated_total_steps
    train_state = TrainState(
        model=model,
        carry=carry,
        optim=optim,
        
        step=0,
        total_steps=total_steps,
        fwd_bwd_dtype=fwd_bwd_dtype,
        use_cuda_autocast=(
            config.distributed_strategy == "ddp"
            and device.type == "cuda"
            and fwd_bwd_dtype != torch.float32
        ),
    )
    return train_state, train_loader, train_metadata

