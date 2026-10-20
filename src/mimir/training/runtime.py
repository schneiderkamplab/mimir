from __future__ import annotations
from pathlib import Path
import os
import torch
import torch.distributed as dist
from torch import Tensor, nn
import coolname
from omegaconf import DictConfig, OmegaConf
from mimir.model.accelerator import (
    AcceleratorType,
    empty_accelerator_cache,
    memory_stats_for_device,
    set_accelerator_type,
    synchronize_device,
    torch_device_for_accelerator,
)
from mimir.training.config import PretrainConfig
from mimir.training.state import TrainState


def trace_print(config: PretrainConfig, rank: int, message: str):
    if config.resume_trace:
        print(f"[resume_trace rank={rank}] {message}", flush=True)



def load_synced_config(hydra_config: DictConfig, rank: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{Path(config.data.path).stem.capitalize()} HLM-torch"
        if config.run_name is None:
            config.run_name = os.environ.get("MLP_TASK_NAME", f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}")  # pyright: ignore[reportPrivateImportUsage]
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore



def move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device, non_blocking=device.type == "cuda") for name, tensor in batch.items()}



def format_mib(value: int) -> str:
    return f"{value / 1024 / 1024:.3f} MiB"



def format_memory_stats(stats: dict[str, int]) -> str:
    return " ".join(f"{name}={format_mib(value)}" for name, value in stats.items())



def maybe_log_memory(step: int, label: str, device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    stats = memory_stats_for_device(device)
    if not stats:
        return
    print(f"[{device.type} memory] step={step} {label}: {format_memory_stats(stats)}", flush=True)



def maybe_empty_cache(step: int, device: torch.device, interval: int) -> None:
    if interval <= 0 or step % interval != 0:
        return
    before = memory_stats_for_device(device)
    empty_accelerator_cache(device)
    synchronize_device(device)
    after = memory_stats_for_device(device)
    if not before and not after:
        return
    changes = []
    for name in sorted(before.keys() | after.keys()):
        changes.append(f"{name} {format_mib(before.get(name, 0))}->{format_mib(after.get(name, 0))}")
    print(f"[{device.type} empty_cache] step={step}: {' '.join(changes)}", flush=True)



@torch.no_grad()
def distributed_state_fingerprint(
    config: PretrainConfig,
    train_state: TrainState,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Return compact distributed checksums for benchmark parity comparisons."""
    buckets: dict[str, list[Tensor]] = {"parameters": []}
    for param in train_state.model.parameters():
        buckets["parameters"].append(param)
    for state in train_state.optim.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                buckets.setdefault(f"optimizer/{key}", []).append(value)

    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    replica_factor = 1
    if config.fsdp_shard_degree is not None and dist.is_available() and dist.is_initialized():
        replica_factor = world_size // config.fsdp_shard_degree

    result: dict[str, dict[str, float]] = {}
    for name, tensors in sorted(buckets.items()):
        sums = torch.zeros(4, dtype=torch.float64, device=device)
        maximum = torch.zeros(1, dtype=torch.float64, device=device)
        for tensor in tensors:
            local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            if local.device != device:
                local = local.to(device)
            sums[0] += local.sum(dtype=torch.float64)
            sums[1] += local.abs().sum(dtype=torch.float64)
            sums[2] += (local * local).sum(dtype=torch.float64)
            sums[3] += local.numel()
            if local.numel():
                maximum.copy_(torch.maximum(maximum, local.abs().max().to(torch.float64).reshape(1)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        normalization = world_size if name == "optimizer/step" else replica_factor
        sums /= normalization
        result[name] = {
            "sum": sums[0].item(),
            "abs_sum": sums[1].item(),
            "square_sum": sums[2].item(),
            "numel": sums[3].item(),
            "abs_max": maximum.item(),
        }
    return result

