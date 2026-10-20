from __future__ import annotations
from typing import Literal, Optional
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch import Tensor, nn
from mimir.model.common import IGNORE_LABEL_ID, prepare_prefixlm_batch, wrap_tensor
from mimir.model.gradient_clipping import clip_grad_norm_mean_units
from mimir.training.config import PretrainConfig
from mimir.training.state import TrainState


def forward_backward_batch_uncompiled(train_state: TrainState, batch: dict[str, Tensor], loss_scale: Tensor, **kwargs):
    device_type = batch["inputs"].device.type
    use_autocast = (
        (device_type in ("mps", "cpu") and train_state.fwd_bwd_dtype != torch.float32)
        or (device_type == "cuda" and train_state.use_cuda_autocast)
    )
    with torch.autocast(device_type=device_type, dtype=train_state.fwd_bwd_dtype, enabled=use_autocast, cache_enabled=False):
        train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
    (loss * loss_scale).backward()
    return metrics



def compiled_forward_backward_batch(mode: str):
    try:
        return _COMPILED_FORWARD_BACKWARD[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported compile_train_batch_mode: {mode}") from exc



def train_batch(train_state: TrainState, batch: dict[str, Tensor], **kwargs):
    metrics = forward_backward_batch(train_state, batch, torch.tensor(1.0, device=batch["inputs"].device), **kwargs)
    train_state.optim.step()
    train_state.optim.zero_grad()
    return metrics



def train_batch_uncompiled(train_state: TrainState, batch: dict[str, Tensor], **kwargs):
    metrics = forward_backward_batch_uncompiled(train_state, batch, torch.tensor(1.0, device=batch["inputs"].device), **kwargs)
    train_state.optim.step()
    train_state.optim.zero_grad()
    return metrics



def _add_metrics(total_metrics: Optional[dict[str, tuple[Tensor, Tensor]]], metrics: dict[str, tuple[Tensor, Tensor]]) -> dict[str, tuple[Tensor, Tensor]]:
    if total_metrics is None:
        return {name: (value_sum.detach(), divisor.detach()) for name, (value_sum, divisor) in metrics.items()}

    for name, (value_sum, divisor) in metrics.items():
        total_sum, total_divisor = total_metrics[name]
        total_metrics[name] = (total_sum + value_sum.detach(), total_divisor + divisor.detach())
    return total_metrics



def _supervised_token_count(config: PretrainConfig, rank: int, batch: dict[str, Tensor]) -> Tensor:
    from mimir.training.runtime import trace_print
    count = (batch["labels"] != IGNORE_LABEL_ID).sum().to(torch.float32)
    if dist.is_available() and dist.is_initialized():
        if config.resume_trace:
            trace_print(config, rank, f"supervised_count_all_reduce_begin local={count.item()}")
        dist.all_reduce(count, op=dist.ReduceOp.AVG)
        if config.resume_trace:
            trace_print(config, rank, f"supervised_count_all_reduce_end avg={count.item()}")
    return count



def train_accumulated_batches(
    config: PretrainConfig,
    rank: int,
    train_state: TrainState,
    batches: list[dict[str, Tensor]],
    use_compiled: bool,
    zero_grad_after_step: bool = True,
    optimizer_step_enabled: bool = True,
    **kwargs,
) -> tuple[dict[str, tuple[Tensor, Tensor]], bool]:
    from mimir.training.runtime import trace_print
    from mimir.training.distributed import unwrap_model
    trace_print(config, rank, f"train_accumulated_begin step={train_state.step} microbatches={len(batches)} compiled={use_compiled}")
    if config.gradient_skip_norm is not None and train_state.carry is not None:
        raise RuntimeError(
            "gradient_skip_norm currently supports only models with no persistent carry; "
            "a skipped stateful batch needs explicit carry rollback semantics"
        )
    supervised_counts = [_supervised_token_count(config, rank, batch) for batch in batches]
    total_supervised = torch.stack(supervised_counts).sum().clamp_min(1.0)
    backward_step = (
        compiled_forward_backward_batch(config.compile_train_batch_mode)
        if use_compiled
        else forward_backward_batch_uncompiled
    )

    trace_print(config, rank, f"zero_grad_begin step={train_state.step}")
    train_state.optim.zero_grad()
    trace_print(config, rank, f"zero_grad_end step={train_state.step}")
    metrics = None
    use_hsdp_reduce_scatter = (
        isinstance(train_state.model, FSDPModule)
        and config.fsdp_accumulation_sync_mode == "reduce_scatter"
        and config.fsdp_shard_degree is not None
        and dist.is_available()
        and dist.is_initialized()
        and config.fsdp_shard_degree < dist.get_world_size()
    )
    try:
        for microbatch_idx, (batch, supervised_count) in enumerate(zip(batches, supervised_counts), start=1):
            is_final_microbatch = microbatch_idx == len(batches)
            sync_context = nullcontext()
            if isinstance(train_state.model, DistributedDataParallel) and not is_final_microbatch:
                sync_context = train_state.model.no_sync()
            elif isinstance(train_state.model, FSDPModule):
                if use_hsdp_reduce_scatter:
                    train_state.model.set_requires_gradient_sync(True, recurse=True)
                    train_state.model.set_requires_all_reduce(is_final_microbatch, recurse=True)
                else:
                    train_state.model.set_requires_gradient_sync(is_final_microbatch, recurse=True)

            loss_scale = supervised_count / total_supervised
            if config.resume_trace:
                trace_print(
                    config,
                    rank,
                    f"forward_backward_begin step={train_state.step} microbatch={microbatch_idx} loss_scale={loss_scale.item()}",
                )
            with sync_context:
                metrics = _add_metrics(metrics, backward_step(train_state, batch, loss_scale, **kwargs))
            trace_print(config, rank, f"forward_backward_end step={train_state.step} microbatch={microbatch_idx}")
    finally:
        if isinstance(train_state.model, FSDPModule):
            train_state.model.set_requires_gradient_sync(True, recurse=True)
            train_state.model.set_requires_all_reduce(True, recurse=True)

    stability_diagnostics = kwargs.get("stability_diagnostics")
    if stability_diagnostics is not None:
        stability_diagnostics.record_model_state(
            unwrap_model(train_state.model), train_state.optim
        )
        stability_diagnostics.finalize(
            detailed_wandb=config.stability_diagnostics_wandb_detailed
        )

    if not optimizer_step_enabled:
        if zero_grad_after_step:
            train_state.optim.zero_grad()
        assert metrics is not None
        return metrics, False

    optimizer_step_skipped = False
    if config.gradient_clip_norm is not None or config.gradient_skip_norm is not None:
        summed_gradient_scale = 1.0
        if isinstance(train_state.model, FSDPModule) and dist.is_available() and dist.is_initialized():
            # This FSDP2 path intentionally disables gradient division because
            # Adam is scale-invariant. Keep the public clipping threshold in
            # mean-gradient units so it remains comparable to DDP/non-sharded
            # training and independent of world size.
            summed_gradient_scale = float(dist.get_world_size())
        clip_metrics = clip_grad_norm_mean_units(
            train_state.model.parameters(),
            max_norm=(
                config.gradient_clip_norm
                if config.gradient_clip_norm is not None
                else config.gradient_skip_norm
            ),
            summed_gradient_scale=summed_gradient_scale,
        )
        assert metrics is not None
        one = clip_metrics.total_norm.new_ones(())
        metrics["grad_norm"] = (clip_metrics.total_norm.detach(), one)
        if config.gradient_clip_norm is not None:
            metrics.update({
                "grad_clip_coefficient": (clip_metrics.clip_coefficient.detach(), one),
                "grad_clipped": (clip_metrics.clipped.detach(), one),
            })
        if config.gradient_skip_norm is not None:
            skip_tensor = (
                clip_metrics.total_norm > clip_metrics.total_norm.new_tensor(config.gradient_skip_norm)
            ).to(torch.int32)
            if dist.is_available() and dist.is_initialized():
                # The norm should already agree, but MAX guarantees identical
                # control flow at a threshold despite rank-local roundoff.
                dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
            optimizer_step_skipped = bool(skip_tensor.item())
            metrics["optimizer_step_skipped"] = (
                skip_tensor.to(dtype=clip_metrics.total_norm.dtype),
                one,
            )

    update_probe = None
    if config.experiment_update_probe_interval and train_state.step % config.experiment_update_probe_interval == 0:
        if config.experiment_metrics_output is None:
            raise ValueError("Update probes require experiment_metrics_output")
        from mimir.model.experiment_diagnostics import ParameterUpdateProbe
        update_probe = ParameterUpdateProbe(train_state.model)
    if optimizer_step_skipped:
        trace_print(config, rank, f"optim_step_skipped step={train_state.step}")
    else:
        trace_print(config, rank, f"optim_step_begin step={train_state.step}")
        train_state.optim.step()
        trace_print(config, rank, f"optim_step_end step={train_state.step}")
    if update_probe is not None:
        from mimir.model.experiment_diagnostics import append_jsonl
        updates = update_probe.finish(next(train_state.model.parameters()).device)
        if rank == 0:
            append_jsonl(str(config.experiment_metrics_output) + ".updates.jsonl", {
                "step": train_state.step,
                "optimizer_step_skipped": optimizer_step_skipped,
                "includes_weight_decay": True,
                "parameters": updates,
            })
    if zero_grad_after_step:
        trace_print(config, rank, f"zero_grad_after_step_begin step={train_state.step}")
        train_state.optim.zero_grad()
        trace_print(config, rank, f"zero_grad_after_step_end step={train_state.step}")
    assert metrics is not None
    trace_print(
        config,
        rank,
        f"train_accumulated_end step={train_state.step} optimizer_step_skipped={optimizer_step_skipped}",
    )
    return metrics, optimizer_step_skipped



@torch.inference_mode()
def validate_batches(
    config: PretrainConfig,
    rank: int,
    train_state: TrainState,
    val_iter,
    num_batches: int,
    device: torch.device,
    local_batch_size: int,
    **kwargs,
) -> Optional[dict[str, tuple[Tensor, Tensor]]]:
    from mimir.training.setup import initial_model_carry
    from mimir.training.runtime import move_batch_to_device
    if num_batches <= 0:
        return None

    was_training = train_state.model.training
    train_state.model.eval()
    old_carry = train_state.carry
    with torch.device(device):
        train_state.carry = initial_model_carry(train_state.model, local_batch_size, dtype=train_state.fwd_bwd_dtype)

    metrics = None
    for _ in range(num_batches):
        try:
            batch, batch_info = next(val_iter)
        except StopIteration:
            break
        batch = move_batch_to_device(batch, device)
        batch_info.pop("resume_info", None)
        batch = prepare_prefixlm_batch(
            batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in batch_info.items()}
        )
        device_type = batch["inputs"].device.type
        use_autocast = (
            (device_type in ("mps", "cpu") and train_state.fwd_bwd_dtype != torch.float32)
            or (device_type == "cuda" and train_state.use_cuda_autocast)
        )
        with torch.autocast(device_type=device_type, dtype=train_state.fwd_bwd_dtype, enabled=use_autocast, cache_enabled=False):
            train_state.carry, _, batch_metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
        metrics = _add_metrics(metrics, batch_metrics)

    train_state.carry = old_carry
    if was_training:
        train_state.model.train()
    return metrics



@torch.inference_mode()
def reduce_metrics(local_metrics: dict[str, Tensor], prefix: str):
    metric_keys = list(sorted(local_metrics.keys()))  # Sort keys to guarantee all processes use the same order.
    # Reduce and reconstruct
    metric_values = torch.stack([local_metrics[k][0] for k in metric_keys] + [local_metrics[k][1] for k in metric_keys])
    if dist.is_available() and dist.is_initialized():
        dist.reduce(metric_values, dst=0)
    # Split and normalize
    metrics, metrics_div = metric_values.chunk(2, dim=-1)
    metrics = (metrics / metrics_div).cpu().numpy().tolist()
    return {prefix + name: metrics[idx] for idx, name in enumerate(metric_keys)}


forward_backward_batch = torch.compile(forward_backward_batch_uncompiled, dynamic=False)
_COMPILED_FORWARD_BACKWARD = {"default": forward_backward_batch, "max-autotune-no-cudagraphs": torch.compile(forward_backward_batch_uncompiled, dynamic=False, mode="max-autotune-no-cudagraphs")}
