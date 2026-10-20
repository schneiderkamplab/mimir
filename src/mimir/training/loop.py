from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
from pathlib import Path
import json
import os
import time
import torch
import torch.distributed as dist
from torch import Tensor, nn
import tqdm
from mimir.training.tracking import wandb
from omegaconf import DictConfig, OmegaConf
from mimir.model.common import IGNORE_LABEL_ID, prepare_prefixlm_batch, wrap_tensor
from mimir.model.stability_diagnostics import StabilityDiagnostics, diagnostic_prefixlm_batch
from mimir.model.accelerator import (
    AcceleratorType,
    empty_accelerator_cache,
    memory_stats_for_device,
    set_accelerator_type,
    synchronize_device,
    torch_device_for_accelerator,
)
from mimir.model.module_learning_rates import configured_module_rates, module_lr_scales, module_lr_metrics, update_auto_module_rates, windowed_cosine_lr
from mimir.model.lr_rewarm import resolve_rewarm, rewarm_lr, rewarm_metadata
from mimir.training.tracking import TrainingWandbLogger


def launch(hydra_config: DictConfig):
    from mimir.training.setup import compute_train_extra_args
    from mimir.training.setup import create_dataloader
    from mimir.training.runtime import distributed_state_fingerprint
    from mimir.training.setup import init_train
    from mimir.training.checkpoint_io import load_checkpoint_metadata
    from mimir.training.runtime import load_synced_config
    from mimir.training.checkpoint_io import load_train_checkpoint
    from mimir.training.runtime import maybe_empty_cache
    from mimir.training.runtime import maybe_log_memory
    from mimir.training.runtime import move_batch_to_device
    from mimir.training.steps import reduce_metrics
    from mimir.training.checkpoint_io import remove_stale_ephemeral_checkpoints
    from mimir.training.checkpoint_io import save_code_and_config
    from mimir.training.checkpoint_io import save_train_checkpoint
    from mimir.training.runtime import trace_print
    from mimir.training.steps import train_accumulated_batches
    from mimir.training.distributed import unwrap_model
    from mimir.training.schedules import update_lr
    from mimir.training.steps import validate_batches
    WORLD_SIZE = 1
    RANK = 0
    DEVICE_ID = 0
    requested_accelerator = OmegaConf.select(hydra_config, "accelerator_type", default="sm100")

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        if requested_accelerator in ("mps", "cpu", "none"):
            raise RuntimeError(f"accelerator_type={requested_accelerator} supports single-process training only.")
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        DEVICE_ID = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(DEVICE_ID)

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK)
    if config.resume_checkpoint_path:
        from mimir.checkpoints.contract import resolve
        config.resume_checkpoint_path = str(resolve(config.resume_checkpoint_path, config.resume_checkpoint_tag))
    Path(config.checkpoint_path).mkdir(parents=True, exist_ok=True)
    wandb.path = str(Path(config.checkpoint_path) / "metrics.jsonl")
    set_accelerator_type(config.accelerator_type)
    device = torch_device_for_accelerator(config.accelerator_type, local_rank=DEVICE_ID)
    suspend_hook = None
    if config.suspend_control_dir:
        if config.suspension_backend == 'cuda':
            from mimir.training.suspension.runtime import SuspendHook
            suspend_hook = SuspendHook(config.suspend_control_dir, config.suspend_after_steps,
                                      device, config.suspend_poll_interval_steps,
                                      config.suspend_wait_poll_seconds)
        else:
            if WORLD_SIZE != 1:
                raise ValueError('Distributed disk suspension requires a whole-launcher controller')
            from mimir.training.disk_recovery import DiskHook
            suspend_hook = DiskHook(config.suspend_control_dir, config.suspend_after_steps,
                                    config.suspend_poll_interval_steps)

    # Seed RNGs to ensure consistency
    import random
    import numpy as np
    random.seed(config.seed + RANK)
    np.random.seed((config.seed + RANK) % (2**32))
    torch.random.manual_seed(config.seed + RANK)

    # --- Training
    local_batch_size = config.global_batch_size // (WORLD_SIZE * config.gradient_accumulation_steps)
    trace_print(config, RANK, "init_train_begin")
    train_state, train_loader, train_metadata = init_train(config, rank=RANK, world_size=WORLD_SIZE, device=device)
    trace_print(config, RANK, "init_train_end")
    val_loader = None
    val_iter = None
    if config.data.validation_path is not None and config.validation_interval > 0 and config.validation_batches > 0:
        val_loader, _ = create_dataloader(
            config,
            local_batch_size,
            drop_last_batch=False,
            rank=RANK,
            world_size=WORLD_SIZE,
            dataset_path=config.data.validation_path,
        )
        val_iter = iter(val_loader)
    resume_state = load_train_checkpoint(config, train_state, rank=RANK, local_batch_size=local_batch_size)
    resolve_rewarm(config, train_state.step if resume_state is not None else None,
                   load_checkpoint_metadata(config.resume_checkpoint_path, resume_state.tag)
                   if resume_state is not None and config.lr_rewarm_steps else {})
    lr_cooldown = None
    if config.lr_cooldown_checkpoint is not None:
        from mimir.model.epoch_lr_cooldown import EpochLRCooldown
        lr_cooldown = EpochLRCooldown(config.lr_cooldown_checkpoint, config.data.path,
                                     train_state.step, config.lr_min_ratio)
    if train_state.step > train_state.total_steps:
        raise ValueError(
            f"Loaded checkpoint step {train_state.step} exceeds training_total_steps "
            f"{train_state.total_steps}"
        )
    start_epoch = 1
    skip_batches = 0
    batch_in_epoch_exact = True
    if resume_state is not None:
        start_epoch = resume_state.start_epoch
        skip_batches = resume_state.skip_batches
        batch_in_epoch_exact = resume_state.resume_mode == "batch"
        trace_print(config, RANK, f"set_epoch_begin epoch_index={start_epoch - 1}")
        train_loader.dataset.set_epoch(start_epoch - 1)
        trace_print(config, RANK, f"set_epoch_end epoch_index={start_epoch - 1}")
        if resume_state.start_row_cursor is not None:
            trace_print(config, RANK, f"set_start_row_cursor_begin row_cursor={resume_state.start_row_cursor}")
            train_loader.dataset.set_start_row_cursor(resume_state.start_row_cursor)
            trace_print(config, RANK, f"set_start_row_cursor_end row_cursor={resume_state.start_row_cursor}")
        else:
            trace_print(config, RANK, f"set_start_batch_begin skip_batches={skip_batches}")
            train_loader.dataset.set_start_batch(skip_batches)
            trace_print(config, RANK, f"set_start_batch_end skip_batches={skip_batches}")
        if RANK == 0:
            print(
                f"Resumed from {config.resume_checkpoint_path} ({config.checkpoint_format}:{resume_state.tag}): "
                f"step={train_state.step}, start_epoch={start_epoch}, skip_batches={skip_batches}, "
                f"resume_mode={resume_state.resume_mode}, row_cursor={resume_state.start_row_cursor}",
                flush=True,
            )

    session_start_step = train_state.step
    session_start_time = time.perf_counter()

    # Optional per-step timing for benchmarking (only active with max_steps).
    bench_step_times: list[float] = []
    bench_last_time: Optional[float] = None
    bench_last_metrics: dict[str, float] = {}
    bench_metric_history: list[dict[str, float | int]] = []

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)

        wandb.init(
            project=config.project_name,
            name=config.run_name,
            id=config.wandb_run_id,
            resume=config.wandb_resume,
            config=config.model_dump() | {"train_metadata": train_metadata.model_dump()},
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore
        training_wandb = TrainingWandbLogger(wandb)
        num_params = sum(x.numel() for x in train_state.model.parameters())
        if resume_state is None:
            training_wandb.log({"num_params": num_params}, train_state.step)
        else:
            wandb.run.summary["num_params"] = num_params  # type: ignore[union-attr]
        save_code_and_config(config, train_metadata)

    # Training Loop
    stop_after_step_reached = (
        config.stop_after_step is not None
        and train_state.step >= config.stop_after_step
    )
    if stop_after_step_reached and RANK == 0:
        print(
            f"stop_after_step={config.stop_after_step} already reached at "
            f"step={train_state.step}; exiting without another optimizer step",
            flush=True,
        )
    consecutive_gradient_skips = 0
    gradient_skip_stop_reached = False
    for epoch in range(start_epoch, config.epochs + 1):
        if stop_after_step_reached or gradient_skip_stop_reached:
            break
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch}")
        trace_print(config, RANK, f"epoch_begin epoch={epoch}")

        # ############ Train Iter
        train_state.model.train()
        accumulation_batches: list[dict[str, Tensor]] = []
        accumulation_resume_info: Optional[dict[str, int]] = None
        batch_start = skip_batches + 1 if skip_batches > 0 else 1
        trace_print(config, RANK, f"dataloader_iter_begin epoch={epoch} batch_start={batch_start}")
        for batch_in_epoch, (batch, batch_info) in enumerate(train_loader, start=batch_start):
            if config.resume_trace and batch_in_epoch == batch_start:
                trace_print(config, RANK, f"first_batch_yielded batch_in_epoch={batch_in_epoch}")
            batch = move_batch_to_device(batch, device)
            if config.resume_trace and batch_in_epoch == batch_start:
                trace_print(config, RANK, f"first_batch_moved batch_in_epoch={batch_in_epoch}")
            resume_info = batch_info.pop("resume_info", None)
            accumulation_batches.append(
                prepare_prefixlm_batch(batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in batch_info.items()})
            )
            if resume_info is not None:
                accumulation_resume_info = resume_info
            if len(accumulation_batches) < config.gradient_accumulation_steps:
                continue

            train_state.step += 1            
            cooldown_ratio = (lr_cooldown.ratio(epoch, accumulation_resume_info)
                              if lr_cooldown is not None else None)
            lr = update_lr(config, train_state, cooldown_ratio)
            trace_print(config, RANK, f"optimizer_step_start step={train_state.step} batch_in_epoch={batch_in_epoch} lr={lr}")
            # Extra train arguments (such as BP warmup etc.)
            train_extra_args = compute_train_extra_args(train_state.model, train_state)
            update_auto_module_rates(config, train_state.model, train_state.optim, train_extra_args.get('bp_steps'))
            backbone = getattr(unwrap_model(train_state.model), "model", None)
            if hasattr(backbone, "configure_gate_updates"):
                backbone.configure_gate_updates(train_state.optim, train_extra_args['bp_steps'])
            trace_print(config, RANK, f"train_extra_args step={train_state.step} {train_extra_args}")
            stability_diagnostics = None
            if (
                config.stability_diagnostics_interval > 0
                and train_state.step % config.stability_diagnostics_interval == 0
            ):
                diagnostics_output = config.stability_diagnostics_output or os.path.join(
                    config.checkpoint_path or "logs", "stability_diagnostics.jsonl"
                )
                stability_diagnostics = StabilityDiagnostics(
                    step=train_state.step,
                    rank=RANK,
                    output_path=diagnostics_output,
                    context={
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "bp_steps": train_extra_args.get("bp_steps"),
                        "lr": lr,
                        "global_batch_size": config.global_batch_size,
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        "distributed_strategy": config.distributed_strategy,
                        "fwd_bwd_dtype": config.fwd_bwd_dtype,
                        "gradient_clip_norm": config.gradient_clip_norm,
                        "gradient_skip_norm": config.gradient_skip_norm,
                    },
                )
            train_call_args = dict(train_extra_args)
            if config.experiment_metrics_output is not None:
                synchronize_device(device)
                experiment_start = time.perf_counter()
            maybe_log_memory(
                train_state.step,
                "before_train",
                device,
                config.memory_log_interval > 0 and train_state.step % config.memory_log_interval == 0,
            )
            if stability_diagnostics is not None:
                if train_state.carry is not None:
                    raise RuntimeError(
                        "stability diagnostics shadow replay currently requires a stateless carry"
                    )
                rng_devices = (
                    [device.index] if device.type == "cuda" and device.index is not None else []
                )
                with torch.random.fork_rng(devices=rng_devices):
                    diagnostic_batches = [
                        diagnostic_prefixlm_batch(
                            batch,
                            max_tokens=config.stability_diagnostics_max_tokens_per_microbatch,
                        )
                        for batch in accumulation_batches
                    ]
                    train_accumulated_batches(
                        config,
                        RANK,
                        train_state,
                        diagnostic_batches,
                        use_compiled=False,
                        optimizer_step_enabled=False,
                        **train_extra_args,
                        stability_diagnostics=stability_diagnostics,
                    )
            if config.experiment_metrics_output is not None:
                synchronize_device(device)
                experiment_train_start = time.perf_counter()
            metrics, optimizer_step_skipped = train_accumulated_batches(
                config,
                RANK,
                train_state,
                accumulation_batches,
                config.compile_train_batch,
                **train_call_args,
            )
            accumulation_batches = []
            if config.experiment_metrics_output is not None:
                synchronize_device(device)
                experiment_train_end = time.perf_counter()
            if stability_diagnostics is not None and RANK == 0:
                training_wandb.log(
                    stability_diagnostics.wandb_summary
                    | stability_diagnostics.wandb_detailed,
                    optimizer_step=train_state.step,
                )
            if config.gradient_skip_norm is not None:
                if optimizer_step_skipped:
                    consecutive_gradient_skips += 1
                else:
                    consecutive_gradient_skips = 0
                metric_reference = next(iter(metrics.values()))[0]
                metric_one = metric_reference.new_ones(())
                metrics["consecutive_gradient_skips"] = (
                    metric_reference.new_tensor(float(consecutive_gradient_skips)),
                    metric_one,
                )
                gradient_skip_stop_reached = (
                    optimizer_step_skipped
                    and consecutive_gradient_skips >= config.gradient_skip_max_consecutive
                )
                if optimizer_step_skipped and RANK == 0:
                    skipped_grad_norm = float(metrics["grad_norm"][0].item())
                    print(
                        f"Skipped optimizer step {train_state.step}: gradient norm "
                        f"{skipped_grad_norm:.6g} exceeded "
                        f"gradient_skip_norm={config.gradient_skip_norm}; "
                        f"consecutive={consecutive_gradient_skips}/"
                        f"{config.gradient_skip_max_consecutive}",
                        flush=True,
                    )
                if gradient_skip_stop_reached and RANK == 0:
                    print(
                        "Maximum consecutive gradient skips reached; saving a regular "
                        "checkpoint and exiting cleanly",
                        flush=True,
                    )

            if config.max_steps is not None and RANK == 0:
                synchronize_device(device)
                now = time.perf_counter()
                if bench_last_time is not None:
                    bench_step_times.append(now - bench_last_time)
                bench_last_time = now
            maybe_log_memory(
                train_state.step,
                "after_train",
                device,
                config.memory_log_interval > 0 and train_state.step % config.memory_log_interval == 0,
            )
            maybe_empty_cache(train_state.step, device, config.empty_cache_interval)

            if train_state.step % config.log_interval == 0 or optimizer_step_skipped or config.experiment_metrics_output is not None:
                trace_print(config, RANK, f"reduce_metrics_begin step={train_state.step}")
                metrics = reduce_metrics(metrics, prefix="train/")
                trace_print(config, RANK, f"reduce_metrics_end step={train_state.step}")
                if RANK == 0:
                    metrics.update(module_lr_metrics(config, lr, train_extra_args.get('bp_steps')))
                    if hasattr(backbone, "gate_alpha"):
                        metrics["train/gate_alpha"] = float(backbone.gate_alpha.item())
                        metrics["train/lr_gates"] = lr
                    if config.experiment_metrics_output is not None:
                        from mimir.model.experiment_diagnostics import append_jsonl
                        append_jsonl(config.experiment_metrics_output, {
                            "step": train_state.step, "epoch": epoch,
                            "batch_in_epoch": batch_in_epoch, "train/lr": lr,
                            **train_extra_args, **metrics,
                            "training_seconds": experiment_train_end - experiment_train_start,
                            "probe_seconds": experiment_train_start - experiment_start,
                        })
                    bench_last_metrics = dict(metrics)
                    if config.max_steps is not None:
                        bench_metric_history.append({"step": train_state.step, **metrics})
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    trace_print(config, RANK, f"wandb_log_begin step={train_state.step}")
                    training_wandb.log(metrics | train_extra_args | {"train/lr": lr}, train_state.step)
                    from mimir.evaluation.metrics import write_record
                    write_record(Path(config.checkpoint_path) / 'progress.json', {
                        'completed': train_state.step-session_start_step,
                        'total': train_state.total_steps-session_start_step,
                        'optimizer_step': train_state.step, 'epoch': epoch,
                        'elapsed_seconds': time.perf_counter()-session_start_time,
                    })
                    trace_print(config, RANK, f"wandb_log_end step={train_state.step}")

            if (
                val_loader is not None
                and val_iter is not None
                and train_state.step % config.validation_interval == 0
            ):
                val_metrics = validate_batches(
                    config,
                    RANK,
                    train_state,
                    val_iter,
                    config.validation_batches,
                    device,
                    local_batch_size,
                    **train_extra_args,
                )
                if val_metrics is None:
                    val_iter = iter(val_loader)
                    val_metrics = validate_batches(
                        config,
                        RANK,
                        train_state,
                        val_iter,
                        config.validation_batches,
                        device,
                        local_batch_size,
                        **train_extra_args,
                    )
                if val_metrics is not None:
                    val_metrics = reduce_metrics(val_metrics, prefix="val/")
                    if RANK == 0:
                        training_wandb.log(val_metrics, train_state.step)

            del metrics

            stop_after_step_reached = (
                config.stop_after_step is not None
                and train_state.step >= config.stop_after_step
            )
            saved_regular_step_checkpoint = False
            regular_interval_reached = (
                config.checkpoint_step_interval is not None
                and train_state.step % config.checkpoint_step_interval == 0
            )
            if regular_interval_reached or stop_after_step_reached or gradient_skip_stop_reached:
                save_train_checkpoint(config, train_state, f"step_{train_state.step}", epoch, batch_in_epoch, RANK, local_batch_size=local_batch_size, resume_info=accumulation_resume_info, batch_in_epoch_exact=batch_in_epoch_exact)
                saved_regular_step_checkpoint = True

            if config.ephemeral_checkpoint_step_interval is not None and train_state.step % config.ephemeral_checkpoint_step_interval == 0:
                if saved_regular_step_checkpoint:
                    remove_stale_ephemeral_checkpoints(config, f"step_{train_state.step}", RANK)
                else:
                    ephemeral_tag = f"ephemeral_step_{train_state.step}"
                    save_train_checkpoint(config, train_state, ephemeral_tag, epoch, batch_in_epoch, RANK, checkpoint_kind="ephemeral", local_batch_size=local_batch_size, resume_info=accumulation_resume_info, batch_in_epoch_exact=batch_in_epoch_exact)
                    remove_stale_ephemeral_checkpoints(config, ephemeral_tag, RANK)

            if suspend_hook is not None:
                def save_suspend_checkpoint():
                    if not saved_regular_step_checkpoint:
                        save_train_checkpoint(config, train_state, f"step_{train_state.step}", epoch, batch_in_epoch, RANK, local_batch_size=local_batch_size, resume_info=accumulation_resume_info, batch_in_epoch_exact=batch_in_epoch_exact)
                    return str(Path(config.checkpoint_path) / f"step_{train_state.step}" / f"checkpoint_state_step_{train_state.step}.json")

                suspend_hook.after_step(
                    train_state.step, save_suspend_checkpoint,
                    stopping=stop_after_step_reached or gradient_skip_stop_reached or (
                        config.max_steps is not None and train_state.step >= config.max_steps
                    ),
                )

            if stop_after_step_reached or gradient_skip_stop_reached or (
                config.max_steps is not None and train_state.step >= config.max_steps
            ):
                break

        skip_batches = 0

        if stop_after_step_reached or gradient_skip_stop_reached or (
            config.max_steps is not None and train_state.step >= config.max_steps
        ):
            break

        ############ EVAL STACK: TBD TODO

        ############ Checkpointing
        if (epoch % config.checkpoint_interval == 0) or (epoch == config.epochs):
            save_train_checkpoint(config, train_state, f"epoch_{epoch}", epoch, 0, RANK, local_batch_size=local_batch_size, batch_in_epoch_exact=True)

    benchmark_fingerprint = None
    if config.max_steps is not None and config.benchmark_state_fingerprint:
        benchmark_fingerprint = distributed_state_fingerprint(config, train_state, device)

    # Benchmark summary (rank 0 only, when max_steps was set).
    if config.max_steps is not None and RANK == 0 and bench_step_times:
        # Drop the first couple of measured intervals as warmup (compile/MIOPEN tuning).
        warmup = min(2, len(bench_step_times) - 1) if len(bench_step_times) > 1 else 0
        steady = bench_step_times[warmup:]
        sorted_steady = sorted(steady)
        median = sorted_steady[len(sorted_steady) // 2]
        summary = {
            "world_size": WORLD_SIZE,
            "nnodes": int(os.environ.get("NNODES", os.environ.get("SLURM_JOB_NUM_NODES", 0))),
            "global_batch_size": config.global_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "measured_steps": len(bench_step_times),
            "warmup_dropped": warmup,
            "median_step_seconds": median,
            "mean_step_seconds": sum(steady) / len(steady),
            "min_step_seconds": min(steady),
            "max_step_seconds": max(steady),
            "all_step_seconds": bench_step_times,
            "last_metrics": bench_last_metrics,
            "metric_history": bench_metric_history,
        }
        if benchmark_fingerprint is not None:
            summary["state_fingerprint"] = benchmark_fingerprint
        bench_path = os.environ.get("BENCH_OUTPUT")
        if bench_path:
            with open(bench_path, "w") as f:
                json.dump(summary, f, indent=2)
        print(f"[bench] {json.dumps({k: v for k, v in summary.items() if k != 'all_step_seconds'})}", flush=True)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()

