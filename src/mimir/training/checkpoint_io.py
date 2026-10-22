from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import json
import os
import re
import yaml
import shutil
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)
from mimir.training.tracking import wandb
from mimir.model.adam_atan2 import AdamATan2
from mimir.model.lr_rewarm import resolve_rewarm, rewarm_lr, rewarm_metadata
from mimir.model import Mimir2
from mimir.corpus.sampled import V1Dataset, V1DatasetConfig, V1DatasetMeta
from mimir.training.config import PretrainConfig
from mimir.training.state import TrainState
from mimir.training.state import ResumeState


def normalize_checkpoint_tag(tag: str) -> str:
    if tag.startswith("fsdp2_"):
        tag = tag.removeprefix("fsdp2_")
    elif tag.startswith("unsharded_"):
        tag = tag.removeprefix("unsharded_")
        if tag.endswith(".pt"):
            tag = tag.removesuffix(".pt")
    elif tag.startswith("carry_"):
        tag = tag.removeprefix("carry_")
    if os.path.basename(tag) != tag:
        raise ValueError(f"Checkpoint tag must be a name, not a path: {tag}")
    if not (tag.startswith("epoch_") or tag.startswith("step_") or tag.startswith("ephemeral_step_")):
        raise ValueError(f"Checkpoint tag must start with 'epoch_', 'step_', or 'ephemeral_step_': {tag}")
    return tag



def load_checkpoint_metadata(checkpoint_path: str, tag: str) -> dict:
    metadata_path = os.path.join(checkpoint_path, f"checkpoint_state_{tag}.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "rt") as f:
        return json.load(f)



def infer_checkpoint_world_size(checkpoint_path: str, tag: str, metadata: dict) -> Optional[int]:
    if metadata.get("world_size") is not None:
        return int(metadata["world_size"])
    carry_pattern = re.compile(rf"^carry_{re.escape(tag)}\.(\d+)\.pt$")
    carry_ranks = {
        int(match.group(1))
        for name in os.listdir(checkpoint_path)
        if (match := carry_pattern.fullmatch(name)) is not None
    }
    return max(carry_ranks) + 1 if carry_ranks else None



def checkpoint_carry_policy(checkpoint_path: str, tag: str, metadata: dict) -> Literal["none", "per_rank"]:
    explicit_policy = metadata.get("carry_policy")
    if explicit_policy is not None:
        if explicit_policy not in ("none", "per_rank"):
            raise ValueError(f"Unsupported carry_policy={explicit_policy!r} in checkpoint {tag}")
        return explicit_policy

    # Legacy no-carry checkpoints wrote one serialized None per rank.
    inferred_world_size = infer_checkpoint_world_size(checkpoint_path, tag, metadata)
    legacy_policy: Literal["none", "per_rank"] = "per_rank"
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        if inferred_world_size is not None:
            carry_paths = [
                os.path.join(checkpoint_path, f"carry_{tag}.{rank}.pt")
                for rank in range(inferred_world_size)
            ]
            if all(os.path.isfile(path) for path in carry_paths) and all(
                torch.load(path, map_location="cpu", weights_only=False) is None
                for path in carry_paths
            ):
                legacy_policy = "none"
    if dist.is_available() and dist.is_initialized():
        policies = [legacy_policy]
        dist.broadcast_object_list(policies, src=0)
        legacy_policy = policies[0]
    return legacy_policy



def same_architecture(previous, current):
    """Older checkpoints predate these opt-in flags; absence means disabled."""
    defaults = dict(h_residual_persistence=False, h_branch_l_mixing=False,
                    moe_aux_loss_free=False, mos_aux_loss_free=False, routing_bias_update_rate=0.001)
    return previous is not None and (defaults | previous) == (defaults | current)


def resolve_resume_state(config: PretrainConfig, current_local_batch_size: int) -> Optional[ResumeState]:
    if config.resume_checkpoint_path is None and config.resume_checkpoint_tag is None:
        return None
    if config.resume_checkpoint_path is None or config.resume_checkpoint_tag is None:
        raise ValueError("Both resume_checkpoint_path and resume_checkpoint_tag must be set to resume training")

    tag = normalize_checkpoint_tag(config.resume_checkpoint_tag)
    metadata = load_checkpoint_metadata(config.resume_checkpoint_path, tag)

    step = metadata.get("step", config.resume_step)
    epoch = metadata.get("epoch", config.resume_epoch)
    batch_in_epoch = metadata.get("batch_in_epoch", config.resume_batch_in_epoch)
    row_cursor = metadata.get("global_row_cursor_in_epoch")
    checkpoint_local_batch_size = metadata.get("local_batch_size")
    checkpoint_world_size = infer_checkpoint_world_size(
        config.resume_checkpoint_path, tag, metadata
    )
    current_world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    batch_in_epoch_exact = bool(metadata.get("batch_in_epoch_exact", True))

    if tag.startswith("epoch_"):
        tag_epoch = int(tag.removeprefix("epoch_"))
        epoch = tag_epoch if epoch is None else int(epoch)
        step = -1 if step is None else int(step)
        return ResumeState(tag=tag, step=step, start_epoch=tag_epoch + 1, skip_batches=0)

    step_match = re.fullmatch(r"(?:ephemeral_)?step_(\d+)", tag)
    if step_match is None:
        raise ValueError(f"Unsupported checkpoint tag: {tag}")
    step = int(step_match.group(1)) if step is None else int(step)
    if epoch is None or batch_in_epoch is None:
        raise ValueError(
            f"Step checkpoint {tag} needs checkpoint_state_{tag}.json or explicit "
            "resume_epoch and resume_batch_in_epoch overrides"
        )
    start_row_cursor = None
    resume_mode = "batch"
    if (
        row_cursor is not None
        and checkpoint_local_batch_size is not None
        and (
            int(checkpoint_local_batch_size) != current_local_batch_size
            or (
                checkpoint_world_size is not None
                and int(checkpoint_world_size) != current_world_size
            )
            or not batch_in_epoch_exact
        )
    ):
        start_row_cursor = int(row_cursor)
        resume_mode = "row_cursor"
    return ResumeState(
        tag=tag,
        step=step,
        start_epoch=int(epoch),
        skip_batches=int(batch_in_epoch),
        start_row_cursor=start_row_cursor,
        resume_mode=resume_mode,
    )



def load_train_checkpoint(config: PretrainConfig, train_state: TrainState, rank: int, local_batch_size: int) -> Optional[ResumeState]:
    from mimir.training.runtime import trace_print
    if config.resume_checkpoint_path is not None:
        from mimir.checkpoints.contract import resolve
        config = config.model_copy(update={"resume_checkpoint_path": str(resolve(
            config.resume_checkpoint_path, config.resume_checkpoint_tag))})
    resume_state = resolve_resume_state(config, local_batch_size)
    if resume_state is None:
        return None

    assert config.resume_checkpoint_path is not None
    metadata = load_checkpoint_metadata(config.resume_checkpoint_path, resume_state.tag)
    previous_config = os.path.join(config.resume_checkpoint_path, 'all_config.yaml')
    previous = {}
    if os.path.isfile(previous_config):
        with open(previous_config) as handle:
            previous = yaml.safe_load(handle)
    if previous:
        current = config.model_dump()
        protected = ('data', 'seed', 'global_batch_size', 'epochs', 'training_total_steps',
                     'lr', 'lr_auto', 'lr_piecewise_points', 'lr_warmup_steps', 'lr_min_ratio',
                     'lr_rewarm_steps', 'lr_rewarm_stages', 'lr_rewarm_start_ratio',
                     'lr_decay_start_step', 'lr_decay_end_step', 'lr_cooldown_checkpoint',
                     'lr_embeddings', 'lr_head', 'lr_h', 'lr_l', 'weight_decay',
                     'beta1', 'beta2', 'ema', 'optimizer', 'muon_momentum', 'muon_ns_steps',
                     'muon_lr_scale', 'fwd_bwd_dtype')
        for key in protected:
            if previous.get(key) != current.get(key):
                raise ValueError(f"Exact continuation configuration changed: {key}")
        if not config.arch.upcycle_from_dense and not config.arch.gate_initialize_from_v1:
            if not same_architecture(previous.get('arch'), current['arch']):
                raise ValueError('Exact continuation architecture changed')
        old_metadata = yaml.safe_load((Path(config.resume_checkpoint_path) / 'train_metadata.yaml').read_text())
        # Re-read the corpus metadata before loading state; the dataset reader normalizes it.
        from mimir.corpus.sampled import V1Dataset, V1DatasetConfig
        corpus = V1Dataset(V1DatasetConfig(seed=config.seed, dataset_path=config.data.path,
                          batch_max_length=local_batch_size, drop_last_batch=True,
                          target_only=config.data.target_only, rank=rank,
                          num_replicas=dist.get_world_size() if dist.is_initialized() else 1))
        if old_metadata != corpus.metadata.model_dump():
            raise ValueError('Corpus metadata changed during continuation')
    if metadata.get('optimizer', previous.get('optimizer', 'adam_atan2')) != getattr(config, 'optimizer', 'adam_atan2'):
        raise ValueError('Optimizer changes require an explicit offline state conversion')
    old_homogeneous = metadata.get('homogeneous_categories', previous.get('data', {}).get('homogeneous_categories', False))
    mode_changed = old_homogeneous != getattr(config.data, 'homogeneous_categories', False)
    layout_changed = old_homogeneous and (
        infer_checkpoint_world_size(config.resume_checkpoint_path, resume_state.tag, metadata)
        not in (None, dist.get_world_size() if dist.is_initialized() else 1)
        or metadata.get('local_batch_size', local_batch_size) != local_batch_size
        or metadata.get('seed', previous.get('seed', 0)) != config.seed)
    if not resume_state.tag.startswith('epoch_') and (mode_changed or layout_changed):
        raise ValueError('Homogeneous sampler changes require an epoch-boundary resume')
    carry_policy = checkpoint_carry_policy(
        config.resume_checkpoint_path, resume_state.tag, metadata
    )
    checkpoint_world_size = infer_checkpoint_world_size(
        config.resume_checkpoint_path, resume_state.tag, metadata
    )
    current_world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if (
        carry_policy == "per_rank"
        and checkpoint_world_size is not None
        and checkpoint_world_size != current_world_size
    ):
        raise ValueError(
            "Cannot resume stateful per-rank carry across a world-size change: "
            f"checkpoint={checkpoint_world_size}, current={current_world_size}. "
            "Sample-keyed carry redistribution is not implemented."
        )
    carry_path = os.path.join(config.resume_checkpoint_path, f"carry_{resume_state.tag}.{rank}.pt")
    if carry_policy == "per_rank" and not os.path.isfile(carry_path):
        raise ValueError(f"Carry file not found: {carry_path}")

    trace_print(config, rank, f"load_checkpoint_begin tag={resume_state.tag} format={config.checkpoint_format}")
    if config.checkpoint_format == "sharded":
        load_sharded_train_state(config, train_state, resume_state.tag)
    elif config.checkpoint_format == "unsharded":
        load_unsharded_train_state(config, train_state, resume_state.tag, rank)
    else:
        raise ValueError(f"Unsupported checkpoint_format: {config.checkpoint_format}")
    trace_print(config, rank, f"load_checkpoint_end tag={resume_state.tag}")

    if config.upcast_optimizer_state_on_resume:
        trace_print(config, rank, "upcast_optimizer_state_begin")
        upcast_optimizer_state(train_state.optim)
        trace_print(config, rank, "upcast_optimizer_state_end")
    if config.reset_ema_on_resume:
        trace_print(config, rank, "reset_ema_to_params_begin")
        reset_optimizer_ema_to_params(train_state.optim)
        trace_print(config, rank, "reset_ema_to_params_end")

    if carry_policy == "none":
        train_state.carry = None
        trace_print(config, rank, "load_carry_skipped policy=none")
    else:
        trace_print(config, rank, f"load_carry_begin path={carry_path}")
        train_state.carry = torch.load(carry_path, map_location="cuda")
        trace_print(config, rank, "load_carry_end")
    if resume_state.step >= 0:
        train_state.step = resume_state.step
    elif resume_state.tag.startswith("epoch_"):
        completed_epoch = int(resume_state.tag.removeprefix("epoch_"))
        train_state.step = int(completed_epoch * train_state.total_steps // config.epochs)
        resume_state.step = train_state.step
    else:
        raise ValueError(f"Cannot infer resume step for checkpoint tag {resume_state.tag}")
    from mimir.checkpoints.contract import restore_rng, validate
    manifest = validate(config.resume_checkpoint_path, hashes=False)
    world = dist.get_world_size() if dist.is_initialized() else 1
    if manifest['world_size'] != world:
        raise ValueError("Exact RNG continuation requires unchanged world size")
    if manifest['total_steps'] != train_state.total_steps:
        raise ValueError("Total-step schedule changed; use an explicit initialization")
    restore_rng(torch.load(Path(config.resume_checkpoint_path) / f"rng.{rank}.pt",
                           map_location="cpu", weights_only=False))
    return resume_state



@torch.no_grad()
def upcast_optimizer_state(optim: AdamATan2):
    for state in optim.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                state[key] = value.to(torch.float32)



@torch.no_grad()
def reset_optimizer_ema_to_params(optim: AdamATan2):
    for group in optim.param_groups:
        if group.get("ema") is None:
            continue
        for param in group["params"]:
            state = optim.state[param]
            if "param_ema" in state:
                state["param_ema"] = param.detach().clone().to(torch.float32)



def sharded_checkpoint_id(checkpoint_path: str, tag: str) -> str:
    return os.path.join(checkpoint_path, f"fsdp2_{tag}")



def unsharded_checkpoint_path(checkpoint_path: str, tag: str) -> str:
    return os.path.join(checkpoint_path, f"unsharded_{tag}.pt")



def load_sharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str):
    assert config.resume_checkpoint_path is not None
    checkpoint_id = sharded_checkpoint_id(config.resume_checkpoint_path, tag)
    if not os.path.isdir(checkpoint_id):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_id}")

    source_config_path = os.path.join(config.resume_checkpoint_path, 'all_config.yaml')
    if os.path.isfile(source_config_path):
        with open(source_config_path) as handle:
            source_config = yaml.safe_load(handle)
        if source_config.get('optimizer', 'adam_atan2') != getattr(config, 'optimizer', 'adam_atan2'):
            raise ValueError('Optimizer changes require an explicit offline state conversion')
    if config.arch.model_dump().get('upcycle_from_dense', False):
        if config.arch.name != 'mimir2':
            raise ValueError('upcycle_from_dense requires HRM v2')
        from mimir.model.v2_checkpoint import load_upcycled_checkpoint
        load_upcycled_checkpoint(train_state.model, train_state.optim, checkpoint_id)
        return

    if config.arch.model_dump().get("gate_initialize_from_v1", False):
        if config.arch.name != "mimir2":
            raise ValueError("gate_initialize_from_v1 requires the HRM v2 architecture")
        with open(os.path.join(config.resume_checkpoint_path, "all_config.yaml")) as handle:
            source_arch = yaml.safe_load(handle)["arch"]
        if source_arch["name"] == "legacy_v1":
            for field in ("H_cycles", "L_cycles"):
                if source_arch[field] != config.arch.model_dump()[field]:
                    raise ValueError(f"v1 -> v2 migration must preserve {field}")
            from mimir.model.v2_checkpoint import load_v1_checkpoint
            if not load_v1_checkpoint(train_state.model, train_state.optim, checkpoint_id):
                raise ValueError("Source claims v1 but contains v2 gate state")
            return
        if source_arch["name"] != config.arch.name:
            raise ValueError("Only the original HRM baseline can initialize v2 gates")

    model_state = train_state.model.state_dict()
    if config.arch.name == "mimir2":
        # DCP loads a requested subset; it must not silently discard opt-in v2
        # modules after a default change (notably early gate-pilot checkpoints).
        metadata = dcp.FileSystemReader(checkpoint_id).read_metadata()
        source_keys = {key.removeprefix("model.") for key in metadata.state_dict_metadata
                       if key.startswith("model.")}
        unexpected = source_keys - set(model_state)
        if unexpected:
            raise ValueError("Checkpoint contains disabled or mismatched v2 features; "
                             "enable the original gates/experts/residual branches explicitly. "
                             f"Unexpected model keys: {sorted(unexpected)}")
    optim_state = get_optimizer_state_dict(train_state.model, train_state.optim)  # pyright: ignore[reportPrivateImportUsage]
    dcp.load({"model": model_state, "optim": optim_state}, checkpoint_id=checkpoint_id)
    if config.distributed_strategy == "ddp":
        train_state.model.load_state_dict(model_state)
        set_optimizer_state_dict(train_state.model, train_state.optim, optim_state)
        return
    set_state_dict(
        train_state.model,
        train_state.optim,
        model_state_dict=model_state,
        optim_state_dict=optim_state,
    )



def load_unsharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str, rank: int):
    from mimir.training.runtime import trace_print
    if config.arch.model_dump().get("gate_initialize_from_v1", False):
        with open(os.path.join(config.resume_checkpoint_path, "all_config.yaml")) as handle:
            source_arch = yaml.safe_load(handle)["arch"]["name"]
        if source_arch != "mimir2":
            raise ValueError("v1 -> v2 initialization currently requires a sharded DCP checkpoint")
    assert config.resume_checkpoint_path is not None
    checkpoint_path = unsharded_checkpoint_path(config.resume_checkpoint_path, tag)
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

    if config.arch.model_dump().get('upcycle_from_dense', False):
        from mimir.model.v2_checkpoint import expand_unsharded_state
        distributed = dist.is_available() and dist.is_initialized()
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        target = get_model_state_dict(train_state.model, options=options)
        optim = get_optimizer_state_dict(train_state.model, train_state.optim, options=options)
        if not distributed or rank == 0:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            target, optim = expand_unsharded_state(target, optim, checkpoint)
        set_state_dict(train_state.model, train_state.optim, model_state_dict=target, optim_state_dict=optim,
                       options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=distributed))
        return

    if dist.is_available() and dist.is_initialized():
        trace_print(config, rank, f"unsharded_load_begin path={checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if (config.distributed_strategy == "ddp" or rank == 0) else {}
        trace_print(config, rank, f"unsharded_load_end keys={list(checkpoint.keys()) if checkpoint else []}")
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=config.distributed_strategy != "ddp",
        )
        trace_print(config, rank, "set_state_dict_begin")
        set_state_dict(
            train_state.model,
            train_state.optim,
            model_state_dict=checkpoint.get("model", {}),
            optim_state_dict=checkpoint.get("optim", {}),
            options=options,
        )
        trace_print(config, rank, "set_state_dict_end")
    else:
        trace_print(config, rank, f"unsharded_load_begin path={checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trace_print(config, rank, f"unsharded_load_end keys={list(checkpoint.keys())}")
        trace_print(config, rank, "model_load_state_dict_begin")
        train_state.model.load_state_dict(checkpoint["model"])
        trace_print(config, rank, "model_load_state_dict_end")
        trace_print(config, rank, "optim_load_state_dict_begin")
        set_optimizer_state_dict(train_state.model, train_state.optim, checkpoint["optim"])
        trace_print(config, rank, "optim_load_state_dict_end")



def save_sharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str):
    assert config.checkpoint_path is not None
    dcp.save({"model": train_state.model.state_dict(), "optim": get_optimizer_state_dict(train_state.model, train_state.optim)},
             checkpoint_id=sharded_checkpoint_id(config.checkpoint_path, tag))



def save_unsharded_train_state(config: PretrainConfig, train_state: TrainState, tag: str, rank: int):
    assert config.checkpoint_path is not None
    if dist.is_available() and dist.is_initialized():
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state = get_model_state_dict(train_state.model, options=options)
        optim_state = get_optimizer_state_dict(train_state.model, train_state.optim, options=options)  # pyright: ignore[reportPrivateImportUsage]
        if rank == 0:
            torch.save(
                {"model": model_state, "optim": optim_state},
                unsharded_checkpoint_path(config.checkpoint_path, tag),
            )
    else:
        torch.save(
            {"model": train_state.model.state_dict(), "optim": get_optimizer_state_dict(train_state.model, train_state.optim)},
            unsharded_checkpoint_path(config.checkpoint_path, tag),
        )



def save_code_and_config(config: PretrainConfig, train_metadata: V1DatasetMeta):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        __import__("inspect").getfile(Mimir2)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(config.model_dump(), f)
    with open(os.path.join(config.checkpoint_path, "train_metadata.yaml"), "wt") as f:
        yaml.dump(train_metadata.model_dump(), f)

    # Log code
    if wandb.run is not None:
        wandb.run.log_code(config.checkpoint_path)



def save_checkpoint_metadata(
    config: PretrainConfig,
    train_state: TrainState,
    tag: str,
    epoch: int,
    batch_in_epoch: int,
    rank: int,
    checkpoint_kind: str,
    local_batch_size: int,
    resume_info: Optional[dict[str, int]] = None,
    batch_in_epoch_exact: bool = True,
    carry_policy: Literal["none", "per_rank"] = "per_rank",
):
    if config.checkpoint_path is None or rank != 0:
        return

    metadata = {
        "tag": tag,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_format": config.checkpoint_format,
        "step": train_state.step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "batch_in_epoch_exact": batch_in_epoch_exact,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "global_batch_size": config.global_batch_size,
        "local_batch_size": local_batch_size,
        "data_path": config.data.path,
        "homogeneous_categories": getattr(config.data, 'homogeneous_categories', False),
        "optimizer": getattr(config, 'optimizer', 'adam_atan2'),
        "seed": config.seed,
        "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        "carry_policy": carry_policy,
        "carry_schema_version": 1,
        "fsdp_shard_degree": config.fsdp_shard_degree,
        "fsdp_reshard_after_forward": config.fsdp_reshard_after_forward,
        "fsdp_accumulation_sync_mode": config.fsdp_accumulation_sync_mode,
    }
    if config.lr_rewarm_steps:
        metadata['lr_rewarm'] = rewarm_metadata(config)
    if config.lr_piecewise_points is not None:
        metadata['lr_piecewise_points'] = config.lr_piecewise_points
    if resume_info is not None:
        metadata.update({
            "global_row_start_in_epoch": int(resume_info["global_row_start"]),
            "global_row_cursor_in_epoch": int(resume_info["global_row_end"]),
            "global_numseq": int(resume_info["global_numseq"]),
            "global_batch_totlen": int(resume_info["global_batch_totlen"]),
        })
    metadata_path = os.path.join(config.checkpoint_path, f"checkpoint_state_{tag}.json")
    temporary_path = f"{metadata_path}.tmp"
    with open(temporary_path, "wt") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, metadata_path)



def save_train_checkpoint(config, train_state, tag, epoch, batch_in_epoch, rank,
                          checkpoint_kind="regular", local_batch_size=0,
                          resume_info=None, batch_in_epoch_exact=True):
    from mimir.checkpoints.contract import publish, rng_state
    import uuid
    if config.checkpoint_path is None:
        return
    if not re.fullmatch(r"(?:epoch|step|ephemeral_step)_\d+", tag):
        raise ValueError("Invalid checkpoint tag")
    root = Path(config.checkpoint_path)
    identity = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(identity, src=0)
    stage = root / ("." + tag + "." + identity[0] + ".partial")
    stage.mkdir(parents=True, exist_ok=True)
    staged = config.model_copy(update={"checkpoint_path": str(stage)})
    if config.checkpoint_format == "sharded":
        save_sharded_train_state(staged, train_state, tag)
    else:
        save_unsharded_train_state(staged, train_state, tag, rank)
    torch.save(rng_state(), stage / f"rng.{rank}.pt")
    if train_state.carry is not None:
        raise ValueError("Mimir2 checkpoint contract requires carry=None")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        for name in ("all_config.yaml", "train_metadata.yaml"):
            source = root / name
            if source.exists():
                shutil.copy2(source, stage / name)
        (stage / "all_config.yaml").write_text(yaml.safe_dump(config.model_dump()))
        save_checkpoint_metadata(staged, train_state, tag, epoch, batch_in_epoch,
                                 rank, checkpoint_kind, local_batch_size,
                                 resume_info, batch_in_epoch_exact, "none")
        publish(stage, root / tag, metadata={"mode": "exact", "tag": tag,
                "step": train_state.step, "total_steps": train_state.total_steps,
                "world_size": dist.get_world_size() if dist.is_initialized() else 1})
    if dist.is_initialized():
        dist.barrier()





def _ephemeral_step_from_name(name: str) -> Optional[int]:
    patterns = (
        r"^fsdp2_ephemeral_step_(\d+)$",
        r"^unsharded_ephemeral_step_(\d+)\.pt$",
        r"^carry_ephemeral_step_(\d+)\.\d+\.pt$",
        r"^checkpoint_state_ephemeral_step_(\d+)\.json$",
        r"^ephemeral_step_(\d+)\.pt$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match is not None:
            return int(match.group(1))
    return None



def remove_stale_ephemeral_checkpoints(config, keep_tag, rank):
    from mimir.checkpoints.contract import validate
    if config.checkpoint_path is None:
        return
    root = Path(config.checkpoint_path)
    validate(root / keep_tag)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        keep_step = int(keep_tag.rsplit("_", 1)[1])
        for entry in root.glob("ephemeral_step_*"):
            if entry.is_dir() and entry.name != keep_tag and int(entry.name.rsplit("_", 1)[1]) < keep_step:
                validate(entry)
                shutil.rmtree(entry)
    if dist.is_initialized():
        dist.barrier()



