from __future__ import annotations
from typing import Literal, Optional
import pydantic
from mimir.model.accelerator import (
    AcceleratorType,
    empty_accelerator_cache,
    memory_stats_for_device,
    set_accelerator_type,
    synchronize_device,
    torch_device_for_accelerator,
)
from mimir.model.lr_piecewise import piecewise_lr, validate_lr_points


from mimir.model import Mimir2Config

class ArchConfig(Mimir2Config):
    model_config = pydantic.ConfigDict(extra='forbid')
    max_seq_len: int = 4096
    n_layers: int = 16
    hidden_size: int = 1536
    num_heads: int = 12
    expansion: float = 4
    init_type: Literal['fixed_normal','lecun_normal','megatron'] = 'lecun_normal'
    norm_type: Literal['pre','post'] = 'pre'
    norm_eps: float = 1e-6
    pos_emb_type: Literal['rope','none'] = 'rope'
    H_cycles: int = 2
    L_cycles: int = 3
    goldfish_strategy: Optional[str] = None
    goldfish_k: int = 50
    goldfish_context_width: int = 50
    goldfish_seed: int = 0

    name: Literal["mimir2"] = "mimir2"



class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    path: str
    validation_path: Optional[str] = None
    target_only: bool = True  # Only supervise Answer.
    homogeneous_categories: bool = False
    routing_categories: bool = False



class PretrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    # Config
    arch: ArchConfig
    data: DataConfig

    # Hyperparams
    global_batch_size: int = pydantic.Field(gt=0)
    epochs: int = pydantic.Field(gt=0)
    gradient_accumulation_steps: int = pydantic.Field(default=1, ge=1)
    gradient_clip_norm: Optional[float] = pydantic.Field(default=None, gt=0)
    gradient_skip_norm: Optional[float] = pydantic.Field(default=None, gt=0)
    gradient_skip_max_consecutive: int = pydantic.Field(default=3, ge=1)
    training_total_steps: Optional[int] = pydantic.Field(default=None, ge=1)

    lr: float
    lr_auto: bool = True
    lr_piecewise_points: Optional[list[tuple[int, float]]] = None
    lr_rewarm_steps: int = pydantic.Field(default=0, ge=0)
    lr_rewarm_stages: int = pydantic.Field(default=1, ge=1)
    lr_rewarm_start_ratio: float = pydantic.Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    lr_rewarm_start_step: Optional[int] = pydantic.Field(default=None, ge=0)
    lr_decay_start_step: Optional[int] = pydantic.Field(default=None, ge=0)
    lr_decay_end_step: Optional[int] = pydantic.Field(default=None, ge=1)
    lr_embeddings: Optional[float] = pydantic.Field(default=None, ge=0, allow_inf_nan=False)
    lr_head: Optional[float] = pydantic.Field(default=None, ge=0, allow_inf_nan=False)
    lr_h: Optional[float] = pydantic.Field(default=None, ge=0, allow_inf_nan=False)
    lr_l: Optional[float] = pydantic.Field(default=None, ge=0, allow_inf_nan=False)
    lr_min_ratio: float
    lr_warmup_steps: int
    lr_cooldown_checkpoint: Optional[str] = None

    weight_decay: float
    beta1: float
    beta2: float
    optimizer: Literal['adam_atan2', 'muon'] = 'adam_atan2'
    muon_momentum: float = pydantic.Field(default=0.95, ge=0, lt=1)
    muon_ns_steps: int = pydantic.Field(default=5, ge=1)
    muon_lr_scale: float = pydantic.Field(default=1.0, gt=0)
    ema: Optional[float] = None
    fwd_bwd_dtype: str = "bfloat16"
    activation_checkpointing: Literal["none", "full", "l_only", "h_only"] = "none"
    accelerator_type: AcceleratorType = "sm100"
    distributed_strategy: Literal["fsdp", "ddp", "none"] = "fsdp"
    fsdp_params_precision: Literal["fp32", "bf16"] = "fp32"
    fsdp_wrap_policy: Literal["transformer_block", "recurrent_level"] = "transformer_block"
    fsdp_shard_degree: Optional[int] = pydantic.Field(default=None, ge=2)
    fsdp_reshard_after_forward: Optional[bool] = None
    fsdp_accumulation_sync_mode: Literal["no_sync", "reduce_scatter"] = "no_sync"
    ddp_params_precision: Literal["fp32", "bf16"] = "fp32"
    ddp_find_unused_parameters: bool = True
    compile_train_batch: bool = True
    compile_train_batch_mode: Literal[
        "default", "max-autotune-no-cudagraphs"
    ] = "default"
    memory_log_interval: int = 0
    empty_cache_interval: int = 0
    resume_trace: bool = False
    stability_diagnostics_interval: int = pydantic.Field(default=0, ge=0)
    stability_diagnostics_max_tokens_per_microbatch: int = pydantic.Field(default=4096, ge=1)
    stability_diagnostics_output: Optional[str] = None
    stability_diagnostics_wandb_detailed: bool = False
    experiment_metrics_output: Optional[str] = None
    experiment_update_probe_interval: int = pydantic.Field(default=0, ge=0)

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_resume: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_format: Literal["sharded", "unsharded"] = "sharded"
    initialize_from: Optional[str] = None
    resume_checkpoint_path: Optional[str] = None
    resume_checkpoint_tag: Optional[str] = None
    resume_epoch: Optional[int] = None
    resume_step: Optional[int] = None
    resume_batch_in_epoch: Optional[int] = None
    upcast_optimizer_state_on_resume: bool = False
    reset_ema_on_resume: bool = False

    # Extras
    seed: int = 0
    checkpoint_interval: int = 1
    checkpoint_step_interval: Optional[int] = None
    ephemeral_checkpoint_step_interval: Optional[int] = None
    max_steps: Optional[int] = None  # Optional early stop after this many optimizer steps (benchmarking/debugging).
    benchmark_state_fingerprint: bool = False
    stop_after_step: Optional[int] = None  # Gracefully stop after checkpointing this global optimizer step.
    suspension_backend: Literal['disk', 'cuda'] = 'disk'
    suspension_settings: Optional[str] = None
    suspend_after_steps: list[int] = pydantic.Field(default_factory=list)
    suspend_control_dir: Optional[str] = None
    suspend_poll_interval_steps: int = pydantic.Field(default=10, ge=1)
    suspend_wait_poll_seconds: float = pydantic.Field(default=2, gt=0)
    dataloader_prefetch_factor: int = pydantic.Field(default=8, ge=1)  # Batches prefetched by the (single) dataloader worker.
    log_interval: int = 5
    validation_interval: int = 0
    validation_batches: int = 0

    @pydantic.model_validator(mode='after')
    def check_intervals(self):
        if self.suspension_backend == 'cuda' and self.accelerator_type not in ('sm90','sm100'):
            raise ValueError('CUDA suspension requires NVIDIA; use suspension_backend=disk')
        if self.initialize_from and self.resume_checkpoint_path:
            raise ValueError("Choose weights initialization or exact continuation")
        from mimir.training.suspension.protocol import steps
        self.suspend_after_steps = steps(self.suspend_after_steps)
        if self.suspend_after_steps and not self.suspend_control_dir:
            raise ValueError('suspend_after_steps requires suspend_control_dir')
        if self.suspend_control_dir and not self.checkpoint_path:
            raise ValueError('Suspension requires a recovery checkpoint_path')
        if self.lr_piecewise_points is not None:
            validate_lr_points(self.lr_piecewise_points)
            if (self.lr_rewarm_steps or self.lr_rewarm_start_step is not None
                or self.lr_decay_start_step is not None or self.lr_decay_end_step is not None
                or self.lr_cooldown_checkpoint is not None or self.lr_min_ratio != 1):
                raise ValueError('Piecewise LR cannot be combined with another LR schedule')
        if self.lr_cooldown_checkpoint is not None and (
            self.lr_rewarm_steps or self.lr_decay_start_step is not None
            or self.lr_decay_end_step is not None
        ):
            raise ValueError('Row-anchored cooldown cannot be combined with step decay or rewarm')
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.checkpoint_step_interval is not None and self.checkpoint_step_interval < 1:
            raise ValueError("checkpoint_step_interval must be >= 1 when set")
        if self.ephemeral_checkpoint_step_interval is not None and self.ephemeral_checkpoint_step_interval < 1:
            raise ValueError("ephemeral_checkpoint_step_interval must be >= 1 when set")
        if self.stop_after_step is not None and self.stop_after_step < 1:
            raise ValueError("stop_after_step must be >= 1 when set")
        if self.log_interval < 1:
            raise ValueError("log_interval must be >= 1")
        if self.validation_interval < 0:
            raise ValueError("validation_interval must be >= 0")
        if self.validation_batches < 0:
            raise ValueError("validation_batches must be >= 0")
        return self

