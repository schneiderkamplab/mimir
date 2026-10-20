# Mimir

An independent Python package for Mimir2 training, checkpointing, export,
evaluation and scheduling. Default model semantics retain the original Mimir
backbone. Gates, expert mixtures, recurrent/residual branches, n-gram memory,
metadata routing, upcycling and Muon remain explicit options.

Linux, Python 3.13 and uv 0.13.3 are the qualified installation target.
Dependencies are locked; other operating systems are not qualified.

```bash
uv sync --extra training --extra test
uv run mimir doctor
uv run mimir train size@arch=XXS data.path=/path/to/sampled \
  checkpoint_path=checkpoints/example
```

For CPU training, also set `accelerator_type=cpu distributed_strategy=none
fwd_bwd_dtype=float32 compile_train_batch=false`. For multi-node GPU training,
launch `python -m mimir train` through `torchrun --module mimir train` with the
usual node/rendezvous options. FA4 is in the `sm100` extra.

The `serving` extra pins base Transformers/vLLM revisions. Mimir2 also requires
the captured integration patches; apply them in new dependency checkouts:

```bash
uv sync --extra training --extra serving --extra evaluation --extra test
uv run python tools/bootstrap_forks.py .runtime/dependencies
uv pip install --no-deps -e .runtime/dependencies/transformers
uv pip install 'cmake>=3.26.1' ninja 'setuptools>=77,<81' \
  setuptools-scm setuptools-rust wheel
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
VLLM_PRECOMPILED_WHEEL_COMMIT=a46abb7ae68acc13a4fc5870db98619b3f97c6e0 \
  uv pip install --no-deps --no-build-isolation -e .runtime/dependencies/vllm
```

After installing patched dependencies, use `.venv/bin/mimir` or `uv run --no-sync`
to avoid replacing those patched installs with the base revisions. Fork patch
hashes and revisions are in `provenance/forks.json`; patches apply to clean bases.
GPU builds and execution still require qualification on the target machine.

```bash
mimir train --suspendable size@arch=XXS data.path=/path/to/sampled \
  checkpoint_path=checkpoints/example 'suspend_after_steps=[100,200]'
mimir checkpoint import /path/to/legacy imported --tag step_100 --mode weights
mimir export checkpoints/example/step_100 exported --tokenizer /path/to/tokenizer
mimir serve exported --gpus 0 --port 8000
mimir eval src/mimir/evaluation/suites/comparison-inspect-v1.json \
  exported results --base-url http://127.0.0.1:8000/v1
mimir schedule queue.sqlite --plan plan.json --run
mimir monitor queue.sqlite --watch
```

Weights imports initialize with `initialize_from=imported`; they cannot be used
as exact resumes. `--mode exact` requires optimizer/EMA, RNG, cursor and schedule
state and rejects missing legacy information. Exact resumes use
`resume_checkpoint_path=checkpoints/example resume_checkpoint_tag=step_100`.

Read [architecture](wiki/pages/architecture.md),
[training and checkpoints](wiki/pages/training.md),
[evaluation protocols](wiki/pages/evaluation.md),
[scheduling](wiki/pages/scheduling.md), and
[validation limits](wiki/pages/validation.md) before operational use.

Run CPU tests after installing the patched Transformers/vLLM dependencies:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q
python tools/qualify_cpu.py .runtime/new-cpu-campaign
python scripts/validate_okf.py wiki
```

This initial implementation has CPU evidence. It is not a production migration
or a claim of completed B200, multi-node, real-server or full-benchmark qualification.
