# XXS DFM15 capacity ablation — prepared, not started

Four independent 20,000-step runs use the packaged XXS defaults and identical
seed (0), data order, global batch (196,608 tokens), optimizer, precision and
recurrence schedule. Each run consumes a nominal 3,932,160,000 training tokens.
The common overrides set a single epoch, a 20K schedule/stop budget and 5K
checkpoint intervals. No weights are shared or upcycled between runs.

| Variant | Change from default | Parameters excluding token embeddings and LM head | Ratio |
| --- | --- | ---: | ---: |
| Default | H3/L3, width 256 | 5,505,024 | 1.0000 |
| Engram | Orders 2/3; 10,496 buckets; 4 heads; contextual gate | 11,010,048 | 2.0000 |
| MoE | Four L MLP experts, top-1 | 10,816,524 | 1.9648 |
| MoS | Three L stacks, top-1 | 11,010,819 | 2.0001 |

The doubling target is 11,010,048. N-gram memory tables count toward added
capacity; only input token embeddings and the LM head are excluded. Those two
components contribute 134,217,728 parameters in every run. Engram means Mimir's
existing contextual n-gram memory, not an implementation of every feature in
an external Engram paper. Its default 2K-step warmup remains enabled.

MoE is 1.76% below the target, MoS 0.007% above. Integer expert counts preserve
the default width and layer dimensions. Dense mixture execution remains the
default: all experts execute and top-k masks their contributions. These runs
match parameter capacity, not FLOPs or wall time. Residual branching, persistent
H streams and branch-specific L mixing remain disabled. Automatic learning
rates and the default recurrence-depth warmup remain enabled for all variants.

## Prepare and inspect

Use the installed Mimir environment with its patched dependencies. This helper
only writes configurations, counts parameters on the meta device, checks bounded
corpus windows and creates a held native scheduler queue. It has no run option.

```bash
python tools/prepare_xxs_ablation.py .runtime/campaigns/NEW_DIRECTORY \
  --data "$DFM15_DATA" --tokenizer "$DFM15_TOKENIZER" --template "$DFM15_TEMPLATE"
```

Pass `--gpus 0,1` only to record an explicitly available future allocation; jobs
still remain held. Without it, GPU allocation is unassigned, training has empty
CUDA visibility and server GPU validation prevents inference launch. Regenerate
in a fresh output directory after deciding the allocation; process count and
resources must agree. Intermediate evaluations are optional via
`--eval-steps 5000 10000 15000 20000` and retain the cumulative 20K schedule.

The live queue is created under `/tmp` on local disk; its location is recorded
in `preparation.json`. An explicit `--database` must also be on local disk.
`queue.backup.sqlite` is an online backup, not a live database on shared storage.
Restore that backup to local disk or import `plan.json` into a fresh local queue
with `mimir schedule LOCAL_DB --plan PLAN`. Import preserves all holds atomically
per job. Do not release jobs or use `--run` during preparation.

```bash
python -m mimir monitor LOCAL_DB
python -m mimir monitor LOCAL_DB --watch
```

The generated directory includes resolved YAML, exact Hydra overrides, capacity
counts, dataset/tokenizer receipts, frozen task manifests, the dependency plan,
and the backup. All training, export, evaluation, merging and monitoring use
Mimir. No legacy runner, scheduler, environment or model code is used.

## Evaluation and prelaunch checks

By default evaluate only the final EMA checkpoint with all tasks in Mimir's
three comparison manifests: Inspect, Danish and EuroEval. No sample truncation,
prompt rewriting or protocol overrides are introduced. Each task uses its own
suite-qualified output directory; compatible task servers can be reused. Inspect
sample records merge into complete per-suite metric records; EuroEval retains
its native complete records. There is no invented cross-benchmark scalar mean.
Training successors depend on terminal evaluation jobs, never successful merges.
Explicit holds still take precedence. W&B is disabled for these preparations.

Before any later launch, qualify the assigned GPU/FA4/compiled execution and
batch memory. Check evaluation prompt lengths against the 4096-token training
context, especially long few-shot and summarization tasks; do not silently
truncate or change a frozen protocol. Full benchmark replay has not been run.
Set `HF_HOME` to the campaign's `cache/huggingface` before launching the scheduler
so in-process evaluation and vLLM inherit the same cache location as train/export.
No dataset downloads occur during preparation.

The initial preparation selected the DFM15 native-reasoning-v2 sampled corpus.
Its own receipt says `validated_staged_not_promoted`; preparation does not promote
it. The dataset references a tokenizer data directory outside Mimir, which is
copied with the dataset-owned chat template into the campaign. No sibling code
or weights are copied. Headers and at most 3072 index rows are checked, not the
entire token stream. Exact paths and hashes are recorded in local `data.json`;
the tokenizer must not be substituted with a different Gemma tokenizer merely
because its vocabulary has the same size.

## Activation update

The user subsequently authorized scheduling when GPUs become free. The local
campaign now has `scheduled-plan.json`, a released queue backup and runner.json.
The original held plan above remains a preparation record. Training uses all
eight GPUs serially; evaluation tasks run on different GPUs concurrently with
one TP=1 server per worker. The native idle coordinator waits for existing GPU
users to leave and never terminates them. It polls every 30 seconds and requires
three clear observations. Use `mimir monitor LOCAL_DB --watch` for queue and GPU
wait status. The live queue path is in preparation.json. GPU availability checks
are cooperative and cannot exclude a competing external launcher.

## Additional authorized runs

`extensions.json` adds, after MoS: `muon` (baseline architecture, optimizer=muon),
`moe_balanced` (same four-expert L MoE with aux-loss-free balancing), and
`mos_balanced` (same three-stack MoS with balancing). Each retains DFM15, seed 0,
20K steps, 5K checkpoints and final evaluation. Balancing update speed is 0.001;
its persistent non-parameter bias adds no trainable capacity. The opt-in router
uses sigmoid affinities, selection-only correction and no auxiliary objective.
See the [routing semantics](../../wiki/pages/aux-loss-free-routing.md).

`python tools/extend_xxs_ablation.py CAMPAIGN_DIRECTORY --release` validates and
appends these three runs to the local native scheduler, preserving existing jobs.
It refuses duplicate registration. Without --release new jobs remain held.
Training successors depend only on terminal evaluation attempts. Tasks retain
single-GPU assignment and TP=1. The generated extension-plan.json and resolved
configs provide the exact commands; the original plan remains unchanged.
