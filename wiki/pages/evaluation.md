---
type: Technical Reference
title: Comparison protocol evidence
description: Frozen task implementations, recovered log selections and remaining reproduction limits.
status: draft
confidence: high
last_updated: 2026-10-20
---
# Evaluation evidence

436 allowlisted HRM-test evaluation headers were recovered, including later
Inspect reruns and September campaigns. Records retain producing-log paths,
header hashes, sample IDs, sample/pass counts and generation settings. Credentials,
server model arguments and authentication configuration were not copied.

Versioned suites live under `src/mimir/evaluation/suites`. Custom tasks are
ordinary Inspect tasks. Eight Danish tasks come from the frozen September 25
run's source snapshot. The direct clients call Inspect AI and EuroEval; no TSV
execution backend, proxy, runtime model patch or score repair was ported.

The later DROP producer reports **9,535 samples**, not the older guide's 9,536.
The broken earlier DROP protocol stopped at a blank line; the selected built-in
Inspect protocol has no such stop sequence. HellaSwag's later multiple-choice
10-shot format and later MMLU/Winogrande/GSM8K variants are separate protocols.
Historical exceptions retaining original Mimir/HRM-Text or anomalous Qwen results
are documented in the manifest, never applied to new measurements automatically.

BERTScore rescoring uses stored final answers, xlm-roberta-large revision
c23d21b0620b635a76227c604d44e43a9f0ee389, layer 17, FP32, no IDF or baseline
rescaling. Empty answers score zero. Truncation is reported. The source counts
are 500 Nordjylland, 960 WMT EN-DA and 973 GovReport observations. No generation
rerun is implied by rescoring.

Superseded during implementation (2026-10-20): current EuroEval checkout 18.2.0
was initially considered for the dependency pin; it is unpublished and is not
the comparison-producing version. Recovered result records identify **17.7.0**,
one pass, few-shot, test split for angry-tweets, DALA, citizen tests and idioms.
The package and lock now pin 17.7.0. Source results preserve individual pass
statistics rather than presenting ten passes that were never run.

Inspect logs and normalized per-sample records preserve generations, targets,
raw scores and epoch/pass IDs. Mean metrics merge sample sums/counts; MCC/macro
F1 and pass@k have separate sufficient-statistic reducers. W&B is optional and
registers each metric explicitly before publishing a whole record with a
fractional epoch axis.

## Remaining comparison qualification

Recovered headers are evidence, not proof that today's downloaded datasets are
identical. Some tasks lack immutable dataset revisions or frozen bootstrap
membership. Historical model-specific chat-template variants are not fully
reconciled; export-owned templates remain the default, with an explicit server
spec required for overrides. Full benchmark replay, BERTScore encoder download,
EuroEval execution and judge-capacity behavior have not run in this initial CPU
campaign. Protocol manifests therefore retain qualification notes. Do not claim
published leaderboard reproduction from the local wiring fixtures.
