# Isolated 16.17 development experiment

The owner authorized a clean 16.17 database and actual model training on
September 11, 2026, to learn which changes merit later testing on 16.18.
This run is development evidence, permanently ineligible for model serving.

## Population and data acceptance

The frozen local source plan is
`data/acceptance/16.17_development_20260911/source_plan.json`. It contains
13,283 hash-bound ranked-solo source pairs: 12,399 original training games and
884 post-cutoff games assigned explicitly to development validation. Every
validation game follows the selected training games in time. These validation
games were used by a prior probe; they are not an untouched confirmation set.
The original 2,955 test IDs remain reserved. Their raw pairs are unavailable
locally, so this generation has no final-test examples; its test file is empty
and its test counts are zero.

The builder rejects whole games containing Magical Footwear rune 8304, Viego,
opaque undo events, malformed sources, unresolved terminal inventories, or
invalid modeled/hidden regular-slot occupancy. It stores only the screened
games in a new database and records every exclusion. This deliberately narrow
first population avoids inventing unobserved boot grants or possession states.
Source inspection found at most 1,517 training and 106 validation games without
either of the first two exclusions; further quality checks reduce those counts.

The database then receives the full independent semantic audit, qualification
and coverage checks. Export preserves the source plan's exact membership and
fold assignment; no held-out game is silently moved to training. No training
starts before those gates and immutable publication succeed.

## Fixed first comparison

Question: does the board transformer improve displayed shopping-policy
imitation over the fixed conditional lookup on these same validation games?

The fixed recipe uses a 128-dimensional, four-layer, eight-head transformer,
256-dimensional feedforward layers, causal gold, runes and existing extra
features. It trains a 16-epoch trunk, followed by a six-epoch frozen-trunk graft
with save/target weights 0.5/0.25. The presence-weight schedule is 8:24.
History, target blending, save splicing and custom weighting remain disabled.
Seed is 16. This is one recorded recipe, not an architecture sweep.

The conditional policy fits training rows only. Both policies receive the same
label-stripped validation inputs and are scored in a common traversal. Primary
metrics are `first_action_exact` on all scored visits and `action_family_match`
on purchases with a known next final. Paired game bootstrap uses 400 resamples
and seed 16. The report includes denominators, uncertainty and budget legality.
Canonical trainer scores are diagnostics and cannot substitute for this report.

Per-visit paired outcomes are retained for role, champion, time, budget and
purchase-type error analysis. Any follow-up change must be justified by these
results and compared on the same frozen export. The result says nothing yet
about newer patches, exact live gold, or whether an imitated purchase wins games.

## Execution and evidence

Run `scripts/run_patch_development.ps1` only through its hidden one-shot Windows
task, `BuildTracker 16.17 Development V1`. The Python supervisor runs one child
stage at a time, with a Windows job-object memory cap of 10 GB, a 40 GB free-disk
floor and a six-hour overall wall budget. The final plan is bound to the
published dataset fingerprint before training. All code and source-plan digests
are checked between stages; serving and hold files are protected by hashes.

The planned paths are:

- Database and frozen sources: `data/corpora/16.17-development-v1/`
- Acceptance reports: `data/acceptance/16.17_development_20260911/`
- Dataset generation: `data/dataset_generations/16.17-development-v1/`
- Protocol, stage logs, artifacts and policy evidence:
  `data/experiments/16.17-development-v1/`

Inspect `status.json` and `completion.json` in the experiment directory for
actual execution state. A configured or launched task is not a training result.
The production hold, final test, 16.18 collector plan and existing served model
are not altered. No promotion, commit or shutdown is authorized by this run.

## Recovered-data comparison (v2)

The second run uses the disjoint union of the first qualified 1,013 games and
9,706 source-consistent Magical Footwear games. These are prospective counts:
fresh reconstruction, semantic audit, qualification and export still decide
the actual training population. The frozen folds contain 9,988 train and 731
development-validation games before these gates. Original test IDs stay reserved.

The process-local `footwear-development-runtime-v1` supplies the same pinned
16.17 rules to reconstruction, export, training, both displayed policies and
the independent affordability replay. Physical item 2422 remains identifiable;
its grant is free, its recipe credit is 300, and its sale value is 90. A selected
rune blocks buying boots before the inferred grant. No production process is
switched to this runtime.

Delivery follows the causal timer and first-known-room rule. A 1,000 ms
decision boundary guard is a sensitivity choice, not a measured upper bound on
server delivery delay. Every forensic event stays in the corpus. The bound
`decision_admission.json` only omits uncertain input rows; source-backed
completed-item targets remain available under `full-source-completions-v1`.

The recipe and paired metrics are the same as v1, but the comparison is made
anew against a train-only lookup baseline on this exact v2 export. Cross-export
headline differences must not be described as model improvement. Viego and
unresolved replay cases remain excluded. Results remain development evidence.

`scripts/run_recovered_development.ps1` runs as the hidden one-shot task
`BuildTracker 16.17 Development V2`. It runs a sequential build, a four-worker
semantic audit, qualification, coverage, export, one trunk, one graft, paired
validation and error analysis. The owned process tree has a 10 GB memory cap,
a 40 GB disk reserve and an eight-hour overall wall budget. Code and source
digests are frozen before build and checked between stages. The published
fingerprint is bound into the run plan before any training.

Actual execution evidence is written under:

- `data/corpora/16.17-development-v2/`
- `data/acceptance/16.17_development_recovered_20260911/`
- `data/dataset_generations/16.17-development-v2/`
- `data/experiments/16.17-development-v2/`

The experiment's `status.json` reports its current stage; `completion.json`
exists only after the paired result and its checks finish.
