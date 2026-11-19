# Authorized continuation after the September 12 reconstruction audit

On September 12, 2026, the owner instructed: "once 1 finishes do 2-4".
This authorizes implementing and running the following development sequence
after the existing 16.18 reconstruction and full semantic audit pass. It does
not authorize promotion, final-test evaluation, a beta rollout, commits,
changes to the running collectors, or a shutdown.

The owning Codex task is `01a09563-0619-7e83-b829-63eb75f0f373`.
The existing `16-18-next-two-nights` follow-up retains collection and acceptance
of its dated samples; it must not launch a duplicate baseline/candidate run.
Later cumulative samples are not silently included in this fixed experiment.

## Prerequisite and execution boundary

Read `CLAUDE.md`, the latest `HANDOFF.md`, this document, and
`scripts/status.py` before acting. Step 1 is the existing task
`BuildTracker 16.18 Reconstructible Sources`, with evidence under
`data/acceptance/16.18_reconstructible_20260912/` and corpus
`data/corpora/16.18-reconstructible-v1/`.

Require `run/completion.json` and `run/status.json` to report complete build
and audit, verify their qualification binding, and verify the successful full
semantic report and its source, corpus, static-data and decision-mask bindings.
A completed screening batch, partial folder or scheduler state is insufficient.
If the audit fails, diagnose its actual report before any export or training;
never skip or weaken the failed check.

Until the reconstruction supervisor finishes, do not edit its pinned code:
it binds every `app/*.py` and the scripts named in its `run/protocol.json`.
Documentation and read-only planning may proceed. Before later edits, inspect
other running supervisors' bindings as well. Do not race or stop Riot jobs.

The source snapshot is immutable. In particular, retain its original
`training_or_export_authorized: false`: that describes its earlier scope.
Record this later owner authorization in a new benchmark/run plan rather than
rewriting old source or qualification evidence. The qualification's
`training_clearance: false` is likewise not an export-admission certificate.

## Step 2: scoped 16.18 benchmark and immutable export

Implement an explicit development admission path for the qualified 16.18
reconstructible population. Do not merely relax the current 16.17 guard or
reuse the one-epoch smoke purpose. Keep existing generations and their readers'
historical evidence valid. The new path must verify the full qualification,
the independent Viego ownership evidence, the footwear/slot runtime and mask,
patch-pinned static data, source/code digests and full-source target semantics.
Retain quantities and all exclusion reasons; delivery timing remains inferred
with the declared one-second boundary guard.

Before reading any data payload for export, review prior split assignments and
game-creation metadata. Reserve all prior test IDs and never open test payloads.
Keep prior validation IDs out of training, disclose prior development exposure,
and verify that every scored validation game follows all selected training
games, keeping timestamp ties together. A new cumulative 70/15/15 split is not
automatically valid. If old held-out dates prevent using newer training games,
explicitly reserve the old holdouts and declare a forward development
validation cohort, or keep the compatible earlier training population. Record
the chosen membership, temporal checks, exclusions and relationship to previous
benchmarks before publication. Do not claim an untouched final test for this
development run.

Publish a new immutable generation with coverage, acceptance, source membership,
split version and export fingerprint. Verify complete train/validation payload
bindings, causal gold, role-slot state, footwear state, quantity encoding and
mask/target handling. Train only after these checks pass. Use unique names;
resume valid checkpoints and never overwrite a completed generation.

## Step 3: fixed transformer versus a train-only conditional policy

Question: does the current transformer recipe improve displayed-policy
imitation over the conditional lookup on these same 16.18 validation inputs?

Keep the reviewed V2 recipe: seed 16, dimension 128, four layers, eight heads,
feedforward size 256, runes, causal gold and existing extras, presence-weight
schedule 8:24; 16 trunk epochs followed by six frozen-trunk graft epochs with
save/target weights 0.5/0.25. History, target blending, save splicing and custom
weighting remain off. Verify actual environment/config parity in the runner.
Fit the conditional baseline on exactly this generation's training rows only.

Predeclare both primary metrics: `first_action_exact` on all scored validation
decisions and `action_family_match` on validation purchases with a known next
final. Score the actual displayed policies in a shared traversal with labels
removed. Use game-paired differences, 400 bootstrap resamples and seed 16.
Record denominators, confidence intervals, per-visit outcomes and independent
wallet/slot/recipe legality checks. Require zero unaffordable displayed options.
Call a supported win only if both primary paired intervals exclude zero in
the favorable direction; mixed or inconclusive results remain such. Complete
the fixed run without hyperparameter searches or early threshold shopping.

Bind the export fingerprint, artifact names, configurations and code hashes
into an immutable machine-readable protocol before training. All results are
development evidence; do not compare percentages across different exports
as model improvement, or describe reconstructed-wallet checks as live evidence.

## Step 4: one additional-copy decoder experiment

The completed 16.17 V2 analysis motivates exactly one stricter rule for
additional copies, not an arbitrary multiplicity cap. After step 3, use its
same 16.18 checkpoint, validation population and export for the original and
modified decoders. Hold first-copy ranking and all other policy settings fixed.
Keep the old checkpoint immutable; bind the decoder policy/version separately.

Before scoring, record the exact proposed rule and why it can reduce excess
copies. Preserve the ability to recommend every legal source-supported
multiplicity. Compare excess-copy and missing-copy errors as well as the two
primary displayed-policy metrics on identical decisions, with paired game
intervals and independent legality checks. Check the real displayed decoder,
not a simplified surrogate. Do not tune additional thresholds after seeing
this comparison or call lower excess-copy counts alone an improvement.
Retain and report a negative or inconclusive result without overwriting the
original candidate. No retraining or architecture sweep is part of this step.

## Resource budget, validation and completion

Run appropriate targeted regressions for split isolation, admission/tampering,
mask causality, target multiplicity, policy parity and legality before launch.
Complete the repository's required checks after implementation. Preserve
existing work; do not stage or commit `data/` or unrelated changes.

Use hidden, logged, durable one-shot Windows jobs for expensive work. One
training process and one expensive child stage at a time; stream JSONL. Plan
a maximum 10 GB child-tree memory cap, 40 GB free-disk reserve and eight-hour
wall budget for each scheduled experiment attempt, verified against available
resources before launch. Use explicit status, stage logs, checkpoint and failure
records. Stop cleanly on errors or resource limits; do not automatically restart
a failed training run under changed settings. Resume only validated stages.

Preserve production hold/serving hashes, completed V2 evidence, source snapshots
and collection/refetch plans. Never loosen production serving eligibility for
these development artifacts. Report each metric with split, n and report path.
Update HANDOFF with actual progress and final evidence. Stop this continuation
after the export, baseline comparison and single decoder comparison are
complete, or on owner cancellation. Notify on meaningful progress, completion,
failure or required user action; stay quiet while jobs are unchanged.
