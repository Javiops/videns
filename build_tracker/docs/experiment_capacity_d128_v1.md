# Pre-registration: state-encoder capacity contrast d64 vs d128 (v1)

2026-09-22 ~01:30, owner-authorized overnight capacity experiment. Run
dir: `data/experiments/16.18-capacity-d128-v1/`. Task: 'BuildTracker
Capacity AB' (queued behind 'BuildTracker Val Window Check'; one GPU job
at a time).

## Question

Does widening ONLY the state encoder (d_model 64 -> 128) improve main-
purchase selection, holding data, objective, basket representation and
the grouped decoder fixed? Not a threshold tune, not a ward-treatment
change.

## Arms (both from clean initialization)

- A control: d_model=64, layers=2, heads=4, ff_dim=128.
- B candidate: d_model=128, layers=2, heads=4, ff_dim=128.

The ONLY deliberate change is d_model. Dimensions that necessarily
depend on it (PrefixModel embeddings/attention/query_proj/output heads,
and BasketNet.state_proj input = 2*d_model) adapt; BasketEncoder, the
scorer STATE_DIM=128, losses, features and every other decision are
identical. d_model is passed EXPLICITLY per arm (PrefixModel kwarg);
module env defaults untouched.

**Why arm A is retrained instead of reusing Bf_ep16**: this spec requires
the same batch order in both arms from a generator independent of model
randomness. Phase-2's Bf drew its batch order from the device-global RNG
entangled with model initialization, so batch-order identity with arm B
cannot be demonstrated. Identity of data/recipe/config therefore fails
on that clause; A is retrained. (Bf_ep16 never trained on the probe;
the exclusion is about the order contract, not contamination.)

## Data and objective (hash-bound)

Same frozen generation 16.18-reconstructible-development-v3; fit/probe
membership digests asserted against FROZEN_FIT_SHA / FROZEN_PROBE_SHA
(first 2,000 TRAIN games fit, last 500 probe). Input tensors reused from
the frozen v1 caches (cache_fit_full_v2.pt / cache_probe_full_v2.pt) —
built under the harness contract: POV inputs + corrected wallet
(PREFIX_POV_GOLD=1, CACHE_VERSION v15), identical for both arms by
construction. Full catalogue WITH wards, identical in both arms. Exact
full-catalogue cross-entropy, microbatch 32, accumulation to effective
256 (pilot must confirm memory). Recipe: Adam 1e-3, cosine T_max=16, 16
epochs, seed 16. Batch order: torch.Generator seeded 16, independent of
model randomness, same sequence in both arms. No new economic features,
no vocabulary change, no new negatives, no cardinality change. The
frozen dev baseline (econ_ft_ep8 + grouped decoder) is an additional
REFERENCE under the same metric — it is not the architectural control.

## Evaluation (identical grouped decoder in both arms)

Score full catalogue -> group ward/no-ward variants (2055, 772043) ->
per-group logsumexp -> legality of the MAIN purchase at the original
budget -> up to three DISTINCT main purchases. Empty group = "no main
purchase". Metric: ward-free first_action_exact over ALL probe
decisions; PRIMARY = B − A, paired-by-game bootstrap. Also reported:
any-of-three (denominator stated: decisions with non-empty stripped
truth); single/multi after stripping; real no-buy and ward-only
separately; family (explicit common definition: family(target) =
recursive components of target plus target, evaluated only when the
stripped truth is non-empty and target_id is known and not a ward; match
= any item of the predicted main basket in the family); illegal/
unaffordable walked candidates and fallback saves; corrected/broken/net
B vs A; wall time, peak GPU memory, inference cost; A and B vs the
frozen econ grouped baseline (reference). All per-decision rows
persisted with identity (match_id, ts, champion); uniqueness and
alignment verified — never match+timestamp alone.

## Result definition and thresholds

The epoch-16 checkpoint per arm is THE result; checkpoints 4/8/16 are
saved for learning diagnostics, never retrospective best-picking.
Practical guidance threshold, pre-declared: B − A >= +0.5 points with a
favorable interval. Outcomes distinguished: statistical improvement
(CI > 0), practical improvement (>= +0.5 with favorable CI),
inconclusive (CI spans 0). No global victory is declared for beating A
if the econ baseline remains better under the same metric.

## Overnight execution contract

Pre-registered before training (this file). A REAL forward+backward
pilot runs first per arm: measures seconds per effective batch and peak
memory, projects total duration; if the projection exceeds 7.0 h, or
free disk is below 40 GB, the run aborts with the block documented in
the report instead of training. Durable scheduled task via
register_task.ps1; waits for the val checker to finish so only one GPU
job runs; closed budget, max 8 h, no automatic extensions or sweeps;
unbuffered prints plus a per-epoch progress JSONL
(progress_full.jsonl); full-state checkpoints recoverable. Host RAM use
stays well under 10 GB (tensor caches ~2 GB). No test reads; no val
reads; the checker's results are not used to shape this experiment; no
promotion or baseline replacement. Riot sequences and unrelated jobs
untouched; the PC stays on. Code state pinned by the launch commit
recorded in the report.
