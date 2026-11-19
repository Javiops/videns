# Fixed Gaussian resource fields with the strongest tree learner

Status: shelved before execution on2026-10-01 after the user rejected the
current predictive-model approach and prioritized improving M2 purchases.
Retained as research history. Do not launch this proposed experiment.

2026-10-01. Declared before computing these features on real player rows or
fitting this experiment. The completed neural recipe remains negative and
unchanged. This study isolates fixed analytical features while retaining the
strong tree recipe; it does not retune that neural experiment.

## Hypothesis and mathematical definition

Separate team summaries discard some health/resource/position associations.
The synthetic collision witness proves such information loss, not useful
real-world prediction gain. The neural test did not demonstrate that gain:
both learned pooling and attention lost to the frozen full-combat classifier,
and the global neural control also lost. Test fixed Gaussian fields with the
same stronger histogram-boosting recipe and a matched spatial-shuffle control.

Use the existing prototype `winprob/features_wp_kernel_v1.py` unchanged.
For valid current positions, K_ij=exp(-d_ij^2/(2*r^2)), r=2000world units,
the same radius already used by frozen rich geometry. Coordinates are divided
by14870 in the raw player tensor; the radius is scaled accordingly. There is
no radius, feature, transformation, horizon or hyperparameter search.

Four nonnegative per-player weight channels are fixed: positive-health
indicator; health/healthMax when valid; that fraction times totalGold/10000;
and that fraction times max(netGoldOutflow/10000,0) times wallet-observed mask.
Unknown positions have kernel row/column0. This does not impute a location.
For each positioned, positive-health anchor, compute blue and red weighted
fields B_i and R_i and (B_i-R_i)/(B_i+R_i), or0 for denominator0. Summarize
mean/min/25th-percentile at each own side's anchors, reversing the balance
orientation for red, then emit blue-minus-red and blue-plus-red summaries.

For each channel normalize the positioned weights separately within each
team. Emit the cross-team kernel mean, squared MMD=b'Kb+r'Kr-2b'Kr, and an
explicit both-positive-masses flag. If either mass is0, cross-mean/MMD/flag
are0. Clamp negative floating-point roundoff only after asserting MMD>=-1e-12.
Add side difference/sum fractions for positioned-positive-health anchors,
ratio-known positioned players and wallet/ratio-known positioned players.
There are exactly42 added inputs:36 channel summaries and6 availability fields.

MMD is a descriptive input here, NOT a two-sample hypothesis test or evidence
of a win advantage. The kernel ignores terrain/reachability/vision/ability
cooldowns. Positive health is a proxy; net gold outflow is not certified item
value; the clipped weight is explicitly a nonnegative observation proxy.
No synthetic rearrangement or model attribution is a causal action value.

Conceptual sources: [Gretton et al.,2012](https://jmlr.org/papers/v13/gretton12a.html)
for kernel discrepancies; [Grinsztajn et al.,2022](https://arxiv.org/abs/2207.08815)
for the importance of tabular learner comparisons. Neither establishes that
this proposed weighting, representation or LoL use will improve anything.

## Exact existing data and two new models

Use the audited player cache manifest SHA256
`121d2afa80722e14fb1dfc7a611d1c1e382a37369cefca6c8b10771d8015ae0c`
and its unchanged global parent manifest SHA256
`7a6852da971246e67051b0ad7622b27b1ff20fd7ce8cb43d9fd76ccf41d89c14`.
The populations remain11468FIT/2866calibration/2861development games.
Use the exact64422 sampled FIT rows and all66279calibration/66864development
rows (197565model rows total), with equal total game weights. Original sampler,
chronological partitions, row order, timestamp, patch, label and evaluation-only
end time stay unchanged. Do not exclude, replace or select any extra game/row.

All2861development games have prior exposure. They are the fixed primary
development cohort, never a fresh test. Report998original pilot and1863old-pilot
complement separately as secondary. All1911own confirmation games and all
sibling/global seals remain closed. Revalidate current reservation metadata.
No new raw archive or source database read. No sibling runtime imports.

Both new pipelines receive unchanged173global inputs plus42field features:
- kernel_actual uses the real joint player tensor;
- kernel_position_shuffled recomputes fields after the position x/y/mask
  bundles are deterministically permuted within each team, leaving all other
  player fields and all173global inputs unchanged.

Use the SAME outcome-free permutation rule as the closed player study,
seed20260929/game_index/timestamp/team/slot, identical in FIT/calibration/
development. Verify it against that study's bound permutation artifact. It
preserves team position marginals and destroys additional player-position
association; it is a statistical control, not a realistic intervention.

Store new packed matrices and row identities with digests after exact
read-back parity checks. Packing changes no input value. Derive42features in
bounded chunks from the already audited raw35-field tensor; no neural
normalization is applied. Report ranges/availability, without row filtering.

Fit two FRESH classifiers via the unchanged `CurrentObservedModel` wrapper:
HGB150iterations,15leaves,learning_rate.06,L2=10,min_leaf100,
early_stoppingFalse,seed20260929. No model warm start, architecture search,
feature selection, ensembling, extra seeds or reweighting. Platt calibration
uses the unchanged equal-game recipe on the same2866calibration games
(C1e6,max_iter500,seed20260929). Save both models/calibrators and complete
training receipts before computing their development predictions or metrics.

The sole strong reference is frozen combat_full from
`data/runs/wp-scale-fit-20260930-v2/experiment`, report SHA256
`6e10781c05ce85d706ef87f723c36bdabebbc75613624193bd2144a8fa2fe014`.
Its predictions must exactly replay on the packed unchanged development173.
No weaker neural or pilot reference substitutes for this benchmark.

## Metrics, gate and use

Compute equal-game/equal-frame-within-game log-loss, Brier, accuracy and ECE.
Use2000paired game bootstrap draws, seed20260929, with relative loss ratios
recomputed inside each draw. Only kernel_actual can be selected. Require ALL:
- complete scope/packing/feature/inference checks, both complete150-iteration fits;
- >=2% relative calibrated log-loss reduction against combat_full, positive
  paired95% log-loss and Brier reduction lower bounds;
- >=0.5% relative calibrated log-loss reduction against the matched shuffled
  fields, positive log-loss lower bound and nonnegative point Brier reduction;
- on frames at least180000ms before verified end, >=0.5% relative log-loss
  reduction versus combat_full, positive log-loss lower bound, nonnegative
  point Brier reduction and >=1700represented games.

Failure of any condition selects none. No secondary cohort, raw score, patch,
time window, alternative radius, weaker comparator or revised gate can rescue
the result. Report both models/reference, raw/calibrated scores, primary and
secondary cohorts, lead stratum and fixed3-10/10-20/20-30/30-40minute/patch slices.
The shuffled control is not a deployment candidate. Intervals condition on
fitted models and this is repeated-development selection, not confirmation.

A positive result would justify a separately declared confirmation/practical
review study, not automatic promotion. The intended contribution is better
retrospective review of local support/exposure with explicit missingness.
Predictive improvement alone is not coaching quality, an item recommendation,
a causal policy effect or completion of the broader breakthrough objective.

## Execution and verification

New own-project files/artifacts only. Prepare, fit/evaluate and audit run as
separate hidden durable CPU1 jobs, each1800seconds cap,2.5GiB startup and0.75GiB
live RAM floor. No GPU and no competing game/benchmark. Bind source metadata,
protocol/code, cache/row/model/control digests and exclusions (none allowed).
Record actual resources. Reverify reservations before publication.

Independently verify all packed global/identity rows against the parent inputs,
all feature algebra/ranges, and a deterministic outcome-free role/patch/time
sample of the42fields against a scalar implementation (absolute float32
tolerance1e-6). Reproduce every development prediction from saved models and
independently recompute primary/cohort/lead metrics and selection. Preserve
failed attempts; do not edit consumed sources, prior artifacts or the index.
