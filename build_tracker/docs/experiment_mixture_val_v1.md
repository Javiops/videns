# Pre-registration: frozen mixture on VAL + full-circuit latency (v1)

2026-09-22 afternoon. Owner's assignment, recorded BEFORE the pass. Run
dir: `data/experiments/16.18-mixture-val-v1/`. Test stays sealed.

## What is frozen before this read

`data/experiments/16.18-frozen-mixture-v1/frozen_mixture_v1.json` binds
every component by sha256: A's checkpoint `econ_ft_ep8.pt`, A's economic
domain parent `Bf_ep16.pt`, the FIT normalization-stats source, B's
checkpoint `wardfree_ep16.pt`, the decode rule, the alignment method, and
the weights (0.5 / 0.5, declared before the probe run, never tuned). The
scoring script is bound by digest too. Nothing in this VAL run may change
any of them.

## Declared VAL exposure

**This VAL split is NOT blind.** It was already read once, on
2026-09-22 ~01:00, for one frozen configuration (A) under
`docs/experiment_val_window_check_v1.md`. This pass adds B and M on the
same population. That prior exposure is stated here so the result is read
as *a check in a different population*, not as a clean out-of-sample
confirmation. A blind confirmation would need the sealed test, which this
experiment does not touch.

## Design

1. **Three arms, one pass, identical decode.** A (grouped econ), B
   (ward-free), M (0.5/0.5 mixture) scored on the same VAL rows in the
   same process, so every comparison is paired by decision.
2. **Everything computed fresh.** No cache covers VAL: parent forward →
   D(x) = top-128, economic features through the pinned `econ_features`
   engine, tuned corrected scores, ward-free forward, permutation by
   canonical key, grouped logsumexp, mixture, decode, legality at
   `gold_est`.
3. **Integrity gates, reported whether they pass or fail.**
   - The alignment permutation must be a bijection over 5,692 actions and
     leave exactly 31 positions unchanged (as on TRAIN).
   - Arm A must reproduce the previous VAL run's persisted rows
     (`16.18-val-window-check-v1/rows_val_baseline_full.jsonl`) cell by
     cell. A nonzero mismatch count invalidates this run's comparisons and
     is published as such.

## Metrics

- **PRIMARY**: ward-free `first_action_exact`, M vs A and M vs B, paired,
  bootstrapped by match (seed 16; 400 draws as in every sibling report,
  plus 2,000 draws for the primary).
- Secondary: `any_of_three_exact`; slices single / multi / no_buy_death /
  ward_only; absolute rates with per-game CIs for all three arms; movement
  counts (corrected / broken vs each parent; cases both parents got right
  that M loses; cases both got wrong that M wins).
- Declared as a cost either way: any slice where M loses to either parent.

## Latency (same run, measured not estimated)

Batch size 1 on a fixed sample of VAL decisions, warmed up first, timed
per stage: (1) parent forward → D(x), (2) economic features for the 128
candidates, (3) tuned corrected scores, (4) ward-free forward, (5)
permutation + grouped logsumexp + mixture, (6) decode + legality walk.
Reported as median and p90 per stage and for the total circuit, with the
device named. The single-model baseline contract (~40.5 ms/decision,
offline estimate) is re-measured here under the same harness so the
mixture's added cost is a measured difference, not an inferred one.

## Correction to the latency figure (recorded 2026-09-22, after the run began)

The in-run `total_single_model_A` is **arithmetic, not a measurement**: it
is the mixture's total minus the ward-free forward, and therefore still
contains the mixture's grouping, blend and its decode. It does not execute
circuit A on its own, so the ratio derived from it is not a complete A-vs-M
comparison and must not be quoted as one. The running process was left
untouched.

The measured answer comes from a separate benchmark,
`scripts/basket_latency_bench.py`, which runs each circuit end to end, per
decision, on a **fixed FIT sample** (not VAL, not the selection probe, not
test), alternating which circuit goes first so neither is favoured by cache
or clock drift, and reports both orders separately as well as pooled plus
the paired per-decision delta. It needs no second VAL pass. Output:
`data/experiments/16.18-mixture-val-v1/latency_bench_full.json`.

## Declared expectations and caveats

- VAL numbers may differ from the probe's; the earlier VAL check saw
  ~0.2-0.4 points below probe for A. The question is whether M's margin
  over both parents survives in the other population, not whether the
  absolute figures reproduce.
- A positive VAL margin is still **not** a promotion. Promotion needs the
  frozen-validation displayed-policy report with zero unaffordable baskets
  and `scripts/promote_served_model.py` binding the digest — a different
  artifact from this diagnostic.
- Running two models is a real product cost. If the latency measured here
  is unacceptable, the mixture's probe win does not by itself justify
  shipping it, and the distillation route (teach one d64 the mixed
  distribution) becomes the relevant follow-up rather than adoption.
