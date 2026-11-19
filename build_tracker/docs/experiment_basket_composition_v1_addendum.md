# Stage-2 addendum — frozen 2026-09-13 (post stage-1 diagnostic)

frozen_policy: sequential-v1

## Diagnostic evidence (bound)

`data/experiments/basket-composition-v1/diagnostic/train_miss_partition.json`,
TRAIN only, 279,748 rows scanned, 274,647 purchase decisions, checkpoint
`candidate.pt` (sha in the run completion). Coverage 138,101 (50.28%).
Miss partition (n=136,546): lead_item_missing 68,593 (50.23%),
budget_infeasible 30,310 (22.20%), other 25,803 (18.90%),
composition_truncated_or_blocked 9,023 (6.61%), count_truncated 2,817 (2.06%).

## Decision under the predeclared mechanical rule

Generator-addressable classes (composition + count + other) = 37,643 =
27.57% of misses ≥ 25% → intervention **A, `sequential-v1`**
(scripts/basket_composition_policy.py, as smoke-verified: SAVE-led decisions
byte-identical, non-lead cards unchanged, composed baskets legal and
affordable by construction). Stopping rule not triggered
(budget_infeasible + other = 41.10% < 80%).

## Honest expectation bounds, stated before the run

- The addressable ceiling is 27.6% of TRAIN misses; half the miss mass
  (lead_item_missing) is a ranking problem no recomposition can touch.
- Known risk: composition extends a lead the model already had exactly right
  (the smoke showed ward-heavy extensions). Single-item decisions carry ~42%
  of current exactness; the win rule and guardrails decide, and no
  composition constant will be tuned in response to this run.

## The one authorized comparison

`scripts/run_basket_composition_stage2.py`: full assert suite, then exactly
one paired evaluation (composed vs original, same frozen checkpoint) on the
v2 validation — 400 game resamples, seed 16, both primaries must favor the
composed arm, ward-only and post-death no-buy recall must not significantly
regress, zero unaffordable options. The run appends the first line of
`data/ml/v2_validation_selection_log.md`. Win or lose, the result is kept and
nothing is re-tuned on this validation.
