# Experiment: whole-basket selection (16.18-whole-basket-v1)

Registered before any tensor is built. Owner authorization: 2026-09-20/21 —
"Adelante. Fija el protocolo y continúa directamente con la implementación y
el piloto A/B", with seven binding conditions incorporated below.

## Question

Does learning a joint score over complete baskets — a shared nonlinear basket
encoder scored against the full fit-side catalogue — make better first
choices than the incumbent formulation (independent item scores + count head
+ rule-based assembly), when both are trained from clean initialization on
the same 2,000 fit games under identical POV inputs and the corrected wallet
contract?

## Evidence this bets on (frozen, reproduced)

`data/reports/strategy_analysis_20260920/` (frozen originals) — reproduced
exactly by the versioned ports `scripts/basket_vocabulary_probe.py` /
`scripts/basket_legality_probe.py` (byte-identical sample, sha256
`4806447777d5…`; all compared fields equal):

- 291,819 TRAIN purchases → only 8,721 distinct baskets; 837 cover 90%.
- Chronological probe (2,000 fit / 500 probe games, cutoff 1789176866930,
  fit digest `d534b9f7…`, probe digest `f53db64a…`): 98.5% of probe baskets
  already in the fit catalogue; 99.1% of legal-at-budget targets in catalogue.
- Only 33.0% of probe purchases have an exact (champion, role, inventory)
  precedent — the transfer gap a neural scorer must close.
- 765/2,877 sampled targets (26.6%) are budget-blocked at `gold_est`.

## Arms (all from clean initialization; NO checkpoint that has seen the
probe games may serve as init or as an "unseen" comparator)

- **A — incumbent formulation**: the current architecture and decode
  (item-presence BCE + copy-count CE + greedy budget-constrained assembly),
  trained by this experiment's shared harness on fit games only. Weights are
  saved in the standard artifact format and decoded through the real
  `app.predictor` path.
- **B — whole-basket scorer**: same state featurization; a permutation-
  invariant nonlinear basket encoder (Deep Sets: per-element MLP over
  item+count embeddings, sum-pool, post-pool MLP — a plain sum + dot product
  is disallowed as it recreates additive item scoring); two-stage scoring:
  full-catalogue bilinear → top-K shortlist → deep interaction scorer
  (state, basket, recursive purchase cost, leftover wallet). Trained on
  complete-basket likelihood (sampled-softmax NLL over the catalogue with
  the true basket always present). SAVE stays an explicit action;
  ward-only baskets are ordinary catalogue members.
- **C — B + item semantic features** (stats, price, tags, recipe links in
  the basket encoder): built ONLY if B's result warrants it (decided on the
  internal probe, before any validation read).

## Shared contract (condition 1)

- POV inputs in every arm (`PREFIX_POV_GOLD=1` semantics).
- Corrected wallet contract: the board gold-diff features consume the SAME
  resolved budget as the query/mask/affordability path
  (`_context_extras(row, budget=…)`, CACHE_VERSION v15). Verified before
  tensor construction; both arms read identical tensors for the shared parts.
- `gold_est` (prequential-v3) is the budget; measured wallets are a
  DIAGNOSTIC ONLY in this experiment (condition 5): joined by
  (match_id, ts, champion_id), exact vs interval reported separately,
  never a training input here, and the A/B contrast does not wait for any
  new generation.

## Data (condition 4)

Generation `16.18-reconstructible-development-v3` (fingerprint `e4ea6431…`),
TRAIN split only. Fit = the 2,000 earliest TRAIN games (digest `d534b9f7…`);
internal development probe = the remaining 500 (digest `f53db64a…`). The
harness recomputes both memberships and REFUSES to run if the digests do not
match the frozen bindings. No validation or test payload is read at any
point before the final paired validation run. `visits_test.jsonl` stays
sealed; promotion stays blocked.

## Metrics and reporting (conditions 6-7)

Primary, on the internal probe (all purchases, and feasible-at-budget
subset separately):

- `first_action_exact` (agreement of the first displayed action with the
  bought basket, SAVE counted as in eval_policy).
- `any_of_three_exact`.

Decomposition per visit (condition 6): catalogue miss (true basket absent
from fit catalogue) / shortlist miss (in catalogue, not in top-K) /
legality removal (in shortlist, filtered by economy at budget) / ranking
miss (legal in shortlist, ranked below the choice). Reported per basket
size (1..5+) and separately for duplicate-copy baskets. Guardrails:
save/no-buy recall and ward-only recall must not degrade
(pre-registered floor: within 2 points of arm A's).

Measured wallets diagnostic: for the ~joined probe subset, re-evaluate
legality gating with exact wallet (and interval ceiling for bounded rows)
vs `gold_est`, reporting how many budget-blocked targets become legal.

## Decision hurdle (pre-registered)

Invest further (build C / go to paired validation) only if B beats A by
**≥ +5.0 points** of `first_action_exact` on the internal probe with a
game-paired bootstrap CI excluding zero, zero illegal displayed baskets,
and guardrails within floor. "48-72 h" is planning guidance, not a stopping
rule: an arm is declared failed only after convergence is checked
(training loss plateau + no metric improvement across ≥3 evaluation
checkpoints), not on a timer (condition 7).

## Order of work

1. Throughput pilot (`scripts/basket_pilot.py`): catalogue build, basket
   encoder forward, full-catalogue scoring, shortlist deep scoring, and
   economy-executor cost per candidate, on the real device. Go/no-go on
   feasibility of full-catalogue scoring.
2. A/B contrast + inspectable examples (50 representative disagreements).
3. C only per the hurdle; then ONE fresh paired validation run of the
   winner vs the conditional baseline under a new run plan.

## What this experiment does NOT do

No promotion, no test reads, no cross-export claims, no simultaneous model
enlargement, no cross-patch data. The completed POV run (0.3806 vs 0.3811)
is context, not a claim of demonstrated statistical equivalence — it passed
its pre-registered non-inferiority checks, which is all it is quoted as.

## Resource budget

One training process; RTX 4060 8 GB confirmed by owner (CUDA availability
verified at runtime by the pilot, not assumed); 10 GB RAM; ≥40 GB free disk;
runs as detached jobs, never children of an agent shell.
