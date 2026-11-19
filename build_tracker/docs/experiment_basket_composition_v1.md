# Experiment protocol — basket composition v1 (predeclared)

Drafted 2026-09-13 at the owner's instruction ("haz 1, 2 y 3"), before the v2
development generation exists. This document freezes the question, the
evaluation contract, the guardrails and the stopping rules. It does NOT
authorize training or evaluation runs by itself: each stage launch needs the
owner-visible ledger entry pattern used on September 12, and stage 2's exact
intervention must be frozen as a dated addendum to this file BEFORE its run
starts. No final-test read, promotion, serving change, commit or shutdown is
part of this experiment.

## Recorded question

Can changing how candidate baskets are GENERATED — not how existing cards are
ranked, gated or thresholded — raise displayed-policy imitation on multi-item
decisions without degrading save/ward behavior?

## Motivating evidence (bound)

All from the completed 16.18 benchmark and its expanded re-derivations:

- Exact-action agreement collapses with recorded basket size (0.4250 → 0.1897
  → 0.1104 → 0.0874 for 1→4 items) while family match rises (0.5197 → 0.7674).
  `data/reports/expanded_metrics_16_18_dev_v1/expanded_metrics_val.json`.
- The recorded basket is absent from all three displayed cards on 10,469 of
  17,715 purchases (59.1%); rank-2/3 rescues are bounded at 1,446. Same report.
- All four September 12 variants (rank, gate, head losses) moved decisions
  only inside that coverage envelope and were rejected under the two-primary
  rule. `data/reports/expanded_metrics_arms_20260912/README.md`.
- The same structure repeats at 5x the sample on the 16.17-development-v2
  export. `data/reports/expanded_metrics_16_17_dev_v2/expanded_metrics_val.md`.

## Data contract

- Export: `16.18-reconstructible-development-v2` (eval version 7), policy
  `forward-validation-v1` — training ends at the frozen v1 snapshot instant
  (2026-09-12T11:03 UTC); validation is every qualified newer game and is
  unshopped at admission. Its `export_fingerprint` and generation-manifest
  SHA256 must be bound into each stage's frozen plan at launch.
- The v1 151-game validation window is retired from selection: those games
  enter neither fold here.
- Reserved test IDs remain sealed; no test payload is materialized or read.
- Validation-spend ledger: every scored comparison on the v2 validation must
  append one line to `data/ml/v2_validation_selection_log.md` (date, stage,
  artifacts). The window starts with zero selection rounds; keep it that way
  until a run is actually scored.
- Never compare v2 headline values against v1 or 16.17 exports as improvement.

## Stage 1 — TRAIN-only candidate-miss diagnostic (no training, no val reads)

On v2 TRAIN rows only, run the current displayed policy (original benchmark
candidate re-trained on v2 TRAIN per the fixed recipe below — or, if the
diagnostic precedes retraining, the v1 checkpoint with that provenance
recorded) with labels stripped, and partition every purchase decision whose
recorded basket is absent from all three cards by mechanism:

1. `lead_item_missing` — no card contains the highest-count recorded item.
2. `composition_truncated` — some card contains a strict subset of the
   recorded multiset and the remainder was affordable at gold_est after the
   subset's recursive cost.
3. `composition_blocked` — as (2) but the remainder was gated (target filter,
   slot, stack, boot family) rather than unaffordable.
4. `count_truncated` — right item set on some card, wrong copy counts.
5. `budget_infeasible` — the full recorded basket exceeds gold_est under
   recursive costs (not recoverable by any generator; reported, not attacked).
6. `other` — none of the above; keep examples.

Deliverable: counts with denominators and 25 bound examples per class under
`data/reports/basket_composition_v1/train_miss_partition.json`. These counts
choose stage 2's intervention; they are never accuracy claims.

## Stage 2 — one generation-side intervention (frozen addendum required)

Candidate interventions, of which EXACTLY ONE is frozen after stage 1:

- **A. Sequential budget-aware composition**: build each card by iterating
  (pick best legal item under remaining gold_est → deduct recursive cost →
  re-rank conditioned on updated inventory/budget → repeat until stop), with
  the stop decided by the existing count/save machinery. No new thresholds
  tuned on validation; any constant is fixed before launch.
- **B. Count-conditioned completion**: keep first-copy ranking; extend the
  lead basket with the count head's conditional copies and the target-recipe
  remainder while legal and affordable.

Either arm must preserve: recursive combine costs, slot/stack/boot legality,
ward stacking, SAVE gating and fallback semantics, multiset targets, and the
candidate cap of three displayed cards. If the intervention involves
retraining, both arms (control = current generator, candidate = new
generator) share the fixed V2 recipe: seed 16, d128/4L/8H/ff256, runes,
causal gold (`prequential-v2` only), pos-weight schedule 8:24, 16 trunk +
6 graft epochs, save/target weights 0.5/0.25 — one run each, no sweeps.

## Metrics, populations and win rule (frozen now)

Primary, on the v2 validation, paired game bootstrap (400 resamples, seed 16):

1. `first_action_exact` — all scored decisions.
2. `action_family_match` — purchases with a known next final.

A supported development win requires BOTH paired intervals to exclude zero in
the candidate's favor.

Guardrails (non-regression gates, same pairing): `ward_only_recall` and
`no_buy_death_recall` paired differences must not exclude zero in the
unfavorable direction. Zero unaffordable displayed options is mandatory.
Secondary reporting (no gate): any-of-three exact, basket precision/recall,
save-card precision per display tier, exactness by basket-size slice with
paired CIs (the crossover chart is the diagnostic this experiment exists to
move: qty>=2 exactness must not be traded away for qty=1 gains — report the
slice table alongside the primaries).

Alignment rule (September 12 lesson): any trained arm's objective must include
both primary behaviors (exact and family), and the guardrail behaviors must be
represented in training supervision or explicitly argued unaffected before
launch. An exact-only objective is not eligible.

## Stopping rules

- Stage 1 shows `budget_infeasible` + `other` >= 80% of misses: stop; the
  generator is not the binding constraint after all; report and re-plan.
- Stage 2 fails the win rule or a guardrail: retain the negative result,
  original candidate stays the reference, no re-tuning on this validation.
- At most ONE stage-2 comparison is authorized by this protocol. A second
  attempt needs a new owner-visible decision and accounts for the validation
  spend ledger.

## Budgets and mechanics

Hidden one-shot Windows scheduled tasks, isolated checkout if any pinned job
is running; full assert suite as the first stage; 8h wall / 10 GB child
memory / 40 GB disk reserve per run; stream JSONL, never load the corpus into
dicts; frozen plan SHA256 recorded at launch; independent verification of the
final report before any claim; every reported number carries metric, split,
n and report path.
