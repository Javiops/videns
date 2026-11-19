# Experiment: data scale across patches — frozen 2026-09-14

Owner authorization: 2026-09-14 ~00:30 UTC+2, in session — arms "16.17-solo +
conjunto", two paired comparisons on the v2 validation, PC on overnight. This
document freezes the rules BEFORE any evaluation; the authorization threshold
lives here, in the parent protocol, not in a post-hoc addendum (lesson from
the basket-composition stage-2 review, 2026-09-13).

## Question (recorded)

Does training the fixed V2 recipe on more adjacent-patch data produce better
displayed-policy imitation on the frozen 16.18 v2 validation than the same
recipe trained on 16.18 data alone?

## Arms and artifacts (no retraining of A or control)

| Arm | Training data | Artifact | sha256 |
|---|---|---|---|
| control | 16.18 v2 train (279,748 rows) | `data/experiments/basket-composition-v1/candidate.pt` | `a9b40beb991c832a743e410889946c97895999c40be082dd97e3d47fa55d0e28` |
| A (16.17-solo) | 16.17-development-v2 train (~4.2x rows) | `data/experiments/16.17-development-v2/candidate.pt` | `13f5b9bf272b848061be8ac4734f97b18503dfb8e322e7b1c75267a63cfad0d7` |
| B (joint 16.17+16.18) | union corpus | **not yet buildable** | — |

Recipe identity verified field-by-field between both existing run plans:
d128/4L/ff256/8H, runes, GOLDX `prequential-v2`, extras, history off, POSW
8:24, 16-epoch trunk + 6-epoch frozen-trunk graft, seed 16. Only the training
corpus differs, which is the question.

Arm B requires a joint dataset generation (publisher + a `patch_development`
purpose decision in `app/dataset_generations.py`) and launches only under its
own dated addendum after the owner reviews that design. Nothing in this file
authorizes it.

**Disclosed grade of arm A (verified in the artifact blob):** the 16.17
corpus is qualified as `qualified_subset` / `isolated_experiment_only`, and
the artifact self-declares `probe_only` provenance. Its training rows were
built by an earlier reconstruction generation
(`decision-start-footwear-v3`, inventory `footwear-timer-inventory-
elixirs-prototype-v1`) than the control's
(`decision-start-footwear-16.18-v1`, `footwear-timer-inventory-16.18-v1`) —
so arm A differs from the control in training-data RECONSTRUCTION SEMANTICS
as well as in patch and volume. A win or loss here is directional
development evidence about the data-scale question, at probe grade; it
authorizes nothing beyond informing the arm-B design, and the confound is
one reason arm B (joint, single current pipeline) remains the decisive test.

## Evaluation contract (arm A, authorized tonight)

- Split: `16.18-reconstructible-development-v2` validation, ALL games, export
  fingerprint `d70bfb2a8d71d312924004f7dace3cadcc213de27473313a8413ac94bb639bc2`.
- Evaluator: `scripts/eval_data_scale_v1.py` through
  `scripts/footwear_runtime.py --entry`. It reuses `eval_policy.score_visit`,
  `summarize`, `paired_diff` and `outcome_capture` verbatim — no reimplemented
  metrics. Paired game bootstrap 400 resamples, seed 16.
- **Static-data governance (predeclared):** economy, legality, display and
  family resolution for BOTH policies are governed by the 16.18 Data Dragon.
  Arm A is loaded with the recorded `dragon_override` waiver
  (`scripts/test_predictor_dragon_override.py` pins the contract: refusal
  without override unchanged, waiver recorded, footwear regime never waived,
  never serving-eligible). Disclosed bound: A's query features are computed
  under 16.18 static data, not its training patch's; if A loses, this
  divergence is one candidate explanation and the joint arm is the control
  for it.
- **Vocabulary bound (disclosed):** A cannot display any item absent from the
  16.17 label vocabulary; per-visit unseen-truth counts are captured in the
  paired outcome evidence.
- Provenance gates: both artifacts must carry `gold_input=prequential-v2`,
  `training_stage=graft`, and the exact `run_plan_sha256` of their declared
  training plans; artifact bytes must match the SHAs above.

## Win rule and guardrails (frozen now)

A supported development win for an arm requires BOTH paired intervals —
`first_action_exact` (all scored visits) and `action_family_match` (purchases
with a known next final) — to exclude zero in the arm's favor.

Guardrails: `ward_only_recall` and `no_buy_death_recall` paired differences
must not exclude zero in the unfavorable direction; zero unaffordable
displayed options under 16.18 static data is mandatory for both policies.

The report MUST include the exactness-by-basket-size slice table (paired, per
size 1..5+) — omitting it was the stage-2 reporting defect.

## Validation spend and stopping rules

- Each executed paired comparison appends one line to
  `data/ml/v2_validation_selection_log.md`. This protocol authorizes at most
  TWO (arm A now; arm B only after its addendum). After the second line, this
  protocol is exhausted regardless of outcome.
- An arm that fails the win rule or a guardrail is a retained negative
  result; the 16.18-only candidate stays the reference; nothing is re-tuned
  on this validation.
- No test reads. No promotion — the production hold stands and neither
  artifact is serving-eligible from this experiment (the override arm never
  is, by construction).
- No comparison of these numbers with the retired v1 validation numbers, in
  either direction; different exports.
