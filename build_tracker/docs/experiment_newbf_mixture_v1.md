# Experiment: frozen 50/50 of the NEW Bf with the OLD frozen mixture (v1)

2026-09-25. Reviewer's step 2 after the v4 stage-1 result (relayed by the
owner): before spending a day of training on stage 2, test ONE frozen
combination of the new Bf (v4) with the previous mixture package. The old
mixture still gets 5,420 cohort decisions right that the new Bf misses
(oracle union 54.43% vs 50.74%). Inference only; no training; no weight
search.

## Question

Does averaging the new Bf's ward-free group probabilities with the old
package's probabilities, 50/50, raise ward-free first-exact on the v4 forward
cohort over the new Bf alone?

## Arms (all scored on EXACTLY the same 146,597 cohort decisions / 1,209 games)

- **new_bf** — `bf_newgen_ep16_full.pt` (sha256 c923d6ea…), v4 catalogue
  13,165 baskets / 9,333 ward-free groups, grouped logsumexp → softmax.
- **old_mixture** — package `mixture-50-50-v1` (identity 56e80a57…),
  `MixturePolicy.probabilities` over its 5,692 groups (features computed
  fresh, wallet `gold_est`).
- **M** — `0.5 · P_new + 0.5 · P_old`, weights declared here and never tuned.

## Alignment rule (declared)

Old groups → new groups by canonical ward-stripped basket key, never by
index. The reviewer verified the old 5,692 groups are a subset of the new
9,333; the script asserts every old group maps to exactly one new group
(injective) and that new-only groups receive probability 0 from the old
side. Group 0 = "no main purchase" on both sides.

## Decode (common to the three arms)

Walk the arm's ordering, best first, to 3 distinct legal main purchases at
`gold_est` (`purchase_sequence`), SAVE always legal — the same decode used in
stage 1 for all three arms.

## Integrity gate (must pass before the mixture counts)

Re-decoding new_bf and old_mixture in this process must reproduce their
stage-1 persisted rows (`rows_new_bf_full.jsonl`, `rows_mixture_full.jsonl`)
with **0 mismatches** in `pred_first`, `first_action_exact`,
`any_of_three_exact`. Any mismatch invalidates the run (the parents would
not be the models the reference was frozen on).

## Primaries (declared before running)

1. **M vs new_bf**, ward-free first_action_exact, paired by game, 95% CI
   (400 and 2,000 draws, seed 16). Reading: M replaces new_bf as the
   practical reference for stage 2 ONLY if the difference is positive with
   the CI excluding zero; otherwise new_bf stays and this is published as
   null/negative.
2. M vs old_mixture (same metric) — must be positive for M to be meaningful.

Secondary: top-3 (any_of_three), per-slice costs (single / multi /
no_buy_death / ward_only), movement (corrected / broken; both-parents-wrong
→ M right; both-right → M wrong), first-recommendation agreement, oracle
union, lazy legality calls. Informational: M vs old_bf from its persisted
rows (same cohort, same identity).

## Scope and exposure

The cohort is the generation's `val` split, already exposed by stage 1. This
is a single pre-declared configuration, not a search; the exposure is
recorded. No VAL/test of earlier generations is read. No promotion.

## Cost (extrapolated, to be measured)

Stage 1 scored three arms in 76 min, dominated by the package's economic
features (34 rows/s). Expected ≈ 80–90 min for the full cohort; smoke
(1,500 rows) first.

## Outputs

`data/experiments/16.18-newbf-mixture-v1/`: `rows_{new_bf,old_mixture,M}_{stamp}.jsonl`,
`mixture_report_{stamp}.json` (aggregates, paired, slices, movement,
reproduction mismatches, alignment facts, digests of both parents).

---

## RESULT (2026-09-25 23:10) — NULL: M does not beat the new Bf; new Bf stays the reference

Run 7,464 s (slowed by the BelowNormal/E-core scheduling trap, fixed mid-run;
computation unaffected). Integrity gate: both parents reproduced their
stage-1 rows with 0 mismatches. Alignment: 5,692 old groups ⊂ 9,333 new,
injective, 51 positions coincide by index.

| arm | first-exact | top-3 |
|---|---|---|
| M = 0.5·new Bf + 0.5·old mixture | 50.80% | 65.60% |
| new Bf (R1) | 50.74% | 65.60% |
| old mixture | 49.20% | 64.20% |

- **PRIMARY M vs new Bf: +0.061 [−0.049, +0.166]** (2,000 draws [−0.044,
  +0.164]); net +89 (3,181 corrected / 3,092 broken); top-3 +0.002
  [−0.085, +0.089]. **CI includes zero → not adopted.**
- M vs old mixture +1.60 [+1.48, +1.72] (necessary, not sufficient).
- Slices M vs new Bf: single +0.19 [+0.06, +0.31]; multi −0.16 [−0.39,
  +0.04]; no_buy_death −0.38 [−1.10, +0.33]; **ward-only −0.54 [−0.98,
  −0.11]** (a real cost).
- Movement: M recovers 3,054 of the old mixture's 5,420 exclusive hits but
  loses 3,092 of the new Bf's exclusive hits; both-parents-wrong → M right
  127; both-right → M wrong 0. M's first recommendation equals the new
  Bf's in 132,336/146,597 (90.3%); 799 novel.
- Verified independently from the persisted rows (identity identical
  across arms and with stage 1; aggregates, deltas and movement reproduce;
  CIs to bootstrap noise).

Reading: the complementarity that exists (oracle union 54.43%) is not
reachable by averaging probabilities — the same conclusion the learned gate
reached on the old parents: the parents' confidences do not say which one
is right. Published as null; **R2 = R1 = new Bf** for stage 2.
