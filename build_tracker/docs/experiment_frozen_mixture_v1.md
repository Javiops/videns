# Pre-registration: frozen 50/50 probability mixture (v1)

2026-09-22. Owner's assignment, recorded before the run. Run dir:
`data/experiments/16.18-frozen-mixture-v1/`. No training. Internal
chronological TRAIN probe only — VAL and test are OUT OF SCOPE and are
not read by this experiment.

## Question

The two frozen models make DIFFERENT errors on the same decisions: on the
probe (58.781 decisions, 500 games) their first recommendation differs in
13.189 cases and their correctness differs in 5.813 (grouped-only correct
2.881, ward-free-only correct 2.932; audit
`data/reports/proposal_checker_review_20260922/input_and_catalogue_audit.json`).

Does a fixed, untuned 50/50 mixture of their normalized probabilities over
the shared main-purchase action space beat BOTH parents on ward-free
first_action_exact?

## Arms (all three decoded identically)

| arm | probability source |
|-----|--------------------|
| A (parent) | `econ_ft_ep8` + ward-marginalized logsumexp over the 7.937-basket catalogue -> 5.692 groups |
| B (parent) | `wardfree_ep16` over its 5.691-basket catalogue + SAVE |
| M (candidate) | `0.5 * P_A + 0.5 * P_B` |

## Design

1. **Alignment is by canonical multiset key, never by index.** The two
   action spaces contain the same 5.692 actions but order them
   differently (only 31 positions coincide: the grouped map numbers groups
   by first encounter in the full-catalogue frequency order; the ward-free
   catalogue sorts by post-stripping aggregate frequency). The script
   builds the explicit permutation from `basket_key(sorted(multiset))`,
   asserts both key sets are identical, asserts the permutation is a
   bijection (every target index hit exactly once), and RECORDS the number
   of coinciding positions as a cross-check against the audit's 31.
2. **Normalize before mixing.** `P_A = softmax` over the 5.692 grouped
   logsumexp scores (this is exactly the ward-marginalized full-catalogue
   posterior); `P_B = softmax` over the ward-free catalogue scores. Both
   are full distributions summing to 1 over the same action space before
   they are averaged. The mixture is over PROBABILITIES, not logits.
3. **Weight is fixed at 0.5/0.5, declared here, before the result.** No
   sweep is run and no weight is adjusted after seeing any number. No
   per-row model selection (e.g. by true basket size) — the reason this
   is a mixture and not a selector.
4. **Decode and legality unchanged** for all three arms: top-128 by the
   arm's probability, walk to 3 DISTINCT legal main purchases via
   `purchase_sequence(action, row, gold_est, dragon)`, group 0 = SAVE =
   "no main purchase", always legal.
5. **Reproduction check (integrity gate).** Arms A and B are re-decoded
   in this process and compared cell-by-cell against their persisted rows
   (`rows_grouped_econft_full.jsonl` sha 46ad5537…,
   `rows_wardfree_full.jsonl` sha e59a7bc7…). Mismatch counts are
   reported. A nonzero count invalidates the alignment, not just the
   mixture, and is published as such.

## Metrics

- **PRIMARY**: ward-free `first_action_exact`, M vs A and M vs B, paired
  and bootstrapped by match (`v1.paired_bootstrap`, seed 16; 400 draws as
  in every sibling report, plus a 2.000-draw CI for the primary).
- Secondary: `any_of_three_exact`; slices `single` / `multi` /
  `no_buy_death` / `ward_only`; size statistics; agreement counts
  (how many of the 13.189 disagreements the mixture resolves each way).
- Reported as costs even if the primary is positive: every slice where M
  loses to either parent.

## Declared expectations and caveats

- **The +4,99 pts figure is NOT a hurdle, an expectation, or a ceiling for
  this arm.** It belongs to an oracle that knows the truth and picks
  between the two first options. A probability mixture is a different
  object: it can lose cases both parents got right (by promoting a shared
  runner-up over each parent's separate winner) and it can win cases both
  parents got wrong (by promoting an option both ranked second). Both
  directions are counted and published.
- This probe was already used for model selection. A positive result here
  is a reason to pre-register a VAL check, not a promotion, and the
  existing VAL usage (one frozen config, once) may not be reused for it.
- Mixing two models does not calibrate either. The pinkward conditional
  that the product rule needs comes from A's within-group mass and is
  untouched and uncalibrated here.
- Online cost of running both models is not measured by this experiment.

---

## RESULT (2026-09-22 15:25, full probe, 206 s)

`data/experiments/16.18-frozen-mixture-v1/mixture_report_full.json`;
per-decision rows `rows_{A,B,M}_full.jsonl`; independent recomputation from
the rows alone in
`data/reports/frozen_mixture_review_20260922/mixture_independent_recompute.json`.

**PRIMARY MET against both parents.** Ward-free `first_action_exact`,
n=58.781, 500 games, paired bootstrap by match:

| arm | first | any-3 | vs A | vs B |
|-----|-------|-------|------|------|
| A grouped econ_ft_ep8 | 0.48487 | 0.63819 | — | −0.09 [−0.32, +0.17] |
| B wardfree_ep16 | 0.48574 | 0.63808 | — | — |
| **M mixture 50/50** | **0.49514** | **0.64519** | **+1.03 [+0.85, +1.19]** | **+0.94 [+0.78, +1.12]** |

2.000-draw CIs agree: M−A [+0.85, +1.20], M−B [+0.76, +1.12]. any-3:
+0.70 [+0.55, +0.84] and +0.71 [+0.56, +0.86].

**Alignment and integrity.** 5.692 actions; the permutation is a bijection
and leaves exactly **31** positions unchanged — an independent reproduction
of the reviewer's audit. Re-decoding A and B in-process reproduced their
persisted rows with **0 mismatches** in `pred_first`,
`first_action_exact` and `any_of_three_exact` (the row files are
byte-identical), and the recomputed A-vs-B reproduces the audit's
2.881/2.932 exactly.

**Movement.** M corrects 1.725 / breaks 1.121 vs A (net +604); corrects
1.828 / breaks 1.275 vs B (net +553). It is right in 68 cases where BOTH
parents are wrong, and wrong in **0** cases where both parents are right.
338 first recommendations match neither parent. First recommendations of A
and B differ in 13.189 rows, as audited.

**Costs (all slices, ward-free first_action_exact).** vs A: single +1.29
[+1.09, +1.50], multi +0.30 [−0.07, +0.64], no_buy_death +1.24 [−0.29,
+2.59], ward_only +0.97 [+0.20, +1.68] — no slice negative. vs B: single
+0.87 [+0.65, +1.09], multi +1.45 [+1.09, +1.81], no_buy_death −1.05
[−2.42, +0.29], ward_only −0.08 [−0.83, +0.62] — the two nominal losses are
the slices where A itself lost to B (−2.29 and −1.05); the mixture recovers
most of that deficit and neither CI excludes zero.

**Against the declared caveat.** The oracle union rate is 0.53475; the
mixture captures 20.6% of that gap. The oracle was never a hurdle or an
expectation, and this is not it.

**NOT a promotion.** The probe was already used for selection. Adopting
this needs a newly pre-registered VAL reading (the existing one-config VAL
usage may not be reused), plus the unmeasured online cost of running both
models and the unchanged fact that neither model's pinkward conditional is
calibrated.
