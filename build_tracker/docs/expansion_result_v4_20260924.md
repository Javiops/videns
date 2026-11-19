# Expansion v4 — stage-1 result (design B, forward cohort 09-20/21)

2026-09-24 22:30 (local). Owner GO 2026-09-23 ~12:00 (`docs/expansion_plan_v1.md`,
authorization block). Chain `scripts/run_expansion_v4.py`; result
`data/experiments/16.18-reconstructible-development-v4-bf-stage1/bf_stage1_report_full.json`
(`finished_utc` 2026-09-24T20:30:43Z). Nothing promoted; test sealed (3,112
reserved IDs never materialised, `final_test_payload_opened: false`).

## Headline (metric: ward-free first_action_exact, grouped decoder, same for all arms)

Cohort: **146,597 decisions / 1,209 games**, played 2026-09-20 00:04 → 09-21
04:46 UTC — later than every training game of every arm (verified on the
corpus DB; equal to the generation's `val` split).

| arm | trained on | first-exact | top-3 (n=138,073) |
|---|---|---|---|
| **new Bf d64 (ep16, fixed)** | v4 TRAIN: 6,354 games / 757,975 visits, 09-09 → 09-19 | **50.74%** | **65.60%** |
| frozen 50/50 mixture (package `mixture-50-50-v1`, id 56e80a57…) | v3 FIT (2,000 games; the 500 probe games were never trained on), 09-09 → 09-13 | 49.20% | 64.20% |
| old Bf_ep16 (the package's domain parent) | v3 FIT (2,000 games) | 48.17% | 63.44% |

Paired by game (bootstrap seed 16, 2,000 draws), points of first-exact:

| contrast | Δ first-exact | 95% CI | net decisions (corrected − broken) | Δ top-3 [CI] |
|---|---|---|---|---|
| **new Bf vs frozen mixture (practical reference)** | **+1.54** | [+1.38, +1.69] | **+2,257** (7,677 − 5,420) | +1.41 [+1.29, +1.51] |
| **new Bf vs old Bf (PRIMARY)** | **+2.57** | [+2.40, +2.74] | **+3,765** (9,281 − 5,516) | +2.16 [+2.03, +2.29] |
| mixture vs old Bf (third population for the mixture) | +1.03 | [+0.92, +1.15] | +1,508 | +0.76 [+0.67, +0.84] |

**Continuation criterion (declared before the run): met** — new vs old
+2.57 ≥ +1.0 with the CI excluding zero. `run_econ_wardfree_chain: true`.
Launching stage 2 remains the owner's decision (the GO covered stage 1).

## Per-segment costs (new Bf vs frozen mixture) — no negative slice

| slice | n | Δ points | 95% CI |
|---|---|---|---|
| single-item truth | 101,005 | +1.46 | [+1.28, +1.65] |
| multi-item truth | 37,068 | +1.66 | [+1.33, +2.02] |
| no_buy_death (SAVE) | 2,640 | +2.20 | [+1.17, +3.40] |
| ward-only | 5,884 | +1.85 | [+1.27, +2.51] |

vs old Bf: single +2.57, multi +2.79, SAVE +2.39 [+1.12, +3.57], ward-only
+1.17 [+0.49, +1.82]. Sizes (hits new/old): 1-item 58,585/55,985, 2-item
8,273/7,362, 3-item 614/495; underfill 15.07% vs 15.60%. Oracle union
new Bf ∪ mixture = 54.43% (complementarity persists: +3.7 over new Bf alone).

## Independent verification

Recomputed from the persisted per-decision rows (`rows_{new_bf,old_bf,mixture}_full.jsonl`,
146,597 each, identity (match_id, ts, champion) identical across the three
arms, 0 duplicates, game set == generation `val`): every aggregate, every
paired delta, every movement count and every slice above reproduce exactly;
CIs reproduce to bootstrap noise (e.g. new-vs-mixture [+1.387, +1.689] vs
report [+1.381, +1.692]).

## Cost, as measured

Reconstruction 09:13 → 11:20 (build 1 h 03, audit 1 h 04; 12,733 → 10,344
games, 1,249,675 visits, 0 hard failures under the corrected audit).
Admission → export → tests: 11:20 → 12:04. FIT cache 1,862 s (built once;
reload 396 s). Steady-state pilot 0.594 s/effective batch → 7.82 h projected;
**measured 16 × 1,734 s = 7.71 h** (13:32 → 21:14; projection within 1.4%).
Scoring three arms on the cohort: 76 min (34 rows/s, mixture package
options). Stage `train_score` end-to-end 9.09 h.

Admitted counts and exclusions: 7,563 quality-accepted; **2,781 excluded as
`prior_validation_exposed`** (old VAL 1,601 + the 1,180 v3 exposure
quarantines); source exclusions at reconstruction (not mutually exclusive):
undo_without_item_ids 1,129, unreconciled_terminal_inventory 974,
unexplained_viego_source_state 957, footwear_source_qa 510,
duration_below_600s 165, others < 40. TRAIN 6,354 of the plan's 7,226
ceiling; cohort 1,209 of 1,466.

## Two aborts on the way, both published

1. 2026-09-23 20:37 — audit hard failure on 1 visit of 1,249,675: a full
   sell-off (KR_8388382904, six opening sales, 5,255 g) written correctly as
   `decision_inventory = []`, misread by the gate's `payload or fallback`
   as a missing field. Data right, gate wrong (v3 precedent 012e562). Rule
   factored, twin fallback in `baseline.py` fixed, 8-case test added,
   corpus rebuilt (manifest pins both scripts).
2. 2026-09-24 12:35 — pilot gate abort at 10.60 h > 10 h: it timed its first
   6 effective batches on a cold CUDA context. Two untimed warm-up batches
   then 12 timed → 7.82 h; recipe, epochs, checkpoint rule and cap untouched.

## Limits (state them with the number)

- **Volume and recency are entangled.** The new Bf saw games up to 09-19; the
  old models' training ended 09-13. Part of the gain may be temporal
  proximity to the cohort rather than data volume. This is the product-
  relevant comparison (shipping models retrain on fresh data), but it is
  not a pure learning-curve point. A same-volume, older-window control was
  not run.
- **Catalogue coverage differs**: 13,165 baskets (v4) vs 7,937 (v3). Some
  cohort truths are reachable only by the new model. The gains on the
  fixed-action slices (SAVE +2.39, ward-only +1.17 vs old) show the model
  itself improved. Reviewer's decomposition by truth support (2026-09-25,
  `data/reports/v4_external_review_20260925/catalogue_support_decomposition.json`):
  of the +2,257 net vs the mixture, +2,049 on main purchases present in BOTH
  catalogues, +167 on no-main-purchase decisions, +41 on new-only truths
  (98.18% on old-supported truths). So the gain is NOT mainly explained by
  hitting new catalogue actions; the other effects a larger catalogue may
  have on the ranking are not isolated (no fixed-catalogue ablation).
- Estimated wallet (`prequential-v3`), same legality limits as all week; not
  live performance.
- The cohort is now exposed by this comparison; it cannot serve to select
  among several later candidates.
- FIT loss still descending at ep16 (1.7527 → 1.7408); ep16 fixed in advance,
  no extension (protocol). Training-log numbers are not harness numbers.
- new Bf is a checkpoint (`bf_newgen_ep16_full.pt`, sha256 c923d6ea…), not a
  package: promotion needs the package + displayed-policy report with zero
  unaffordable baskets, as for the mixture.

## What the numbers say for the product

On the forward cohort the retrained single model beats our best deployed
configuration (the frozen mixture) by **+1.54 points (≈ +2,257 net
decisions of 146,597)** and its own predecessor by **+2.57 (+3,765)**, with
every slice positive. The learning-curve hypothesis held at the next
doubling-and-a-half without projection. Continuing pays by the declared
criterion; the next step (stage 2: econ cache + twin econ fine-tune +
ward-free partner + new mixture on the new FIT, all on this cohort) is a
new owner decision with its own measured pilots.

Correction (reviewer, 2026-09-25): the mixture's components were trained on the
2,000 FIT games, not 2,500 — the other 500 were the internal probe. Result unchanged.

## Addendum (2026-09-25 23:10) — step 2 null, budget for stage 2 measured

Frozen 50/50 of the new Bf with the old mixture on the same cohort: 50.80%
vs 50.74%, +0.061 [−0.049, +0.166], net +89, ward-only −0.54 [−0.98, −0.11]
→ not adopted; the new Bf stays the accuracy reference (R2 = R1). Stage-2
pilot (measured, within bounds): econ cache 3.61 h CPU, twin econ
fine-tune 8.56 h GPU, ward-free partner 6.51 h GPU → ≈ 15 h machine plus
one day of engineering. Details: `docs/experiment_newbf_mixture_v1.md`,
`docs/experiment_stage2_econ_wardfree_v1.md`.
