# Data expansion plan v1 — inventory, temporal cut, staged training

2026-09-23 ~11:45. Reviewer's technical go-ahead for a new snapshot and
qualification, under three conditions (count what is really available;
preserve temporal validity, not just IDs; train in stages). This document
is the NEW plan the continuation protocol requires: the historical
snapshot's `training_or_export_authorized: false` is untouched, and the
authorization and scope for the expansion must be granted here, by the
owner, before anything is qualified or trained.

**Why**: the learning curve on the existing FIT games is steep by the
pre-registered rule — 500 → 1,000 → 2,000 games gave 42.64% → 45.92% →
48.26% ward-free first-exact (+3.28, +2.34; ratio 0.72). Read correctly:
THIS recipe benefits from more data and the training that comes with it
(16 fixed epochs also mean more optimizer updates). It does not by itself
prove the cause of every negative result this week, and it does not
guarantee another two points from expanding. No projection beyond 2,000
games is made anywhere in this plan.

## 1. Inventory — verifiable, metadata only

Source: `data/tracker.db` `games` table + the v3 generation's JSONs. No
payload was read; no test payload was opened. Files:
`data/reports/expansion_inventory_20260923/{inventory.json,
temporal_cut.json, new_queue420_ids.json}` (ID list digest recorded).

| class (patch 16.18, all queue 420) | games |
|---|---|
| v3 TRAIN (fit 2,000 + probe 500) | 2,500 |
| v3 VAL | 1,601 |
| v3 quarantined / temporal-inventory exclusions | 1,180 |
| v3 reserved test IDs present in `games` | 149 (of 3,112 reserved; the rest are metadata-only) |
| prior generations only (reusable evidence) | 10 |
| **NEW — never in any snapshot** | **7,442** |
| total in DB | 12,882 |

The 7,442 are new by construction: no prior qualification evidence exists
for them, so nothing is double-counted and nothing is reused wrongly. The
10 prior-generation games carry reusable evidence and are the only
deduplication case. The earlier "~4,500 admissible" figure is WITHDRAWN as
a planning number: the 78% yield of the v3 snapshot mixes qualification
with exposure-based exclusions, and this plan budgets on counted games,
with the qualification yield to be measured, not assumed.

## 2. Temporal cut — the condition that actually binds

Game-creation windows (UTC): TRAIN 09-09 → 09-13 12:55; VAL 09-13 12:59 →
09-14; NEW 09-09 → 09-21, with most mass on 09-15/16/18/20. **6,126 of the
7,442 new games were played after VAL's last game.** Per the continuation
protocol, every scored validation game must follow every selected training
game, so those cannot join TRAIN while the current VAL remains a forward
evaluation.

| design | joins TRAIN | evaluation population | what stays comparable |
|---|---|---|---|
| **A · keep current VAL** | only the **892** new games created ≤ last-TRAIN ms | current VAL (1,601) and current probe | everything measured so far; but +892 games is a 45% growth, not a doubling |
| **B · forward cohort** | 5,976 new games played before 09-20 + existing 2,500 = **8,476** before qualification | NEW forward validation cohort **09-20 → 09-21, 1,466 games** (≈ current VAL's size); old VAL 1,601 kept OUT of training; 3,112 reserved test untouched | nothing from the old probe/VAL — every arm (old parent, new Bf, frozen mixture) is re-scored on the new cohort |

**Recommendation: B.** A alone cannot test the hypothesis the curve
raised — it adds less than one doubling — while B is the design the
protocol already contemplates ("reserve the old holdouts and declare a
forward development validation cohort"). Its cost is stated plainly: the
old probe/VAL figures stop being the yardstick; the new cohort becomes it,
and the frozen mixture must be re-scored there before any comparison.

Ties: 0 at the TRAIN/VAL boundary; the same tie-together rule is applied
at the new cohort boundary (09-20 00:00 UTC) when the split is materialised.

## 3. Staged training — measure early whether it pays

Stage 0 — snapshot + qualification of the 7,442 new games through the
existing isolated pipeline (`prepare_reconstructible_1618.py` /
`qualify_patch_corpus.py` / `run_reconstructible_1618_v2.ps1`), retaining
quantities and every exclusion reason; new immutable generation with
coverage, acceptance, source membership, split version and export
fingerprint; prior VAL and test IDs excluded/reserved; temporal checks
recorded. Measured cost of the v3 run: 5,281 games in 3,605 s.

Stage 1 — **Bf d64 only**, the fixed phase-2 recipe, on the new FIT, with
a pilot gate projecting the run before it starts (the recipe's epoch is
~430 s at 2,000 games; the projection is measured, not extrapolated from
this sentence). Compared, on the new cohort, against (i) the old parent
`Bf_ep16` and (ii) the frozen 50/50 mixture — all three re-scored on the
same population with the same grouped decoder. PRIMARY: new Bf vs old Bf;
practical reference: new Bf vs frozen mixture.

Stage 2 — only if stage 1 is positive: the economic cache, twin econ
fine-tune and ward-free partner on the new FIT, then a new mixture. Not
promised in advance, and not budgeted as "a night of GPU" until stage 1's
pilot has measured the epoch.

Test stays sealed. Nothing here is a promotion.

## 4. Gates before any training

- New generation published and immutable; `training_or_export_authorized`
  granted explicitly for it in its own manifest.
- Split membership: prior VAL IDs absent from training; reserved test IDs
  reserved; every cohort game later than every training game (ties
  together); membership digests recorded.
- Qualification yield measured and published, with exclusion reasons.
- Stage-1 pilot projection below the declared budget, else abort.

## 5. Corrections carried from the review

The 13,668 probe purchases excluded from the "legal" denominator are not
all budget-blocked: **13,209 are blocked by budget and 459 by structural
constraints.** The 63.38% / 85.73% figures describe the 41,579 main
purchases feasible at the ESTIMATED wallet, not live performance with an
exact wallet.

---

## AUTHORIZATION (owner, 2026-09-23 ~12:00) — design B

Authorized, without confirmation between stages: new snapshot, qualification,
export, and the FIRST training (Bf d64 only) per this plan. Recorded here;
the historical snapshot's `training_or_export_authorized: false` is NOT
altered. Conditions, binding:

1. Current generation and current mixture stay intact.
2. Real game dates verified. Excluded from TRAIN: old VAL, exposure
   quarantines, all reserved test IDs. The new cohort is later than ALL
   training games.
3. 8,476 / 1,466 are pre-qualification counts; the admitted numbers and
   every exclusion are published.
4. Bf d64 only, recipe and final checkpoint fixed in advance (phase-2 Bf,
   16 epochs, ep16 is the result); cost measured by a pilot; no retrospective
   epoch choice.
5. New candidate, old Bf_ep16 and the frozen mixture scored on EXACTLY the
   same cohort games. PRIMARY ward-free first_action_exact; secondary top-3;
   paired by game with CIs and per-segment costs.
6. Continuation criterion fixed BEFORE looking (below).

**Continuation criterion (declared now):** the economic + ward-free chain is
run only if new Bf vs old Bf on the cohort is positive with the 95% CI
excluding zero AND the point estimate is at least +1.0 point of ward-free
first-exact. Anything less is published and the chain is not run.

Old probe/VAL become historical references once invalid for this
generation. Test stays sealed. A null or negative result is published.

---

## CORRECTED INVENTORY (2026-09-23 ~12:40) — before launch

The first inventory classified as NEW every game absent from the v3
assignment/qualification sets. That missed the games the v3 snapshot DID
select but that failed reconstruction or audit. Recomputed against the v3
`source_plan.matches` (6,541 selected):

| class (patch 16.18, queue 420) | games |
|---|---|
| v3 admitted (TRAIN + VAL) | 4,101 |
| v3 quarantined at admission | 1,180 |
| v3 snapshotted, FAILED build/audit (played 09-09..09-14) | **1,260** |
| reserved test present in `games` | 149 |
| **NEW, never snapshotted** | **6,192** (not 7,442) |

Of the 6,192 new games, 4,726 were played before the boundary and 1,466 on
09-20/21. The v4 snapshot selects everything non-reserved with a raw pair
(~12,733 games): the 1,260 build/audit failures and the 1,180 quarantines
are RE-ATTEMPTED under the same rules, not reused.

**Dry run of the forward split with the explicit boundary
2026-09-20T00:00:00Z**, upper bound (every new game admitted): TRAIN
**7,226** (2,500 old TRAIN + 4,726 new), cohort **1,466**, old VAL 1,601
excluded as `prior_validation_exposed`, reserved test never reached. Last
training game 2026-09-19 23:59:17 UTC, first cohort game 2026-09-20
00:04:14 UTC; `validate_1618_split` passes. The earlier "8,476" is
withdrawn; 7,226 is the pre-qualification ceiling.

Pipeline changes made for this run, all additive with defaults untouched:
`prepare_development_1618_v2.py` gains `--forward-boundary-utc`
(digest-bound to this document), `--authorization` and
`--authorization-document`; `validate_1618_admission` accepts one new
literal: *owner authorized design B expansion (snapshot, qualification,
export, Bf d64) on 2026-09-23*. Historical literals and the prior-snapshot
boundary rule are unchanged.

Chain: `scripts/run_expansion_v4.py` (snapshot → recon → verify →
admission → coverage → export → tests [non-fatal: known pre-existing
stale-path failure] → Bf d64 stage 1 + cohort scoring). Cost measured
inside the run: reconstruction wall 6 h max; training pilot gate 10 h max.

**Restart note (16:44):** attempt 1 stopped at 16:35, 26 minutes into
reconstruction, to apply the reviewer's correction: the tests stage was
non-fatal for ANY failure; it is now fatal except for the single identified
failure (stale old-root `FileNotFoundError` in two named test files), gate
validated on four cases including "same exception on the new root" -> abort.
The 16:09 snapshot is reused unchanged; reconstruction restarts from zero.

**Restart note B (20:37; relaunch pending disk):** attempt 2 completed the
build (16:44 -> 18:36; 12,733 selected, 10,344 accepted, 1,249,675 shop visits)
and failed the semantic audit (18:36 -> 20:37) on ONE hard failure,
`decision_state_binding`, KR_8388382904 (a cohort game, 2026-09-20 09:50 UTC),
ts 1368998. Root cause, verified on the raw timeline and the stored event: Zed
sold all six items (5,255 g) before buying 6676; the reconstruction wrote
`decision_inventory = []` — the correct state — and the audit's
`payload or inventory_before` fallback read the empty list as a missing field,
substituted the pre-sale inventory and failed correct data. Data right, gate
wrong: the second occurrence on this invariant (v3 precedent 012e562). Fix:
the rule now lives in `decision_state_binding()` ("[] is a state; only an
absent field falls back"), the identical fallback in `scripts/baseline.py` is
corrected, failures carry both inventories, and
`scripts/test_decision_state_binding.py` pins 8 cases. The invariant itself is
unchanged (a freed slot must have been paid for). Checked on the quarantined
DB: exactly 1 empty-decision-with-sales visit in 1,249,675; 0 empty decision
inventories with a non-empty pre-visit inventory and no sales; every shop
event of the game passes the corrected rule. The corpus manifest pins the
audit script and baseline.py, so the fix costs a full rebuild (3 h 53 min
measured today). Quarantined, not deleted: corpus ->
`16.18-reconstructible-v4_failed_gate_b` (with `quarantine_note.json` and the
game's 112 stored events), acceptance `run_failed_gate_b/`,
`semantic_audit_failed_gate_b.json`, chain `status_attempt2_failed_gate_b_2037.json`
and `recon_attempt2_failed_gate_b.log`. The 16:09 snapshot is reused
unchanged; reconstruction restarts from zero. Blocking: 69.7 GB free, the
rebuild writes ~28.7 GB and the chain holds a 40 GB reserve, so ~15 GB must be
freed first — which files to delete is the owner's decision (candidates: the
quarantined, reproducible `tracker.db` 27.7 GB; the attempt-1 `.partial`
7.5 GB).

**Relaunch (24 Sep 09:13, attempt 3):** the owner authorized removing the
quarantined, reproducible `tracker.db` (27.7 GB) and the attempt-1 `.partial`
(7.5 GB) — 97.9 GB free afterwards; everything else in the quarantine stays.
Full test suite after the fix: 56 pass + the two identified stale-path
failures, gate rule satisfied. Chain relaunched from `recon` via
`resume_from.txt`; snapshot 16:09 unchanged; corpus rebuilt from zero under
the corrected audit.

**Attempt 3 (24 Sep 09:13 -> 12:35): seven stages passed; the pilot gate
aborted the training.** Reconstruction 09:13 -> 11:20 (build 1 h 03, audit
1 h 04; corrected audit: 10,344 games, 1,249,675 shop visits, 0 hard
failures). Admission under the explicit boundary: 7,563 quality-accepted,
2,781 excluded as `prior_validation_exposed` (old VAL 1,601 + the 1,180 v3
exposure quarantines); **TRAIN 6,354 games / 757,975 visits (09-09 23:25 ->
09-19 23:59:17 UTC), cohort 1,209 games / 146,597 visits (09-20 00:04:14 ->
09-21 04:46:46 UTC)**, cohort later than every training game (verified on
the corpus DB), reserved test 3,112 never materialised. Source exclusion
reasons at reconstruction (not mutually exclusive): undo_without_item_ids
1,129, unreconciled_terminal_inventory 974, unexplained_viego_source_state
957, footwear_source_qa 510, duration_below_600s 165, others < 40 each.
Tests: 56 pass + the two identified stale-path failures (gate rule OK).
Export: generation `16.18-reconstructible-development-v4`, fingerprint
6c78398c…, `promotion_allowed: false`. Stage 1 setup: FIT cache 757,975
rows (1,862 s), catalogue 13,165 baskets / 9,333 groups. **Pilot: 0.81 s per
effective batch -> 10.60 h projected > 10 h cap -> abort, as declared.**
Reading: the pilot timed only its first 6 effective batches on a fresh CUDA
context (kernel loading, cuBLAS/cuDNN set-up, allocator growth), so the
projection is inflated by one-off costs; the catalogue is also 1.7x the v3
one (13,165 vs 7,937), a real cost of the fixed recipe. Measurement fix
(not a recipe change): two untimed warm-up batches, then 12 timed. The
recipe, epochs, checkpoint rule and the 10 h cap are unchanged; the FIT
cache and targets are reused; the training re-seeds after the pilot, so
the pilot length does not touch the run. Relaunched from `train_score`
(attempt 4). If the steady-state projection still exceeds 10 h, the chain
aborts again and the cap becomes the owner's decision.

---

## RESULT (2026-09-24 22:30) — stage 1 POSITIVE; continuation criterion MET

Attempt 4 (relaunched 13:25 from `train_score`, cache reused): steady-state
pilot 0.594 s/effective batch -> 7.82 h projected; measured 16 x 1,734 s =
7.71 h (13:32 -> 21:14); three arms scored on the cohort in 76 min.

Cohort 146,597 decisions / 1,209 games (== generation `val`), ward-free
first_action_exact, grouped decoder, paired by game (seed 16, 2,000 draws):

| arm | first-exact | top-3 |
|---|---|---|
| new Bf d64 ep16 (v4 TRAIN 6,354 games) | **50.74%** | **65.60%** |
| frozen 50/50 mixture (package 56e80a57...) | 49.20% | 64.20% |
| old Bf_ep16 (package parent) | 48.17% | 63.44% |

- **new Bf vs old Bf (PRIMARY): +2.57 [+2.40, +2.74]**, net +3,765
  (9,281 corrected / 5,516 broken); top-3 +2.16 [+2.03, +2.29].
- **new Bf vs frozen mixture: +1.54 [+1.38, +1.69]**, net +2,257
  (7,677 / 5,420); top-3 +1.41 [+1.29, +1.51].
- mixture vs old Bf on this third population: +1.03 [+0.92, +1.15].
- Slices new vs mixture: single +1.46 [+1.28, +1.65]; multi +1.66
  [+1.33, +2.02]; no_buy_death +2.20 [+1.17, +3.40]; ward-only +1.85
  [+1.27, +2.51] -- no negative slice. Oracle union new|mixture 54.43%.
- Verified independently from the persisted rows (identity identical across
  arms, 0 duplicates; every aggregate/delta/movement/slice reproduces).

**Criterion declared above: new vs old >= +1.0 AND CI excludes 0 -> MET
(+2.57 [+2.40, +2.74]).** `run_econ_wardfree_chain: true`. Stage 2 is NOT
launched by this document: the authorization block covers stage 1 only.

Limits: volume and recency entangled (new Bf trained up to 09-19, old
models up to 09-13; no same-volume older-window control); catalogue
13,165 vs 7,937 -- coverage share of the gain not decomposed (fixed-action
slices SAVE +2.39 / ward-only +1.17 vs old show the model itself improved);
estimated wallet; cohort now exposed; ep16 fixed with FIT loss still
descending (no extension); checkpoint, not a package -- nothing promoted.
Full write-up: `docs/expansion_result_v4_20260924.md`; report
`data/experiments/16.18-reconstructible-development-v4-bf-stage1/bf_stage1_report_full.json`
(checkpoint sha256 c923d6ea...).

**Reviewer's read (2026-09-25) and next steps adopted:** independent
catalogue-support decomposition (`data/reports/v4_external_review_20260925/`):
of the +2,257 net decisions vs the old mixture, +2,049 are on main purchases
present in BOTH catalogues, +167 on no-main-purchase decisions, +41 on
truths only the new catalogue supports -> 98.18% of the net gain is on
answers the old mixture could already give (accounting by truth support,
not a causal ablation: the gain is not mainly explained by hitting new
catalogue actions, but the other effects a larger catalogue may have on the
ranking were not isolated). Correction: the mixture's components were trained
on 2,000 FIT games (the 500 probe games were never trained on). Steps:
(1) the new Bf is FROZEN as the accuracy reference
(`frozen_reference_v4.json`; candidates must beat 50.737% on this cohort
and be compared with the old mixture); (2) a single frozen 50/50 of the new
Bf with the old mixture, pre-registered in
`docs/experiment_newbf_mixture_v1.md`, launched 2026-09-25 21:05; (3) the
economic + ward-free chain afterwards, with its own pilots and budgets.

**Step 2 result (2026-09-25 23:10): NULL.** The frozen 50/50 of the new Bf
with the old mixture scores 50.80% vs 50.74% (+0.061 [−0.049, +0.166]);
not adopted; the new Bf remains the reference (R2 = R1). Details in
`docs/experiment_newbf_mixture_v1.md`. Stage 2 budget measured
(`docs/experiment_stage2_econ_wardfree_v1.md`): ≈ 15 h of machine within
every declared bound, plus one day of engineering; launch is the owner's
decision.
