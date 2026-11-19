# Patch 16.19 benchmark — protocol v1 (owner's job 2, 2026-09-29; rules before any measurement)

Owner's decision after the null transfer result: keep M2, and "preparar una
evaluación limpia en 16.19 antes de otro entrenamiento largo": qualify 16.19
under the current pipeline, fix a temporal cohort, measure the frozen
references on it under verified 16.19 economics, and only then decide any
training from the admitted counts and the references' performance. **The bar
for 16.19 is what M2 achieves on the same 16.19 games, not 51.53 %.**

## 1. Patch day (done 2026-09-28)

Official schedule: 16.19 on Wednesday 2026-09-23 (Pacific); Data Dragon latest
16.19.1. `app/config.py`: start lower bound 2026-09-22 UTC (DTO decides),
`PATCH_DATA_VERSIONS["16.19"] = "16.19.1"`, `CURRENT_INGEST_PATCH = "16.19"`;
`daily_pull.py` now collects `16.18,16.19`. Static review 16.18.1 → 16.19.1
(`prepare_reconstructible_1619.static_evidence`, 65 checked ids: boots,
support line 3865–3877, wards 2055/772043, elixirs, rune 8304): **rune 8304
unchanged, 0 mechanical field differences**, 2 new items (773161 Moonflair
Spellblade, 773162 Cloak and Dagger), 0 price changes on common items.
Patch-day tests 5/5.

## 2. Reconstruction chain (derived, ledgered)

`scripts/{reconstructible_1619_runtime, viego_source_gate_1619,
prepare_reconstructible_1619, process_reconstructible_1619,
run_reconstructible_1619(.ps1), verify_reconstructible_1619_evidence}.py`,
derived from the reviewed 16.18 chain by global patch-identifier replacements
(counts recorded) plus asserted single-occurrence edits; ledger
`data/acceptance/16.19_reconstructible_20260929/derivation_ledger.json`.
Labels: `decision-start-footwear-16.19-v1`, `footwear-timer-inventory-16.19-v1`,
runtime `reconstructible-16.19-v1`, economics `footwear-economics-16.19-v1`,
recovery `footwear-viego-16.19-v1`; the runtime loads the same three reviewed
16.17 sources with the 16.19 guards, exactly as the 16.18 variant did. Holdout
scan shared with the 16.17 history chain (`history_1617.split_files`, 45 files,
3,112 reserved tests + 7,837 validation holdouts) at the snapshot, the build's
plan validation and the evidence verifier. Parent collection state inherited
from `16.18-reconstructible-v4/state.json` (no scheduled 16.19 sample exists).

## 3. Snapshot (2026-09-29 12:55 UTC) and the cohort rule

Database: **4,992 games of 16.19** (2026-09-23 04:42 → 2026-09-28 23:41 UTC);
**4,992 selected**, 0 excluded (no 16.19 game is a reserved test or a
historical holdout anywhere). Smoke snapshot of 100 games beside it.

**Temporal cohort (proposal to the owner, matching their own count):** forward
boundary **2026-09-27T00:00:00Z**; games created at or before it train (3,384
raw), later ones form the cohort (1,608 raw, the two most recent days). Same
`forward-validation-v1` rule as v4 (validation strictly after every training
game; reserved tests never materialised). Counts after qualification will be
smaller; the boundary is declared before any qualification result is read.

## 4. Measuring the frozen references on the 16.19 cohort (rules)

- R1 (new Bf, `bf_newgen_ep16_full.pt`) and R2 = M2 (`mixture-v4-m2-v1`) are
  scored **frozen**: their own 16.18 vocabulary, catalogue and group space
  (never rebuilt from the 16.19 generation), the grouped decoder, legality at
  `gold_est`, ward-free first-action exact; paired-by-game bootstrap for M2 − R1.
- **Economics world for scoring:** the rows are reconstructed under the 16.19
  runtime (16.19.1 static data), and the frozen models featurise and check
  legality with **their own pinned 16.18.1 Data Dragon** (a frozen model is a
  frozen world; the static review found no mechanical difference among the
  checked ids and no price change among common items). Rows whose inventory or
  label carries an item unknown to 16.18.1 (the two new items) are counted and
  reported as a measured cost, never silently dropped or remapped.
- Gates: both references must reproduce their persisted v4-cohort rows 0/0/0 in
  the scoring process before scoring 16.19 (proves the decoder and legality are
  the frozen ones); the 16.19 generation must pass `verify_generation`.
- Published: exact and top-3 for R1 and M2 on the 16.19 cohort with per-game
  intervals, per-slice (single/multi/no_buy_death/ward_only), the count of rows
  with unknown items, and the admitted-count table. No promotion, no training
  decision inside this document.

## 5. What is pending

Reconstruction + full audit of the 4,992 sources (~1.5 h), admission with the
declared boundary, coverage, publication as a generation, the 16.19 static
economics check against the 16.18 package (`scripts/test_shop_econ.py` +
package legality on 16.19 rows), then the scoring. Training on 16.19 is not
part of this protocol.

## 6. Execution log (2026-09-29)

- 12:55 smoke snapshot (100 sources) + full snapshot (4,992 sources, 0 excluded); static review clean.
- 14:59–15:01 smoke reconstruction: 100 → **85 qualified**, 10,203 admitted rows, audit 0/0, 77 s.
- 15:02 shared modules generalised (`SCOPED_DEVELOPMENT` in `app/dataset_generations.py`;
  `report_corpus_coverage`, `publish_patch_dataset`, `corpus_rows.load_qualified`,
  `footwear_runtime` dispatch); `benchmark_1619_runtime/entry`, `prepare_benchmark_1619`,
  `prepare_development_1619` derived (ledger section `admission_helpers_from_1618`);
  `test_benchmark_1619_admission` passes and the 16.18 twin is unchanged (suite 57/59,
  the two known stale-path failures).
- 15:04 full reconstruction launched (`BuildTracker 16.19 Reconstructible Sources`).
- 15:06–15:12 smoke admission → coverage → publication (`data/dataset_generations_smoke/16.19-smoke-v1`,
  boundary 2026-09-23T10:00Z for the smoke only): train 37 games / 4,282 rows, val 48 / 5,921 rows,
  `verify_generation` OK, runtime `reconstructible-16.19-v1`, reconstruction
  `decision-start-footwear-16.19-v1`.
- Scoring tools: `scripts/score_frozen_on_generation.py` (R1 through `footwear_runtime`
  with the v4 generation as PREFIX_ML_DIR; M2 directly through the package) with the
  v4-prefix reproduction gate and the unknown-item accounting, and
  `scripts/pair_frozen_on_generation.py` (paired M2 − R1, slices, movement). Smoke runs
  on the smoke generation in progress.
- Consequence recorded: `corpus_rows.py` changed, so the 16.17 history export's reviewed
  row-builder digests no longer match live code; its validator will refuse it for new
  training runs until re-exported (deliberate; nothing planned needs it).
- 15:10–15:18 dress rehearsal of the scoring on the smoke generation (48 games / 5,921
  decisions of 2026-09-23, not the deliverable): R1 gate 0/0/0 on 2,000 v4 rows, M2 gate
  0/0/0; R1 50.04 %, M2 50.84 %, M2 − R1 +0.79 [+0.41, +1.24]; 0 rows with unknown items.
  The full chain (`scripts/run_benchmark_1619.ps1`: admission with boundary
  2026-09-27T00:00Z → coverage → publication `16.19-development-v1` → R1 → M2 → pairing)
  is registered to start when the reconstruction completes.
