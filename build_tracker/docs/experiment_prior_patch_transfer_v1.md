# Prior-patch transfer with target-patch adaptation — protocol v2 (rules closed; long trainings NOT launched)

2026-09-27. Reviewer's next investment after stage 2, with the corrections of
2026-09-27 12:00 applied. Owner's GO covers: preparing and qualifying 16.17
under the current pipeline, closing these rules, and measuring the cost of
the corrected contrast. The two long trainings stay pending until the design
and the budget are concrete. Antecedent: `docs/experiment_data_scale_v1.md`
(a 16.17-trained model on 16.18 improved the global rate but was rejected for
worsening SAVE; other architecture and reconstruction — not an answer).

## Question

On the v4 forward cohort (146,597 decisions / 1,209 games of 16.18 played
after every training game), does a Bf d64 that first learns on patch 16.17
and then adapts on the 16.18 FIT beat the target-only Bf given the SAME
updates and the SAME adaptation — first option, top-3, per-segment costs?
Three readings are kept, never one: H vs T (does the historical origin of
the data help), H vs R1 (does it beat the useful single model), H vs R2 = M2
(does it beat our best configuration). A deteriorated control could make
the first favourable without improving any reference.

## 1. Two symmetric phases (reviewer's correction 1)

Both arms train in two phases with identical mechanics; only the DATA of
phase 1 differs.

- **Phase 1 (U updates)** — T: U updates on the 16.18 FIT; H: the same U
  updates on the 16.17 FIT. Same recipe (full-catalogue CE, micro 32 /
  effective 256, Adam 1e-3, cosine over U), same seed, same effective batch
  size; each arm uses the catalogue of the data it sees.
- **Reset** — both arms re-initialise Adam and the scheduler.
- **Phase 2 (adaptation, identical)** — 16 epochs on the 16.18 FIT with R1's
  exact recipe (Adam 1e-3, cosine T_max 16), the v4 catalogue, and the SAME
  batch order (one generator, seed 16, consumed identically by both arms).
  ep16 is the result of each arm; no epoch choice.

So the restart is not confounded with the history: T restarts too.

## 2. Common vocabulary (reviewer's correction 2)

Champion index, item index (state encoder), label index and the inventory
columns feeding `query_proj` are indexed by vocabulary. Re-indexing only the
basket encoder would load valid weights with wrong meanings. Rule: a
**single vocabulary built exclusively from the two FITs** (16.17 FIT ∪
16.18 FIT) is used by both arms in both phases, for every indexed
structure: champ_index, item_index, label_index, and the basket encoder's
item table (the union of the items of both catalogues). Each phase keeps its
own ACTION catalogue (16.17 FIT truths for H's phase 1; the v4 catalogue of
13,165 baskets / 9,333 groups for phase 2 and for T's phase 1). Nothing is
re-indexed between phases; the tables simply contain both patches' IDs.
The cohort rows are featurised with the same common vocabulary; R1's rows
are the persisted stage-1 rows (its own vocabulary — the reference is the
persisted policy, not a re-featurised one).

## 3. Qualify first; then fix the budget in updates (reviewer's correction 3)

The 17,904 raw pairs do not license assuming a FIT like the current one,
and equal updates are not equal hours when catalogues differ. Order:

1. **Qualify 16.17 under the CURRENT pipeline** (reconstruction with the
   16.17.1 prices, recipes and items; the same gates: source screening,
   footwear QA, prefix check, decision-boundary mask, full semantic audit
   including `decision_state_binding`). **Before reconstructing**, exclude:
   every reserved test ID of any generation (fold `test` anywhere), and
   every historical holdout of BOTH patches — the 16.17 validations
   (`16.17-development-v1` val 73, `16.17-development-v2` val 731, any
   recovered-generation validation) and the 16.18 validations/cohort (by ID;
   they are 16.18 games, excluded by patch anyway). Publish the admitted
   counts and every exclusion, as for v4.
2. **Then fix U**: U = the updates of **four epochs of the 16.18 FIT** =
   4 × ceil(757,975 / 256) = **11,844 updates** (declared now, before any
   result). H spends those 11,844 updates on the 16.17 FIT (whatever its row
   count, cycling if smaller, truncating if larger, same batch size); T
   spends them on the 16.18 FIT. A negative result discards this dose only.
3. **Measure the cost** of the corrected contrast with pilots on the real
   FITs (2 untimed warm-up + 12 timed batches per phase and per arm): phase 1
   per-update cost depends on each catalogue's size; phase 2 is R1's
   measured 1,734 s/epoch. Bounds declared with the pilots before launch.

## 4. Primaries and reading (declared)

1. **H vs T**, ward-free first_action_exact on the cohort, paired by game,
   95% CI (400 + 2,000 draws, seed 16). Positive with the CI excluding zero
   = the historical origin of the data adds value at this dose.
2. **H vs R1** and **H vs R2 (M2)** — practical readings; progress for the
   product requires beating R2 with the CI excluding zero. All three are
   published; none alone declares progress.
3. Secondary: top-3; slices single / multi / no_buy_death / ward_only with
   CIs (SAVE is where the antecedent failed); movement counts; T vs R1 (does
   the extra phase-1 budget alone move the single model).
4. Integrity gates: R1 reproduces its stage-1 rows (0/0) in the scoring
   process; both arms' rows persisted; the 16.17 corpus audit passes with 0
   hard failures; the common vocabulary's digest recorded in both arms'
   reports.

## 5. Inventory (metadata only, 2026-09-27)

| class | count |
|---|---|
| patch 16.17 games in `games` (queue 420) | 20,859 (2026-08-25 23:49 → 2026-09-10 02:34 UTC) |
| with raw match + timeline pairs | 17,904 |
| historical 16.17 holdouts to exclude | val 73 (`16.17-development-v1`) + val 731 (`16.17-development-v2`) + any recovered validation |
| existing `16.17-development-v2` | old pipeline (reconstruction `decision-start-footwear-v3`, gold `prequential-v2`, prototype inventory) — NOT reusable |
| static data | Data Dragon 16.17.1 cached (item/champion/runes/summoner) |

## 6. Costs (anchors; pilots replace them)

Reconstruction: v4 rebuilt 12,733 games in 2 h 07 → ≈ 3 h for ≈ 17,000
games; admission/export ≈ 45 min; engineering to run the current gates
under the 16.17 runtime and static data: measured by doing it. Training:
phase 1 ≈ 11,844 updates ≈ 1.9 h on the 16.18 FIT (catalogue-dependent on
16.17); phase 2 = 16 × 1,734 s ≈ 7.7 h; ≈ 9.6 h per arm, ≈ 19–20 h GPU for
both, sequential. Nothing runs until the pilots and the owner's GO.

## 7. What it shows and what it does not

It separates "history helps" from "more updates help" (T spends the same
updates and restarts the same way). It does not separate temporal proximity
from volume (a farther patch would), and R1/R2 comparisons keep stage 1's
recency confound. The latency difference between packages (+3.7 ms) stays
descriptive: it does not isolate the larger catalogue's effect and does not
block this work.

## 8. Execution log (2026-09-27) — first snapshot withdrawn, review fixes, smoke chain

**Snapshot r1 (10:59) withdrawn.** It excluded 3,687 historical validation
holdouts and 2,955 reserved test IDs, but its scan covered only
`data/ml/split_assignment.json` and `data/dataset_generations/*/split_assignment.json`.
The 16.17 development validations (884 `val` games of `16.17-development-v1/v2`,
evaluated in `docs/experiment_data_scale_v1.md`) live only in
`preserved_split_assignment.json` / `split_plan.json` files (acceptance,
corpora and dataset_generations directories) → **153 evaluated 16.17
validation holdouts were inside the 14,217 selected sources**. Found by the
adversarial review (workflow `wf_4a33520c`, 33 agents, 11 findings
confirmed by two verifiers each). The reconstruction started from r1 at 11:04
was ended at 11:44 (owner condition 1: holdouts are excluded *before* any
payload is opened; the build is not repairable at export); its orphaned
supervisor failed by itself at 12:36 on its own disk guard ("40 GB disk
reserve reached") and its 25.2 GB partial corpus was deleted. Nothing of r1
was exported; its metadata is kept as `data/acceptance/16.17_history_20260927_superseded_r1/`.

**Fixes applied before any new payload (ledger: `derivation_ledger.json`,
section `review_fixes_20260927`)**:

| # | finding (severity) | fix |
|---|---|---|
| 6 | 153 holdouts in the selection; no gate could refuse them (blocker) | one shared scan `history_1617.split_files` (every `split_assignment` / `preserved_split_assignment` / `split_plan` under `data/`, both layouts) used by the snapshot, the build's `validate_plan`, the evidence verifier and the history validator; 45 files, 3,112 reserved + 7,837 val ids; the r1 plan is now refused by `validate_plan` |
| 1 | `validate_plan` checked fold `test` only (minor) | also the plan's holdout list, fold `val` in prior assignments, and the live scan |
| 7 | exporter built rows with live code not bound to the audited build (major) | before any row, every file of the completed run's `protocol.json` `code_sha256` must equal the live file; manifest records `protocol_sha256`; the validator cross-checks `row_builder_code` against the protocol |
| 8 | free-form `export_exclusions` (major) | reasons restricted to the three `corpus_rows.iter_games` can emit AND recomputed from the frozen corpus (duration, shop perspectives, admitted rows) for every excluded game |
| 9 | rows matched by count only (minor) | every row must be exactly one audited `admitted_decision_ids` entry of its game (set equality per game) |
| 10 | manifest re-hashed evidence (minor) | manifest carries the digests `verify_prerequisite` certified (`bindings`) |
| 11 | R1/R2 rows unbound (major) | trainer refuses unless `rows_new_bf_full.jsonl` and `rows_M2_mix_full.jsonl` match the digests in `frozen_reference_v4.json` / `frozen_reference_stage2.json`; digests + reference metrics recorded in the report |
| 12 | phase-2 RNG not reset (minor) | `torch.manual_seed(16)` again after the reset in both arms (same dropout masks, same order) |
| 14 | `catalogue_sha256` never checked (minor) | part of the cache `expect` for both FIT caches |
| 15 | CWD-relative reference path (minor) | ROOT-anchored |
| 3 | launcher absent from the ledger (minor) | parametrised (`-TaskName -Base -Corpus -MaxWallSeconds`, wall unchanged at 21,600 s) and ledgered; `register_history_1617.ps1` registers with read-back |

Refuted by the verifiers (no change): static-review basis, runtime not
forbidding a pre-imported `baseline`, two coexisting 16.17 declarations,
unused cache tensors on the GPU.

Validator proof: 22 synthetic rejection cases (partial rows, digest mismatch,
altered mask, infiltration through each split-file layout and through the
plan list, mixed versions, duplicates, row outside the audited ids, partial
state, foreign scope, changed row-builder code, export code ≠ build code,
relabelled partial export, free-form reason, silently dropped game) + 1
control accepted.

### 8.1 Smoke chain (300-game snapshot, both phases, 2026-09-27 11:59 → 12:16)

The whole contract, end to end, on a `--limit 300` snapshot taken with the
corrected scan (`data/acceptance/16.17_history_smoke_20260927`, 45 split
files scanned, 3,112 reserved + 7,837 val ids excluded before selection):

| step | result | time |
|---|---|---|
| build + full audit (`BuildTracker History 1617 smoke`) | 300 sources → **257 qualified** (footwear 228 / no-footwear 29); source exclusions: undo_without_item_ids 17, viego 15, unreconciled_terminal_inventory 15, duration 7, footwear QA 6, physical replay 2, slot overflow 2, invalid source 2; **31,131 admitted rows, 121 boundary-excluded**; audit passed, 0 hard failures, 0 quarantined | 4.6 min (job peak 86 MB) |
| export (`export_history_1617.py --allow-limited`) | evidence verified 6 s; live code = protocol code; 257 games / 31,131 rows / 0 export exclusions / single versions (recon `decision-start-footwear-v4`, inventory elixirs-prototype-v1, gold `prequential-v3`, slots `role-slots-v1`, patch 16.17); manifest + validator accepted (`data/history/16.17-history-smoke`, 124.5 MB) | 68 s |
| caches 16.17 then 16.18 (`prepare_transfer_caches.py --smoke`) | common vocabulary `156c2217…`: 173 champions, 205 items, 198 labels, 197 basket items (0 ids only in 16.17; 5 champions / 2 items / 2 labels only in 16.18); catalogues 16.18 **13,165** (= stage 1) / 16.17 smoke 2,778; cohort 1,500 rows; identical digest asserted on the second run | 379 s + 89 s (the 758k-row 16.18 first pass dominates) |
| trainer smoke (`basket_transfer_ab.py --smoke`, U=40, 1 epoch, both arms) | setup 9 s; R1/R2 rows bound to the frozen digests; both arms trained, reset, adapted, scored on 1,500 rows; every report field present (paired CIs, slices, movement, decision); checkpoints saved | 89 s |

The smoke accuracies (T 7.1%, H 7.9% on 1,500 rows after 40 updates + 1
epoch) are a harness check only and are **not measurements** of anything.

### 8.2 Budget (pilots, steady state: 2 untimed + 12 timed effective batches, RTX 8 GB)

`pilot_report_smoke.json` (`--pilot-only`, 12:16, while the withdrawn CPU
build was still running beside it):

| arm | phase 1 (11,844 updates) | phase 2 (16 × 2,961 batches) | total |
|---|---|---|---|
| T (16.18 catalogue 13,165) | 0.610 s/batch → **2.01 h** | 0.602 s/batch → **7.92 h** | **9.92 h** |
| H (16.17 catalogue) | 0.193 s/batch → 0.63 h with the **smoke** catalogue (2,778); the full 16.17 catalogue will be larger → re-pilot on the full history before launch | 0.604 s/batch → 7.94 h | ≥ 8.6 h |

The pilots inside the trainer smoke a minute earlier read 0.585 / 0.580 /
0.174 / 0.579 s per batch (9.56 h for T). Phase 2 reproduces R1's measured
epoch (1,734 s ≈ 0.586 s/batch). Both arms sequential ≈ **19–20 h GPU**
(H phase 1 to be re-measured with the full catalogue). Bounds to
declare with the full pilot: phase 1 ≤ 3 h per arm, phase 2 ≤ 9 h per arm
(same rule as stage 2: a pilot above its bound stops that segment).

### 8.3 Full snapshot r2: blocked on disk (owner's decision)

The reconstructed corpus costs ≈ 2.45 MiB per game (v4: 29,462 MiB for
12,733; r1 partial 16,916 MiB at 6,900, 25.2 GB when it died at ≈ 8,900)
→ **≈ 37 GB** for the ≈ 14,064 expected sources. The supervisor refuses
to run below **40 GB free** (40e9 bytes; bound inherited from the 16.18
chain, not changed). Free space after the r1 cleanup (12:40): **62.5 GiB
(67 GB)**; **≥ 77 GB (71.7 GiB) needed at launch** → ≈ 10 GB short. Candidates measured 12:04 (MB): `16.17-development-v2`
29,529 (old pipeline, not reusable), `16.18-reconstructible-v3_failed_gate_a`
15,015 (quarantined), `16.18-reconstructible-v3` 15,015, `-v2` 9,872, `-v1`
6,073, `16.18_role_slots_v1` 3,200, `16.18_repaired_v2` 2,961,
`16.18_initial` 2,495, `16.17-development-v1` 2,261, `-v4_failed_gate_b`
1,095. Nothing is deleted without the owner's word. Ready to run once
space exists: `prepare_reconstructible_1617.py` (r2, expected 14,064
sources) → `register_history_1617.ps1` (≈ 1.8 h build + ≈ 1.5 h audit) →
export → caches (full) → `--pilot-only` → owner GO for the two long arms.

### 8.4 Snapshot r2 and launch (owner GO, 2026-09-27 13:29)

The owner freed space by deleting only the quarantined `16.18-reconstructible-v3_failed_gate_a/tracker.db`
(14.1 GiB; metadata, audits, code and sources kept; record in `data/reports/cleanup_20260927/`).
Snapshot r2 (`data/acceptance/16.17_history_20260927`, 13:29 local): 20,859 games in the
database → **14,064 selected** (= r1 minus exactly the 153 holdouts): 3,840 historical
validation holdouts + 2,955 reserved test IDs excluded before any payload, 45 split files
digest-bound. Reconstruction task `BuildTracker History 1617` started 13:34 (wall 21,600 s);
`BuildTracker Transfer Prep` waits for its completion and runs export → caches → `--pilot-only`
(`scripts/run_transfer_prep.ps1`). The two long arms wait for that pilot report, the declared
bounds and the owner's GO.

### 8.5 Second adversarial review (workflow `wf_2703ca2a`, 32 agents, 13:35–14:00) — fixes before the full export

10 findings confirmed (two verifiers each), 4 refuted. Fixed before any full artefact is produced:

| finding | fix |
|---|---|
| `--arm H` alone crashed in the comparison (pair filter tested one side only) | pairs require both sides; single-arm runs report H vs R1/R2 |
| R1/R2 digest gate ran only after ~20 h of training | `bind_references` runs at setup, before the pilots; `--compare-only` recompares persisted `rows_{arm}` files without retraining |
| protocol §4.4 "R1 reproduces its stage-1 rows (0/0)" not executed | `r1_gate`: R1 re-instantiated with the stage-1 vocabulary, cohort decoded with the live decoder/legality/static data, compared with the frozen rows; any mismatch aborts (exit 2) before training |
| declared bounds not enforced in the full run | `BOUNDS_H = {phase1: 3 h, phase2: 9 h}` per arm; a pilot above its bound writes the report and stops (exit 3) |
| a `--limit` (smoke) export accepted for full caches/training | `prepare_transfer_caches` and `basket_transfer_ab` refuse a history whose snapshot carried `selection_limit` unless `--smoke` |
| "live code = build code" gate skipped the row builder itself (`corpus_rows.py`, `build_recovered_patch_corpus.py` are export-only code, outside the build protocol) | `REVIEWED_ROW_BUILDER` digests declared in the exporter (the reviewed builder the smoke chain ran with, plus `train_prefix.py`); a changed builder refuses to export until the declaration is updated deliberately; recorded in the manifest |
| prep launcher could wait forever on a dead supervisor | 8 h deadline; status.json must be rewritten within 15 min (the supervisor rewrites it every 10 s); guarded reads |
| trailing separator in `-Base/-Corpus` garbled the registered command | trimmed |
| row payload bound only to its own manifest (a consistent two-file forgery passes; refuted by one verifier, minor by the other) | after the reconstruction finishes (the validator is pinned build code while it runs): validator spot-checks admitted rows' labels against the frozen corpus database; `build_recovered_patch_corpus.py` added to the pinned row-builder set |

Refuted: 20 "probe-val" 16.17 games in r2 are sub-600 s remakes that never produced a row anywhere (the row builder excludes them: `duration_below_600s`); version-field homogeneity; `targets` cache not digest-bound (bound through row count + cache meta); GPU headroom (16.17 cache ≈ 1.6 GB + 16.18 0.82 GB fit the 8 GB card).

**Build r2 done (13:34 → 15:50, 2 h 16).** 14,064 sources → **11,562 accepted**
(no-Viego footwear 10,069 / no-Viego no-footwear 1,493), 2,502 excluded by the
source gates; **1,381,011 admitted decisions, 4,725 boundary-excluded**;
corpus `data/corpora/16.17-history-v1` (database 33.2 GB, ≈ 2.9 MB per accepted
game). Full semantic audit running (≈ 100 games/min → ≈ 17:45). Reasons of the
2,502 exclusions are published by the qualification after the audit.

**Audit r2 done (15:51 → 18:10; whole run 4 h 36, job peak 588 MB).** Full semantic
audit **passed: 0 hard failures, 0 quarantined**; qualification
`qualified_reconstruction_snapshot`: **11,562 games, 1,381,011 admitted decisions,
4,725 boundary-excluded**. Source exclusions (2,502): undo_without_item_ids 1,289,
unreconciled_terminal_inventory 785, unexplained_viego_source_state 749,
footwear_source_qa 419, duration_below_600s 268, physical_inventory_replay 56,
total_regular_slot_overflow 51, modeled_inventory_slot_overflow 34,
unsupported_or_invalid_source 30, future_prefix_input_dependency 1.

**Validator hardening, final form.** `history_1617.py` is pinned build code of the
completed runs (the exporter refuses to run when any pinned file drifts — it did
refuse when the hardened validator was installed over it), so the file was restored
to its pinned digest and the row-content check lives in `scripts/history_rows_check.py`,
called after `validate_history` by the exporter, the cache builder and the trainer:
rows of 300 sampled games checked field by field and label-wise against the frozen
corpus events (save sentinel 999999 handled), plus the exporter's
`reviewed_row_builder` digests re-checked live. Proof on the smoke export: 31,131
rows accepted in 5 s; a consistently re-hashed relabelling, a drifted builder digest
and an undeclared builder rejected (3/3); validator suite 18/18 + 4/4; smoke history
re-exported deterministically (rows digest `6af7c28a…` unchanged → smoke caches
still bound). Prep chain (`BuildTracker Transfer Prep`) launched 18:15: export →
caches (full) → `--pilot-only`.

### 8.6 Full history export (18:15 → 18:41)

`data/history/16.17-history-v1`: evidence verified in 426 s (33 GB database and
every raw pair re-hashed); **11,562 games → 11,562 exported, 0 export exclusions;
1,381,011 rows = the decision mask's admitted rows exactly** (5.52 GB, 953 s
through the shared `corpus_rows.iter_games` under the 16.17 runtime); single
versions (reconstruction `decision-start-footwear-v4`, inventory
`footwear-timer-inventory-elixirs-prototype-v1`, gold `prequential-v3`, slots
`role-slots-v1`, labels `purchase-net-v1`, patch 16.17); validator + row check
accepted it (36,389 sampled rows re-checked against the frozen corpus). Rows digest
`f085022312a1…`. Caches (common vocabulary from this FIT and the v4 16.18 FIT)
building next, then the full pilots.

### 8.7 Full caches (common vocabulary)

16.17 (18:41 → 19:34): history validated + rows re-checked in 196 s; common
vocabulary digest **`156c2217…` — identical to the smoke's**: 173 champions, 205
items, 198 labels, 197 basket items (every 16.17 id already exists in the 16.18
FIT; 1 champion only in 16.18); **16.17 catalogue 17,436 baskets** (16.18: 13,165)
from 1,381,011 FIT rows; `cache_fit_1617_full.pt` 1.50 GB (1.08 KB/row) + targets.
16.18 cache (v4 runtime) building next, then `--pilot-only`.

### 8.8 Full pilots — the budget (`pilot_report_full.json`, 20:28 → 20:36)

Setup 404 s (history re-validated + rows re-checked, both FIT caches and the
146,597-row cohort resident on the GPU, R1/R2 rows digest-bound). Steady state,
2 untimed + 12 timed effective batches per arm and phase:

| arm | phase 1 (11,844 updates) | phase 2 (16 × 2,961 batches) | total | bound check |
|---|---|---|---|---|
| T (16.18 catalogue 13,165) | 0.679 s/batch → **2.23 h** | 0.672 s/batch → **8.84 h** | **11.07 h** | 2.23 ≤ 3 ✓, 8.84 ≤ 9 ✓ |
| H (16.17 catalogue 17,436) | 0.863 s/batch → **2.84 h** | 0.674 s/batch → **8.87 h** | **11.71 h** | 2.84 ≤ 3 ✓, 8.87 ≤ 9 ✓ |

Both arms sequential ≈ **22.8 h of GPU** (+ R1 gate ≈ 5 min, scoring ≈ 15 min,
setup). Phase 2 measures 0.67 s/batch against R1's 0.586 s/batch epoch and the
smoke pilots' 0.60: 12–15 % slower with both full caches resident and the desktop
sharing the GPU; descriptive. Both phase-2 projections sit within 1.5 % of the
declared 9 h bound: if the machine is busier at launch, the in-run pilot stops the
run before training (exit 3) and the owner decides. The bounds stay as declared;
only the owner changes them. Launch = `register_task.ps1 -TaskName 'BuildTracker
Transfer Train' -ScriptName scripts\run_transfer_train.ps1` (refuses without the
full pilot report and history export). Nothing launched: owner GO pending.

### 8.9 Owner GO (2026-09-27 21:20) — bounds recorded before launch

GO to both long arms, chained without further confirmation (R1 reproduction →
pilots → T → H → evaluation). Bounds as the owner set them: **phase 1 ≤ 3 h per
arm; phase 2 ≤ 10 h per arm** (raised from the 9 h declared in §8.2 because the
8.84–8.87 h pilots left no operational margin). Recipe intact: 11,844 initial
updates, Adam + scheduler reset in both arms, 16 identical adaptation epochs; the
result is the **final checkpoint** (no epoch selection, no extension). A failed
gate or a pilot above its bound stops the segment and publishes the reason. The
~24 h remain a projection. The result must answer separately whether H beats the
equal-budget control, the current Bf (R1) and M2 (R2): beating a weakened control
does not count as a useful accuracy gain. Launched 21:2x as task `BuildTracker
Transfer Train` (`scripts/run_transfer_train.ps1`).

### 8.10 Launch log (2026-09-28)

00:13 task started (after a first start at 21:36 the day before was stopped in
setup at the owner's request, nothing trained). Setup 426 s. **R1 reproduction
gate: 0 / 0 / 0 mismatches (pred_first / first_action_exact / any_of_three_exact)
over the 146,597-row cohort**, 1,610 s. In-run pilots vs the owner's bounds
(3 h / 10 h per arm): T phase 1 0.658 s/batch → 2.16 h, phase 2 0.651 → 8.57 h;
H phase 1 0.842 → 2.77 h, phase 2 0.659 → 8.67 h; all within bounds → T phase 1
started 00:49. Projection: T done ≈ 11:35, H done ≈ 23:00, report ≈ 23:30
(2026-09-28). Progress files `transfer_progress_{T,H}_full.jsonl`.

**T scored (11:44).** Phase 1 11,844 updates in 7,725 s; 16 epochs at 1,928–1,948 s
each; final checkpoint `T_ep16_full.pt`, rows `rows_T_full.jsonl` (146,597).
First-action exact **50.744 %** vs R1 50.737 % — net **+10 decisions**, same first
action as R1 in 85.9 % of decisions (recomputed from the persisted rows; the paired
CI comes with the final report). Reading so far: the equal-budget control with
11,844 extra updates on the same 16.18 FIT adds nothing over R1, so any H − T
margin will be attributable to the 16.17 history rather than to more training.
H phase 1 started 11:44 (0.85 s/update → ≈ 14:35), then 16 epochs → ≈ 23:15.

## 9. RESULT (2026-09-29 09:25) — NULL for the history; H is not practical progress

`data/experiments/16.18-prior-patch-transfer-v1/transfer_report_full.json`, cohort
146,597 decisions / 1,209 games (v4 forward, exposed), ward-free
first_action_exact at gold_est, grouped decoder, paired-by-game bootstrap (400
draws; 2,000 for the primaries). Integrity: R1 reproduced its frozen rows 0/0/0
in the scoring process; R1/R2 rows digest-bound (`b8bdd145…`, `67d14d30…`);
recipe as declared (11,844 updates → reset → 16 identical epochs, same batch
order); final checkpoints (`T_ep16_full.pt` 02d94e2f…, `H_ep16_full.pt`
7597ffec…); every number below recomputed independently from the persisted rows
(`rows_T_full.jsonl` 3deb9184…, `rows_H_full.jsonl` 7b5076e8…).

| model | first-action exact | any-of-three |
|---|---|---|
| R1 (new Bf, frozen) | 50.737 % | 65.600 % |
| T (equal-budget control) | 50.744 % | 65.559 % |
| **H (16.17 history → 16.18)** | **50.854 %** | 65.616 % |
| R2 = M2 (practical reference) | 51.531 % | 66.087 % |

| reading (paired, points) | diff | 95 % CI | net decisions | verdict |
|---|---|---|---|---|
| **H vs T** (does the history help?) | **+0.111** | **[−0.028, +0.244]** (2,000 draws [−0.018, +0.244]) | +162 (5,282 corrected / 5,120 broken) | **null** |
| H vs R1 (does it beat the useful single model?) | +0.117 | [−0.015, +0.270] | +172 | null |
| H vs R2 = M2 (does it beat our best?) | −0.677 | [−0.818, −0.540] | −992 | **no** |
| T vs R1 (does more training alone help?) | +0.007 | [−0.117, +0.134] | +10 | null |
| H vs T any-of-three | +0.057 | [−0.047, +0.169] | — | null |

Slices (H vs T): single +0.147 [−0.002, +0.307] (n 101,005; borderline), multi
+0.011 [−0.277, +0.274], no_buy_death −0.49 [−1.63, +0.57], ward_only +0.39
[−0.15, +0.91]. H vs R1 no_buy_death −1.25 [−2.29, −0.35] (n 2,640) is
measured, but H vs T in the same slice is unresolved (−0.49 through zero), so
that SAVE cost cannot be attributed specifically to the historical patch
(owner's precision, 2026-09-29). H vs M2 is negative in every slice except
ward_only (+0.83 [+0.26, +1.40], M2's known ward cost).

**Decision by the rules declared before the run (§4, §8.9):** the history does
NOT add value (H − T's interval includes 0), and H is NOT practical progress
(below M2 by 0.68 points, 992 net decisions). The direction is positive but
within noise (+0.11 points ≈ 160 decisions), concentrated in single-item
decisions. The control is clean in rate, not in decisions: T matches R1's
rate (+0.007) while changing 20,725 first recommendations (14.1 %; H differs
from T in 23,082), so the equal-budget control was not weakened.

**Owner's decision (2026-09-29):** keep M2; close this transfer training
without adopting it; no more H epochs. Two bounded follow-ups: (1) an
inference-only check with the trained models — frozen ½ M2 + ½ H and
½ M2 + ½ T against M2 and against each other, no weight search, to separate
"another model helps" from "the history helps" (complementarity from the
rows: H right where M2 is wrong 4,205, T 3,875;
`data/reports/transfer_review_20260929/saved_row_complementarity.json`);
(2) a clean 16.19 evaluation before any long training: qualify 16.19, fix a
temporal cohort, measure frozen M2 and R1 on it under verified 16.19
economics; the bar for 16.19 is what M2 achieves on those games, not 51.53 %.
This result closes one transfer recipe; it does not establish that prior-patch
data has no value.

Costs: T 2.15 h + 8.6 h; H 2.77 h + 9.0 h (epochs 2,018–2,097 s with the pull
beside them); R1 gate 27 min; scoring + comparisons ≈ 10 min; wall 33.2 h
including a 9 h 49 min owner-requested pause (epoch 2 of H, resumed cleanly).
All pilots and bounds honoured; no gate tripped; no repair, no epoch selection.

What it does and does not show: it separates "history" from "more updates"
(T ≈ R1) and answers the reviewer's question negatively at this dose (four
epoch-equivalents of an adjacent patch, 1.38 M decisions). It does not test
other doses, longer pretraining, or mixing history INTO the current FIT; the SAVE slice is worse than R1 but not resolved against T, so no no-buy bias
can be pinned on the 16.17 prior from this run. Nothing is promoted; R2 = M2 remains the practical
reference; the lever stays more data of the current patch (16.19 collection
running: 4,992 games by 2026-09-29 02:43).

## 10. Follow-up 1 (owner, 2026-09-29): frozen mixtures of M2 with the trained arms — rules declared before the run

Inference only, no training, no weight search: **M2H = ½ P(M2) + ½ P(H)** and
**M2T = ½ P(M2) + ½ P(T)**, where P(M2) is computed exactly as stage 2 did
(½ softmax of the grouped econ-corrected scores + ½ softmax of the ward-free
group scores, stage-1 vocabulary) and P(H), P(T) are the softmax of the grouped
full-catalogue scores of the final transfer checkpoints (common vocabulary,
same 9,333-group space, asserted identical). Same cohort (146,597 / 1,209),
ward-free first-action exact at gold_est, paired-by-game bootstrap (2,000
draws for the primaries). Integrity gates: M2 reproduces `rows_M2_mix_full`
0/0/0, H reproduces `rows_H_full`, T reproduces `rows_T_full`, all in the
scoring process; every checkpoint digest-bound to the frozen references and
the transfer report. Readings: M2H vs M2 and M2T vs M2 (does adding another
model help?), **M2H vs M2T (is the history specifically worth anything?)**, and
each mixture vs R1 and vs its own arm. **Decision rule:** nothing is adopted
unless a mixture beats M2 with the 95 % CI excluding 0; a history-specific
advantage requires M2H − M2T with the CI excluding 0; the gates must pass.
Complementarity from the rows (owner, before the run): H right where M2 is
wrong 4,205 decisions, T 3,875 — an oracle union, not attainable performance.
Script `scripts/basket_transfer_mixture.py` (allowlisted); smoke passed on the
smoke chain (gates 0/0/0 ×3, 55 s); full run launched 14:5x.

### 10.1 RESULT (2026-09-29 15:27) — no mixture beats M2 on the primary; the history-specific term is small but resolved

`mixture_report_full.json` (146,597 decisions / 1,209 games, 41 min; gates: M2
reproduces its stage-2 rows byte-for-byte — same digest 67d14d30… — H and T
reproduce their transfer rows 0/0/0; recomputed independently from the rows).

| model | first-action exact | any-of-three |
|---|---|---|
| R1 | 50.737 % | 65.600 % |
| M2 (reference) | 51.531 % | 66.087 % |
| M2T = ½ M2 + ½ T | 51.467 % | 66.151 % |
| **M2H = ½ M2 + ½ H** | **51.597 %** | **66.221 %** |

| reading (paired, points) | first-action diff | 95 % CI (2,000 draws) | net | top-3 diff | 95 % CI |
|---|---|---|---|---|---|
| M2H vs M2 (does adding H help?) | +0.066 | [−0.029, +0.162] | +97 | +0.134 | [+0.066, +0.200] |
| M2T vs M2 (does adding T help?) | −0.064 | [−0.157, +0.031] | −94 | +0.064 | [+0.001, +0.134] |
| **M2H vs M2T (is the history specifically worth anything?)** | **+0.130** | **[+0.031, +0.231]** | +191 | +0.070 | [−0.009, +0.140] |
| M2H vs R1 | +0.860 | [+0.752, +0.972] | +1,261 | +0.621 | [+0.531, +0.708] |

Agreement of the first recommendation: M2H = M2 on 135,915 rows (92.7 %),
M2H = M2T on 134,541. Oracle union M2 ∪ H 54.40 %, M2 ∪ T 54.17 %.

**Decision by the rules declared in §10:** nothing is adopted — neither mixture
beats M2 on the primary with the interval excluding zero. The history-specific
term is resolved on the primary only: with the same weights, the mixture with H
beats the mixture with T on first action by +0.13 points [+0.03, +0.23] (191
net decisions). On top-3 BOTH mixtures improve M2 (H +0.13 [+0.07, +0.20];
T +0.06 [+0.00, +0.13], a very small margin) and M2H vs M2T stays unresolved
(+0.07 [−0.01, +0.14]), so no additional top-3 advantage of the history is
demonstrated (owner's correction, 2026-09-29). The 16.17 history carries a
small, real first-action signal inside a mixture that is not enough to move
the first-action bar. M2 stays the practical reference; the transfer recipe
is closed with everything the trained models could give without training again.
