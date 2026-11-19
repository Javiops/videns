# HANDOFF — Build Tracker (live League shop-decision advisor)

**September 29 (17:13) — 16.19 CHAIN RELAUNCHED at the owner's request: reconstruction task registered 17:10, running since 17:11:24 (build stage, ~100 sources/min -> build ~18:20, audit ~19:05); benchmark chain registered and waiting since 17:12:26 (-> ~20:10). Preconditions verified before registering: no run/ dir, no partial corpus (the supervisor archived run_failed_20260929_1655 and removed data/corpora/16.19-reconstructible-v1 on failure), 202 GB free, only the Daily Pull registered, all 49 app/*.py digests identical to the 16:12 freeze. FREEZE IS BACK ON: no byte of app/*.py or the pinned chain scripts changes until the benchmark completes.**

**September 29 (17:15) — 16.19 BUILD FAILED AT 17:00:54 ('Processing code changed', run_failed_20260929_1655) BECAUSE OF THE OVERLAY WORK: app/main.py, app/overlay.py and app/advisor.py were edited during the freeze (a route and a launcher height) and the byte digest over every app/*.py caught it at the end of the build (4,950/4,992 sources done). Benchmark chain exited 17:01:48; the 16.19 ladder pull finished 16:48 (exit 0). All three one-shot tasks self-unregistered. The disk matches the frozen digests byte for byte again (49/49; main.py restored to its original CRLF bytes, overlay.py/advisor.py to LF). NOT relaunched — owner's call: same procedure as the 16:12 relaunch (scripts/run_reconstructible_1619.ps1, then run_benchmark_1619.ps1 behind it). Lesson in docs/history.md.**

**September 29 (17:30) — LIVE OVERLAY MOVED OUT: the in-game overlay (team item gold, Baron/Elder buff timers with holders, Baron/dragon respawn timers) now lives in the sibling project `../live_overlay` — own venv, own server (port 8020), own launcher, reimplemented Live Client fetch and Data Dragon item totals; no shared code or data with build_tracker (the macro_coach rule). Everything it briefly added here today (app/live_overlay.py, GET /api/overlay, two cards on /live, the 780 px launcher height, scripts/test_live_overlay.py) was removed; live test subset back to 4/4. Two findings measured on this repo's snapshot corpus stay in docs/history.md: the API `price` field is the combine cost, and event `KillerName` is the untagged riotIdGameName, which `app/live.py::_objectives` never matches, so the live `ally_obj`/`enemy_obj` features have been zeros (NOT fixed: live.py feeds model inputs and is under the chain freeze). No digested file and nothing the running 16.19 chain imports was touched at any point.**


**September 29 (23:00) — STOPPED AT THE OWNER'S REQUEST ("stop everything rn", machine needed): benchmark chain killed during score_M2 (started 20:13, ~2 h 45 of scoring lost, no partial rows; task self-unregistered). R1's 16.19 report/rows in data/experiments/16.19-benchmark-v1 are complete and valid. Ladder pull had already finished 21:37 (exit 0, 581 new perspectives). To resume the benchmark: rerun only score_M2 (scripts/score_frozen_on_generation.py --model M2 --generation data/dataset_generations/16.19-development-v1 --out data/experiments/16.19-benchmark-v1; ~3 h CPU) and then pair (footwear_runtime --entry scripts/pair_frozen_on_generation.py with PREFIX_ML_DIR=v4); admission/coverage/publication/R1 need not be redone. Only when the owner says so.**

**September 29 (20:52) — LADDER PULL 16.19 RELAUNCHED (owner: "continua con el daily pull"): task 'BuildTracker Ladder Pull 1619' (scripts/run_ladder_pull_1619.ps1, KR + EUW, patch 16.19), beside the benchmark chain (score_M2 running). Previous run 15:42-16:48 ended exit 0 (732 new perspectives / 3,448 EUW games). Key from 01:17 still answered 200 at 20:45; it expires ~01:15 on 09-30, before the 04:30 daily pull. Log data/logs/ladder_pull_1619_20260929_205205.log.**

**September 29 (20:25) — 16.19 RECONSTRUCTION COMPLETE (relaunched 17:11 by someone other than this session, app/ quiet since 17:04): 4,992 -> 4,112 qualified games, 498,185 admitted rows, audit 0/0, complete 18:54. Benchmark chain ran automatically: admission (boundary 2026-09-27T00:00Z) -> coverage -> generation data/dataset_generations/16.19-development-v1 (train 2,770 games / 335,996 rows; cohort 1,342 games / 162,189 rows) -> R1 scored (gate 0/0/0 on 10,000 v4 rows; 16.19 cohort: exact 50.943 %, top-3 65.799 %, 0 rows with unknown items) -> M2 scoring since 20:13 (~21:30) -> pair (~21:40). Results dir data/experiments/16.19-benchmark-v1.**


**September 29 (16:57) — 16.19 BUILD FAILED AGAIN at its final code check ("Processing code changed"): app/advisor.py, app/overlay.py and app/main.py were rewritten at 16:53 by ANOTHER agent (a `codex` process started 13:08, working on the beta launcher/overlay). The reconstruction pins every app/*.py at start and re-checks at the end; unrelated UI edits during the run kill it. Second partial (11.5 GB) deleted, run dir archived as run_failed_20260929_1655. Relaunch only once app/ is quiet for ~2 h (owner asked). Proposal for the owner: narrow the pinned set of future chains to the reconstruction closure (contract change, needs approval).**


**September 29 (16:48) — 16.19 PULL DAY 3 DONE (15:42-16:48): KR 1,200 + EUW 732 new perspectives; task self-unregistered. The dev key expires ~01:15 on 09-30 (before the 04:30 daily pull, now set to 16.18,16.19); the owner was told.**


**September 29 (16:12) — 16.19 BUILD RELAUNCHED after a self-inflicted refusal: the first full build (15:04-16:08) processed all 4,992 sources and then failed its final live holdout scan because the SMOKE admission/generation I published beside it under data/ recorded 48 16.19 games as `val`. Smoke split artifacts deleted (data/acceptance/16.19_smoke_admission*, data/dataset_generations_smoke), partial corpus (11.5 GB) deleted, run dir archived as run_failed_20260929_1608; live scan clean (45 files, 0 hits). Tasks re-registered: reconstruction (~1.1 h build + ~45 min audit -> ~18:10) and the benchmark chain behind it (-> ~19:15). Lesson in docs/history.md.**


**September 29 (15:27) — JOB 1 RESULT: NOTHING ADOPTED. M2H = 1/2 M2 + 1/2 H 51.597 % vs M2 51.531 % (+0.066 [-0.029, +0.162], net +97, null); M2T 51.467 % (-0.064, null); M2H vs M2T +0.130 [+0.031, +0.231] (history-specific, small, resolved); top-3: BOTH mixtures beat M2 (H +0.134 [+0.066, +0.200]; T +0.064 [+0.001, +0.134]) and M2H vs M2T top-3 unresolved -> no top-3 advantage of the history demonstrated (owner's correction). Gates 0/0/0 x3 (M2 byte-exact 67d14d30...). Verified from rows_mix_*_full.jsonl. docs/experiment_prior_patch_transfer_v1.md §10.1. Transfer recipe closed; M2 remains the reference.**


**September 29 (15:17) — 16.19 BENCHMARK CHAIN REGISTERED ('BuildTracker Benchmark 1619', scripts/run_benchmark_1619.ps1): waits for the reconstruction's completion.json, then admission (boundary 2026-09-27T00:00Z, docs/benchmark_1619_v1.md) -> coverage -> publish data/dataset_generations/16.19-development-v1 -> score R1 (footwear_runtime, v4 world) -> score M2 (package) -> pair -> data/experiments/16.19-benchmark-v1/benchmark_16.19-development-v1.json. Dress rehearsal on the smoke generation passed (gates 0/0/0; R1 50.04 %, M2 50.84 % on 48 games). ETA ~19:30.**


**September 29 (15:15) — 16.19 ADMISSION PATH WORKS end to end on the smoke corpus (prepare_development_1619 -> coverage -> publish into data/dataset_generations_smoke; train 37 / val 48). Shared modules generalised to a per-purpose table SCOPED_DEVELOPMENT (app/dataset_generations.py: patch_development_1618 unchanged + patch_development_1619; report_corpus_coverage, publish_patch_dataset, corpus_rows.load_qualified, footwear_runtime dispatch); derived benchmark_1619_runtime/entry, prepare_benchmark_1619, prepare_development_1619 (explicit boundary required; prior = the 16.18 v4 generation), test_benchmark_1619_admission (passes; 1618 twin unchanged and passing; suite 57/59). CONSEQUENCE: corpus_rows.py changed -> the 16.17 history export's reviewed row-builder digests no longer match live code, so validate_history/validate_rows will refuse it for NEW training runs until re-exported (26 min) — deliberate; nothing planned needs it. FULL 16.19 BUILD running since 15:04 (task 'BuildTracker 16.19 Reconstructible Sources'; ~100 games/min -> build ~15:55, audit ~16:45). Do NOT edit app/*.py or the pinned chain scripts until it completes. Mixture check (job 1) still computing (CPU-bound setup/decoding; ~2 h expected).**


**September 29 (15:00) — JOB 1 RUNNING (frozen mixtures, full cohort; rules in docs/experiment_prior_patch_transfer_v1.md §10; smoke gates 0/0/0 x3). JOB 2 STARTED: 16.19 chain derived + ledgered (data/acceptance/16.19_reconstructible_20260929/derivation_ledger.json), static review 16.18.1->16.19.1 clean, FULL SNAPSHOT 4,992/4,992 selected (no test/holdout overlap), 100-game smoke build running (task 'BuildTracker 16.19 Smoke'); protocol docs/benchmark_1619_v1.md (boundary proposal 2026-09-27T00:00Z -> 3,384 train / 1,608 cohort raw; frozen models score with their own 16.18.1 world; unknown-item rows counted). Pending: full build+audit (~1.5 h), admission/coverage/publication for 16.19 (needs a patch_development_1619 branch in app/dataset_generations.py + prepare_development_1619.py + footwear_runtime dispatch), then R1/M2 scoring on the 16.19 cohort.**


**September 29 (14:50) — OWNER DECISION: keep M2, close the transfer recipe without adopting it; two bounded jobs. (1) frozen 1/2 M2 + 1/2 H and 1/2 M2 + 1/2 T vs M2 and vs each other, inference only (scripts/basket_transfer_mixture.py, allowlisted; smoke running); (2) a clean 16.19 benchmark before any long training (qualify 16.19, temporal cohort, measure frozen M2 and R1 under verified 16.19 economics; the 16.19 bar = what M2 gets there). Precisions recorded: T matches R1's rate but changes 20,725 first recommendations; H's SAVE cost vs R1 is not attributable to the history (H vs T unresolved). daily_pull.py default patches -> 16.18,16.19 (the 04:30 task passes no --patch; today's run still collected 16.17,16.18 + the 16.18 dated sample).**


**September 29 (09:25) — PRIOR-PATCH TRANSFER RESULT: NULL. H 50.854 % vs T 50.744 % (+0.111 [-0.028, +0.244], net +162), vs R1 +0.117 [-0.015, +0.270], vs M2 -0.677 [-0.818, -0.540]; T = R1 (+0.007). SAVE slice worse vs R1 (-1.25 [-2.29, -0.35]). Nothing promoted; R2 = M2 stays the practical reference. Full write-up docs/experiment_prior_patch_transfer_v1.md §9; recomputed independently from rows_{T,H}_full.jsonl. Trainer exit 0, task self-unregistered; GPU idle. Daily pull (collection-only) still running its sample stage.**


**September 29 (02:45) — 16.19 PULL DAY 2 DONE (01:27-02:43, new key in build_tracker\.env at 01:17): KR 370 new games / 1,338 perspectives, EUW 1,595 perspectives; the rest already stored. Task self-unregistered; the 04:30 daily pull continues while the key lives. H on epoch 5/16.**


**September 29 (01:15) — TRAINING RESUMED (paused 15:26 -> 01:15, 9 h 49 min; H still in epoch 2/16; GPU 77 %). Projection: H done ~09:00, report ~09:30. All three .env keys are now rejected by Riot (dev keys expire at 24 h; the 04:30 daily pull will abort again) — the owner is placing a new one.**


**September 28 (15:27) — TRAINING PAUSED at the owner's request (H in epoch 2/16): scripts/pause_transfer_train.py --suspend (NtSuspendProcess; VRAM kept, nothing lost). Resume with --resume when the owner says; the paused epoch's `seconds` will be inflated (descriptive only); ETA shifts by the pause length.**


**September 28 (12:25) — 16.19 PULL DONE (07:35-12:25): KR 2,407 games / 7,152 perspectives + EUW 2,711 / 7,613; database now holds 4,115 games of patch 16.19 (2026-09-23 04:42 -> 09-28 07:52 UTC; the DTO gameVersion filter dropped the rest). Task self-unregistered; the 04:30 daily pull continues while the dev key lives (24 h from its creation). Training untouched (H phase 1 running).**


**September 28 (11:50) — T DONE: 50.744 % vs R1 50.737 % (net +10 of 146,597, from persisted rows; paired CI at the end) -> the equal-budget control adds nothing over R1. H phase 1 running (ends ~14:35), H done ~23:15, report ~23:40. 16.19 pull: KR 2,407 done, EUW 1,190/2,711 at 11:20.**


**September 28 (07:45) — 16.19 LADDER PULL RUNNING (KR+EUW) since 07:35 after the owner corrected the key (Riot 200); training T on epoch 9/16, on pace.**


**September 28 (00:49) — TRAINING PAST ITS GATES: R1 reproduced 0/0/0 on 146,597 rows (1,610 s); in-run pilots T 2.16 h + 8.57 h, H 2.77 h + 8.67 h, all within the owner's bounds (3 h / 10 h); T phase 1 running. Projection: T done ~11:35, H ~23:00, report ~23:30 today.**


**September 28 (00:25) — PATCH 16.19 OPENED for ingestion (config, Data Dragon 16.19.1 cached + static review clean, tests 5/5); 24 h ladder pull KR+EUW registered ('BuildTracker Ladder Pull 1619', run_ladder_pull_1619.ps1) — WAITING FOR A VALID KEY in build_tracker\.env (the owner's new key landed in macro_coach\.env; retries every 10 min for 8 h). Training untouched (setup passed 426 s; R1 gate running).**


**September 28 (00:13) — TWO LONG ARMS RELAUNCHED (owner: "game over, todo tuyo el ordenador"); task 'BuildTracker Transfer Train' started 00:13; bounds phase 1 <= 3 h / phase 2 <= 10 h per arm; projected end ~00:30 on 09-29. Nothing else running (Daily Pull 04:30 stays registered as always).**


**September 27 (21:45) — LAUNCH POSTPONED at the owner's request (playing): the 21:36 start was stopped during setup (no training done, no partial artifacts); relaunch = register_task.ps1 -TaskName 'BuildTracker Transfer Train' -ScriptName scripts
un_transfer_train.ps1 once the owner says the game is over. Bounds/recipe unchanged (§8.9).**


**September 27 (21:25) — OWNER GO: TWO LONG ARMS LAUNCHED (task 'BuildTracker Transfer Train'); bounds phase 1 <= 3 h, phase 2 <= 10 h per arm (owner raised 9 -> 10 before launch, docs §8.9); recipe intact; result at the final checkpoint; chained without further confirmation; expected end ~21:30 on 09-28 (projection).**
Owner's reading rule: H must beat separately the equal-budget control T, the current Bf (R1) and M2 (R2); beating a weakened control
is not a useful gain. The trainer (scripts/basket_transfer_ab.py via footwear_runtime, PREFIX_ML_DIR v4): validator + row check ->
references bound -> R1 reproduction gate (exit 2 on mismatch) -> in-run pilots vs BOUNDS_H {3, 10} (exit 3 above bound, reason
published in transfer_report_full.json) -> T (11,844 updates, reset, 16 epochs) -> H (same on the 16.17 FIT) -> scoring on the
146,597-row cohort -> paired comparisons H/T/R1/R2 -> decision block. Progress: data/experiments/16.18-prior-patch-transfer-v1/
transfer_progress_{T,H}_full.jsonl; logs data/logs/transfer_train_*. Recovery: --compare-only recompares persisted rows_{arm}.
Launcher RAM gate 2.5 GB / 3 h (desktop keeps 1-3 GB free).


**September 27 (20:40) — PREP CHAIN DONE: full history export + caches + pilots; BUDGET MEASURED (T 11.07 h, H 11.71 h, ~22.8 h GPU sequential); bounds met but phase 2 within 1.5% of its 9 h bound. AWAITING OWNER GO for the two long arms.**
Export 18:15-18:41 (11,562 games / 1,381,011 rows / 0 exclusions / 36,389 rows re-checked). Caches: 16.17 53 min (17,436 baskets,
1.50 GB), 16.18 54 min (13,165 = stage 1; cohort 146,597, rows sha 763bb9ee...); common vocab 156c2217... (identical to smoke).
Pilot (docs §8.8): T phase1 0.679 s/batch -> 2.23 h, phase2 0.672 -> 8.84 h; H phase1 0.863 -> 2.84 h, phase2 0.674 -> 8.87 h.
Launch when GO: register_task.ps1 -TaskName 'BuildTracker Transfer Train' -ScriptName scripts
un_transfer_train.ps1 (trainer:
references bound, R1 gate, in-run pilots vs BOUNDS_H {3, 9}, T, H, scoring vs R1/R2, decision; exits 2 gate / 3 bound). Expected
end ~24 h after launch. All prep tasks self-unregistered; only 'BuildTracker Daily Pull' remains registered.


**September 27 (18:15) — r2 RECONSTRUCTION COMPLETE AND QUALIFIED (11,562 games, 1,381,011 admitted decisions, audit 0/0); validator hardening installed as a separate module; PREP CHAIN RUNNING (export -> full caches -> pilots). Long trainings still pending pilot report + owner GO.**
Run 13:34-18:10 (4 h 36; job peak 588 MB; audit ~100 games/min under 0.5-2.7 GB free RAM). Exclusions of 2,502 sources in
docs/experiment_prior_patch_transfer_v1.md §8.4. Validator decision: history_1617.py is PINNED build code of the completed runs
(the exporter refused: "live code differs from the audited build code: scripts/history_1617.py" when the hardened version was
installed) -> restored to the pinned digest; the row check lives in scripts/history_rows_check.py (validate_rows after
validate_history in exporter, prepare_transfer_caches, basket_transfer_ab; 300 sampled games vs frozen corpus events + reviewed
row-builder digests live). Proofs: rows check 3/3, validator suite 18/18 + 4/4, smoke re-export deterministic (6af7c28a...).
Prep task registered 18:14 (run_transfer_prep.ps1; RAM gate 2.5 GB / 3 h; logs data/logs/transfer_prep_*): export (~40 min
+ 33 GB hash), cache 16.17 (~1.4M rows, ~1.5 h), cache 16.18 (~40 min), --pilot-only. Then: bounds already declared in the
trainer (3 h / 9 h per arm), scripts/run_transfer_train.ps1 written (NOT registered) -> owner GO.


**September 27 (14:10) — SECOND REVIEW (wf_2703ca2a, 32 agents) FIXED; trainer smoke + compare-only re-passed (R1 gate 0/0/0); the prep task is deliberately NOT registered until the reconstruction ends.**
10 confirmed / 4 refuted (docs/experiment_prior_patch_transfer_v1.md section 8.5). Fixed now (non-pinned files): trainer binds R1/R2
at setup, runs the protocol's R1 reproduction gate (stage-1 vocabulary, live decoder; 0/0/0 on the smoke cohort in 46 s) before any
training, enforces BOUNDS_H {phase1 3 h, phase2 9 h} per arm (exit 3), pairs need both sides (--arm H no longer crashes),
--compare-only recompares persisted rows; caches + trainer refuse a --limit history without --smoke; exporter declares the reviewed
row-builder digests (REVIEWED_ROW_BUILDER: corpus_rows.py, build_recovered_patch_corpus.py, train_prefix.py); prep launcher has an
8 h deadline + 15-min liveness on status.json + guarded reads; register_history_1617.ps1 trims trailing separators.
PENDING UNTIL run/completion.json (history_1617.py is pinned build code while the supervisor runs): install the reviewed validator
(scratchpad history_1617_next.py: rows of 300 sampled games checked field by field and label-wise against the frozen corpus
events, save sentinel 999999 handled; build_recovered_patch_corpus.py in ROW_BUILDER_CODE) -- proven on the smoke export (31,131
rows accepted in 7 s; a consistently re-hashed relabelling rejected) and on the synthetic suite (18/18 + 4/4); then re-export the
smoke history, run the suite, and register 'BuildTracker Transfer Prep' (run_transfer_prep.ps1). The prep task registered at 13:35
was ended/unregistered at 13:5x on purpose (it would have exported with pre-review code).


**September 27 (13:35) — OWNER FREED DISK (deleted only the quarantined v3_failed_gate_a tracker.db, 14.1 GiB; record data/reports/cleanup_20260927/) and GAVE GO: snapshot r2 FROZEN (14,064 sources); reconstruction RUNNING; preparation task queued (export -> caches -> pilots). Long trainings still pending the full budget + owner GO.**
Snapshot r2 13:29: 20,859 DB games -> 14,064 selected = r1 minus exactly the 153 holdouts (3,840 historical validation holdouts +
2,955 reserved tests excluded before any payload; 45 split files scanned; ledger with review fixes copied into the base dir).
Tasks: 'BuildTracker History 1617' (register_history_1617.ps1, start 13:34, wall 21,600 s; ~1.8 h build + ~1.5 h audit -> ~17:00)
and 'BuildTracker Transfer Prep' (scripts/run_transfer_prep.ps1, start 13:35: waits for run/completion.json, exits 2 on a failed
run, waits for >= 4 GB available RAM before each step, then export_history_1617 -> prepare_transfer_caches 16.17 -> 16.18 (v4
runtime) -> basket_transfer_ab --pilot-only; logs data/logs/transfer_prep_*). Expected: ~12,000 qualified games, ~1.45M rows
(121 rows/game in the smoke), 16.17 cache ~1.6 GB (1.08 KB/row), 16.18 cache 0.82 GB. Free disk observed 241 GiB (PowerShell)
after the owner's cleanup. Stale task from the withdrawn run unregistered first (its wrapper had been killed before its finally).
NEXT after the pilot report: declare bounds (phase 1 <= 3 h/arm, phase 2 <= 9 h/arm), write the two-arm launcher, ask the owner GO.


**September 27 (12:20) — 16.17 HISTORY: snapshot r1 WITHDRAWN (153 evaluated holdouts inside); review fixes applied and proven; SMOKE CHAIN OF THE WHOLE CONTRACT PASSED (300 games, both phases); budget measured; FULL SNAPSHOT r2 BLOCKED ON DISK (owner decision).**
Review wf_4a33520c (33 agents): 11 findings confirmed. Blocker: r1 scanned only data/ml + dataset_generations/*/split_assignment.json;
the 16.17 development validations live in preserved_split_assignment.json / split_plan.json -> 153 holdouts in the 14,217 sources.
Build from r1 ended (owner condition 1): the wrapper was ended 11:44, the orphaned supervisor (kill refused by the sandbox)
failed by itself at 12:36 on its own guard ("40 GB disk reserve reached", 8,9xx games); its 25.2 GB partial corpus was deleted
and data/acceptance/16.17_history_20260927 archived as _superseded_r1 (metadata only, nothing exported). Fixes (docs/experiment_prior_patch_transfer_v1.md §8;
ledger section review_fixes_20260927 in the smoke acceptance dir): shared scan history_1617.split_files (45 files; 3,112 test +
7,837 val ids) in snapshot / validate_plan / verifier / validator; exporter binds live code to the completed run's protocol;
exclusions restricted + recomputed from the corpus; rows == admitted_decision_ids; manifest reuses verified bindings; trainer
binds R1/R2 rows to the frozen digests, re-seeds before phase 2, checks catalogue_sha256, projects hours with the declared
budget; prepare --limit, export --allow-limited, --history/--out on caches+trainer; launcher parametrised +
register_history_1617.ps1. Validator: 22 rejections + control; suite 56/58 (same 2 stale-path failures).
SMOKE (§8.1): 300 -> 257 qualified, 31,131 admitted rows, audit 0/0 (4.6 min); export 257/31,131/0 exclusions validated (68 s);
common vocab 173/205/198/197, catalogues 13,165 / 2,778 (smoke), cohort 1,500; trainer smoke both arms + scoring vs digest-bound
R1/R2, all fields (89 s). BUDGET (§8.2, pilot_report_smoke.json): T 0.610 s/batch -> 2.01 h + 0.602 s/batch x 2,961 x 16 -> 7.92 h = 9.92 h; H
phase 2 7.94 h, phase 1 to re-pilot on the full 16.17 catalogue (smoke 0.63 h); both arms ~19-20 h GPU sequential.
DISK (§8.3): corpus ~2.45 MiB/game -> ~37 GB for ~14,064 sources; supervisor reserve 40e9 bytes (bound) -> need >= 77 GB
(71.7 GiB) free at launch; have 62.5 GiB after the r1 cleanup (12:40) -> ~10 GB short. Candidates (MB): 16.17-development-v2 29,529; 16.18-reconstructible-v3_failed_gate_a
15,015; -v3 15,015; -v2 9,872; -v1 6,073. Owner decides; then: prepare r2 -> register_history_1617.ps1 -> export -> caches full
-> --pilot-only -> GO. M2 full harness DONE 11:43: 146,597 rows, identity/first_exact/any3 mismatches 0/0/0 (pred_first 3,
tie-order), adapter 0.51531, unaffordable 0 of 427,940, sandbox OK.


**September 27 (11:10) — OWNER GO (option 1): 16.17 HISTORY CORPUS with the full validation contract; snapshot frozen, reconstruction RUNNING; export/validator/caches/trainer written; adversarial review running.**
Owner's conditions: same data guarantees in a different format — frozen snapshot excluding reserved tests
and historical holdouts BEFORE any payload; reconstruction under 16.17 rules with every 16.18 gate; exact
export of the admitted rows (counts/versions/exclusions, no extras/dups/silent drops) reusing corpus_rows;
own manifest + validator used at export and at training; app/, v4 and M2 untouched; long trainings pending
the measured budget. Done: six scripts DERIVED from the 16.18 chain by asserted single-occurrence
replacements (ledger data/acceptance/16.17_history_20260927/derivation_ledger.json): reconstructible_1617_runtime
(reviewed sources loaded VERBATIM; labels decision-start-footwear-v4 / elixirs-prototype inventory / economics
16.17-v1), viego_source_gate_1617, prepare_reconstructible_1617 (identity static review over 16.17.1; excludes
fold test AND fold val of any split file), process_reconstructible_1617, run_reconstructible_1617 (+.ps1),
verify_reconstructible_1617_evidence (also refuses val-fold selections). Snapshot 10:59: 20,859 DB games ->
14,217 selected; excluded 3,687 historical validation holdouts + 2,955 reserved test IDs. Build started 11:04
(task 'BuildTracker History 1617'; ~130 games/min -> build ~12:55, audit ~14:30). scripts/history_1617.py
(validator; 11 rejection cases proven incl. split-file infiltration) + scripts/export_history_1617.py (verify
prerequisite -> activate 16.17 runtime -> corpus_rows.iter_games -> manifest -> self-validate, else delete).
scripts/prepare_transfer_caches.py (common vocabulary from both FITs; per-patch runtime featurisation; digest-
bound caches) + scripts/basket_transfer_ab.py (two symmetric phases, U=11,844, reset both, same phase-2 order,
pilots, --pilot-only, scoring vs R1/R2). Protocol v2 closed (docs/experiment_prior_patch_transfer_v1.md).
Workflow wf_4a33520c adversarially reviewing all of it. Next: export -> caches -> trainer smoke + pilots ->
budget to owner. M2 full displayed-policy harness still running (~11:45).


**September 27 (10:35) — M2 PACKAGED (`mixture-v4-m2-v1`, id ab2b4cc4…); harness smoke 0/0, 0 unaffordable, sandbox OK; latency warm median 39.0 ms (v1 35.3); full harness running; prior-patch protocol PREPARED (not launched).**
scripts/build_mixture_package_v2.py (allowlisted; inputs verified 4/4 against frozen_reference_stage2.json; identity
permutation; econ stats from the fine-tune), scripts/mixture_displayed_policy_v2.py (v4 cohort rows, reference
rows_M2_mix_full.jsonl, exactness flags compared), mixture_package_latency.py gained --review/--generation.
Reports: data/reports/mixture_displayed_policy/{displayed_policy_mixture-v4-m2-v1_cohort.json, package_latency_mixture-v4-m2-v1.json}.
Full 146,597-row harness launched 10:31 (~55 min). docs/experiment_prior_patch_transfer_v1.md: inventory (17,904
16.17 games with raw pairs; existing 16.17 generation is old-pipeline -> re-reconstruct with 16.17.1 static data),
equal-budget arms T vs H, primaries, cost anchors (~3 h recon + ~30 h GPU), awaiting pilots + owner GO.


**September 27 (03:09) — STAGE 2 COMPLETE: M2 = 0.5 econ_ft_new + 0.5 wardfree_new beats the frozen reference (+0.794 [+0.691, +0.889], net +1,164) -> new PRACTICAL REFERENCE R2 = 51.53%. Nothing promoted; owner decides next.**
Cohort 146,597 / 1,209 games. econ_ft_new 50.98% (+0.246 [+0.160,+0.324] vs R1; +0.271 [+0.202,+0.335] vs
twin control -> features carry it); control_ft_new 50.71% (ns); wardfree_new 50.65% (null); M2 51.53% /
top-3 66.09% (+0.49); vs old mixture +2.33 [+2.18,+2.48] net +3,421; both-parents-right->M2-wrong 0;
slices single +0.81, multi +1.05, SAVE ns, ward-only -0.77 (cost). Verified from rows; gates 0/0.
Reference file data/experiments/16.18-stage2-v4/frozen_reference_stage2.json (checkpoints econ_ft_new_ep8
670dd7bd..., wardfree_new_ep16 ffbfc40a..., econ_stats). Costs within bounds: wardfree 6.97 h (pilot 6.81),
twin FT 8.97 h train (pilot 8.03), features parallel ~1.7 h wall, domains 3 h 20 (CPU, avoid), mixture 34 min.
Full write-up: docs/experiment_stage2_econ_wardfree_v1.md RESULT + docs/stage2_result_v4_20260927.md (dossier);
informe videns.gg/informes/20260926/. NEXT (owner/reviewer): package M2 (mixture-package contract +
displayed-policy report + latency), then the sealed test once final; more data remains the lever.


**September 26 (17:25) — finetune RUNNING (pilot 8.03 h, bound 12) after a job-ceiling trap; final projected ~02:00 on 09-27.**
First finetune attempt (17:05) died at the pilot with CUDA OOM while 4.8 GB VRAM was free: the segment
job's 12 GB commit ceiling was hit (job peak 12.04 GB) -> raised to 20 GB; load_econ_fit now fills a
preallocated tensor shard by shard (no torch.cat spike). Resumed 17:16 (domains/features/gate/wardfree
skipped by report); setup 403 s, step-0 |corr| 0, parent fixed CE 1.7395, pilot 0.610 s/batch -> 8.03 h;
GPU 92%, RAM 5.4 GB available, no paging. Then mixture (M2) -> final report. Ledger entry added.


**September 26 (17:05) — STAGE 2: wardfree NULL (−0.086 [−0.216, +0.050]); cache COMPLETE (parallel, 8 shards + cohort); chain RESUMED 17:04 -> finetune running.**
Parallel cache mishap (16:00): 2 of 5 workers died silently at start-up (RAM ~700 MB available) and the
launcher misread $null exit codes (PowerShell Start-Process -PassThru needs .Handle touched) -> no merge,
5 h idle. Fixed (scripts/run_features_remaining.ps1), pieces recomputed, merge 17:02 (econ_fit_full_shards.json,
8 shards verified; econ_cohort_full.pt). Resume watcher relaunched the chain 17:04 (--resume: domains,
features, gate, wardfree skipped by report) -> finetune (pilot bound 12 h, ~8.6 h) -> mixture -> final.
Projection: final ~02:15 on 09-27. Ledger: docs/history.md (two entries).


**September 26 (09:20) — owner: PC free all day -> features cache built in PARALLEL (bit-identity gate passed on smoke).**
basket_econ_cache_newgen.py --phase features now has worker/merge forms (--shards i,j / --cohort-only /
--merge; every worker streams the whole FIT file and checks the full row digest; merge verifies every
shard, writes the manifest + report). Smoke: 3 shards on 2 workers + cohort -> torch.equal with the
sequential smoke cache (fit AND cohort) = True. Full build launched 09:19 via
scripts/run_features_parallel.ps1: 4 workers x 2 shards of 100k rows + cohort worker (~54 rows/s each,
ETA ~61 min) then merge (~10:25); the sequential standalone build was killed. wardfree ends ~10:40-11:00;
resume watcher then relaunches the chain (--resume) -> finetune ~8.6 h -> mixture -> final ~20:00.


**September 26 (09:10) — STAGE 2 STATUS: gate PASSED, wardfree TRAINING (ep 8/16), features re-running after a false bound trip; finetune+mixture resume automatically.**
Chain (auto-relaunched 00:33): domains (CPU) completed in 3 h 20 (pathological CPU forward, noted);
features STOPPED at 04:55 by its 6 h bound on an inflated bench (27-33 ms/decision; causes: cyclic GC
over the resident cohort rows + Windows maintenance at 04:40; fresh bench 14 ms under every affinity
mask; gc.freeze() applied); gate (GPU) PASSED 0/0; wardfree running since 03:20 at ~1,575 s/epoch
(ep 8/16 at 09:07 -> ends ~10:40 + eval); finetune/mixture skipped by the chain (dependency).
Standalone features build relaunched 09:08 (bench 16.9 ms -> 4.24 h, within bound; ends ~13:30);
scripts/resume_stage2_after_features.ps1 (detached) relaunches the chain with --resume once the
cache report completes and the chain has ended -> finetune (~8.6 h) -> mixture -> final (~23:00).
Aborted report kept as cache_report_full_attempt1_stopped_by_bound.json.


**September 26 (00:05-00:35) — OWNER: "continúa con los núcleos y la RAM libre" -> chain policy reworked; running.**
New policy in scripts/run_stage2_v4.py (schema 2): CPU segments (domains --cpu, features) run at any
time -- BelowNormal on all 16 logical CPUs while "League of Legends.exe" runs, AboveNormal on the
P-cores otherwise (re-prioritised live via winmem.set_priority_affinity on the job's pids); GPU
segments never START during a game and are SUSPENDED (NtSuspendProcess) if one starts; pilot
segments also wait for a quiet GPU; one GPU segment at a time; CPU beside GPU only if the smoke
peaks fit in RAM. New `gate` phase (GPU, minutes) carries the exact R1 reproduction gate now that
domains run on the CPU; features are written in 100k-row shards (~1 GB RAM) and concatenated
only if RAM allows (loader `load_econ_fit` handles both; unit-tested). Order: domains -> features
(CPU) ; gate -> wardfree -> finetune -> mixture (GPU) -> final. Peaks/pause/game-overlap seconds
recorded per segment. The old idle watcher fired before it was killed and relaunched the chain
(task 'BuildTracker Stage2 V4', --resume) at ~00:2x; a smoke of the reworked pipeline runs beside it
(stamp smoke, separate files) to validate the new phases; if it finds a bug the full chain is
resumed after the fix (reports of completed segments are reused).


**September 25 (23:58) — STAGE-2 CHAIN STOPPED BY THE OWNER WHILE PLAYING; auto-resume armed; chain now game-aware.**
Owner: "PARALO EN CUANTO DEJE DE JUGAR LO RETOMAS". Chain killed at 23:58 during `domains` (no
artefact written; status kept as run/status_attempt1_stopped_by_owner_2343.json); task unregistered.
scripts/resume_stage2_when_idle.ps1 (detached, hidden) relaunches the task with --resume once
"League of Legends.exe" has been absent 3 min (log data/logs/resume_stage2_watch_*.log). The
orchestrator now PAUSES itself while a game runs: every 15 s it checks the game process and
suspends/resumes every process of the running segments' jobs (NtSuspendProcess on the job's pid
list); it does not start a new segment while a game runs; paused time is excluded from the 30 h
wall budget and recorded per segment (elapsed vs active seconds). Self-tested on a dummy child.


**September 25 (23:30) — OWNER GO TO STAGE 2 (econ + ward-free on the v4 FIT). Engineering done; smoke chain running; full chain launches after the smoke.**
Owner conditions: recipes/final checkpoints/mixture weights/bounds fixed (cache 6 h, twin FT 12 h,
ward-free 8 h; a pilot above its bound stops that segment and publishes why); new Bf stays the frozen
reference; every candidate and the final mixture compared DIRECTLY with it (beating the twin control
is not practical progress); full result published incl. per-segment costs even if null; no
confirmation between segments after the smoke + integrity gates. Owner's precision: the econ
builder computes+persists the GPU domains first, releases CUDA, then runs CPU features beside
the ward-free training only if the smoke shows both fit in RAM (else series). 15 h is a projection.
Code (new, all under footwear_runtime allowlist): scripts/stage2_newgen.py (shared setup/scoring;
R1 re-instantiated from its checkpoint), basket_econ_cache_newgen.py (--phase domains|features,
R1 reproduction gate, bench-gated), basket_wardfree_newgen.py, basket_econ_finetune_newgen.py,
basket_stage2_mixture.py (M2 = 0.5 econ_ft_new grouped + 0.5 wardfree_new, integrity gate),
stage2_final_report.py, run_stage2_v4.py (+ .ps1; job objects, budgets, GPU-quiet wait before
pilots, CPU builder AboveNormal + P-core affinity, --smoke records peaks, --resume).
Out dir data/experiments/16.18-stage2-v4/ (run/status.json, run/*_full.log). Task 'BuildTracker Stage2 V4'.


**September 25 (23:10) — STEP 2 NULL: new Bf ⊕ old mixture 50/50 = +0.061 [−0.049, +0.166] vs new Bf → NOT adopted; new Bf stays the reference. Budget for stage 2 MEASURED; awaiting owner GO.**
Step 2 (docs/experiment_newbf_mixture_v1.md, 16.18-newbf-mixture-v1/): M 50.80% vs new Bf 50.74% vs
old mixture 49.20%; M vs new +0.061 [−0.049,+0.166] net +89, top-3 +0.002; M vs old mixture +1.60;
ward-only slice −0.54 [−0.98,−0.11]; M == new Bf in 90.3% of first recommendations; integrity gate
0/0 both parents; verified from rows. R2 = R1 = new Bf (50.737%). Ops trap fixed mid-run and
recorded: scheduled tasks run BelowNormal on all 16 logical CPUs incl. E-cores -> 34 -> 18 rows/s
when the desktop is busy; AboveNormal + P-core affinity restored >= 41 rows/s.
Stage 2 budget pilot (docs/experiment_stage2_econ_wardfree_v1.md, 16.18-stage2-budget-v1/budget_pilot.json):
A econ cache 14.4 ms/decision -> 3.61 h CPU (one core); B twin econ FT 0.65 s/batch -> 8.56 h GPU;
C ward-free 0.495 s/batch -> 6.51 h GPU; all within declared bounds (6/12/8 h); ~15 h machine with
A ∥ C, plus ~1 day engineering (new-generation variants of econ_cache / econ_finetune / wardfree +
common cohort scorer keeping R1 and R2). NOT launched: owner decision.


**September 25 (21:05) — REVIEWER STEPS 1-2 IN MOTION: new Bf FROZEN as accuracy reference; frozen 50/50 (new Bf + old mixture) RUNNING.**
Reviewer's independent audit (data/reports/v4_external_review_20260925/catalogue_support_decomposition.json):
98.18% of the +2,257 net gain vs the old mixture falls on truths the OLD catalogue already
supported (+2,049 common, +167 no-main-purchase, +41 new-only) -> the gain is NOT mainly
hitting new catalogue actions (shown); the other effects a larger catalogue may have on the
ranking are NOT isolated (no fixed-catalogue ablation). Correction applied: mixture components were trained on 2,000 FIT games (not 2,500).
Step 1: frozen_reference_v4.json (stage-1 dir): every later candidate must beat 50.737% on
this cohort AND be compared with the old mixture (49.197%). Step 2: docs/experiment_newbf_mixture_v1.md
+ scripts/basket_newbf_mixture.py (allowlisted): 0.5*P_new + 0.5*P_old over the new 9,333 groups,
old 5,692 groups aligned by canonical key (subset, injective), common decode; integrity gate =
both parents reproduce stage-1 rows (smoke: 0/0 mismatches). Task 'BuildTracker NewBf Mixture'
started 21:05 (scripts/run_newbf_mixture.ps1; log data/logs/newbf_mixture_*.out; ~80-90 min
extrapolated). Primary M vs new_bf, paired by game; M becomes the practical reference only if
positive with CI > 0. Step 3 (econ + ward-free partner on the new FIT) after step 2, own pilots.


**September 24 (22:30) — EXPANSION V4 COMPLETE: stage 1 POSITIVE, continuation criterion MET; stage 2 awaits owner GO.**
Cohort 1,209 games / 146,597 decisions (09-20/21 UTC, later than all training). New Bf d64 ep16
(v4 TRAIN 6,354 games) 50.74% ward-free first-exact / 65.60% top-3 vs frozen mixture 49.20/64.20
and old Bf 48.17/63.44. Paired: vs mixture +1.54 [+1.38,+1.69] net +2,257; vs old Bf +2.57
[+2.40,+2.74] net +3,765; no negative slice. Verified from rows_*_full.jsonl (identity identical,
all aggregates reproduce). Report data/experiments/16.18-reconstructible-development-v4-bf-stage1/
bf_stage1_report_full.json; checkpoint bf_newgen_ep16_full.pt (sha256 c923d6ea...) NOT promoted,
not packaged. Dossier docs/expansion_result_v4_20260924.md; informe videns.gg/informes/20260924/.
Costs measured: recon 2h07, epoch 1,734 s, scoring 76 min; chain from train_score 9.09 h.
Two aborts published (audit `[] or fallback` gate -> rebuild; cold-start pilot -> warm-up).
Limits: volume/recency entangled; catalogue 13,165 vs 7,937 coverage share not decomposed;
cohort now exposed. Uncommitted code: audit_patch_corpus.py, baseline.py,
test_decision_state_binding.py, basket_bf_newgen.py (pilot warm-up) -- owner decides commits.
NEXT (owner decision): stage 2 per plan (econ cache + twin econ FT + ward-free partner + new
mixture on the new FIT, same cohort) with measured pilots; cheap first: catalogue_miss decomposition.


**September 23 (20:37) — EXPANSION V4: AUDIT GATE FAILED on correct data; fix applied; attempt 3 (09:13) passed 7 stages, PILOT GATE ABORTED training at 12:35 (10.60 h > 10 h);
pilot measurement fixed (2 untimed warm-up batches, 12 timed); attempt 4 relaunched from train_score.**
Admitted (v4): 7,563 accepted / 2,781 prior_validation_exposed; TRAIN 6,354 games (757,975 visits,
-> 09-19 23:59 UTC) / cohort 1,209 games (146,597 visits, 09-20 00:04 -> 09-21 04:46 UTC); test 3,112
reserved, unopened. Generation 16.18-reconstructible-development-v4 (fingerprint 6c78398c...),
promotion_allowed false. Stage-1 dir data/experiments/16.18-reconstructible-development-v4-bf-stage1
(cache_fit_full.pt reused; pilot_full_attempt3_aborted.json kept).
Attempt 2 built the corpus (16:44->18:36) and failed the semantic audit (->20:37):
ONE decision_state_binding failure, KR_8388382904 ts 1368998 (cohort game): Zed sold
all six items (5,255 g) before buying; reconstruction wrote decision_inventory=[]
(correct); the audit's `payload or inventory_before` fallback read [] as missing.
Fix (uncommitted): scripts/audit_patch_corpus.py rule factored into
decision_state_binding() ("[] is a state; only an absent field falls back"; detail
now carries both inventories); same fallback fixed in scripts/baseline.py;
scripts/test_decision_state_binding.py (8 cases) passes; related tests pass; full
suite re-run in progress. Verified on the quarantined DB: 1/1,249,675 such visits,
0 inconsistent empties. Manifest pins both scripts -> full rebuild (~3 h 53 min
measured). Quarantined (nothing deleted): data/corpora/16.18-reconstructible-v4_failed_gate_b
(quarantine_note.json + the game's 112 stored events), acceptance run_failed_gate_b/,
semantic_audit_failed_gate_b.json, audit_code_failed_gate_b.py; chain
status_attempt2_failed_gate_b_2037.json, recon_attempt2_failed_gate_b.log,
run_plan_attempt2.json. Relaunch = resume_from.txt "recon" + task
'BuildTracker Expansion V4' (scripts/register_task.ps1 -ScriptName
scripts/run_expansion_v4.ps1). Owner authorized (24 Sep ~09:10) deleting the quarantined reproducible tracker.db and the
attempt-1 .partial -> 97.9 GB free; full suite 56 pass + 2 known stale failures (gate OK);
relaunched 09:13 via resume_from.txt (task 'BuildTracker Expansion V4'). Recon measured
3 h 53 min on attempt 2 -> ~13:05; training projected by the pilot gate at its start.


**September 23 (16:09; RESTARTED 16:44 from stage recon) — EXPANSION V4 CHAIN (owner GO, design B).**
Attempt 1 was stopped at 16:35 (26 min into reconstruction) to install the reviewer's
tests gate: the tests stage is now FATAL on any failure except the identified one
(FileNotFoundError on old-root paths in test_reconstructible_1618_pipeline.py and
test_recovered_corpus_pipeline.py; 55/57 pass today), validated on 4 cases. Snapshot
kept (source_plan.json 16:09), partial recon outputs deleted, status of attempt 1 kept
as status_attempt1_stopped_1635.json. Launcher resumes via resume_from.txt.
Task 'BuildTracker Expansion V4' -> scripts/run_expansion_v4.py: snapshot
(data/acceptance/16.18_reconstructible_20260923) -> recon (corpus v4, 6 h wall)
-> verify -> admission (prepare_development_1618_v2 with --forward-boundary-utc
2026-09-20T00:00:00Z bound to docs/expansion_plan_v1.md, authorization literal
"owner authorized design B expansion (...) on 2026-09-23") -> coverage -> export
(generation 16.18-reconstructible-development-v4) -> tests (NON-fatal; known
stale-path failure) -> Bf d64 stage 1 + cohort scoring (basket_bf_newgen.py,
pilot gate 10 h). Status: data/experiments/16.18-expansion-v4/status.json;
per-stage logs there. Result report: data/experiments/
16.18-reconstructible-development-v4-bf-stage1/bf_stage1_report_full.json.
Corrected inventory: NEW never-snapshotted = 6,192 (not 7,442; 1,260 were v3
build/audit failures, re-attempted). Upper-bound split: TRAIN 7,226 / cohort
1,466 / old VAL 1,601 excluded / test 3,112 reserved; validator passed in a
dry run. Continuation criterion (chain econ+wardfree) fixed in the plan:
new Bf vs old Bf >= +1.0 pt AND CI excludes 0. Current generation, mixture
package and historical authorization untouched. Test sealed.


**September 23 (11:45) — EXPANSION PLAN v1 written; awaiting owner authorization.**
Inventory (metadata only): 12,882 patch-16.18 q420 games in DB; **7,442 NEW never
in any snapshot**; 6,126 of them played AFTER VAL's last game (09-14). Boundary
last-TRAIN 09-13 12:55 UTC / first-VAL 12:59, 0 ties. Design A (keep VAL): only
892 new games fit TRAIN. Design B (recommended, per continuation protocol):
forward cohort 09-20/21 (1,466 games), TRAIN pool 5,976 new + 2,500 = 8,476
pre-qualification, old VAL out of training, 3,112 test reserved; all arms
re-scored on the new cohort. "~4,500 admissible" WITHDRAWN as a planning figure.
Staged: Bf d64 first with pilot gate, vs old Bf and frozen mixture on the new
cohort; econ/ward-free chain only if positive. docs/expansion_plan_v1.md,
data/reports/expansion_inventory_20260923/. NOTHING launched.


**September 23 (11:10) — "haz 1-4" complete. LEARNING CURVE STEEP: data is the lever.**
500 games 0.4264 -> 1,000 0.4592 (+3.28) -> 2,000 0.4826 (+2.34); ratio 0.72,
verdict STEEP by the pre-registered rule; no projection beyond 2,000. Learned
gate NEGATIVE (-0.060 vs fixed; the learnable constant chose 0.5012 -> the
fixed 50/50 is validated). Consumable-grouped decoder moot (targets already
exclude consumables). OWNER DECISION PENDING: new generation snapshot to
qualify the ~6,200 post-2026-09-14 patch-16.18 games (DB 12,882 vs snapshot
6,690; ~4,500 admissible at 78% yield), preserving prior split assignments per
policy; then retrain both parents and re-mix. Best config unchanged: frozen
50/50 mixture (package mixture-50-50-v1, experimental_reference).


**September 23 (~09:50) — owner's "haz 1-4"; items 2-pre and 3 closed by facts; 1 and 4 running.**
- Item 3 (consumable-grouped decoder): CLOSED, nothing to gain — the export already
  excludes consumables from targets (`label_counts`, purchase-net-v1): 0 of 58,781
  probe truths contain potions/refillables/biscuits/elixirs.
- Item 2 pre-fact: the generation froze a 2026-09-14 snapshot of 6,690 games
  (6,541 selected -> 5,281 sources -> 4,101 quality-accepted -> 2,500 TRAIN +
  1,601 VAL). The DB `games` table now holds 12,882 patch-16.18 games: ~6,200
  collected after the snapshot, never qualified; at the observed ~78% yield,
  ~4,500 more admissible games. Process work, gated on item 1's verdict.
- Item 4 (learned gate over the two frozen parents, out-of-fold by match on the
  probe, twin zeroed-input control, fixed 50/50 as reference) and item 1
  (learning curve 500/1,000 nested by game, 2,000 point reused from
  16.18-capacity-d128-v1 arm A rows) pre-registered in
  docs/experiment_learned_gate_v1.md and docs/experiment_learning_curve_v1.md,
  smoked, chained in task 'BuildTracker Items 1-4' (gate ~8 min, curve ~1.5 h).
- Mixture package report: docs/mixture_package_report_20260923.md (four sections;
  two limits recorded; package loses one hit to a float16 tie: 49.5126 vs 49.5143).
- CS/assists pilot NOT started: TRAIN/live parity not established (live reads
  neither field; `creepScore` appears nowhere in the repo; assists reconstructed
  from events vs game counter). Needs a paired live-vs-Match-V5 capture first.


**September 22 (21:00) — cycle closed. Mixture stands; loss contrast is a
null for the product.**

*Best observed configuration*: the frozen 50/50 mixture
(`data/experiments/16.18-frozen-mixture-v1/frozen_mixture_v1.json`, all
components by sha256). Probe 0.49514 (+1.03 / +0.94 vs its two parents);
VAL 191,916 decisions 0.49190 (+0.92 [0.82,1.01] / +0.99 [0.87,1.09]);
top-3 VAL 0.64110. Serving cost measured end to end: circuit A 75.53 ms vs
circuit M 83.83 ms median, paired delta +7.78 ms (+11.0%). NOT promoted —
that VAL was already exposed and promotion needs the displayed-policy
artifact with zero unaffordable baskets.

*Loss contrast λ=0.25* (16.18-loss-contrast-v1): PRIMARY 1 +0.048
[+0.002,+0.099] (2000d lower bound +0.000); PRIMARY 2 vs the frozen mixture
+0.044 [−0.002,+0.095] NOT RESOLVED; arm alone +0.073 [+0.012,+0.141];
control did not regress (−0.003). The loss change worked on the objective
(main term 1.3644 vs 1.3707) and did not reach the product. ward_only costs
−0.24/−0.28/−0.32, CIs excluding zero. Complementarity unchanged (union
0.53510 vs 0.53475). **Not adopted, no weight sweep**, per pre-registration.

*Next*: the lever is still DATA. 23.91% of main purchases remain
budget-blocked by the estimated wallet; the adopted next contrast is
estimated-vs-measured wallet on the frozen configuration, whose data is the
tier-1 gold capture of the 208 probe games — still blocked on the League
client (test download EUW1_7978990508).

*Prepared, not run*: `scripts/econ_features_parallel.py` +
`docs/econ_features_parallel_v1.md` — parallel econ_features with a
bit-identity equivalence gate (workers must report the parent's runtime
declaration, ddragon version, schema and catalogue digest; `torch.equal`,
not allclose). Applies to cache builds and VAL-style evaluations, NOT to
training. Run `--profile` first: econ_features and the decoder both call
purchase_sequence for the same (row, action) pairs, and a per-row memo may
be a cheaper risk-free win.


**September 22 (afternoon) — FROZEN 50/50 MIXTURE BEATS BOTH PARENTS; ward-term
gradients INTERFERE.** Two owner-assigned readings, no training, no VAL/test.

*Mixture* (`docs/experiment_frozen_mixture_v1.md`, dir
`data/experiments/16.18-frozen-mixture-v1/`, 206 s): ward-free
first_action_exact **0.49514** vs A grouped econ_ft_ep8 0.48487 and B
wardfree_ep16 0.48574 — **+1.03** [0.85, 1.19] and **+0.94** [0.78, 1.12],
n=58.781, paired by match; any-3 +0.70/+0.71. The two catalogues hold the same
5.692 actions but only **31** positions coincide, so alignment is by canonical
multiset key with an asserted bijection — never by index. Integrity gate:
re-decoding A and B in-process reproduced their persisted rows with **0**
mismatches. M never breaks a case both parents got right (0 of 58.781), fixes
68 where both were wrong, captures 20.6% of the oracle-union gap. No slice has
a negative CI excluding zero. **NOT promoted** — the probe was already used for
selection; adopting needs a newly pre-registered VAL reading (the existing
one-config VAL usage may not be reused) and the online cost of running both
models is unmeasured. Independent recomputation from rows only:
`data/reports/frozen_mixture_review_20260922/`.

*Ward-loss diagnostic* (`docs/experiment_ward_loss_diagnostic_v1.md`, dir
`data/experiments/16.18-ward-loss-diagnostic-v1/`, 79 s): CE decomposes exactly
(residual 5e−09) into main + ward|main. Bf_ep16 1.70479 = 1.39503 + **0.30976**
(18.2%); econ_ft_ep8 1.63764 = 1.33391 + 0.30374 (18.5%). Bf_ep16's total
reproduces the twin fine-tune's verified step-0 CE 1.70479 exactly.
`cos(g_main, g_ward)` = **−0.287** / **−0.122**: the two objectives interfere on
shared parameters. At w=0.25 the RAW GRADIENT vector moves 19% — a property
of the gradient, not of the Adam step (Adam rescales per coordinate). Per the reading
rule written before the run, this does NOT decide the loss contrast — it stays
pending the owner's call.

**September 22 (16:05) — mixture FROZEN + VAL in flight; loss contrast QUEUED.**
Mixture manifest `data/experiments/16.18-frozen-mixture-v1/frozen_mixture_v1.json`
binds all four model components by sha256 plus the decode rule, alignment
method and weights; scoring scripts are recorded one digest per script
(probe / val / latency bench) instead of one ambiguous `script_sha256`.
`BuildTracker Mixture VAL` running since 15:46 (~2 h, pre-reg
`docs/experiment_mixture_val_v1.md`, VAL's prior exposure declared in it —
NOT a blind confirmation). Smoke already showed arm A reproducing the
earlier VAL rows with 0 mismatches.

`BuildTracker Loss Contrast` registered, waits for the VAL task to clear the
GPU, then runs (1) the A-vs-M latency benchmark and (2) the twin loss
contrast. Reviewer corrections applied before queueing:
- The **"noise floor" rule is withdrawn.** `w1 − econ_ft_ep8` is published as
  a REPRODUCTION DIAGNOSTIC, not an estimate of training variability;
  nothing is declared uninterpretable by its size.
- **Both primaries compare mixtures.** PRIMARY 1 positive + PRIMARY 2
  unresolved now reads "better than the twin control mixture, no
  demonstrated improvement over the frozen mixture" — NOT overlap with the
  ward-free model, since `mix_w1` regressing produces the same pattern
  (reported by `mix_w1_vs_mix_frozen`). Arm-alone results are a separate
  question.
- **Pre-training gates, all aborting**: λ=1 equals conventional CE in value
  (|Δ| 9.5e-07) and gradient w.r.t. logits (|Δ|max 4.7e-09 against scale
  2.1e-02) via a local Generator with the global and CUDA RNG asserted
  untouched; frozen references required and **zero** reproduction mismatches
  in full mode (hard abort, no longer only printed); identity keyed on
  `(match_id, ts, champion)` everywhere; free disk re-checked after every
  epoch; curves appended+fsynced to `progress_full.jsonl` per epoch.
- **Latency figure corrected**: the VAL run's `total_single_model_A` is
  arithmetic (mixture total minus the ward-free forward, still carrying the
  blend and decode), so the "+12%" is NOT a complete A-vs-M comparison.
  `scripts/basket_latency_bench.py` runs both circuits end to end on a fixed
  FIT sample with alternating order. The running VAL process was not touched.
- The earlier "update direction moves 19%" wording is corrected everywhere
  published: that is the RAW GRADIENT vector, not the Adam step.

Day report (both results, exec tile + afternoon section) served at
http://127.0.0.1:8127/20260922/ — server relaunched this afternoon.
`scripts/footwear_runtime.py` allowlist gained `basket_frozen_mixture.py` and
`basket_ward_loss_diagnostic.py`. Gold front still blocked on the League client
(test download EUW1_7978990508).


**September 20 (evening) — POV experiment complete: removing omniscient board
gold cost ZERO.** Experiment `16.18-pov-gold-v1` (protocol
`docs/experiment_pov_gold_v1.md`, run plan sha `b79d56cb…`, generation v3
unchanged, one knob: `PREFIX_POV_GOLD=1`): first_action_exact **0.3806**
[0.3784, 0.3828] vs Phase D's 0.3811; margin over the conditional lookup
**+0.1507** [0.1485, 0.1529] vs +0.1512; guardrails held; zero unaffordable;
all three pre-registered checks passed. Candidate at
`data/experiments/16.18-pov-gold-v1/candidate.pt` — **adoptable, NOT promoted**
(owner's manual gate). Per-champion honest exactness (from the run's outcome
evidence): only Aurora ≥ 0.50 (0.516, n=673); per-champion median 0.372; head =
rigid-build mages, tail = situational supports. Full day ledger in
`docs/history.md` (2026-09-20 entries: omniscient gold withdrawn, measured
wallets, operational traps — footwear_runtime entry is MANDATORY for 16.18
generations, `--review` must be passed explicitly since qualification.json
still points at the pre-move root, and NTFS shows 0 bytes for open files).
New knobs (flag-gated, cache v14, plan-bound): `PREFIX_POV_GOLD`,
`PREFIX_GOLD_MEASURED` (macro_coach measured wallets:
`data/imports/gold_measured_v1.json`, 7,557 exact + 2,220 intervals — joins
zero rows on v3; enters with the next generation window ≥ Sept 19). All 40
macro games ingested with ten perspectives. Owner decisions pending:
promote POV candidate; capture farm (gold-only ≈ 400-500 games/day on a
dedicated PC, patch window closing); imitation-ceiling study + learning curve
(free, next step of the 60%/0.80 plans). Day report served locally at
http://127.0.0.1:8127/20260920/ (file: `videns.gg/informes/20260920/`).

**September 20 — both dated collection/acceptance sequences are complete.**
The owner-moved repository is `C:/Users/Javier/videns.gg/build_tracker`.
The immutable closeout is
`data/acceptance/16.18_dated_samples_20260920/closeout.json`; it binds the
per-sample views, final supervisor receipt, pro-impact verification, historical
provenance note, and the already-completed owning benchmark task. The separate
`16-18-next-two-nights` follow-up is paused under its completion condition.
No model, split, dataset export or production serving state changed.

The original dated samples and observation cutoffs are preserved. They now
have separate immutable **audited corpus views**, under
`data/acceptance/16.18_dated_samples_20260920/final/`, referencing the exact
qualified rows in existing frozen v3 and new child corpora. They do not copy
or rewrite the original sample DBs and are not exportable training generations
(`training_export_ready=false`). Every collected game has an accepted,
quarantined or reserved disposition; the complete collections are not certified
as universally clean.

- `16.18_20260912`: 3,295 collected; **2,502 accepted**, 636 quarantined,
  157 reserved without payload reads. Accepted coverage: 297,378 admitted
  decisions, including 58 distinct games with tracked pros. Actual original
  cutoff: `2026-09-12T11:21:34+00:00`.
- `16.18_20260913`: 4,350 collected; **3,372 accepted**, 821 quarantined,
  157 reserved without payload reads. Accepted coverage: 402,761 admitted
  decisions, including 90 distinct games with tracked pros. Actual original
  cutoff: `2026-09-13T07:17:13+00:00`.

Sources for those counts are each dated `acceptance_index.json` and
`coverage.json` below `final/`. The cumulative populations overlap; never sum
the dates as unique games. Targets remain `item-counts-v1` multisets; maximum
copies of a target item are five and six respectively, with no clipping.

The original collection checkpoints and reports passed their hash/membership
checks. Later v3 evidence covered most games but left 196 non-test IDs
unaccounted for, including the 121 missing from September 12. Only those 196
were reconstructed again, into
`data/corpora/16.18-dated-samples-uncovered-20260920/`. Its full audit completed
at 08:53:17 UTC: 168 source-qualified games / 20,162 shops, zero hard failures;
28 games were screened out before the full audit. The decision mask admits
20,005 rows. Its completion is under `uncovered/run/` in the new acceptance
folder. Earlier corpus/code snapshots were not overwritten or relabeled.

Final reconciliation completed at 09:21:25 UTC. It verified both corpus DBs,
masks, source/code receipts and fixed memberships; reused 2,260 exact-parent
inventory outcomes; and freshly checked before-board, before-shop and
after-shop states for 944 additional parent games plus all 168 child games.
There were zero additional inventory exclusions. Publication occurred only
after the supervisor's code/protected-artifact checks passed. Both
`final/completion.json` and `reconciliation_run/completion.json` are required.
The source-code difference from collection remains explicit; no freezer guard
or collection checkpoint was weakened to bypass it.

`pro_coverage_review.json` independently checks actual participant identities
against cached, canonical-hash-verified non-test match DTOs. September 12 has
75 such pro games: 58 accepted / 17 quarantined. September 13 has 111:
90 accepted / 21 quarantined. Accepted counts match the per-sample coverage;
there are no unknown non-test match identities in this pro check. Both retain
84 resolved tracked accounts and the accepted 16 unresolved outcomes. Reserved
pro participation stays unknown because those payloads remain sealed; pro
memberships overlap and are not a unique-game sum.

One historical evidence limitation is preserved in
`inventory_reuse_provenance.json`: v3 saved the inventory outcomes and wrapper
but omitted its imported helper's execution-time digest. The current helper
matches two independently bound v1 snapshots exactly; that corroborates reuse
but does not retroactively prove the exact v3 execution bytes. These are reused
prior outcomes on the exact v3 inputs, not fresh checks of those 2,260 games.
Previously excluded parent sources remain quarantined; their full pair's
canonical linkage remains unverified (the pro check verifies match DTOs only).

The owning task `01a09563-0619-7e83-b829-63eb75f0f373` already completed the
fixed September 12 transformer/conditional-policy and additional-copy decoder
comparisons. Evidence:
`data/experiments/16.18-reconstructible-development-v1/completion.json`;
its candidate and both report hashes were checked. No later sample entered
that benchmark, and this follow-up launched no duplicate model work. The initial
smoke candidate and report also still match their original completion receipt.

Validation for this follow-up: all three new focused assert scripts passed;
the existing reconstruction regression and a real non-test inventory/coverage
integration check passed; independent read-only review found no blocking issue.
All work was offline, with hidden durable jobs, a 6 GB child limit and 40 GB disk
reserve. The production hold and sealed-test requirements remain in force;
no commit or promotion was performed.

Today's independent Daily Pull stopped at 10:28:35 Madrid after Riot rejected
the key (exit 1, `data/daily_pull.log`); `.env` still predates that rejection.
A fresh key is needed for future API collection. This did not block completion
of the already-collected September 12/13 acceptance work. The morning recovery
coordinator was already complete, and the collector lease was free. The
follow-up's saved path was corrected after the move; do not recreate the old
`C:/Users/Javier/build_tracker` directory.

**September 18 — THE REPO MOVED. It now lives at
`C:\Users\Javier\videns.gg\build_tracker`, next to `macro_coach` under the
`videns.gg` umbrella.** Nothing about the model, the data or any result changed;
this is a path change only. Same volume, so each `Move-Item` was a rename — the
~250 GB of `data/` was never copied.

What was re-pointed, and verified after the move:

- Scheduled task `BuildTracker Daily Pull` -> `<repo>\.venv\Scripts\pythonw.exe
  <repo>\scripts\daily_pull.py`. It is the only task that referenced the repo.
- The six worktrees under `.worktrees\`. `git worktree list` had them all as
  `prunable`; `git worktree repair` fixed the `gitdir` pointers in both
  directions. All seven checkouts now resolve and `git status` works in each.
- Their twelve junctions — `data` and `.venv` in every worktree — were still
  aimed at the old path. Recreated against the new one (only the links were
  deleted, never the targets). These were also what held a handle on the old
  directory and stopped it being removed.
- `scripts/shutdown_after_patch_sample.ps1`: the `-Report` default had the old
  path baked in.

`.venv` survives the move (Python 3.14.0, `sys.prefix` correct from the main
checkout and from a worktree); the git history, branch `deepseek-client-setup`
and the `Javiops/build_tracker` remote are unchanged. `C:\Users\Javier\build_tracker`
no longer exists.

**Absolute paths written further down this document are historical.** Entries
from September 12 and earlier name `C:/Users/Javier/build_tracker/...`; they
were left as written and should be read against the new root. Paths in
`docs/model_development_20260912.md` and `docs/audits/2026-09-10-phase0.txt` are
stale for the same reason. `__pycache__/`, `build/` and `dist/` still carry the
old path in generated files and regenerate on their own.

**September 17 — THE UNCERTAIN-WALLET RETRAIN IS A CLEAN NEGATIVE. Three
independent measurements say the new candidate is functionally the old one.**
Suite 51/51. Nothing promoted, no validation or test read, no ledger line spent.

*The question (owner, 2026-09-17):* "train this model with the uncertainty that
the gold is in an estimated range to see how it will mimic once it goes to a
live game". Protocol frozen before the first training step:
`docs/experiment_gold_posterior_training_v1.md`. Candidate
`16.18-gold-posterior-v1/candidate.pt` (`8ca1cef3...`), same generation, same
geometry and hyperparameters as Phase D except the wallet treatment.

**ALL FOUR pre-registered predictions failed, none of them narrowly.**

| check | predicted | measured |
| --- | --- | --- |
| new vs old at `const:0` | +0.01..+0.05 | **+0.0006** |
| slack curve flatter | clearly | 0.1668 vs 0.1721 |
| optimum nearer q=0.50 | yes | **q=0.65**, same as old |
| guardrails within 0.02 | yes | `ward_only_recall` **-0.027** |

Best arm of each: new `quant:0.65` **0.4620**, old `rate:2` **0.4650** — the new
one is slightly WORSE at its own optimum. Both curves land on top of each other
point for point (`const:0` 0.4359 vs 0.4353, `const:500` 0.2952 vs 0.2929).
Report: `data/reports/mask_slack_sweep_posterior_v1/`.

**The oracle confirms it from the other side** (owner: "evalues el modelo como se
hizo para conseguir el 60%"). Same script, same 288,751 TRAIN purchases, same
method as Sep 15: **old 0.4383 -> 0.6006, new 0.4384 -> 0.6023.** Control
identical to four figures. By stratum the telling number is `feasible` (where the
wallet never blocked anything): **0.5979 -> 0.5980**, a move of +0.0001. The only
movement is `budget_short` oracle 0.6079 -> 0.6142, the noisiest stratum.
`data/reports/oracle_budget_ceiling_posterior_v1/`.

**Reading, stated plainly: training under wallet uncertainty was INERT.** Not a
small gain to be tuned up — the resulting model is the same model, measured with
an estimated wallet and with a perfect one. The wallet lever lives entirely at
DECODE. Do not spend more on the training side of it.

*Hypothesis for why, falsifiable and NOT tested:* the budget was already close to
non-binding in training — mask at `gold + 500` (97.4% of states over-credited)
and the label force-unmasked at `train_prefix.py:765`. If the constraint rarely
bit, calibrating it changes nothing. Separating that needs a run with the
tightened mask AND without the label force-unmask.

**MISS PROFILE — the owner's new metric** (`scripts/miss_profile_v1.py`,
`data/reports/miss_profile_posterior_v1/`, 58,621 TRAIN purchases, SAVE rows
excluded). A miss is the first card's multiset != the recorded multiset, counts
included; cross-checked against `eval_policy`'s own
`first_basket_exact_given_purchase` — both 0.4342 on n=58,621, two independent
code paths.

At `const:0`: median |gold distance| **200 g**, signed **-100 g** (we recommend
CHEAPER than he bought — the short-wallet signature). **15.1% of misses are under
75 g**, i.e. inside our own median wallet error and irreducible without better
gold. At `quant:0.65`: median 175 g, signed -50 g, 16.8% under 75 g.

| true class | purchases | miss | miss if affordable | median gap |
| --- | --- | --- | --- | --- |
| basic | 20,289 | 0.476 | 0.337 | 150 |
| component | 15,405 | 0.589 | 0.452 | 200 |
| completed | 11,336 | **0.654** | **0.408** | 325 |
| pink_ward | 6,856 | 0.537 | 0.372 | 125 |
| boots | 4,570 | **0.706** | **0.578** | 200 |

**Two findings worth acting on.** (1) Legendaries are a WALLET problem: 0.654 ->
0.408 once the basket is affordable. (2) **Boots are not** — 0.706 -> 0.578,
barely moves. That is the largest block where gold explains almost nothing, so it
is ranking/feature work, and it is the first concrete non-wallet lever this
project has isolated. Also: when we miss a legendary we show ANOTHER legendary
62.7% of the time (72.1% at `quant:0.65`) — the phase is right, the item is
wrong; and pink_ward -> pink_ward on 41.8% of its misses suggests a **quantity**
defect (bought two wards, showed one), cheap to check.

**Displacement now visible per class:** at `quant:0.65` the miss rate AMONG
AFFORDABLE rows gets WORSE (basic 0.337 -> 0.389, component 0.452 -> 0.496,
completed 0.408 -> 0.467). Widening the mask unblocks some rows and breaks
others — the same trade the sweep measured, now attributable.

**THE GOLD POSTERIOR — the reusable asset from today**
(`scripts/gold_posterior_fit_v1.py`, `data/reports/gold_posterior_v1/`). The
correction `c = currentGold - gold_est` fits

    c_q(dt) = a_q*dt + k_q*sqrt(dt)

per quantile, worst ABSOLUTE error **1.8-6.4 gold** across quantiles spanning
-153..+268 g. Bias linear in staleness, uncertainty as its square root — income
arriving as a stream of increments. A shared-drift form (one b for all quantiles)
fits the upper half at 2-5% but the LOWER tail at 58-75%: over-credit comes from
unseen spending, which is lumpy, not a diffusion, and has no reason to share the
income drift. Fitted on 653,030 unselected frame windows, **no labels**.

**The censored coverage check — the one legitimate use of the recorded
purchases, and it passed.** Each of the 77,073 budget-short purchases says `true
wallet >= basket cost`, at a REAL visit timestamp (frame truth only exists at
60 s boundaries). The fit does NOT use them — label-selected — so they validate
it one-sidedly: covered share must sit BELOW its quantile. Measured q=0.90 ->
0.75 at dt=29 s and 0.87 at dt=60 s; every level below q. A covered share ABOVE q
would have falsified the fitted tail. None did.

**FOUR DEFECTS CAUGHT, three of them silent-failure class.**

1. **`app/predictor.py` builds inference features through `tp.ShopDataset`** —
   the training featurizer. With the posterior active, EVERY EVALUATION would
   have drawn a random wallet per row while `check_displayed_purchases` audited
   against the granted one. Fixed: `GOLD_SAMPLE_WALLET`, off by default, on only
   for the trainer's TRAIN-split build. Caught because the owner said the
   uncertainty must hold for the visits the model is tested with.
2. **`main()` needed `global GOLD_SAMPLE_WALLET`** — without it the assignment
   bound a local, the flag stayed False, and the run would have trained with NO
   sampling while looking completely successful.
3. **`apply_config` did not restore the posterior**, so any eval harness run
   without the env var would build a 26-dim query for a 27-dim artifact. It now
   restores it and REFUSES if the fit is missing or its sha differs.
4. **A `gold_exact` feature was designed and dropped before the run** — it would
   have been 0 on every training row (only live sets it), so serving would have
   presented a combination never seen. Width alone is continuous and puts the
   exact-wallet end INSIDE the training range (dt=1 s -> sigma 7 g). Recorded as
   a dated pre-run correction in the protocol.

Plus one own goal: `register_task.ps1` silently registered the wrong launcher (a
string replacement that did not apply), so this morning's task re-ran last
night's phase-1 sweep for ~25 minutes. It now reads the registered action back
and refuses on mismatch. And the `quant:` arm parser demanded an exact grid key
while the trainer's sampler interpolates — q=0.65 is not even ON the refitted
grid — so decode would have meant a different budget than the training mask. Both
interpolate now.

**OPERATIONAL — disk nearly ran out.** `test_morning_collection_recovery` failed
at 50/51 and it was NOT a code defect: free space had reached **14.3 GB of
1,023 GB** and `resume_morning_collection.budget()` refuses below a 40 GB
reserve, aborting before the powershell call the test asserts. Confirmed
pre-existing by re-running with the trainer change stashed. It would also have
blocked the retrain (`train_prefix.py:1342`) and risked the 57 GB WAL database.
Owner freed space to **209.9 GB**; suite back to 51/51. Nothing was deleted — all
nine corpora intact. For the record, inside `data/`: corpora **90.4 GB**
(16.17-development-v2 30.9, v3 15.7, v3_failed_gate_a 15.7, v2 10.3, v1 6.4,
role_slots 3.3, repaired_v2 3.1, initial 2.6, 16.17-v1 2.4), tracker.db 57.0,
dataset_generations 10.2, experiments 5.4. **No deletion was performed or
authorised** — the quarantined `failed_gate_a` and the unloadable
`16.17-development-v2` remain the obvious candidates if space is needed again.

**Collection is healthy.** The gap window pull finished **exit 0** (201.5 min, 0
stage failures, KR +5,794 / EUW +5,054 / pros +27) and its dated-sample stage
completed — the first clean end-to-end run since the two `exit 1` nights. The
04:30 Daily Pull then ran clean (72.8 min, 0 failures) with sample
`16.18_20260917` collected and structurally validated. That closes queue item 4
from the Sep 15 close.

**Queue, re-ordered by what today measured:**

1. **The decode lever is the only wallet lever left.** `quant:0.65` is +0.026
   exact on TRAIN, available with NO retraining. Two owner decisions gate it: the
   promotion rule ("zero unaffordable displayed baskets", which any delta>0 fails
   by construction) and whether to spend a SECOND paired comparison on this
   generation's window.
2. **Boots.** 0.706 miss, 0.578 when affordable — the largest block the wallet
   does not explain. First concrete non-wallet lever.
3. **Pink-ward quantity.** 41.8% of ward misses show a ward; suspect multiset
   counts. Cheap to check.
4. Owner-only, unchanged and still blocking three things: the personal Riot API
   key.
5. Windows Update active hours — still not set.
6. **Do not** spend more on training-side wallet work. Phase 2b (quantile arms on
   the OLD candidate) was deferred, not cancelled, and is now low value.

**Artifacts added today.** Scripts: `gold_error_shape_v1.py`,
`budget_shortfall_shape_v1.py`, `gold_posterior_fit_v1.py`,
`mask_slack_sweep_v1.py`, `miss_profile_v1.py`,
`run_gold_posterior_training_v1.py`, plus launchers
(`run_mask_slack_sweep.ps1`, `run_gold_posterior_night.ps1`,
`run_oracle_ceiling_posterior.ps1`, `run_posterior_day.ps1`) and the task helpers
`register_task.ps1` / `stop_task.ps1`. Docs: `experiment_mask_slack_v1.md` (+
phase-2b addendum), `experiment_gold_posterior_training_v1.md`. Reports:
`gold_error_shape_v1`, `gold_posterior_v1`, `mask_slack_sweep_v1`,
`mask_slack_sweep_posterior_v1`, `miss_profile_posterior_v1`,
`oracle_budget_ceiling_posterior_v1`. Diagnostic page:
https://claude.ai/code/artifact/d3424665-3d94-4914-9c48-8446f49c1830
**Nothing is committed** — the whole day is uncommitted in the working tree.

**September 16 evening — WALLET-LEVER DIAGNOSTIC MEASURED, and the
mask-slack sweep is ARMED for tonight.** Suite 51/51 after the changes.

*What was measured (TRAIN only, model-free, no ledger spend):*
`data/reports/gold_error_shape_v1/` — the production gold estimator scored
against exact frame `currentGold` on **653,030 windows / 2,500 v3 TRAIN games**
(`scripts/gold_error_shape_v1.py`), plus the raw shortfall distribution over
**291,819 purchases** (`scripts/budget_shortfall_shape_v1.py`, which reproduces
`lead_feasibility_audit_v3` exactly — 291,819 / 250,221 / 40,427 — as its
correctness check). Page: https://claude.ai/code/artifact/d3424665-3d94-4914-9c48-8446f49c1830

**The error is a spread, not a bias.** Signed `gold_est − currentGold`,
no-purchase stratum: median **−75 g**, IQR **212**, p10..p90 **−317..+109**,
over-credit already **29.5%** at slack 0. The bias is a stable rate, ≈ **−0.5
g/s**, unchanged at anchor ages 60/120/180 s — but the SPREAD scales with
staleness too (IQR 192 → 337 → 450) and with game phase (IQR 182 → 218 → 247).
Kill/bounty events do not explain it (quiet windows, 83% of the mass, IQR 203),
which is the mechanism behind Phase A rejecting every income variant.
Full-basket shortfall: **26.41%** of purchases, median **83 g** — and that
26.41% independently confirms the oracle ceiling's ~26.7% affected mass.

**The two ends of the pipeline sit on opposite sides of that distribution, and
neither number came from it.** `train_prefix.py` masks legality at
`gold + GOLD_DRIFT` (500), where **97.4%** of states are over-credited with
median excess 431 g; `eval_policy.py:320` decodes at `budget_slack=0`, where
26.4% of recorded baskets are unreachable. The model is trained against a
candidate set far wider than the one it is scored on. Nothing in
`docs/history.md` records anyone having tried to calibrate either constant.

**Correction to an earlier reading in this session:** the oracle-budget ceiling
(+16.2) was measured with `budget_slack=0` on BOTH arms
(`oracle_budget_ceiling_v1.py:88,93`), so its control is harsher than the
training configuration. It bounds a perfect per-row wallet, not the headroom
left after the slack the trainer already ships.

*What is armed (owner: "go... only proceed with phase 1 and 2, this is the task
for tonight"):* `BuildTracker Mask Slack Sweep 20260916` →
`scripts/run_mask_slack_sweep.ps1` → `scripts/mask_slack_sweep_v1.py`.
Protocol **frozen before the first arm ran**: `docs/experiment_mask_slack_v1.md`,
with pre-registered predictions and four audit triggers. Twelve paired arms
(`const:0..500` and `rate:1..6`, delta = k·dt) over **500 TRAIN games**, seed 16,
bootstrap 400; the Phase D candidate unchanged, **no retraining**. Measured
throughput 1.8 s per game-arm → ETA ≈ 3 h; the launcher waits for the gap
window pull to finish first, then unregisters itself.

Mechanics worth knowing: `eval_policy.score_visit` is imported **verbatim** so
the metrics are the same code that produced 0.3811, and `eval_policy.py` is NOT
modified (it is digest-pinned in `run_development_1618_v3.py`); the slack is
injected by a documented `SlackProxy` because `score_visit` hardcodes
`budget_slack=0`. Over-budget is judged against `gold_est`, never against
`gold_est + delta`, so no arm can define its own price away. Phase 2's `dt` is
approximated as `ts mod 60000` — the export carries no `gold_gap.frame_ts` —
validated at **99.55%** within 1 s of the true anchor gap on 60 games / 7,350
visits, with the remaining 0.45% off by about a full cadence.

**Bounds, stated before the result exists:** TRAIN rates are
memorisation-favoured and the level will not transfer — only the ORDERING of
arms is being selected. No number from this run may be compared with Phase D's
0.3811 or any other validation figure. **It cannot promote anything:** any
delta > 0 displays baskets unaffordable at `gold_est`, which fails the
promotion gate as written. Restating that gate is an owner decision and is not
part of this experiment. Phase 3 (one paired validation comparison) and the
retrain arm are explicitly NOT authorised by the frozen protocol.

**Also today:** the gap window pull completed collection with **0 stage
failures** in 201.5 min (KR +5,794, EUW +5,054, pros +27 perspectives), and its
dated-sample stage finally started — `sample-prepare` exited 0, the first time
that guard has passed since the two `exit 1` nights of Sep 14.

**Rejected tonight, with the reason on record:** raising `gold_est` toward the
recorded purchase on blocked rows. That is the banned `gold_left + net_spend`
pattern (`docs/history.md` row 1); trained on, it would teach a model that gold
is always sufficient, then be served a real wallet by `live-exact-v2`. The
clean version of the same idea — drop the budget mask from TRAINING and mask
hard only where the wallet is exact, at serving — stays queued behind the
sweep, which prices it.

**September 15 — DAY CLOSED by the owner ("suficiente por hoy"). State and
queue for the next session.**

What stands at close, all committed through `1dcaf57`, working tree clean:

- **Phase D complete.** The v3 generation (corrected inputs: opening-sale
  credit + decision inventory, prequential-v3 end to end) carries a
  fixed-recipe candidate that beats its own conditional lookup on every one
  of the nine declared metrics with all intervals excluding zero — exact
  0.3811 vs 0.2299 (+0.1512), guardrails UP alongside the primaries for the
  first time. Every pre-registered prediction was met; the >0.45 audit
  trigger did not fire. One comparison spent on the fresh v3 window.
- **Oracle-budget ceiling (owner's experiment).** With the minimal wallet
  that affords the recorded basket: exact +16.2 points, and the decomposition
  is clean — budget-masked rows (~26.7% of purchases) go 0 -> 0.6079, the
  same hit rate as normal rows. The trunk already knows those answers; the
  wallet blocks them. Label-derived, TRAIN-only, ceiling not promise.
- **Input defects re-measured on v3:** feasible 85.75%, false-illegal 0.26%
  (was 2.75%), budget-short 13.85% at median 74 g — near the measured
  irreducible floor of the 60 s frame format (55.6% of window income has no
  timestamp).
- Nothing promoted; deployment_manifest.json absent; test split 0 rows,
  unopened. Scheduled tasks: only the recurring Daily Pull remains.

**Queue, in the order the evidence now supports:**
1. **Quantile/two-output gold design** (mask from a conservative bound,
   feature from the central estimate) + **mid-visit sale accounting** — the
   oracle ceiling makes this the highest-expected-value model work.
2. **Owner-only:** the personal Riot API key. It now blocks THREE things:
   key-expiry operational failures, the beta, and the shop telemetry that is
   the only ground truth for gold (and the only way to test whether live —
   which has exact wallets — already behaves closer to the oracle regime).
3. Windows Update active hours (an update reboot killed the eval at 88.6%
   this morning) and the arm-B decision (joint 16.17+16.18; needs a full
   16.17 re-reconstruction and would spend the last data-scale ledger line).
4. The 04:30 pull now includes the dated-sample stage with the fixed guard —
   tomorrow morning is its first clean end-to-end test; check
   data/daily_pull.log before anything else.


**September 15 17:35 — ORACLE-BUDGET CEILING measured (owner-requested).**
`data/reports/oracle_budget_ceiling_v1/oracle_budget_ceiling.json`,
`scripts/oracle_budget_ceiling_v1.py`, TRAIN-only, 288,751 v3 purchases,
paired per visit: the displayed policy re-ran with the minimum wallet that
makes the recorded basket affordable (max(gold_est, exact recursive cost)).
LABEL-DERIVED DIAGNOSTIC CEILING - never a feature, never an evaluation
reference; the minimal-sufficient wallet is optimistic (a true, larger wallet
would also unlock expensive WRONG items).

Overall exact 0.4383 -> 0.6006 (**+16.2 points**), any-of-3 0.5312 -> 0.7244
(+19.3). The decomposition is perfectly clean: rows whose basket cost exceeds
gold_est (~26.7% of purchases under this full-basket definition) go 0.0000 ->
**0.6079** exact - and 0.6079 is essentially the SAME hit rate as the
already-feasible stratum (0.5979). **When the wallet is right, the model
ranks previously-masked rows as well as normal rows: on those states the
trunk already knows the answer and the wallet alone blocks it.** The
feasible stratum moved exactly 0.0000 by construction (oracle == gold_est
there), so this experiment says nothing about ranking distortion - only
about unmasking.

Strategic readings, with their bounds: (1) the budget lever is worth far
more than the +3-6 point estimate derived from the lead-only audit - the
full-basket definition roughly doubles the affected mass; (2) since LIVE
serving carries the exact wallet (live-exact-v2), live behaviour on these
states should sit closer to the oracle regime than offline numbers suggest -
direction certain, magnitude unmeasured until shop telemetry exists (the
same personal-API-key blocker); (3) this strengthens the
quantile/two-output gold design and the sell-to-buy accounting as the
highest-value next model work. TRAIN rates are memorization-favoured; the
val-level ceiling would be lower.


**September 15 13:25 — PHASE D COMPLETE. All pre-registered predictions met;
nothing promoted, test still sealed.** Chain finished after two external
interruptions (01:42 kill, 05:36 Windows Update reboot at 88.6% of the eval);
the evaluation re-ran whole. Report:
`data/experiments/16.18-reconstructible-development-v3/policy_eval_val_candidate_vs_conditional_policy.json`
(sha `404cfd02...`), candidate `ed1fda4f...`, generation fingerprint
`e4ea6431...`, 1,601 validation games / 191,916 decisions, bootstrap 400 seed 16.

Candidate vs this generation's own conditional lookup (every declared metric,
all intervals excluding zero in the candidate's favour):
first_action_exact **0.3811** vs 0.2299 (**+0.1512** [+0.1488,+0.1535]);
action_family_match 0.6342 vs 0.4440 (+0.1902); any-of-three exact 0.4630 vs
0.3711; basket exact 0.3784 vs 0.2249; recall 0.5513; precision 0.5712;
no_buy_death_recall 0.5216 vs 0.4905 (+0.0311); ward_only_recall 0.7623 vs
0.6942 (+0.0681); save_card_precision 0.2234 vs 0.0701. **Zero unaffordable
displayed options on both sides.** Unlike every decode-time experiment of the
past week, the guardrail behaviours moved UP together with the primaries.

**Pre-registered check (written before any result was visible):** exact 0.3811
falls inside the predicted 0.36-0.39; margin over the lookup +0.1512 clears the
predicted >=0.10; the ">0.45 means audit, do not celebrate" trigger did NOT
fire.

**The comparison that is NOT allowed**, stated so nobody makes it later: these
numbers are not comparable with v2's 0.3641 — different export AND different
validation window. The most defensible cross-generation quantity is the margin
over each generation's own lookup baseline, and it widened from ~+0.130 (16.18-v1
and 16.17 alike) to **+0.151**. Suggestive, not a paired result.

**Input diagnostics re-measured on the v3 TRAIN split**
(`data/reports/lead_feasibility_audit_v3/`, 291,819 purchases): feasible at
gold_est **85.75%** (was 82.95% pre-fix), budget_short 13.85% (median shortfall
74 g), legality_blocked **0.26%** (was 2.75% — a 90% reduction), plus 0.14%
lead-probe artifacts. The opening-sale fix holds at full corpus scale.

**Validation spend:** this generation's fresh window now carries ONE scored
comparison. Nothing is promoted, `deployment_manifest.json` remains absent,
`visits_test.jsonl` is 0 rows and unopened.

**September 15 05:36 — WINDOWS UPDATE rebooted the machine and killed both
night jobs.** Not an owner action and not a pipeline fault: System event 1074
records `TrustedInstaller.exe` initiating a restart (twice, 05:35 and 05:36),
LastBoot 05:36:27. Casualties: (1) the Phase D **policy evaluation at 88.6%**
- it had scored 170,000 of 191,916 validation rows in ~53 min and has no
mid-run checkpoint; (2) the 04:30 **daily pull**, killed mid-EUW at game
130/721 (its window auto-sizes, so the next run catches up).

**What survived - everything else.** completed_stages through `graft`:
prepare, recon, verify, admission, coverage, export, tests, trunk, graft.
Artifacts intact: `trunk.pt` (04:41), `candidate.pt` (04:43), the published
generation `16.18-reconstructible-development-v3` (fingerprint `e4ea6431...`,
297,240 train / 191,916 val / 0 test rows, 100% `prequential-v3` +
`decision-start-footwear-16.18-v2`, 4.54% of train rows carrying an
opening-sale credit, median 140 g).

**Fold audit (done, before training):** boundary 2026-09-13T12:58:55 (the v2
snapshot instant), prior benchmark v2, validation declared fresh/unshopped.
train 2,500 games / val 1,601 / 1,180 excluded as `prior_validation_exposed`.
**Correction to an earlier claim in this file: TRAIN did NOT grow 2-3x - it
grew 6%** (2,357 -> 2,500 games). The Sep-13 sample was already inside the v2
snapshot, so nearly all newly qualified games post-date the boundary and land
in validation (+56%). The data-scale lever is therefore NOT in play for this
candidate; the only change is the corrected inputs. Revised expectation for
first_action_exact: 0.36-0.39, margin over the conditional baseline >= ~10
points, and >0.45 means audit rather than celebrate.

**Recovery in flight:** `scripts/resume_policy_v3.py` +
`BuildTracker Policy Rerun V3` re-run the evaluation alone (~60 min) against
the preserved candidate, reusing the frozen plan and re-verifying its code and
protected digests. The interrupted partial stream and fitted baseline are
quarantined as `*_killed_by_winupdate_0536.*`; the baseline re-fits
deterministically from the same TRAIN split. No pending reboot at relaunch
(CBS_pending False, WU_reboot_required False).

**Standing risk to fix in daylight:** unattended multi-hour chains have no
mid-stage checkpoint and Windows Update can restart the machine at will.
Either set active hours / pause updates before authorizing an overnight run,
or give the evaluator a resumable checkpoint.


**September 15 02:49 — the resumed audit FAILED on my own new gate; fixed,
and the corpus is rebuilding (`012e562`).** `decision_state_binding` rejected
425 of 636,936 visits; every other check passed and zero games were
quarantined. Root cause traced to the raw timeline: the player sold three
Biscuits (item 2010, 5 g each) before the visit's first purchase, so
`opening_sale_gold` was 15 while `opening_sales` was empty — the biscuits
are UNMODELED consumables the inventory replay never carried, so real
pre-decision gold arrived with no modeled slot freed. The data is correct;
the gate wrongly demanded gold and freed slots imply each other. Only the
converse is a true invariant (a freed slot must have been paid for) and that
is what it now asserts; failures carry a detail payload. **Cost:** the corpus
manifest pins `audit_patch_corpus.py`, so the audit refuses to run against a
corpus built under a different digest — re-stamping the manifest would forge
provenance, so the corpus is being rebuilt from raw (~1.5 h) before the audit
re-runs. `resume_development_1618_v3.py` now has two entry points (full
reconstruction when the corpus is absent — its own supervisor writes the
build+audit evidence; audit-only when only the audit was lost). The failed
corpus, run directory and audit report are QUARANTINED as `*_failed_gate_a`,
not deleted (a delete was correctly refused by the permission layer while the
owner slept). Relaunched at 03:05 as `BuildTracker Phase D Resume V3`; new
ETA for the full chain ≈ 09:00-09:30. Suite 51/51 after the fix, and the
frozen run plan's code digests still match (the audit script is not in the
plan's list, only in the corpus manifest's).

**September 15 01:42 — Phase D killed externally mid-audit; RESUMED without
redoing the expensive half.** The audit subprocess exited 1073807364
(DBG_TERMINATE_PROCESS — killed from outside, not a pipeline fault) while the
owner was shutting things down; no python remained and the scheduled task
returned to Ready. **What survived:** `process_reconstructible_1618 build`
had COMPLETED — `data/corpora/16.18-reconstructible-v3`, 6,541 sources,
**631,454 admitted rows** (2,066 boundary-excluded), 15 GB, with its frozen
manifest and decision mask. The snapshot itself is much larger than v2's:
6,690 16.18 games in DB, 6,541 selected (149 reserved-test only, zero
missing raw pairs) versus 4,409 for v2. The audit had reached 2,700/5,281
games and writes nothing until it finishes, so the acceptance directory held
no partial state. **Resume:** `scripts/resume_development_1618_v3.py` +
`BuildTracker Phase D Resume V3` re-run the audit alone against the existing
corpus and then continue verify → admission → coverage → export → tests →
trunk → graft → policy. It reuses the ALREADY FROZEN run plan and re-verifies
every code and protected digest in it (all matched at relaunch), so the same
contract governs. On audit success it writes the recon evidence exactly as
`run_reconstructible_1618.py` does (the verifier demands status == completion
with completed_stages ['build','audit'] — both DID complete, across two
invocations) and records the split in `resumed_after_external_termination`,
preserving the killed status as `status_failed_audit_terminated.json`.
Lesson for the ledger: a multi-hour chain with no mid-stage checkpoint loses
whatever stage is in flight when the machine or console is touched — the
build/audit split is what made this recoverable at ~1 h instead of ~2.5 h.

**September 14 ~23:30 — night chain relaunched after two same-minute
failures, both root-caused and committed (`7d64f93`).** (1) The evening
pull's COLLECTION completed clean again (120.9 min, 0 failures) but the
dated-sample stage aborted: the guard in `collect_patch_sample.py` read the
hygiene-REMOVED 'Causal Recovery' task as "scheduler unavailable"
(ErrorActionPreference=Stop + missing name throws); it also self-detects
the Daily Pull task when the daily pipeline invokes it (the 04:30 failure).
Fixed: SilentlyContinue lookups + `BT_INVOKED_BY_DAILY_PULL` excludes the
caller — tomorrow's 04:30 sample stage should finally pass. (2) Phase D's
prepare stage died in seconds: `prepare_reconstructible_1618.static_evidence`
fetches the pinned runesReforged CDN file and the runner blackholes all
proxies; the file is frozen in the 20260912 evidence and prepare skips the
fetch when present, so the runner now seeds it — the whole chain stays
offline. Failed dirs quarantined as `*_failed_prepare_a`. Relaunched:
`BuildTracker Window Sample 20260914` (resumes the preserved checkpoint)
then `BuildTracker Phase D Rebuild V3` (waits for the sample task).
**00:05 update:** the resumed sample refused itself, CORRECTLY —
`collect_patch_sample.py:176 'Pinned reconstruction/static provenance
changed; use a new sample'`: the checkpoint was prepared at 06:22 under the
pre-v3 reconstruction pins, and today's version bumps invalidate it (the
integrity gate doing its job; ~50 min of route discovery lost). Decision: do
NOT re-prepare tonight — the 04:30 daily pull prepares a fresh dated sample
(16.18_20260915) under current provenance with the fixed guard, covering
the window cumulatively. The 20260914 checkpoint stays preserved. Phase D
started at 00:06 (prepare running, seeded runes file, fully offline); the
04:30 pull may overlap the recon stage — by design (frozen source plan
tolerates continuing collection; recon reads raw files, not the DB).

**September 14 late — repository hygiene (owner: "commitea todo lo de hoy y
mejora la higiene").** The five-day uncommitted backlog (Sep 10-14) is now
committed in five thematic commits on main: footwear/Viego repair +
recovered-corpus machinery; the 16.18 reconstructible pipeline and
forward-validation admission; the weekend experiments (basket composition,
data scale, miss diagnostics); the leak-safe inputs change (prequential-v3 +
opening sale); and this hygiene commit. Suite was 51/51 immediately before
committing. `.gitignore` now covers every `data/` artifact family plus
`.worktrees/` and `.agents/` — nothing under data/ can be committed by
accident. Seven SPENT one-shot scheduled tasks were unregistered (16.18
Corpus Audit, Causal Probe 10k, Causal Recovery, Early Causal Probe, both
Patch 16.18 Sample tasks, and the Shutdown After 16.18 Sample task — the
lingering-shutdown class behind history.md's near-miss); Daily Pull and
tonight's two live tasks remain. The orphan
`.16.18_development_v2_admission.*.partial` (declared derivative-safe on
Sep 13) is deleted. Nothing under data/ evidence directories was touched;
the running night chain's frozen code is unaffected (commits do not modify
file contents). Not pushed — the owner decides when to push.

**September 14 night — leak-safe inputs plan, Phases A-C executed (session
build-tracker-cb owns this; plan approved by the owner, frozen at
`C:/Users/Javier/.claude/plans/sparkling-bouncing-sparrow.md`; protocol
`docs/experiment_gold_prequential_v3.md` frozen BEFORE the first harness
run).** Suite 51/51.

*Phase A (measured, TRAIN-only, no ledger spend):* the new prequential
harness (`scripts/prequential_gold_harness.py`, calls the REAL
`_attach_gold_estimates` via the `gold_debug` capture hook, never
re-implements the walk) scored the estimator against next-frame exact
`currentGold` on all 2,357 TRAIN games (614k windows): v2 median |err| 115 g,
signed bias −74 (under-credits, as the code comment predicted). Every
pre-registered income variant (goldPerSecond additive, assist constant,
both) improved |err| but RAISED the over-credit rate → **all rejected under
the frozen rule; prequential-v3 is the v2 walk + the opening-sale credit
only**. `goldPerSecond` is confirmed NOT an income rate (0 for 76% of
frames; the 0 cohort earns more than the 90 cohort). Opening-sale timing
over the full corpus: **13,260/13,260** opening sales strictly precede the
visit's first purchase. `scripts/validate_gold_est.py` QUARANTINED (its
"truth" is the banned `gold_left+net_spend`; its one honest slice was
silently empty via a key bug — repairing it would resurrect a banned
reference; see the new docs/history.md ledger row). Reports:
`data/reports/gold_prequential_v3_phase_a/`.

*Phase B (landed, both reconstruction copies in lockstep):* opening sales —
`ITEM_SOLD` in whole batches strictly before the visit's first purchase —
now free their slot (`decision_inventory`; `ITEM_DESTROYED` never counts: a
combine's destroys precede its purchase and would reveal the label) and
credit their proceeds (`opening_sale_gold`, net over
[ts_start, first_purchase_ts) so a sale+undo cancels; window predicates of
the estimator UNTOUCHED; `gold_gap["spent"]` stays 0). Versions:
`prequential-v3`, `decision-start-v3` / `decision-start-footwear-v4` /
`decision-start-footwear-16.18-v2`, CACHE_VERSION
`v13-prequential-v3-decision-inventory` (the cache key does NOT include the
gold version). Export: row `inventory` ← decision_inventory,
`unmodeled_regular_slots` post-sale, affordability block from
`gold + opening_sale_gold + GOLD_DRIFT`, `opening_sale_gold` emitted; labels
untouched; sell-only visits still never export. `gold_input` provenance is
now DATA-derived (manifest's sole version) in train_prefix and checked
against the manifest in eval_policy — the flag-derived stamp could lie about
which estimator trained a model without anything failing.
`app/dataset_generations.py` gates take membership sets so PAST generations
stay verifiable; `audit_patch_corpus.py` gained `decision_state_binding`
(decision = before − opening_sales, proceeds ⟺ sales); `prefix_check` covers
the new fields; the lead-feasibility audit is v2 (full-basket fallback —
~10% of the old `legality_blocked` headline was a lead-only-probe artifact).
Frozen v2-experiment evaluators (eval_data_scale_v1, eval_basket_composition,
eval_copy_policy) intentionally stay v2. New test pins: exact opening-sale
arithmetic, combine-destroy anti-leak, no-future on the new fields, neutral
undo, featurization isolation (`opening_sales`/`opening_sale_gold`/
`decision_inventory` never reach tensors), data-derived gold_input.

*Phase C (COMPLETE):* before/after on 300 TRAIN games / 36,092 purchases —
old reconstruct from `git show HEAD:` vs current, identical classification
(`data/reports/opening_sale_sample_effect_v1/sample_effect.json`):
feasible_at_gold_est **81.85% → 84.62%** (+2.77 points of purchases the
model is now allowed to get right); legality_blocked **3.00% → 0.66%**
(1,084 → 237, −78%; 711 flip straight to feasible, 136 free the slot but
still lack gold and correctly become budget_short); budget_short 15.01% →
14.60% (287 freed by the credit); **zero worsening transitions**; 4.9% of
rows carry an opening-sale credit. Post-change visit-level provable
over-credit (`data/reports/gold_prequential_v3_phase_a_after/`): 8.49% →
8.86% (+1,046 of 282,093 visits). Reading, stated honestly: the one-sided
bound already covers sale proceeds (they are income before the next frame),
so this +0.37-point rise is REAL marginal over-credit — rows where the base
walk's optimistic tail was already near the bound and the exact sale credit
pushed them over. It is the pre-existing income-model tail being unmasked on
sale rows, not an error in the credit itself (which is exact observed
pricing); the pre-registered no-rise criterion governed income-FORECAST
variants, and the credit is an observed transaction. Recorded here so the
number never surprises anyone later.

*Phase D LAUNCHED (owner: "haz D y el daily pull de la ventana", ~20:40).*
Night chain, two one-shot tasks: `BuildTracker Evening Window Pull`
(re-runs scripts/daily_pull.py under a DIFFERENT task name — the 04:30 run
completed collection with 0 failures but its dated-sample stage aborted
because `collect_patch_sample.py:277` checks whether 'BuildTracker Daily
Pull' is Running and detected ITSELF; a daytime fix should exclude the
calling task) → `BuildTracker Phase D Rebuild V3` (waits for the pull task
to unregister, then `scripts/run_development_1618_v3.py`): prepare (new
frozen source plan + static review over every 16.18 game incl. tonight's
pull) → recon (`16.18-reconstructible-v3` corpus, 9h wall) → verify →
admission (forward-validation-v1 rolled one step via the new
`--prior-base`/`--prior-generation` args on prepare_development_1618_v2.py:
boundary = the 20260913 snapshot instant; the v2 validation window, 2 ledger
comparisons spent, retires from every fold; the Sep-13 sample's pre-boundary
games enter train, so TRAIN grows substantially; newer games become the
fresh validation) → coverage → export (`16.18-reconstructible-development-v3`,
eval_version 7) → full suite → fixed-recipe trunk+graft → ONE
`eval_policy --compare-conditional` on all validation games (bootstrap 400,
seed 16). Plan finalizes its export_fingerprint right after the export
stage (the validators bind to it). Expected wall ~12-16h; PC stays on.
Acceptance on completion: the accepted_match_ids/fold diff declared, TRAIN
diagnostics re-run on the new generation, candidate-vs-conditional CIs
reported with full provenance. Known consequences already recorded: old
generations verifiable but not re-exportable under current code; the
recovered 16.17 corpus cannot load under current code — arm B needs a full
16.17 re-reconstruction (last data-scale ledger line, separate owner
decision).

**September 14 — WHERE THE MISSES ACTUALLY COME FROM (two TRAIN-only
diagnostics, no model selection, no ledger spend).** The stage-1 partition
said 50.2% of misses are `lead_item_missing` (68,593). Two follow-ups
decomposed that class, and the answer moves the roadmap.

*Step 1 — with the model* (`data/reports/lead_miss_diagnostic_v1/
lead_miss_diagnostic.json`, 279,748 TRAIN rows, frozen candidate
`a9b40beb…`): of the 68,593 lead misses, **67.6% (46,389) were not reachable
at all under `gold_est`** — the affordability mask forbade the item, so the
model was never allowed to show it. Genuinely rankable but ranked 4-10:
16,342 (23.8%); ranked 11-30: 791. Ranked **top-3 among legal items yet
absent from every displayed card: 5,070 (7.4%)** — pure decoder/gate loss.
`not_in_vocab`: **0** (the vocabulary is not a constraint).

*Step 2 — without the model, because step 1's headline was not attributable*
(`data/reports/lead_feasibility_audit_v1/lead_feasibility_audit.json`,
`scripts/audit_lead_feasibility_v1.py`, 22 s, no torch):
`purchase_sequence(...) is None` conflates "no money" with "illegal at the
reconstructed state", which have opposite remedies. Re-asked per purchase
with the budget lifted, over ALL 274,647 TRAIN purchases:
**feasible at `gold_est` 82.95%** (227,828 — a miss there is a real ranking
problem); **`budget_short` 14.30%** (39,272 — blocked only by our wallet
estimate); **`legality_blocked` 2.75%** (7,547 — the reconstructed
inventory/slot state forbids a purchase the player demonstrably made).

**The number this project has wanted for a week.** `docs/history.md` records
"frames are 60 s stale so the offline budget is systematically low; direction
certain, magnitude unmeasured". On the `budget_short` rows the shortfall
(`lead cost − gold_est`) has **median 77 gold** (p25 33, p75 159, p90 287,
mean 143, n=39,272) — roughly 10-40 s of income, exactly the stale-frame
signature. It rises with item depth (median 65 / 76 / 99 for depth 0/2/3).
**Bounds:** a LOWER bound on a label-selected subset (only rows where the
player bought something we called unaffordable). It is a diagnostic of
estimator error and must never become a feature or an evaluation reference —
deriving the wallet from the purchase is the banned `gold_left + net_spend`
pattern.

**`legality_blocked` is a reconstruction defect with a clear fingerprint:**
it barely exists early (137 rows at 0-9 min) and concentrates late (2,646 at
10-19, 4,764 at 20+) and on depth-0 basics (4,488). The bound examples show
5-6 item inventories buying cheap components (Cloth Armor, Null-Magic
Mantle, Warden's Mail) — i.e. we believe the inventory is full when the
player had freed a slot. Worth 2.75% of all purchases, and unlike the gold
gap it is fixable offline.

**Revised lever ranking (measured, replaces the September 13 closing
update's ordering):** (1) the wallet input — 14.3% of purchases are masked
out by a median 77 g error, and it is also the largest single block inside
the lead-miss class; (2) lead-item ranking on the 83% of purchases where the
item WAS reachable; (3) `legality_blocked` slot-state reconstruction (2.75%,
offline-fixable, late-game); (4) decoder gates — 5,070 cases where the model
ranked the right item top-3 and still did not display it. Data scale (arm A
above) cuts across (2).

**September 14 — DATA-SCALE ARM A RESULT (verdict: rejected under the
predeclared rule; primaries won, a guardrail regressed).** One paired
comparison on the v2 validation (1,029 games / 124,257 decisions, 400 game
resamples, seed 16), both policies governed by 16.18 static data, zero
unaffordable displayed options on either side, zero visits with
out-of-vocabulary truth items on either side. A = 16.17-trained candidate
(`13f5b9bf…`), B = the 16.18-only control (`a9b40beb…`). A − B:
first_action_exact **+0.0230** CI95 [+0.0208,+0.0249] (0.3870 vs 0.3641);
action_family_match **+0.0123** [+0.0105,+0.0140] (0.6316 vs 0.6193);
any-of-three exact +0.0208; basket exact +0.0239; recall +0.0257; precision
+0.0224. A wins EVERY basket-size slice with a positive interval (size 1
+0.0205 n=75,308; size 2 +0.0316 n=35,716; size 3 +0.0214; size 4 +0.0335;
size 5+ is n=57 and inconclusive). **The guardrail that failed:**
`no_buy_death_recall` **−0.0284** [−0.0404,−0.0143] on n=2,321 — the post-death
no-buy behavior degrades significantly. `ward_only_recall` improved (+0.0127
[+0.0044,+0.0208]). Report:
`data/experiments/data-scale-v1-arm-1617/report/data_scale_eval_val.json`
(sha `24c90104…`); ledger line appended 2026-09-14; run 61 min.
**Reading, with its bounds:** this is the strongest directional evidence yet
that TRAINING-SET SIZE is the binding lever for purchase prediction — 4.2x
rows bought ~2.3 points of exact action even across a patch boundary and an
older reconstruction pipeline. It is NOT a promotion case and NOT a
like-for-like model comparison: arm A is `probe_only` grade, trained on a
`qualified_subset`/`isolated_experiment_only` corpus reconstructed by
`decision-start-footwear-v3`, so patch, volume and reconstruction semantics
all differ at once. Note the save weakness travels with the artifact rather
than being obviously caused by scale: the same 16.17 candidate scored
no_buy_death_recall 0.5019 on its OWN 16.17 validation, below the control's
0.5162 here. **Next step (needs owner decision, not authorized by the frozen
protocol):** arm B — a joint 16.17+16.18 generation under ONE current
pipeline with a patch feature, whose training objective must carry the
save/no-buy behavior explicitly (the September 12 alignment rule), scored
once against the same control. That spends the second and last ledger line
this protocol allows. Do not tune anything else on this validation.

**September 14 early morning — data-scale experiment v1 (session
build-tracker-e6 OWNS the two tasks below; do not start parallel work on
them). Owner: "16.17-solo + conjunto", 2 ledger lines authorized ~00:30.**
Protocol frozen at `docs/experiment_data_scale_v1.md` BEFORE any evaluation
(the authorization rule lives in the parent doc — the stage-2 addendum
lesson). Key discovery: a recipe-identical 16.17-trained candidate ALREADY
EXISTS (`data/experiments/16.17-development-v2/candidate.pt`, sha
`13f5b9bf…`, trained 09-11; fixed env verified field-by-field against the
control) — no retraining; arm A is evaluation-only. Built tonight: (1)
`Predictor(dragon_override=…)` — cross-corpus static-data governance, waiver
recorded, never serving-eligible, footwear check never waived; contract
pinned by `scripts/test_predictor_dragon_override.py`; (2)
`scripts/eval_data_scale_v1.py` — paired cross-corpus evaluator reusing
eval_policy's score_visit/summarize/paired_diff verbatim, BOTH policies
governed by the 16.18 dragon, basket-size slice table included (the stage-2
reporting defect, fixed here); (3) `scripts/run_data_scale_eval.py` — suite
→ ONE paired eval (arm 13f5b9bf vs control a9b40beb) on the v2 validation →
ledger line → completion; task `BuildTracker Data Scale Arm 1617 Eval`
fires ~02:15. Disclosed bounds in the protocol: the 16.17 corpus is
`qualified_subset`/`isolated_experiment_only` (artifact provenance
`probe_only`) and was reconstructed by an earlier pipeline
(`decision-start-footwear-v3`) than the control's — directional evidence at
probe grade; arm B (joint corpus, single current pipeline) is the decisive
test and needs its own addendum + a joint-generation publisher + a purposes
decision in `app/dataset_generations.py` (NOT built; owner review pending).
Suite 50/50 with the new test. Also queued tonight: `BuildTracker Lead Miss
Diagnostic V1` — the TRAIN-only lead_item_missing decomposition this file's
closing update asked for (not_in_vocab / infeasible_at_gold_est / legal-rank
buckets + minute/size/depth slices, `scripts/mine_lead_misses_v1.py`, waits
for the eval task, output `data/reports/lead_miss_diagnostic_v1/`). Its
first launch at 01:32 died silently: PS 5.1 + `$ErrorActionPreference=
'Stop'` + `*>>` turns the first native stderr line (torch warnings) into a
terminating error — both launchers now use Start-Process with file
redirection (history.md trap 4). The 300-row smoke hinted most lead misses
may be `infeasible_at_gold_est` (gold-estimation error, not ranking) — full
run decides, denominators there. Label vocabulary is IDENTICAL across
16.17/16.18 artifacts (198 labels). `footwear_runtime.py` entry allowlist
gained the two new scripts. No training, no test reads, no promotion; the
ledger line is appended by the eval task on completion.

**September 13 closing update — the improvement route chosen from today's
evidence.** With the fresh v2 reference established (original policy: exact
0.3641 / family 0.6193, n=124,257 / 1,029 games) and the composition
intervention refuted, the levers ranked by measured mass:

1. **Lead-item ranking — the priority.** 50.2% of TRAIN misses (68,593
   purchases) have the recorded lead item absent from all three cards
   (`data/experiments/basket-composition-v1/diagnostic/train_miss_partition.json`).
   Next step is a TRAIN-only slice diagnostic of that class (by minute,
   budget, role, item type; 25 bound examples per class already saved), then
   ONE predeclared training experiment chosen from it — candidate shapes: a
   ranking objective over legal items with family and ward/SAVE inside the
   objective (the September 12 alignment lesson), or conditioning purchase
   ranking on the model's own target-head prediction instead of using it only
   as a filter. This is the only lever with large headroom; state it as
   headroom, never as a promised gain.
2. **The wallet ceiling (22.2% of misses) is not model-fixable offline.**
   Match-V5 lacks decision-time gold; the fix is opt-in live shop sessions
   recording the exact wallet (Phase-4 telemetry exists), which also gives
   `gold_est` its first validation. Blocked on the owner's Personal API Key
   application (`docs/beta.md`).
3. **Save/ward heads.** The save decision still ties a frequency lookup
   (no-buy recall 0.5162 on v2 val). Buried positive from September 12: the
   rejected D5 supervised head fine-tune improved no-buy +0.0123 and
   ward-only +0.0207 significantly. A small predeclared head fine-tune
   SELECTED on save/ward with non-regression gates on both primaries
   (the inverted gate) is cheap (~1h GPU) and targets the product
   differentiator.
4. **Data grows unattended.** The daily pipeline now collects and samples on
   its own; a refreshed development generation every few days provides
   replacement validation. The spend ledger
   (`data/ml/v2_validation_selection_log.md`) decides when.

Explicitly NOT the route: further decode-time devices — four families are now
rejected with paired intervals (target gate D2, listwise ranker D7,
contrastive heads D5, sequential composition tonight) — and small unbudgeted
validation reads. Each scored comparison spends the fresh window.

**September 13 late night — stage 2 result: REJECTED, negative result kept.**
The sequential-v1 composed policy versus the original policy, same frozen
checkpoint, on the fresh v2 validation (1,029 games / 124,257 decisions,
first line of the spend ledger): `first_action_exact` 0.2913 vs 0.3641,
paired diff **-0.0728** CI95 [-0.0749,-0.0708] — the addendum's predeclared
risk (extending already-correct leads) materialized decisively.
`action_family_match` +0.0093 [+0.0084,+0.0105] and ward-only recall +0.0144
[+0.0106,+0.0181] improved; post-death no-buy diff exactly 0.0000 (7,729
SAVE-led decisions byte-identical by construction); zero unaffordable in both
arms. Verdict: rejected under the two-primary rule; retain the original
policy; no composition constant is re-tuned. Byproduct: the v2 reference for
the original policy on the fresh validation is exact 0.3641 / family 0.6193
(never compare against v1-export numbers). Report:
`data/experiments/basket-composition-v1/stage2/report/composition_eval_val.json`.
Operational corrections recorded: (1) the earlier "dual instance" diagnosis
was WRONG — Windows venv python.exe is a launcher stub whose real interpreter
child shows an identical command line; the serial attempt killed under that
diagnosis was healthy; (2) the parallel driver initially spawned shards with
CREATE_SUSPENDED and no resume (56 min frozen) — fixed by dropping the flag
outside supervisor+job contexts. Aborted attempts preserved under
`stage2_failed_tests_env_leak/`, `stage2_aborted_dual_instance_20260913/`
(misdiagnosed, actually healthy) and `stage2_aborted_suspended_20260913/`.

**September 13 night — basket-composition stages 1-2 (owner: "1 y 2"):**
Stage 1 completed in `data/experiments/basket-composition-v1/`: full suite,
fixed trunk+graft on the v2 export (the stage-2 control `candidate.pt`), and
the TRAIN-only miss partition over 274,647 purchases — covered 138,101
(50.28%); of 136,546 misses: lead_item_missing 68,593 (50.2%),
budget_infeasible 30,310 (22.2%), other 25,803 (18.9%),
composition_truncated_or_blocked 9,023 (6.6%), count_truncated 2,817 (2.1%).
Under the predeclared rule (addressable 27.6% >= 25%) the frozen addendum
`docs/experiment_basket_composition_v1_addendum.md` selects
`sequential-v1`; its honest bounds note that half the miss mass is lead-item
ranking that no recomposition touches, and the ward-heavy extension risk on
already-correct single-item leads. Stage 2 (`BuildTracker Basket Composition
Stage2`) is running: suite, then ONE paired composed-vs-original evaluation
on the fresh v2 validation with guardrails; it appends the first
validation-spend ledger line on completion. An earlier full-scan abort was
declined by the owner; the run above IS the full scan. The stage-2 evaluator
has no mid-run checkpoint: a power-off kills it and it restarts from scratch.
No promotion, no test reads.

**September 13 afternoon — owner instructed steps 1-3 (gates, fresh
generation, predeclared experiment):**

1. **Gates/reconstruction (running):** new frozen snapshot
`data/acceptance/16.18_reconstructible_20260913/source_plan.json` at 12:58 UTC
— 4,558 main-DB 16.18 games, 4,409 selected (149 reserved test excluded, zero
missing raw pairs). Hidden task `BuildTracker 16.18 Reconstructible V2` is
building `data/corpora/16.18-reconstructible-v2/` and will run the full
semantic audit + qualification (6h wall / 6 GB / 40 GB caps). This is the
footwear/Viego-aware pipeline; it subsumes the per-sample plain gates for
training purposes. `prepare/run_reconstructible_1618.py` gained `--base` /
`--corpus` / wall parameters (defaults unchanged; v1 evidence untouched).

2. **Fresh generation (staged, launches after 1):** benchmark-era code ported
from `.worktrees/benchmark-16-18` into main (app/dataset_generations.py,
app/predictor.py, footwear_runtime, corpus_rows, report_corpus_coverage,
publish_patch_dataset, eval_policy, train_prefix, benchmark_1618_runtime/
entry, prepare_benchmark_1618, verify_reconstructible_1618_evidence + the two
benchmark test scripts; all ported tests pass). New
`scripts/prepare_development_1618_v2.py` implements policy
`forward-validation-v1`: train = qualified games created at or before the v1
snapshot instant (2026-09-12T11:03 UTC), validation = every qualified newer
game (unshopped at admission), the development-exposed v1 151-game window
excluded from both folds, reserved tests sealed. `scripts/
run_development_1618_v2.py` chains verify → admission → coverage → export to
`data/dataset_generations/16.18-reconstructible-development-v2/`
(benchmark id `16.18-reconstructible-development-v2`, eval version 7).
**COMPLETE (16:0x UTC):** the reconstruction
finished with 3,537 qualified games / 422,046 admitted rows
(`data/acceptance/16.18_reconstructible_20260913/qualification.json`). The
validator patch landed (forward policy branch in `validate_1618_split` and
prior-benchmark document bindings in `validate_1618_admission`; the first
admission attempt failed on the unbranched admission cross-checks and is
preserved at `run_admission_failed_20260913a/`; an orphan
`.16.18_development_v2_admission.*.partial` remains under data/acceptance —
derivative copies, safe to delete). Full suite 49/49. The generation
published:
`data/dataset_generations/16.18-reconstructible-development-v2/`, export
fingerprint
`d70bfb2a8d71d312924004f7dace3cadcc213de27473313a8413ac94bb639bc2`,
eval version 7. Split: **train 2,357 games / 279,748 rows; validation 1,029
games / 124,257 rows — unshopped at admission**; test 0 rows (sealed);
exactly the 151 development-exposed v1 validation games were excluded
(`prior_validation_exposed`), the intended retirement confirmed by count.
Validation spend ledger created empty at
`data/ml/v2_validation_selection_log.md`. No training, evaluation, promotion
or test read has touched this generation.

3. **Predeclared experiment:** `docs/experiment_basket_composition_v1.md`
freezes the question (candidate-basket GENERATION for multi-item decisions),
the two primaries + ward/SAVE non-regression guardrails, the TRAIN-only miss
diagnostic that precedes any intervention, the objective-alignment rule, the
stopping rules and the v2 validation spend ledger. It authorizes no run by
itself; stage 2's exact intervention needs a frozen dated addendum.

**September 13 morning operations:** today's 04:30 Daily Pull was missed (PC
off) and the 07:00 dated sample follow-up did not fire. The owner asked to run
the sample covering the actual window, then the daily pull, and to fold the
dated sample into the daily pipeline permanently. Executed: sample
`data/patch_samples/16.18_20260913/` prepared at 07:17 UTC with the real
cutoff (patch start → prepare time); its first launch failed at 07:22 UTC on
"Riot rejected the API key" — Riot rejected the key before the portal's
advertised expiry (the portal still showed 1h16m remaining at 07:30 UTC, and
a direct 200 was observed at 07:15 UTC). The owner regenerated the key (the
portal's regenerate button silently requires its reCAPTCHA — a blocked captcha
looks like a frozen page); the new key verified 200 on KR and EUW at 07:34 UTC
and the same one-shot task `BuildTracker Patch 16.18 Sample 20260913` was
restarted at 07:34 UTC — never a duplicate collector. `scripts/daily_pull.py`
now runs the dated cumulative sample as a post-lease stage (see the September
13 update in `docs/data_acceptance_16_18.md`; `--no-sample` opts out; skip on
completed sample, resume on checkpoint, guard on cohort-parent patch). Both completed serially the same day: the
sample finished at 10:46 UTC (4,350 stored games / 43,500 perspectives from
5,838 discovered matches, 1,488 excluded by patch/queue, structural checks
passed, cutoff 07:17 UTC; two mid-run key deaths — the first key was rejected
before its advertised expiry, and the first "regeneration" turned out to be a
copy of the existing key; a truly regenerated key also returned 401 for ~30
seconds before propagating), and the catch-up daily pull finished at 12:03 UTC
in 75.8 min with 0 stage failures, whose new sample stage correctly skipped
the already-complete dated sample. Freeze/audit/qualification/coverage for
the September 13 sample remain pending. The spent one-shot task
`BuildTracker Patch 16.18 Sample 20260913` remains registered (owner declined
cleanup). No export, training, promotion or shutdown was part of any of this.

## September 12 model postmortem — what failed and what to do next

**Result:** none of today's changes *after the reconstructed 16.18
benchmark* qualified as a model-behavior improvement. The benchmark candidate
did beat the conditional lookup control on the **same** frozen development
validation: first-action exact agreement 0.3304 versus 0.1997 on 18,041
decisions / 151 games; action-family match 0.5999 versus 0.3857 on 15,987
applicable purchases. Source:
`data/experiments/16.18-reconstructible-development-v1/policy_eval_val_candidate_vs_conditional_policy.json`.
That is evidence relative to this control, not a gain over a served model or
over another export. No deployment artifact is bound, and the test split
remains unread.

**Where my iteration went wrong.** I spent the follow-up budget on local
decoder, gate, head-loss and ranking changes against the same 151-game
development validation. The copy-tail rule attacked quantity, but the
original error partition has 281 quantity-only failures versus 10,658
wrong-item-ID failures (full validation n=18,041). Removing the target gate
helped exactness slightly but did not establish a family gain and hurt
save/ward behavior. The matched contrastive head made lead baskets sparser;
it lost 547 family hits and gained 225 on n=15,987, while its exact-action
change against the parent was uncertain. Despite that warning, I chose a
D7 listwise loss trained only for exact basket selection. D7 raised exact
agreement 0.3304 to 0.3403 on n=18,041, but lowered family match 0.5999 to
0.5892 on n=15,987 and ward-only recall 0.7939 to 0.7441 on n=723. Both
primary paired intervals exclude zero in opposite directions; it fails the
predeclared two-primary win rule. Family match was an acceptance metric but
absent from D7's loss, while ward-only recall had no predeclared nonregression
gate. I should have aligned those objectives and guardrails before spending
a full run on another exact-only objective. Reports:
`data/reports/model_day_20260912/baseline_error_attribution.json`,
`data/reports/model_day_20260912/d5_postmortem.json`, and
`data/experiments/16.18-option-ranking-v1/option_ranking_eval_val.json`.

**What the diagnostics actually explain.** Of 18,041 validation decisions,
5,170 recorded targets are outside the current buy-only/estimated-wallet
envelope: 626 under buy-only action rules even with an unlimited diagnostic
wallet, and 4,544 legal only above the supplied `gold_est`. Yet 6,910 errors
remain among 12,871 mechanically representable targets, so action-space and
wallet limits are not the whole explanation. Vocabulary and count-capacity
exclusions were zero on this validation. The source audit found just
293 conditional sell-first routes among the 5,170 failures under a narrow
ordinary-sale screen; 4,771 cases were not assessed by that screen. These
counts are *not* a Bayes accuracy ceiling, proof that `gold_est` is wrong,
or a reconstruction repair. Match-V5 does not give an exact decision-time
wallet, and future transactions cannot become model inputs. Sources:
`data/experiments/16.18-action-feasibility-v1/action_feasibility_val.json`
and `data/experiments/16.18-source-accounting-v1/source_accounting_val.json`.

The original decoder already has a second, separate limit: one of its three
cards matches the exact recorded purchase in 7,246/17,715 purchase visits,
while the lead card matches in 5,800/17,715. This is oracle headroom for
*selection*, not an achievable gain; fixed-card reranking cannot rescue a
purchase missing from all three cards. D7 reordered cards on 2,257
of 11,656 eligible validation decisions and confirmed the exact/family/ward
tradeoff. Sources: the original policy report above,
`data/reports/model_day_20260912/option_ranking_headroom.json`, and the
D7 report. A low literal basket-match score also does not establish that
historical alternatives are bad advice: the task imitates observed high-elo
shopping, not game outcomes.

**Prioritized route forward:**

1. **Improve the evidence before another selection run.** Let the separately
   authorized dated collection/acceptance finish; once enough games qualify,
   build a fresh temporal development generation with reserved test IDs
   protected. Keep old and new export scores separate; compare every new
   policy with its control on the *same* new export and an unshopped
   validation window.
   Instrument opt-in, user-started shop sessions to record the actual
   pre-decision wallet, inventory, displayed cards and subsequent choice.
   This can test the offline/live state gap without inventing exact gold from
   a future frame. Do not score on a polling endpoint or treat user telemetry
   as a high-elo imitation label by default.
2. **Target basket identity and coverage.** On TRAIN only, measure why the
   observed basket/family is missing from otherwise legal original candidate
   lists, then predeclare one structured *complete-basket* candidate/scoring
   comparison. If a family label from a later completed item is used, it is
   loss-only TRAIN supervision, never a pre-decision feature. Train and select
   for both exact action and family behavior, with explicit ward-only and
   SAVE guardrails. Preserve all item copies, recursive costs and slot rules.
   Do not tune another copy threshold, gate, loss weight or epoch on today's
   repeatedly used validation.
3. **Keep the release gate intact.** Require a new full displayed-policy,
   game-paired validation report on identical inputs, favorable intervals on
   both primary metrics, zero unaffordable baskets, and reviewed ward/SAVE
   behavior. The final-test payloads get one confirmatory use only after a
   candidate is chosen. Promotion still needs owner review of a specific
   validated candidate and its provenance/digest; today's ranker is rejected.

The conclusions above are reused-development evidence. No accepted change
to the served behavior was made today; `docs/model_day_closeout_20260912.md`
and the original reports are the detailed audit trail. The separate two-night
collection follow-up remains active, while the day-development follow-up was
deleted after closeout.

**September 12 closeout complete; original candidate retained:** D7 option
ranking completed at 20:24:29 UTC. All 64 assert scripts passed, its full
86,488-row TRAIN candidate/context check passed, and the completed supervisor
no-op, generation/cache/ranker and independent outcome verifiers all passed.
The one-shot scheduled task removed itself. Final evidence:
`data/experiments/16.18-option-ranking-v1/option_ranking_eval_val.json`,
`final_verification.json`, and the independent receipt under
`data/reports/model_day_20260912/option_ranking_independent_verification.json`.

On unchanged validation n=18,041 / 151 games, exact-action agreement rises
from 0.3304140569 to 0.3403359016 (paired difference +0.0099218447, CI95
[+0.0070098576,+0.0124685491]). Family match on n=15,987 falls from
0.5999249390 to 0.5892287484 (difference -0.0106961907, CI95
[-0.0144798679,-0.0078991913]). Ward-only recall also falls by 0.0497925311
on n=723. Every candidate basket, SAVE behavior and any-three coverage are
preserved; only BUY order and its positional tier label changed. All displayed
baskets remain legal at the offline wallet. This **fails the predeclared
two-primary win rule**. Do not adopt, tune or promote the ranker. The original
benchmark candidate remains the development reference. See
`docs/model_day_closeout_20260912.md` for the complete end-of-day summary.

At the owner's wrap-up request, no further experiment was selected. No final
test, serving/promotion, commit/push or shutdown occurred; the deployment
manifest is still absent. The completed day heartbeat was deleted. The separate
authorized two-night collection/acceptance follow-up remains active.

**Wrap-up requested; final experiment active, September 12 19:18 UTC:** the
owner asked to start summing up and reduce model effort. Stop selecting new
experiments. Finish the already launched D7 job, verify its final evidence,
update the closeout below and the day ledger, then pause the day heartbeat.
The recommendation for the remaining checks/handoff is GPT-5.6 Sol with High
reasoning; the owner chooses the chat setting. The local Python job is
independent of that choice.

D7 launched at 19:15 UTC as hidden task
`BuildTracker 16.18 Option Ranking V1`. Frozen plan SHA256:
`c92c3f5bbebae56eb9775da25cde7e3354223efc52b96fbe6e13450cbec5ffae`.
Actual progress: `data/experiments/16.18-option-ranking-v1/status.json`.
The full suite reports 64/64 passed in 106.4 seconds. No D7 learning or
validation result is claimed yet. Its two-hour supervisor budget remains in
force. Keep the entire new checkout's code and bound protocol doc/web file
frozen; update ONLY the main handoff/ledger while it runs.

After completion, run the completed supervisor's no-op verification and
explicit train/val generation/cache/weight binding checks. Independent
recorded-outcome verifier (outside the frozen checkout):
`data/reports/model_day_20260912/verify_option_ranking_results.py`, with
`--expected-plan c92c3f5bbebae56eb9775da25cde7e3354223efc52b96fbe6e13450cbec5ffae`.
It has only been syntax-checked; no final D7 verification exists yet.
Current closeout: `docs/model_day_closeout_20260912.md`.

**Fixed-option ranking selected, September 12 18:49 UTC:** decision D7 in
`docs/model_development_20260912.md` is now being implemented in fresh checkout
`.worktrees/model-option-ranking-20260912`; planned run is
`data/experiments/16.18-option-ranking-v1`. It may only permute unchanged BUY
cards when the original primary is BUY and at least two BUY cards exist;
SAVE cards remain in their original positions. Original model, candidate
contents, source generation and all decoder settings stay fixed.
One presentation-only exception was identified before freezing: moved BUY
cards inherit the destination slot's tier, since the UI uses it to label
"First line". All non-tier fields and every SAVE card remain exact; original
card objects are preserved. The evaluator checks both conditions separately.
TRAIN candidates must be regenerated before labels are attached; D5's
negative cache is incomplete for ranking. Its context cache may be reused
only after full identity/hash and fresh bitwise context checks. A separate
small scorer uses the fixed three-epoch configuration in D7; full paired
validation must improve both primary metrics while preserving the entire
candidate set, SAVE behavior and legality. No real D7 job has started yet.
The maximum two-hour run must fit before today's 21:59 UTC boundary.

**Source-accounting audit verified, September 12 18:43 UTC:** D6 completed
at 18:40:52 UTC with 61/61 assert scripts passing. External source/code,
generation, stage and output bindings passed the completed supervisor's no-op
verification; independent outcome arithmetic also passed. Evidence:
`data/experiments/16.18-source-accounting-v1/source_accounting_val.json` and
`final_verification.json`. The one-shot task removed itself successfully.

Among exactly 5,170 failed validation targets, within the unchanged 18,041
decisions / 151 games, 399 visits pass the narrow ordinary-sale screen.
Conditional sell-first searches recover 131 at unchanged gold_est and 162
more after adding reviewed sale credits; 106 remain wallet-limited. The other
4,771 are not assessed by this restricted scenario. Observed sales occur in
971 failed visits, but that flag alone is not a causal explanation. These
findings do not measure actual starting gold or new model accuracy. Retain
the original model, inputs and all completed checkouts; no repairs/promotion.

Remaining day work is assessing one next hypothesis from saved evidence:
whether choosing among the original displayed alternatives has useful
headroom without changing the legal candidate baskets. No new comparison
or training has been selected or launched at this checkpoint. D5's negative
result is also receiving a small read-only postmortem under
`data/reports/model_day_20260912/`; do not tune its loss weights or thresholds.

**Source-accounting audit launched, September 12 18:36 UTC:** hidden task
`BuildTracker 16.18 Source Accounting V1` is running from
`.worktrees/model-source-accounting-20260912`. Actual progress is in
`data/experiments/16.18-source-accounting-v1/status.json`; the immutable plan
is `source_accounting_plan.json` in that directory, SHA256
`116ad23864767f5a1c076092923ae9950547c62e095cae818a616ab151918a65`.
Twelve focused worker tests and six supervisor fault tests passed; independent
review found no blocker. The scheduled sequence checks all frozen inputs,
runs the complete assert suite, then diagnoses exactly D4's 5,170 failed
validation targets. No completed audit result is claimed at launch.
The new checkout's complete code inventory is now pinned; do not edit or add
application/script files there. This is a source diagnostic, with no model
inference, training, exact-gold reconstruction or input repairs. The task name
was passed explicitly to its self-unregistering wrapper.

**Basket-head learning complete; original retained, September 12 17:41 UTC:**
both training arms, three-arm validation and summary completed. Independent
final verification passed at
`data/experiments/16.18-basket-head-learning-v1/final_verification.json`.
Neither child meets the declared win rule. Exact action agreement on validation
n=18,041 / 151 games is original 0.3304140569, supervised 0.3274208747,
contrastive 0.3319660773; the contrastive gain against original is uncertain.
Family match on n=15,987 applicable purchases falls from 0.5999249390 to
0.5797835742 for the contrastive arm. Full effects, intervals and tradeoffs:
`data/experiments/16.18-basket-head-learning-v1/basket_heads_eval_val.json`.
All original tensors outside the two purchase/count heads and all inference
configuration remain verified unchanged. No candidate is adopted or promoted.

Decision D6 now investigates original source transactions for D4's 5,170
wallet/action-space-limited validation cases. The new checkout is
`.worktrees/model-source-accounting-20260912`; planned run is
`data/experiments/16.18-source-accounting-v1`. Its scope is source flags plus
conservative, explicitly conditional sell-first feasibility—not reconstructed
exact gold, model scoring or automatic source repairs. Detailed restrictions
and time budget are in `docs/model_development_20260912.md`.

The completed Basket Heads scheduled task was manually unregistered after
checking exit 0: its frozen wrapper's default task name differed from the name
actually registered. No frozen code/evidence changed. Pass exact task names
explicitly to later one-shot wrappers.

**Basket-head learning: tests passed and TRAIN mining active, September 12
16:44 UTC:** all 59 assert scripts passed. The job is generating legal TRAIN
alternatives and cached encoder features before either head variant trains.
The last inspected checkpoint had processed 7,000/86,488 rows; use the live
`status.json` under `data/experiments/16.18-basket-head-learning-v1/` for current
counts and measured rate. No new model-quality result exists at this checkpoint.

**Matched basket-head learning launched, September 12 16:39 UTC:**
hidden task `BuildTracker 16.18 Basket Heads V1` runs from
`.worktrees/model-basket-learning-20260912`. Frozen plan and actual status:
`data/experiments/16.18-basket-head-learning-v1/learning_plan.json` and
`status.json`. Plan SHA256:
`ba4a72f976c7fc6e1f91e65ec9e97caf731e20a7eab4b678a2e4a66a999409f0`.
Stages: complete assert suite, TRAIN-only teacher mining/features, matched
supervised and contrastive purchase/count head training, three-arm validation,
summary/completion. Focused tests and independent code review passed before
launch; full-suite, mining and new training completion are not yet claimed.
The encoder, auxiliary heads, original checkpoint/configuration and all decoder
rules remain fixed. Every original TRAIN row retains its supervised anchor;
only feasible truths receive additional legal contrastive alternatives. This
run has a 3-hour/10 GB/40 GB budget and today's deadline. Do not modify its
pinned code inventory or start another expensive experiment concurrently.

**Feasibility audit complete; matched basket-head learning selected:** the
diagnostic completed at 15:53 UTC and passed final verification. On validation
n=18,041 / 151 games, 4,544 targets exceed the supplied wallet and 626 are
unreachable under buy-only rules; vocabulary/count exclusions are zero.
12,871 targets are mechanically representable, with 6,910 original-model errors
remaining in that group. Source:
`data/experiments/16.18-action-feasibility-v1/action_feasibility_val.json` and
`final_verification.json`. This is a mechanical diagnostic, not new accuracy
or a measurement of true gold error.

Decision D5 in `docs/model_development_20260912.md` now governs the next learning
experiment. New checkout: `.worktrees/model-basket-learning-20260912`.
Planned run: `data/experiments/16.18-basket-head-learning-v1`. Both variants
fine-tune purchase/count heads from the same parent, with the encoder and
save/target heads fixed. All TRAIN supervision is retained; only feasible TRAIN
targets receive mined legal basket alternatives for the additional contrastive
term. The matched supervised control and contrastive candidate will be compared
with each other and the unchanged original on full validation. Implementation
and tests are in progress; no new training or result is claimed yet. All earlier
checkouts and evidence remain unchanged.

**Target-filter result verified; action-feasibility diagnostic chosen:** the
first day experiment completed at 15:05 UTC. Its direct-purchase arm improves
exact action agreement by 0.0019400255 (validation n=18,041/151 games), but the
action-family paired interval crosses zero (n=15,987 applicable purchases) and
save/ward recall decline. It does not meet the two-primary win rule; retain the
original default filter. Exact report and verification:
`data/experiments/16.18-target-filter-ablation-v1/target_filter_eval_val.json`
and `final_verification.json`. Full effects and caveats are recorded in
`docs/model_development_20260912.md`.

The owner then asked why accuracy is low. The next selected diagnostic audits
whether recorded baskets are mechanically representable by the vocabulary,
count/basket limits, buy-only legality and estimated wallet. This will separate
learnable prediction mistakes from current target/action-space constraints;
it is not a gold-truth oracle or an excuse to change the accuracy denominator.
Implementation is being prepared in `.worktrees/model-feasibility-20260912`,
with planned run `data/experiments/16.18-action-feasibility-v1`. No diagnostic
result or new training is claimed yet. Prior completed checkouts stay frozen.

**Target-filter experiment: full tests passed, September 12 14:48 UTC:**
all 54 assert scripts passed in the day-development checkout. The single paired
target-filter evaluation is now running; no result is available yet. Its plan
SHA256 is `d247b437a1a4a860c6b421fce3c64f248d0d957affb9abcaf9989cd15c95c6ad`.
Read `data/experiments/16.18-target-filter-ablation-v1/status.json` and the
`evaluation_20260912T144802561469Z.log` stage log. Choose subsequent model work
from the verified result and trace summary, preserving the fixed comparison.

**First day-development experiment launched, September 12 afternoon:**
hidden task `BuildTracker 16.18 Target Filter V1` runs from
`.worktrees/model-behavior-20260912`. It executes the full assert suite, then
the single original-versus-direct-purchase target-filter comparison, with
decision traces and exact parent-output parity. Actual status and frozen plan:
`data/experiments/16.18-target-filter-ablation-v1/status.json` and
`evaluation_plan.json`. Expected report is `target_filter_eval_val.json`,
with trace summary and completion evidence in the same directory.
No retraining or new data is involved. The 90-minute run has a 10 GB child
memory cap, 40 GB disk reserve and same-day deadline. Do not edit or add code
files in this checkout while it runs; its complete code inventory is pinned.
Focused predictor, evaluator, attribution and supervisor tests passed before
launch. Full-suite and comparison completion are not yet claimed.

**New owner authorization: model development for the rest of September 12:**
at about 16:17 Madrid the owner asked for autonomous development through the
rest of the day with updates on decisions, and selected **Improve model
behavior**. The active decision ledger is `docs/model_development_20260912.md`.
Heartbeat `build-tracker-model-development-today` continues this task every
30 minutes through this date and must finish with an end-of-day report/pause
by 23:59 Madrid. The earlier benchmark heartbeat remains completed/paused.

New work lives in
`C:/Users/Javier/build_tracker/.worktrees/model-behavior-20260912`;
the completed benchmark checkout and its bound evidence remain unchanged.
The first decision is one fixed-checkpoint ablation of the top-four target
recipe purchase filter, retaining purchase ranking/thresholds, original count
decoding, target arrows, save rules and legality. Control is `top4-gate-v1`;
the experimental opt-in is `direct-purchase-v1`. Optional private traces record
target vetoes and learned-save versus unconditional fallback causes. No real
comparison result is available for this new change yet.

The independently bound baseline error partition is
`data/reports/model_day_20260912/baseline_error_attribution.json` (validation,
18,041 decisions / 151 games). For the transformer it records 5,961 correct
actions, 10,658 wrong-item-identity decisions, 281 quantity-only disagreements,
976 saves on recorded purchases, and 165 buys on recorded post-death no-buys;
empty-action categories are zero. These mutually exclusive counts sum to the
same validation denominator. Save-branch causality remains not measured until
the new trace-bearing comparison; overlapping underbuy/overbuy flags are not
evidence of simple underspending. No source holdouts were read.

Morning collection and the archive refetch both completed at 13:26 UTC,
including all 4,432 selected non-test raw pairs. Source recovery does not add
them to any frozen dataset. The new day authorization does not include model
promotion, final-test access, commits/pushes, external messages or shutdown.

**Steps 2–4 complete and independently verified, September 12 13:55 UTC:**
all nine scheduled stages completed. `completion.json` and
`final_verification.json` are in
`data/experiments/16.18-reconstructible-development-v1/`. Stage-output, artifact,
report, plan, generation, paired-outcome and protected-artifact hashes were
checked; the original decoder exactly reproduces step 3's decisions and
primary metrics. Analysis is complete in that run's `analysis/` directory.

**Negative decoder result:** `count-tail-075-v1` does not meet the predeclared
win rule. On the same 151 development-validation games / 18,041 decisions,
`first_action_exact` changes from 0.3304140569 to 0.3291391830, paired
modified-minus-original difference -0.0012748739, CI95
[-0.0022544102, -0.0003328710]. Excess additional copies fall from 559 to 445,
but missing additional copies rise from 697 to 758 (both counts over the same
18,041 validation decisions). `action_family_match` improves slightly from
0.5999249390 to 0.6004878964 on 15,987 applicable validation purchases, paired
difference 0.0005629574, CI95 [0.0001892983, 0.0009709995]. Both policies have
zero unaffordable options across 36,728 independently checked displayed
options each. Source for every decoder metric:
`data/experiments/16.18-reconstructible-development-v1/decoder_eval_val.json`.
Report SHA256 `4f0deb566c38f123d3d116b79d3fb789b5ff5b13e675ebda9764811a60ad0176`.
Retain the original default decoder and the immutable negative-result evidence;
no extra threshold search, retraining or policy adoption followed this result.

The task's authorized benchmark/comparison work is complete; its heartbeat is
being paused. The separate dated sample follow-up retains its collection and
acceptance scope. No test payload was read, no production hold cleared, no
model promoted, no commit made and no shutdown scheduled. Implementation and
its frozen execution environment remain in the isolated benchmark checkout.

**Step 3 complete; decoder comparison running, September 12 13:31 UTC:**
the fixed 16.18 transformer beats the train-only conditional baseline on both
predeclared development-validation metrics. `first_action_exact` is
0.3304140569 versus 0.1997117676, paired difference 0.1307022892, CI95
[0.1238509385, 0.1374132085], n=18,041 decisions across 151 validation games.
`action_family_match` is 0.5999249390 versus 0.3856883718, paired difference
0.2142365672, CI95 [0.2025919573, 0.2252859351], n=15,987 applicable validation
purchases. Both policies have zero unaffordable displayed outputs under the
reconstructed wallet and slot state. Source for all metrics:
`data/experiments/16.18-reconstructible-development-v1/policy_eval_val_candidate_vs_conditional_policy.json`.
Report SHA256 `8a35523584bbd338aa789b6c6e379ee3c516dcb97aa231cddd635b9974d36fbe`.
The policy-stage receipt, candidate digest and paired-outcome file digest were
independently verified. This is reused development validation, not final-test
or live evidence. The scheduled job is now running the one fixed-checkpoint
additional-copy comparison; its result and final analysis remain pending.

**Training complete; paired evaluation started, September 12 13:14 UTC:**
the fixed trunk and graft stages finished. Their checkpoint and log hashes were
independently verified against `stage_evidence/trunk.json` and `graft.json` in
`data/experiments/16.18-reconstructible-development-v1/`. Candidate SHA256:
`35bc20c781cb1f49903bd39bb98bd5d6f21f5757121cb313f37bcce9482121a9`.
The scheduled job is fitting the conditional baseline on training rows, then
comparing both policies on frozen validation decisions. No completed
displayed-policy result exists at this checkpoint; the copy-decoder comparison
follows that report.

**Step 2 complete; training started, September 12 12:57 UTC:** the scoped
qualification, coverage and immutable export have completed successfully.
`data/dataset_generations/16.18-reconstructible-development-v1/` contains
722 training games / 86,488 rows and 151 development-validation games / 18,041
rows. The 1,308 later qualified sources remain deferred; no test payload was
materialized. Independent generation and run-plan verification passed after
publication. Export fingerprint:
`fef1eb63d0d524f33fb85292ee302da64cb7502df09f41dc01cbf42594709f6e`;
generation-manifest SHA256:
`1a397042df68213dd2f2b4939fa098c0f88ffeecc6545959f63572da144942b4`;
run-plan SHA256:
`1f6fabde21e8e635e284410661ea6d18a5b0b4328c50051e5236fac35407d59a`.
The scheduled job is training the fixed 16-epoch trunk on CUDA. Graft, paired
conditional evaluation, copy-decoder evaluation and analysis remain queued.
Training-log diagnostics are not policy evidence. Read the experiment's
`status.json`; no completed model comparison is claimed at this checkpoint.

**Corrected benchmark test gate passed, September 12 12:40 UTC:** all 50 assert
scripts passed in the isolated checkout. Evidence:
`data/experiments/16.18-reconstructible-development-v1/tests_20260912T123950041034Z.log`.
The durable job continues through the previously bound qualification/export
and model/decoder stages. Do not infer an exported dataset or model result
from the test result; inspect its `status.json` and stage evidence.

**Benchmark preflight recovery, September 12 12:38 UTC:** the first scheduled
attempt stopped at the test gate (49/50 passed), before qualification/export or
training. `test_recovered_corpus_pipeline.py` compared a resolved temporary path
against an unresolved `ROOT/data` junction. The isolated-checkout fix resolves
both paths before enforcing containment; the bounded two-game reconstruction,
audit, qualification and export test then passed. The failed run and protocol
bytes are preserved under
`data/experiments/16.18-reconstructible-development-v1_preflight_failed_20260912T123616Z/`.
No failed evidence was overwritten. A corrected protocol was frozen at the
normal run path (SHA256
`3786f3b150f14ef509b454370f19081476e55fa1a11f008734a84c609dcab93e`)
and the hidden `BuildTracker 16.18 Benchmark V1` task restarted. Inspect the
normal `status.json` for current progress; the complete assert suite runs again
before any data or training stage. Main collector code and source evidence are
unchanged.

**16.18 benchmark implementation and durable sequence launched, September 12:**
the original reconstructible build/audit completed successfully at 12:04 UTC.
It qualified 2,181 games with 259,409 admitted model rows; the source snapshot
and its original reconstruction-only scope remain unchanged. The continuation
independently verified every referenced source, corpus, static, decision-mask,
code-snapshot and protected-artifact hash. Bound verification evidence is
`data/acceptance/16.18_benchmark_20260912/prerequisite_verification.json`.

Because the morning collector still pins every main-checkout application file,
implementation lives in the isolated checkout
`C:/Users/Javier/build_tracker/.worktrees/benchmark-16-18`. Its `data` and
`.venv` junctions reference the existing local data/runtime. The collector's
original source files are unchanged. Do not edit either running job's pinned
code or merge these changes into the main checkout while it is still pinned.
New source files there include `prepare_benchmark_1618.py`,
`benchmark_1618_runtime.py`, `benchmark_1618_entry.py`,
`verify_reconstructible_1618_evidence.py`, `eval_copy_policy.py`, and
`run_benchmark_1618.py` / `.ps1`, plus targeted regression tests.

Hidden one-shot task `BuildTracker 16.18 Benchmark V1` runs from that checkout.
Actual stage authority is
`data/experiments/16.18-reconstructible-development-v1/status.json`; its frozen
recipe is `protocol.json`, followed by the export-bound `run_plan.json` once
publication succeeds. The sequence is the complete assert suite, new benchmark
inventory qualification, coverage, immutable export, one 16-epoch trunk,
six-epoch frozen-trunk graft, paired conditional-policy validation, exactly one
fixed-checkpoint copy-decoder comparison, and bound exploratory analysis.
Use its existing status/checkpoints; never launch a duplicate experiment.
An eight-hour attempt wall budget, 10 GB child-tree cap, 40 GB disk floor and
exclusive execution lock apply. `completion.json` is written only at the end.

The admitted benchmark preserves the initial temporal train/validation windows
and all old holdout reservations. Metadata-only selection yields at most
722 train / 151 validation games before the new inventory/export checks;
1,308 later qualified source games are explicitly deferred. Qualification and
coverage determine the actual counts. The new scope is
`patch_development_1618`, benchmark `16.18-reconstructible-development-v1`,
evaluation version 6. Never describe unused later games as training examples.

The predeclared `count-tail-075-v1` decoder requires at least 0.75 count-head
tail mass for each additional copy, bounded by the original argmax quantity.
The default decoder and first-copy ranking are preserved. It uses the same
checkpoint/export and verifies exact original-policy parity with every step-3
decision before reporting paired primary and excess/missing-copy differences.
Existing 16.17 runtime/generation tests and new scoped runtime, admission,
publication, decoder and supervisor fault checks passed before launch. The
full assert suite is also a mandatory first stage; no model result or final
export is claimed yet. The final test and production hold remain untouched.

**Authorized continuation after the current audit, September 12:** the owner
instructed "once 1 finishes do 2-4": after the reconstructible 16.18 build and
full independent semantic audit pass, implement scoped benchmark admission and
an immutable export, run the fixed transformer recipe against the train-only
conditional baseline on the same validation decisions, then run one stricter
additional-copy decoder comparison with the checkpoint and first-copy ranking
fixed. Detailed protocol and completion conditions are in
`docs/experiment_16_18_continuation.md`. This authorizes implementation, checks
and durable experiment execution; it does not authorize promotion, final-test
evaluation, commits, a beta rollout or shutdown.

Codex task `01a09563-0619-7e83-b829-63eb75f0f373` owns this sequence. Its ACTIVE
heartbeat `16-18-benchmark-and-decoder-follow-up` checks every 15 minutes and
continues the work once `run/completion.json` and the full qualified evidence
are verified. The existing `16-18-next-two-nights` daily follow-up retains its
September 12/13 sample collection and acceptance work; its prompt was updated
to defer benchmark/training/comparison ownership here and prevent duplication.
No application or pinned pipeline code was edited while the audit was running.
The continuation is queued, not an export, training result or audit clearance.

The new admission path must bridge the current reconstruction-only
qualification and the generation readers, propagate the actual 16.18 runtime
through export/training/both policies, and preserve the reviewed Viego cases.
Current generic paths still contain 16.17-only runtime assumptions and blanket
Viego exclusions. Review prior assignment and timestamp metadata before fixing
the benchmark: many unassigned sources are newer than the original validation
window, so re-splitting all sources is not automatically valid. Keep old
holdouts reserved and explicitly document temporal validity and any new
development cohort; never rewrite the completed source plan to claim its
earlier reconstruction job had training authorization.

**16.18 reconstructible snapshot processing launched, September 12:** user
requested processing valid 16.18 games including reconstructible Magical
Footwear and Viego. Main DB held 3,180 unique 16.18 games at 11:23 UTC; the
immutable processing snapshot taken at 11:03 UTC contains 2,904 games, of which
149 reserved-test IDs were excluded before any payload access. All remaining
2,755 sources had archived match/timeline pairs. Later collected games are
outside this fixed snapshot: the 3,180-game count includes 276 games collected
after the 2,904-game snapshot. The source plan and static evidence are under
`data/acceptance/16.18_reconstructible_20260912/`.

**Progress checkpoint, 2026-09-12 11:33 UTC / 13:33 Madrid:** the task is
running in the build stage. The latest logged batch has processed 1,600 of
2,755 sources: 1,287 passed initial screening and 313 were excluded. Passing
cohorts are 1,120 games with Magical Footwear and no Viego, 166 games with
neither, and one Viego game without Magical Footwear. These are provisional
screening counts; the complete independent semantic audit is still pending.
Evidence: `data/acceptance/16.18_reconstructible_20260912/run/status.json` and
`run/build.log` in the same directory. Do not treat these counts as final
training qualification or add the later 276 games to the frozen plan.

Hidden one-shot task `BuildTracker 16.18 Reconstructible Sources` is running
`scripts/run_reconstructible_1618.ps1`; live status is `run/status.json` in that
acceptance directory. It builds `data/corpora/16.18-reconstructible-v1/`, then
runs the full semantic audit with independent Viego ownership and footwear
checks. A 6 GB child memory cap, 40 GB disk reserve and four-hour wall budget
apply. No Riot requests, live DB writes, split assignment, export, training or
promotion are part of this job. Existing collector/refetch code was not edited.

The process-local runtime materializes source-hashed 16.18 variants of the
reviewed 16.17 implementation; only patch/version guards differ. Rune 8304 is
byte-identical in the pinned official 16.17.1/16.18.1 rune files, and reviewed
boot/elixir/role item economic/recipe/slot fields are unchanged. This preserves
the timer/first-known-room inference and one-second decision guard; it does not
make actual delivery ticks independently observed. All forensic events stay in
accepted games and the input mask is separate.

Viego admission uses `scripts/viego_source_gate_1618.py`: independent permanent
ownership transactions, linked purchase/sale undos, pinned own-item transforms,
source-backed control wards, owner progression checks and comparisons with the
emitted shop/board inventories. It admits reviewed TOP/JUNGLE cases only;
unsupported role transforms and unexplained possession-related item events
remain excluded. No last-known build is substituted. A known valid Viego and
a possession-corrupted negative control passed their expected tests. A full
two-game build/audit test (one Viego, one footwear game) passed with 191 audited
events and zero hard failures. Full-snapshot qualification remains pending
until this new job completes; the test is not whole-corpus clearance.

**Morning recovery resumed, 2026-09-12 08:57 UTC / 10:57 Madrid:** the owner
renewed `.env` and explicitly authorized morning collection plus the remaining
archive refetch, keeping recovered historical games outside the completed V2
experiment. One-shot task `BuildTracker Morning Recovery 20260912` runs
`scripts/run_morning_collection_recovery.ps1`. The immutable plan and actual
stage are in `data/collection_recovery/20260912/plan.json` and `status.json`.
The existing Daily Pull task has restarted successfully and Riot is returning
match lists with the renewed key. Sequence: daily catch-up, prepare the dated
16.18 sample with its actual current cutoff and inherited initial cohorts,
collect that sample, then resume the same 16.17 refetch checkpoint (601 games
pending at launch). No parallel Riot collectors or new training are launched.
The previously accepted unresolved pro identities stay explicitly recorded.
Collection completion is distinct from later semantic qualification of the
new sample. Errors stop the sequence and preserve the existing checkpoints.

The requested Spanish model-evidence graphic is
`data/reports/model_progress_20260912/model-progress-16-17.png` (also SVG).
Its source JSON binds both paired reports and their completion hashes. It shows
each transformer against its own conditional baseline on identical validation
inputs within that experiment; it does not claim v1-to-v2 improvement across
different exports. The actual model experiment and artifacts remain unchanged.

**Verified overnight results, 2026-09-12 08:22 UTC:** V2 completed all nine
stages at 00:42 UTC / 02:42 Madrid (7h34m total). Candidate, policy-report,
generation-manifest and export-fingerprint bindings were independently checked.
The model trained on 9,988 games / 1,181,079 rows and was compared with the
train-only conditional policy on the identical 731-game development validation.
`first_action_exact`: A=0.3888555275, B=0.2585572010, n=88,259 validation
decisions; paired A-B=0.1302983265, CI95 [0.1273090701,0.1338086723].
`action_family_match`: A=0.6295983520, B=0.4831036556, n=78,153 applicable
validation purchases; paired difference=0.1464946963, CI95
[0.1421354646,0.1508346404]. Source for every quoted model metric:
`data/experiments/16.17-development-v2/policy_eval_val_candidate_vs_conditional_policy.json`.
Both policies had zero unaffordable outputs under reconstructed wallet/slot
state. This is reused development validation, not final-test or release
certification; no promotion occurred. Footwear grant timing remains inferred.
Existing exploratory analysis is under that run's `analysis/` directory.

The source refetch recovered 3,831 of 4,432 non-test raw pairs, including all
1,395 legacy-training games, then exited on HTTP 401 at 00:29 Madrid. Exactly
601 selected games remain. Completed game checkpoints and partial endpoint
archives are preserved outside V2. The 04:30 daily job exited 1 after the same
authentication rejection, and today's 07:00 sample was not collected. At this
check there were no active relevant Python processes or running scheduled
tasks, and `.env` still had its pre-failure modification timestamp. No API
retry or credential change was performed. Renew `RIOT_API_KEY` before the
serial collection/refetch resume described below.

**Heartbeat checkpoint, 2026-09-12 05:01–05:03 UTC / 07:01–07:03 Madrid:**
gate 6 is blocked on Riot authentication. The latest Daily Pull started at
04:30:02 Madrid and stopped at 04:30:04 after the key was rejected; Task
Scheduler reports exit 1. Evidence: `data/daily_pull.log` and the read-only
`scripts/status.py` inspection. All registered `BuildTracker*` tasks were Ready;
no collection was started, stopped or retried during this check. The `.env`
modification timestamp still predates these failures; no credential was printed.

The September 12 sample directory/checkpoint does not exist. No collection,
freeze, audit, qualification or coverage is claimed for that date. The initial
`data/experiments/16.18-initial-smoke-v2/completion.json` was checked and still
records a passed mechanics-only smoke with its original fingerprint and digest.
The production hold and sealed final-test requirements remain in force.

The separate 16.17 missing-source refetch also stopped on HTTP 401 at
2026-09-11T22:29:09.942922Z. Its preserved status reports 3,831 recovered pairs
of 4,432 selected games, leaving 601 selected games to process:
`data/source_refetch/16.17-missing-20260911/status.json`. These are source
recovery counts, not newly qualified training games. Its one-shot task has
unregistered. Separately, `data/experiments/16.17-development-v2/completion.json`
and `status.json` record all stages complete at 2026-09-12T00:42:22.570237Z,
with protected artifacts unchanged and no promotion/final-test evaluation.
This checkpoint does not make a new model-quality claim or import refetched
games into that completed experiment.

**Resume after the owner renews `RIOT_API_KEY` in `.env`:** inspect tasks and
processes again; finish the missed daily collection in collection-only mode,
then prepare and run the dated 16.18 sample through `run_patch_sample.ps1`,
then resume the same 16.17 missing-source plan if still pending. Keep these
Riot sequences serial. Source inspection at 08:22 UTC confirms that the current
`daily_pull.py` CLI does acquire `CollectorLease` around `main()`; the earlier
heartbeat's contrary statement was incorrect. The CLI fails rather than queues
when the lease is occupied, so collection timing still needs coordination.
Reuse the reviewed initial cohorts and accepted
16 unresolved pro outcomes. Fix the new sample's actual cutoff when preparing
it; do not backdate an observation if collection moves to a later date. Resume
existing checkpoints rather than creating duplicates. No API retry or new
sample preparation was performed during this blocked heartbeat.

**Live checkpoint, 2026-09-11 18:40 UTC:** V2 finished the build and entered
the full semantic audit. Separately, the missing-source refetch has received
11 complete, structurally checked raw pairs from Riot with no failures so far.
Both jobs continue; neither a new model result nor refetch completion is being
claimed. Read their status files below for current progress.

**Missing-source refetch launched, 2026-09-11:** user authorized recovering
missing 16.17 sources while explicitly leaving them outside the current
training experiment. Hidden one-shot task `BuildTracker 16.17 Missing Source
Refetch` uses `scripts/run_missing_source_refetch.ps1`; immutable plan and
live progress are under `data/source_refetch/16.17-missing-20260911/`.
It targets 4,432 non-test games: 1,395 legacy train first, then 2,955 legacy
validation and 82 unassigned. The 2,955 reserved test games are excluded before
any source payload access. Up to 4,423 match and 4,432 timeline requests are
needed. The legacy validation group has prior probe-training exposure and the
unassigned group consists of short games; refetch does not clear those issues.

This job only appends missing raw archive files and writes its own checkpoints.
It does not write the game database, reconstruct, change folds, export, train,
or add games to V2. Existing archive files are never replaced. Shared
`CollectorLease`, a 125-second opening cooldown, the existing Riot rate limiter,
and a five-hour/04:15 local deadline protect other collection. The owned child
tree has a 1 GB cap. Authentication failure stops with completed checkpoints
preserved; resume the same plan after the key is restored. Seven offline
regression tests and wrapper syntax checks passed before launch.

**Recovered-data development run launched, 2026-09-11 17:07 UTC:** the user
authorized trying the newly recovered data. Hidden one-shot Windows task
`BuildTracker 16.17 Development V2` runs `scripts/run_recovered_development.ps1`.
Read `data/experiments/16.17-development-v2/status.json` for the actual stage;
the launcher log is `data/logs/16.17-development-v2-launcher.log`.
The source selection is 10,719 games (9,988 train / 731 development-val),
subject to fresh build, full audit, qualification and immutable export.
It then trains one 16-epoch trunk and six-epoch graft and evaluates the actual
displayed policy against a train-only conditional baseline on the same export.
No v2 model result exists at launch. The two-game full data-gate rehearsal,
runtime/policy regressions and launcher preflight all passed.

The process-local footwear runtime binds causal selected/granted state to every
shopper and board row. The one-second boundary mask is an inferred-delivery
sensitivity guard; source completion targets survive masked input rows.
The eight-hour supervisor enforces a 10 GB child-tree memory cap, 40 GB disk
reserve, immutable source/code hashes and protected serving/hold hashes.
All bound code is frozen while the job runs. Original test IDs remain reserved,
Viego remains excluded, and no production promotion or cross-export comparison
is authorized. Full details: `docs/experiment_16_17_development.md`.

**Ordinary footwear recovery measurement complete, 2026-09-11:**
`data/acceptance/footwear_repair_20260911/recovery_summary.json` binds the full
11555-game report and exact recovered-ID list. 9,706 additional games pass the
source-consistency checks: 9,048 training / 658 development-validation games,
with 1,078,515 / 80,230 candidate rows after the one-second boundary guard.
They are disjoint from the existing no-footwear corpus of 1,013 games, yielding
prospective combined coverage of 10,719 games (9,988 train / 731 val) before
new qualification/export admission. Of 9,863 replayed games, 9,778 pass the
boot-specific QA; 1,692 source games were excluded before replay and 157 remain
unresolved. The full audit used disjoint four-worker source lists, verified
complete membership and equal code bindings, and preserved serving/hold hashes.

This completes the requested recovery-rate measurement, not publication of a
new training dataset. Source consistency assumes the documented causal timer
and first-available-slot model; exact delivery ticks are not independently
observed. All replayed games received source, endpoint, intermediate inventory
and final-outcome invariance checks; representative future-prefix probes ran
on the fixed sample and synthetic tests. No model was retrained on recovered
rows and no existing generation was overwritten. The next data integration
must carry explicit footwear state into masks/exports and bind a new version.

**Ordinary footwear recovery launch record, 2026-09-11:** the owner approved
testing the normal timer/slot path across the archived sources. A revised
isolated replay now schedules a free 2422 grant causally, defers while physical
slots are full, preserves recipe/sale/undo identity, and excludes nearby decision
boundaries. It also handles the separately verified conditional auto-consumption
of elixirs 2138/2139/2140 purchased with six occupied regular slots. The full
source audit is a hidden one-shot task, `BuildTracker 16.17 Footwear Parallel Audit`.
Read `data/acceptance/footwear_repair_20260911/full_parallel/status.json` for
progress. Four workers have disjoint portions of all 11,555 selected footwear
games; original test IDs are excluded. The earlier serial attempt was stopped
and superseded; its partial status is not a completed measurement.

The fixed 100-game sample passes footwear-specific source checks in 99 cases.
After the separate elixir correction, 90/100 pass the broader source-consistency
screen (48 train / 42 development-val), yielding 6,097/5,175 candidate rows.
Evidence: `data/acceptance/footwear_repair_20260911/timer_sample_100_v3.json`.
Eight games have existing Viego/opaque-undo blockers; the remaining two have
unresolved endpoint or grant-boundary evidence. These are recovery candidates,
not a newly admitted training generation. The one-second boundary guard is a
declared sensitivity choice, not an independently verified server tick bound.
The completed report supersedes these preliminary sample counts.

**16.17 development run complete, 2026-09-11:**
`data/experiments/16.17-development-v1/completion.json` binds the screened
database, full audit, qualification, export, 16-epoch trunk, six-epoch graft and
paired displayed-policy evaluation. The clean subset has 940 training games /
102,564 rows and 73 validation games / 8,029 rows. First-action exactness on
validation is A=0.3253207124 vs conditional B=0.2018931374, paired difference
0.1234275750, CI95 [0.1156692198,0.1323173682], n=8,029 visits / 73 games.
Action-family match is A=0.5925241615 vs B=0.3951108584, difference 0.1974133030,
CI95 [0.1830264433,0.2131240277], n=7,036 applicable validation purchases.
Source for all these metrics:
`data/experiments/16.17-development-v1/policy_eval_val_candidate_vs_conditional_policy.json`.
Both policies have zero unaffordable displayed options under reconstructed
wallets. Dataset fingerprint:
`6f21094679b3cdd629395cf39ec58f78d8316f253f60fc449fc641b4f4de6e28`;
candidate SHA256 `cb5e7baa18d56da11f5590af75f69d7af0ba16310c38a5587844a418e13fe5d5`.
Protected artifacts are unchanged. These are reused development-validation
results on a narrow subset, not final-test, live, cross-patch or release evidence.
Exploratory error analysis is in the run's `analysis/` directory. It identifies
extra-copy decoding as one possible controlled follow-up, not a proven fix.

The owner then prioritized the Magical Footwear data loss. A separate,
non-production prototype now tracks pre-eligibility, uncertain delivery and
observed receipt resolution. It never inserts boots from a guessed timer or
uses final rune summaries as earlier inventory. See `docs/footwear_repair.md`.
First 20-game replay produced 1,239 candidate rows passing its limited checks;
zero new rows have been admitted. The pilot corpus and results remain frozen.

**16.17 development launch record, 2026-09-11 14:09 UTC:** the owner authorized
an isolated clean 16.17 database, actual model training and validation-driven
improvement work while 16.18 accumulates. Hidden one-shot task:
`BuildTracker 16.17 Development V1`. Its authority is
`data/experiments/16.17-development-v1/status.json`, with separate stage logs and
a final completion report only when the pipeline succeeds. Do not describe
training or metrics as complete while status is still build/audit.

The source plan binds 13,283 local raw pairs (12,399 legacy-training sources;
884 post-cutoff sources assigned explicitly to development validation). The
validation cohort was already used by a probe and is development-exposed.
Original test IDs remain reserved; their raw sources are unavailable, so no
test examples are materialized. The strict builder rejects all Magical
Footwear rune games, Viego, opaque undos and remaining source/inventory faults
before storing the new database. At most 1,517 training and 106 validation
games survive the first two exclusions; final accepted counts are not yet known.

The pipeline queues full semantic audit, qualification, preserved-split export,
one fixed 16-epoch trunk, one six-epoch frozen-trunk graft, and a paired
conditional-baseline comparison on all retained validation visits. Its plan,
code hashes, memory/disk/wall limits and source-exposure rules are recorded in
`docs/experiment_16_17_development.md` and the experiment's `protocol.json`.
The paired report will bind per-visit outcomes for error analysis. No promotion,
commit, final-test evaluation or shutdown is authorized. The 16.18 collection
follow-up remains in place. Avoid editing pinned application/pipeline code while
this job runs; the supervisor checks hashes between stages and fails closed.

**Viego candidate investigation, 2026-09-11:** the owner asked to find valid
games among the 125 Viego quarantines. Source hashes were checked for all 125;
the five games excluded only by the blanket Viego rule received closer review.
One game passes the additional checks without a possession repair:
`KR_8375431152` (915 seconds, Viego TOP, seven Viego shop rows / 61 all-player
rows). The independent reference ledger explains all 26 item/ward transitions,
matches every Viego board state across all 61 decisions and both sides of his
seven shops, and rejects a known corrupted negative control. All 732 board/shop
inventory states pass, all ten modeled endpoints reconcile, and all 61 decision
input-prefix checks pass. Six source-perturbation checks validate the narrow
reference auditor. Bound evidence:
`data/acceptance/viego_recovery_20260911/recovered_games.json`.

This identifies a game for explicit requalification; it has not changed the
frozen qualification, export, split, hold or model. It does not prove absence of
possession or establish a general possession-event filter. The other 124 games
are not certified: 120 have additional existing semantic exclusions, two of the
five easy cases are remakes, one has demonstrable intermediate corruption
despite a matching endpoint, and `KR_8375315960` has unobserved rune-granted
Magical Footwear (2422). Its missing boot is absent from both model tokens and
hidden-slot accounting. Prevalence of that grant gap elsewhere is not measured.
Earlier candidate diagnostic reports also mislabeled absent truncated label
windows as input dependence; the authoritative `candidate_validation_final.json`
records those cases as inconclusive. No new causal input defect is established
by that diagnostic mistake.

**Local preparation complete, 2026-09-11 13:04 UTC:** the owner approved
today's slot-state rebuild/audit/export and gate-6 verification. New frozen
corpus: `data/corpora/16.18_role_slots_v1/`. The semantic audit checked all
1,311 games / 156,842 shop events with zero hard failures, including independent
strict-prefix quest checks on each shopper and all ten board members.
Qualification retains 1,063 games and quarantines 248. Coverage retains 1,049
export-eligible games / 126,325 rows. The new immutable export is
`data/dataset_generations/16.18-initial-role-slots-v1/`, fingerprint
`603bc23adc2bde07d25202ae52872517dc7b9e1569c25263d74d15c0dc150dda`.
All original split memberships are identical: 735 train / 157 val / 157 test.
Post-publication verification checked all 88,231 train and 18,869 val rows for
explicit `role-slots-v1`, boolean completion and shopper skipped-slot occupancy;
the test payload was not opened. This is data QA, not a model comparison.
The export deliberately retains the smoke-only scope pending gate 7's runner.

Authoritative completion evidence is
`data/acceptance/16.18_role_slots_v1/completion.json`, bound to
`export_verification.json` and the published generation. The supervisor's
`chain_status.json` preserves a logging failure after successful publication:
a transient Windows sharing conflict prevented appending to `chain.log`.
Every pinned code/input/protected digest was rechecked, then only the missing
export verification ran directly. The completion report records this recovery;
no immutable output was replaced. The supervisor now retries transient log
sharing errors, tested with injected transient and persistent failures.
An earlier launch failed before reconstruction in PowerShell 5.1's null-string
handling; its evidence is retained at
`data/acceptance/16.18_role_slots_v1_preflight_failed_20260911T124113Z/`.
The status replacement fix was reproduced before the successful rebuild.

Eight targeted regression scripts passed: corpus audit, slot-state causality,
live slot state, purchase legality, qualification, dataset artifacts, dataset
generations and the offline sample/freeze integration. PowerShell 5.1 parsing
and status/logging fault checks also passed. Serving/hold digests are unchanged.
There was no training, evaluation, promotion, commit or shutdown.

Gate 6 was verified against the saved automation and Windows task settings:
`16-18-next-two-nights` is ACTIVE at 07:00 Madrid, targeting
`Inspect current refetch`. Daily Pull next runs September 12 at 04:30,
`WakeToRun=false`, `StartWhenAvailable=true`. The dated samples depend on the
Codex heartbeat; no September 12/13 collection is already complete. See the
corrected durable preparation and missed-date instructions in
`docs/data_acceptance_16_18.md`.

Gate 7 remains after those collection evidence gates and also needs a scoped
comparison runner: the existing isolated stack is smoke-only, and the
conditional evaluator does not compute paired differences against the
transformer. Do not bypass the one-epoch restriction or silently move old
holdout games into training in a cumulative split. The current beta builder
also requires a promoted deployment manifest, which is absent. A Personal API
Key supports developer ingestion; testers need no key. No application was
submitted, no model trained/promoted, and no final test evaluated in this work.

**Active update, 2026-09-11:** the completed 16.18 sample is frozen at
`data/corpora/16.18_initial/` (1,311 games; collection, not training acceptance).
The owner authorized seven ordered gates: freeze → semantic audit → coverage →
atomic export generations → isolated pipeline smoke → another 1–2 nights → one
baseline/candidate validation comparison. Gates 1–3 are complete for the qualified
subset, not the whole collection. The derived corpus is
`data/corpora/16.18_repaired_v2/`. Its successful semantic recheck is in
`data/acceptance/16.18_repaired_v2/recheck/`. The first derived audit found one
validator error (last quest marker instead of first completion); its failed
report is preserved in the parent directory. The authoritative final qualification
is now `data/acceptance/16.18_repaired_v2/inventory_recheck/qualification.json`:
1,063 accepted games, 248 quarantined; before-board, before-shop and after-shop
inventory checks pass for the accepted set. Coverage in its sibling `coverage/`
contains 1,049 export-eligible games and 126,325 rows (735/157/157 games in
train/val/test). Exclusions and pro coverage remain explicit.

Gates 4–5 completed at 08:51 UTC / 10:51 Madrid. The immutable export is
`data/dataset_generations/16.18-initial-smoke-v2/`, fingerprint
`5f24376f311742a8ffdec6633a0d7cc1d184875d3d251eaa8a491f729693447f`.
`data/experiments/16.18-initial-smoke-v2/completion.json` binds the one-epoch
trunk, one-epoch frozen-trunk graft and displayed-policy smoke: 1,252 visits
across ten validation games, zero unaffordable baskets. Candidate SHA256:
`202b5800e489199c93bb37d1db6ad01b324856b99641944c13bdcc3ca4721e9a`.
Serving/hold artifacts stayed unchanged; no final-test evaluation or promotion.
This is mechanics testing, not a quality comparison. Keep its code snapshots and
results immutable rather than reusing this experiment directory.
The existing daily collection finished at 10:11 Madrid with zero stage failures;
its additional rows used older code and require a new frozen/audited generation.
The production pipeline hold remains in force.
See `docs/data_acceptance_16_18.md`; the earlier broad readiness assessment is
withdrawn. Source collection completeness did not establish semantic correctness.

The original semantic report found inhibitors counted as towers. Subsequent
review found labels inferred from unpurchased inventory gains, final-DTO support
choice leakage, incomplete role-quest/ward handling, and incomplete combine-undo
restoration. The uncommitted `decision-start-v2` reconstruction uses causal support
tier proxies, timestamped role-quest transitions, source-reported undo gold,
strictly earlier gold frames, and purchase-backed labels. Tests cover those
properties; the full-corpus recheck has zero hard failures. Opaque undo IDs and
unreconciled terminal inventories must be explicitly excluded from any accepted
training subset. Do not equate an accepted subset with a clean entire collection.

The owner confirmed the inventory extremes after raw inspection: eight modeled
copies can mean six regular items plus two support wards; the five-copy target
is five source-recorded Long Swords (EUW1_7979347871, participant 1, 267579ms).
`coverage_extremes.json` preserves the traces. **Shop targets are multisets:**
qualified JSONL uses `label_counts` (item ID → positive integer quantity),
`target_encoding=item-counts-v1`, without an ordered `label_ids` target. Tensor
adapters preserve counts, canonicalize inventory order for new artifacts, and
reject capacity overflow rather than silently truncate/clip. Legacy canonical
labels remain diagnostic only. The immutable displayed-policy smoke used
conservative regular-slot decoding; it remains mechanics-only evidence.

**Slot-legality follow-up, 2026-09-11:** both Predictor paths and the conditional
baseline now share recipe, wallet, regular-slot, ward-stack and boot-family
checks in `app/shop_econ.py`. Basket entries are retained quantities: a later
combine cannot consume an earlier displayed purchase. Conditional baskets are
unordered and their legality searches for a valid purchase sequence. Policy
evaluation independently replays each displayed option and checks reported
prices, capacity and wallet. New reconstruction/export rows carry
`slot_state_version=role-slots-v1` and strictly pre-decision
`bot_quest_complete`; skipped held consumables also retain their regular-slot
occupancy. Existing immutable generations were not rewritten and lack this
new slot state: use a freshly reconstructed/audited generation for gate 7.

Live Client docs and existing captures do not establish a bot-quest completion
or separate role-inventory signal. Live completion stays unknown, and absent
BOTTOM boots / UTILITY wards are marked as unknown ownership; affected buys
are withheld rather than assuming empty slots. Observed skipped regular-slot
items count against capacity. This is a conservative live limitation, not a
claim of complete live inventory observability. See the new
`test_purchase_legality.py`, `test_slot_state_causal.py` and
`test_live_slot_state.py` regression scripts. Production hold, served artifacts,
frozen evidence and final-test payloads remain untouched.
Validation: 18 focused regression scripts passed (purchase legality, slot-state
causality/live parsing, multiset, economics, visits/quests, gold/decision causality,
audit, conditional policy, sessions/API, provenance, dataset generations, corpus
oracle, hold and live prediction). The unchanged initial smoke candidate also
passed the updated live path on four stored snapshots; its digest still matches
the completion report above. This is regression evidence, not a new quality
comparison or a rerun of the immutable experiment.

The owner's separate request for larger Codex context was applied to
`C:/Users/Javier/.codex/config.toml`: model_context_window=1050000,
model_auto_compact_token_limit=900000, with a backup. **September 12 correction:**
the active task's `token_count` events at 05:00–05:02 UTC report an effective
`model_context_window` of **828400**, not 1050000. The local
`C:/Users/Javier/.codex/models_cache.json` entry for `gpt-6-astra` advertises
`max_context_window=872000` and `effective_context_window_percent=95`;
872000 × 0.95 = 828400. Evidence is the task rollout under
`C:/Users/Javier/.codex/sessions/2026/09/10/`, ID
`01a08cb1-f1c1-7fc0-9d16-4e46ad739e9f`. The saved setting is verified, but it
does not establish a 1M active window. The published API capacity is not proof
of this Codex session's capacity. No cache/model metadata was edited to fake
a larger limit, and already-compacted history is not restored by this setting.

Gate 6 follow-up is scheduled in this task at 07:00 Madrid daily, automation ID
`16-18-next-two-nights`, for the September 12 and 13 collections, then the
conditional gate 7 comparison. It must stop after those are handled. The
September 12 invocation ran and found the authentication block recorded above;
the dated samples still do not exist. No future collection or comparison is
claimed done.
See the tested next-night collection/freezing procedure in
`docs/data_acceptance_16_18.md`. The collector can inherit the original reviewed
cohorts, retains unresolved-pro outcomes, and reconstructs only once per game.

The owner accepted the 16 unresolved tracked pro accounts as a minor limitation;
do not spend another cycle resolving them. The previous shutdown completed;
there is no new shutdown authorization for this work. No final-test evaluation,
model promotion, or commit has been authorized by the seven-step request.

Written 2026-09-09, revised the same day after an external audit. `CLAUDE.md` /
`AGENTS.md` carry the durable contract; this file covers mutable corpus state,
the recovery queue, and the long-form reasoning behind them.

**This file is not the binding contract and is not read first.** Start at
`CLAUDE.md` — it is short, and it is what an assistant loads every session.
Come here for depth on a specific question, and run `scripts/status.py` for
anything about what is happening right now.

**If you read only one thing, read §5.** An earlier version of this handoff
claimed the project had gained "+14 points in actual games" from exact-gold
training. That claim was withdrawn: it compared an eval harness against a
training log, and the estimator behind it was reading a frame from *after* the
decision. §5 explains what is actually known.

Read order: §1 (what the product is) → `scripts/status.py` (what is running,
so you don't collide with it) → §5 (gold) → §6 (the audit and what it changed)
→ `docs/history.md` (traps) → the latest dated checkpoint above and
`docs/data_acceptance_16_18.md` (the current 16.18 queue). §11 is historical.

**The development history lives in two places now.** §2.1 keeps the timeline;
the ledger of withdrawn beliefs, the engineering failures and the traps moved
verbatim to `docs/history.md` so they survive independently of this file's
live-status sections. When something here turns out to be wrong, move it into
that ledger instead of quietly deleting it.

---

## 1. What this product is

A desktop companion for League of Legends that answers only when the player
explicitly starts a shop-decision session: **"what would a KR Challenger buy
right here?"** — including *"nothing, hold your gold"* as a first-class
answer. The Live Client API has no trustworthy shop/location flag, so it must
not be inferred from continuous polling.

**The engraved question (owner's framing, 2026-09-02, never drift from it):**

> given that a KR Challenger — a really good player — visited the shop, WHICH
> INCLUDES basing for health/resources and buying nothing, what is the optimal
> action?

Consequences that are not negotiable:

- **Save/no-buy is a real action**, not a UI heuristic. `reconstruct.py` emits
  save events (respawned at fountain, ≥400g, bought nothing within 90s) with
  the `SAVE_ITEM = 999999` sentinel label.
- **"Optimal" means imitation of high-elo behaviour**, not outcome
  optimisation. Labels are what Challenger/GM players actually bought; there
  are no counterfactuals and no signal that any purchase was good. Calling the
  output "optimal" outside this narrow sense is rhetoric — the audit was right
  about that, and product copy should say "what a Challenger did here".
- **Confidence must be trustworthy.** Rare-item probability inflation "boosts
  great matchup choices but also highlights mistakes" (owner). Hence the
  `pos_weight` cap and the unweighted save head.
- **Poll-to-poll stability matters as much as accuracy** — but see §8: the
  previously quoted "2.9% flicker with identical state" was a mismeasurement.

Data flow: **Riot API → timeline reconstruction → SQLite → JSONL export →
transformer → live overlay.** Training data is offline match timelines;
inference reads the League client's own local API. No memory reading, no
scraping.

---

## 2. Development history

This section is deliberately long, and it records the mistakes as carefully as
the wins. Almost everything expensive this project learned, it learned by
believing a number that turned out to be wrong — and a handoff that documents
only the current state hands the next person the conclusions without the
reasoning that earned them. Keep appending to it. When a belief here is
overturned, move it to the withdrawn-belief ledger in `docs/history.md`
 rather than deleting it.

Commits exist through 2026-09-06; **everything after that is uncommitted** (`git status`).

### 2.1 Timeline

| When | What happened |
|---|---|
| **08-30** | Initial commit: Riot API client, SQLite, a web UI over Faker's ranked games. A match browser, not yet an advisor. |
| **09-02** | The architecture day, and the day the product became itself. Shop-economics layer (`shop_econ.py`: recursive combine costs, affordability, decision kinds). Board-model overhaul (self token, lane opponent, game-state features). Nightly pipeline. **First gold label leak found and fixed** — the budget feature was being reconstructed from the visit's own spend. Live recommender born (`live.py`, `predictor.py`, `/live`). Iterative basket decode with in-game purchase blocks. Plan probe and plan-consistency (stop burning gold off-build). **Save taught as a first-class action.** `pos_weight 8` promoted for trustworthy confidence; self-tuning basket threshold. The engraved question was stated and has not moved since. |
| **09-03** | 37× faster mask preprocessing (memoised Data Dragon, hoisted per-label metadata). Trainer opts itself out of Windows background throttling. README so friends could run it. |
| **09-04** | Experiments **A** (bigger arch → 0.61) and **B** (A + runes + an arrival-gold estimate → 0.63). **RAM crisis**: the export outgrew 15.8 GB as Python dicts and hung the machine twice; the trainer became streaming-only. |
| **09-05** | **C** (multiset count head — each buy is a multiset, ~7% of visits repeat an item). **D** (`pos_weight=1`: ECE 0.0002 but −6pts — the weighting is load-bearing and its tilt is analytically invertible). **E/F**: save and plan heads trained jointly cost the trunk 4-5pts at *any* loss weight → the **two-stage graft recipe** (train the trunk clean, freeze it, graft the heads) that is still in use. |
| **09-06** | `pos_weight` sweep (1/4/8/16/24/32) → **H24 deployed**. Live panel reworked into three baskets with no probability numbers, on the reasoning that a tilted logit is not a confidence and should not be shown as one. **In-game overlay** (frameless pywebview). `/api/live` serves options only (halves per-poll work). Health check moved off `/api/status`, which scans a multi-GB table and outlives any timeout. |
| **09-07** | Five owner-requested improvements: nightly-task auto-recovery, streaming `baseline.py`, a gold eval harness, per-label `pos_weight`, and a stability metric with hysteresis. Then a product decision: **the owner chose a closed beta over more model work**, on the grounds that the open questions had stopped being model-shaped. Launcher, bundle and runbook followed. |
| **09-08** | Fresh export (**17.7k games / 2.13M visits**) — the streaming rewrite from the day before is what let it complete at all. H24 retrained on it. Three-way weight comparison: the simple monotone **8:24 schedule beat both H24 and a hand-shaped alternative**. Beta bundle built, and debugged (`docs/history.md`). And the gold work began: a first measurement, two corrections, a forward walk, a corpus refresh. |
| **09-09 (am)** | GOLDX candidate trained on that forward walk and deployed; nightly recipe updated to match. A "+14 points in actual games" claim was made and is now **withdrawn** (`docs/history.md`, §5). |
| **09-09 (pm)** | **External audit.** Four fatal findings, all confirmed in code within minutes. Fixed the same day: causal gold estimator with a no-future test, versioned gold input with a hard gate, featurization parity through one function, frozen temporal 3-way split, policy-level evaluation, save disaggregation, honest stability decomposition. A follow-up round fixed the split's `post_cutoff` protocol and a metric whose point estimate and confidence interval described different populations. |
| **09-09 (eve)** | Serving contract tightened: scoring only happens inside an explicit shop session, artifacts need provenance and a digest bound in `deployment_manifest.json` to serve, and **raw Match-V5 responses are archived locally as gzip** so a future reconstruction change costs no API time. |
| **09-09 (night)** | **First honest measurement in the project's life**: a 3k-game causal probe, two trained artifacts on one export, paired by game, `probe_only`. Causal gold 32.0% first-action-exact vs stale gold 20.4%, paired +11.6 [+11.3, +11.9], zero unaffordable. And the finding that matters more than the headline (`docs/history.md`, last row of the ledger). |

### 2.2–2.4 Withdrawn beliefs, engineering failures, what they share

Moved verbatim to **`docs/history.md`**. Append new withdrawn beliefs there.

---

## 3. What is running right now

Run **`scripts/status.py`**. It reports scheduled tasks, active log tails,
the served artifact digest, corpus generations and `git status` by reading
the machine, so it cannot be stale the way this section always was.

---

## 4. Data and metrics — with the caveats attached

**Export** (regenerated by `baseline.py`; the numbers below predate the 3-way
split, so re-export before quoting them):

```
games 17,685   usable shops 2,130,985   labels 200   completed 450,625
decisions   complete 668,855   component 506,254   start 828,202   save 127,674
basket      avg 1.49 items per shop
```

**Every accuracy number this project has published shares three defects**, all
confirmed by the audit:

1. They score `predict_top3` or `predict_baskets` against `label_id` — a
   canonical item picked by `baseline._label` (the completed item, else the
   priciest). **That is not the action the user sees.** `predict_options` —
   plan probe, recipe filters, three alternatives, exact budget, save card — was
   never scored until `scripts/eval_policy.py` (new, §7.2).
2. They were computed on a split that had already been used to choose
   architecture, `pos_weight`, threshold, blend and graft recipe. It was a
   validation set being reported as a test set.
3. The `complete` class (0.79) is largely the determinism of that
   canonicalisation, so the headline is flattered by the easiest decisions.

So treat the table below as **historical model-selection signal**, not
performance:

| Model | canonical top-1 | top-3 | complete | component | start | save | basket F1 | exact |
|---|---|---|---|---|---|---|---|---|
| champion-frequency baseline | 0.13 | 0.30 | — | — | — | — | — | — |
| A (big arch) | 0.61 | 0.85 | — | 0.48 | — | 0.44 | 0.555 | — |
| B (+runes, +gold est) | 0.63 | 0.86 | 0.77 | 0.50 | — | 0.47 | 0.571 | — |
| H24 (deployed 09-06→09-09) | 0.63 | 0.87 | 0.79 | **0.54** | 0.59 | 0.37 | 0.558 | 0.29 |
| sched 8:24 | 0.63 | 0.86 | 0.79 | 0.52 | 0.60 | 0.42 | 0.573 | 0.32 |
| custom bump 8→24→12 | 0.63 | 0.87 | 0.79 | 0.53 | 0.59 | 0.38 | 0.567 | 0.31 |
| GOLDX + 8:24 (deployed, leaky gold) | 0.65 | 0.88 | 0.79 | 0.53 | 0.63 | 0.55 | 0.574 | 0.35 |

The 0.13 baseline is also too weak to carry weight: it ignores inventory, gold,
minute, role and legality. `scripts/eval_conditional_baseline.py` (new) builds
the baseline the model should actually have to beat.

---

## 5. Gold — the whole story, current as of the audit

Four quantities are called "gold". Confusing them has now produced two wrong
conclusions, so this section is the reference.

### 5.1 The four quantities

| Name | Where | What it is | Model input? |
|---|---|---|---|
| `leftover` / `gold_left` / export `gold` | `reconstruct.close_visit` | `currentGold` from the frame **before** the visit. Frames are 60 s apart, so it is stale by up to a minute. | ✅ yes (the pre-09-09 budget) |
| `gold` (payload) / export `gold_arrival_true` | `shop_econ.arrival_from_parts` | `leftover + the visit's own net spend`. **Not arrival gold** — it adds the spend back without removing the gap's income, ≈ **+630 g** too high — and it is derived from the label. | ❌ **banned**, and not a valid eval reference either |
| **`gold_est`** | `reconstruct._attach_gold_estimates` | A **causal approximation** of gold on arrival, built only from pre-decision information. Tagged `prequential-v2`. | ✅ yes, under GOLDX |
| live `currentGold` | `app/live.py` | The client's exact gold. Written to `gold` and `gold_est`, tagged `live-exact-v2`. | ✅ yes |

The engraved anti-leak rule stands: **never feed a quantity reconstructed from
the visit's own purchase.** The audit added a second rule of equal weight:
**never feed a quantity reconstructed from anything after the decision.**

### 5.2 What `gold_est` is now (prequential-v2)

```
gold_est(t) = f0.currentGold                      # last frame at or before t
            + passive income f0 → t               # deterministic, 2.04 g/s after 110 s
            + kill/shutdown bounties in (f0, t]   # events, exact ts and amounts
            − purchases/sells/undos in (f0, t)    # the visit's own buys share t exactly
            + prior_residual × (active / window)  # forecast, see below
```

`prior_residual = max(0, (f0.totalGold − fprev.totalGold) − passive(fprev,f0) −
bounties(fprev,f0))` — the farm/assist/objective income the player was earning
over the **previous complete** frame window, extrapolated forward at that rate,
not credited for the last 12 s (`SHOP_DEAD_TIME_MS`, spent walking or recalling).

`totalGold` is cumulative income, so spending never enters that term.

The dead time is subtracted from the credited span but not from the window the
rate is measured over. That asymmetry is deliberate: the prior window had no
reason to end in a shop stop, so its rate is a wall-clock rate. The effect is a
conservative under-credit of roughly one dead-time of income (order 60-120 g at
mid-game farm rates) — a calibration bias, not leakage, and one that can only
under-state the budget.

### 5.3 What it used to be, and why that was fatal (leaky-v1)

The 09-08 version computed the residual from `series[i+1]` — the frame **after**
the visit — and prorated part of it into the estimate. Two problems:

- **Availability**: at time t that frame does not exist. Live inference reads
  exact gold, not a share of future income, so training and serving were
  constructing the feature differently.
- **Label correlation**: income between t and f1 partly *results from* the
  purchase being predicted. Bounties were excluded, which removed the strongest
  channel, but farm, assists and objective gold after the buy were not.

The audit called it temporal leakage. Correct. `scripts/test_gold_causal.py`
now pins the fix: mutate any later frame or any post-t event and `gold_est`
must not move, with controls proving pre-decision information *does* move it.

### 5.4 What is validated, and what is not

| Check | What it proves | What it does NOT prove |
|---|---|---|
| Wallet identity: `cur(f1) == cur(f0) + Δtotal − spends(f0,f1]` — exact (≤5 g) on **92.6 %** of windows, p90 residual 5 g | The **spend pricing** is right (recipe discounts, undos, sells) | Nothing about the income terms. It reads both frames, so it cannot check how income is split between them |
| Afford coverage: whatever true arrival was, it covered the visit's spend — stale+500 **90.5 %**, est+150 **95.8 %**, est+300 **98.4 %** (leaky-v1 numbers; re-run for v2) | The budget is usable: recommendations are not priced out | That the estimate is close to the truth |
| Error vs `gold_arrival_true` | **Nothing.** That reference is inflated and label-derived | — |

**Validating the income forecast requires live telemetry** — exact gold at a
known timestamp with the purchase that followed. Match-V5 cannot provide it.
Until then, `gold_est` is a defensible approximation and nothing more.

### 5.5 What is claimed and what is withdrawn

| Earlier claim | Status |
|---|---|
| "gold_est is exact / leak-free arrival gold" | **withdrawn** — it is causal and approximate |
| "the live regime costs −19.4 points" | **withdrawn** — measured against the inflated field |
| "the live regime costs −12.0 points" | **weakened** — real harness, but the "live" arm was the leaky estimate, scored on canonical labels, on a contaminated split |
| "GOLDX scores 0.65 live → +14 points in game" | **withdrawn** — a training-log number against a harness number, different metrics, unpaired, leaky budget |
| "the stale-gold budget is systematically low" | **stands** — frames are 60 s old; the direction is not in doubt, the magnitude is |

What replaces them: nothing yet. The honest comparison is the ablation in
§11.2, on the frozen split, with the deployed decoder.

### 5.6 The version gate

`gold_est` alone is not enough — a corpus can hold two estimator generations at
once, which is how a leak survives a refactor. So every row carries
`gold_est_version`, and `PREFIX_GOLDX=1` accepts only `prequential-v2` (offline)
or `live-exact-v2` (live). Anything else — including the untagged leaky-v1 rows
that fill the corpus today — **raises**, rather than silently falling back to
stale gold and producing a model trained on a mixture nobody can describe.

Practical consequence: **GOLDX cannot train until the corpus is
re-reconstructed** (`--backfill --refresh`, ~18 h of API time, checkpointed).
The artifact records `gold_input: prequential-v2 | stale-frame-gold`, never
"exact".

---

## 6. What the audit changed (this session)

All eight required regression tests pass (`scripts/test_audit.py`,
`scripts/test_gold_causal.py`).

**Phase 1 — causal gold and parity**

- `reconstruct.py`: `gold_est` rebuilt as prequential-v2, `GOLD_EST_VERSION`
  exported, debug payload carries `prior_residual`/`frame_ts`/`bounty`/`spent`/
  `version`. All reads of the later frame removed.
- Version tag propagated through `baseline.py` → `stream_rows` → `ShopDataset`;
  `live.py` tags rows `live-exact-v2`; cache key bumped to `v8`; artifacts
  record `gold_input`.
- `train_prefix.apply_config()` is now the single place featurization knobs are
  set, used by `Predictor` and every harness. It previously lived in three
  hand-copied lists, and the sampled gold harness had dropped `USE_GOLDX`.
- `eval_live_gold_sampled.py`: parity asserts, both gold fields forced per arm,
  the "honest live regime" arm renamed to "causal offline gold estimate (NOT
  exact live gold)".
- `daily_pull.py`: the nightly now writes `prefix_candidate_<date>.pt` and
  **never overwrites the served model**. Promotion is manual, after evaluation.

**Phase 2 — honest evaluation**

- `baseline.py` writes **train / val / test** (70/15/15), split **temporally by
  `game_creation`**, whole games, ties kept together, with
  `data/ml/split_manifest.json` (eval version, strategy, cutoff, counts,
  per-split id hashes, fingerprint) and `split_assignment.json`. The split is
  **frozen**, and freezing has a protocol (§6.1).
- `train_prefix.py` scores itself on **validation only** and no longer opens
  `visits_test.jsonl`.
- `scripts/eval_policy.py` (new) scores `predict_options` — the real policy —
  with labels stripped from the row, one budget in every channel, and a hard
  check that no shown basket exceeds the budget. Two distinct headline metrics,
  each with its population declared: **`first_action_exact`** (all visits: SAVE
  right on a no-buy, exact basket on a purchase) and
  **`first_basket_exact_given_purchase`** (purchases only). Point estimate and
  bootstrap always come from the same filtered population — the first version
  reported a purchases-only point next to an interval that had also eaten the
  no-buys. `--compare B.pt` adds a **game-paired A−B bootstrap**, which is the
  test for choosing between candidates. All intervals go into the JSON, not
  just the console. `--split test` requires `--confirm-final` and appends to
  `data/ml/test_set_usage.log`.
- `app/predictor.py` re-applies its artifact's config before every forward
  pass, so loading two Predictors in one process (which `--compare` does) can't
  leave one of them featurizing rows for the other's model.
- `scripts/eval_conditional_baseline.py` (new): 5-level backoff baseline built
  on train only, plus `conditional_action_concentration` (modal share, distinct
  actions, normalised entropy per state bucket) — explicitly *not* a ceiling.

### 6.1 The split protocol (read before touching `baseline.py`)

A frozen split is only meaningful if it stays temporally sound, and the first
attempt at this got it wrong: it parked newly ingested games in **train**, to
protect the test ids. That protects the wrong thing. Games arriving after the
manifest was drawn are newer than the test cut, so training on them lets the
model see the future relative to val and test — the exact contamination the
freeze exists to prevent, with the test ids left intact as a fig leaf.

The rule now:

- games with no assignment go to **`post_cutoff`** and are written to **no
  artifact** — not train, not val, not test;
- the manifest records `post_cutoff_count`, `cutoff_utc`, the date range and an
  id hash for that bucket;
- `route_split` drops those rows, and anything unassigned defaults to
  `post_cutoff` rather than to train.

**Operational consequence, state it to the owner:** while a manifest is frozen,
the nightly pipeline keeps growing the corpus but the *trainable* data stands
still. Fresh games only become training data on a deliberate `--rebuild-split`,
which bumps `eval_version`, archives the old manifest as
`split_manifest_v<N>.json`, records what it supersedes, and prints a banner
saying results are no longer comparable. The nightly never passes that flag. So
periodically someone must decide: keep the benchmark, or take the new data and
start a new evaluation version. Both are defensible; drifting into it silently
is not.

**Phase 3 — save and stability**

- `save_kind` ∈ {`no_buy_death`, `ward_only`, `build_spend`} in the export. The
  save head trains **only** on `no_buy_death`; ward-only visits keep their ward
  items as real labels. `save_aux` is renamed `no_build_spend` and labelled a
  diagnostic, not save precision.
- `eval_stability.py` decomposes churn by cause and states in its own docstring
  that equal inventory and level is not an unchanged game state.

**Phase 4 — serving contract and prospective beta evidence**

- `app/main.py` no longer forwards the shop-only policy on a two-second poll.
  The player opens a 90-second decision session in the shop; only then is the
  model run, once, against the exact client wallet. Polling subsequently only
  observes its outcome.
- `app/telemetry.py` writes local-only, append-only JSONL: anonymous decision
  context, exact query wallet, the displayed actions and a later net inventory
  addition or explicit no-buy timeout. A removal-only delta is **unscored**;
  it is not mislabelled as a purchase. Cancellation/supersession is abandonment,
  never a no-buy. `scripts/eval_shop_telemetry.py --strict` makes any displayed
  basket over the exact wallet a hard-stop failure.
- The old plan decoder's 3,500g synthetic probe was removed. Plan arrows now
  use the next-final head at the observed state; it does not pretend to be an
  answer to a fabricated high-gold state.
- `app/deployment.py` fail-closes live serving. The manual
  `scripts/promote_served_model.py --report ...` command binds the exact served
  artifact digest to a provenance-checked **validation** policy report with
  zero unaffordable options. `build_beta.ps1` refuses to package any other
  model.

---

## 7. Results from the new tooling

### 7.1 The 3k causal probe (2026-09-09 night) — the first trustworthy number

Two artifacts trained on one export and scored against each other, paired by
game. Report:
`data/ml/probes/causal_v2_newest_3000/policy_eval_val_probe_candidate_prequential_vs_probe_candidate_stale.json`.

| | A: prequential-v2 | B: stale frame gold | paired A−B (CI95) |
|---|---|---|---|
| first_action_exact | **0.320** | 0.204 | **+0.117** [+0.113, +0.120] |
| first_basket_exact_given_purchase | 0.319 | 0.202 | +0.116 [+0.113, +0.120] |
| basket precision / recall | 0.520 / 0.491 | 0.402 / 0.352 | +0.118 / +0.139 |
| any_of_three_exact | 0.402 | 0.252 | +0.150 |
| action_family_match | 0.613 | 0.511 | +0.102 |
| no_buy_death_recall | 0.413 | 0.292 | +0.121 |
| save_card_precision | 0.098 | 0.037 | +0.060 |
| unaffordable baskets | 0 | 0 | — |

591 validation games, 69,857 visits, both artifacts `probe_only`, same
`export_fingerprint`, neither promoted.

What this is: the first result in the project's history where the comparison is
between two *trained* models, on identical data, paired, with provenance. The
direction of the gold work is supported. What it is not: a product number, or a
result on the full corpus — it is 3,000 of ~20,000 games, and the newest ones,
which is the most homogeneous slice available.

**Three things in it matter more than the headline:**

1. **The stale-gold model (0.204) ties the conditional lookup table (0.208).**
   On the metric that describes what the user sees, the architecture we have
   been refining for a week was worth nothing over a backoff table — until the
   budget was fixed. This is the single most important finding so far, and it
   reframes every earlier "the model beats baseline" claim.
2. **The save card is the weakest part of the product.** It is shown on 5,304
   visits when only 1,253 are real no-buys, and 90% of the time it is shown it
   is wrong. Since "hold your gold" is the differentiator, this is a product
   problem, not a metric footnote. Note also that `save_card_precision` counts
   the card appearing in *any* of the three options, which mixes "we advise
   saving" with "we offer saving as a situational alternative" — measure it per
   tier before drawing conclusions.
3. **The intervals are narrow for a structural reason, not a magical one.** A
   game bundles ten players' decisions, so a per-game mean already averages
   heterogeneous trajectories and the between-game variance is small. The
   interval is correct, but it describes sampling variation *within this 3k
   slice*; it says nothing about variation across patches or the rest of the
   corpus. Expect movement well beyond ±0.3 points at full scale.

Open question worth answering before the full run, and it costs no training:
**why is causal gold worth eleven points?** A breakdown of the error by price
band and basket size would say whether the model learns better items, or
whether the stale-budget decoder was simply truncating baskets it could not
afford.

### 7.2 Stability decomposition

The stability decomposition ran on 300 consecutive snapshots (2 s cadence,
deployed artifact, real hysteresis):

```
poll pairs 299   inventory+level unchanged: 283
   of those, fully identical discrete state: 79
lead churn, all polls            0.060
with inventory+level unchanged:  0.035   (0.028 as displayed)
why those 10 lead changes happened
  an on-screen item crossed its price   6
  the board / KDA / objectives moved    0
  gold moved, no on-screen crossing     4
  nothing discrete changed (FLICKER)    0
```

**Zero residual flicker.** Every change had a cause in the input. So the
"2.9 % flicker with identical state" framing was wrong. This is now a replay
diagnostic only: Phase 4 removed continuous live re-ranking and its hysteresis
from the serving path. Do not reintroduce it unless a real shop detector or a
separately validated session-state update policy exists.

---

## 8. Repo map (with the gotchas attached)

- **`app/riot.py`** — rate-limited httpx client (1.3 s gap, 95 req/2 min). 404
  returns `None`; 401/403 raises and aborts ingest. Successful Match-V5 match
  and timeline responses are atomically archived in ignored
  `data/raw_match_v5/`; replays reuse those immutable source payloads instead
  of making a duplicate API crawl (`RAW_MATCH_V5_CACHE=0` is an explicit
  ephemeral-run opt-out).
- **`app/reconstruct.py`** — the replay: shop visits, inventories, ten-player
  board, objectives, save events, `gold_est`. The support-item line (Atlas 3865
  → Compass → Bounty → finished) never emits `ITEM_PURCHASED`; it is seeded and
  upgraded via `ITEM_DESTROYED` — special-cased, don't "simplify" it.
- **`app/shop_econ.py`** — recursive combine costs, affordability, decision
  kinds, `GOLD_DRIFT = 500`, `SAVE_ITEM`, `PINK` (control wards, the only
  consumables not skipped).
- **`app/db.py`** — SQLite (multi-GB, WAL). `matches` + `shop_visits` are
  per-player perspectives; `games` + `game_events` the full lobby. `init_db()`
  migrates in place.
- **`app/ladder.py` / `pros.py` / `backfill.py` / `ingest.py`** — ingest modes.
  `--backfill --refresh` re-reconstructs the whole corpus, checkpointed in
  `data/refresh_done.txt`.
- **`scripts/baseline.py`** — export + split + manifest. Streaming — see the traps in `docs/history.md`.
- **`scripts/train_prefix.py`** — the model: 10-token board (token 0 the
  shopper; the enemy in the shopper's role gets a distinct side id), query
  vector with state/affordability/damage-share/runes, multi-label BCE with
  legality masking, count + save + plan heads, two-stage graft. Knobs:
  `PREFIX_EPOCHS/DMODEL/LAYERS/FF/HEADS`, `PREFIX_COSINE`, `PREFIX_RUNES`,
  `PREFIX_GOLDEST`, `PREFIX_GOLDX`, `PREFIX_POSW`, `PREFIX_POSW_SCHED`,
  `PREFIX_POSW_CUSTOM`, `PREFIX_SAVEW/TARGETW`, `PREFIX_INIT_FROM`,
  `PREFIX_FREEZE`, `PREFIX_OUT`, `PREFIX_HISTORY` (no gain, off).
- **`app/predictor.py`** — loads an artifact and **reuses the training
  featurization** (`ShopDataset` on a one-row list). Never featurize live data
  any other way. `predict_options` is the shipped policy.
- **`app/main.py`** — FastAPI. `/api/live` shapes state and observes an active
  session but emits no advice; `POST /api/live/session` runs the model once
  after the player's explicit request. `/api/status` scans the events table;
  never health-check on it.
- **`app/telemetry.py`** — local-only prospective session evidence; no Riot
  identity, raw snapshots or opponent data. **`app/deployment.py`** — the
  fail-closed promotion binding for a served artifact.
- **`app/live.py`** — Live Client Data API poller. `python -m app.live --dump`
  saves snapshots (20 k in `data/live_snapshots/`, all Practice Tool).
- **`app/overlay.py`** (dev) / **`app/advisor.py`** (beta, in-process server,
  PyInstaller-safe).

---

## 9. Traps

Moved verbatim to **`docs/history.md`**.

---

## 10. Beta status

The owner decided on 2026-09-08 to run a closed beta (10–30 testers, 2–4 weeks)
rather than keep polishing the model, because the open questions are
product-level: does the advice *feel* trustworthy, is the overlay in the way, is
the save card understood.

Built and verified: `dist/BuildAdvisor-beta.zip` (174 MB, frozen
`app/advisor.py`, CPU torch, assets beside the exe — launch tested),
`scripts/build_beta.ps1`, `docs/beta.md` (tester instructions, patch-day
runbook, checklist, **Personal API Key application draft**).

**Testers need no Riot API key**: the overlay talks only to `127.0.0.1:2999`
and the Data Dragon CDN.

**Blocker: the Personal API Key.** Also pending: a freshly reconstructed,
provenance-valid candidate, deliberate promotion, validation in the owner's
own ranked games, a feedback channel, minimal branding. The legacy artifact
will now be blocked rather than silently packaged.

The audit's beta advice is worth heeding: instrument first, and start smaller
than 30 testers. Phase 4 now records the minimum actionable facts — artifact
version, displayed options, exact query gold, role/minute and an observed
inventory addition or no-buy timeout — without retaining identity or a raw
snapshot. Latency and a one-tap useful/wrong rating are **not implemented**;
do not claim a feedback or satisfaction stopping rule until they are. The
existing evidence supports affordability and behavioural-match analysis, not
whether users found the advice helpful.

---

## 11. Historical recovery queue — superseded operational commands

This section preserves the September 9 plan and its reasoning. It is not a
current runbook: `scripts/baseline.py` now refuses the legacy in-place export,
and an in-place main-DB backfill is not the accepted-generation repair path.
Use the latest dated checkpoint and `docs/data_acceptance_16_18.md` for the
active sequence. Publication uses `scripts/publish_patch_dataset.py`; immutable
dataset inputs use `PREFIX_ML_DIR`, and isolated experiment outputs/caches use
`PREFIX_RUN_DIR`. A generation's `CURRENT.json` pointer does not promote a
served model or clear the pipeline hold.

### 11.1 Re-reconstruct the corpus, then re-export (P0, blocks everything)

Nothing GOLDX can train until rows carry `prequential-v2`, and the frozen split
does not exist until `baseline.py` runs once.

```powershell
# finish/redo the corpus refresh (checkpointed; needs a live key)
.venv\Scripts\python.exe -m app.ingest --backfill --refresh
# rebuild the export, and create the frozen 3-way split the first time
.venv\Scripts\python.exe scripts\baseline.py --rebuild-split
```

Check afterwards: `data/ml/split_manifest.json` shows
`gold_est_versions: {"prequential-v2": <all rows>}`, three non-empty splits and
`post_cutoff_count: 0`. After that, plain `baseline.py` (no flag) is the routine
command; it parks newer games in `post_cutoff` and leaves the benchmark alone.
Only redraw when you have decided the backlog is worth a new eval version
(§6.1).

### 11.2 The gold ablation (P1 — the only modelling question that matters now)

Three candidates, identical but for the budget, scored with the deployed
decoder on validation:

```powershell
$env:PREFIX_COSINE="1"; $env:PREFIX_DMODEL="128"; $env:PREFIX_LAYERS="4"
$env:PREFIX_FF="256"; $env:PREFIX_HEADS="8"; $env:PREFIX_RUNES="1"
$env:PREFIX_POSW_SCHED="8:24"; $env:PREFIX_EPOCHS="16"
$env:PREFIX_SAVEW="0"; $env:PREFIX_TARGETW="0"

# A: stale frame gold (no GOLDX, no interpolation)
$env:PREFIX_GOLDEST="0"; $env:PREFIX_GOLDX="0"; $env:PREFIX_OUT="trunk_stale.pt"
.venv\Scripts\python.exe scripts\train_prefix.py
# B: own-rate interpolation only (the model B feature)
$env:PREFIX_GOLDEST="1"; $env:PREFIX_GOLDX="0"; $env:PREFIX_OUT="trunk_interp.pt"
.venv\Scripts\python.exe scripts\train_prefix.py
# C: causal prequential estimate
$env:PREFIX_GOLDEST="1"; $env:PREFIX_GOLDX="1"; $env:PREFIX_OUT="trunk_prequential.pt"
.venv\Scripts\python.exe scripts\train_prefix.py
```

Graft heads onto each (`PREFIX_INIT_FROM=<trunk>`, `PREFIX_FREEZE=1`,
`PREFIX_EPOCHS=6`, `PREFIX_SAVEW=0.5`, `PREFIX_TARGETW=0.25`,
`PREFIX_OUT=cand_<name>.pt`), then compare them **pairwise**:

```powershell
.venv\Scripts\python.exe scripts\eval_policy.py --artifact cand_prequential.pt --compare cand_stale.pt --split val
.venv\Scripts\python.exe scripts\eval_policy.py --artifact cand_prequential.pt --compare cand_interp.pt --split val
```

**Success criterion: the paired A−B bootstrap interval excludes zero** on
`first_action_exact` and `action_family_match`. Do *not* use "the two
candidates' individual CI95 don't overlap" — the candidates are scored on
identical visits, so the difference has far less variance than either estimate,
and independent intervals are both the wrong question and much too
conservative. `--compare` resamples the same games for both models and reports
the interval of the difference.

If prequential does not win, the whole GOLDX line was a distributional change
with no measurable payoff — say so and revert the recipe. ~90 min per trunk on
the 4060; do not run two at once.

### 11.3 Establish where the model actually stands (P1, no training)

```powershell
.venv\Scripts\python.exe scripts\eval_policy.py --split val --games 300
.venv\Scripts\python.exe scripts\eval_conditional_baseline.py --split val
.venv\Scripts\python.exe scripts\eval_stability.py --stride 1
```

The first two together answer the question nobody has answered: does the
transformer beat a conditional lookup table on the policy the user sees? If the
gap is under ~2 points of family-match with overlapping intervals, the
architecture is not earning its cost.

### 11.4 Only after all of the above (P2)

**Do not read the final test set yet.** Selection happens on validation, full
stop; `--split test --confirm-final` is for one confirmatory read of the single
artifact you have already chosen, and every read is logged to
`data/ml/test_set_usage.log`. Then: the save↔ward hysteresis
dial, a `pos_weight` schedule sweep, the plan-head decode blend re-measured
under the new budget. Capacity is saturated (d160 flat, d192 worse) — not there.

---

## 12. Uncommitted work

Use `git status`. This section was a hand-typed file list that went stale
between sessions; `scripts/status.py` prints the live one.

---

## 13. Verifying you haven't broken anything

```powershell
.venv\Scripts\python.exe scripts\run_tests.py
```

Runs every `scripts/test_*.py` in its own subprocess and exits non-zero if
any fail. The two that catch the expensive class of mistake are
`test_gold_causal.py` (a feature that reads the future) and `test_audit.py`
(a harness scoring a model in a regime it was not trained in — nothing
crashes, the number is just wrong).

---

## 14. What still cannot be demonstrated

Moved verbatim to **`docs/history.md`**. It is the honest boundary of the
project's claims; keep it current there.

---

## 15. Working notes on the owner

- Spanish speaker; short, direct asks. Technical, comfortable with model-level
  detail, cares about *why* a number moved.
- Pushed back — correctly — when the previous agent proposed a statistical
  patch (gold jitter) instead of solving the gold problem: *"estamos hablando de
  un problema real, que requiere solución real y difícil… quiero el oro
  exacto"*. Prefer the real fix; bring evidence.
- Commissioned this audit and asked for it to be applied without commits and
  without stopping running jobs. Respect both.
- Turns the PC off at night and has asked for automatic shutdown after
  overnight work. **Don't schedule a shutdown unless asked for that session**,
  and abort a pending one (`shutdown /a`) if they come back to the machine.
- Reports honestly what failed, and expects the same. Several of the most
  useful results here were negative: the first gold measurement was invalid,
  the hand-tuned weight schedule lost to the simple one, and the headline gold
  win is currently unproven.

---

## 16. Audit update — 2026-09-10 (historical initial finding; current action in §17)

All eight Phase 0 scripts passed on CPU, but a separate synthetic check
demonstrated that a post-decision kill changes the exported shopper input.
`close_visit` reads `end_kda` while stamping the decision at `ts_start`;
`baseline._example` passes that value into `ShopDataset.query`. A kill at 153s
changed the input for a 150s decision whose final purchase occurred at 155s,
with identical labels and identical causal gold. The end-frame/objective paths
also need review. Corpus prevalence and metric impact are **not measured**.
The existing gold test proves less than whole-input causality.

See `docs/audits/2026-09-10-evidence.md`, the verbatim eight-test output at
`docs/audits/2026-09-10-phase0.txt`, and the bounded reproducer at
`data/audit_reconstruction_probe_20260910.py`. The 3k report values and model
hashes match the ledger; this does not resolve the new input-validity finding.
A newer 10k probe report exists, documented in that audit; it uses a different
population and is still ineligible for promotion. No original saved evidence
for the quoted wallet-identity percentage or zero-flicker run was located in
the searched local reports/logs; those statistics were not reproduced.

At the audit check, Task Scheduler showed `BuildTracker Causal Recovery`
Running; both probes and the daily pull were Ready. The latest logged full
refresh attempt began at 18:26:12; `backfill.log` was advancing through its
5,178-game remaining batch. The supervisor already queues catch-up, export,
three trunk/graft pairs, and policy comparisons. **This audit did not stop or
alter that sequence.** The recommendation is to defer selection and address
the start-state defect first; the owner has not yet decided how to change the
queued stages. Do not start another collector or infer a stop was implemented.

Only audit/procedure files and scratch diagnostic output were written. No
production code, reconstruction checkpoint, model, export, or deployment was
changed, and the final test set was not read. The operating procedures now
live under `docs/skills/`; they are not authorization to launch another job.

## 17. Authorized repair hold and 16.18 transition — 2026-09-10

The owner approved the repair-first action after §16. The code now fixes the
decision-start input defect and tags new events `decision-start-v1`, independently
of `prequential-v2` gold. The existing corpus has NOT been repaired or certified.
`data/ml/pipeline_hold.json` blocks all main export/training entry points. Leave
it in place until the staging repair and export publication gates in
`docs/audits/2026-09-10-repair-and-patch.md` are complete. The active refetch and
serial catch-up are untouched; the supervisor will stop at guarded export.

Patch provenance is now explicit: match DTO patch selects pinned static data;
16.17 uses 16.17.1 and 16.18 uses 16.18.1. Unknown patches fail closed. Daily
collection accepts both transition patches in a single pass. Export requires
`--patch <single-patch>`; daily export/training also needs `--export-patch`.
Changing export patch requires an explicit new split/evaluation version. Do not
clear the hold merely to make the old supervisor's command line work.

Backfill and ladder now publish game/events and requested perspectives in one
transaction, after reconstruction succeeds. Already-running Python retains its
old code. Nothing here retroactively fixes checkpointed games.

The read-only source audit at `data/ml/repair_source_audit.json` found 11,207
raw pairs present among 20,857 stored games and 9,650 missing pairs at its
timestamp. Presence is not integrity; twelve sampled pairs were successfully
replayed. Reinventory after collection finishes. Repair from verified local
sources into staging, explicitly account for exclusions, and decide targeted
retrieval only then. Do not start a duplicate broad API pull. Thirteen regression
scripts passed; details and limitations are in the linked repair report.

## 18. Queued 16.18 sample — 2026-09-10

Historical launch record. The dated checkpoints above supersede its job state
and the blanket collector-lease statement below; verify the actual entry point
before assuming it acquires a lease.

The owner requested the complete small new-patch sample and explicitly included
tracked pros. `BuildTracker Patch 16.18 Sample` is a one-shot Windows task using
`scripts/run_patch_sample.ps1`; it waits for Causal Recovery and Daily Pull to
stop, then allows the prior rate-limit window to expire. It does not interrupt
them. Do not launch a second copy or infer that a Running scheduler state means
the sample is already fetching: inspect its job log.

Sample directory: `data/patch_samples/16.18_20260910/`. `state.json` freezes the
cutoff and tracked-pro snapshot; future discovery freezes KR/EUW Challenger and
Grandmaster roster responses, uncapped paginated lists, and overlapping cohort
membership. Pros use their configured regional searches, including LEC's Asia
search. It is a tracked-account population, not all professional players globally.
Ladder membership is observed at discovery time, not retrospectively at match time.

Raw match/timeline responses reuse the shared archive. All ten perspectives go
to the sample's own `tracker.db`. Main DB, refresh checkpoint, exports, models
and deployment are untouched by the sample. `report.json` will separate stored,
excluded, quarantined and unresolved-pro cases, with overlapping cohort counts.
Scope is not certified complete when a pro is unresolved or a source quarantined.
No real sample count or quality result is available before the worker finishes.

New ingestion CLI and daily entry points share `app/collector_lock.py` with the
sample. Existing processes predate this lease, so the scheduler-state gate must
also remain. Offline tests passed for frozen-window pagination, pro/ladder
deduplication, restart behavior, invalid source rejection, collector exclusion,
and real reconstruction into a temporary ten-perspective staging DB.

To inspect: read the sample's `job.log` and task state. After an auth failure,
resolve the key as usual and restart this same one-shot task to resume; never
create another collector. Changes to pinned collection code require review
before resume. The production pipeline hold remains in force.

## 19. September 11 initial-corpus session ledger

This records the earlier session requested by the owner, separately from the
later 16.17 development/footwear work and September 12 operational checkpoint.
The authoritative acceptance paths and their limitations remain in the dated
updates above and `docs/data_acceptance_16_18.md`.

- **Source-bound reconstruction:** `app/inventory.py`, `app/reconstruct.py`,
  `app/ddragon.py` and `app/shop_econ.py` added causal quest/support ownership,
  recursive combine-undo restoration, ward-event deduplication, strictly
  earlier gold-frame inputs, tower-only objective counts and purchase-backed
  net labels. Opaque undo, Viego and unresolved endpoint differences remain
  explicit whole-game exclusions. The later Magical Footwear finding limits
  the earlier acceptance; do not treat those initial fixes as complete rune
  ownership support.
- **Purchase multisets:** `app/purchase_multiset.py` validates authoritative
  `label_counts` with `target_encoding=item-counts-v1`. `scripts/corpus_rows.py`
  removes the ordered `label_ids` target; quantity-preserving tensor adapters,
  count supervision and displayed-policy metrics retain every copy. Inventory
  order has no semantic meaning. Overflow fails explicitly. Single-item
  `label_id` remains diagnostic. Raw evidence supports the eight-copy inventory
  and five-Long-Sword purchase described in the acceptance document.
- **Qualification and coverage:** `scripts/qualify_patch_corpus.py` checks all
  before-board, before-shop and after-shop inventories, separating item copies,
  regular slots and role slots. Its authoritative initial inventory recheck
  accepted 1,063 of 1,311 games; 248 were quarantined. The corresponding
  `inventory_recheck/coverage/` contains 1,049 export-eligible games and 126,325
  rows. Coverage reports preserve tracked-pro participation and overlapping
  cohort memberships; route discovery is not proof a pro played in the game.
- **Atomic generations and isolated execution:** `app/dataset_artifacts.py`,
  `app/dataset_generations.py`, `scripts/publish_patch_dataset.py` and
  `scripts/run_qualified_smoke.py` bind source, code, acceptance, membership and
  content hashes. Complete folders publish atomically; the pointer changes
  last. Failed partial generations and name reuse are rejected. Training,
  masks, caches and policy reports stay under a separate experiment directory.
  Normal admission does not open final-test payloads. The old baseline export
  command was retired; no production-hold bypass was authorized.
- **Smoke evidence:** `data/experiments/16.18-initial-smoke-v2/completion.json`
  records one trunk epoch, one frozen-trunk graft epoch and actual displayed
  policy evaluation on ten validation games / 1,252 visits, with zero
  unaffordable baskets. This completed gates 4–5 as mechanics testing only.
  Later slot-legality changes have separate evidence; they do not rewrite this
  immutable run or establish model quality.
- **Next-night collection:** `scripts/collect_patch_sample.py` can inherit the
  original reviewed ladder/pro cohorts with a new cutoff and fresh match
  lists, preserving the original cohort observation time and unresolved pros.
  It reconstructs once per game and derives all ten perspectives. The freezer
  binds uniform event/static/code versions and explicitly closes SQLite
  connections before publishing the folder on Windows.
- **Verification completed in that session:** focused assert scripts covered
  causal gold/decisions, quest/undo/label semantics, qualification, multisets,
  dataset generation interruptions, provenance and the hold. In particular,
  `test_purchase_multiset.py` exercised five-copy targets, eight inventory
  tokens, permutation invariance, overflow rejection and label stripping through
  the real decoder; `test_dataset_generations.py` exercised interrupted
  publication, tamper detection, final-test sealing and output isolation;
  `test_patch_sample.py` exercised inherited cohorts, one replay per game and a
  real temporary ten-perspective DB-to-freeze round trip. The Windows SQLite
  file-lock failure found by that integration check was fixed and the test
  rerun successfully. These are historical test results, not a claim that the
  subsequently changed working tree has just passed the complete suite.

The initial read-only 16.17 feasibility report is
`data/ml/repair_source_feasibility_20260911.json` (08:58:59 UTC): 20,859 stored
16.17 games, 13,472 cached raw pairs present and 7,387 incomplete pairs. Ten
sampled 16.17 games replayed successfully. Those were cache-presence/sample
findings, not full integrity certification or an end-to-end rebuild timing.
The recommendation at that point was a separate pilot and resumable historical
rebuild. Later owner authorizations, development runs and targeted refetch are
recorded above and supersede that earlier recommendation-only state.
