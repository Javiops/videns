# Research progress — 2026-09-29

Goal: achieve a demonstrably useful breakthrough in win-probability-based
decision evaluation for videns.gg. Goal remains active; no breakthrough or
better decision policy has been established.

The owner has stopped other workloads and authorized RAM/GPU use. They want
updates about substantive findings, not narration of routine implementation
failures. Keep frozen sibling code, the report index, and sealed data intact.

## Completed evidence

- Versioned admission v2 unites 71 build_tracker split records with an exported
  macro_coach reservation artifact: 3,118 globally sealed IDs denied, 1,911 own
  confirmatory IDs still reserved. Existing historical validation never fits.
- Clean W1 v3: 6,000 games, 138,831 states; fit 4,002, calibration 1,000,
  development 998. Exact-state features; no future interpolation.
- Snapshot log loss 0.499117 versus logistic 0.509568: paired improvement
  0.010451 [0.005579,0.015539], 2.05% relative. Trajectory log loss 0.497963:
  gain over snapshot 0.001154 [0.000064,0.002277], 0.23% relative, below the
  0.5% practical gate. Shuffled-label control is clear. No confirmation yet.
- Source: `data/runs/w1-pilot-20260929-v3/experiment/report.json`; independent
  provenance/count/partition audit: that run's `integrity_report.json`.
- Short-horizon experiment: predict >=1,500 gold-difference loss OR >=3
  kill-difference loss over complete three-minute intervals; retrieve two
  intervals per game. Development: 998 games, 721 with a positive window,
  6,178 windows. History risk macro recall 35.48%, logistic risk 34.08%,
  worst preceding gold slope 31.77%. History does not clearly beat logistic.
- Independent post-result time baseline: latest two eligible windows have
  44.16% recall, materially better than history. The positive comparison to
  gold slope alone is inadequate. This is retrospective window selection;
  it is not an online alert policy or demonstrated coaching benefit.
- Source: `data/runs/setback-20260929-v1/experiment/report.json` and independent
  `docs/setback_result_v1.md`. Do not change the endpoint to seek a positive.

## Rich-state and feature-family results

Rich run `data/runs/rich-20260929-v1/experiment/report.json` completed on the
same population. Rich trajectory reduces log loss by 0.003621
[0.000176,0.007357], 0.727% relative, versus the original trajectory; Brier
gain 0.001194 [-0.000103,0.002553] is unresolved, so the full declared gate
fails. The setback branch still loses to the latest-two control. Independent
audit and full interpretation are in `docs/rich_result_v1.md`.

The fixed feature ablation
`data/runs/rich-ablation-20260929-v1/experiment/report.json` finds:

- Absolute-gold-only log-loss gain: 0.000149 [-0.000588,0.000901], unresolved.
- Allocation (32 features): 0.003403 [0.000059,0.006890], but Brier unresolved.
- Geometry (48 features): -0.001035 [-0.002378,0.000383], unresolved.
- Combined versus allocation: 0.000218 [-0.001047,0.001525], unresolved.

The evidence points to allocation as the promising block; it does not
establish extra geometry value or a practical breakthrough. Do not chase
small forecast gains indefinitely instead of evaluating decisions.

## In progress: actual purchase decisions

Counts-only feasibility was implemented for the first defensive boot
choice (Steelcaps 3047 versus Mercury's Treads 3111), using only the exact
clean 6,000-game source bindings. Candidate focal player is selected by a
deterministic hash among pregame fighter/tank champion tags. All tier-2 boot
choices stay visible in the support census; outcomes must not select a group.

Anchor context before the entire shop visit, not the final upgrade event:
components purchased in that visit reveal the treatment. Latest frame must
strictly precede the visit's first event. A conservative wallet lower bound
subtracts known full-price purchases between the frame and visit; never add
the chosen purchase's cost to observed leftover gold. Unknown costs/undo
ambiguity fail affordability confidence. Full-price dual affordability can
be tested conservatively; slot/whole-basket legality is not thereby proven.

`statistical_review` owns `scripts/boots_support.py`, its protocol and tests.
`data_admission` owns a static-only export of the three DDragon versions into
`artifacts/boots-static/20260930-v1/boots_static.json` plus sidecar. No current
run or existing core module may be changed. Root must review then launch the
counts pass durably; no actual action-value result has been measured yet.

The draft `docs/confirmation_design_v1.md` has not been activated. The full
rich WP gate failed, so no own confirmation or original test is opened.

## Mathematical decision-learning continuation (2026-09-30)

The user explicitly redirected emphasis toward mathematical strategies and
papers, rather than predictor tuning or reporting routine implementation
failures. `docs/mathematical_strategy_v1.md` and `winprob/decision_math.py`
implement the algebraic kernels for a reference-policy exponential tilt,
return-conserving WP shaping, IQL expectile weighting and IPO/DPO losses.
Eight synthetic mathematical properties pass. This is not a trained IQL or
IPO policy and does not establish an improvement in decisions.

The boots support census is complete at
`data/runs/boots-support-20260930-v1/experiment/report.json`: among 6,000 games,
3,034 first defensive choices were observed, but only 19 meet the strict
full-price/unmixed/known-inventory support rule; only two are development.
The proven ordinary-Boots discount sensitivity has 70 overall and 12
development cases. Do not fit an underpowered terminal-effect model or call
the unfiltered 3,034 observations interchangeable decision opportunities.

The revision-pair census is complete at
`data/runs/undo-pairs-20260930-v1/experiment/report.json`: 1,295 strict different-
item revisions in 1,156 games. Fit: 866 pairs / 777 games; calibration:
200 / 178; own development: 229 / 201. There are 233 same-item roundtrip
controls. Original certificates are conditional on B-time feasibility;
context comes from before A, and no win targets were extracted.

Crucially, own-WP development is not automatically unseen by M2. M2 fitted all
6,354 v4 source training games, including much of the 16.18 own calibration/
development data (and six historical macro exposures that remain excluded
from every new operation). Primary preference evaluation must use 16.19:
the pilot has 56 eligible pairs in 52 games out of 303 permitted games.
Expansion to all 859 already admitted 16.19 development games is being built
without relaxing the certificate or opening the 1,911 confirmation IDs.

`docs/ipo_experiment_design_v1.md` specifies an at-most-768-parameter bilinear
residual. IPO is weighted ridge regression after normalizers cancel. Exact
M2 group masses and canonical pre-A input rows still require a source-owned
export; do not invent M2 probabilities or substitute a weak classifier.
M2 groups are whole retained purchase multisets minus control wards, not
single main items. The first adapter must prove zero-residual native parity.

The outcome-free context cache is complete at
`data/runs/revision-contexts-20260930-v1/cache`: 5,861 games, 135,667 frames,
fit4,002/calibration1,000 and all859 admitted 16.19 development games. It reuses
5,305 overlapping contexts exactly and opens only556 new admitted raw games.
There is no y array. Reuse parity is not independent raw re-extraction.

The source-owned trajectory smoke is complete at build_tracker's
`data/exports/winprob-trajectories/trajectory-smoke-20260930-v1`: ten fit games,
100 player-games,93 complete episodes. The independent consumer compiles these
to993 transitions and93 terminals;47 winning returns are paid exactly once,
on the transition into the terminal. Seven gapped player-games are excluded
whole. Postmatch roles remain only in legacy M2 compatibility metadata.
Evidence: `data/reports/trajectory_consumer_smoke_20260930.json`. No IQL fit.

The source alignment smoke verifies ten fit16.18 revision contexts against
qualified replay, preinventory, strict-prefix wallet and label/future mutation
checks. Exact M2 scoring then reproduces all9,333 group masses unchanged under
those mutations. Artifacts are build_tracker's `data/exports/` releases
`winprob-revision-contexts/alignment-smoke-20260930-v1` and
`winprob-m2-pairs/m2-pair-smoke-20260930-v1`. These are adapter evidence on fit
data, not held-out performance. Full vectors/native zero-residual decoder
identity and cross-patch mapping still precede real IPO fitting.

Current work owners: statistical_review implements the expanded census using
the frozen original certificate and small IPO/winner-only training kernels;
data_admission generalizes indexed pre-A context export to17/18/19 and exact
M2 reference export; direction_audit independently reviews the mathematical
direction. Root owns durable launchers and real experiment runs. Ordinary
qualified rows cannot silently replace earlier pre-A undo contexts.

The expanded census completed in235.36s at
`data/runs/undo-pairs-20260930-v2/experiment/report.json`, reusing the exact
original receipt-certificate code. Fit remains866pairs/777games and calibration
200/178. The complete859-game16.19 development cohort yields158 strict
revisions in144games:55 cheaper B,78 dearer B,25 equal price. The minimum
population gate passes before qualification/catalogue exclusions. No outcome
targets or preference predictions were extracted in this census.

The small IPO implementation is `winprob/ipo_residual.py`; its frozen feature
map is `docs/ipo_feature_map_v1.md`:22 POV context coordinates and23 static
action coordinates,506 parameters. Comparison is exact weighted ridge IPO
versus full-catalogue winner-only CE, both relative to M2. An independent
consumer checks source artifact identities and support; paired evaluation
resamples whole games. Zero residual must preserve original float32 reference
bytes and compose with the source-native decoder parity check before fitting.
These implementations alone are not measured preference benefit.

Full native pre-A alignment completed at
`data/runs/revision-alignment-20260930-v1/alignment_summary.json`:17 has764/767
aligned pairs,18 has299/299,19 has158/158. Total1,221 contexts; three17 pairs
remain explicitly excluded. Source exports use new
`build_tracker/scripts/export_winprob_revision_contexts_v3.py` and preserve
all previously bound v1/v2 producer bytes.

As of2026-09-30 00:12UTC, exact full-catalogue scoring is running under
`Videns Data revision-score-full`. Output will be source-owned
`build_tracker/data/exports/winprob-m2-pairs/m2-preference-20260930-v1`.
It checks all-row label invariance and original native options against the
exported-P decoder path. CPU throughput is about one context/second; the
25-minute launcher and30-minute scheduled-task bounds remain active.

`Videns Data ipo-fit` is queued at `data/runs/ipo-revisions-20260930-v1` and
waits for the source scorer before calling the independently validated loader.
The final code is frozen pending this run. It requires sufficient mapped
support, original-float32 zero-residual byte identity, and convergence of the
full-catalogue winner-only control before evaluating the fixed16.19 challenge.
It reports price/frequency controls, game-bootstrap intervals, design rank,
feature collisions and policy drift.19 synthetic IPO/CLI tests pass, plus
7 consumer and7 paired-evaluation tests. Training results are not yet known.

## Practical integration now verified

M1 curves are exported at `artifacts/macro-curves/20260930-v1/` and copied as
versioned artifacts into `macro_coach/data/imports/winprob_curves_v1/`.
All34 permitted macro train/validation games have curves:810 points and708
observed three-minute windows. Judge exposure:23 unseen,7 calibration,4
development,0 fit. The frozen original snapshot model is used, not a rich
model that failed its complete gate. No terminal outcomes are exported.

The macro review loader and UI were integrated with guards before per-game
corpus reads. An isolated headless-browser test passed: actual plot rendered,
observed-decline navigation changed the selected decision, preceding-point
marker and blue-team toggle worked, a nonexistent unadmitted ID returned404,
and no browser errors or feedback writes occurred. Evidence:
`data/reports/macro_review_browser.json` and `.png`. The local service is
http://127.0.0.1:8011; its owned scheduled task is `MacroCoach Winprob Review`.
This verifies functionality, not human coaching benefit or the broader goal.

## Rich feature implementation

Rich-state hypothesis: where resources are located may matter more than the
total lead. `winprob/features_rich.py` adds per-team gold allocation, dispersion,
distance/proximity and resource-weighted position summaries. Gold is not
combat strength; coordinates do not establish alive status or player vision.
Offline OMNI only. Original modules stay unchanged for model provenance.

The completed matched comparison reopens only the same 6,000 digest-bound inputs,
preserves every original row/label and split, then tests fixed rich WP models
against frozen W1 trajectory and rich setback risk against logistic, history,
and latest-time controls. No final reservation is opened during exploration.

## Operational paths

- Python: `.venv/Scripts/python.exe` (numpy/scipy/sklearn/joblib; own environment).
- Tests: `python -m unittest discover -s tests`.
- Long work: one-shot scheduled tasks; never attach multi-minute work to an
  agent shell. `scripts/run_rich.ps1` is prepared for the next experiment.
- The original31 protected app/index files retain their recorded digests.
  New source-owned producers and the explicit macro review integration are
  described above; original source model/training code remains unchanged.
- W1 v2's cross-project reservation omission is documented separately and its
  result quarantined without metric inspection. No need to repeat procedural
  details in user updates; do not suppress the record or use that result.

## IPO result and next outcome-directed experiment

The full reference export completed in1,193.04s:all1,221 aligned rows scored
across9,333 groups, with1,221 native zero-residual and label-mutation checks
and zero mismatches. Allrows retained positive mapped pair support:
fit863pairs/775games, calibration200/178, development158/144.

`data/runs/ipo-revisions-20260930-v1/experiment/report.json` completed in13.02s.
IPO weighted preference accuracy is53.5256%, nativeM2 is54.4872%, winner-only
imitation53.5256%. IPO minusM2 is-0.9615percentage points, paired95% interval
[-3.9396,+1.9231]. IPO andwinner-only make identical classifications on all158
pairs despite different coefficients/logodds; both change sixM2 predictions
(two improvements,four regressions). The declared usefulness gate fails.
Price-dearer is57.2115%, cheaper42.7885%, retained-frequency53.3654%; these are
diagnostics, not a replacement selected after results. Both optimizers passed
stationarity/convergence checks. Independent provenance/prediction/bootstrap
audit: `docs/ipo_result_v1.md`. Close this fixed recipe; do not promote it or
retune repeatedly on the same challenge. No confirmation set was opened.

The next direction is full-action IQL versus matched uniform behavioral
cloning, with nativeM2 kept as a strong comparison. The source package contains
13,166 full actions mapped to9,333 main groups. A metadata-only versioned
catalogue is published in build_tracker at
`data/exports/winprob-action-catalogue/full-action-catalogue-20260930-v1`.
SAVE,oneward,twowards remain distinct full actions. `docs/iql_action_contract_v1.md`
explains the coherent full-action reference lift and why main-marginal identity
does not imply full-action argmax/native-display identity.

This project's own environment now has torch2.13.0+cu130. A synthetic CUDA
autograd check passed onRTX4060 with17MB peak allocated GPU memory:
`data/reports/iql_runtime_20260930.json`. No source environment is imported.
`winprob/iql_kernels.py` defines observed-action expectile/Bellman/AWR kernels;
`winprob/joint_reference.py` independently checks the mathematical full-action
lift. The actor compiler `winprob/iql_features.py` uses immutable champion IDs,
count-preserving inventories and their public static attributes, own wallet,
public levels/counters/objectives and item-value allocation. It ignores roles,
totalGold, opponent wallets, targets, outcomes and quest-derived flags.

The independent real FIT smoke validates10games,82complete catalogue-supported
player chains and986transitions, plus all13,166 action descriptors:
`data/reports/iql_feature_smoke_20260930.json`. Six feature-boundary/invariance
tests,three catalogue-identity tests andfour lift properties also pass.

Full preparation now runs as `Videns Data iql-trajectories`, owned run directory
`data/runs/iql-trajectories-20260930-v1`. New source-owned producer
`build_tracker/scripts/export_winprob_fit_trajectories_v2.py` processes the exact
original4,002FIT identities in separate17/18 processes. It streams gzipped
complete sequences; any gap or out-of-catalogue action excludes the whole
player-game. No calibration/development outcomes are opened. This conditions
the training population on source/catalogue coverage, which must be reported.
The40-minute launcher/45-minute task is bounded and resource-gated. No IQL
policy has yet been fitted or evaluated.

The subsequent fixed recipe is now recorded in
`docs/iql_experiment_protocol_v1.md`: two whole-game critic folds, .7 expectile,
gamma1, temperature.1, raw advantage cap20, one global FIT-mean weight rescale,
and matched actor continuation from a common BC warm start. Full-action support,
training size and opposite-fold action coverage have explicit pre-fit gates.
`winprob/iql_models.py` implements flexible public-board Torch encoders and
independent twin critics/value/actor; it is new, separate from the frozen IPO
code. `winprob/iql_features.py` supplies the exact37 global/22 inventory inputs.
Dataset-cache construction and the complete training/evaluation runner remain
to be implemented and verified after source preparation. Do not mistake a
passing kernel or CUDA check for an actual trained IQL policy.

## Resumed after account-limit interruption: complete IQL cache

The source trajectory exports finished successfully while model tool access
was interrupted.16.17:3,518eligible games,28,821chains,340,492transitions;
16.18:483games,4,391chains,51,956transitions. One originally requested game
is excluded by native qualification. The current user explicitly resumed work.
The first attempted cache-writer patch had not executed; it was subsequently
created and verified after tool access returned.

`data/runs/iql-cache-20260930-v1/cache/manifest.json` is complete and authorized
for the fixed training recipe:392,448states,33,212episodes,4,001FIT games,
zero feature-encoding exclusions,456.82seconds. Whole-chain survival is
33,212/36,029=92.1813% of independently validated source-complete chains.
The two folds contain2,000/2,001games and195,611/196,837transitions, with
98.9172%/98.8498% opposite-fold observed-action support and both terminal classes.
The tensor cache uses native-patch observed-action attributes and public
state inputs; next-state references remain inside complete player-games.

An independent audit matched ALL392,448transitions to source artifacts and
verified29cache files,142bindings,25arrays and39,498static action vectors.
Feature/ragged-bag parity was exact on32episodes/379states; ignored-field
mutation checks passed. Evidence:
`artifacts/audits/iql-cache-20260930-v2/audit.json` and
`docs/iql_cache_audit_20260930_v1.md`. No raw or sealed sources were reopened.

New trainer `scripts/train_iql.py` and helpers `winprob/iql_training.py` passed
seven CPU helper tests and a real full-cache loader check. Independent review
confirmed fold isolation, exact warm actor/Adam cloning, matched continuation
orders and full-group evaluation. A source-bound synthetic CUDA preflight
`data/runs/iql-preflight-20260930-v1` stopped BEFORE real model fitting when
available host RAM fell below the0.75GiB live floor. Its task/process are
terminal; no candidate or result exists. The next operational revision releases
host feature mappings once their exact GPU copies are ready, preserves the
small focal-champion vector needed by diagnostics, and retains the RAM floor
and the statistical recipe. Retry must use a new preflight directory.

Training launchers are `scripts/register_iql_job.ps1` and `run_iql_job.ps1`:
separate preflight/train modes, initial2.5GiB host gate, game/sibling-job
watchdog,90-minute worker training limit and owned-process termination only.
Full training remains pending successful resource preflight. No IQL policy
improvement, win benefit, serving change or goal completion is established.

## Real IQL training launched (2026-09-30 09:07 UTC)

The memory revision closes the seven large host feature mappings after
synchronized exact CUDA preload; the small owned focal-champion vector remains
for frequency diagnostics. Ten helper tests and two independent artifact/
diagnostic-preservation tests pass. Preflightv2 passed memory (minimum available
host1.82GB) but its76.49-minute conservative projection missed the additional
85%-of-remaining-time guard by about12seconds. Neither the guard nor the recipe
was relaxed, and no real training occurred in either failed preflight.

A subsequent exact-execution revision skips repeated GPU-to-CPU input-content
checks only on the fully validated immutable cache. Structure/dtype checks,
forward arithmetic, output/loss finiteness and gradient guards remain; strict
content checks are the default and are restored for the challenge. Six
independent CPU tests prove exact full13166-class actor forward/loss/gradient/
Adam parity and mixed-terminal Q/V/Polyak/Adam parity. A separate root check
also reproduced exact numeric parity. No architecture, data, objective, epoch,
seed, sampling or learning-rate change was made.

`data/runs/iql-preflight-20260930-v3/experiment/preflight.json` completed:
CUDA states resident; actorstep mean0.008716s, criticstep0.039827s; conservative
full-recipe projection2,850.39s (47.51min), including1.5factor and300s allowance;
peakCUDAallocated887,395,328bytes, minimumhostavailable1,937,100,800bytes.

The real job `Videns IQL train` is running under
`data/runs/iql-outcome-20260930-v1/experiment`. It started09:07UTC and was
verified live with sourceworkerPIDs29008/30900; at427s it had completed13of20
epochs for the critic that will score fold0. Status/epoch files are progress
only, not policy benefit. All bound implementation/data/protocol files are now
frozen until the run is terminal. A code snapshot of15bound own-project code/
protocol files plus the runtime requirements is saved in its`code_snapshot/`.
Continue by checking the actual scheduled task/process and progress; never
restart solely because an observation call timed out. Evaluation and the
broader breakthrough goal remain pending.

## IQL run completed: candidate fails, BC control merits a careful audit

The job completed at2026-09-30 09:37UTC in1,784.00s (29.73min). Authoritative
report: `data/runs/iql-outcome-20260930-v1/experiment/report.json`. All fixed
epochs completed; no serving or confirmation action occurred.

On the same158revision pairs/144development games: IQL53.2051%,M254.4872%,
BC20 59.6154%,warmBC10 52.5641%. IQL minusBC20 is-6.4103pp with paired95%
interval[-12.1795,-1.2577]; IQL minusM2 is-1.2821pp[-8.0784,+5.5241]. The
candidate fails the predeclared useful cross-signal gate. Do not call this
a successful outcome-learning policy or dismiss all IQL from one recipe.

An explicitly exploratory contrast of the already-declared BC control is
saved in `exploratory_bc_control_contrasts.json` beside the report. BC20 minus
M2 is+5.1282pp[-0.6330,+10.5599], unresolved; versuswarm10 it is+7.0513pp
[+1.8987,+12.4203], and versusprice-dearer+2.4038pp[-7.1192,+12.7499]. This
does not turn BC into the original primary candidate or justify confirmation/
promotion on a retrospectively selected favourable point estimate.

OOF advantage weights retain98.8834% observed-action support, but effective
weight count is70,854.84 of392,448transitions (18.05%);2.1794% hit the raw cap.
This is a diagnostic of concentration, not proof of why preference ranking
worsened. Independent final provenance/checkpoint/OOF/metric audit is assigned
to statistical_review. Keep the fullrun code/data/checkpoints frozen. The
broader useful-breakthrough objective remains active and unachieved.

## Frozen BC20 ordinary-decision trial, 2026-09-30 continuation

The independent IQL result audit completed successfully; see
`docs/iql_result_v1.md` and `data/reports/iql_result_v1_audit.json`. The negative
IQL result and unresolved exploratory BC comparison stand. A separate
complementarity diagnostic found only20 M2/BC disagreements among158 revisions
(14 BC-only correct,6 M2-only correct); no mixture was fitted or selected.

The next inference-only trial was fixed before ordinary purchase labels or
scores: `docs/bc20_ordinary_eval_protocol_v1.md`. Primary is frozen BC20
reranking the unchanged native M2 up-to-three options. A native incremental
spend rule is the cheap control; fixed half-geometric pooling and directly
decoded BC are secondary. Preserve original main-basket ward-only group0
credit; also report typed and complete baskets. Success requires at least700
games,95% prediction coverage,+3pp primary gain with positive paired95% lower
bound versus M2, a positive paired lower bound versus spend, unchanged
candidates and complete mutation/parity/economic receipts. This is exposed
development purchase agreement, not fresh confirmation or win benefit.

Metadata selection chose one focal player and one native admitted decision in
each of859 development16.19 games, with no replacement. Root independently
recomputed every selection hash. Source selection SHA256:
`611c1f7d3943074325c88659dcae59693518745bc5abcceebf52c5e67ce27f85`.
The complete source-owned context export reconstructed859/859 in284.671s:
`build_tracker/data/exports/winprob-ordinary/ordinary-contexts-20260930-v1`.
Targets are separate from all prediction inputs. Source replay internally
deserializes final match fields; no win target is exported or analyzed.

Frozen BC20 export completed859/859, no feature fallbacks or mutation failures,
in18.066s at `data/runs/ordinary-actor-20260930-v2/export`. A numerical v1
failure is preserved: float32 softmax accumulation across13,166 classes
exceeded the2e-6 normalization guard. V2 uses float64 normalization and grouped
accumulation before float32 exports, retaining the same guard, checkpoint,
features and model logits. No target or accuracy informed this correction.
Independent audit `data/reports/ordinary_actor_v2_audit.json` verified135
bindings,859 rows,3436 array hashes, all full-to-group sums, and bitwise
re-inference of16 deterministic rows with the original batch shape. Maximum
group-mass error is6.20e-8. All confirmation and global test payloads stay shut.

New source-owned reference/audit v2 scripts use unchanged native M2 inference
and decoding. The first reference attempt failed before inference because
Torch exposes a nonexistent pseudo-module path during code-digest enumeration;
v2 skips nonexistent paths but binds every real local source file. V1 is
preserved. Fourteen source synthetic tests and24 own evaluation/reader tests
pass; independent reviews covered ward credit, exact candidate membership,
whole-game denominators, integer+3pp gate, unknown target support, every
upstream identity, and prediction-before-target boundaries.

The inherited W1 launcher start threshold2.5GiB unnecessarily delayed this
small inference stage. A source-runtime-only measurement used0.489GiB working
set; all M2 weight files total7.18MiB and the two complete prediction matrices
61.17MiB. Estimated full reference working set0.8–1.1GiB with1.3GiB planning
allowance is an estimate, not an empirical bound. The stage-specific launch
gate is now2.0GiB, retaining CPU1/batch16, no-game/no-competing-job guards and
the0.6GiB pressure stop. Waiting attemptv2 was cancelled before any worker or
source output existed. No other user's process was stopped. This task's own
macro review preview on8011 was temporarily stopped and must be restored
after the heavy stage.

The actual reference run is `ordinary-reference-20260930-v3`, source exporter
codev2, started11:28:46UTC. At70.93s it had96/859 scored contexts. Measured
worker working set stayed around0.7GiB with1.4–1.5GiB host RAM free. Resource
telemetry is in its own launcher directory. Do not restart a live worker.
Next: complete the reference, run native19 auditv2 in a separate process to
`ordinary-native19-20260930-v3`, then the fixed own evaluator. No ordinary
accuracy has been computed yet; the goal remains unachieved.

### Ordinary trial completed; the reranker is worse than M2

Reference inference completed859/859 in552.931s, with every full-distribution
mutation, original-P decoder replay, and both M2/direct-BC option-mutation
check passing. Native19 audit completed in8.253s on3211 exact candidate-union
objects. Every declared candidate was feasible in both specified economic
worlds under the reconstructed wallet/slot assumptions. No target was opened
by either producer. The own evaluator completed in6.110s; completed durable
stage computation totals870.031s (14.50min), excluding waiting and failed
pre-inference attempts. The preview on8011 was restored and HTTP200 verified.

Report: `data/runs/ordinary-evaluation-20260930-v1/experiment/report.json`,
SHA256`311ccbb5d3a3b09862d8e402934c11a8f162cf08f494fa8e01b12e0a495c4127`.
All859 selected games have complete predictions and observed target ledgers.
Primary main-basket accuracy: M252.8522%,BC reranker49.1269%,spend37.1362%.
BC reranker minusM2 is-3.7253pp, paired95% interval[-6.1700,-1.2806];40 games
improve and72 regress. The fixed primary gate fails. Versus spend, BC gains
11.9907pp[8.2654,15.3667], which cannot replace the failed M2 comparison.

All secondary arms remain visible. Direct decodedBC is47.8463%, versusM2
-5.0058pp[-7.5669,-2.5582]. Fixed half-geometric pooling is53.7835%,+0.9313pp
[-0.8178,2.6775], unresolved and below the practical threshold. Do not tune a
mixing coefficient or switch primary endpoints on these859 exposed games.
Main-candidate top3 recall is68.8009% for M2 and64.6100% for direct BC.
Unfiltered catalogue BC top1/top3 is48.8941%/73.4575%; those are not legal
displayed recommendations. The raw-to-decoded recall loss merits a separate
post-result action-semantics diagnosis; it is not a successful policy result.

Target mix:678 other purchases,128 mixed purchases with wards,38 ward-only,
15 explicit observed no-buy. Four full baskets and two main baskets are
outside the frozen catalogue and remain misses. The15-game no-buy slice is
not estimated separately under the predeclared30-game rule. Purchase-only
results still favour M2 over the primary reranker. Main, typed-main and
complete-basket diagnostics retain their different ward meanings.

An independent result audit is in progress with a separate NumPy/stdlib
recomputation. The final evaluators and all completed producer code/artifacts
are now frozen. The report index SHA256 remains
`94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46` across the
continuation's checks. No confirmation or sealed-test payload was opened and
no serving policy was changed. The broad breakthrough objective is unachieved.

Before observing this result, the research review selected a different next
hypothesis: does the previous two completed own-shop visits add information
for predicting the immediate main purchase beyond current public state?
Proposed controls are fresh state-only BC and a same-capacity history model
whose prior-basket contents are shuffled from different training-fold games
while recipient timing/masks remain unchanged. Use existing game folds, no
all-FIT BC20 encoder, one fixed focal/hash visit per game, and paired whole-game
uncertainty. A metadata-only census finds3350 available focal episodes among
4001FIT games, preserving651 absent focal episodes without replacement.
The target is current action; future completed-item goals are optional later
diagnostics, never inference features. Strict prior-visit closure must be
proved from existing source-export visit_end_ms sidecars; timestamp ties mask
the corresponding history slot without choosing an older replacement.
No new history model, goal label extraction or recipe is fitted/frozen yet.

### Independent-audit access pending; action-feasibility diagnosis

The independent audit implementation is ready and syntax-checked at
`data/reports/ordinary_evaluation_audit_v1.py`. Automatic approval review
rejected its execution twice before process creation, requiring explicit
user authorization for reading the existing859-game development purchase
target artifact. A transparent retry supplied verified source/report scope
(development only; no raw, sealed, confirmation or win-outcome data) and was
also rejected. No independent final auditJSON was produced. Do not execute
the same read through another agent/path or remove its explicit target-read
flag as a workaround. Root asked the user through the input tool; authorization
is pending. This is an audit limitation, not a failed model run or a claim
that the reported metric is already independently verified.

Before that access block was relayed, source-side artifact overlap analysis
found631 catalogue-top3 hits and555 directly decoded-top3 hits:523 shared,
108 catalogue-only lost,32 direct-only gained,196 neither. All108 lost targets
have unique probability ranks1/2/3 (51/33/24), excluding rounding/ties and
shortlist truncation as explanations. None appears in M2's displayed set.
Code therefore implies a native18 feasibility rejection; direct economic
receipts have not yet been obtained. Lost-visit median duration is200.5ms,
90th percentile2.23s, maximum11.279s;71 are single-main-item targets,37 are
multi-item, and six have an opening sale. These facts weaken the initial
long-visit hypothesis. They do not establish actual wallet error, structural
failure, a patch defect, or a better policy.

Further target-dependent diagnostics are on hold pending the same user
authorization. A new source-owned, synthetic-tested diagnostic can be prepared
without accessing those files: compare original-wallet feasibility with a
fixed diagnostic-only generous wallet, bisect minimum feasible wallet where
applicable, and repeat the same fixed cases in separate native18/native19
processes. It must never alter candidate predictions, rescore a modified
primary policy, claim measured actual gold, or change the failed gate.

### Explicit authorization received; independent result audit passed

The user explicitly authorized reading the exported purchases from these859
development games for the independent audit and legality diagnosis. The
previous access block is resolved for that scope. The independent audit ran
the original transparent command and completed in20.299s:
`data/reports/ordinary_evaluation_audit_v1.json` and its SHA256 sidecar.
It recomputed every game, primary and fixed-secondary rank/credit, main/typed/
complete metric, paired2000-game bootstrap, slice, candidate-union/cost receipt,
catalogue map, reservation intersection and bound input/output identity.
The negative primary result and unresolved fixed mixture are confirmed.
No raw archive, sealed-test, confirmation or win-outcome payload was opened.

Prepared source diagnostic `diagnose_bc20_ordinary_feasibility_v1.py` has10
passing synthetic tests. It compares the same fixed108 cases in separate
native18/native19 processes; changed supplied wallets are diagnostic only.
The native18 run is `ordinary-feasibility-16.18-20260930-v2`. The first waiting
attempt was cancelled before inference to replace the inherited2GiB gate
with the native-only1.25GiB gate; the existing0.6GiB pressure stop is checked
each second, CPU1 and120-second computation limits remain. This loads no
policy model and changes neither predictions nor the failed success gate.

### Wallet-sensitivity diagnosis complete in both economic worlds

Native18 diagnostic completed43.561s; native19 completed9.230s. Each made2589
native economic calls on the same108 post-result cases. Every observed action
was infeasible at the original supplied wallet and feasible when only that
wallet was increased. All108 minimum integer wallet thresholds and216
generous/minimum purchase sequences are identical case-by-case across the two
patches. There is no recovered original-wallet feasibility from switching to
the native19 static world in this subset.

Required additional supplied gold has min1,p25=20.75,median69.5,p75=148.25,
p90=240.6,max1658. These are mathematical thresholds of the reconstructed
economic model, not measurements of historical wallet error. The source
engine's feasibility is monotone in the supplied wallet: its budget branch
checks cost against budget, other inventory/role constraints are unchanged,
and the purchase-sequence search explores all retained-item orders.
The diagnostic uses integer bisection and records an infeasible receipt at
the immediately preceding integer. The generous wallet is never a suggested
serving input or an adjusted evaluation policy.

Artifacts:
- `build_tracker/data/exports/winprob-ordinary/ordinary-feasibility-16.18-20260930-v2/manifest.json`,
  SHA256`a5ce6e599719d0f99c72d668df0783b0c21aaa973ad574f08390a45d6e91279d`.
- `build_tracker/data/exports/winprob-ordinary/ordinary-feasibility-16.19-20260930-v1/manifest.json`,
  SHA256`9cfeb3b56f7f9a9440377859efc97d8a00bc18fd0a6a36f99fb413157787457f`.
- `data/reports/ordinary_wallet_diagnostic_audit_v1.json`, independently
  verified188 bindings,432 sequence receipts and216 source/context joins;
  SHA256`877564c8a519da4147a5edd380ec89d6af3df8a02991e35bb63d815f8430de47`.
  Its reproducible own-project script is
  `data/reports/audit_ordinary_wallet_receipts_v1.py`; no source-code import
  or duplicate native economic search was used by that audit.

The diagnosis identifies supplied-wallet/context incompatibility on these108
selected losses. It does not isolate gold estimation from inventory discounts,
visit timing, sales/undo or retained-target semantics.51/108 visits have zero
exported duration,71 have a single main-item target; long waiting during the
visit is therefore not a sufficient general explanation. The failed primary
policy comparison remains failed, and the108 cases cannot estimate wallet
error rates across all ordinary decisions.

Source-code review establishes a different live observation regime:
`build_tracker/app/live.py` reads `activePlayer.currentGold`, sets both gold
and gold_est to that value and marks `gold_exact=True`, `live-exact-v2`.
The native scorer and decoder use that same supplied wallet. This is a code
finding, not a test of live operation. Do not inflate a live budget by69.5gold,
derive a posterior from label-selected shortfalls, or transfer frame-boundary
error coverage to sub-minute shopping stops without evidence.

Next algorithmic candidate remains the fresh cross-fitted history-information
probe. Its primary unfiltered main-basket log loss/accuracy must be separated
from this reconstructed-budget legality issue. In the ideal Bayes setting,
the best log-loss improvement from adding history is conditional mutual
information I(A;H|O). Actual finite-model improvement is not an estimate of
private intent or causal policy value. Compare state-only, true past baskets,
and timing-preserving shuffled baskets with identical model capacity and
training; a positive result still needs an aligned known-wallet evaluation
before a better-recommendation claim. No history model has been fitted yet.

Both authorized audits and both diagnoses are complete. No further permission
is pending for those completed reads. The local preview remains available;
no source serving policy, sealed test, confirmation payload or report index
was changed. The broader useful-breakthrough goal remains unachieved.

## History-information experiment implemented and launched

Automatic goal continuation resumed with an active goal and available account
usage. The previous usage-state explanation did not advance the research;
this continuation implements the proposed next experiment rather than leaving
it as another unexecuted plan. The complete recipe is frozen in
`docs/history_probe_protocol_v1.md`, SHA256
`364f718890d63b754b347c4b3c126e56dcfd61d5776eff99553b3d5466c787d2`.

New own-project modules implement the causal-history adapter, same-capacity
state/history/shuffled models, action-only trainer, fixed frequency baseline
and paired OOF evaluator. All old model/source/benchmark code is unchanged.
Thirty-five focused synthetic tests pass. The history head uses the fresh
public encoder plus two prior purchase slots: state128+2*(item-sum32+native
static23+closure-age1+present1), then242→128→9333 main classes. Three fresh
models per fold share initial weights and batch orders, with20epochs each,
Adam3e-4,batch256 and no outcome/reward weighting. Total fixed workload is
92040 updates. No all-FIT BC20 checkpoint is reused.

Pre-fit review found a real future-information pathway: source synthetic SAVE
markers are stamped15seconds after death but qualified with later no-purchase
and endgame windows. The frozen new recipe excludes synthetic SAVE nodes
BEFORE selecting previous-purchase adjacency. Ward-only purchases remain.
Each of the two previous real purchases must close strictly before the current
decision; ties are masked without an older replacement. Current SAVE targets
retain their original future-window interpretation and separate reporting.
Synthetic insertion/removal of SAVE nodes leaves real purchase memory unchanged.

`data/runs/history-cache-20260930-v1/cache` completed preparation in102.829s
(terminal status102.966s), with392448 states,33212 episodes and4001 FIT games.
The metadata-first selection fixes3350 focal-game decisions and651 unavailable
focal episodes, never replaced. The matched shuffled control lacks a donor on
only2/1673 and7/1677 held-out decisions; those keep recipient timing/masks and
zero content.6480 synthetic SAVE nodes were removed before history adjacency;
there are no closure ties or overlapping purchase slots in this cache.

Independent `data/reports/history_cache_v1_audit.json` checked179 bindings,
all392448 histories/784896 slots and every784896 donor hash/choice, the complete
3350/651 ledger and sampled source closure receipts (20 episodes per patch).
SHA256`90088d225ba28e878558dd8cf87a40b0bb4ca366c34792a8afdeac7d8430d8aa`.
It opened no raw archive, reward array or model. Root separately loaded the
real cache through the trainer interface successfully; peak working set was
1282985984bytes with1343361024bytes host RAM still available at the final mark.

Primary success requires>=2% relative unfiltered main-class NLL reduction and
>=2pp top1 improvement versus state BC, positive paired95% lower bounds for
both, and a resolved NLL gain over timing-preserving shuffled content. Both
game folds remain disjoint per learner;2000 whole-game resamples use shared
draws. A positive result is internal predictive evidence, not legal-policy,
win-effect, private-intent or external-confirmation evidence. Every selected
prediction also receives a current/future-label mutation check over all9333
logits; altered held-out targets cannot touch training-fold donor contents.

First model attempt `history-probe-20260930-v1` rejected its mandatory synthetic
preflight at69.559s before ANY real fitting. Projected remaining time1843.383s
failed85% of the1730.455s remaining in its1800s operational cap. The failed
attempt, original protocol and trainer remain intact. The goal has no user
time limit, so `docs/history_probe_runtime_v2.md` explicitly extends ONLY the
model-stage time budget to2700s/45minutes. All data, model, optimizer, epoch,
control, metric and statistical gates remain fixed. The1.25 projection factor,
180s allowance,85% margin and0.75GiB live RAM floor remain. This is a disclosed
operational revision after measurement, not a claim that the old cap passed.

The new `scripts/train_history_probe_v2.py` has only runtime/addendum receipt
changes and runs mandatory preflight again. Durable task `Videns History train`
launched `data/runs/history-probe-20260930-v2/experiment` at13:22:35UTC. Its own
worker was verified live asPID13636 (venv parent5272); at31.995s it was in
synthetic_preflight. No real result exists yet. Keep ALL bound code/data/
protocol/addendum files frozen while that worker runs. Check the same live
handle before any restart; a polling timeout is not a terminal state. The
independent final audit is being prepared without reading partial checkpoints
or scores. The broader breakthrough goal remains active and unachieved.

### Runtime-v2 passed time preflight, then stopped at the RAM floor

Attemptv2 passed its new mandatory preflight:2069.189s projected remaining
within85% of2654.365s remaining. It began actual state-only fold0 training and
finished two epochs/1538 updates, but at69.008s the live guard measured only
0.652GiB host RAM available and stopped. The launcher independently recorded
the same pressure stop. No final checkpoint or selected-game evaluation score
was produced; this is an incomplete resource-stopped fit, not a result.

Inspection found no obvious retained CPU object over0.5GiB after preload:
remaining required arrays total~37MB, while old host feature-map closure had
already reclaimed~583MB of working set. The last six-second pressure change
was largely external/unattributed: available RAM fell~366MB while the worker's
own working set fell~101MB. The real worker's working set reached~1.44GB.
Do not relax the0.75GiB floor or interpret partial training as evidence.

The next attempt `history-probe-20260930-v3` reuses the exact frozenv2 trainer,
cache, protocol/runtime addendum, seeds, model, optimizers, epochs and gates.
Only the launcher startup headroom is raised from2.5GiB to3.25GiB. The own
macro preview on8011 was temporarily stopped again and must be restored after
the heavy run. No unrelated application or process was stopped. The new
durable task is registered and waiting for resources, with about2.85GiB free
at registration. Root asked the user whether they can release approximately
1GiB by closing unused applications/tabs; the job starts automatically when
its resource/no-game/no-competing-job checks pass. This is a physical RAM
condition, not an account usage limit or a new permission requirement.

The independent final auditor is prepared at
`data/reports/history_probe_v1_audit.py`; its default is being updated to the
v3 run. It must not read partial checkpoints, and it has not run. Keep all
bound code/data/protocol files frozen. Verify the queued task's actual live
handle and inspect the same run before considering any restart. The active
goal remains unachieved; the full experiment and practical validation remain.

### Attemptv3 is now training with sufficient headroom

The resource condition cleared automatically and v3 started at13:33:48UTC
using unchanged `train_history_probe_v2.py`. Its preflight passed with
2111.350s projected remaining. At elapsed220.710s it was training state-only
fold0 epoch13; epoch12 had completed9228 updates. Root verified the actual live
Python workerPID42700 (venv parent16532, launcher35940), not just a state file.
The latest launcher mark showed2451701760bytes host RAM available and
1458081792bytes own-worker working set. No result/partial checkpoint audit has
run. Continue monitoring the SAME run and handle; do not restart. The own
review preview is still paused and must be restored after the heavy run.

### Full history experiment completed and independently audited: gate fails

Attemptv3 completed all six fresh20-epoch models and92040 updates in1468.481s
(24.47minutes); the source task became terminal at13:58:40UTC. The original
fixed recipe was unchanged. All3350 selected games received OOF predictions
from all three neural arms and the opposite-fold frequency control;651 games
remain explicitly unavailable because their fixed focal episode was absent.

Main-class results on this INTERNAL FIT-derived cross-validation population:
- State: NLL2.361115873,top1=43.910448%,top3=69.462687%.
- Real purchase history: NLL2.327677072,top1=44.238806%,top3=69.582090%.
- Timing-preserving shuffled content: NLL2.352118240,top1=44.268657%,top3=69.134328%.
- Fixed champion/inventory frequency: NLL3.985319483,top1=21.522388%,top3=38.119403%.

History minus state top1 gain is0.328358pp, paired95% interval
[-0.865672,+1.522388]pp, with210 improved and199 regressed games. Relative NLL
reduction is1.416229%, interval[-0.106521,+2.897666]%. Against shuffled content,
absolute NLL reduction is0.024441168, interval[-0.010875387,+0.060468469]; top1
is lower by0.029851pp. Every declared useful-signal condition fails except
minimum population and complete coverage. Do not claim private-intent recovery,
useful basket memory, improved recommendations, or a breakthrough from these
small unresolved differences. These percentages are not comparable directly
with the earlier859-game16.19 displayed-policy percentages.

Every proof gate passed: independent current/future-label mutation invariance
over all10050 selected neural predictions/full9333logits, same initialization,
fresh Adam,40 shared per-fold epoch permutations,120 epoch records, whole-game
OOF exposure, complete row hashes and frozen input/code bindings. The final
report is `data/runs/history-probe-20260930-v3/experiment/report.json`, SHA256
`26b4b2dc4bc788dc7347cdf59133eda39bf0a8e90178b8cf668e211566d8054d`.

Independent `data/reports/history_probe_v3_audit.json` verified219 bindings,
all six checkpoints and receipts, all metrics and shared2000-game bootstrap
intervals without importing the evaluator. Manual CPU gathering/re-inference
on96 selected rows reproduced every sampled top3 exactly (zero tie exceptions),
with maximum target-log-probability error6.614683e-6; sampled future-mutation
full logits were identical. Audit SHA256
`50ebaf2d639ca5b567f9948924b0317aee68b4f4588df96a1aad328ec67de060`.
No reward, raw archive, sealed or confirmation payload was read by that audit.

The local macro preview has been restarted after the heavy job. The history
recipe is closed; do not tune its length, epoch count, seed or a post-hoc slice
until the result changes sign. The goal remains active and unachieved. A new
read-only feasibility diagnosis is checking whether the already frozen M2/BC
legal candidate union has enough oracle top3 coverage headroom for a different
fixed-size robust-shortlist strategy. That is post-result hypothesis screening,
not a new policy result, and cannot rescue either failed experiment.

### Fixed candidate-union route rejected by an exact headroom bound

`data/reports/ordinary_union_headroom_v1.json` rechecks the existing authorized
859-game ordinary artifacts without fitting or scoring a new policy. M2's
native top3 contains591 observed main targets, BC's555;540 are shared,51 are
M2-only,15 BC-only and253 in neither set. The full union's target oracle is
606/859=70.5471%, compared with591/859=68.8009% for M2. Even an oracle selecting
three items from that union can add at most15/859=1.7462pp, below the proposed
3pp useful-gain threshold (which would require26 extra correct games).

Therefore do not implement the contemplated minimax-regret fixed-size union
shortlist or score alternative mixtures against these labels. The simple
mathematical ceiling already rejects that route. This is explicitly post-hoc
hypothesis screening, not an achieved policy gain. Expanding the catalogue
alone is also insufficient on this endpoint:857/859 main targets are already
represented, leaving only0.233pp absolute vocabulary-coverage headroom.

The remaining research question is a different mechanism with credible
information or structural benefit, rather than another rearrangement of the
same candidate universe. A supervised latent completed-item goal could share
information across known rare baskets, but must be compared on the SAME
immediate-purchase endpoint with predicted, not future-oracle, goals and a
matched-capacity control. It is not yet selected, implemented or evaluated;
review existing M2 factorization first to avoid rebuilding an existing method.
The user goal remains active and unachieved, with all confirmation games closed.

### Future completion census completed and independently audited

The new census used only already admitted/versioned FIT artifacts and public
static definitions. Its v2 definition fixes106 eligible completed-item IDs
on each of16.17/16.18/16.19 and257 counted nonempty goal multisets. Two NONE
diagnostic codes merge into one EMPTY identity, giving258 model identities.
Public definitions were published before future target timing/action access.

Across all392448 states,33212 complete episodes and4001 FIT games, the fixed
strictly-later six-minute window yields178490 nonempty future goals (178285
single,205 multiple) and213958 EMPTY labels. A separate including-current
diagnostic has79251 current completions and158096 later completions; its
current completions account for33.39% of nonempty labels. The next prototype
therefore uses the strictly-later definition to avoid direct current-action
coarsening. This choice precedes every new model score.

State-weighted empirical H(G|current main basket,patch) is1.427990729 nats
(2.060155142 bits) for strictly-later goals. This establishes descriptive
within-action variation, not predictability, conditional mutual information,
private intention, or practical gain. No win/outcome label was analyzed.

The census completed in17.1215s. Independent chronological reconstruction
matched both target modes and qualifying visits for every state, all counts,
EMPTY merging, entropy, coarsening and cross-game pair calculations. There
were zero reserved-game intersections. Census report:
`data/runs/goal-census-20260930-v1/census/report.json`, SHA256
`367293d55cdeae9aba107055be2eca8af0b4bd03c95de75fcc19dd424624d566`.
Independent evidence: `data/reports/goal_census_audit_v1.json`.

Root selected one bounded next hypothesis: auxiliary future-goal supervision
into the representation actually used for immediate purchases. Draft
`docs/future_goal_protocol_v1.md` defines matched main-only, real-goal and
shuffled-goal models, with no future label at inference and the SAME immediate
main-basket endpoint. M2 already has compositional baskets; composition alone
is not new. Lambda1, strictly-later six-minute targets, three matched copies,
20 epochs, two original game folds, same3350 selected decisions, and a60minute
model budget are fixed before new fitting/timing. Independent protocol review,
implementation and synthetic tests are in progress; preparation/fitting have
not started. A positive internal signal would still need a separate practical
test. The breakthrough goal remains active and unachieved.

### Future supervision protocol frozen; auxiliary cache prepared

Independent review accepted `docs/future_goal_protocol_v1.md`, SHA256
`545dd8121281159516fb14661edf663d5e1d65f51ee17a7383e6175ae573b6bb`.
Before any real preparation,11 synthetic data tests and4 paired-evaluation
tests passed. The public static fallback was narrowed to genuinely missing
native item metadata; other compiler failures propagate. Forbidden outcome/
timing paths are rejected before hashing by the new preparation adapter.
The original census's timing arrays are opaque recorded ancestors here and
were not reopened; the derived labels and all consumed bindings were checked.

Preparation `future-goal-cache-20260930-v1` completed in68.132964s, with all
392448 states,4001 games,3350 selected decisions and258 goal identities.
Cache manifest SHA256:
`f1e4043e57c98e2b4ad9c95600f7f0aaf4928b9106b0aafcece5c426ce47ab3b`.
Every9333 main class and258 goal class has valid native attributes on all
three static patches; no fallback was actually required.

For held-out fold0, its196837 training rows have3571 no-donor fallbacks
(1.8142%);65076 goals changed,72.4185% of89861 nonempty labels. For held-out
fold1,195611 training rows have3533 no-donor fallbacks (1.8061%);64056 goals
changed,72.2743% of88629 nonempty labels. EMPTY status was preserved. Every
held-out shuffled label/donor remains sentinel-1; original goals are retained
when no other training game is eligible. These are control-coverage figures,
not model performance. Independent cache audit and model/trainer integration
are in progress; no new model timing, fitting or score has occurred.

### Auxiliary cache independently verified; constructor diagnostic inconclusive

`data/reports/future_goal_cache_audit_v1.json` independently passed in29.997s.
It checked every strict-later/EMPTY label and every training donor using a
different rank-selection calculation, all3350 original selections, every
counted basket and all23 native static coordinates, and zero reserved-game
intersections. It opened neither reward/timing array values nor any model.
The preparation and audit remain successful and frozen.

A separate CPU-only constructor/API diagnostic stopped at its own0.75GiB
host RAM floor after12.2085s:699056128bytes (0.651GiB) were available. This
diagnostic did NOT record its initial available memory or own working/private
set and only required0.75GiB at startup; it therefore cannot identify whether
the drop came from constructor allocation or another workload. The constructor
did not finish. This is inconclusive integration evidence, not a failed cache,
failed predictive recipe, or justification for changing model/loader semantics.
No model, GPU preflight, fit or score occurred. The actual durable training
launcher retains its3.25GiB startup gate and0.75GiB live floor, with host and
own-process memory sampling. Complete synthetic/source review first, then
run that guarded stage when resources permit; do not repeat low-headroom probes.

### Future-supervision implementation reviewed; durable trial queued

The model/trainer/CLI are complete. Eleven focused CPU synthetic tests passed
for the author and independent reviewer. They verify exact validated/fast
forward, loss, gradient and Adam behavior; auxiliary gradients reaching served
state/item/basket representations; baseline goal-only parameters inactive;
native composition masked after the MLP; no target inputs to main inference;
mixed-patch resource timing and the fixed92040-update recipe. The independent
reviewer read the full loop, receipt binding, unchanged paired evaluator,
mutation coverage, and final metric/integrity gate conjunction and found no
blocking issue. Cache audit SHA256:
`6e85d442a9d8b764a6118c5433ce5387ed5b88b974ed15f9730cb93fe0f9a9b5`.

Frozen model SHA256:
`280ca35c8947b78ca596e2e455ca67d33d4988baf31355bc9ac8babb79b78629`.
Trainer helper SHA256:
`3521a062800d2c8fef09c455d6797ad742aaa0ae686c75ad518590642c2aa4d8`.
CLI SHA256:
`f1bb717404a49e03cbec709fda12e8cd5c5b76a83fd0b6b81760a90c2726f445`.

Root registered `Videns Future Goal train` at15:36:35UTC for
`data/runs/future-goal-probe-20260930-v1`, using the unchanged prepared v1
cache. The real PowerShell launcher PID36636 was verified live, rather than
inferring activity from a status file. At15:36:41UTC it was waiting for host
memory:1.66GiB available against3.25GiB startup threshold; GPU ready, no game
and no competing benchmark. No real model preflight or fit had yet begun.
Continue following THIS task/run/handle; do not restart because observation
times out or the resource wait is slow. Outer wall90minutes includes waiting;
the complete model stage still has its fixed3600-second cap and mandatory
cost projection before real fitting. The local review preview remains up.

This goal turn made progress through independent full-cache audit, completed
implementation/tests/review and a queued live experiment. It establishes no
predictive improvement or breakthrough. The broader goal remains active, all
1911 confirmation games closed, and the original index SHA256 still equals
`94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46`.

### Resource wait resolved; the same future-supervision trial is training

The existing launcher automatically obtained headroom and started its worker
at15:39:10UTC. No restart or recipe change occurred. Root verified venv parent
PID13932 and actual Python workerPID37528 under launcherPID36636. Mandatory
synthetic timing passed: projected remaining2512.738752s versus3560.341846s
remaining in the fixed3600-second model stage, including the existing1.25
work margin and180s allowance. All92040 updates remain required.

At elapsed383.269s, mainonly fold0 had finished20 epochs and future fold0 was
on epoch7. At15:45:43UTC the launcher measured3481268224bytes available host
RAM and440909824bytes combined own worker resident memory. This successful
guarded start supersedes the earlier resource-inconclusive CPU constructor
diagnostic; it does not establish the cause of that earlier host-memory drop.
No held-out comparison metrics or partial checkpoint results have been read.
Continue the SAME task and process handles; the run is genuinely live.

While waiting, root implemented `data/reports/future_goal_result_audit_v1.py`,
SHA256 `6254e31af887e5b6665e592e244cc0c75dcf2fa1f80aa7ace705a76795abe7e6`.
Three synthetic tests pass, including independent bootstrap/gate parity and
intentional metric tampering rejection. The audit reuses the prior independent
audit's arithmetic, not the production evaluator; the new arm mapping is
explicit. It also checks the exact preflight budget/projection and complete
named integrity-gate set. It will check all selected predictions, six fresh initializations,
optimizer active/inactive parameters, all120 epoch records/40 shared orders,
static basket buffers, and96 deterministic CPU re-inference rows only AFTER
the task is terminal and its final report complete. It opens no partial
results now. Preparation of this verifier and the newly running experiment
are concrete progress; the broader practical breakthrough remains unproven.

### Final audit queued behind the live producer, without partial-result reads

New own operational scripts `scripts/run_future_goal_audit.ps1` and
`scripts/register_future_goal_audit.ps1` passed PowerShell parsing. Root
registered `Videns Future Goal audit`, run directory
`data/runs/future-goal-result-audit-20260930-v1`, at15:50:08UTC. Its actual
launcher PID9740 was verified live. At15:50:11UTC it reported
`waiting_for_producer`, having found both live Python processes for the exact
producer experiment. It reads status/process metadata only while fitting.

The audit requires the producer task and exact worker command lines to be
terminal, launcher state completed, final experiment status complete, and a
final report before invoking the verifier. It additionally waits for3.25GiB
host RAM/no competing benchmark or game; CPU1,0.75GiB live floor,300-second
verifier ceiling and six-minute outer worker wall. Total dependency/resource
waiting wall is100minutes. It stops on producer failure and never treats a
missing/partial final report as completed. Its output will be
`data/runs/future-goal-result-audit-20260930-v1/audit.json`; do not launch a
duplicate final audit while this task is pending or running.

The producer remained live (PID37528) at elapsed663.835s, shuffled fold0
epoch3, meaning mainonly/future fold0 had each completed20 epochs. Available
host RAM at15:50:28UTC was3053989888bytes. This goal turn added the durable
dependent audit and verified the same running producer; no comparison score
was opened or interpreted. Both model and scientific recipe stay unchanged.

### Verified wait: second partition is progressing normally

Root repeatedly confirmed the SAME actual Python PID37528 and dependent audit
launcherPID9740 live. No task or worker was restarted. At elapsed1344.167s,
the producer was future fold1 epoch10: four complete20-epoch models and the
fifth in progress. No worker error was present; the audit continued waiting
for the exact producer. This continuation is a verified wait on live work,
not a new result or a blocked goal. All model/control settings remain fixed;
no partial prediction/checkpoint comparison has been inspected.

### Future-supervision experiment complete: useful-signal gate fails

The SAME `future-goal-probe-20260930-v1` worker completed all six20-epoch
models and92040 updates, both fixed frequency controls, complete3350-game
OOF predictions and10050 full-logit target-mutation checks. Runtime was
1891.665595s (31.53minutes); the producer became terminal at16:10:51UTC.
No fit was resumed, shortened or selected using an intermediate score.
Report SHA256:
`2e76b99d515554acb2bbaca90982f7452b009d5cb7ff3fd8dd2eb64758d7e6fd`.

Same immediate main-basket endpoint on the exposed internal FIT-derived
cross-validation population, with no legality filtering:
- Matched compositional main-only: NLL2.165135848,top1=44.388060%,top3=70.179104%.
- Real future supervision: NLL2.216514806,top1=43.313433%,top3=68.417910%.
- Matched shuffled supervision: NLL2.179695094,top1=44.119403%,top3=70.059701%.
- Fixed frequency control: NLL3.985319483,top1=21.522388%,top3=38.119403%.

Real future supervision minus main-only top1 is-1.074627pp, paired95% interval
[-2.388060,+0.149254]pp,224 improved and260 regressed games. Relative NLL
reduction is-2.373013%, interval[-3.744283,-0.954438]%: NLL is significantly
worse under the fixed conditional bootstrap. Against shuffled supervision,
NLL reduction is-0.036819712, interval[-0.063058678,-0.010895748], and top1
gain-0.805970pp, interval[-2.059701,+0.447761]pp. Every statistical/practical
gain condition fails; coverage/admission/integrity conditions pass. Auxiliary
goal accuracy cannot rescue this negative immediate-purchase result.

This closes the declared lambda1, six-minute strictly-later auxiliary recipe.
Do not retune its weight, horizon, target vocabulary or successful slice to
change this result's sign. The new compositional baseline's comparison with
older architectures is not the preregistered future-supervision contrast;
M2 already has compositional baskets. Do not rebrand this failed hypothesis as
a new policy benefit. Nothing here establishes causal action value or wins.

### Numerical audit failure preserved; full GPU replay verifies every prediction

The first final auditor independently reproduced the complete3350-game
metrics/paired bootstraps, then failed its CPU spotcheck threshold2e-5. Its
failed run and unchanged code remain preserved. Diagnostic
`data/reports/future_goal_numerical_diagnostic_v1.json` (SHA256
`27dabecd0439644045aae89d9d6a96c002fd1125d73471490197a60acbb8a89e`)
located the first discrepancy on main-only fold0: CPU maximum target-logP
error3.690337e-5, small-batch GPU error3.923486e-5, but original-batch GPU
error exactly0 for every sampled row, including all rankings. This identifies
batch/backend floating-point effects in that diagnostic; the original CPU
threshold is not falsely recorded as passed or silently relaxed.

NEW `data/reports/future_goal_result_audit_v2.py`, SHA256
`81ab52398122ccb7131fe14745c4a53a50e5f5784350cfafb2fc988adaa43982`,
retains all original metadata, metric, optimizer and epoch checks, records
every CPU discrepancy, and additionally requires independent manual gathering
and exact GPU replay of ALL selected rows using original256-row batches and
global inventory padding7. Its dedicated v2 task waited for resources, ran
after verified producer completion, and finished at16:22:56UTC in38.166492s.

Final evidence `data/runs/future-goal-result-audit-20260930-v2/audit.json`,
SHA256 `6f66f1c337a64fc7452da3ec7703af93947e7029778cf29ba9b7a788c8e089bf`,
passed256 bindings, all3350 prediction rows, six fresh initialization/final
optimizer receipts,40 epoch permutations and120 epoch records. All10050
neural target log probabilities were bitwise identical and all top3 rankings
matched in the independent GPU replay. Among96 CPU spotchecks,6 exceeded the
original2e-5 threshold, maximum4.624571e-5, with zero rank differences. These
CPU findings remain explicitly reported; no training or metric gate changed.

All producer/audit tasks are now terminal. The original index SHA256 remains
`94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46`;
all1911 confirmation games stay closed. This continuation yielded completed,
audited negative evidence, not the requested practical breakthrough. The
broader goal remains active and unachieved.

### Recorded component-initiation census: earlier timing, insufficient clean support

After the negative future-supervision trial, a distinct temporal-abstraction
hypothesis was specified in `docs/commitment_census_protocol_v1.md`: examine
the FIRST recorded defensive component initiation before tier-two boots and
its subsequent recorded recipe consumption, rather than assuming a finished
boot is the initial strategic choice. The options/SMDP framework motivates
the unit (Sutton, Precup and Singh,1999,
https://www.sciencedirect.com/science/article/pii/S0004370299000521); it does
not itself identify an observational action effect or validate private intent.

Correction of interpretation, not of the old data: the earlier19 boot cases
were reliable/unmixed/full-price conservative monetary certificates on6000
games, with70 under the separately declared common ordinary-Boots sensitivity.
They were not a count of all actually legal defensive decisions. This new
4001FIT-game population is different, so no19-to-new-count improvement ratio
is meaningful. The old result and protocols were left intact.

New code `winprob/commitment_census.py` and `scripts/commitment_census_v1.py`
passed12 synthetic tests. Native public recipe graphs also passed on868,
868 and870 items respectively; no source gameplay payload was opened for that
check. The durable counts-only run `commitment-census-20260930-v1` completed
in41.199026s at16:44:03UTC, with33212 recorded episodes across4001FIT games.
Primary analysis kept the same3350 metadata-fixed focal episodes and651
unavailable focal games, without using outcomes or selecting a replacement.

Among3350 focal episodes,658 have a neutral single retained component
initiation:307 pure-main and351 mixed-main baskets. Pure Cloth Armor counts
are111/89 in the two folds; Null-Magic Mantle counts52/55. There are301
matching defensive-boot traces (194 armor,107 MR),272 traces to OTHER items
(199 armor,73 MR), and85 ambiguity/censoring cases. The matching-boot elapsed
time median is147017ms (2.45minutes), interquartile range106383–213519ms.
The recorded component is therefore not an independently observed boot intent.

The predeclared500-pure-focal minimum and100-per-action-per-fold minimum both
FAIL; the200-matching-boot minimum passes. No value regression, outcome study
or GPU fit is justified by this specific census. Do not loosen the purity or
sample gate after seeing these counts. Estimated-wallet early/late cross-tabs
remain diagnostics, with no true-wallet/slot-legality certificate or policy gain.

Report: `data/runs/commitment-census-20260930-v1/census/report.json`, SHA256
`0860d848c7ad65f49bc8e00ff0d204b58195fca54dc509c35e87d63e72c57419`.
`data/reports/commitment_census_aggregate_audit_v1.json` independently checked
183 bindings, all33212 record identities, seven aggregate summaries, the
3350-game focal denominator, the exact readiness gates and zero reserved
intersections in5.443s. This audit does NOT independently reconstruct the
item-token traces; those remain the producer's synthetically tested and
explicitly limited inference from recorded inventories/recipes. No reward,
terminal target, raw archive, database or model was read by this census/audit.

This continuation added and executed a new structural feasibility screen,
which rejects fitting this component-to-boots unit on the current fixed focal
sample. The broad goal remains active and unachieved. A next mathematical
direction to inspect is privileged-future-state distillation for the WIN
probability target itself: for m(X,Z)=E[Y|X,Z], E[m|X]=E[Y|X] and squared-risk
minimization against the true m differs from the Y objective by a constant.
That is a possible variance-reduction mechanism, not the failed auxiliary
goal-class objective and not a promise of finite-sample improvement with an
estimated teacher. Inspect the strongest completed state-only W1 baseline
and available frame artifacts before selecting a fixed protocol. No new
distillation protocol, teacher, model, target cache or result exists yet.

### Privileged future-state WIN distillation completed and audited: gate fails

The next hypothesis was fixed in `docs/wp_distillation_protocol_v1.md`, SHA256
`3a591840b59c1823ce5904021c6bf1c4da6797c5bffdad11b8d003a806684ca1`.
For the true conditional mean m(X,Z)=E[Y|X,Z], current-only squared risk against
m differs from risk against terminalY by an irreducible constant. This supplies
a training-target variance argument, not a finite-sample guarantee for an
estimated teacher. Both teacher types retain currentX; only the joint teacher
also receives futureZ. No future-only conditional-expectation shortcut is used.

The same clean6000-game rich cache supplies61 current retrospective OMNI
features and unchanged4002FIT/1000calibration/998development games. Four game
folds inside FIT create eight teachers: two fitting folds, one calibration
fold, one OOF target fold each. Future state is the earliest qualified frame
three clock-minute bins later in the SAME FIT game. All92294FIT frames stay:
80288 have a future state;12006 retain zero future features plus a missing
flag. Actual available horizons are180002–180295ms (median180053ms). No
calibration/development game supplies teacher fitting or future targets.

Three matched squared-error HGB students use only61 CURRENT features and the
original22554 sampled FIT frames. Targets are hardY, current-only teacher OOF
probability, or current/future teacher OOF probability; no mixing/horizon
search. All share the fixed150-iteration/15-leaf recipe and1000-game external
Platt calibration. The already published rich classifier is a fourth strong
reference, reused unchanged. Thirteen synthetic tests passed before fitting,
including role isolation, missing-frame retention, nested-information layout,
the risk identity and rejection of privileged student inputs.

`wp-distill-20260930-v1` completed in146.078016s; its producer was terminal at
17:24:03UTC. On all23366 development frames from998 games (equal game weights):
- Frozen rich classifier: log-loss0.494341441,Brier0.162921454.
- Hard-target regressor: log-loss0.507103902,Brier0.163203399.
- Current-teacher student: log-loss0.495681786,Brier0.163248238.
- Future-teacher student: log-loss0.495306223,Brier0.163023887.

Future student versus strong frozen reference has relative log-loss reduction
-0.195165%, paired95% interval[-0.736782,+0.358527]%. Absolute reduction is
-0.000964783, interval[-0.003681593,+0.001747241]; Brier reduction-0.000102432,
interval[-0.001037926,+0.000737422]. Neither is a demonstrated improvement.
Against the matched hard regressor, relative log-loss reduction2.326482%
(interval[0.924863,3.963536]%) is real under the fixed conditional bootstrap,
but Brier remains unresolved and the stronger existing classifier is unbeaten.
Against current-teacher smoothing, relative log-loss reduction0.075767% has
interval[-0.168081,+0.339826]%. Five of seven primary gain gates fail.

The joint teachers themselves have14.381414% lower OOF FIT log-loss than
current-only teachers. This is privileged-information diagnostic evidence on
FIT, not a deployable forecast or development improvement. That advantage did
not transfer to the current-only student. Do not substitute the weaker hard
regressor comparison or privileged teacher score for the declared primary gate.
This specific distillation recipe is closed, with no promotion or causal claim.

Independent `data/reports/wp_distillation_audit_v1.json` (SHA256
`036c9fa11a48088413b96083a8f330419454ee24c1c5c75811a0351b6f08fc24`)
passed in26.389100s after a resource-gated durable audit run. It revalidated62
bindings, reconstructed all FIT future alignment and four-way memberships,
recomputed the exact22554-frame sampler, replayed184588 OOF teacher
probabilities and70098 development student probabilities, and independently
recomputed primary/raw/teacher metrics, paired intervals and all seven gates.
Both maximum probability replay errors were exactly0. It refitted no model.
Secondary time/patch slices were not separately audited and are not used to
change the decision. The final report digest is bound in that audit; a matching
report sidecar was added after verifying that digest, without changing report
bytes. All tasks are terminal, confirmation remains closed, and the original
index hash remains94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46.

This turn produced an implemented, executed and audited negative experiment;
the overall breakthrough goal remains active and unachieved. A distinct next
question is whether the present61-variable representation loses useful CURRENT
information: champion identity/resource ownership, unspent versus invested
gold, and recorded combat statistics are absent from that feature list. Review
their existing source availability and define observational-only forecasting
scope before selecting an experiment. Such fields must not be repurposed as
an action-neutral counterfactual item judge. No new feature extractor, cohort
expansion, model or positive result for this direction has been created yet.

### Observed current-combat information: complete extraction and fixed screen

Ten FIT games were selected by a fixed hash, five per training patch, before
opening their20 bound raw sources. `current-state-schema-20260930-v1` inspected
234 exact existing frames/2340 player frames in9.895756s, without indexing wins
or final participant fields. Health/maximum health/wallet and the inspected
combat fields were complete. Health was zero in220 records. AbilityHaste was
constant0 and was excluded; attackSpeed100–425 was treated as a reported index,
not attacks per second. These are schema observations, not performance evidence.

The new frozen protocol `docs/current_combat_protocol_v1.md`, SHA256
`d122d4bdb4730a1b38be741dab78864772a8f0f897f894a66340fa095efaed84`,
declares two CURRENT-only candidates and a fixed simpler-model preference.
`features_current_combat_v1.py` pools112 new fields from health availability,
health/resource proxies, native public champion tags and reported combat
statistics. It is permutation-invariant and emits blue-minus-red/blue-plus-red
features. Health-only uses four columns. Neither positive health nor
totalGold-currentGold is treated as certified alive status/inventory value.
The models are retrospective observational WP, explicitly NOT counterfactual
item judges or live-safe predictors. Fourteen synthetic feature tests and eight
model/selection tests passed before their real stages.

The same `current-combat-cache-20260930-v1` task waited for resources, then
completed without restart at18:53:19UTC in189.475544s. It reopened only12000
compressed paths already bound to the original clean6000 games and retained
all138831 original frames. It read immutable roster fields and exact current
participantFrames, not raw wins/final roles/performance. The feature-preparation
NPZ reader never accessed cachedY. Separate GAME_END timestamp metadata is
evaluation-only, never a feature; a known end conflicting with a current frame
would abort. Current gold exactly matched the two original gold summaries.

Manifest SHA256:
`117f7ad81732ce9758cbf088037a57ff680a858331d76f8da2e863ef1236bfed`.
Health, valid health ratio, wallet and dynamic-health-series coverage were100%
in every original role. All quality gates passed. End timing was known for all
6000 games, with range617781–3001664ms. This guards against merely broadcasting
final stats and supports the fixed retrospective lead-time check.

`current-combat-fit-20260930-v1` completed both calibrated150-iteration models
in55.948574s at19:05:05UTC, with identical22554FIT frames/4002games and1000-game
external calibration. On the unchanged998-game/23366-frame development set:
- Reference rich classifier: log-loss0.494341441,Brier0.162921454.
- Health availability(65 current inputs): log-loss0.494480331,Brier0.163051609.
- Combat/resources(173 current inputs): log-loss0.489673652,Brier0.161340409.

Combat versus reference relative log-loss reduction is0.944244%, paired95%
interval[-0.333950,+2.257483]%. Absolute reduction0.004667789 has interval
[-0.001623665,+0.011249926]; Brier reduction0.001581045 has interval
[-0.000815091,+0.004071347]. Health-only relative reduction is-0.028096%,
interval[-0.191291,+0.133575]%. No declared candidate passes the fixed2%
overall point gain and resolved proper-score gates.

At least180000ms before verified end, all998 dev games remain represented.
Combat's relative log-loss reduction1.047772% has interval
[-0.178506,+2.325285]%; the point0.5% margin condition passes but its interval
does not. This future-defined retrospective stratum is not a live eligibility
rule. No time/patch slice is used to rescue the screen. Selected candidate=null;
there is no promotion, confirmed gain or breakthrough claim.

Report SHA256:
`e6b55e79b73b03a4a48eca6909976927bd8ea6e235433555f4acc55756db8c6c`.
Independent `data/reports/current_combat_audit_v1.json`, SHA256
`ea74ba7b1440ab1cbee96b6e9654776286fbaa5b50bd51b852d02497d1048fca`,
passed in11.228107s. It checked full138831-frame algebra/identity/quality
invariants, independently reconstructed112 features on225 frames from10
metadata-selected role/patch games (maximum error0), replayed both models on
all23366 dev rows (46732 predictions,total maximum error0), and recomputed
primary/raw/lead-time metrics, intervals and the exact selection decision.
Its source validation was a sample, not an independent re-extraction of all
12000 raw files. No models were refitted. Secondary time/patch slices were
not separately audited. The input-only position-count/health-count equality
diagnostic is descriptive and does not prove conditional redundancy.

All tasks are terminal; confirmation remains untouched and the original index
hash remains94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46.
This continuation produced new data, two models and audited but insufficient
evidence. The full user goal remains active and unachieved.

### Next scope fact: the pilot leaves substantial admitted training unused

A read-only metadata check of unchanged admissionv2 still gives11468 train,
2866 calibration,2861 development,3282 development_exposed and1911
confirmatory_reserved games. Outside the current6000-game pilot are7466 train,
1866 calibration and1863 development games; these are metadata-role counts,
not newly verified feature coverage. No payload from that complement was read
in this check. The development complement is outside this project's pilot;
do not call it globally untouched or a confirmatory holdout.

Before another architecture or post-result tweak, inspect a separately
manifested scale comparison using the admitted training/calibration pool and
the same frozen rich/observed-combat recipes, with matched data-budget controls
and explicit old-pilot versus newly evaluated development reporting. Preserve
the current failed screen, all frozen code and all1911 confirmation games.
No expansion manifest, source feature extraction, fitted full-pool model or
result for that scale comparison exists yet. This is the next direction to
make concrete, not permission to relax the completed screen's gates.

### Full-pool scale study declared; all new archive sources are now bound

`docs/wp_scale_protocol_v1.md` fixes a distinct data-budget comparison before
any gameplay deserialization outside the pilot. Requested metadata scope is
17195 games:11468FIT/2866calibration/2861development. The original6000 pilot
IDs/order remain unchanged;11195 additional IDs are appended by date/ID.
All3282 development_exposed and1911 confirmation entries remain excluded,
together with current/prior external/global test/validation reservations.

The original admission and `extraction_candidates` initial6000-game cap were
not edited or bypassed by passing an out-of-range argument. A NEW explicit
study scope was published, retaining the parent roles/reservations and the
original training_authorized_by_manifest=false distinction: identity/source
admission is not itself a fitting authorization. Scope and compressed-source
binding artifacts are also marked training_authorized=false; model work awaits
the new protocol's feature/data-quality checks.

New `wp_scale_scope_v1.py` and `prepare_wp_scale_scope_v1.py` implement this
selection/binding. Four synthetic tests passed for unchanged pilot order,
safe complement, reserved/historical rejection and role/order tampering.
Scope `artifacts/wp-scale-scope/20260930-v1/scope.json`, SHA256
`a58b208c80703f79f0b05e2b1cfbef0302d4241f5ea66b7ab0da18bdb96b5391`,
was published before new archive-byte access.

The durable `wp-scale-bind-20260930-v1` run completed at19:52:35UTC in
216.304009s. It hashed22390 NEW compressed files for11195 additional games,
836999259bytes total, with zero missing endpoints. No gzip was deserialized
and no outcome field was indexed. The old6000 compressed bindings were reused
without reopening their raw sources. Combined bound scope is17195 games.
Sources manifest SHA256:
`056e0e4bd9933eefb7507240c0742d290bee414e74e49159986b507e6b117f5c`.

Independent `data/reports/wp_scale_scope_audit_v1.json` checked set/role
accounting, exact22390 unique new byte-access ledger entries, pilot reuse,
scope/source digests and zero reserved intersections without rehashing raw
sources. This proves availability/binding, not feature qualification or model
quality. The task is terminal; no new gameplay deserialization, expanded
feature cache, model fitting or complement-development score exists yet.

Next implement the streaming expanded cache using the UNCHANGED base/rich/
current-combat extractors, with exact pilot parity and explicit missing/zero-
frame exclusions, no replacement or outcome-based filtering. Reusing pilot
arrays and writing new per-game shards avoids a large RAM concatenation.
Keep the original22554 sampled FIT rows identical by retaining old game
indices0..5999; append new qualified games after them. Primary evaluation is
the newly evaluated development complement (metadata count1863), old998 and
union reported separately. Do not call the complement globally untouched or
confirmation. Protocol includes frozen original models, pilot-tree/full-Cal
recalibration controls, fresh full-FIT rich/combat candidates, fixed gates and
deterministic selection. All1911 confirmation games remain closed. This turn
made concrete scope/source progress; the full breakthrough goal is unachieved.

### Streaming scale-cache implementation prepared and queued

New `winprob/wp_scale_cache_v1.py` and
`scripts/prepare_wp_scale_cache_v1.py` reuse the unchanged rich61 and observed-
combat112 feature definitions. They preserve all original6000 game indices
and every original row/label/timestamp, write immutable per-game NPZ shards,
and append new qualified games in the previously declared order. Each shard
also stores the unchanged deterministic FIT sampler's local indices. Pilot
shards are read back and checked exactly; the original22554 sampled FIT rows
must remain. Missing archives/zero base-qualified frames are explicit
exclusions, while malformed/mismatched data aborts. No outcome-score filtering
or replacement is introduced. Six synthetic converter tests pass, including
outcome/future mutation, wrong queue/patch and exact pilot parity checks.

The new `wp_scale_models_v1.py` provides bounded per-role loading of sampled
FIT/all calibration/all development rows and the protocol's fixed four-
candidate hierarchy. Six synthetic tests pass for mandatory comparison to
BOTH frozen references, data/lead-time minima, fixed simpler-model preference,
and shard label/sample integrity. This helper has not fitted a model. The
full-scale training CLI and final audit still need implementation after this
cache stage is ready; do not imply a scale result already exists.

Root registered `Videns Data wp-scale-cache` for
`data/runs/wp-scale-cache-20260930-v1`. Its actual PowerShell launcherPID11660,
created20:05:12UTC, was verified live. At20:08:51UTC it was waiting for2.5GiB
startup RAM (1.67GiB observed); no Python extractor/source payload stage had
started. Keep following the SAME task/PID/run, not a replacement. The stage
has its declared1800-second extraction cap, CPU1 and0.75GiB live floor; outer
task limits include resource waiting. No original source/code/data, reserved
game or index was changed. This turn implemented/tested the scale-cache and
model-data/selection plumbing and established a verified resource wait. The
overall goal remains active and unachieved.

### RAM released; expanded cache running and fit/audits prepared

The user explicitly freed RAM. At20:20:44UTC the SAME queued cache task
started its Python worker (actualPID39496, venv parent41768, launcher11660).
Available RAM rose to4.41GiB, later3.95GiB. The latest verified live checkpoint
was15450/17195 games, zero exclusions, elapsed677.53s. Do not restart this
existing run because an observation call times out. Confirmation stays closed.

The full scale runner scripts/train_wp_scale_v1.py is implemented: frozen
rich/combat references, copies recalibrated on the full calibration pool,
fresh fixed-recipe full-FIT models, complement-only primary selection and
separate pilot/union reports. The missing summarize import was corrected
before any real fit. A synthetic calibration test verifies that recalibrating
a deep copy preserves original raw trees and predictions exactly.

New scripts/audit_wp_scale_cache_v1.py checks scope and raw-access accounting,
every shard, independent frame-sampler reconstruction, exact6000-game pilot
parity and deterministic new-game raw replay. This replay intentionally uses
the already audited frozen extractors, not a second feature implementation.
The runner requires a passed cache audit before fitting.
New scripts/audit_wp_scale_result_v1.py independently checks proper scores,
paired game bootstrap intervals and selection, and replays all six pipelines.
Operational launchers now expose separate cache-audit, fit and result-audit
stages. These audit/fit stages have not yet run at this checkpoint.

### Full scale extraction complete

The original cache job completed at20:33:57UTC in778.088656s: all17195
requested games qualified,396667 frames, zero exclusions. Roles are11468FIT,
2866calibration and2861development; the new complement is7466/1866/1863.
All declared quality/coverage gates pass. Cache manifest SHA256:
7a6852da971246e67051b0ad7622b27b1ff20fd7ce8cb43d9fd76ccf41d89c14.
Sampled resource marks show340.0MiB peak combined worker working set and
3.83GiB minimum available host RAM. Seventeen scale synthetic tests pass.
The published index hash is still94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46.

The separate cache audit started20:34:05UTC (actualPython39468, venv45280,
launcher46052) and was verified still running with no stderr. Follow this
same audit run rather than duplicating it. Training remains contingent on
its passed artifact. No new model result or breakthrough exists at this point.

### Cache audit passed; full-scale training launched

Cache audit completed20:38:17UTC in245.551572s. It verified all17195 shards,
all6000 pilot games and original22554FIT sample rows exactly, reconstructed
the sampler independently, accounted for all22390 new payload reads and
replayed237 frames from10 deterministic new role/patch games (20raw sources).
All checks passed; zero reserved intersections. Full roles have263524FIT,
66279calibration and66864development frames; full sampledFIT contains64422.
Cache audit SHA256: 5f8ed1ae30200181ff10f825adc7f40b821b2d1171649b586a504edbf2beb484.

The SAME queued fit task automatically started20:38:41UTC:
run data/runs/wp-scale-fit-20260930-v1, launcher33836, venv parent33712,
actualPython43516. It is loading and verifying the17195 shards before any
fit. Do not restart this live run. All fit-consumed project modules and the
training CLI are now frozen. The result auditor was registered separately
as Videns Data wp-scale-audit (run wp-scale-result-audit-20260930-v1) and waits
for the fit task to finish. It requires a completed producer/report before
reading outputs, independently recomputes scores/selection and replays all
six pipelines. No development score has yet been inspected; the breakthrough
goal remains active and unachieved. Continue following these exact tasks.

### Scale fit v1 stopped before new fitting or comparison; execution fix v2

The v1 fit stopped after238.187657s at the first recalibration identity check.
Its directory contains only experiment_lock/status files: no fitted candidate,
training_lock, prediction archive or development comparison. The dependent
v1 result auditor correctly failed closed because the producer had failed.
Both attempts are preserved and terminal; do not relaunch their directories.

Read-only diagnostics isolated the failure: joblib hashes include array
strides. In each of150 loaded HGB trees, deepcopy changes both EMPTY(0,8)
categorical bitset arrays from strides(32,4) to(0,0). Their dtype, shape and
bytes are unchanged; all named node fields and node bytes are identical.
The initial node-padding diagnostic disproved padding as the explanation.
Recalibration does not change any raw model field. All23171 pilot-calibration
raw predictions and original calibrated predictions remain exactly identical.
This is an execution identity check error, not an empirical model failure.

New wp_scale_identity_v2.py identifies every tree array by dtype,shape and
C-order bytes, retaining the entire remaining model state, indices and scaler.
It ignores storage strides only. Two tests cover empty-array layout, a loaded
serialized model, recalibration invariance and detecting a changed tree value.
Preflight on BOTH frozen rich/combat models passes exact23171-row checks.
Evidence: data/reports/wp_scale_copy_diagnostic_v1.json,
wp_scale_node_padding_diagnostic_v1.json,wp_scale_copy_layout_diagnostic_v1.json,
and wp_scale_identity_preflight_v2.json. No complement score was inspected.

scripts/train_wp_scale_v2.py and audit_wp_scale_result_v2.py preserve the
same frozen scientific protocol, cache, features, models, calibration recipes,
gates and selection. Only raw identity and execution-version paths change.
The original v1 sources stay frozen. Operational launchers now target v2.

The v2 fit started20:47:44UTC: launcher39952, venv parent44268,
actualPython31672. The result audit v2 was queued only after this worker was
verified running, avoiding two waiting jobs blocking each other. It is
Videns Data wp-scale-audit / data/runs/wp-scale-result-audit-20260930-v2.
At20:51:25UTC actualPython31672 remained live,CPU55.34s, still loading the
verified cache with empty stderr. No v2 candidate comparison exists yet.
Follow these exact live jobs; do not restart for a stale stage timestamp
(the loader reports only once before the per-game loop).

### All four v2 candidates fitted; comparison publication still pending

The corrected identity check passed. By elapsed271.642s, rich_full and both
pilot-recalibrated models had been saved and combat_full was fitting. At
elapsed351.410s the run entered complement evaluation; at368.234s it entered
union evaluation after the prior fixed cohorts. The process remains verified
live at20:56:07UTC (actual31672,parent44268,launcher39952), with no stderr.
No completed report has been published or read yet. Last status can remain
union while secondary summaries and final17195-shard hash checks run.
The v2 result auditor is queued, waiting for the SAME active fit job to end.
Follow these exact jobs and read the completed report only when published.
Do not infer failure from an unchanged stage timestamp or start a duplicate.
The mathematical protocol and selection gates remain unchanged; no useful
improvement is yet established and all1911 confirmation games stay closed.

### Full-pool scale study completed and independently audited

Fit v2 completed20:58:02UTC,607.416511s. Audit v2 completed21:01:55UTC,
222.175937s. Both tasks are terminal; no worker needs following or restarting.
The audit independently verifies42 proper-score/ECE summaries,69paired
comparisons, all six model pipelines/401184 development predictions (maximum
replay error exactly0), split/identity/calibration controls and the deterministic
selection. It checks77bindings and zero reserved intersections; no raw payloads
are reopened during this final audit. Time/patch slices are secondary and not
separately audited. All1911 confirmation games remain closed. Index hash still
94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46.

Report data/runs/wp-scale-fit-20260930-v2/experiment/report.json SHA256:
6e10781c05ce85d706ef87f723c36bdabebbc75613624193bd2144a8fa2fe014.
Audit data/reports/wp_scale_result_audit_v2.json SHA256:
388d0e963a00aba6f96ec7d19bf28198f0ed7ebfa87ddf800bd65ae39ce29c7f.
Primary cohort:1863development-complement games outside the old6000pilot.
It has NOW been evaluated; never call this cohort untouched in a later study.
The998old-pilot and2861union results remain separate secondary analyses.
FIT11468games/64422sampledframes; calibration2866games/66279frames.

Primary equal-game log-loss/Brier:
- frozen rich:0.486866085/0.161999014;
- frozen combat:0.482555904/0.160476481 (stronger frozen reference);
- rich recalibrated:0.487010310/0.162120392;
- combat recalibrated:0.482828995/0.160629876;
- rich full:0.483797278/0.161003058;
- combat full:0.478832998/0.159085455.

Combat full vs stronger frozen combat improves relative log-loss0.771497%,
paired95% interval[0.070386,1.462065]%. Paired Brier reduction0.001391026,
interval[0.000078204,0.002689331]. Its at-least3-minute-to-end reduction is
0.694412%, interval[0.013874,1.377384]%, with all1863games represented.
Against frozen rich, full combat improves1.649958% (interval[0.901214,2.424992]%).
Both2% point-reduction gates FAIL; every other full-combat gate passes.
No candidate is selected or promoted. Do not weaken the declared gate.
This is a resolved small developmental forecast improvement, not no signal;
it is also NOT the useful-signal threshold, confirmation, causal action value
or a practical breakthrough. Recalibration-only controls do not improve;
full combat vs recalibrated combat improves0.827622% (CI[0.158225,1.506789]%).
More FIT data contributes modestly; no further scaling claim is justified.
The full combat model is the strongest measured research comparator for a
future declared study, even though no production promotion is authorized.

### Mathematical next direction: preserve joint player-state relationships

Before fitting anything else, a synthetic witness was implemented and run:
scripts/probe_wp_spatial_health_collision_v1.py ->
data/reports/wp_spatial_health_collision_v1.json. Two blue players have equal
resources/maxHP/tags/stats at different fixed positions. Exchanging1000HP and
100HP between the player near enemies and the distant player leaves ALL173
existing features exactly identical in four qualifying frames (max difference0).
A health-weighted Gaussian proximity statistic with fixed2000-unit width
changes by0.8722405795. Any deterministic model on the173existing inputs must
give identical predictions on these two constructed states. No real payload
was read and no model was fitted for the witness.

This proves representation information loss only. It does NOT prove that
these states have different true win probabilities, that exact collisions
are frequent in real data, that the proximity statistic measures combat
power/causal value, or that an added relation will improve predictions.
The candidate hypothesis is to retain per-player health/resources/position
and learn permutation-invariant player relations, using the strongest fixed
full-combat comparator and matched controls. No new real-data study/protocol
has yet been frozen or run. Do not retune scale recipes or inspect confirmation.

Primary references inspected:
- Deep Sets (Zaheer et al., NeurIPS2017):
  https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Set Transformer (Lee et al., ICML2019), attention among set elements:
  https://proceedings.mlr.press/v97/lee19d.html
- Relation Networks (Santoro et al.,2017), relational reasoning module:
  https://arxiv.org/abs/1706.01427
These support architectural mechanisms, not LoL-specific performance claims.
Their relevance to the observed aggregation loss is our inference. A next
step should define a falsifiable representation comparison and useful
coaching/decision application, not equate forecast accuracy alone with goal
completion. The broad goal remains active and unachieved. This turn produced
a completed audited scale result plus an executable mathematical witness.

### Joint-player representation study declared and extraction running

New docs/wp_player_relations_protocol_v1.md is frozen before new token-source
access, SHA2564095f3f2b908e174ed80b438c316bf303973e03e6307e3f4df0c81905e762763.
It uses the SAME17195audited games/396667timestamps and64422FIT sample rows.
Primary evaluation is now explicitly the full previously EXPOSED2861development
games, with998pilot/1863complement secondary. Do not call these untouched.
Strong reference is the frozen full-combat model from wp-scale-fit-v2, not the
weaker pilot reference. All1911confirmation and sibling/global seals stay closed.

Fixed arms: global173-only MLP; DeepSets-style learned mean pooling; two-block
attention plus learned-query pooling; same attention with deterministic
same-team position x/y/mask bundle shuffling. Four arms x three seeds20260929,
20260930,20261001 =12models. Each30epochs,batch512,AdamWlr0.0003/wd0.0001,
clip1,float32,no dropout/schedule/early stopping/checkpoint/seed selection.
All models keep173global inputs. Player tensor10x35: position3;gold1;wallet2;
net-outflow1;health2;healthMax2;ratio2;positive-health1;level2;sixreported
combatstats with masks12;sixtagfractions6;side1. No champion embedding,
per-playerXP/damage/CS,identity,outcome,event/future or item inventory inputs.

Global head237->128GELU->64GELU->1, with64zeros for global-only control.
DeepSet encoder35->64GELU->64GELU then mean10. Attention35->64 projection,
2self-attention blocks width64/fourheads, post-residualLayerNorm and
64->128GELU->64 FF; learnedquery plus same attention/FF block for pooling.
No ordinal encodings. Blue/red side explicit, no map-symmetry assertion.
Normalize globals/tokens using sampledFIT equal-game/player weights only.
Three raw seed sigmoid probabilities are averaged and Platt-calibrated as an
ensemble on all2866calibration games with the existing weighted recipe.
Save every model/preprocessor/calibrator/receipt before NEW dev inference.

The fixed gate keeps>=2%resolved log-loss and positiveBrierCI versus full
combat,>=0.5%resolved gain/nonnegativeBrier versus neural-global, same lead-time
condition versus full combat with>=1700games, at least2/3positive raw seeds
versus raw reference. Attention must also beat its matched position-shuffle
control by>=0.5%resolved LL/nonnegativeBrier. Prefer qualifyingDeepSet unless
attention adds a resolved0.5%. Neither control is a promotion candidate.
See the protocol for exact fields and limits; no training has happened yet.

Implemented new own modules:
- features_wp_players_v1.py: exact current-frame constituent values from frozen
  rich/current-combat validity rules;35featureallowlist and explicit missing masks.
  Seven synthetic tests pass (shape/tags/sides,missing values,invalidratio,
  outcome/future invariance,health-location retention,order,exact timestamp).
- wp_player_models_v1.py: four fixed architectures, manual multihead operations,
  no hidden positional ordering, public retrospective-only capabilities.
- wp_player_preprocessing_v1.py: deterministic SHA256same-team position shuffle;
  equal-game FIT normalization with a stable two-pass centered variance. The
  initial one-pass variance expression was corrected before real fitting to
  avoid cancellation in constant observed-mask columns.
- wp_player_data_v1.py: single-pass verified parent-shard reading plus mmap
  player slices; assembles raw x/players/y/game/timestamp/end/pilot/patch perrole.
  It has been pycompiled, not run on the unfinished player cache.
- wp_player_selection_v1.py: exact two-candidate gates and simpler-first rule.
  Six model/preprocessing tests and five selection tests pass. Tests verify
  permutation invariance, identicalattention/shuffleinitialization, no token
  use byglobalcontrol,nonzeroQ/K/V/pool gradients,shuffle preservation,
  equal-game normalization/constants,mandatorycontrols/strongreference,
  all12models,seedstability,leadcoverageandBrier. Total18newtests pass.

scripts/prepare_wp_players_v1.py is RUNNING and now frozen along with its
feature module and protocol. It reads only bound oldscope IDs, logs34390
expectedrawreads, reproduces EVERY112combat value exactly against oldshards,
checkspositionavailability, and writes contiguous players.npy (396667,10,35)
plus game offsets/metadata/hashes. No labels are extracted for tokens; labels
remain in the frozen parent cache. No raw features are inferred/interpolated.

LIVE task: Videns Data wp-players-cache.
Run: data/runs/wp-players-cache-20260930-v1.
Started21:15:06UTC, launcher44304, venv parent41096, actualPython29300.
Last verified live at elapsed420.287s:6350/17195games,146753/396667frames,
empty stderr, CPU221.64s. Do not restart this process due to unchanged/late
observations. Startup RAM2.5GiB/live0.75, CPU1, extractioncap1800s, outer55min.
No GPU work is running. Operational data launcher supports wp-players-cache.

NEXT WORK: finish the training/evaluation CLI and its durableGPU launcher
(start>=4GiBVRAM,2.5GiBRAM;stop ownjob forLeague/BuildTracker or liveRAM<0.75;
5400s stage cap, adequate outerwait). No such training CLI/job exists yet.
Use load_player_data once (avoid repeatedly opening17k shards); train12models
with independent fixed epoch permutations, then calibrate3seed ensembles,
freeze outputs before newdev predictions, verify oldfull-combat replay,
compute fixed full-dev/secondary/lead comparisons and use selectionhelper.
Implement independent player-cache audit and all-primary prediction/metric/
selection audit. Preserve GPU original batch512 for exact later replay;
CPU/GPU float roundoff must not silently loosen a failing audit. All source
models/caches and previous recipes remain frozen. No automatic promotion.

This goal turn made concrete progress: frozen falsifiable protocol, extractor,
models/preprocessing/selection implementation,18tests,and verifiedlive extraction.
No relational-model performance or practical breakthrough is yet established.

### Player tensor and independent cache audit complete; twelve-model job live

Player extraction completed21:32:10UTC,1018.903902s:17195games/396667frames,
players.npy555333928bytes. Manifest SHA256:
121d2afa80722e14fb1dfc7a611d1c1e382a37369cefca6c8b10771d8015ae0c.
Independent cache audit completed21:32:48UTC in19.175525s. It checked every
frame's token algebra/quality,34390raw-access ledger entries, allgameoffset/
parentmetadataalignment, and independently reconstructed227frames from10
role/patch samplegames exactly(maxerror0). Zero reserved intersections.
Audit SHA25653db155622183ef84c69df97f552e274c3f2aafdbe984184dc1ab8db591d3eba.
Cache source/feature code remains frozen. All1911confirmation games stay closed.

New wp_player_training_v1.py implements fixed30epoch/batch512/AdamW training,
equal-game weights, deterministic epoch permutations, state hashes and exact
batch inference. Three synthetic backend tests pass: repeated training exactly
matches, every row is visited, matchedattention/control initializations and
batch orders match, and exactlythree seed probabilities are required.
GPU synthetic preflight in data/reports/wp_player_gpu_preflight_v1.json passed
allfour arms with513syntheticrows/batch512(two epochs each; no realdata).
GPUreplay was exact, runtime2.385083s. Parametercounts:global38785,deepset45249,
attention141569,matchedshuffle141569. No real experiment model was fitted by
that preflight. Before the real job, RAM3.91GiB and GPU6371MiB were available.

scripts/train_wp_players_v1.py is implemented and NOW RUNNING/FROZEN. It loads
verifiedrows once, reproduces the strongest full-combat reference exactly,
fits FIT-only normalization, fixes positional permutations, trains all12models,
calibrates each3seed ensemble, publishes a complete training_lock before NEW
development inference, then computes the fixedcohorts/gates and final hashes.
No hyperparameter/epoch/seed selection. It checks all30epochs,3780updates and
1932660rowvisits per model; matchingattention/control initial weights and
allarm epochpermutations must match. Cal-only player-order invariance tolerance
is1e-5. Exactmodelweights/receipts and original512inferencebatch are saved.

LIVE TRAIN task: Videns Player WP train.
Run: data/runs/wp-players-fit-20260930-v1.
Started21:40:45UTC, launcher40944, venv parent45824, actualPython26800.
At elapsed829.580s the first11models were saved; lastmodel
attention_position_shuffled/seed20261001 was at8/30epochs. Empty stderr.
Follow this SAME process; do not restart due to a stale or timed-out observation.

New durable scripts/run_wp_players_job.ps1 and register_wp_players_job.ps1
implement train/audit modes. Startup2.5GiBhost/>=4GiBfreeGPU, live0.75GiB,
CPU1; stop ownworker ifLeague orBuildTracker starts. Training5400s Pythoncap,
outer120min/task125; audit1800sPythoncap,outer60min/task65. Hidden one-shot
jobs; no modification of older source-owned jobs. The final auditor is queued
as Videns Player WP audit / wp-players-result-audit-20260930-v1, waiting for
this active training job, with ExperimentVersion wp-players-fit-20260930-v1.

scripts/audit_wp_players_result_v1.py and scripts/wp_player_audit_helpers_v1.py
are complete/pycompiled. The auditor independently reconstructs FIT normalization,
position permutations, all neural operations from state-dict tensors, and all
calibration/development probabilities onGPU with batch512; raw probabilities
must match exactly. It refits only the four Platt calibrators for verification,
then independently recomputes proper scores/ECE/bootstrap intervals/seed gates
and selection. It checks the same strong reference and all12trainingreceipts.
No neuralweight refit and no raw gameplay reopening. It shares the already
verified row loader, with that limitation declared. Time/patch secondary
slices are not separately audited. It requires a completed producer/report.

Two independent-helper tests pass. An initial synthetic exact-forward test
exposed ~1.5e-8..3e-8 differences for attention when state tensors had detached
requires_gradFalse flags. Passing the native nn.Parameter.requires_gradTrue
flags made ALL4architectures match exactly onCPU, even without calling their
forward implementation. The helper now preserves those flags before inference;
no numerical tolerance was relaxed and no real-result audit has run yet.
Normalization is independently recomputed by equal-game means/centeredvariance
and checked at rtol1e-12/atol1e-10; calibration coefficient comparisons use1e-9.
All these audit checks were implemented before inspecting new dev predictions.

NEXT: follow train26800 and its queuedGPU auditor. Preserve source modules now
bound by the running training job. Read only the completed report; no interim
seed/development cherry picking. If an audit fails, preserve it and diagnose
rather than editing a consumed source or widening thresholds silently.
No relation-model performance has been inspected; no practical breakthrough
or completion is established. This turn adds21->23 totalnewsynthetic tests,
completed/audited cache, working trainingCLI/GPUjob and complete finalauditor.

### Joint-player neural study closed: audited negative result

The full12-model/30epoch run completed21:58:43UTC (2026-09-30),1068.784522s.
Every model completed3780updates/1932660rowvisits; total45360updates. The
primary cohort is2861PREVIOUSLY EXPOSED development games; no fresh-test claim.
Report data/runs/wp-players-fit-20260930-v1/experiment/report.json SHA256:
2775cbccdc795a977d3622dd2b932614fc59ecdef1bd4377a1decfe579a673eb.
Primary equal-game log-loss/Brier:
- full-combat frozen reference:0.482955487/0.159914985;
- global neural:0.491002939/0.163036138;
- deepset:0.491620902/0.163194234;
- attention:0.510100302/0.170388518;
- attention_position_shuffled:0.508115799/0.169360174.

Relative log-loss INCREASE versus strong reference:
- global1.666293%,paired95%CI[0.707003,2.576448]%;
- deepset1.794247%,CI[0.799276,2.682580]%;
- attention5.620562%,CI[4.306275,6.875293]%;
- shuffledattention5.209654%,CI[3.961853,6.433508]%.
Deepset does not resolve improvement over global (relative reduction-0.125857%,
CI[-0.362716,+0.117506]%). Attention does not beat matched position shuffle
(relative reduction-0.390561%,CI[-0.883997,+0.127332]%). All real candidates
fail the declared gates; selected_candidateNone. Neither neural family is
promoted. Do not retune these completed recipes or weaken their gates.
This rejects the tested neural recipe, not all possible joint-player information.
Global neural also loses to HGB, so learner choice is a material confound when
interpreting the proposed representation's value.

Auditorv1 is preserved FAILED at elapsed288.280621s. It had completed exact
functionalGPUreplay and primary/secondary ensemble metrics, then stopped on
rawseedlogloss. The old independent proper-score helper retained float32 for
seed probabilities, whereas the producer's frozen metrics usefloat64; its ECE
weighted residual also retained float32. Diagnostic
 data/reports/wp_player_seed_metric_precision_diagnostic_v1.json
checked every seed: float32 LL differences~1e-9 and ECE~1e-11; explicitfloat64
proper-score arithmetic and ECE residuals recover the published values to
original1e-12 tolerance. Keep ECE bin assignment in the probability's native
dtype. Nothing in the experiment/report/model/predictions was changed.

New scripts/audit_wp_players_result_v2.py copies/preservesv1 and overrides
only independent numeric semantics. It passed a12-seed summary/interval/ECE
preflight before its real run. All metric tolerances remain unchanged.
The fullv2 audit completed22:13:27UTC (2026-10-01local),285.759129s.
Artifact data/runs/wp-players-result-audit-20260930-v2/audit/report.json SHA256:
87dc9d833758ea528851e410510c82923c0bb2da5f54ae58b64314b850fd2afd.
It independently replayed ALL795348calibration+802368development raw neural
probabilities =1597716, with EXACT0error for every model. All four independently
refitted verification Platt calibrators match coefficients/intercepts exactly0.
FIT normalization/position controls were independently reconstructed;87bindings,
47summaries,30paired comparisons and deterministicselection verified. Zero
reserved intersections; no new raw payload read during result audit. Original
neural weights were not refitted. Secondary time/patch slices remain unaudited.
Both fit andv2audit jobs are TERMINAL; no Videns task remains running.

Measured run resource marks:1629.1MiB peak combined worker working set,
1.98GiB minimum available hostRAM; peakGPU allocated370603008bytes and reserved
404750336bytes onRTX4060. Index remains94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46.
All1911confirmation games and frozen sibling artifacts remain untouched.
README now records the closed negative result. The broad goal is still unmet.

### Next prototype: analytical Gaussian fields while retaining strong HGB

Created winprob/features_wp_kernel_v1.py and five synthetic tests, allpassed.
This is a PROTOTYPE ONLY: no real player tensor was processed with it, no
kernel-feature cache/model fit/evaluation exists, and no next protocol is frozen.
No change was made to any consumed neural or parent-cache source.

The prototype uses the raw10x35player tensor and a fixed Gaussian radius
2000worldunits (same as the existing rich geometry scale), withKij=exp(-dij^2/
(2r^2)) and explicit position masks. Four NONNEGATIVE observation-weight
channels:positive-health indicator,healthfraction,healthfraction*totalGold/10000,
andhealthfraction*max(netGoldOutflow/10000,0)*walletObserved. They are proxies,
not certified alive status/combat power/inventory value. For each positive-
health positioned anchor, compute normalized blue-minus-red local field
balance. Summarize mean/min/q25 within each own side (red orientation reversed),
emitting blue-minus-red andblue-plus-red. Also compute normalized cross-team
kernel inner product and squared MMD, plus both-positive-masses availability.
Missing/zero masses yield0 with their flag0. Additional side count/fraction
features reveal positive-healthposition,ratio-position andwallet-ratio-position
coverage. Exactly42extras; proposed future total input215=173+42.

Squared MMD iswb_normalized' K wb_normalized +wr_normalized' K wr_normalized
-2wb_normalized' K wr_normalized. Clamp only negative roundoff after asserting
>=-1e-12. Tests cover identicaldistributions(zeroMMD/balance),within/allplayer
permutation,side-sign orientation,missingpositions(no inventedcontact), the
health-location witness, squaredMMDbounds andexplicitzeromass handling.
This discerns the mathematical witness without requiring a learned encoder.
It does not establish real-data predictive value or practical benefit.

Candidate NEXT ACTION: predeclare a matched strong-HGB comparison with these
fixed analytical joint fields and a same-team position-shuffled-field control;
retain the strongest full-combat reference and its fixed HGB150/15/.06/L2=10/
minleaf100/seed20260929 recipe. Do not present this suggestion as already run
or committed. New feature computation can reuse the audited player tensor and
parent rows, with no new raw archive reads. Prefer loading the sampled FIT/all
cal/all dev rows once via frozen load_player_data; it already aligns197565
modelrows, so control hashes need only those timestamps. The stored neural
position_permutations NPZ is a bound deterministic control artifact if reused.
A versioned packed matrix could avoid reopening17kshards for every later fit,
but no such new packing has been implemented. Any experiment must keep all
own/sibling/global seals closed and label all2861development games exposed.

Primary sources inspected for this direction:
- Grinsztajn et al.(2022), tabular-tree vs neural benchmarks:
  https://arxiv.org/abs/2207.08815
- Gretton et al.(2012), A Kernel Two-Sample Test/MMD:
  https://jmlr.org/papers/v13/gretton12a.html
The first supports checking learner inductive bias rather than assuming NN
superiority; it does not prove our next HGB augmentation will work. The second
supports the kernel discrepancy concept, not this LoL weighting/decision claim.
The fixed weighted fields are our proposed application, not a published win
improvement or causal strategy. No breakthrough or practical benefit yet.

### User steering: reject the current predictive-model research approach

The user stated: "te puedo asegurar que este current approach no tiene futuro".
Treat this as a strategy correction within the still-active broad goal. Stop
pursuing the current chain of win-probability representation/model variants.
Do NOT run the newly written wp_kernel_protocol_v1.md experiment. At this
point it is an unexecuted proposal: no real kernel features/cache/model scores
have been computed. Preserve the prototype and prior negative evidence, but
do not continue them automatically merely because previous notes proposed it.

The assistant acknowledged the strategic error: optimizing a proxy predictor
has not demonstrated better concrete purchases or macro decisions. A concise
optional preference question is pending: prioritize M2 purchase decisions,
actionable macro-review decisions, or another useful videns.gg capability.
This is not a request for permissions already granted and does not pause or
complete the broad goal. Reframe the next work around an explicit practical
benefit, a concrete decision and a falsifiable comparison before another model
or expensive experiment. Do not replace the rejected line with an equally
unsupported new family of models by default.

### User priority resolved: improve M2 purchase decisions

The user answered the strategy question: "Mejorar las decisiones de compra
de M2". This supersedes the pending-question note. The active broad goal now
prioritizes practical purchase decisions; do not resume WP/kernel variants.
wp_kernel_protocol_v1.md has been explicitly marked shelved before execution.
Its prototype and five synthetic tests remain historical, not an active study.
No kernel data/model experiment or additional predictive training was launched.

Read-only reassessment followed build_tracker/AGENTS.md -> CLAUDE.md and
reviewed docs/history.md before proposing another breakthrough. Ran the
required source-owned scripts/status.py at22:35UTC. No BuildTracker job was
running (Daily Pull ready), and no deployment_manifest was bound for serving.
Do not conflate the M2 experimental reference with a promoted/live policy.
No source code/data/job was changed; no source runtime was imported into winprob.

Verified in app/mixture_policy.py lines264-306 and308-347: M2 combines two
normalized grouped imitation distributions with fixed0.5/0.5weights, ranks
the top128groups and displays up to3distinct actions that pass the native
purchase_sequence at the same wallet. Its catalogue has9333main groups.
Source manifest says6354TRAIN games/757975visits, d64/twolayers/fourheads,
historyFalse, POVgoldTrue; these are artifact metadata, not new evaluations.
There is no explicit outcome/decision-utility comparison in this options path.

Mathematical interpretation: when the sorted shortlist yields three legal
options, those options maximize the sum of M2's own predicted masses among
legal three-option sets (ties aside). Simply reranking the same probabilities
cannot create an independent notion of purchase quality. This says nothing
about calibrated human propensities, causal item effects or actual best buys.
The earlier859-game and M2/BC-union results remain exposed, closed evidence.
Do not recycle them as proof of a new decision benefit or train on their targets.

Reviewed app/review.py and relatedroutes: the inspected review code is a
read-only timeline/inventory display, not an expert purchase-quality rater.
The project's history explicitly distinguished observed purchases from whether
advice is good. A current comprehensive quality-label inventory is NOT yet
established; do not claim there can be no other annotation artifact elsewhere.

Inspected only the bound public item export boots_static.json (no gamepayload):
patch16.17/16.18 have868items,16.19has870; numeric stats expose13attribute keys.
Descriptions/recipes/publictags exist, but this does not establish an executable
model of every passive/active/champion interaction. Do not manufacture exact
combat utility from those13stats or silently ignore unmodeled effects.

Next work should specify a concrete purchase-quality criterion and verify what
independent evidence can support it BEFORE picking another learner/experiment.
M2 remains the fixed reference and a source of plausible legal options. A new
candidate must demonstrate situational purchase benefit under explicit criteria;
imitation exactness or lower generic win-prediction loss alone is insufficient.
Potential evidence routes (not selected protocols or promised gains) include
verified mechanical consequences under stated scenario/goal assumptions and
independent expert preference evidence. Preserve ties/unknowns and tradeoffs;
do not invent a unique 'best' label from an observed historical purchase.
Do not launch a broad simulator, another model family or an expensive pipeline
until the benefit, comparison and applicable evidence are concrete.

No new performance number or M2 decision improvement is claimed. This turn
made the user-directed strategy pivot, preserved the rejected line, and verified
the actual decision objective/available interfaces in source code. The broad
goal remains active and unachieved; no pause or completion was requested.

### M2 quality-criterion clarification and a concrete blind rubric pilot

Asked one optional clarification: should improvement primarily mean better
purchase quality even when different from the logged action, or better
anticipation of high-Elo purchases? No explicit reply was received during this
turn. After allowing time, provisionally prioritized purchase quality based on
the user's earlier steering; do not record that provisional choice as an answer.
The explicit user priority remains "Mejorar las decisiones de compra de M2".

Prepared an eight-case RUBRIC PILOT, not a performance benchmark, new policy
or proved improvement. Script scripts/prepare_m2_quality_review_v1.py uses only
existing versioned859-game ordinary contexts/M2options/native19legality receipts.
It checks admission/current reservations and source hashes before reads. No
raw archives, SQLite, target file, outcome, probability matrix or model is read.
One previously admitted context's field names (not its values) were inspected
to understand the schema before publishing the selection plan.

Selection plan published before option-content selection:8<=minute<=25,
at least2M2buy options, all shown options verified native19feasible at the
supplied wallet, known public item names, no unmodeled focal slots. Eligible394
of859; exclusions324outsidewindow,116unmodeledslots,25fewer-than2buyoptions.
Rank bySHA256('m2-quality-review-v1/'+sample_id), at mostone per champion,
take8. Option display order is separately deterministic/blinded. Selection
uses no logged purchase, outcome, correctness or model probability. It is a
conditional exploratory sample; never generalize an acceptance rate to allM2.

Artifacts: artifacts/m2-quality-review/20261001-v1/{plan.json,cases.json,
escrow.json,manifest.json}, each core file digest-bound. cases.json SHA256:
ad0289aa8465cc6dcb9b383900f43453228da4cf8f3e228a4d1c5e1e59908490.
The escrow contains original M2ranks/sourceidentities; do not expose it during
blind judgement. Public view omits match/player identifiers, original ranks,
probabilities, historical targets, outcomes, totalGold, enemy wallets and
postmatch-inferred roles. It shows suppliedESTIMATEDbudget, current reconstructed
inventory/levels/focalKDA/CS, visible boards/objectivecounts, exactnative19
incremental costs and copied public16.19item descriptions. Do not mistake the
estimated budget or reconstructed state for an observed live exact snapshot.
Empty main group is shown as no main purchase, not an invented fullSAVE action.

Cases are C01Renekton,C02Qiyana,C03Renata,C04Rumble,C05Blitzcrank,C06LeeSin,
C07Syndra,C08Thresh. Each option can be preferred/acceptable/discouraged/
insufficient_information; ties and reasons/assumed goals are supported.
NO HUMAN QUALITY LABELS EXIST YET. Do not invent them from the logged purchase,
from our own judgement or from UI test interactions. A user's later preferences
would be rubric evidence, not causal win effects or objective-best ground truth.
Do not train a candidate or claim policy benefit on this selected pilot.

User-facing comparison is an inline visualize fragment:
C:/Users/Javier/.codex/visualizations/2026/09/29/01a0eefa-9b66-79a3-bb60-f1afe758a8d9/m2-purchase-quality.html
It embeds all8cases, has no externaldata fetch, lets the user rate each option,
keeps local drafts and offers an explicit follow-up submission. Autosaved state
is marked submitted:false; only an actual user-submitted message is feedback.
It includes the cases SHA256 for exact version matching. Never treat an
unsubmitted/draft widget state as independent quality labels.

Read canvas and visualize skills. No matching current-workspace Cursor canvas
runtime directory was identified; used the app-supported inline visualization
for the comparison. CUA preview failed at runtime (deny-read-ACL helper error);
this was a tool startup problem, not an auto-review rejection. Completed QA via
the existing bundled Playwright in an isolated headless Edge, no personalprofile.
All8case views,4allies/5enemies,option counts,grade changes and draft persistence
were checked; syntheticQAgrades were cleared and NOTHING was submitted.
No browser errors, no horizontaloverflow at900/360 widths. Screenshots inspected;
fixed a blank fallback-output region and wrapped champion labels. Final QA:
artifacts/m2-quality-review/20261001-v1/ui_qa_v2.json andpublic_view_audit_v2.json.
Temporary local preview servers62182/94600 were intentionally stopped; no
ongoing experiment or UI preview job remains. Source M2 and index untouched.

IMPORTANT: Include the visualization content reference in THIS turn's final
response if not yet sent. Future visual edits require reloading the full
visualize skill before creating/updating the fragment. Do not restart shelved
WP/kernel experiments. The goal remains unmet. Next work must use the clarified
criterion and/or real independent rubric feedback, not assume this UI is itself
a breakthrough or force labels on ambiguous purchase choices.

### Review ingestion prepared; independent purchase-quality input still missing

No actual submitted ratings or answer to the primary success-criterion question
arrived in this goal turn. Do not infer them from UI drafts or synthetic tests.
The previous turn produced a real eight-case blind rubric pilot; this turn
completes the bounded receiving/analysis preparation, not an M2 improvement.

New winprob/m2_quality_review_v1.py validates explicit submitted payloads only,
requires the exact case version/hash, rejects unknown cases/options/grades and
duplicateJSON keys, preserves partial/insufficient ratings and same-grade ties,
and constructs strict preference pairs only between explicitly comparable
rubric grades. It never opens the M2-rank escrow, targets or game outcomes.
It does NOT authenticate the reviewer: the operator must bind an actual user
submission, not assert that a draft or QA fixture is human evidence.

scripts/ingest_m2_quality_review_v1.py writes a new immutable receipt directory
only when provided an actual reviewJSON and source note. It preserves original
submittedBYTES and their digest separately from the validated interpretation.
No real review was ingested or generated. Eight synthetic tests pass, including
draft/type/hash/identity rejection, ties/unknowns, partial notes, byte preservation
and refusal to overwrite. Synthetic CLI output lived only in a verified temp
subdirectory and was cleaned; no human-rating artifact was created.

Additional safe inventory searched fileNAMES (including ignoredfiles) and
review interface code in build_tracker/macro_coach. Fourteen matching paths:
our own unrated review bundle, prior M2 revision/preference artifacts, and
macro_coach/data/feedback/feedback.jsonl. The latter exists,4039bytes, modified
2026-09-19T16:23:40UTC. Its CONTENTS WERE NOT READ. The inspected macro UI emits
match_id,pid,t0,token,verdict,model_top3,note: macro-token feedback, not a verified
M2purchase-option rating schema. Do not claim this file is empty or that no
human feedback exists anywhere. Its suitability/admission for M2has not been
established, and no cross-project feedback import was made. Source-owned
versioned admission/export would be needed before any such reuse.

The remaining dependency is a concrete accepted purchase-quality criterion
and independent judgement/evidence for it (or an explicit choice to optimize
historical purchase prediction instead). Existing purchase imitation labels and
our own model outputs do not independently establish better buys. The question
and review pilot have now been outstanding in TWO consecutive goal turns.
Autonomous preparation is complete: do not manufacture more experiments,
ratings or peripheral tooling to conceal the missing input. On a further
continuation, first inspect whether genuine user feedback/criterion has arrived.
If the same dependency persists and no meaningful authorized independent work
is available, apply the goal's three-turn blocked audit rather than repeating
status updates while leaving it active indefinitely. Never mark the goal complete.
No permission, RAM or account-usage restriction is causing this dependency.
No model job, revised M2policy or active WP/kernel experiment was launched.

### Third consecutive unresolved quality-reference dependency

Rechecked the authoritative state: the8-case pilot remainsprepared_unrated
with0qualitylabels; no submitted review/receipt path was found in winprob
artifacts. No actual user-submitted ratings or explicit choice of the primary
success criterion arrived in the conversation. Macro feedback metadata is
unchanged (4039bytes,2026-09-19); its content remains unread and its use as M2
purchase-quality evidence is not established.

The same dependency has now persisted across THREE consecutive goal turns:
(1) criterion question and blindpilot preparation, (2) complete receipt/parser
preparation plus independent-source inventory, (3) this revalidation. The
remaining need is evaluative evidence or an accepted operational criterion,
not more formatting/code or a resource permission. The user rejected the
predictive-model search; substituting another proxy or generating our own
'correct buy' labels would not demonstrate the requested purchase benefit.
The rejected kernel proposal remains explicitly shelved and unexecuted.

Autonomous preparation is complete and no meaningful justified experiment
can be selected/evaluated under the still-undefined purchase-quality criterion.
Mark the existing goal blocked, not complete or paused. Preserve the full
objective and all artifacts. Resume when actual purchase-quality feedback or
another usable independent reference is provided, or when the user explicitly
chooses historical purchase-prediction accuracy as the operational target.
No breakthrough/M2improvement is claimed. No account quota/RAM blocker exists.

### 2026-10-01: user resolves the criterion -- observed purchase is correct

The user explicitly states: "la respuesta correcta a todas estas compras es
las que se compra el jugador". This selects historical purchase prediction as
the operational objective for improving M2. It supersedes the provisional
subjective-quality rubric and the preceding missing-reference blocker. Do not
ask for option ratings or require independent expert preferences to proceed.

The statement is a target-definition instruction, not a submitted set of human
option ratings. The unsubmitted widget draft supplies no ratings. No review was
ingested; preserve the eight-case pilot/parser as unused historical artifacts.
Do not use the selected eight cases as a new performance cohort.

Recorded the current contract in docs/m2_imitation_objective_v1.md and linked
it from README. The observed retained multiset, including multiplicities, is
ground truth. Keep M2's main-basket exactness and top-3 coverage explicit,
with strict full-basket/ward accounting separate. Reuse the existing evaluator;
no relabeling, retrospective metric changes or new training run occurred.

Read the existing ordinary report/evaluator and prior source experiment docs.
Existing M2 baseline remains 454/859 main first-exact, 591/859 main top-3,
and 354/859 complete first-exact (main-only exported recommendation interface).
These are old exposed-development results, not progress in model accuracy.
The prior learned gate/probability-mixture and basket-composition work must be
considered before proposing a supposedly new intervention. WP/kernel studies
remain shelved, and the failed BC reranker/union ceiling are not reopened.

All training/calibration/selection/confirmation boundaries persist; neither the
859 development targets nor the eight review cases become training data. All
confirmation remains closed, frozen source assets and index remain untouched.
The backend goal was previously marked blocked; no new goal was created and
no unsupported status transition was attempted. The dependency itself is now
resolved by the user's instruction. Continue toward demonstrated purchase
prediction benefit; do not claim a breakthrough or completion from this reset.

### 2026-10-01: purchase memory executed; positive internal signal, M2 unresolved

Previous goal turn classified PROGRESS: it recorded the user's explicit
observed-purchase criterion and removed the false subjective-label dependency.
Current get_goal returned ACTIVE; no replacement goal or status override.

Reviewed the earlier learned-gate/whole-basket/mixture studies. Context-dependent
mixing of the old parents had already been tried; it was not presented as new.
Read the primary kNN-LM paper abstract/source and adapted its retrieval plus
neural interpolation idea to actual main-purchase multisets. No WP/kernel work
was restarted. No extra user annotation was requested.

Declared docs/purchase_memory_protocol_v1.md before running the new method.
Implemented purchase_memory_v1.py,8meaningful synthetic tests, bounded runner
and hidden one-shot resource-gated launcher. Two original state-only encoders,
same3350fixed OOFqueries,opposite-fold same-champion training memory,32distinct
games,normalized128latent,squaredL2,adaptive fixed-form temperature,50/50mix.
Shuffled FIT labels within champion/patch/3minute bins are the negative control.
No training, parameter sweep, reward/raw/sealed payload read or source import.

Run data/runs/purchase-memory-20261001-v1 completed in80.9s. State43.9104%,
memory47.2836%,shuffled43.2239% main top1. Memory-state+3.3731pp CI
[2.3284,4.5672];244corrected131broken. Top3+2.3582pp;NLL2.361116->2.282992.
The internal gate passes. Old state outputs reproduce exactly,3350heldout-label
mutations leave representation unchanged. All query games excluded from own
encoder/memory. ReportSHA e6a7c38f099e5312cc10bacc81477217fbd8071df73f7daedc04e6e54c662ba9.
Independent audit data/reports/purchase_memory_audit_v1.json passed:105967
neighbor receipts,mixture targetNLL maxerror3.56e-15,all metrics/pairedCIs.
This audit does not replay nearest search/full ranking. No claim that this
weaker unmasked internal reference is M2;651missingfocalepisodes disclosed.

Then declared docs/purchase_memory_m2_protocol_v1.md BEFORE a new859-game
comparison. Same two memories averaged,0.5M2+0.25R0+0.25R1,only reorder frozen
M2legal options; maintain exact candidate objects. The new script reuses the
original own validators/source-owned versioned artifacts and public compiler.
It writes complete new orders before reading targets. Prior BC artifacts are
opened only to validate old reference lineage, not used as the candidate.
All source models/code remain frozen, and no source runtime is imported.

Run data/runs/purchase-memory-m2-20261001-v1 completed in90.6s. M2main454/859
(52.8522%),memory464/859(54.0163%),shuffle458/859(53.3178%). Memory-M2
+1.1641pp CI[-0.3492,2.5611],26corrected16broken; memory-shuffle+0.6985pp
CI[-0.6985,1.9790]. Primary gate FAILS, not enough evidence for useful M2gain.
Complete main-card exactness354->360/859 (+0.6985pp CI[-0.5821,1.9790]);
top3unchanged591/859. Zero feature fallbacks,859forbidden-feature checks,
exact old M2main/full/top3reproduction, unchanged native18/native19legal options.
The receiving audit is data/reports/purchase_memory_m2_audit_v1.json; inspect
its final status before calling that audit complete. No weights/checkpoints
or metrics are retuned to rescue the failed gate.

Both durable tasks completed and removed themselves. No new model job remains.
docs/purchase_memory_result_v1.md and README record both results and scopes.
New candidate generation from memory remains UNMEASURED; current evidence is
only reranking and cannot establish that a new decoder will help. Preserve all
artifacts and seals. This is concrete research progress, not the requested
breakthrough or practical integration. Goal remains active and unachieved.

Final receiving-audit result: purchase_memory_m2_audit_v1.json completed and
passed. It independently recomputed all 1718 candidate orderings, 859 observed
multiset scores, corrected/broken counts and paired intervals. The negative
M2 gate reproduces. No encoder/search replay or new native legality run is
claimed. Both experiment/audit workers are terminal; the report index SHA256
remains 94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46.

### 2026-10-01: full native memory candidate experiment completed and rejected

Previous turn classified PROGRESS: two executed/audited memory studies changed
the next action. This turn tested the previously unmeasured candidate-generation
question while preserving the exact learned representations, neighbor receipts,
weights and scores. No parameter search or retuning on the 859 targets.

Read source AGENTS/CLAUDE and ran scripts/status.py at08:58UTC before source
integration work. No source job running, Daily Pull ready, no model bound for
serving. Existing source working-tree changes were left intact. Read the exact
MixturePolicy.options/options_override/native activation/audit helpers.

Declared docs/purchase_memory_candidates_protocol_v1.md, then added a versioned
winprob distribution exporter and evaluator, plus one NEW source-owned adapter
build_tracker/scripts/decode_purchase_memory_v1.py. Source runtime imports occur
only in the source process; the consumer exchanges hashed artifacts. No frozen
policy/economic/model/database/scheduled-job code was changed. The adapter calls
the unchanged native options method with captured probability matrices and
audits unchanged candidate objects in a separate clean native19 process.

Own run: data/runs/purchase-memory-candidates-20261001-v1. Source artifacts:
build_tracker/data/exports/winprob-memory/purchase-memory-candidates-20261001-v1
under decoded/ and native19/. Full distributions are exactly 0.5M2+0.25R0+
0.25R1; old restricted candidate scores reproduce. All predictions/native
receipts existed before the new evaluator opened the actual-target file.

Result: 464/859 main first exact (54.0163%) versus M2 454/859 (52.8522%),
gain+1.1641pp CI[-0.3492,2.5611];27 corrected17 broken. Restricted reranker
also464; full access corrects one/breaks one, zero net additional first gain.
Top3 memory582/859 vs M2591/859: -1.0477pp CI[-2.3283,0.2328],12corrected
21broken. Candidate sets change on362games. Full shuffled control455first/
573top3. Complete first360 vs354. No secondary result rescues the primary.

All2516memoryoptions legal in BOTH native18 and native19; source baseline
replay0mismatches and2577arm/context target mutations unchanged. The gate
fails on first-gain size, unresolved primary intervals and mean top3 loss.
Fixed recipe CLOSED without more weight/neighbor/checkpoint tuning. Internal
state-only gain is not an M2gain; no breakthrough, promotion or integration into
live serving. All1911own confirmation games and external seals remain closed.

ReportSHA f71c8ea6fa848895d896e8738338fd742af7ced78faf3b390707c5184f489a0a.
Independent receiving audit data/reports/purchase_memory_candidates_audit_v1.json
passed;SHA c6e71a675b90bf3e4234fd3010f2a700cc47bbf6d1d3aa28cb0f42255619ab01.
It recomputed observed multiset metrics/movement/pairedCIs/gates and checked
source receipts; no claim of independently rerunning native economics.
Source decode manifestSHA e3bf5354db0608198d94670ebb9374429cf978591ded213fcb67acd2a5b6fa3f;
native19 manifestSHA96ccd7efc2c200aacb9e2a7e2bad2347dad644e2a9c1f65eeab5154e4890f940.
CPU1 stage times: distribution14.6s,decode23.1s,native19audit4.9s. No GPU or
new training. Durable chain completed and removed its own task; no worker
remains. README and result doc updated, index SHA unchanged. Goal active and
unachieved. Future work needs a genuinely different justified mechanism; do not
reopen this fixed recipe or disguise another weight sweep as continuation.

### 2026-10-01: exact unordered purchase likelihood implemented and benchmarked

Previous goal turn classified PROGRESS: it completed and audited the full
native candidate test and closed the fixed memory recipe. This turn starts a
distinct learning objective, not a memory retune or a return to WP/kernel work.

Read the real BasketEncoder/BasketNet in source train_basket_ab.py and the
whole-basket protocol: M2 already has nonlinear joint basket interactions;
never claim it simply predicts independent items. The new model instead
factorizes over next item/STOP and sums every distinct sequence yielding the
same retained multiset. Prefix state is itself a multiset, allowing a small
acyclic dynamic program. Duplicate copies do not receive factorial multipliers.
Consulted primary Vinyals et al. Order Matters (arxiv1511.06391) and Uria et al.
NADE (JMLR17/16-272); the recurrence is our application, not a claimed formula
or League result from those sources.

Added docs/multiset_likelihood_design_v1.md, winprob/multiset_likelihood_v1.py,
tests/test_multiset_likelihood_v1.py and a bounded synthetic benchmark script.
Eight tests PASS: exact exhaustive probabilities and gradients, repeated-copy
counting, whole-vocabulary normalization, empty/forced STOP, permutation/padding
invariance, invalid bags, outside-catalogue mass and shared-vs-local DP equality.
No game payload/target/outcome was read for the mathematical tests/benchmark.

The immutable public catalogue has9333main groups,196itemIDs,maxsize6,max32local
submultisets,10037unique shared prefixes,60625localnodes,85274localedges and
max120distinctorders per target. Shared-prefix inference avoids rescoring the
same partial basket for every catalogue target. Metadata-only admission counts
were read: FIT9988patch16.17+1480patch16.18; calibration2866patch16.18;
development2002patch16.18+859patch16.19; exposed731/1209/1342; confirmation1911.
No identity/output list was used to open reserved game payloads.

Synthetic run data/runs/multiset-likelihood-synthetic-20261001-v1/report.json
completed in6.14s in a bounded test shell.40synthetic optimizer steps,10warmup,
head63805parameters,batch256 meanstep0.016973s,p95 0.034181s. Shared catalogue
inference batch8 mean0.006583s (0.000823s/query throughput, NOT live latency),
peak CUDA allocated217949696bytes. Shared/local float32 logmass maxerror
9.5367431640625e-7. Gradients finite. No actual training data, model checkpoint,
accuracy score or gain; the common state encoder/gather costs remain unmeasured.

Declared docs/multiset_probe_protocol_v1.md BEFORE real fitting: three matched
encoder arms (categorical, exact marginal, canonical-order-loss control),20epochs,
batch256,Adam3e-4,whole-game OOF. Primary marginal must beat BOTH matched
categorical and the stronger existing47.2836% internal memory reference by3pp
with positive paired lower bounds; top3 no mean loss. Do not use a gain over
the weaker43.91% reference to imply practical M2benefit. Raw catalogue mass and
conditional NLL must remain distinct. No heldout-label-dependent generation.

NEXT REQUIRED WORK: implement and verify the real training harness, canonical
control, exact shared encoder initialization and full tensor-store preflight.
Only start training if the projected complete run fits80% of a60minute bound.
No real training job is running now; this is a finished synthetic feasibility
stage. Preserve the consumed prototype/benchmark versions if a correction is
needed. M2-native direct comparison/confirmation remain later requirements.
Index unchanged, seals closed, goal active and unachieved.

### 2026-10-01: real multiset training launched after full preflight

Previous turn classified PROGRESS: exact likelihood/gradient implementation,
eight synthetic tests, measured head-only feasibility and a fixed real-training
protocol. This turn implemented the full matched comparison and began real
training; no result is inferred from its training losses.

New winprob/multiset_probe_v1.py implements shared initial encoder copies,
identical initial marginal/canonical heads, the single canonical-order loss,
target-independent full-catalogue prediction, raw/conditional probability
accounting and the two-primary gate versus categorical AND frozen memory.
Five new tests pass alongside the eight likelihood tests (13 total): matched
tensors without aliasing, canonical versus marginal exactness/gradients,
complete deterministic epoch orders, explicit raw catalogue mass, and refusal
to rescue a failure against memory by beating a weaker categorical control.

scripts/train_multiset_probe_v1.py validates the original 4001-game/392448-state
FIT cache, 3350 selected OOF decisions and both immutable references. Frozen
memory predictions SHA204e23aa1843e237f6517cc5cdf40262a69402f64ae5bd782d9639b6fa3d3fc2;
original state SHA27eb094eb1dcfc1551b375446684b0b423d86c03b79077b2ccb5772c8c0968f5.
They reproduce1584 and1471 main hits respectively. All models use their explicit
opposite-fold games only, 20epochs, batch256, Adam3e-4 and gradient cap10.
Checkpoints freeze before scoring; actual held-out labels are not an input to
the all-catalogue prediction method. Raw catalogue mass is retained separately.

Durable task: `Videns Multiset Probe v1`; own run directory
data/runs/multiset-probe-20261001-v1. Started 2026-10-01T09:46:10UTC.
The wrapper PID is3824 and actual Python worker PID42524 (parent3824); always
revalidate those live handles/task before interpreting status, never rely only
on this note. No other source job was running at launch, League was absent,
available host RAM about3.9GiB and GPU free about6.2GiB. Startup floor3.25GiB,
live floor0.75GiB, CPU1, protected CUDA execution and60minute worker bound.

Full preflight PASSED with zero real updates:90synthetic optimization steps;
state cache preloaded. Measured end-to-end step means categorical0.010681s,
marginal0.024110s,canonical0.013996s; gather0.000436s. Inference batch8 means
0.011274/0.022381/0.021334s respectively. Whole planned work is92040updates;
projection2158.86s including1.25safetyfactor and180s overhead, below80% of the
remaining3564.10s. Real training then began. These timings are no accuracy claim.

At the last process-verified observation (2026-10-01T10:01:19UTC), all three
fold0 checkpoints and fold1 categorical were complete; fold1 marginal was on
epoch3, and PID42524 was alive. Do NOT restart or duplicate the run. All six
models must finish before aggregate comparison. The read-only NumPy warning
comes from copying row indices to CUDA; the index tensor is only used for
gathering and the source arrays are never written. No operational failure.

Prepared scripts/audit_multiset_probe_v1.py plus run/register_multiset_audit_job_v1.ps1.
The audit will independently verify epochs/Adam counts/IDs/initial hashes,
reference arrays, all metrics/CIs and replay all10050 neural query predictions
on the same GPU/batch/store path. Its launcher is NOT registered or started yet.
Only run it AFTER the training job completes and its report is complete; a
second running Videns task would trigger the current job's concurrency guard.
Audit source is fixed to this v1 experiment; use a NEW audit version directory.

NEXT: poll the existing task/process, then inspect completed report and launch
the prepared audit with e.g. register_multiset_audit_job_v1.ps1 -Version
multiset-probe-audit-20261001-v1. No training result, M2 improvement or breakthrough
is claimed yet. Do not edit consumed core/trainer/protocol files while it runs.
All seals and index remain untouched; the goal is active and unachieved.

### 2026-10-01: multiset experiment completed, exactly replayed, and rejected

Previous turn classified PROGRESS (real six-model training launched) with a
verified live wait at handoff. This turn polled the SAME task/PID, observed
normal progress and completion, then launched the prepared independent audit
only after the training task had exited/unregistered. No duplicate or restart.
One status-tool observation timed out transiently; re-polled its same exec cell,
which returned live progress. It was not treated as a training failure.

Training run data/runs/multiset-probe-20261001-v1 completed in1425.98s with all
six20epoch models /92040updates. Source selection4001games,3350fixed OOFqueries,
651unavailablefocalepisodes; no population change. Encoder164224parameters,
categoricalhead1220469, each generativehead63805; capacity difference disclosed.

Primary main first exactness: marginal1481/3350=44.2090%; categorical1515=
45.2239%; canonical1491=44.5075%; frozenmemory1584=47.2836%; oldstate1471=
43.9104%. Marginal-memory -3.0746pp CI[-4.4179,-1.7604],222corrected325broken.
Marginal-categorical -1.0149pp CI[-2.3582,0.2687],229corrected263broken.
Marginal-canonical -0.2985pp CI[-1.4336,0.8963]. Top3marginal69.4925% vs
memory71.8209%,delta-2.3284pp CI[-3.6418,-1.0448]. Fixed primary gate FAILS.

Conditional NLL is better (marginal2.133272 vs categorical2.333686/memory2.282992),
but this is NOT the user's correct-purchase criterion and cannot rescue failure.
Marginal rawNLL2.142141, mean retained catalogue mass.9915698, p05.9553137,
minimum.5366025; no hidden normalization. No calibration-error claim. No
canonical-control selection, epoch extension or follow-up859score is justified.

Source reportSHA704000a609c45f6623ee7ad5c13eb4ee9c4852dd35aecfbd353375afe92f7770.
Functional audit data/runs/multiset-probe-audit-20261001-v1/audit/report.json
PASSED, SHA c0f627d003093da43161167df193fb39006ba933dc9244f3b21eb864932420ff.
It verified120epochs/92040Adamsteps, memberships/initialhashes/referencearrays,
parameter counts, every metric/slice/pairedCI and resource/primary gates. It
replayed all10050neural predictions: zero top3 mismatches and EXACT0 error in
conditional logp,raw logp and catalogue logmass. Audit elapsed50.25s. No new
fitting in audit. All source/model/code digests remain valid.

Peak Torch CUDA allocation1012843008bytes; minimum sampled host availability
1893617664bytes >.75GiB floor. No resource failure. Both durable tasks and
workers are now terminal/removed. README and docs/multiset_result_v1.md updated.
The tested multiset recipe is CLOSED; mathematical correctness/probabilistic
score improvement is not a practical purchase improvement or breakthrough.
This does not disprove the entire model family, but future work needs a
different justified hypothesis and must preserve these consumed versions.

No new859-target evaluation, raw archive, reward/outcome, sealed payload,
source code/model/database edit or serving change. All1911confirmation games
remain closed. IndexSHA remains94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46.
Goal remains active and unachieved; there is no missing user annotation,
permission, account-usage or RAM blocker.

### 2026-10-01: omitted public scoreboard recovered without changing snapshots

Previous turn classified PROGRESS: completed and exactly replayed the negative
multiset result. This turn changes the information hypothesis; it does not
retune either failed method or reopen its859evaluation. Inspected source code
and found all-player cumulative KDA retained by _board_snapshot but dropped by
baseline._others and absent from M2's query/other-player tensors. Current live
adapter also omits other-player scores. Own PublicFeatureCompiler has focal
KDA only. Do NOT claim champions/levels/items/damage traits were missing.

Checked primary Riot Live Client API documentation: player scores expose
kills/deaths/assists. Direct base-URL fetch failed; the official indexed page
with query parameters supplied the schema. No third-party mirror used as
technical authority. No local live game/API was queried and no new collection.

Found an important design constraint in frozen footwear_reconstruction: no-buy
examples are timestamped death+15000ms after a90second no-purchase qualification
and use the earlier death snapshot. Exact death-age/recency features could
reveal that sampling rule. They are EXCLUDED. The new feature is only existing
cumulative KDA at the ORIGINAL stored snapshot, with no observation-time change.
Source purchase boards use start_kda (copied at visit start), not ending KDA.

Declared docs/scoreboard_information_protocol_v1.md before new payload access.
Added NEW source-owned export_winprob_scoreboard_v1.py and seven synthetic
tests; all pass. Ran required source status at10:27UTC, no source job active,
no served model bound, existing source changes preserved. The source publisher
ran as a durable CPU1 task and read only the4001 original FIT games via indexed
read-only immutable SQL. Every full indexed game/event digest matches its
previous certified FIT export. No raw archives, broad table scan or source
runtime/model/database modification. Labels/outcome metadata are necessarily
deserialized/hashed for identity but NEVER used/analyzed as feature values.

Source release completed in464.917s, full392448states/4001games, zero roster or
focal-score mismatches, one forbidden-field mutation check per game. Other-player
scores nonzero in363576states; column maxima K34/D21/A49. These are availability
facts, not conditional-information or accuracy evidence. Source manifest:
build_tracker/data/exports/winprob-scoreboard/scoreboard-export-20261001-v1/manifest.json
SHAfee70f9d7be28235e07b22b409471833bd4674d33a10abfc9939ae046fa2eeb2.

Added own scoreboard_features_v1.py and five tests (all pass), independent
source admission/episode/focal parity checking, and a target-free preparer.
Own cache data/runs/scoreboard-features-20261001-v1/cache/manifest.json completed
in26.0s, SHAc3a0f735523efac4eb040ca138f720196cb8a323194f3a29666119d9173ae786.
scores.npy contains K/D/A divided by20 with focal channel ZERO; permutations.npy
preserves complete KDA triples within allies1..4 and enemies5..9. The frozen
identity-hash control changes362823feature rows. No purchase-target/outcome array,
source runtime, raw source, test or confirmation payload was opened by preparer.

Declared docs/scoreboard_probe_protocol_v1.md before real fitting. Planned arms
zero/scoreboard/shuffled have identical copied encoder/head initialization and
a3->64 no-bias per-player score projection initialized to0 (192 extra parameters
in every arm). Same categorical loss,20epochs,shared epoch orders,2game folds.
Primary real-score accuracy must beat BOTH zero and frozen47.2836% memory by3pp
with positive paired lower bounds; top3 cannot fall. Additional positive lower
bound on NONEMPTY main purchases versus zero guards against synthetic no-buy/
ward-only-only gains. Real/shuffle comparison attributes individual assignment;
it cannot rescue a failed primary. No exact death-age/timing inputs.

NEXT: implement matched scoreboard model/tensor-store wrapper, unit-test zero
initialization and target independence, run complete resource preflight, then
train only if the fixed60minute budget passes. No scoreboard model has yet been
trained. Source export and own preparation tasks completed/removed; no job
remains active. Source data/feature/code versions consumed here must be preserved.
README and scoreboard_information_result_v1.md updated. All1911confirmation
games and external seals stay closed; index untouched; goal active/unachieved.

### 2026-10-01: matched scoreboard experiment implemented and training

Previous turn classified PROGRESS: completed source recovery/independent feature
preparation and fixed the next protocol. This turn implemented and started the
real six-model test, without changing its data, folds, objectives or gates.

Added winprob/scoreboard_probe_v1.py. ScoreCache revalidates the entire feature
lineage, all normalized arrays and all392448 deterministic control permutations.
ScoreStore supplies the same public state plus true/shuffled KDA; focal channel
is zero. ScoreEncoder adds the declared3->64 no-bias projection at the first
per-player preactivation using a scoped hook on the unchanged own encoder.
Hooks are removed even on exceptions; no source runtime import. Every control
has the same192extra parameters, initialized exactly0, and independent copied
encoder/head storage. Prediction takes no true target argument.

Eleven own model/feature tests PASS: exact zero-initialization/base parity,
no shared parameter storage, zero arm ignoring KDA, nonzero projection gradient,
hook cleanup, focal/NaN rejection, deterministic complete epoch orders, team-
local KDA permutations, and stronger-reference guard behavior. A synthetic
fixture initially tried assigning to an immutable PublicBoardBatch; corrected
the fixture with dataclasses.replace before any real run. No model/data change.

scripts/train_scoreboard_probe_v1.py implements the fixed20epoch/3arm/2fold
recipe, source/input hashes, all-real-data field validation, base-parity
preflight, full optimizer/epoch receipts, checkpoint-before-evaluation and
held-out-label mutation checks. Existing common writer/reference helpers are
reused from the own multiset runner; its failed scientific recipe is NOT run
or retuned. The scoreboard experiment has its own protocol/code/run artifacts.

Durable task `Videns Scoreboard Probe v1`, run
data/runs/scoreboard-probe-20261001-v1. It initially waited for its fixed3.25GiB
host floor, then started normally at2026-10-01T11:16:43UTC when that gate passed.
No threshold was lowered, no app/source job stopped and no new permission asked.
Wrapper PID19828, actual Python PID12596. ALWAYS recheck live handles/current
task before interpreting these notes; do not restart a live or pending run.

Full preflight PASSED,90synthetic updates and0real updates at that stage.
Initial prediction parity verified onGPU, state cache preloaded. Step means
zero0.010108s/scoreboard0.012995s/shuffled0.007902s; gather0.0005165s.
Inference batch256 means0.24860/0.23799/0.24394s (including complete probability
summary; NOT single-shop latency). Projected whole work1454.00s includes1.25
margin+180s overhead and is below80% of remaining3528.44s. Plan92040updates.

Last process-verified observation2026-10-01T11:27:33UTC: three fold0 checkpoints
completed; fold1 zero arm on epoch17/20. PID12596 alive, task Running. No
aggregate accuracy has been inspected; all six models must finish first.
The read-only NumPy warning concerns CUDA-copy row indexes used only for
gathering; no source array is written and no operational failure occurred.

Prepared scripts/audit_scoreboard_probe_v1.py and run/register_scoreboard_audit_job_v1.ps1.
They will verify complete features/permutations, initial hashes, training IDs,
epochs/Adam/settings, frozen references, every aggregate/slice/guard and replay
all10050 neural queries using the same batch256/store path. Audit launcher is
NOT registered or started; do so only AFTER the training task has completed
and unregistered, to avoid triggering its running-Videns-task concurrency guard.
Use a fresh audit version, e.g. scoreboard-probe-audit-20261001-v1.

NEXT: poll the existing task/process, inspect completed report, run the prepared
audit, and apply BOTH top1 primaries plus the nonempty-main-purchase guard.
No score, M2 improvement or breakthrough is claimed now. Consumed core/trainer/
protocol files must not be edited during this run. README updated; all seals
and the original index remain untouched. Goal active and unachieved.

### 2026-10-01: scoreboard experiment completed; exact audit confirms failure

Training finished at11:33:13UTC: all six20-epoch models,92040updates,980.846s.
On3350 fixed exposed whole-game OOF decisions, scoreboard1515correct45.2239%,
zero1500correct44.7761%, shuffled1497correct44.6866%, frozen memory1584correct
47.2836%. Real-zero+0.4478pp95%CI[-0.6567,+1.6418]; real-memory-2.0597pp
[-3.4634,-0.7164]. Real top3 is69.1642% versus memory71.8209%. Nonempty-main
guard3169games: real-zero+0.4733pp[-0.6937,+1.6944]. All primary gates fail.
No direct M2/859-game trial is justified by this result. Close the fixed recipe.

Registered original durable audit v1 after training completed. Its3.25GiB
startup gate initially waited. A2.5GiB launch-only adjustment was proposed in
scoreboard_audit_runtime_v2.md, but before any replacement/stop the controller
found the v1 Python worker had started at11:42:35UTC. It correctly ABORTED
replacement without mutation. No v2 launcher/run/controller-stop was created.
The user reported RAM freed ("libre"); v1 completed normally at11:45:58UTC,
launcher completed/unregistered11:46:05UTC. The proposal now says NOT APPLIED.

Independent audit passed: all120epochs/92040updates/settings/trainIDs, complete
score normalization/permutations, frozen references, every aggregate/slice/
paired interval and guard. Six-checkpoint replay of all10050 neural predictions
has0top3-cell mismatches and exactly0target-logp error. Audit190.698s.
Report SHA91286915a4ad16970031a9541f96efc214cbb867d6b7a3a03e60b24f4e342a17
at data/runs/scoreboard-probe-20261001-v1/experiment/report.json.
Audit SHA6a9b944137548beeab66404b7be8a56360218a6bcaccb37d0f2fde4d773907ef
at data/runs/scoreboard-probe-audit-20261001-v1/audit/report.json.

Training peak physical working set1670987776bytes, maxprivate3386388480,
minhostavailable1455337472 (>805306368floor), peakTorchCUDA961451520bytes.
Both own tasks/knownworkerPIDs are gone. Available hostRAM after completion
4146108KiB. No threshold was changed, no source task stopped, no failed recipe
retuned. Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
unchanged. New docs/scoreboard_probe_result_v1.md records evidence, limitations,
costs and closure. README updated; consumed code/protocol/artifacts preserved.

Goal remains ACTIVE/UNACHIEVED. All1911confirmation games and source seals
remain closed. WP/kernel remains shelved; observed retained purchase with
counts remains the correct target. This is negative evidence about the fixed
KDA learner, not proof that KDA is useless in every model. NEXT: select a
different justified mechanism after checking prior source experiments; do not
restart KDA, multiset, memory, subjective ratings or wallet-diagnostic loops.

### 2026-10-01: native-M2 recipe-transition correction implemented; selection frozen

Previous goal turn classified PROGRESS: completed/audited negative scoreboard
result, closed its fixed recipe. Goal still ACTIVE/UNACHIEVED. This turn chose
a different supervised mechanism after source/history review, without fitting
or tuning against the859development games or reopening failed recipes.

Source status11:52UTC: only Daily Pull Ready, no source training/benchmark,
no served deployment manifest. Reviewed native BasketNet/PrefixModel and econ
features: M2 already uses nonlinear basket interactions and recursive economics.
Its ten economic features contain consumed count/value but not consumed-stat
vectors. The own categorical probe is a different representation. Do not claim
M2 assumes item independence or that those older failures invalidate every
native-prior residual. Related primary ResLogit/TB-ResNet papers consulted;
neither establishes this particular net-descriptor formula or a League gain.

Implemented winprob/recipe_transition_v1.py:26context x26candidate descriptors,
676-parameter bilinear logit correction to frozen M2's original up-to-three
legal options, with an explicit OTHER category retaining all shortlist misses.
Likelihood is exact for the coarsened choices, convex plus positive ridge;
analytic gradient/Hessian-vector and deterministic bounded L-BFGS-B solver.
Zero coefficients preserve original order; target-free prediction only reorders
the original candidates. Native hypothetical after inventories yield purchased
attributes minus consumed-component attributes. These are10static descriptors,
not actual combat stats/passives or causal outcomes.11synthetic tests PASS.

Added new source probe scripts and hidden resource-gated one-shot launchers.
v1 reached native scoring but failed exact full-probability mutation equality;
preserved unmodified, no real data used. v2 added deterministic algorithms/
CUBLAS_WORKSPACE_CONFIG, same-input exact replay and direct parity of all six
model-input tensors. All v2 checks PASS. GPU nondeterminism is supported as the
explanation; responsible kernel was not isolated. No semantic test weakened.
Source v2 report SHAad1bf78aeec9929367a3f5e3bb364fb21f3f171ceb82f77c0a8d5b1a669dbe6c
at build_tracker/data/exports/winprob-recipe-transition/recipe-native-probe-20261001-v2/report.json.
16distinct synthetic states/112scored rows including repeats;48native receipts,
6fixtures,13.143s complete. Median16-row batch1.009661s,63.10ms/row; receipt
replay48options10.61ms; peakTorchCUDA164663296bytes. Synthetic cost only.

Own independent consumer audit verifies native receipt conservation and stat
parity, and benchmarks the CPU1 convex solver on8192synthetic rows (2028OTHER),
676parameters:9iterations,0.0631s,maxgradient9.23e-8. Both launcher tasks and
workers finished/unregistered. Own audit SHA6bd020579b7c9f714becef0774b55a4ce195d1c87c44c011989ad5a3b725f372
at data/runs/recipe-transition-synthetic-20261001-v1/audit/report.json.

Declared docs/recipe_transition_probe_protocol_v1.md BEFORE new real-data
predictions/fitting. Scope only3518original16.17FITgames, avoiding M2's6354
16.18TRAINgames; verified source train-ID digest matches package and overlap0.
Same original whole-game folds1761/1757;2900original focal evaluations1449/1451,
618original focal unavailability disclosed. At most16label-blind hashed states
per game including original focal when available. Raw/net/shuffled-net matched
controls, train-only game-weighted standardizers, ridge.01, six fixed fits.
Gate net-M2>=3pp with positive lower95% bound AND positive lower bound vsraw;
nonempty-main positive lower bound; original option sets/top3/legality invariant.
No real fitted correction exists. Passing only justifies a separately frozen
forward859development comparison. No seal is opened and no promotion implied.

Before freezing, discovered FIT exporter encode_record removes legacy
reference_compatibility. Corrected unconsumed protocol: recover role fields
ONLY in new source sidecar from indexed, read-only original16.17qualified games,
matching each full indexed game/event digest to previous certified receipt.
baseline._others uses board.team_position; focal uses event.team_position.
Preserve original prestate/order, tag legacy roles separately for native M2,
never residual features. No raw reconstruction or mixed dataset needed. Native
M2 scoring must run in its clean16.18process after the sidecar is published.

Added/ran metadata-only select_recipe_transition_v1.py. BaseCache's historical
metadata reader revalidates all reservations before arrays; no current labels,
rewards, source trajectories or DB payload reads. Reproduced original focal
selection from metadata. All3518games have16selected states=56288, fold28176/
28112. Selection manifest SHA979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6,
selection SHA97b7842e8811b3b4ed3baf4cc2918185c98061d8997589b52ba11459213b4191
in data/runs/recipe-transition-selection-20261001-v1/selection.
Consumed protocol SHAe87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e.
DO NOT edit these consumed code/protocol/artifacts; fixes need new versions.

NEXT: implement source-owned role-sidecar recovery/identity audit for exactly
the frozen selection, then source M2/transition release. Fixed200selected-state
real resource pilot must project the full56288states with1.5margin+900s inside
80% of remaining180min before full scoring. No target-dependent sample changes.
Then implement own train-only standardizers and fixed six convex fits, freeze
OOF predictions, evaluate all2900focals against SAME native M2, independently
audit. No such real source job is running now. All1911confirmation games and
source seals stay closed; index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
unchanged. Goal remains active; mathematical/operational feasibility is progress,
not the requested demonstrated purchase improvement.

### 2026-10-01: real recipe inputs verified; native M2 release v2 LIVE

Previous goal turn classified PROGRESS: convex core/synthetic native validation
and frozen metadata-only selection. Current source status12:25UTC showed only
Daily Pull Ready, no source training/benchmark, no served manifest. All source
changes already present were preserved; no frozen app/model/database edit.

Added NEW source export_recipe_roles_v1.py plus5synthetic tests (PASS). It reads
exactly3518original16.17FITgames using the existing indexed read-only/immutable
ApprovedGameReader. All3518whole indexed game/event digests match the original
certified receipts, roster/selected event identities agree, one forbidden-field
mutation per game leaves role projection unchanged. Recovers baseline semantics:
focal event.team_position and other board players' team_position, empty-string
fallback only as native baseline defines. These legacy fields are explicitly
reference-only, not certified live-safe or new residual inputs.

Role release completed340.566s, all56288states. Task Videns Recipe Roles v1
completed/unregistered12:34:55UTC; knownworker40148/wrapper33404 are gone.
Source artifact build_tracker/data/exports/winprob-recipe-transition/recipe-roles-20261001-v1/manifest.json
SHAb5537e3e4bb521e724212ae9787a8351c8233cd2bfcbe65a5128847e6a195805.
No original source trajectory payload, raw archive or whole-table scan. Indexed
purchase/outcome fields are necessarily deserialized/hashed for identity only.

Added NEW source prepare_recipe_contexts_v1.py plus4tests (PASS). Joins the
roles to the original certified FIT prestates by exact selected identity and
source_event_sha256; preserves original other-player order and copies all
other fields unchanged. Separate residual_public is only seven explicit keys;
all56288poison checks prove target/role/private-gold/future fields cannot change
that context. Streams28821original FIT episodes, verifies compressed and full
uncompressed hashes, writes selected actual multiset targets SEPARATELY.
Terminal metadata deserialized but never analyzed. No model was loaded/fitted.

Context release completed84.409s; task Videns Recipe Contexts v1 completed/
unregistered12:38:17UTC. Source artifact
build_tracker/data/exports/winprob-recipe-transition/recipe-contexts-20261001-v1/manifest.json
SHAf45e22753294b0d349c1c58bf6db4a23575352d5a9fc2ae5af1bf4d6b65bc25a.
Its contexts.jsonl.gz and targets.jsonl must stay separate. The native M2 scorer
does not open targets, original trajectories or databases, even for hashing.

Added own winprob/recipe_transition_probe_v1.py: equal-game/equal-within-game
row weighting, train-fold-only context/candidate standardizers, each present
candidate equally weighted within a row, fixed intercept, constant variance
fallback, raw/net/shuffled-net controls and exact identity-hash permutations.
Raw zeros only net coordinates AFTER shared standardization; shuffle preserves
the entire10-vector and never moves padding. Six tests PASS, including scaler
serialization without refitting on evaluation, same common block, explicit
intercept checking and no target/fold dependence of control permutation. No
actual real fit performed and no empirical gain inspected.

Added source export_recipe_m2_v1.py and3coarsening tests (PASS), new hidden
GPU/CPU1 launchers with3.25GiBhost/.75GiBlive/3072MiBGPU/startup guards and
180min worker/210min launcher bound (resource wait max30min). v1 FAILED during
warmup receipt packaging: the prior synthetic-only helper directly indexed
role_slot_inventory_unknown, an optional field absent from the certified FIT
prestate. Native purchase_sequence uses bool(row.get(...)), so this was an
adapter assumption, not a native recommendation/legality failure. No targets,
accuracy or new model fit were accessed. Preserved v1 code/run unchanged.

Added source recipe_native_receipts_v1.py using EXACT native optional-field
defaults for replay arguments, plus2tests (PASS). New export_recipe_m2_v2.py
retains the entire scientific recipe/selection/model/protocol/resource gate;
only calls the corrected packaging helper. Three native fixtures prove parity
with the old helper when fields are explicit and parity of omitted versus
native-false role-slot flag. All frozen apps/weights are untouched.

LIVE TASK: Videns Recipe M2 Release v2.
Own launcher logs: data/runs/recipe-m2-release-20261001-v2.
Source output: build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2.
Started12:46:42UTC; launcherPID14272,wrapper36584,actualPython36252. RECHECK
these handles before relying on this note; never restart on stale logs/timeout.
Last process-verified observation after94.64s:1028/56288native rows written,
worker36252 alive (~1.44GBworking set). No complete release manifest yet.

Real200-state pilot PASSED. Full native M2 probabilities/options and all six
actual input tensors are exactly invariant to ignored-label/future mutation
for all200. Mean end-to-end native scoring+receipt cost0.05544403s/row.
Projected whole56288state work5581.250s (93.02min, includes1.5margin+900s)
<80% of10763.108remaining seconds. Two16row warmups omitted from timed mean;
pilot predictions reused unchanged. Scoring is deterministic CUDA, batch16,
frozen native16.18 M2, no targets/database/raw/trajectory reads. It publishes
only source recommendations, four coarsened prior masses, receipts and public
residual inputs; no correction or purchase-accuracy result exists yet.

NEXT while the SAME live native release proceeds: implement own artifact-only
consumer (verify native static/group/selection identity and receipt conservation),
then the fixed six CPU convex fits using training folds/standardizers only,
freeze models before OOF predictions, evaluate all2900focals against the SAME
native M2 and controls, and independently audit. Only start real fitting after
the complete source manifest and its task termination are verified. All1911
confirmation games/seals stay closed; old failed research recipes remain closed.
Do not confuse source M2 inference with training the new676parameter correction.
Goal ACTIVE/UNACHIEVED. No promotion or demonstrated practical gain yet.

### 2026-10-01: own recipe consumer/fits/evaluation/audit ready; native export remains live

Previous goal turn classified PROGRESS: role/context releases completed and
native M2 v2 started after its real200-state pilot. This turn did independent
implementation while polling the SAME verified-live process. No restart.

Added winprob/recipe_transition_data_v1.py. Admission/reservations and frozen
selection are checked before payloads. Native source schema/package/groups/
statics are bound to the existing full catalogue and own static release. It
independently compares each native query's public and reference-projection
digests with the context release, checks original options/counts/wallet and
every hypothetical receipt, then creates float64 inputs and fixed permutations.
All56288rows/3518games/2900focals and original fold counts must survive. Actual
targets open only after a durable inputs_lock matches every feature digest.
In-memory arrays are rehashed as well as files. Five synthetic consumer tests
PASS, including OTHER rather than replacement, counted purchases, same wallet,
ward-only versus full SAVE and no source-runtime imports.

Added winprob/recipe_transition_evaluation_v1.py. Prediction accepts no targets;
scoring is separate. Stable log probabilities preserve valid NLL even when an
exponentiated probability underflows. Exact original candidate membership and
top3 coverage are mandatory. Reports main/full/typed first exactness, coarsened
NLL, same2000whole-game bootstrap draws, corrected/broken, nonempty guard and
descriptive slices. Gate is EXACTLY frozen: net-M2>=3pp+positive lower95%,
positive lower95%vsraw and nonempty-main positive lower95%. No secondary rescue.
Five tests PASS for gate, stronger-control requirement, membership, ward semantics
and extreme log probabilities. Original eleven numerical and six control tests
also pass; all27 ran together successfully.

Added scripts/train_recipe_transition_v1.py (not run). It validates/freeze inputs,
fits only training-fold standardizers, verifies zero prior/order parity and
benchmarks actual geometry with synthetic category0 targets BEFORE real labels.
Runtime estimate is fixed in docs/recipe_fit_runtime_v1.md:1.5*max gradient time
*300*8*6+180s, must fit80% of remaining1800s. Eight line-search evaluations is
an operational proxy, not a worst-case proof; unchanged maxls50 and hard guard
remain. No data/recipe/solver/gate change. Then exactly six676-parameter fits,
lambda.01, max300; success and maxgradient<1e-6 required. All models/scalers
freeze before held-out prediction; predictions freeze before metric evaluation.
No real coefficient fit, target read by this consumer, or score exists yet.

Added scripts/audit_recipe_transition_v1.py (not run). It independently computes
weighted train-only scaler means/stds, checks checkpoint train IDs and population,
recomputes each saved convex objective/stationarity and every summary/interval/
gate, and requires exact probability/logp/order replay of8700correction queries
plus2900native baseline queries. It reuses source receipts and does NOT rerun
all56288native M2 neural forwards; scope explicitly disclosed. Two additional
audit-helper tests PASS (29own synthetic checks total). Python syntax checks pass.

Prepared run/register_recipe_fit_job_v1.ps1 and run/register_recipe_audit_job_v1.ps1;
PowerShell syntax checks pass. Neither task is registered or running. Fit source
is the current native v2 release; planned fit run name is
recipe-transition-fit-20261001-v1, output experiment/. Audit launcher points
to that exact experiment; use a new output such as recipe-transition-audit-20261001-v1.
Registration REFUSES if another protected task is running. Do NOT register a
waiting fit/audit alongside M2: the source concurrency guard would see it and
stop its own worker. Verify source completion AND task/process termination first.

LIVE native task unchanged: Videns Recipe M2 Release v2; worker36252 under
wrapper36584 (launcher14272). Last handle-verified observation:49166/56288rows,
elapsed2871.574s, worker alive (~331MBworking set), task Running. Only existing
PyTorch nested-tensor warning in stderr; no complete native manifest yet.
Own logs data/runs/recipe-m2-release-20261001-v2; source publication
build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2.
Never infer completion or restart from a stale note/observation timeout.

NEXT: wait/poll the SAME native handle to complete, inspect its manifest and
resource accounting, then register the prepared fit after its source task has
unregistered. Fix any execution-only defect with a new version after consumption;
do not modify native v2 or the frozen scientific protocol. Once all six fits and
scores complete, run the prepared exact auditor, report measured gate outcome,
and only consider the predeclared forward859trial if the internal gate passes.
Index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
and protocol SHAe87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e
verified unchanged. All1911confirmation games/source seals stay closed. No policy
promotion or demonstrated gain; goal ACTIVE/UNACHIEVED.

### 2026-10-03: native-M2 recipe experiment finished and audited; no useful gain

Continuation classification: PROGRESS. The original native M2 release completed
on October1, and the own consumer/optimization issues have now been resolved and
the complete fixed comparison audited. This entry supersedes the stale live-job
notes above. No recipe worker or scheduled task is currently active.

Native M2 completed56288states in3286.4166s (54.77min), peakTorchCUDA164065792B.
Its task unregistered at2026-10-01T13:41:35UTC. Source manifest:
build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2/manifest.json
SHAf0bf5e91a06a04151654c2293364bfe47e46d8b9acbfe96eb96822c3020c9abe.
Actual targets were never opened by that native scoring stage.

Own fit v1/v2 completed input validation and164731native receipt checks but
stopped before target loading because their operational timing projections
failed. v1 projected1573.386s against80% of1735.375remaining under30min. v2 raised
only the execution cap to45min and projected2466.732s against80% of2636.492.
Its maximum single measurement0.105867s was above its eleven other measurements
0.036425-0.056784s; source of timing variation was not isolated. v3 fixes the
estimator using two warmups plus20consecutive passes/arm/fold and the maximum
of six block means, preserving the original formula and80% gate. Runtime notes
recipe_fit_runtime_v2.md/v3.md record these nonstatistical changes. v3 passed
preflight (projected1150.815s), then failed its first raw-control precision gate.
All v1/v2/v3 inputs reproduce the same locked feature digests; scientific data,
features, lambda, controls, splits and success thresholds never changed.

On October1, the attempted diagnostic file creation/execution was NOT executed:
automatic approval review reported a usage limit and could not complete. No
unsafe-action determination was issued. On October3 the normal reviewed tools
worked again; read-only revalidation found no live process and no diagnostic
files, so the diagnostic was then actually created and run. Source status at
14:34UTC showed Daily Pull Ready and no competing source job. Source collector
API-key failure is unrelated; no new collection or key edit was needed here.

Diagnostic recipe-solver-diagnostic-20261003-v1 reproduced the first raw/fold0
training solve:28112states,79iterations, success by relative-loss tolerance,
objective0.8797179570121145, maxgradient2.115549009e-6 (>required1e-6).
Three directional derivative checks agree within6.512e-11. No held-out
predictions/accuracy were computed. Diagnostic SHA
7ab1062c0a2c892cab4be9a16583e14a5e5f3f5122a99bac76e71fbcde47a284.
Unqualified diagnostic coefficients are preserved and never warm-started a fit.

Declared recipe_solver_precision_v2.md before any evaluation. New own wrapper
recipe_transition_solver_v2.py changes ONLY ftol1e-12 to0; L-BFGS-B, objective,
zero initialization, gtol1e-7,maxiter300,maxls50 andlambda.01 stay fixed. This
refines the same unique strictly convex optimum, not a new statistical model.
Two numerical tests PASS (31own tests total). New train_recipe_transition_v4.py
also persists every solver receipt before checking the convergence gate, so
another numerical failure cannot discard its diagnostic. Old code preserved.

Actual completed run: data/runs/recipe-transition-fit-20261003-v4/experiment.
All six fits converged on projected gradient:88-125iterations,619total,671function
evaluations. Max gradients6.057e-8 to8.422e-8. Optimization25.5367s in total;
whole input/preflight/fit/evaluation125.3802s. All coefficients/scalers freeze
before OOF prediction; prediction artifact freezes before metric inspection.
Final report SHA0ed980c46e99c31f342b983935e99fc4feb7492242374410302b8583cac2c2d8.
Fit task completed/unregistered2026-10-03T14:48:25UTC.

PRIMARY RESULT on2900fixed distinct-game internal focal decisions:
M21483/2900=51.1379%;net1481=51.0690%;raw1489=51.3448%;shuffled_net1475=50.8621%.
Net-M2-0.0690pp,95%CI[-0.6552,+0.5172],37corrected39broken. Net-raw-0.2759pp
[-0.6897,+0.1724]. Net-shuffle+0.2069pp[-0.3103,+0.7241]. Nonempty-main2736games:
exactly0pp[-0.5900,+0.5883]. Empty-main population119ward-only+45no-buy-death.
All arms retain top31959/2900=67.5517%. Full first net1104vsM21106. Coarsened
NLLnet.887774 versusM2.883664; delta+.004111[-.000750,+.008935]. No useful signal.
DO NOT confuse this population with859ordinary16.19 development or compare its
coarsened NLL to earlier full-catalogue NLL figures. Internal screen is exposed,
retrospective16.17, with native16.18 M2 trained on later16.18 games.

Independent audit code audit_recipe_transition_v2.py additionally verifies the
stronger ftol0 settings. Completed74.118s; all six training-only scalers,
train IDs, objectives and gradients verify; max independent scaler discrepancy
1.708633e-13. Exact replay of ALL8700correction predictions plus2900native
baseline queries, including probabilities/logp/order. Every metric/paired
interval/slice/guard and the FALSE gate reproduced. It does not rerun every
source M2 neural call; that limitation is explicit. Audit artifact
data/runs/recipe-transition-audit-20261003-v4/audit/report.json,
SHAa629f732b26d96ed7b98fbfd7a5746156fbf750d69e21708ecb774c1244e33a7.
Audit task completed/unregistered14:53:58UTC. All recipe processes/tasks absent
on final verification; unrelated services preserved.

Decision: CLOSE this fixed net-recipe correction. No lambda/feature/solver
accuracy sweep, no selection of the raw control after failed primary, no859
trial and no promotion. This failure does not rule out every representation
or additional information source. No causal/win/live benefit claimed.
New docs/recipe_transition_result_v1.md is the complete result; README and
feasibility status updated. All consumed code/protocol/model/artifacts preserved.
Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
and scientific protocol SHAe87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e
verified unchanged. All1911confirmation games/global/source seals stay closed.

The goal remains ACTIVE/UNACHIEVED. NEXT: choose a distinct, evidence-grounded
information or decision-representation hypothesis after reviewing prior studies.
Do not revive the closed recipe through secondary metrics, small parameter
changes, subjective ratings, WP kernels or another wallet-diagnostic loop.
No next hypothesis has been selected or implemented in this turn.

### 2026-10-03: personal kill-graph information recovered; next test fixed

Previous turn classified PROGRESS: completed/audited the negative net-recipe
comparison and closed it. This turn selected NEW relational information, not
retuning that recipe. No personal-graph model has yet been fitted/evaluated.

Verified in source that native M2 lacks directed killer-victim edges. Different
histories can share ALL players' K/D/A but differ in who killed the buyer.
Official Riot docs and linked sample events verify ChampionKill/KillerName/
VictimName/Assisters fields. Direct raw sample URL initially failed through
web open; clicking the official docs links succeeded. Read-only inspection of
live_overlay contract/tracker confirms observed untagged/hidden-name aliases
and honest ambiguity handling. No overlay code/data imported, live capture or
current game API queried. Do not call kills a measure of damage or pressure.

Mathematical feature: enemy killer-victim counts C[j,i] minus the degree-null
expectation d[i]*K[j]/T, shrunk by d[i]+5. Integer numerator T*C-d*K sums0.
Aggregate against17public enemy profile coordinates. If no personal excess,
the proposed edge-only residual must stay exactly M2. Null is labelled-stub
matching separately by team direction, preserving every player's kills and
champion deaths. Environment deaths and original assists remain unchanged.
This does not synthesize a temporally valid alternate match or causal effect.
Read Karrer/Newman arXiv1008.3926 as primary motivation for degree correction;
no claim their community-detection results prove a purchase improvement here.

Critical cutoff finding: frozen reconstruction processes ORIGINAL frame/event
order, not a new timestamp sort. Each valid CHAMPION_KILL increments one death,
so all-player death sum uniquely selects its stored KDA prefix. Every selected
state must match all10K/D/A triples and all included timestamps must<=its
decision timestamp. This preserves the earlier death snapshot of synthetic
no-buy rows, even when additional kills happen before death+15s. No recency,
exact death age or timestamps are predictor features. Assists are parity only.

Declared docs/personal_kill_graph_information_v1.md BEFORE new raw payloads.
Reuses EXACT original3518FIT16.17games/56288states/2900focals, same selection
SHA979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6.
No new selection, no859read, seals untouched. New source-owned exporter reads
only already certified timeline envelopes (not match payloads/SQL), uses old
scoreboard sidecar and public roster from prior role receipts, and publishes
sanitized needed kill prefixes plus count/KDA/roster/audit-time arrays.

Own personal_kill_graph_v1.py passed8synthetic checks; independent source v1
passed4. Source status16:02UTC had no active source job. Its new task initially
waited for3.25GiB, then STARTED normally16:11:38UTC. It did NOT remain waiting:
it completed812games and failed an extra assist validation. A scoped diagnostic
on the next certified FIT timeline (not a broad scan) found kill48 of needed52
at1304856ms: killer10,victim5,assisters[2,3,5,6,7,9]. Victim/same-team assist
metadata are valid IDs but violate assumptions we had added; frozen source
reconstruction simply counts each listed valid ID.

Preserved v1 partial run/code. information_v2.md records the correction:
retain exact raw assist counter semantics (including multiplicity if present),
keep killer/victim opposition, ID and temporal guards, and never use assists as
model graph edges. No source engine/data changed. The CPU-only extraction/audit
startup gate is2.5GiB (live.75 unchanged), rather than inherited GPU3.25. No live
worker was cancelled; v1 was terminal. Own personal_kill_graph_v2.py9tests and
source export_personal_kill_graph_v2.py5tests PASS.

Source release completed127.754s: all3518games/56288states,170648needed kill
events,222411631compressed timeline bytes. All exact KDA/temporal checks pass,
one future-suffix mutation per game passes. Two raw events list victim/same-team
assisters;0duplicate/0killer-listed. These lists are not certified physically
correct; graph uses only killers/victims. Tasks source v2 and info audit v2
completed/unregistered, no graph process remains. Source manifest SHA
771d1888d270285a4651ff0dbdec8f375acdd54c4ad6828d1dfd2b910b52eed8 at
build_tracker/data/exports/winprob-personal-kill-graph/personal-kill-graph-20261003-v2/manifest.json.
Information protocol v1 SHA3b622f042af6027c7d33fd54e0a45435e70dfd505eceb8aa513f87e5514b95fa;
v2 amendment SHA03d345a03a51d3d500fe88fedc7b47820b36cf44e16b1b88cb6edd2e435ae7ed.

Own independent audit_personal_kill_graph_v2.py replays ALL56288counts from
sanitized source event ledgers and compares KDA to ORIGINAL old scoreboards,
rosters to original role artifacts. No raw, source runtime or target file read.
Every null row/column degree verified exactly. Nonzero real integer edge excess
41608states,2094/2900focals. Null changes48201whole graphs and1766focal incoming
vectors. Loose outcome-blind change-capacity bound72.2069%; passes necessary
87case/3pp minimum, but is NOT an expected gain or purchase-accuracy result.
Audit37.535s, report SHA490ba260928feec51cc79596e087dd552edf446452513e1befd14101e3c0a6bd
at data/runs/personal-kill-graph-audit-20261003-v2/audit/report.json.
null_counts.npy SHA2786f6cac428fe78565b5ec30dbc2b810135a5cb7792d7364dfa7fec203814ea;
active_excess.npy SHA74d0620d464486be81301cb6f5fe8685018fda72b3a095ee5a79c0a4833631ea.

Declared personal_kill_graph_probe_protocol_v1.md AFTER availability but BEFORE
new target analysis/fitting. SHA cdd3c2f163f10b6862eae9737a130a96258768629d9a31eefca070d93e663d91.
Selected final design is EDGE-ONLY incoming kills (not assists, outgoing edges,
GNN, time decay or another net-stat recipe): g17excess profile x f16existing
common/raw native candidate descriptors=272parameters. No intercept/centering;
training-fold-only RMS, g independently per arm and same f scales. Two arms
personal/degree_null, two folds,fourfits. Same exact coarsened M2+OTHER likelihood,
lambda.01 and precision-v2 L-BFGS-B300/gtol1e-7/ftol0/maxls50. All states retained.
Personal-M2>=3pp+positive lowerCI, personal-null positive lowerCI, nonempty
personal-M2 positive lowerCI, exact options/top3 and zero-profile M2 parity.
New bootstrap seed20261003. No next859trial unless this fixed gate passes.

Added own personal_graph_models_v1.py with public profile join (six champion
tags,10scaled owned-item stats,level), RMS-only scaling and exact reference
mass/order override for zero-g rows or zero coefficients. Three model tests
PASS: no forbidden field use/mirror-champion confusion, multiplicities,
RMS zero preservation and exact zero-information M2 parity. Model prediction
accepts no target. No real learner/accuracy work has run for this hypothesis.

NEXT: implement GraphData joining the verified graph/control artifacts to the
existing native M2 priors/candidates and context public enemy profiles (read
only champion_id/team_id/items/level; ignore legacy roles/other gold). Can reuse
own RecipeData solely as a validated artifact reader and its first16candidate
features; do NOT refit/rescue the closed net-recipe. Compile g using v2 graph
code; bind every input; count actual nonzero projected focal profiles before
targets (must>=87). Then implement fixed4fit/evaluation pipeline and independent
audit, with frozen models/predictions before scoring, CPU1/2.5GiB/.75live/45min
worker and sustained preflight1.5*maxmean*300*8*4+180 <=80%remaining. No job is
currently registered/running. Consumed source/protocol/graph artifacts must not
be edited; changes after use get a new version. Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
verified unchanged, all1911confirmation games/source seals stay closed. Goal
ACTIVE/UNACHIEVED; relationship availability is not the requested breakthrough.

### 2026-10-03: personal kill-graph prediction test completed/audited; gate failed

Previous turn classified PROGRESS: recovered/audited graph information and
fixed the next comparison. This turn implemented and executed the fixed test
without changing its states, folds, features, solver settings or metric gates.

Added personal_graph_data_v1.py. Uses RecipeData only to validate/reuse original
M2 probabilities/candidates/common-raw descriptors, never to refit the closed
net recipe. Binds source graph771d1888... and own info audit490ba260..., checks
every fixed degree-null graph against its original hash seed, joins only public
champion/team/items/level to verified participant rosters, and compiles real/null
g17profiles. Keeps f16, all56288states and original2900focals. Validated graph
feature digests freeze before target access. Source runtime/raw/seals untouched.
Actual nonzero personal profiles41608states/2094focals; zero personal14680/806,
zero null14734/809. Necessary capacity gate passes; this was not a gain claim.

Added personal_graph_evaluation_v1.py and four tests (PASS), run with the three
model tests (7PASS). Checks actual option membership, all zero-profile order
parity, unchanged top3, correct ward-only/full/typed semantics, same2000paired
game resamples(seed20261003), corrected/broken and nonempty guard. The primary
cannot be rescued by a slice or degree-null control. Python syntax checks pass.

Added/ran train_personal_graph_v1.py through the hidden CPU1/RAM-gated task
Videns Personal Graph Fit v1. All training-only RMS scales freeze before labels;
candidate scales equal between arms, context scales independently fitted on
each training fold. Sustained preflight PASSED: projected726.917s. All four
272parameter models completed from0, lambda.01,L-BFGS-B300/gtol1e-7/ftol0/maxls50.
Iterations58/60/60/60 (238total),266function evaluations. Gradients6.298e-8 to
9.035e-8. Four optimizations8.5964s; complete input/profile/preflight/fit/scoring
225.1634s. Models freeze before prediction; predictions before scoring. All
zero-g rows retain exact native probabilities/logp/order, including the
806personal and809null focal rows. No training/checkpoint/hyperparameter sweep.

RESULT on2900same fixed distinct-game focals:
M21483/2900=51.1379%;personal1491=51.4138%;degree_null1484=51.1724%.
Personal-M2+0.2759pp,95%CI[-0.1034,+0.6552],19corrected11broken.
Personal-null+0.2414pp[-0.2069,+0.6897],26corrected19broken.
Nonempty2736: +0.3289pp[-0.0366,+0.6978]. All useful-signal requirements fail.
Top3 unchanged1959/2900=67.5517%. Full first1112vs1106 (+0.2069pp,
CI[-0.1379,+0.5517]). Coarsened NLLpersonal.886984493, null.886870649,
M2.883663520; personal-M2+.003320973[+.000559899,+.006339686] is WORSE.
No full-catalogue NLL, causal/live or forward-confirmation claim.

Fit report data/runs/personal-graph-fit-20261003-v1/experiment/report.json,
SHA949e5aba479d79c7aa017a8b283836d4753dd626252a874a00a6d2cf8809fe6a.
Task completed/unregistered17:04:16UTC. Four saved models retained unchanged.

Added/ran audit_personal_graph_v1.py. Independently reconstructs all graph
profiles and training-only RMS (maximum discrepancy5.134781e-16); verifies
training identities, scaler-before-target lock, fixed numerical settings and
all four objective/gradient receipts. EXACT replay of5800correction queries
plus2900native baselines, probabilities/logp/orders, zero-profile overrides,
option sets, every metric/interval/slice/guard and FALSE gate. Audit186.1277s.
It reuses bound native M2/graph artifacts rather than redoing all source neural
forwards or raw recovery; that scope is explicit. Audit report
data/runs/personal-graph-audit-20261003-v1/audit/report.json,
SHAadc8f708b52eb4aea2f29f35f364a3a513de5c96cd12f9f287219362563e5472.
Task completed/unregistered17:07:52UTC; no new model is promoted.

Decision: CLOSE this fixed incoming-kill/excess-profile/linear correction.
No new859trial, alternative shuffle, shrinkage/window/lambda search, extra
epochs or null-control selection after the failed primary. Exact feature
availability does not establish predictive benefit. The small unresolved
eight-hit difference is not the requested breakthrough. This does not prove
all personal interactions or representations are useless.

New docs/personal_kill_graph_result_v1.md records the complete result and limits;
README and information-stage status updated. A multi-file documentation patch
initially failed exact-context validation; verified that no partial file/edit
was applied, then applied the corrected patch. No scientific artifact changed.
Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
and protocol SHAcdd3c2f163f10b6862eae9737a130a96258768629d9a31eefca070d93e663d91
verified unchanged. All1911confirmation games/global/source seals stay closed.
Goal ACTIVE/UNACHIEVED. NEXT: select a substantially different justified route,
not another small rescue of the now-closed M2 correction or graph recipes.
No next hypothesis has been selected, predeclared, or run here.

## 2026-10-03 — route-marginal mathematics, public census and full-prior retention

Continued under the user's observed-purchase objective and resource availability.
No graph/net-recipe rescue, subjective ratings, WP/kernel revival or sealed reads.

Added docs/route_marginal_design_v1.md and winprob/route_marginal_v1.py.
For row-stochastic recipe attribution R, J0(a,z)=p0(a)R(z|a), m0=p0R,
J1=J0 alpha/m0 and p1=p0*(R(alpha/m0)). This is the minimum-KL supported
joint with marginal alpha; neutral alpha=m0 preserves p0 bit-exactly and
KL(p1||p0)<=KL(alpha||m0). Identical routing rows preserve action odds.
These properties do NOT validate an estimated alpha as fresh information.
The closed future-goal auxiliary did not feed goal probabilities to inference;
its strictly-later completion labels are neither private intent nor causal inputs.

RecipeFrontiers matches counted multisets to public recipe antichains without
counting an ancestor and descendant in the same branch. Duplicate recipe nodes
retain separate capacities. Native legality, wallet, current inventory, footwear
transforms and passives are outside this descriptive compatibility calculation.
Eight initial mathematical/recipe tests PASS.

Public census data/runs/route-static-census-20261003-v1/census/report.json
SHA08b426d97260521b52c9271c235f705a60c3b899ac9d449fddf67a1a8c951def:
9333baskets,258forests,197reachable recipe nodes,2407914matrix payload bytes;
9274baskets match at least one node,2520fully match some forest,59match none.
5874exact compatibility-row equivalence classes; largest59 includes empty and
non-upgradeable purchases. These are CATALOGUE counts, not purchase frequencies.
Old major-item ontology excludes boots and is not a complete route ontology.
Runtime28.8628s; synthetic full-catalogue bridge6.17-7.14ms. No real policy
kernel, route predictor or accuracy selected. Consumed files remain immutable,
including a harmless old design-text typo '851/859development'; actual boundary
was no development/game payload read.

Added winprob/route_capacity_v1.py and eight more tests (16total PASS0.108s).
The bridge is a convex mixture of E(a,z)=p0(a)R(z|a)/m0(z). Pairwise max/min
margins equal expert-column extrema, but independent pairwise wins do NOT imply
simultaneous top1 feasibility (counterexample tested). Maximum KL and TV over
the family are attained at a vertex. Unsupported routes are excluded.
Implemented alpha proportional to m0*exp(x theta), exactly neutral at zero,
and relative observed-purchase NLL logsum(m0 expu)-logsum(R(y)expu), gradient
x^T(alpha-beta_y). Verified against full bridge NLL and finite differences.
This objective is NONCONVEX, not the earlier convex net-recipe objective. Actual
NLL also needs the constant -log p0(y); no current purchase/future target read
or actual fit happened in this stage. A real kernel/input/control/fit protocol
is still required. No future-goal/private-plan prediction claim is made.

Public independent audit data/runs/route-static-audit-20261003-v1/audit/report.json
SHAf1e6d16d642cede7238d0a3b142de42a2a29b7d1b1732dab212325e539cc014f,
complete0.4291s. All matrix summary/equivalence counts recomputed;768distinct
recipe/basket pairs verified via independently unfolded node occurrences and
antichain assignment, not the original capped-submultiset algorithm. This is
NOT an independent recomputation of all2407914entries. With SYNTHETIC uniform
p0 and softmax(fraction) routing, max attainable KL=.0331055783 and TV=.108121174;
max action amplification2.98512134 and largest routing pair ratio5.03821817.
These are exact hypothetical-family bounds, NOT native-M2 gain estimates.
Both public tasks completed/unregistered; no games or model payloads opened.

Next concrete input need: the old native release saved full-row hashes but only
three option probabilities plus OTHER. Those four masses do not determine
m0=p0R. New docs/route_prior_export_protocol_v1.md
SHAf76ffa31a92ea3d3d8d0a563366dee03478a1184050cda703f83cf326502d2cb
fixes an INPUT-ONLY full-probability replay, not a route policy trial.

Ran source scripts/status.py at18:25UTC: no active source task, daily pull Ready,
no served deployment manifest. Existing API-key failure unrelated and untouched.
Created NEW build_tracker/scripts/export_route_prior_v1.py from the frozen v2
exporter; old file SHA3c75ac5f5b09f1af54e7a13501468fdde05d862efa793acba297bad314da631f
unchanged. New code SHA3cd6dc666c4ff5e1f537a4b92bd7523326ba1461b4c469a70225a321b1403f2e.
Admission now also precedes hashing the context payload, not just parsing it.
Retains exact old200pilot/remaining batch16 sequence, original native options,
receipts and model inputs; checks every full9333-float32 row hash and old option/
coarsened/projection record. Disk-backed[56288,9333] array payload2101343616bytes,
flush by1024outputs; rechecks every saved row before marking complete.
Source tests3PASS: tail changes invisible to coarsening detected, exact ordinal
mmap roundtrip and swapped rows detected, dtype/action-axis rejection.

Started hidden task Videns Route Prior Release v1 at18:31:29UTC, workerPID30664.
Version route-prior-20261003-v1, logs winprob/data/runs/route-prior-20261003-v1;
output build_tracker/data/exports/winprob-route-prior/route-prior-20261003-v1.
Latest observed launcher RUNNING; no completion claim. CPU1, idle GPU3072MiB,
startup2.5GiB RAM/live.75GiB,5GiB disk guard, no League/competing source jobs;
180minute worker with200state cost gate. The prior same inference took3286s;
new ETA is not measured until its pilot finishes. New startup threshold explicitly
uses streaming/disk-backed storage, not loading2.1GB into host RAM. Never start
another protected Videns task while it runs; do not restart from stale logs.

Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
verified unchanged. No new route fit/accuracy or improvement claimed. Goal stays
ACTIVE/UNACHIEVED. Next: inspect actual prior release state, verify exact replay;
then fix a full-catalogue purchase-supervised route comparison and matched
control before any new fitting/scoring. All859development/1911confirmation and
global/source seals remain closed in this stage.

Route-prior live verification at approximately18:33UTC: pilot200/200 PASSED,
including exact prior hashes and forbidden-field mutation parity. Mean native
seconds/row.05751816; guarded whole-release projection5756.3733s (includes
1.5margin+900s, not a promised duration). Full export observed1028/56288states
at113.0775s. Launcher/worker30664 remained running; only standard PyTorch
nested-tensor warning, no exception. Available host RAM1342580KiB (~1.28GiB),
above the declared.75GiBfloor. Do not launch competing heavy work. This input
stage has still performed no training or accuracy scoring. Inspect live task,
children and final manifest before any retry or downstream artifact use.

## 2026-10-03 — full-catalogue route comparison declared and implemented

Previous goal turn: PROGRESS (verified public geometry, exact bridge/capacity
code and source prior launch). This turn makes independent implementation
progress while verifying the SAME live source handle; no restart/duplicate.

Declared docs/route_probe_protocol_v1.md, current SHA
f6e0a8de1fdaffa76dc7775457cb1f618058c384858eafe71997cb4d5d23ec18.
It was clarified BEFORE any real route marginal/fit/target read to use original
CUDA topk during source decoding, while route fitting remains CPU1. Initial
draft SHAebc6b4d... was never used for a scientific run; all executable pins now
usef6e0a8de.... No already consumed scientific artifact or source export changed.

Fixed real kernel softmax(fraction_units/60), temperature1,258routes/9333actions.
Control permutes complete action ROWS within purchased-copy-count bins0..6,
seed20261003, preserving each bin's row multiset and empty group. A route-column
permutation would merely rename the latent axis and is explicitly rejected.
Inputs reuse exactly26role-free current context features (intercept, time,
level,wallet,10owned-item attribute sums,6own champion tags,6enemy tag fractions).
Each fold's scaler uses training-only equal-game moments, fixed intercept.
Two arms x two folds,6708parameters each, same original3518FITgames/56288states,
2900original distinct-game focals, no population/label resampling.

Alpha=m0 exp(x theta)/sum(m0 exp(x theta)), m0=p0R with FULL native prior.
Observed-current-purchase relative NLL plus(.01/258)*||theta||^2; actual NLL
also needs-log p0(y). Four zero-initialized L-BFGS-B fits,maxiter500,gtol1e-7,
ftol0,maxls50, gradient<1e-6 and success required before any heldout score.
Objective is NONCONVEX, no global optimum or uniqueness claim. All selected
targets must have positive original native probability; no epsilon/drop rescue.
No future-completion labels, private-intent targets, graph features or memory
recipe are used. This is explicit structured purchase prediction, not WPA.
The existing human direction supersedes the old WP-only objective wording.

New modules:
- winprob/route_probe_v1.py: deterministic kernels/count-preserving control,
  training-only scaler, exact batched objective/gradient, bounded optimizer,
  target-free full-catalogue prediction and neutral/support checks.
- winprob/route_probe_data_v1.py: completed prior verification, existing
  RecipeData only for immutable inputs/admission, full-vs-coarsened parity,
  disk-backed full-route marginals, durable input lock before target ledger.
- scripts/fit_route_probe_v1.py: input/cache/preflight/scaler locks, four models
  frozen before predictions, all9333-class outputs on2900focals and source
  exchange manifest. No scoring occurs in this worker. Real fit NOT RUN yet.
- scripts/run_route_probe_fit_job_v1.ps1 and register_route_probe_fit_job_v1.ps1
  prepared/parser-checked; NOT registered. They refuse a competing protected
  task. Do not launch while the native prior process remains live.

Source NEW adapter build_tracker/scripts/decode_route_probe_v1.py implements
unchanged MixturePolicy.options with supplied probabilities and originalCUDA
float32 baseline/topk. Corrected rows float64; exactly neutral rows use original
float32 to preserve native ties (verified against normalized prior). Full old
M2 probabilities and native options must replay. Forbidden-field mutations
checked on every query/arm. Separate audit17 process uses the EXACT
reconstructible_1617_runtime and reviewed16.17statics, NOT the older
footwear_runtime: original trajectory runtime is reconstructible-16.17-v1.
Static review SHAafe5bb33d4922bf0395737f598d191640e215e889d99469868d45f88a37fb279.
No new native activation/actual decoding has run; source adapter currently
unit-tested only. No decoder launcher prepared yet.

New winprob/route_probe_evaluation_v1.py implements paired displayed main first,
top3, full/typed first, nonempty guard, full-catalogue NLL, changed option sets,
corrected/broken, and2000same whole-game bootstrap draws(seed20261003).
Useful screen requires>=3pp real-M2first gain and positive95%lower, positive
real-permuted/nonempty intervals, no top3/NLL worsening, and zero17/18legality
failures for ALL arms. Candidate membership can change. Control successes or
slices cannot rescue a failed primary. Inference and legality gate stay separate
from accuracy. End-to-end evaluation CLI and independent result audit remain
TO IMPLEMENT before declaring this comparison complete.

Tests:26own route tests PASS0.088s, including6new learner/control tests and4new
scoring tests. Verified chunked/full objective agreement, finite derivatives,
synthetic learnability, neutral/support behavior, nonconvex counterexample,
count-stratified control, training-only equal-game scaling, option membership
changes, ward-only vs SAVE, and primary cannot override failedtop3/NLL/legality.
Source2neutral-decoder tests PASS. Actual data loader/worker syntax verified;
these unit tests do NOT verify a real fit, native decoder run or improvement.

Primary motivation source browsed: Jacobs/Jordan/Nowlan/Hinton,
Adaptive Mixtures of Local Experts (1991), author-hosted primary PDF
https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf . It motivates supervised
mixture gating only; our experts are fixed M2 conditionals under a specified
kernel, and no League gain follows from that paper.

LIVE PRIOR: task Videns Route Prior Release v1 Running; actual childPID2336
(parent30664) verified live repeatedly. Latest observed1955.4645s,
33796/56288states exported (~60.04%). Worker working set765698048bytes.
Only standard nested-tensor warning, no failure/exception. Source exporter SHA
3cd6dc666c4ff5e1f537a4b92bd7523326ba1461b4c469a70225a321b1403f2e unchanged;
original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
unchanged. Do not consume incomplete prior files or restart a live worker.

NEXT: verify prior completion/manifest and task exit, then register the prepared
route fit with a NEW immutable run version (proposed route-probe-fit-20261003-v1).
Finish decoder launcher, actual-scoring CLI and independent replay/metric audit
while respecting one protected job at a time. Use small synthetic checks only
alongside the source GPU export. No actual route marginal, training, new target
read or accuracy was performed this turn. All859development,1911confirmation
and global/source seals remain closed. Goal ACTIVE/UNACHIEVED.

## 2026-10-03 — route scoring, complete replay audit and native chain ready

Previous goal turn: PROGRESS (fixed recipe/protocol, fitter, source adapter and
pure metrics). This turn completes independent work while the same native
prior childPID2336 (parent30664) remains LIVE; no timeout-driven restart.

Added winprob/route_probe_io_v1.py: reads only complete fit/distribution/native
manifests, admits originalFIT identities before prediction payloads, validates
model/input/scaler/prediction locks, code snapshots, original2900focals,
probability precisions, query order, counted option identities, per-world audit
totals and costs. Targets open only after all prediction/native receipts bind.
M2 catalogue NLL uses float64 normalization of the stored native float32 prior;
corrected distributions retain their already-normalized float64 probabilities.

Added scripts/evaluate_route_probe_v1.py: actual paired scoring after both
native worlds, frozen M2 must reproduce1483first/1959top3 on2900games, publishes
all endpoints/gates/costs and explicit retrospective/exposed limits. A positive
internal result still does not authorize promotion or establish the goal.

Added scripts/audit_route_probe_v1.py: reconstructs all56288route marginals,
real/control kernels and count-bin permutation; independently recomputes each
training fold's equal-game mean/variance; verifies saved-model fold IDs, settings,
objective/gradient and directional finite differences at every real solution.
Replays all5800corrected full distributions in the original32-row batches with
EXACT equality, verifies2900source priors and neutral flags, and checks the
alternative direct exponential formula within2e-14. Recomputes every paired
metric/interval/gate and separately counts native exactness in Python.
It binds/revalidates consumed caches/full priors and accurately states it does
NOT rerun the source neural model or decoder during this own arithmetic audit.
None of these real-data stages has run yet.

Prepared scripts/run_route_probe_evaluation_job_v1.ps1 and registration script.
Arguments Version plus FitVersion. Sequential protected stages: originalCUDA
native18 decode (31min outer cap), cleanCPU native17 audit (31), own evaluation
(31), own replay audit (61),225min whole launcher. Resource wait15min/stage,
startup2.5GiB/live.75GiB RAM, idle GPU3072MiB for decode only, no League/competing
jobs. Registration refuses existing/running protected tasks and incomplete fit.
They are PARSER-CHECKED but NOT registered/launched while prior is active.

Syntax/import checks pass. Verified importing decode_route_probe_v1 leaves all
native17 forbidden-runtime consumers absent, preserving clean activation.
Read native purchase_sequence and footwear_economics.try_purchase: unknown
items already returnNone (explicit not dragon.item), so native17 unknown item
failures will be measured rather than silently treated as free purchases. No
frozen runtime change was needed or made. Existing26own+2source route tests
remain the computational verification; new IO/audit/chain not yet exercised
against real completed route models.

Latest observed prior progress:55303/56288 at3171.3528s; same task Running and
child2336 live,760172544-byte working set; host available2016000KiB. No final
manifest yet, so downstream fitting remains UNSTARTED. Continue by verifying
manifest, exact56288 saved/replayed rows, source code bindings and task exit,
then launch prepared register_route_probe_fit_job_v1.ps1 with new version
route-probe-fit-20261003-v1. Do NOT register another waiting protected task
while source is live, as competing-job guards would stop it.

No route target read, fit, accuracy, promotion, raw/DB or sealed read this turn.
Goal ACTIVE/UNACHIEVED; latest objective remains correct observed purchases.

## 2026-10-03 — exact prior completed; fit v1 RAM stop and bounded-reader v2

The full native prior COMPLETED at19:25:39UTC; task unregistered and both source
handles2336/30664 disappeared. All56288states/3518games replay exactly and all
saved rows verify. Source elapsed3243.9853s (~54.07min). Manifest
build_tracker/data/exports/winprob-route-prior/route-prior-20261003-v1/manifest.json
SHA35113430e6997ec1ef993da5d9dc3f9325941138e21db24fab300ea3b5f470ac;
full_probability.npy SHAa2c4b14975027a7aafafe5a4a879322c2ea74f2116f9fd021d3128be4641b678,
float32[56288,9333],2101343616payload bytes. Available RAM then2919936KiB.

Started route-probe-fit-20261003-v1 at19:27:08UTC (parent25748, child14240).
It TERMINATED at19:28:44UTC with authoritative launcher reason 'Host RAM floor
reached', at38912marginal rows/93.2521s. No input lock/scales/preflight/model/
target read/score was reached; only status, code_snapshot and partialinput_cache.
Both handles missing and task unregistered; not a timeout or inferred failure.
Preserve all v1 files/partial cache. Do not reuse its unwritten marginal tail.

Resource-only amendment docs/route_reader_amendment_v2.md,
SHA0f0b940e528dfc11f3d63f0e1cfc094b6970968a10521694385e1f48c3096506.
The prior mmap can retain resident pages despite a128-row loop. This is a
resource risk, not proof that other host consumers had no contribution.
Do not lower startup2.5GiB/live.75GiB or ask the user to absorb an avoidable
allocation. Replace only the2.1GBnative-prior mapping with explicit bounded
header/seek/read access. No model/loss/population/control/optimizer/gate change.

Added winprob/bounded_npy_rows_v1.py: numeric C-order NPYv1/v2, exact header/
shape/dtype/length checks, scalar/contiguous/irregular/paired row-column reads,
maximum512rows per allocation, no whole-file mmap/materialization. Three
functional fixture tests PASS, covering exact bits and order/count preservation,
paired indices, bounds, dtype/shape/layout/truncation rejection. Creates no new
source probabilities. Reader consumes the same completed native artifact.

NEW preserved versions winprob/route_probe_data_v2.py and
scripts/fit_route_probe_v2.py bind the amendment and snapshot the reader; v1
scientifically consumed code remains untouched. Downstream own audit (not yet
run) also uses bounded prior reads, clamping its final64-row batch. Prepared
run/register_route_probe_fit_job_v2.ps1, parser-checked, same limits.

Started NEW immutable route-probe-fit-20261003-v2 at19:37:21UTC,
Task Videns Route Probe Fit v2. Parent28856, actual child21468 observed LIVE.
At69.1831s it reached6144marginals with313692160-byte working set and host
available2377408KiB. This is partial evidence of the bounded reader working;
not yet a full comparison or fit result. No targets/new accuracy have opened.
Do not register evaluation while this task runs. On successful completion,
use register_route_probe_evaluation_job_v1.ps1 with FitVersion
route-probe-fit-20261003-v2 and a fresh evaluation Version.

All source export/index/frozen policy files untouched; no859/1911sealed reads,
no promotion, goal ACTIVE/UNACHIEVED. Next: inspect live21468/28856, wait for
input cost gate/models, then actual native decode/evaluation/replay audit.

## 2026-10-03 — route comparison fully trained, decoded, scored and CLOSED

The bounded-reader v2 resolved RAM: all56288marginals completed at108.6575s.
It then stopped BEFORE TARGETS/FITS at the original CPU cost gate. Sustained
objective means.69191486/.70226790/.74218842/.58266054s projected18712.5221s,
exceeding80% of10658.7149remaining. Launcher failed19:39:49UTC, handles gone;
not a timeout. No labels/models/scores from v1 or v2 exist. Both retained intact.

New computational amendment docs/route_gpu_objective_amendment_v3.md,
SHAe22716ced993861ae726a4201890b5b475ec9feed08aba4be977640e0896df73.
Added route_gpu_objective_v1.py: SAME analytic relative purchase loss/gradient,
fixed log bases cached on CUDAfloat64, SciPy L-BFGS-B and every statistical/
optimizer/data/control/gate setting unchanged. No autograd/mixed precision,
no TF32, deterministicCUBLAS. CPU1/host RAM2.5startup/.75live and GPU3072MiB,
original180min limit and original cost projection unchanged. Two GPU fixtures
PASS: CPU value/gradient parity, directional derivatives, and saved synthetic
fit checked using CPU stationarity. This is a backend repair, not a new model.

New scripts/fit_route_probe_v3.py and run/register_route_probe_fit_job_v3.ps1
preserve all consumed v1/v2code. Real v3started19:49:26UTC,parent4804/child25052.
Full input marginals completed108.3660s with bounded reader. Before targets,
each arm/fold's16input-only rows plus nonzero seeded coefficients/dummygroup0
checked against original CPU: maxvalue error5.551115e-17,maxgradient3.469447e-18.
Full objective means.020027/.017721/.017765/.017748s; guarded projection
1380.648s PASSED versus10673.284s remaining. Zero model remains exact prior.

All FOUR actual fits converged under unchanged max500,gtol1e-7,ftol0,maxls50,
ridge.01/258 fromzero. Iterations279/359/178/280=1096, evaluations296/381/183/306
=1166, gradients6.773e-8..9.774e-8. Total actual optimization22.7618794s.
All models froze before full9333-class predictions on2900focals. Fit worker
211.9501s. Task completed/unregistered19:53:05UTC; parent/child absent.
Fit report data/runs/route-probe-fit-20261003-v3/experiment/report.json,
SHA838056d4849ebbe3d5db8e293023352a5168f88cfb2782ccdcf2ddfecbe345a3.
Distribution manifest SHAb69a407f794bbebebaf0be7da8c52d50a61629b2f755df3f92c7d95df3173e2e.

Ran hidden Videns Route Probe Evaluation v1 against FitVersionv3, new version
route-probe-evaluation-20261003-v1. Native unchangedCUDA decoder replays every
M2option exactly and passes8700forbidden-field mutation queries. Each arm
has8488options; ZERO infeasible in native16.18 AND clean reconstructible16.17.
Native18stage68.6682s, native17stage12.0065s. Source manifests under
build_tracker/data/exports/winprob-route/route-probe-evaluation-20261003-v1:
decoded SHA24009edcb1fd27ff80017ffafa05f60dbff80c54d7fc250153fe467a1a7611b5,
native17 SHA93d64afc74b1eef17124541b062d36c836d817e1e99a09c1db1cd78248825360.

FINAL AUDITED RESULT on2900original distinct-game focals:
M21483/2900=51.1379%, recipe1485=51.2069%, permuted1480=51.0345%.
Real-M2+0.0690pp,95%CI[-0.3448,+0.4828],20corrected18broken.
Real-control+0.1724pp[-0.3103,+0.6897],31corrected26broken.
Nonempty2736: +0.0365pp[-0.3682,+0.4429]. All primary/control/nonempty gates FAIL.
Top3real1963vsM21959vscontrol1955; real-M2+0.1379pp[-0.1034,+0.4138].
Fullfirst1107vs1106; typedfirst1383both. Main/full quantities preserved;
ward-only remains distinct from observed no-buy. Real changes94first choices
and183option sets (control66/117), so candidate access was actually exercised.
Catalogue NLLreal1.534942877,M21.536004128,control1.536564856.
Real-M2-.001061251[-.004799384,+.002724951], unresolved. Mean top3/NLL and
both-world legality guards pass, but cannot rescue the failed primary.
Intervals2000pairedwhole-game bootstrap(seed20261003), conditional on models;
exposed retrospective16.17 under native16.18 M2, NOT forward/live confirmation.

Evaluation72.8256s, report data/runs/route-probe-evaluation-20261003-v1/evaluation/report.json
SHA10a88830ce17bfc064c4ab4e4301e4ed7017e4413bc9a1b3b1d587546c4718af.
Audit153.0847s,PASSED. Recomputes all56288marginals (error0), independent train
moments(error1.708633e-13), real CPU objectives/gradients and directionalFD
(maxerror3.293231e-14); EXACT5800correction predictions+2900originalpriors;
alternative direct formula maxerror1.332268e-15. Every interval/gate and
independent Python exactness counts rechecked. Does not repeat source neural
forwards/decoder during own audit, or claim global nonlinear optima.
Audit data/runs/route-probe-evaluation-20261003-v1/audit/report.json,
SHA751dcd8301695f7a5a2023c043c50a4738445421887a0adb1dba9c76f7daef6f.
Prediction arrays541320312bytes. Source prior and all four fitted models retained.

Pipeline completed/unregistered20:01:10UTC; no own route job remains running.
New docs/route_probe_result_v1.md and README record the complete negative result,
repairs and limits. Fixed formula CLOSED: no temperature/ridge/ontology sweep,
no alternate feature rescue, no control promotion, no forward859trial.
Two extra hits are not the requested useful breakthrough. This does not prove
all possible latent/sequential/route models useless, nor establish latent intent.
Mathematical coherence and faster fitting alone do not meet the objective.

Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46,
original native exporter SHA3c75ac5f5b09f1af54e7a13501468fdde05d862efa793acba297bad314da631f,
and main protocol SHAf6e0a8de1fdaffa76dc7775457cb1f618058c384858eafe71997cb4d5d23ec18
verified unchanged. No serving change/newcollection/859/1911sealed read. Goal
ACTIVE/UNACHIEVED. NEXT: select a substantially different, justified information
or learning strategy; do not quietly rescue this closed static route correction.
No next predictive recipe has been selected/predeclared/run in this turn.

## 2026-10-03 — causal proxy evidence ratio, mathematical/prefix stage

Previous goal turn: PROGRESS (route comparison completed, independently audited,
fixed recipe closed). No static-route, graph or net-recipe rescue is being run.
New hypothesis: closed prior choices can reveal persistent forecasting errors
not captured by current state alone. Evaluate past actions relative to forecasts
made from their own pre-purchase states; do not equate this with private intent.

Initially considered adapting M2's native parent-expert weights directly from
past likelihoods. Code inspection found a temporal-input problem: train_prefix
query explicitly includes role, _context_extras uses same-role opponent, and
basket_econ_cache uses role. Native past forecasts therefore carry uncertified
final-role metadata. Do NOT introduce that as a new causal history feature.
Current frozen M2 retains its already declared reference limitation only.

Chosen feasibility formulation instead uses existing OWN public-state proxies:
OOF state-only HistoryProbeActor and its fixed purchase-memory predictor. Verified
PublicFeatureCompiler in iql_features.py explicitly excludes legacy roles,
other-player wallets, total gold, target metadata and quest-derived fields.
The state-only checkpoint hashes remain e050c340... and5381e72e... from the
original history-probe-20260930-v3 experiment. Existing memory geometry:k32,
exact champion, one neighbor per distinct opposite-fold game, immutable row tie
break, distance temperature max(d_last/4,1e-6), .5state/.5retrieval. No new model
or neighbor-parameter fitting. These are proposed causal proxy forecasts,
not a repeat/promotion of the closed raw M2/memory averaging policy.

For proxy forecasts S,M, initial expert weights(.5,.5), update Bayes using ONLY
real visits closed strictly before the focal query. Let B_prior=.5S+.5M,
B_post=w_S S+w_M M; correction q proportional to M2 * (B_post/B_prior).
No history/no posterior change gives exact M2; identical proxies also give
neutrality for valid supported inputs; original M2 zero support stays zero.
Hypothesis: transfer relative historical evidence without replacing M2's
absolute probabilities with a weaker proxy. This can still make M2 worse.
Ordinary Bayesian cumulative log-loss regret holds for the PROXY sequential
forecaster, NOT for the anchored M2 correction or focal/top1 accuracy. A
synthetic counterexample to transferring the bound is explicitly tested.
No causally identified intent/reward or expected improvement is claimed.

New docs/bayes_proxy_ratio_design_v1.md SHA
4adf131406d85d0ad1d9c97850e9eca8f40df9d1b6d3e110e7c1530289847550;
new winprob/bayes_proxy_ratio_v1.py SHA
359e82efffe7dc2740d48296ba344861ab89f79c550d5239a43c64faf9d76c25.
Seven synthetic tests PASS0.020s: likelihood product/log-space posterior,
long prefixes/impossible observations, neutral identity, density-ratio bounds,
support, proxy regret and non-transfer counterexample, strict closure and
synthetic-SAVE insertion/deletion invariance. Model forecasting has NOT run.

Primary source read: Orseau/Lattimore/Legg, Soft-Bayes (PMLR76,2017),
https://proceedings.mlr.press/v76/orseau17a.html . Relevant log-loss aggregation
motivation only. Our prototype is ordinary Bayes, NOT that paper's complete
Prod/tracking algorithm. Also inspected Hutter's author page
https://www.hutter1.net/ai/bayespea.htm . No League-specific guarantee follows.
A PMC inverse-preference page returned a browser check; do not cite it as read.

Declared docs/bayes_proxy_prefix_protocol_v1.md SHA
 d08dfa0d1acd18f34dc3c0f5c54426a5885db609b313b7907bedf4d0b57870c0.
Implemented scripts/census_bayes_proxy_prefix_v1.py and hidden CPU1 task
Videns Bayes Proxy Prefix Census v1. Revalidates original4001FIT admission,
keeps original2900p17focals from3518games, publishes query ledger before past
full-action-kind access. It hashes the entire action-ID file but interprets
ONLY scalar IDs at indices strictly preceding each query in its own episode,
just to remove synthetic SAVE before closure/counting. No current/future ID is
indexed and no main labels, model/forecast payloads, raw/DB or outcomes opened.
Real visit physical start is the prefix identity, independent of synthetic
node insertion; state indices are only storage pointers. It does not reconstruct
whole prefixes by walking two-slot links, which can lose intermediate closure
ties. Full prefix checks strict end<query start. Ward-only real visits remain.

Census COMPLETED9.1657s, task unregistered20:50:52UTC. Result:
2900queries;15573unique closed real past visits;18473proxy forecast rows needed
including current queries.2613queries have history,287none. Prefix lengths0..19.
Fold0:1449queries/7753past; fold1:1451/7820.161synthetic nodes removed before
closure;0real closure ties and0overlaps. Every original query stays included;
618unavailable original focal games remain disclosed. Availability is not a
signal/gain estimate and must not be presented as one.

Report data/runs/bayes-proxy-prefix-20261003-v1/census/report.json,
SHA89ff1f75c4031bd88c22ea31050e157eed2f50cb6adfed9a37b0f0d309aa1b35.
query_selection.json SHA6343e7f28bac2b3dcd75e8cf6ff7648c274e98b79d80521551896dc63bf9dfd4;
prefix_state_indices.npy SHA4a124b5ad0070d3bd1a44bc778dbd71a48d5ae70d8f82493429fda7f0640d5d7;
prefix_offsets.npy SHA1b3d264b8f316e6aaced3806e49e23a668c9f2584da48e99653f190120c56fe3.

IMPORTANT next implementation constraint: HistoryDataset eagerly loads and
validates ALL current action labels, so do NOT use that constructor for the
new forecast stage while claiming query labels unopened. Use BaseCache for
metadata/admission and a state-only store for champion_indices, inventory_static,
visible_level, global_public and ragged inventory arrays. TensorStore's state
packing is reusable logic but its constructor unnecessarily loads action-static
catalogue tensors. A new small state-only adapter can omit targets entirely.
In each scored fold, memory labels may be read only from opposite-fold states;
past evidence labels only from that fold's published closed-prefix indices.
The other fold's queries can legitimately be training rows, so any no-target
claim must be RECIPIENT-FOLD-specific, not falsely global across OOF models.
Current/future target mutation must not change that fold's predictions.

Forecasting/retrieval can reuse the fixed own state checkpoints and old memory
functions; no native past inference or source final roles are needed. At query,
reuse the already exact frozen M2 priors on the2900focals from the completed
route input artifact; no 54minute native re-export is needed. Preserve k32 and
all old memory parameters. A separate predictive protocol MUST still freeze
controls, batches, materialization/memory bounds, native decoding and success
thresholds BEFORE generating/scoring candidate recommendations. No final control
set or actual new predictive comparison was fixed/run this turn. Candidate ideas
considered but NOT selected include a static-proxy ratio control and metadata-
seeded sign-scrambling of past likelihood evidence; do not treat those thoughts
as a predeclared experiment.

Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
unchanged. No active own prefix/route task, no new model fit or purchase score,
no859/1911/global/source sealed reads or promotion. Goal ACTIVE/UNACHIEVED.
NEXT: fix a genuinely history-sensitive predictive/control protocol, implement
its target-deferred causal forecast pipeline, run against native M2 and audit.

## 2026-10-04 — fixed Bayes proxy comparison forecast and native stages complete

Previous goal turn: PROGRESS (seven mathematical tests and exact closed-prefix
census). Fixed the predictive recipe BEFORE forecasts in
 docs/bayes_proxy_probe_protocol_v1.md,
SHAa60292830d167824f8d14a608020c4f75da3a023b65ef85d5c42ad7f2b91837b.
Four arms: unchangedM2; real closed-history Bayesian proxy ratio; same evidence
with metadata-seeded likelihood-pair sign scrambling; static-memory ratio on
the same history-available population. All287empty histories retain exactM2.
Primary>=3pp displayed first gain, positive95%lower vsM2 AND both controls,
positive nonempty guard, no mean top3/NLL worsening, zero17/18legality failures
and complete audits. No control promotion or retuning after a failed primary.

Implemented bayes_proxy_data_v1/v2, StateOnlyStore, LabelGate, bounded row writer,
fixed controls/decision module and tests. State inference receives only public
state arrays; no label getter/array. For scored foldk, label permissions allow
only opposite-fold TRAIN rows or k's published closed-past indices. Current/
future requests fail before raw label access. Across OOF models the other
fold's queries can legally be training rows: no falsely global no-label claim.
16Bayes/label/decision/writer tests PASS; two bounded tensor-reader tests PASS.
Two additional four-arm scoring tests PASS, requiring BOTH controls and every
legality/NLL/top3guard. No source runtime imported by the proxy forecaster.

Frozen state checkpoints unchanged; k32/champion/distinct-game/temperature/
.5state+.5memory unchanged. Bank and forecast rows sorted, fixedGPU256batches,
CUDAfloat32 neural inference/noTF32, float64 geometry/probabilities. No optimizer
or neural training. Forecaster builds392448bank embeddings across two folds and
18473current/past forecasts, then uses15573allowed past likelihood pairs.

Attemptv1 (bayes-proxy-forecast-20261003-v1) failed before its input lock because
Torch exposed virtual relative _ops.py. V2 changed only the provenance filter
to p.is_file(), preserving v1. V2 then hit the host.75GiB RAM floor while encoding
its second bank, after first-bank TRAIN labels and64input-only forecast timings;
no recipient past evidence or posterior/candidate score was computed.
Both were terminal (task/handles gone), not timeout restarts. No partial bank reused.

Resource amendment docs/bayes_proxy_memory_amendment_v3.md,
SHAa36f2eaff0235b4c5e599229d821ccaa709c4ad34c23600cd55fe5d9838fab38.
New bounded_npy_array_v1 exact numeric leading-row reader and bayes_proxy_data_v2
close only the four large dense state mappings after validation and byte-exact
row0/middle/last comparisons. Smaller ragged tables remain. No feature, model,
batch, statistical rule or RAM threshold changed. Native priors/proxy outputs
use bounded read/seek-write as well, not complete resident output mappings.

Forecast v3 COMPLETED132.7611s, task unregistered21:49:54UTC2026-10-03.
All18473proxy queries;15573past pairs; all392448TRAIN labels in their proper
opposite folds. No empty retrieval. All three corrections have exactly287
neutral queries. Preflight projected1069.6421s, passed before past evidence.
Neural checkpoints not refitted. Report:
data/runs/bayes-proxy-forecast-20261003-v3/experiment/report.json
SHA9901c9d7f6827dc6a092f206f18468ba1e92ee1aaaf65cd66031697c0c7759eb.
Distribution manifest SHA6b9c9302f83b223a89302dce1c705fd1474686f7617625a436d3683b20dc7d3c.
All current distributions/posteriors/evidence/forecast hashes frozen before scoring.

New source decoder decode_bayes_proxy_v1/v2, four-arm metrics, completed-artifact
reader, evaluation CLI, independent full-forecast replay and metric-audit CLI
prepared. Native v1 (bayes-proxy-native-20261003-v1) hit hostRAMfloor after2432
queries; no complete decoder artifact or score. Native v2 changes only its
four probability input mappings to source-owned bounded ProbabilityRows reads.
No consumer runtime import, forecast regeneration, filtering or model change.
Amendment docs/bayes_proxy_decode_memory_amendment_v2.md,
SHAde770b42d94453f1c2cb55b302e41459fde23c25ecc4a25f08bd3a669f0b2ad7.
Source bounded-reader exact32/64-bit fixtures PASS. Own scoring reader also
uses bounded probability blocks before its first real use.

Native v2, source output
build_tracker/data/exports/winprob-bayes-proxy/bayes-proxy-native-20261004-v2,
COMPLETED22:12:17UTC2026-10-03. Native18decode96.4109s; clean native17audit
14.3038s. Exact frozen M2 replay and11600target-mutation queries. Options:
M28488,history8488,scrambled8487,static_memory8488; ZERO infeasible in BOTH
worlds for ALL arms. No target accuracy was read/claimed at that stage.

ACTIVE final job: Videns Bayes Proxy Evaluation v1,
Version bayes-proxy-evaluation-20261004-v1,
ForecastVersion bayes-proxy-forecast-20261003-v3,
NativeVersion bayes-proxy-native-20261004-v2.
Starts with own GPU forecast replay, then CPU scoring, then independent metric
audit. Parent28100/actualchild18972 verified LIVE in forecast_audit stage.
At80.4889s it had replayed fold0 banks/queries and fold1bank, then was replaying
fold1queries. Working set1429721088bytes. No error output. Do not restart from
stale logs or launch another protected task while it runs.

Forecast auditor verifies all392448bank embeddings,18473proxy distributions,
all retrieval IDs/weights/distances and15573past likelihoods; tests current/
future label rejection BEFORE loader access; exactly replays8700anchored
outputs, plus independent likelihood-product posterior and direct-ratio checks.
Scoring is still pending this replay. No improvement or breakthrough claimed.
Source status read21:34UTC: daily taskReady, unrelated expired APIkey left alone;
no served policy. Original index and closed experiment files untouched.
Goal ACTIVE/UNACHIEVED. NEXT: inspect this specific final job; finish and inspect
both audits before reporting metrics or deciding whether to close this recipe.

## 2026-10-04 — Bayes proxy ratio fully audited and CLOSED NEGATIVE

The forecast replay PASSED155.1948s. Exact392448bank embeddings,18473proxy
queries,15573past likelihood pairs and8700anchored distributions;2900original
priors and empty-history parity verified. Rejected2900current and2625future
label requests BEFORE raw storage access. Independent posterior maxerror
4.718448e-15 and direct-ratio maxerror1.221245e-15. Forecast-audit report
 data/runs/bayes-proxy-evaluation-20261004-v1/forecast_audit/report.json,
SHA7ba2295f1d847618e3490242b0f13d15996c2d9925705da672810b5e45562106.

Scoring COMPLETED103.9984s and final metric audit PASSED98.5423s. The launcher
briefly waited at2.41GiB free before the final CPU audit; it later met2.5GiB and
completed automatically. No RAM/resource/statistical threshold was lowered.
All own tasks/workers completed/unregistered22:28:04UTC2026-10-03 (00:28Madrid
onOctober4). No outstanding process, restart, approval or resource request.

FINAL observed counted-main displayed first accuracy, same2900internal focals:
M21483=51.1379%;history1469=50.6552%;scrambled1482=51.1034%;static_memory1476=50.8966%.
History-M2 -0.4828pp,95%CI[-0.9310,-0.0690],15corrected29broken. Fourteen fewer
correct first purchases; deterioration resolved in this conditional internal
comparison. Nonempty2736: -0.4751pp[-0.9141,-0.0366].
History-scrambled -0.4483pp[-0.9310,+0.0690]; history-static_memory
-0.2414pp[-0.8629,+0.4138]. No positive-control requirement passes.

History top3covers1967vsM21959 (+0.2759pp[-0.0690,+0.6552], unresolved),
scrambled1959/static1970. This cannot rescue the failed primary.
Fullfirsthistory1094vsM21106 (-0.4138pp[-0.8276,0]);typed1369vs1383.
NLLhistory1.538909950vsM21.536004128, delta+.002905822
[-.001741631,+.007932452]. Mean worsens and fails its guard, though unresolved.
ScrambledNLL1.546268925,static1.546149222. Changed first/option sets:
history94/239,scrambled96/222,static232/480. All287empty-history queries stay
exactM2 in all three corrections. Every declared game retained.

Zero native17/native18legality failures for all arms:8488options each forM2,
history andstatic;8487scrambled. ExactM2 source replay and11600mutation checks.
Native18elapsed96.3931s, native17elapsed14.2918s. No option replacement/filter.
No fresh source neural forward in own audit; source decoder receipts were
separately bound. No prediction of private intent or causal purchase value.

Evaluation report data/runs/bayes-proxy-evaluation-20261004-v1/evaluation/report.json
SHA4bfa9892ccd0cb7da645dc6436233dfe1c23ab5ba704fc582cb96ac565a0dc42.
Final audit data/runs/bayes-proxy-evaluation-20261004-v1/audit/report.json
SHA047587c8234cdd9b092229592246df51d6312f00b17ef7a030f21aa5883b9eb5.
Metric auditor rechecked all means, intervals, gates and independent Python
native exactness counts. GateFALSE. Bootstrap2000same game draws(seed20261003),
conditional on fixed models; no training uncertainty or multiplicity adjustment.
Exposed retrospective16.17/nativen16.18 screen, not forward or live confirmation.

Decision: CLOSE the tested causal proxy pair + full closed-prefix Bayesian
likelihood update + anchored ratio. Do not tune priors, forgetting, clipping,
experts or controls to rescue these scores. No forward859trial, control
promotion or serving change. Negative result does not prove all history useless;
it rejects this estimator/transfer to the stronger reference. A correct Bayesian
proxy calculation is not a guarantee for M2 recommendation accuracy.

New docs/bayes_proxy_result_v1.md and README record the complete negative result,
audit scope, preserved resource/provenance attempts and costs. Original index,
frozen source models/code and all859/1911/global/source sealed payload boundaries
preserved. Goal ACTIVE/UNACHIEVED. NEXT: reassess the research strategy before
another small adapter; no next predictive recipe was selected/run here.

## 2026-10-04 — strategy reassessment and complete causal student prepared

Previous goal turn: PROGRESS (Bayes proxy experiment closed after complete
negative evaluation/audits). No existing failed adapter is being retuned.
Re-read source history and actual training/scoring code. The own state/memory
proxies pool per-player features to means/maxima and use a flat9333-class head;
native M2 uses board attention and counted-basket nonlinear state/action
scoring. This is a verified implementation difference, not a causal attribution
of the performance gap. Source data-scaling gains are real, but old transfer/
gating/capacity recipes largely failed and remain closed.

Metadata-only source16.19-generation check:2770TRAIN/1342VAL. Its2770TRAIN
IDs are EXACTLY859own development plus1911own confirmation. No recent train
population was silently admitted or any game/seal opened. This statement
concerns that published generation, not an unmeasured entire future archive.

Selected next approach: complete causal state/action learner, optionally
regularized by frozen M2 teacher distributions on TRAIN rows only. Current
student inference receives public board/runes/catalogue, never labels, final
roles or teacher outputs. Actual purchase remains the hard ground truth.
This is generalized distillation/representation learning, not another runtime
M2 multiplier. Primary papers inspected:
https://arxiv.org/abs/1511.03643 (Lopez-Paz etal., generalized distillation),
https://proceedings.mlr.press/v80/furlanello18a.html (Born Again Networks),
https://arxiv.org/abs/1503.02531 (Hinton etal.). No League gain follows from them.

New docs/causal_student_design_v1.md SHA
9e3323a6f9b9150d0699da0d62712e93884cbbdee6a40c8b960b440a82b74fb6.
New winprob/causal_student_v1.py SHA
6e74d7b1d5496c291cb6c530c482604ab5afbda879e484093aefe9b3405d032b.
Prototype: public10player tokens + global/rune query; champion16/shared item32
embeddings, role-free buyer/ally/enemy type IDs, d64/4heads/2layers/FF128/.1dropout;
state query+buyer->64. Counted action sum plus23attributes,55->64->64, learned
empty-MAIN vector. Bilinear +192->256->64->1 GELU interaction, chunks512 and
checkpointed deterministic deep component. Inventory/action counts never truncate.

Initial quantity test exposed the reused MultisetSum's zero embedding
initialization. BEFORE any real/synthetic artifact use, explicitly initialized
new student item weights normal0,1 with zero padding, matching its champion
embedding convention. Re-ran5tests: PASS0.961s. Tests cover player permutation,
item order/counts/empty identity, checkpoint/chunk values AND gradients,
observed-label vs teacher gradient, and a teacher/label-free inference signature.
No old helper or frozen model was edited. KD objective prototype CE+KL has
analytic logit derivative2q-onehot(y)-teacher; teacher detached. Training caller
will scale KD by.5 to align total loss coefficient with hardCE as fixed below.

Synthetic resource pilot COMPLETE3.0467s through hidden task (now gone).
B128,N9333,actionchunk512;2warmups+4timed steps, dummygroup0/uniformteacher only.
Reserved uninitializedCUDAfloat32[56288,9333],2101343616bytes, without opening
real teacher probabilities. Mean step.1919173s, timings.1917204/.1953201/
.1874197/.1932090. Peakallocated2479704064bytes including reservation,
peakreserved2533359616; conservative projectedfree4917821440 and1GiBheadroomPASS.
Synthetic201585parameters uses180champs/1024items; actual public vocab173/870
implies196545parameters by dimension accounting (not a real fit).
Hostafteravailable1358794752bytes (> .75GiB), working1444872192, peak1469992960.
No checkpoint saved. Resource report
 data/runs/causal-student-resource-20261004-v1/experiment/report.json,
SHA88ea8187473300d8786a21fa66256e1635db3e005881107f4531b9a5c6055044.
These are synthetic cost/memory measurements, NOT real-data runtime or accuracy.

New input protocol and prepare_causal_student_inputs_v1.py bind code before
reading data and AST-check rune vocab parity without importing model/runtime.
Original admission validated first. Read only keystone_id/sub_style from
versioned source_contexts (other JSON fields including roles may deserialize,
not indexed). No target/teacher array, raw/DB or source runtime. Compile count-
preserving9333main baskets with native16.18public attributes and own item IDs.

Input artifact COMPLETE23.1373s, task gone:
 data/runs/causal-student-inputs-20261004-v1/inputs/manifest.json,
SHA4b45bb0f8e68d68645d1ac40c3f1a6c06780fe8e09667f9366146e966d5b8c2a.
56288states/3518games/2900focals;25530selected player-games. Rune pairs constant
among SELECTED observations (not a new audit of every visit);0missing keystones,
2missingsecondary styles,0unknown nonzero IDs. All states retained.9333actions,
padded distinct-item width5,totalcopycap6,one empty-main class. No training/score.
Files include rune/raw IDs, original state/fold/focal indices and action tensors;
all hashes in manifest. Selection JSON exactly97b7842e... (original unchanged).

Prospective predictive protocol NOW FIXED before any real fit:
 docs/causal_student_probe_protocol_v1.md,
SHAfc3168bffd7865eb9ab53a18450501d8732474cf8df8077f16e61ffac7abbd5e.
Candidates:hard anddistilled;shuffled_teacher is ONLY diagnostic, not eligible.
Twofolds x3arms=6fresh models, identical within-fold initialization/order/dropout
seed, Adam3e-4,batch128,20epochs, finalcheckpoint only. Actualupdates26460:
scoredfold0 TRAIN28112 =>220batches/epoch;fold1 TRAIN28176 =>221. Prospective
draft initially said26400; corrected BEFORE training or protocol consumption.
Partial-batch mean loss multiplied bybatch_rows/128 beforebackprop; epoch loss
accounting uses actualrowcounts. No scheduler, AMP, TF32 or weight reuse.

Hard optimizesCE;distilled/shuffled optimize.5*(CE+teacherKL),temperature1.
Teacher-shuffle TRAIN-only withinchamp/3min bins, separatefixedNumPyseed.
Same sourcefullprior alreadyavailable, no54minre-export. Each model may gather
only opposite-fold teacher/target rows for loss; forward API accepts neither.
Two eligible candidates versusM2 use reported95% plus adjusted97.5%two-sided
primary intervals (.0125/.9875 quantiles) for conservative two-candidate handling.
Each must gain>=3pp first, adjustedlower>0, nonempty95%lower>0, no mean top3/NLL
worsening, zero17/18legality, completeaudits. Ifbothpass choosehigherfirst,
thenlowerNLL,thenhardontie. Attribute teacher benefit separately; no claim if
KD doesn't beat hard/shuffled. No control promotion ifcandidatesfail.

Real worker NOT IMPLEMENTED/STARTED yet. Next: build the selected public tensor
loader, GPU preload (only56kstates+teacher, boundedhostcopies), guarded fold/
teacher/label permissions, matched6-fit trainer, real-input dummy-label cost
pilot, native decoder and result audits. Protocol allows6h; pilot criterion
1.5*max_timed_step*26460+900 <=.8remaining,1GiBdeviceheadroom. Reset throwaway
pilotmodel/optimizer/RNG beforeactual training. Keep2.5/.75GiBhost and idleGPU/
noLeague/no competingprotectedjobs. No extra architecture/epoch/weight sweep.

A first patch attempt to change nonexistent protocol text failed atomically;
then the exact26460/partial-batch correction applied. A PowerShell property-count
read produced arrays ofones rather than counts; subsequent Python metadata read
verified actual173champions/870items. Neither affected artifacts or scores.
Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
unchanged. All859development/1911confirmation/source/global seal boundaries
preserved. No active student task or realtraining. Goal ACTIVE/UNACHIEVED.

## 2026-10-04 — complete causal student training RUNNING

Previous goal turn: PROGRESS (full architecture, prototype tests, resource/input
artifacts and prospective two-candidate protocol fixed). This turn implemented
and launched the actual six-model training; no student accuracy computed yet.

Added winprob/causal_student_data_v1.py. Revalidates original BaseCache admission,
selected3518games/56288states/2900focals, prepared rune/action artifact4b45bb...,
and exact native teacher artifact351134.../full probability SHAa2c4b149....
Reads public dense tensors through bounded NPY readers; preloads ONLY selected
public states/runes/actions and teacher matrix toGPU, with bounded host copies.
Closes smaller host ragged mappings after GPU state packing. Public inference
store contains no teacher/label attributes. Separate training getters reject
held-out ordinals AND cross-fold teacher donors before gathering supervision.

Metadata-only shuffled-teacher map is fixed within TRAIN champion/3min bins;
seed20261004+fold. Epoch ordering uses the predeclared SHA salt, independent of
global NumPy RNG. True targets open only after a durable input lock and successful
cost pilot. Source/full-to-main mapping preserves item quantities. Three new
permission/shuffle/order tests PASS; with five architecture/loss tests,8PASS.

Added scripts/train_causal_student_v1.py plus hidden run/register launchers.
Six fresh models:hard/distilled/shuffled_teacher x two folds. Per-fold identical
seed/init/order/dropout sequence,20epochs,batch128,Adam3e-4,partial loss scale
batch_rows/128,checkpointed deep chunks512.26460updates total. Hard optimizes
actualCE; distilled/control optimize.5*(actualCE+teacherKL). No teacher field
in CausalStudent.forward. Final epoch20checkpoint only; all six models freeze
before any held-out prediction. Save optimizer, initialization/final hashes,
trainIDs/orders,loss terms and counts. Release all supervision GPU/host tables
before teacher-free student prediction; originalfloat32M2 is separately copied
for unchanged native reference. No source runtime, training-set change or
validation-driven selection. Protocol SHAfc3168bffd7865eb9ab53a18450501d8732474cf8df8077f16e61ffac7abbd5e unchanged.

Started hidden Task Videns Causal Student Train v1 at23:47:37UTC2026-10-03,
version causal-student-train-20261004-v1. WrapperPID11904, venv launcher13160,
actual Python child28776. Logs data/runs/causal-student-train-20261004-v1;
experiment output under its experiment/ directory. DO NOT register another
protected waiting task while this one runs; Busy guards would terminate it.

Real selected-input preload completed46.426s. Cost preflight PASSED before
actual labels: timed update seconds.1894888/.1917054/.1887981/.2070966;
projection9119.6641s under80% of21551.3355remaining; conservativeGPU headroom
4712300544bytes. Dummy group0labels only, opposite-fold teacher128rows; throwaway
model/optimizer deleted and real seeds reset. Real training then began.

LATEST VERIFIED LIVE: child28776, parent13160, TaskRunning. At487.2952s,
fold0/hard epoch10of20 complete,2200updates,0of6final models yet. Epochs6-10
42.07-42.48s. Working set447528960bytes; available hostRAM2154700KiB, abovefloor.
No stderr. Earlier childWS1124548608bytes during initialization. This is actual
training progress, not accuracy. Do not compare train losses with native scores.
Never restart because observation times out or logs are temporarily quiet.

Independent work prepared while GPU trains:
- New source-only build_tracker/scripts/decode_causal_student_v1.py, cloned into
  a NEW version from the verified bounded decoder. Four arms M2/hard/distilled/
  shuffled_teacher; exactM2replay, nativeCUDA decoding, target-mutation checks,
  separate17world. No old decoder/runtime edited. NOT run yet.
- winprob/causal_student_evaluation_v1.py and3tests PASS. Two candidates only;
  shuffledcontrol cannot be promoted. Computes95% and adjusted97.5%primary
  intervals, nonempty/top3/NLL/legal guards; deterministic first/NLL/hard tie
  selection; teacher-specific contrast separate from practical M2 benefit.
- Source scripts/status.py read23:52UTC: dailyReady, no active source task,
  unrelatedexpiredAPIkey left alone, no served policy. It does not list own
  Videns task, which was independently confirmed live.

Remaining implementation while training continues: completed-student receipt
reader; score CLI; replay audit for checkpoints/teacher permissions/exact
inference and metrics; native/evaluation launchers. Respect RAM: use bounded
probability reads rather than four full mappings. No source decoder or other
GPU job may run beside the current protected learner. Do NOT alter consumed
student/data/trainer/protocol files; new scientific versions if repair needed.

Goal ACTIVE/UNACHIEVED. No score, breakthrough, promotion, forward859trial,
new collection or1911/source/global sealed payload read. Original index/frozen
source models remain untouched. Next inspect PID28776 and task status, continue
independent evaluation/audit implementation, then finish six fits and compare.

## 2026-10-04 00:41 UTC — student assessment prepared; live fit unchanged

Previous goal turn: PROGRESS. Current continuation completed the evaluation
implementation while the fixed learner remained active. Goal ACTIVE and
UNACHIEVED. User's correct answer remains the actual retained purchase multiset.

Latest observed training at3181.3514seconds: fold1/hard epoch13,2873updates,
three of six final models complete. Actual child28776, launcher13160 and
wrapper11904 still live; Videns Causal Student Train v1 is the only running
protected task. worker.err is empty. At00:36UTC host freeRAM2.4009GiB; GPU
earlier3886MiB used/4071MiB free/90% utilization. No new GPU worker launched.
These are timestamps, not a claim that the process is still live when reread.

Checkpoint compatibility issue detected before this continuation: trainIDs
were serialized as NumPy str_ scalars, which the original post-training
weights-only torch.load rejects. Synthetic and first actual final checkpoint
loads reproduced it. A scoped safe_globals loader for three NumPy metadata
types succeeds while preserving checkpoint bytes/tensor hashes and keeping
weights_only=True. One checkpoint loader test passed. This is NOT a reason to
stop the live trainer: let all six fixed20epoch fits finish unchanged. Its
expected failure after models_lock and M2-copy is not actual terminal evidence.

Prepared only, NOT RUN:
- docs/causal_student_metadata_load_amendment_v2.md (SHA
  dc2ee5dae1e4ec027b6041571b5d302caa6ec52472cfce8340320914dbfd81ce).
- winprob/student_checkpoint_v1.py, student_public_inference_v1.py and
  scripts/predict_causal_student_v2.py. After actual expected failure, copy all
  six unchanged checkpoints/receipts into a new run; no training restart.
  Verify26460updates and folds/orders, then predict from only2900public focals
  at the original global inventory padding width and128-row batches. No teacher
  or observed-label GPU storage. Bind original-file copies, terminal error,
  launcher status and recovery lineage with digests. Separate training cost
  from inference-only recovery duration.
- run/register_causal_student_predict_job_v2.ps1. Proposed version
  causal-student-predict-20261004-v2, TrainingVersion
  causal-student-train-20261004-v1. Actual failed launcher plus all-six lock
  required first. Registration arguments include TrainingVersion. NOT registered.
- winprob/causal_student_io_v1.py (CompletedStudent), score CLI
  scripts/evaluate_causal_student_v1.py, model auditor
  scripts/audit_causal_student_models_v1.py, metrics auditor
  scripts/audit_causal_student_metrics_v1.py. Final metric auditor creation was
  verified after prior truncated output. Bounded probability readers throughout.
  Model audit checks copied original bytes,20epoch/26460update traces, seeds,
  training identities, optimizer receipts, fold/teacher permissions, rejection
  of56288held-out supervision requests,8700exact public-only student replays,
  and unchanged M2. It does not repeat all26460training updates.
- run/register_causal_student_assessment_job_v1.ps1, proposed Version
  causal-student-assessment-20261004-v1, PredictionVersion
  causal-student-predict-20261004-v2. One hidden task runs model audit, isolated
  source18decode, source17legality, scoring, and metric audit sequentially.
  Source artifacts under build_tracker/data/exports/winprob-causal-student/
  plus the assessment version. Existing source adapter decode_causal_student_v1
  remains unused. CPU1, host2.5GiB startup/.75live, GPU3072MiB idle startup,
  no League/capture/protected competing task. No waiting task registered.

Validation this continuation: all four prediction/assessment PowerShell files
parse; all six new Python prediction/reader/audit/score modules compile; three
synthetic candidate-selection tests pass (control cannot be selected, adjusted
intervals, no-harm gates/tie break). Real checkpoint replay and native scoring
remain pending, so syntax/tests do not establish pipeline success or buy gains.
Frozen trainer SHAa3f09fa9e4cb79afadcf4e17281b275be30472aec33629ee9bec39cd6d618a28,
protocol SHAfc3168bffd7865eb9ab53a18450501d8732474cf8df8077f16e61ffac7abbd5e,
and original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
all reverified unchanged. README now records real training, replacing stale
prototype-only status. No current student purchase score, promotion,859trial,
1911/global/source sealed payload read or new collection.

NEXT: inspect live PID/task/logs before any action. Do not restart or register
a protected waiting task beside training. After actual terminal metadata error,
verify all-six lock and invoke prepared prediction v2, then after its completion
inspect source scripts/status.py and invoke prepared assessment v1. Inspect
all metrics/uncertainty/legality and reproduce before any improvement claim.

## 2026-10-04 00:47 UTC — focal inference transfer verified before use

Previous continuation: PROGRESS (complete assessment implementation prepared).
Current continuation: PROGRESS plus verified live training wait. Added two
synthetic CPU tests for the new public-focal inference materializer. A130focal
fixture crosses the128row boundary with nonidentity global-state selection,
nonidentity output minibatches, repeated item quantities and global padding5.
Independent nested-loop reconstruction matches all item IDs/counts exactly;
board/runes retain query alignment and4ally/5enemy masks. Nonfocal dense states
changed to NaN leave focal inputs unchanged. Any live teacher or label storage
is rejected BEFORE feature/device access. Both tests PASS (0.668seconds).

student_public_inference_v1 now accepts an explicit CPU device only for those
synthetic tests; production recovery and replay still require CUDA and call its
unchanged CUDA default. The helper has not yet been used for real predictions.
Every code digest consumed by the live original trainer was rechecked unchanged.
No real target/prediction scoring, GPU contention, model/optimizer change or
test/confirmation read. Four final checkpoint files now exist; the observed
live process28776 advanced to fold1/distilled epoch1 at3521.7503seconds, with
four completed models. Exact checkpoint replay remains deferred until all six
models finish. Continue the recovery/assessment sequence already prepared above.

## 2026-10-04 20:13 UTC — interrupted final control recovered, first epoch exact

Previous continuations: VERIFIED WAIT on actual process28776, advancing to
five final models at01:00UTC. Current continuation discovered a real terminal
interruption after a long wall-clock gap; this is not an observation timeout.
At20:01UTC the original11904/13160/28776processes were absent, no process had
the original training command, task Videns Causal Student Train v1 was Ready
with LastTaskResult1073807364, and last Windows boot was09:51UTC. The original
launcher_status.json says running but is stale. Exact cause/termination time
is not established. All original artifacts remain unchanged. Only the stale,
nonrunning own scheduled task was removed after recording its identity/state.

Interruption evidence artifacts/recovery/causal-student-interruption-20261004-v3.json
SHAa0a6fb7073a3779562e0cf0c289ff777de43fb9fe5e4a1d136b58552a5aa83c8.
Five full20epoch checkpoints exist: threefold0plus hard/distilled fold1.
Original trace has101rows:100kept epochs plus fold1/shuffled_teacher epoch1,
221updates. No sixth checkpoint, models_lock, prediction directory or report.
No student purchase score has been computed. Original metadata-only v2 recovery
is NOT applicable: no terminal NumPy load error occurred. Do not fake failure
status or try its launcher. The scoped safe checkpoint loader remains useful.

Declared docs/causal_student_interruption_amendment_v3.md BEFORE recovery,
SHA140e574bae56d1346cc6f71cc02c7f2c9ce5837b928b67661b8407a249e78174.
Resource-interruption exception to no-restart rule: preserve five finished
models; recompute ONLY the unsaved final control with seed20261005,20epochs,
4420updates and all original settings. No warm start, checkpoint choice or
retuning. Require exact first-epoch loss/order/count replay excluding seconds.
At least221updates/42.8186013seconds were discarded; later partial updates,
if any, are unknown. Effective kept family still26460updates/120epochs.

Implemented scripts/resume_causal_student_v3.py,
SHAe3f3d10d1ad0e66fd8ff62046e72aa93f7a4e4f6e0f674ee0ec24b67550e83f9,
derived from immutable original trainer. The actual minibatch-update loop AST
is EXACTLY equal to the original, not just manually considered equivalent.
Validates admission/old bindings/checkpoints/trace, copies five final files,
uses original conservative resource preflight, trains only missing control,
then freezes all six and predicts using the original128-row public CUDA batches
with the scoped weights-only loader. Report declares training_restarted=true,
completed_models_retrained=false and discarded compute as a lower bound.
Original v1trainer SHAa3f09fa9... remains unchanged. New code is now consumed:
do not modify recovery worker, amendment or its code/input dependencies live.

Hidden Task Videns Causal Student Recovery v3 registered and running from
20:10:10UTC. Run data/runs/causal-student-recovery-20261004-v3; wrapperPID10988,
venv launcher21700, actual Python12872. Same6hour bound,CPU1,host2.5/.75GiB,
CUDA3072MiB idle startup and League/capture/protected workload guards. Do not
register another waiting protected task while it runs. All prepared launchers
parse and recovery/audit/evaluation Python modules compile.

ACTUAL RECOVERY EVIDENCE at20:13UTC: public preload65.106s; resource gate passed;
all five checkpoints copied unchanged by83.425s. Epoch1 completed126.960s and
passed exact parity: optimization_loss5.71798984764788, hard_loss6.4305617028105875,
teacher_KL5.005417957825799,28176states/221updates, order SHA608e19d94d4ed2d48a9a42c6efe1302cad56aac529a85e008240f41a27774e73.
Only elapsed time differs (43.4957s vs42.8186s). Durable proof is
experiment/interrupted_epoch_replay.json. Latest live control epoch2/20,
442updates at170.350s; child12872working set1386217472bytes, no stderr.

Prepared model auditor now verifies interruption lineage, five unchanged model
bytes, all100preserved epoch receipts and exact lost-epoch parity. It also
binds audit-specific inference/helper code. Score CLI requires that proof and
reports retained-fit compute separately from known discarded compute; total
discarded partial compute is unknown. These audit/scoring files remain unused.

NEXT after ACTUAL completed recovery report/process exit: source status.py,
then scripts/register_causal_student_assessment_job_v1.ps1 with Version
causal-student-assessment-20261004-v1 and PredictionVersion
causal-student-recovery-20261004-v3. That sequential task runs model replay,
native18decode/native17legality, fixed metrics and metric audit. Recovery v3
already performs prediction: do not run the old predict-v2 step. No accuracy,
benefit, promotion,859trial,1911/source/global sealed read or completion claim.
Goal ACTIVE/UNACHIEVED. Current continuation is PROGRESS, not a blocker.

## 2026-10-04 20:37 UTC — complete student family fails; all audits complete

Previous continuation: PROGRESS, bounded recovery launched with exact lost-
epoch parity. This continuation completed the recovered sixth model, all
predictions, native comparison and both audits. Current classification: PROGRESS.
The full goal remains ACTIVE/UNACHIEVED; this fixed family is CLOSED negative.

Recovery causal-student-recovery-20261004-v3 completed at20:26:54UTC. Five old
checkpoints are byte-identical; fold1/shuffled_teacher completed20epochs and
4420new updates. The retained six-model family has26460updates/120epochs.
All original128-row predictions completed before any accuracy was scored.
Report SHA0f869f5947d0103e02e5df9c28109b80fa2f14e6f5ae97283a5e806146136716;
distribution manifest SHA237fe8b1bfd9ce1d9c86f5a3c8a2d6406cd1d79bd2a27e6d80bfbce0e65dd8af.
Do not rerun old training/predict/recovery launchers. All consumed scientific
code and artifacts are frozen; original stale running JSON remains evidence.

After confirmed process exit, source status.py read20:26UTC: dailyReady,
no source job running, no served policy, unrelated expired API key untouched.
Started Videns Causal Student Assessment v1, version
causal-student-assessment-20261004-v1, PredictionVersion
causal-student-recovery-20261004-v3. It completed at20:34:17UTC; subsequent
process/task inspection found no remaining student tasks/workers.

Audit verified8700exact student forecasts,2900native baselines,26460training
batch permissions and56288held-out supervision request rejections. Five old
checkpoint bytes,100preserved epochs and exact interrupted-epoch replay verified.
Native M2 options replay exactly;11600mutation queries pass. Every option is
legal in both16.18and reconstructible16.17:8488M2,8486hard,8499distilled,
8460shuffled. Final audit independently counts and recomputes all metrics,
intervals and gates. Both integrity audits pass; metric success is FALSE.

ACTUAL paired displayed main first on2900unique-game original focals:
M2 1483/2900=51.1379%; hard1003=34.5862%; distilled1032=35.5862%;
shuffled909=31.3448%. Distilled-M2=-15.5517pp, adjusted97.5% interval
[-17.5172,-13.6897];116corrected/567broken. Hard-M2=-16.5517pp,
adjusted[-18.4836,-14.6207];116corrected/596broken. No selected candidate.
Main top3 counts1959/1569/1596/1451; full first1106/726/748/655;
typed main first1383/904/932/808 (same M2/hard/distilled/shuffled order).
Nonempty2736:distilled-M2=-16.3743pp[-18.0958,-14.7099]95%;
hard-M2=-17.4708pp[-19.2257,-15.7084]. NLL1.536004128M2,
2.803743168hard,2.625575244distilled,3.133880872control.
Distilled-M2 NLL+1.089571116[1.032701429,1.146565331]95%.

Within-student teacher contrast:distilled-hard+1.0000pp[0.0690,1.9319]95%,
distilled-shuffled+4.2414pp[2.9310,5.5517]. This conditional diagnostic is NOT
a useful M2 improvement. Do not promote the weak-student contrast, blame a
particular feature/training cause not isolated by this design, or launch a
hyperparameter/epoch/data slice rescue of the failed family.

Cost:5128.079s retained-model training; at least42.819s/221updates discarded
in interrupted control, later partial updates unknown. Recovery987.888s,
model audit69.457s,decode105.193s,native17audit15.673s,score78.348s,
metric audit78.240s. Six-model compute is not the987.888s recovery wall time.

Evaluation report at data/runs/causal-student-assessment-20261004-v1/evaluation/report.json
SHAf369133d8ef2d3d22972679beca271e51d4a0e36035636854077610ff00cb00d;
model audit SHAbbfe4b0f96279579ad739e8bd6d51d8675dd5d443404e7b5d2347f1aa06b55a7;
final audit SHA294f40aef5099ad3bda14e7ae999430ad7ea51bc234b769bb1c234368f5d4bd8.
Native source manifests under build_tracker/data/exports/winprob-causal-student/
causal-student-assessment-20261004-v1:decoded4b9c7fe74d01c4a2a1a19543b8544fa9a84ff5df274b9724072c18bfd8e91d92,
native17a728f2cdbd0052a664de288015c69ef06ffc921feabf8fd34d35a719ebe8099b.

Written docs/causal_student_result_v1.md and updated README with the negative
result. Original index SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46
reverified unchanged. No forward859trial,1911/source/global sealed reads,
new collection or serving. M2 remains the stronger measured reference. The
requested breakthrough has not been established. Further work must reconsider
the practical M2 improvement mechanism; do not equate a reproducible weaker
standalone model or another small correction with completion of the goal.

## 2026-10-04 — native class-bias calibration prototype, no real fit

Previous continuation: PROGRESS, six-student family fully audited and closed
negative. Current continuation: PROGRESS, primary-source review, a new explicit
mathematical hypothesis, executable reference kernel and prospective protocol.
No running learner/assessment task was found at the latest22:57Madrid check.
No new real-purchase score, class-frequency residual diagnostic or fit occurred.

Reviewed primary papers: Kull et al. Dirichlet calibration (NeurIPS2019),
Lipton et al. BBSE (ICML2018), Alexandari et al. calibrated label-shift maximum
likelihood (ICML2020), and Zhao et al. decision calibration (NeurIPS2021).
Links and applicability are in the new protocol. Do not claim p(x|y) is stable
across patches or apply unlabeled adaptation to seals. Source-history search
found a proposed per-basket frequency offset and cardinality diagnostics, not
evidence that this exact calibration trial already ran. The September21v3
dossier explicitly withdrew its irreducibility reading from aggregate/group
calibration; retain that correction.

New method: q(a|x) proportional to M2(a|x)*exp(b_a), one constant per9333
counted main basket. This is the W=I subfamily of softmax(W log p+b), not full
Dirichlet calibration, a new League theorem, or identified label-shift transport.
It retains M2's complete native signal; no weaker independent predictor is fit.
Targets stay actual purchases, without balancing rare classes or changing
semantics. It inherits the native reference's legacy-role/live limitation.

Objective F=sum_i log(sum_a p_i(a)exp(b_a))-n dot b+.5||b||^2, with TRAIN class
counts n and fixed unit ridge. Hessian=I+sum categorical covariances gives a
unique minimum and numerical gradient-based gap/distance bounds. Positive
actual target support must be checked before counts are aggregated. Zero bias
returns exact float64-normalized native probabilities; zero support is retained.
Counts, gradients and optimum are invariant to permutation of TRAIN labels
that preserves counts, so label shuffling would be an INVALID negative control.
Use a predeclared permutation of fitted offsets within public basket-size
strata, seed20261004, as the diagnostic instead; it is never a candidate.

Implemented winprob/native_bias_calibration_v1.py with stable NumPy objective,
manual gradient/Hessian-vector product, fixed L-BFGS-B, guarded2000optimizer
calls plus one final gradient recheck, and the class-control permutation.
Source SHA5094bbaf38afebdc8ba995a485b7bd7583efe170094d794d2febb50f1eae271a.
Five tests in tests/test_native_bias_calibration_v1.py PASS: zero identity/zero
support/extreme logits, independent likelihood/finite-difference derivatives/
strong convexity, invalid label-permutation control, count-stratum permutation,
and a concrete lower-NLL/worse-accuracy example (3/5 first hits become0/5).
After adding the strict evaluation-call cap, the solver/counterexample test
was rerun and passed. These are synthetic functional checks, not M2 gains.

Frozen-before-real-fit protocol docs/native_bias_calibration_protocol_v1.md,
SHA8e91013fc3f637684605a84e43ed755e2bf3edf68bb5c57a5b83e56efdb6ce85:
same56288states/3518FIT16.17games, two whole-game folds and2900original focals;
exact full-prior artifact35113430... /matrixa2c4b149...; two fits only, b=0,
ridge1, maxiter500/maxfun2000/maxls20/maxcor20/gtol1e-6/ftol0, solver success
and max-gradient<1e-5. Both maps freeze before prediction. Sole candidate is
calibrated; control is class-permuted. Same native decoder and both legality
worlds;8700mutation checks and exact1483first/1959top3 M2 replay expected.
At least3pp first gain with paired95% lower>0, positive nonempty lower, no mean
top3/NLL worsening, zero illegality and all audits; control contrast separately
reported. Bootstrap2000games, seed20261004. Failed family closes without tuning.
Internal success would not complete the broader goal.

REMAINING IMPLEMENTATION, not yet run: bounded admitted prior/count reader,
GPU objective (256-row scratch, current TRAIN fold log-priors only), CPU/GPU
parity and dummy-label resource preflight, hidden fit/predict worker, source
decoder adapter and complete evaluation/replay/metric audit. Do not use the
NumPy reference on the full matrix in the agent shell. Resource policy CPU1,
host2.5/.75GiB, idleCUDA3072MiB,3hour worker; preflight bound is
1.5*max_objective_seconds*4002+900 <=80%remaining, plus1GiB device headroom.
Before labels freeze inputs/code/control map and verify all reservations.
Reuse existing versioned artifacts; no source runtime import, new collection,
859training,1911/source/global sealed reads, index edits or promotion.
Goal ACTIVE/UNACHIEVED. Do not revive any of the closed student/route/history
families or confuse a convex numerical certificate with practical improvement.

## 2026-10-04 21:31 UTC — full native-bias pipeline prepared; game guard active

Previous continuation: PROGRESS (reference math, five tests and fixed protocol).
Current continuation: PROGRESS (bounded data reader, GPU objective, fit worker,
native adapter and complete assessment implemented). No real fit or score yet.

New own files:
- winprob/native_bias_data_v1.py: RecipeData admission before payloads, exact
  full-prior binding and BoundedNpyRows(max256), fixed class-count control,
  explicit opposite-fold count requests, paired positive target support and
  exact16selected states per TRAIN game. Inputs lock precedes actual labels.
- winprob/native_bias_gpu_v1.py: deterministicCUDA FP64, only current TRAIN
  fold log-priors resident,256-row scratch, analytic value/gradient and explicit
  close. No automatic CPU fallback. New tests/test_native_bias_gpu_v1.py uses
  a synthetic513-row/17-class case, partial batches, zero support and wrong
  count rejection. This GPU test is PREPARED BUT NOT EXECUTED.
- scripts/fit_native_bias_v1.py: two small CPU/GPU parity checks per fold,
  two warmups/five full calls, conservative headroom/time gate BEFORE labels;
  exact fold counts and identities, both bias fits frozen before all5800
  predictions and2900baseline rows. Actual-label table released before fit/
  inference; only appropriate TRAIN aggregate counts enter each fit. Fixed
  objective/optimizer/protocol unchanged. No speculative gain is recorded.
- run/register_native_bias_fit_job_v1.ps1: hidden3hour worker, startup2.5GiB
  host/3072MiB GPU, live.75GiB floor, no League/capture/protected workload.
  Registration explicitly refuses an active game. Proposed Version
  native-bias-fit-20261004-v1, NOT registered.
- winprob/native_bias_io_v1.py (CompletedNativeBias),
  winprob/native_bias_evaluation_v1.py, scripts/evaluate_native_bias_v1.py,
  scripts/audit_native_bias_models_v1.py and audit_native_bias_metrics_v1.py.
  Model audit rechecks TRAIN counts per recipient fold, identity/support,
  current GPU gradients and convex numerical bounds, then exact5800forecast/
  2900baseline replay. It explicitly acknowledges opening the TRAIN ledger
  for per-model count verification; it does not falsely claim no evaluation-
  game label appears anywhere across both cross-fits. Labels are released
  before query replay. Current-query labels never enter prediction.
- run/register_native_bias_assessment_job_v1.ps1: model audit -> native18
  decode -> isolated17audit -> score -> metric audit. Proposed Version
  native-bias-assessment-20261004-v1 with PredictionVersion
  native-bias-fit-20261004-v1. NOT registered. Source artifacts under
  build_tracker/data/exports/winprob-native-bias plus assessment version.

Source status.py read21:21UTC before writing NEW
build_tracker/scripts/decode_native_bias_v1.py. DailyReady, no active source
job, no served artifact; unrelated expired API key left alone. Adapter uses
unchanged source runtime, bounded probability reader, three arms
m2/calibrated/permuted, zero-bias originalfloat32 path,8700mutation queries
and two legality worlds. No old source script, runtime or benchmark edited.

Validation: all four new PowerShell launchers parse; all new Python modules,
workers and source adapter compile. Three tests in
tests/test_native_bias_evaluation_v1.py PASS: control remains diagnostic and
ineligible even when better than the candidate, NLL/legal guards cannot be
ignored, and fold requests are explicit/nonempty. Combined with the previous
five math tests,8CPU tests pass. The synthetic GPU test remains unverified.
An rg invocation with Windows literal wildcard paths failed; the corrected
directory/-g search succeeded, finding only the intended8700native count.

RESOURCE STATE: at21:09UTC the GPU test command stopped at its very first
guard, BEFORE any GPU test or training work, because League of Legends PID3300
was active (start21:07:09UTC). At21:21/21:31UTC it was still live; host freeRAM
2.107/2.276GiB, also below the2.5GiB startup threshold. No protected experiment
task is running and no native-bias run directory/task was created. Do not ask
the user to stop their game or run around these guards. The earlier4.33GiB/
6541MiB idle snapshot is stale. Default exec remains broken with apply-deny-read
ACL helper; read-only escalation works. This is not an account-usage limit.

NEXT: check current League/resource/process state. Once idle and resources
qualify, run the prepared synthetic GPU parity test first (CPU/GPU tolerance
2e-10). Then register_native_bias_fit_job_v1.ps1 -Version
native-bias-fit-20261004-v1. Inspect its actual handle/preflight; no targets
before the preflight passes. After a completed fit and confirmed process exit,
run prepared assessment. Do not edit consumed files, retune on scores, replace
M2 with the failed student, open seals, touch the original index or promote.
No breakthrough or new M2 improvement measured. Goal ACTIVE/UNACHIEVED.
