# Privileged future-state distillation for current win probability

2026-09-30. Fixed before constructing new teacher targets, fitting a teacher
or student, or computing any new development comparison. This is a distinct
WIN-probability experiment; prior purchase/auxiliary-goal recipes stay closed.

## Mathematical question

Let Y be the terminal blue-team win, X the61 currently available retrospective
rich-state features, and Z a recorded state three clock-minute bins later.
For the true conditional mean m(X,Z)=E[Y|X,Z], the tower property gives
E[m(X,Z)|X]=E[Y|X]. For a current-only forecast f(X),

    E[(f(X)-Y)^2] = E[(f(X)-m(X,Z))^2] + E[Var(Y|X,Z)].

The cross term vanishes conditional on(X,Z). Also
Var(m(X,Z)|X) <= Var(Y|X). Soft-label cross entropy against the true m has
the same expected risk as cross entropy against Y because it is linear in
its target. These facts motivate a less noisy TRAINING target; they do not
add observations at inference. An ESTIMATED teacher has bias/variance and
can make the student worse. Cross-fitting removes own-game teacher fitting;
it does not make an estimated teacher equal the true conditional expectation.
Both teacher types retain X. A future-only teacher need not satisfy the
tower identity because its information need not contain X.

Training-only information is motivated by Lopez-Paz et al.,
*Unifying distillation and privileged information*, ICLR2016:
https://leon.bottou.org/publications/pdf/iclr-2016.pdf.
The following is our fixed tabular adaptation, not a novel statistical theorem
or a claim that this paper demonstrated a League-specific improvement.

## Frozen population and inputs

Use the completed `rich-20260929-v1/experiment` artifacts and the clean W1v3
ancestor, NEVER W1v2. Required rich cache SHA256:
65414f41331d4a82601ef38d9a95ff4b465bc90a758629730683593a93e729cb.
Required rich report SHA256:
f5e566b4187f02858c7fecdf7fb9ff64d689336461235d5080102d12d4f4808d.
Reuse all61 rich features, including current geometry/allocation and the
already causal past gold slope. This is retrospective OMNI, not live-safe.

Keep all138831 frames:92294FIT frames/4002 games,23171 calibration frames/1000
games,23366 development frames/998 games. The existing deterministic
game/five-minute frame sampler supplies exactly22554 student fit frames.
Keep its seed20260929 and indices unchanged; no outcome/time-window filtering.
The development cohort is repeatedly exposed, not fresh confirmation. All
1911 confirmation games and global/external sealed reservations remain closed.

Before NPZ loading, validate clean source/admission metadata, current full
reservation union, quarantine exclusions, source role/patch/time identities
and bound hashes. Reuse the immutable rich verifier and add explicit current
external/confirmation reservation checks before and after work. No new raw,
database or sibling-project code imports. Existing NPZ containers contain
all admitted roles' labels; the inherited verifier loads/compares them for
integrity. This is not a claim that their bytes were never deserialized.
External calibration labels are used only for final student calibration;
development labels only for final evaluation/integrity, never fitting or
teacher target construction. No new analysis of reserved labels is allowed.

## Privileged state alignment, FIT only

Construct teacher matrices only for FIT rows. Group by exact game and
floor(timestamp_ms/60000). For each current FIT row, use the earliest
qualified frame in clock bin current_bin+3, following the existing clock-bin
convention. Do not interpolate, select a nearer other-bin state, or cross a
game boundary. Distinct timestamps within a bin remain current rows; duplicate
exact(game,timestamp) keys fail. Report actual elapsed-time quantiles because
three clock bins are not an assertion of exactly180000ms.

If the future bin is unavailable, retain the current row, zero-fill the
future61 block and set future_available=0. Do not infer an absorbing winner,
game-ending reason or post-cache state. For available rows the flag is1.
The joint teacher has123 inputs: current61, future61, availability1. The
current teacher has only the same current61. The missing flag is privileged
training information. Students receive neither future features nor that flag.

Publish fit-frame alignment, selected fit indices and whole-game fold ledgers
before fitting any teacher. A synthetic mutation of external calibration/
development features must not change FIT joint matrices; mutations of outcomes
must not change feature alignment. No construction or validation condition
drops a current FIT frame because its future is missing.

## Four-way teacher cross-fitting entirely inside FIT

Assign each FIT game to fold int.from_bytes(SHA256('wp-distill-v1:'+match_id)
[:8],'big') modulo4; no label influences assignment. For target fold k,
teacher calibration uses fold(k+1)mod4, and teacher fitting uses the other
two folds. All three game sets are disjoint. Teacher fitting uses the original
sampled frames in its two fitting folds; teacher calibration uses every FIT
frame in its calibration fold with equal-game/equal-frame weights. Predict
all frames of target fold k. Repeat for k=0..3 and for both teacher types.
Every FIT row receives exactly one calibrated OOF prediction from each type.

Each teacher is the existing CalibratedModel histogram classifier:150
iterations,15 leaves,.06 learning rate,L2=10,min leaf100,no early stopping,
random state20260929; equal games, equal selected frames within game, weights
rescaled to mean1. Platt calibration is the existing LogisticRegression on
clipped logits(C=1e6,max_iter500,seed20260929), with equal-game weights.
No warm start, architecture search, horizon search or teacher selection.
Save all eight teachers, membership/fit/calibration/prediction receipts, OOF
targets and the declared role constraints. Teacher OOF metrics are descriptive
diagnostics, not permission to select a successful fold or stop an arm.

## Three matched current-only students and a strong frozen reference

Fit three fresh HistGradientBoostingRegressor models, squared-error loss,
150 iterations,15 leaves,.06 learning rate,L2=10,min leaf100,early_stopping=False,
random_state20260929, using identical22554 current-only61-feature FIT rows and
equal-game normalized weights. Targets are fixed by arm:
- hard: original terminal Y;
- current_teacher: current-state teacher's calibrated OOF probability;
- future_teacher: joint current/future teacher's calibrated OOF probability.

There is no target mixing coefficient. All three students use the same
external1000-game Platt calibration recipe as the original models. Clip raw
regressor outputs to[1e-7,1-1e-7] before logits; report raw clipping rates and
uncalibrated proper scores, without choosing raw versus calibrated after
results. The inference class accepts EXACTLY61 current columns and never
loads/calls a teacher or alignment artifact. Save students separately from
training-only teachers. Synthetic tests reject a123-column inference matrix.

Additionally retain the immutable published rich_trajectory/combined
classifier as a strong reference. Its development probabilities and the
matched23366 row identities come from the bound existing artifact; no refit
or weakening of that reference. Its log-loss0.4943414406 and Brier0.1629214542
are already exposed results. New students must beat this reference as well
as the matched hard-target regressor.

## Evaluation and fixed useful-signal gate

Use all998 development games and23366 current frames. Equal game mass, equal
qualifying frames within each game. Reuse the unchanged game-weighted metric
kernels and2000 paired whole-game bootstrap resamples with seed20260929.
For relative loss intervals, recompute the ratio of aggregate paired losses
inside each resample. Intervals condition on these fitted models; they do
not include retraining uncertainty or establish fresh confirmatory evidence.

The candidate is future_teacher. ALL conditions must hold:
- Complete original population, fixed fit sample, OOF teacher coverage and
  strict teacher/student role separation/inference/provenance checks.
- At least2% relative calibrated log-loss reduction versus BOTH hard and the
  frozen rich classifier, with each paired95% log-loss reduction lower bound>0.
- Paired95% Brier reduction lower bound>0 versus BOTH those controls.
- Paired95% calibrated log-loss reduction lower bound>0 versus current_teacher.

Report all arms, raw/calibrated proper scores, ECE20/reliability, fixed3–10,
10–20,20–30,30–40minute bands and patches (subgroup metrics only at>=30 games).
These are secondary; none can rescue a failed primary gate. No post-result
mixing, horizon/calibration/feature/weight/checkpoint change or cherry-picked
time band. No future teacher is used to score a current development frame.

## Execution and scope of a positive result

CPU1, no GPU,600-second complete stage cap,2.0GiB startup,.75GiB live host
floor, no game/competing benchmark and a hidden durable one-shot task. Bind
all called own code, input hashes and this protocol before fitting; recheck
before publication and retain failures without overwriting. Store model,
prediction, alignment/fold and calibration receipts in a new run only.

A pass establishes an internal forecasting signal, not better purchases,
causal WPA attribution or breakthrough completion. It would justify a frozen
confirmatory/practical review comparison before an integration decision. A
failure closes this recipe. Existing source code, tests, models, confirmation
data and informes/20260929/index.html remain unchanged; no automatic promotion.
