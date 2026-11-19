# W1: an honest value signal before a new policy

Protocol v1, declared 2026-09-29 before this project's outcome inspection or
model fitting. Status: exploratory development; no breakthrough claimed.

## Objective and research question

Build a reliable retrospective probability of winning and test whether recent
trajectory contributes useful information beyond the current scoreboard.
If it does, a review can distinguish an ordinary fluctuation for that state
from an unusually adverse sequence. This is a falsifiable route to better
review prioritization; it does not establish which action should replace it.

The practical goal remains improved decision evaluation and advice. A useful
predictive diagnostic is infrastructure and an intermediate result. A positive
development score alone will not complete the goal.

## Corrections to the initial dossier

1. M2 is a learned imitation policy. Its normalized grouped probabilities are
   estimates, not known human logging probabilities. Neither its accuracy nor
   its normalization establishes the calibration required for importance
   ratios. Historical action support and hidden confounding remain unverified.
2. Staying in M2's top three does not guarantee overlap with historical human
   actions. Report action support, clipping bias, effective sample size, and
   sensitivity before attempting observational off-policy evaluation.
3. A visit-level contextual estimate with terminal rewards addresses switching
   one decision with natural continuation. It is not an estimate of deploying
   a policy at every shop visit. Sequential evaluation needs its own estimand.
4. Minute-frame interpolation at a purchase reads a future frame. Use an exact
   existing frame for W1; later decisions use the latest preceding frame and
   events strictly before the decision, with explicit source ages.
5. Removing items alone does not remove item effects from currentGold or
   championStats. The first judge omits both; it remains an observational
   predictor, not an action-neutral counterfactual simulator.
6. Omniscient timeline positions do not identify what a player could see.
   W1 is offline OMNI only. A POV model awaits a separate observed-input spec.
7. Rank is not established as constant: the current status log mentions master,
   grandmaster and challenger. No rank proxy enters v1; population limitations
   must describe the admitted archive rather than assume challenger-only.

## Data and isolation

Inputs are explicitly admitted match and timeline archive envelopes, read-only.
No broad database event scan. Source metadata reservations exclude the union
of existing sealed test/holdout IDs before either payload is opened. Existing
training and development IDs are allowed; exposed development cohorts are
labelled as such. Never call an already scored 16.19 cohort fresh.

Pre-results amendment after the cross-project reservation audit: this union
also includes macro_coach test IDs through a versioned, digested metadata
artifact. `w1-pilot-20260929-v2` is quarantined for omitting that reservation;
its metrics were not inspected or used to select a model. The statistical
recipe and thresholds remain unchanged. See
`incident_20260929_cross_project_seal.md` for the actual access record.

Record source manifest hashes, per-file compressed SHA-256, missing/corrupt
sources, patch, match start time, qualifying game counts, feature schema and
code hashes. No model input may include game ID, PUUID, final duration, final
participant totals, GAME_END, winner, or final-item inventory.

Sample existing complete frames in [3,40] minutes; exclude any frame containing
GAME_END and all subsequent frames. Exclude games with invalid queue, missing
ten-player coverage, or inconsistent identity. Items and championStats are not
features. Events must be at or before the sampled timestamp. A frame can be
post-purchase for retrospective prediction; it must never be re-labelled as a
pre-purchase state.

All primary averages give each game equal weight, then its admitted frames
equal weight. Training samples one deterministic state per five-minute bin
per game. Evaluation uses all qualifying frames. This avoids treating a long
game as many independent games. No state from one game crosses a split.

## Population and selection

Use a frozen chronological split of eligible, explicitly admitted source-train
games: first 60% fit, next 15% calibration, next 15% development, last 10%
reserved for one later confirmatory check. Equal start timestamps stay on one
side of a boundary. Existing sealed tests are excluded from every partition.
The union of historical validation IDs is excluded from this split and may
later serve only as a separately labelled exposed transfer cohort. No former
validation game is fitted or calibrated. The new reservation is sealed after
admission metadata is frozen; do not open
its raw payloads or feature cache during development.

First bounded pilot uses a deterministic hash sample of at most 6,000 permitted
games from fit/calibration/development only, preserving their temporal roles.
It is explicitly a pilot. A final check and broader transfer evaluation require
a frozen candidate and a separately declared comparison, not repeated peeks.
If metadata cannot safely establish these populations, stop before data reads.

## Models fixed before fitting

- Constant: fitted training win frequency.
- Baseline: logistic regression on time, gold difference, kills difference,
  towers difference and dragons difference, standardized on fit only.
- Snapshot: histogram gradient boosting on the admitted current-frame features.
- Trajectory: same boosting recipe with past-only differences/slopes appended.

No architecture search. Fixed random seed 20260929. Histogram boosting uses
150 iterations, 15 leaves, learning rate .06, L2 regularization 10,
min_samples_leaf 100, early_stopping=False. Logistic C=1, max_iter=500.
Use sigmoid calibration on the separate calibration games, for all nonconstant
arms. Report raw and calibrated predictions to expose calibration effects.
No forced team symmetry unless it is separately measured: blue/red map-side
effects can be real, so complementary team perspectives share the same game
prediction rather than asserting that swapping arbitrary features is causal.

## Metrics and gates

Primary: paired game-weighted log-loss improvement of trajectory over snapshot.
Secondary: snapshot over logistic, Brier score, accuracy, and ECE (20 bins),
overall and by ten-minute band. Reliability plots include game counts.
Report 2,000-resample paired game-bootstrap 95% intervals. Confidence intervals
are conditional on the fitted model and population; they are not training-set
uncertainty and do not prove causal value. Exploratory slices are labelled.

Useful predictive-signal gate (development, not deployment): at least 0.5%
relative reduction in log loss for trajectory over snapshot, with a paired
95% interval strictly above zero; overall ECE must not increase by more than
0.005 (0.5 percentage points), a predeclared descriptive guard rather than
a significance test of calibration. Time-window calibration is also reported.
W1 infrastructure gate: snapshot clears the logistic baseline in paired log
loss and Brier score, and the reliability report reveals no unreported time
window failure. If trajectory is null, retain the simpler model and report it.

Negative control: permute outcomes at the game level within patch and start-
date bands, with all states in a game receiving the same shuffled label. A
control result outside its game-level chance uncertainty triggers inspection.
No row-level label permutation because it violates within-game dependence.

Mandatory invariants: future mutation cannot change past features; terminal
metadata cannot change features; labels are accessed separately; split
intersections are empty; bootstrap and point estimates share a population;
source digests and code digests remain fixed for an experiment.

## Resource bounds and stopping

Current machine has 15.8 GiB RAM and a live benchmark. No heavy run while the
benchmark or reconstruction is running. A one-shot launcher waits for those
jobs to finish, no actual League game process, and at least 2.5 GiB available
RAM. Use BelowNormal process priority, one CPU thread, no GPU, at most 6,000
pilot games and a 90-minute whole-run deadline. Never stop a sibling job.
Stop on admission/schema/leakage failure, stale hashes, memory pressure,
insufficient population, or deadline. Save explicit status and diagnostic;
do not silently shrink the comparison population or publish partial metrics.

No edits to today's index.html, frozen apps, served policies or databases.
No policy promotion, no automated in-game actions, no new Riot collection.

## Methods references

Probability calibration follows the separate-fit/calibration discipline in
the [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).
Fixed boosting parameters use the documented
[HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
The distinction between a contextual decision and a sequential policy matters
for [doubly robust sequential evaluation](https://proceedings.mlr.press/v48/jiang16.html).
The identification falsifications and their separate primary references are
in `ope_scope.md`.
