# Rich-state matched-cohort experiment v1

Fixed before enriched feature extraction or fitting. This is an exploratory
follow-up on the same exposed development cohort, not confirmation. The aim
is a useful improvement in forecasts and automatic retrieval of upcoming
scoreboard setbacks. No action-value, causal, coaching-benefit or win-rate
improvement claim follows from passing these gates.

## Population and source contract

Use exactly the 6,000 games and every qualifying frame in the clean W1 v3
cache. Keep the original fit, calibration and development roles, numeric
game indices, timestamps, labels and five-minute fit sampling. No new game,
archive lookup, split boundary or test/confirmation payload is admitted.
Historical validation and the six external macro reservations remain barred.

Require the original cache metadata sidecar, source manifest, admission
manifest/sidecar, external reservation sources and all prior frozen roles
to validate. Recompute current forbidden test/validation membership and
reject quarantined inputs. Existing W1 and setback artifacts must bind the
same clean cache and unchanged implementations. Before every raw payload is
decoded, its compressed bytes must match the original source-manifest hash,
size, endpoint and identity. Read one original game at a time in source order.

The rich extractor must return exactly the same game/frame population. All
19 original feature values, cast to float32, must equal the old cache exactly;
labels, side-independent metadata and timestamps must also match. Missing
new fields use the feature module's declared availability mechanism, never
a changed population or future frame. Any mismatch fails the experiment.
The enriched matrix is saved in a new directory with source and code digests.

## Fixed hypotheses and models

R1: additional as-of state and past information improve calibrated WP beyond
the original frozen W1 trajectory model. Fit two independent fixed models:
rich snapshot on the explicitly declared snapshot feature subset, and rich
trajectory on all declared rich features. Both use W1's existing 150-tree
recipe: 15 leaves, learning rate .06, L2=10, minimum leaf population 100,
no early stopping, seed 20260929. Fit the identical hash-selected state in
each game/five-minute bin, with equal game mass. Calibrate by the existing
sigmoid recipe on calibration games only. The original trajectory stays
frozen and is evaluated on the identical development rows.

R2: rich past-and-present features improve three-minute setback retrieval
over BOTH stronger existing fitted controls (direct risk with the original
history features and direct snapshot logistic risk) AND the time-only
ranking that takes each game's latest two eligible windows. The time-only
comparator was identified after the prior experiment's results; it is a
diagnostic addition declared before this enriched run, not a retrospectively
prespecified control for the old result. Fit one rich direct risk model
using the same 150-tree recipe and the exact existing complete windows,
labels and partition memberships. Fit and calibrate only on their respective
partitions. No WP-generated feature enters this risk model.

The setback definition, fixed focal team, start times and two-window budget
are inherited unchanged from setback_protocol_v1.md. Original difference
and slope features, plus new signed differences, flip for a red focal team.
Sums, counts, availability indicators and other expressly invariant fields
do not flip. The feature module's complete signed-feature list is frozen
before fitting; append focal_is_blue. Reuse the original controls' persisted
development scores only after verifying their digest and exact window,
label, focal-side and selection alignment. Do not refit or select controls.

## Metrics and gates

WP: game-weighted log loss and Brier score with 2,000 paired whole-game
bootstrap intervals. Report ECE20 and paired ECE20 differences with the same
cluster resamples, alongside reliability bins. Compare both rich candidates
with the frozen original trajectory, and rich trajectory with rich snapshot.
The primary WP gate is rich trajectory versus original trajectory: at least
0.5% relative log-loss reduction, paired log-loss interval strictly above
zero, and paired Brier improvement interval strictly above zero. Positive
paired reductions mean improvement. ECE is diagnostic and bin-sensitive.

Retrieval: retain the exact existing macro recall among positive games and
precision over all selected windows, including games with no setbacks.
Use 2,000 paired whole-game bootstrap resamples. The primary gate must pass
against all three controls: at least 100 positive development games, at least
10% relative recall improvement, paired absolute recall-gain interval
strictly above zero, and point precision loss no worse than .02. Report
precision intervals; the point guard is not a noninferiority test. Also
report probability diagnostics and patch comparisons with at least 30 games.
WP and direct risk proper-score comparisons are additionally reported in
fixed start-clock bands 3–10, 10–20, 20–30 and 30–40 minutes, where at least
30 games exist. This separates prediction quality within clock ranges from
a ranker's tendency to select later opportunities. The latest-two clock
score is not a probability and receives no proper-score diagnostic. All
retrieval is retrospective on eligible complete replay windows; candidate
pool membership contains hindsight, so no early-warning benefit is claimed.

All declared methods and comparisons remain visible. No hyperparameter
search, feature selection based on development results, or revised endpoint
is permitted within this run. Passing is an exploratory forecasting or
retrieval result, requiring a later untouched confirmation and a separate
human-usefulness evaluation. All existing confirmation games stay sealed.

## Execution and artifacts

Use one CPU thread, BelowNormal priority and no GPU. Default wall budget is
40 minutes; resource checks require at least 2.5 GiB available memory before
work, every 50 raw games and between model/evaluation phases. The external
launcher enforces the hard deadline and sibling-workload gate. Status updates
record bounded progress. Start only in a new output directory.

Write rich_states.npz, experiment_lock.json, three fitted model artifacts,
matched development predictions, report.json with SHA256 sidecar, and
status.json. Freeze source, model, implementation and protocol digests before
work and recheck them before success. Existing caches, reports, app files,
website index and sibling project code remain untouched.
