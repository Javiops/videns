# Short-horizon setback retrieval — protocol v1

Declared before this experiment reads its labels. This is a development
experiment, not confirmation or evidence that any decision causes wins.

## Hypotheses and fixed endpoint

H1: disagreement between W1's snapshot and trajectory forecasts can retrieve
upcoming material setbacks better than simple scoreboard or trend rankings.
The fixed candidate score is focal-team snapshot WP minus trajectory WP.
Both probabilities are evaluated on the original blue-oriented input, then
complemented for a red focal team. We do not flip a blue model's inputs and
pretend that the map is symmetric.

H2: a direct short-horizon risk model with the existing past-only gold slope
can improve retrieval over a snapshot-only risk model. A full snapshot
logistic regression provides an additional strong, inexpensive risk baseline.
All methods are declared now; the best development method is not subsequently
called a confirmatory selection.

Each game's focal team is blue when the first byte of SHA256(match_id) is
even, red otherwise. No labels or player identity enter this selection.
Candidate nonoverlapping windows have nominal starts at minutes
6, 9, 12, ..., 36 and endpoints three clock minutes later. To accommodate
Match-V5 timestamp jitter, each clock-minute bin maps to its earliest actual
sampled frame. Adjacent windows reuse the identical observed endpoint/start.
Actual timestamps and durations are retained; no state is interpolated.

A window is positive when focal-team gold difference changes by <= -1500
OR focal-team kill difference changes by <= -3 between its observed endpoints.
These labels use realized scoreboard quantities, not a future WP judge.
The start features cannot read the endpoint, game outcome, duration, or
terminal metadata. Direct risk inputs orient all differences and the slope
to the focal team, retain time/availability, and add an explicit focal-blue
indicator so the model can learn real side effects rather than assume symmetry.

Only complete observed windows enter the primary estimand. A missing start
or endpoint is recorded as unavailable; it is never inferred to be terminal.
The report counts every planned clock slot, all exclusions, source games,
games with complete windows, and actual horizon durations by partition.
This population is retrospective complete replay intervals, not all live
decision states. Early-ending and missing intervals are an explicit coverage
cost. Two retrieved three-minute nominal windows constitute a six-minute
nominal review budget; games with fewer complete windows use less budget.

## Models, baseline selection and leakage guards

Use only a supplied clean W1 cache, its source manifest, and the W1 artifacts
fitted on that exact cache. Require matching SHA256 digests, complete cache
metadata, no test/confirmatory/historical-validation payload reads, and
consistent source-to-cache game roles. Quarantined inputs are inadmissible.
Reject any QUARANTINED.md ancestor before opening an input. Bind the original
admission manifest and its SHA256 sidecar from both cache metadata and source
manifest, revalidate current split/external/prior reservations, and verify
every cached ID and frozen role independently of payload-read booleans.
The script never opens raw archives or sibling-project databases.

The two cheap ranking baselines are worst preceding focal-team gold slope
and lowest current focal-team snapshot WP. Choose the stronger on calibration
windows by macro recall, then precision, then the fixed method name. Freeze
that choice before evaluating development. Each method ranks start-only
scores and retrieves at most two windows per game; ties use earlier actual
start time. Calibrated W1 outputs are ranking inputs, not calibrated
probabilities of the new short-horizon endpoint.

Direct logistic risk: standardized current snapshot features plus focal-blue,
C=1, max_iter=500. Direct snapshot and history risk: histogram gradient
boosting with 150 iterations, 15 leaves, learning_rate=.06, L2=10,
min_samples_leaf=100, early_stopping=False, random_state=20260929. The history
arm additionally receives only the existing gold-slope and availability
features. Fit on fit windows with equal game weight, then fit a sigmoid
calibrator on calibration windows using equal game weight. Neither direct
risk model consumes W1 predictions, avoiding in-sample generated features.
Require at least 100 eligible games and both classes in fit and calibration.

Freeze all methods before development labels are evaluated. The existing
chronological roles remain intact; all windows from a game share its role.
No hyperparameter search, new raw collection, GPU work or policy promotion.
At most 6,000 cached source games; one CPU thread and BelowNormal priority.
Deadline and free-memory
checks occur between bounded phases; an external gated launcher remains
responsible for hard process limits and avoiding active sibling workloads.

## Metrics and useful-signal gate

Primary recall is the mean per-game fraction of positive eligible windows
retrieved, among games with at least one positive eligible window. Precision
is total positive retrieved windows divided by all retrieved windows,
including selections in games without any positives. Also report equal-game
precision, the fraction of positive games with any positive retrieved, the
fraction of all eligible games with a positive retrieved, actual selected
minutes, and source coverage. Use 2,000 paired whole-game bootstrap resamples.
The paired difference is candidate minus reference on the identical games;
no independent bootstrap comparison or outcome-dependent method selection.

For each prespecified candidate versus the calibration-selected cheap baseline:
at least 100 positive development games, >=10% relative macro-recall gain,
a paired 95% interval for absolute recall gain strictly above zero, and a
point precision decline no greater than .02. Precision intervals are also
reported; the .02 guard is descriptive, not a noninferiority test.
If baseline recall is zero the relative-improvement gate is undefined and
does not pass. Insufficient positive games means underpowered, not failure
of the scientific hypothesis. H2's history-specific claim separately requires
the same gate against direct snapshot risk. All fixed comparisons and all
method results remain visible, regardless of signs.

Log loss, Brier score and calibration are diagnostics for the direct risk
probabilities. Report retrieval by patch where at least 30 eligible games
exist, with counts, and disclose directional reversals. Pooled improvement
does not establish generalization to a patch with unresolved or adverse
results. Multiple candidate comparisons are exploratory; no familywise
confirmatory inference is claimed.

Passing establishes an automatic retrieval/forecast benefit for an objectively
defined endpoint at a fixed nominal budget. It does not show preventable
mistakes, appropriate alternative actions, human review-time savings,
coaching usefulness, causal value or improved win rate. Those need separate
blinded review or prospective experiments. It does not complete the broader
breakthrough goal by itself.
