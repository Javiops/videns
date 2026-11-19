# Rich-state mechanism ablation v1

Declared before this ablation fits or reads performance results. The existing
development population is already exposed; every result is exploratory.
The question is whether richer WP predictions add information beyond knowing
the absolute amount of gold in the game. A positive full-model result alone
does not identify geometry or player allocation as its source.

Use the completed, digest-bound rich run and its exact 6,000-game matrix.
Keep the original roles, every frame, labels, five-minute fit sampling,
equal-game weights and separate calibration games. Validate the rich report
sidecar, all bound implementations/inputs/model files, original cache/source
metadata, current reservations, and exact original-19 feature projection.
Reject quarantine ancestry before opening an input. No raw archive payload,
database, test game or confirmation game is opened by this experiment.

Three new fixed arms use the original 19 features plus:

- **total:** only `rich_gold_total_sum`.
- **allocation:** that total plus both difference and sum for gold standard
  deviation, quartiles 25/50/75, maximum player share and top-two share.
- **geometry:** every remaining rich feature, including geometry, gold-position
  coupling, counts and availability, but excluding the allocation statistics
  above and `rich_gold_total_sum`. This block cannot separate geometry from
  its declared availability and resource-position coupling; do not claim it can.

The combined arm is the already fitted rich trajectory model. The reference
is the already fitted original trajectory model. Reuse their digest-bound
development probabilities; do not refit or select either control. The union
of allocation and geometry feature sets must equal the full rich feature set.

Each new arm uses the unchanged `CalibratedModel` histogram-boosting recipe:
150 iterations, 15 leaves, learning rate .06, L2 10, minimum leaf population
100, no early stopping, seed 20260929. Fit the identical deterministic state
per game/five-minute bin; apply the same separate sigmoid calibration.
No radius, model, feature, threshold or hyperparameter search.

Report all five arms' game-weighted log loss, Brier, accuracy, ECE20,
reliability bins and 2,000-resample whole-game intervals where implemented.
Report raw and calibrated probabilities for new arms; raw control predictions
are unavailable in the supplied frozen prediction artifact and are not invented.

Fixed comparisons: each new arm and combined versus original; combined versus
each new arm; allocation versus total; geometry versus total. Positive paired
log-loss/Brier/ECE reductions mean improvement. ECE gets paired game-bootstrap
differences as well as point estimates. Also publish all fixed comparisons by
patch and by 3–10, 10–20, 20–30 and 30–40 minute bands when at least 30 games
exist; otherwise retain an explicit insufficient-population record.

Mechanistic reading is limited to these fixed trained feature-set contrasts.
If combined does not improve on total, the results do not establish extra
geometry/allocation value beyond absolute wealth. Allocation versus total and
combined versus allocation address distinct incremental questions. Nonlinear
interactions mean the gains need not add; these are not causal attributions.
All signs remain visible. No score from this run is called confirmation and
no model is promoted. Any future confirmation candidate must be frozen after
this analysis under a separate protocol.

Use one CPU thread, BelowNormal priority, no GPU, an external hard-deadline
launcher and a 20-minute internal deadline. Read only derived caches and
provenance metadata. Write a new output directory containing the experiment
lock, three model artifacts, matched development probabilities, report and
SHA256 sidecar. Recheck all bound inputs/code before publishing success.
