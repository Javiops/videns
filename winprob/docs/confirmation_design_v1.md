# One final check of the frozen rich-state win-probability model

Prospective design only. Writing this document reads no outcome, metric,
holdout manifest, raw payload or model artifact. Counts and candidate names
below are recorded instructions for the planned check, not independently
measured results. No confirmation has been performed by this document.

## Question and entry condition

Does the already selected rich-trajectory model improve retrospective
win-probability forecasting over the original trajectory model and the
original gold/time logistic baseline on this project's own reserved games?

Proceed only if the frozen rich-trajectory candidate passes its predeclared
primary development gate and all source/feature/negative-control integrity
checks. The candidate is that exact fitted and calibrated artifact. A failed
development gate does not authorize trying another candidate on confirmation.
No confirmation result selects the model, feature set, calibrator, horizon,
time band, patch, loss, weighting, clipping constant or reporting threshold.

This is a forecast-quality check. It does not estimate action effects, which
purchase is better, preventable mistakes, coaching usefulness or improved wins.
A positive result does not, by itself, complete the broader research goal.

## Reservation and honest independence claim

The planned population is **1,911 own `confirmatory_reserved` IDs** in the
second admission manifest, with the original reservation preserved. Verify
that count and each surviving ID's original reserved role using metadata
before any payload is opened. A mismatch is an admission failure, not a
reason to silently redefine the cohort. Preserve the full original reservation
ledger, including any previously documented metadata-only exclusions.

The known **3,118 existing global sealed-test IDs** remain forbidden. The
actual denylist is the current union of all authoritative test reservations,
including external project exports; it may be larger than that recorded
count. It overrides every allowlist. Own confirmation must be disjoint from
that union and from historical validation IDs, and from every game used to
fit, calibrate, select or develop any of these WP candidates. Do not open
existing global tests even for unsupervised feature extraction.

These are source-project training games deliberately reserved before this
WP program's payload access. Describe them as **previously unexposed to this
WP development/evaluation**, subject to the access audit. Do not describe
them as never used by any other project or model. Source-project work may
have used them, and repeated players may occur across temporal partitions.
The new check is game-disjoint and temporally later, not player-disjoint or
a randomized deployment experiment.

## Binding record required before payload access

Write one immutable-by-contract `confirmation_binding.json` in a new versioned
output directory, then compute its SHA256 and reference it in every output.
No placeholder, unresolved path or missing digest may reach the evaluator.
The binding must contain these exact categories of evidence:

1. **Protocol:** this document's absolute path and SHA256, protocol version,
   candidate name `rich_trajectory`, the fixed comparator names
   `original_trajectory` and `original_logistic`, and the successful development
   report path/SHA256 that authorized this one candidate.
2. **Admission:** absolute path/SHA256 of admission v2, its SHA256 sidecar,
   its prior-manifest binding and prior reservation digest, and the exact
   confirmation membership file and digest. For the ID-set digest use
   `SHA256(UTF8("\n".join(sorted(ids)) + "\n"))`. Record each ID's frozen role,
   patch, creation timestamp and source-project membership. The expected own
   reserved count is 1,911; do not sample a favourable subset.
3. **Reservations:** the exact current split-metadata paths and digests,
   external-reservation artifact/source bindings, forbidden test and historical
   validation ID-set digests/counts, and zero intersections with own
   confirmation. Run the existing metadata-only current-reservation verifier.
   Repeat it before extraction batches and at completion; changes stop access
   and invalidate completion until the population integrity is resolved.
4. **Exposure:** fit/calibration/development source-manifest and cache digests
   for original and rich WP runs, their game-ID-set digests, and empty
   intersections with confirmation. Bind a metadata audit of prior WP payload
   access, including quarantined/failed runs: reservation exposure in any such
   run cannot be erased by discarding the run. Reject QUARANTINED.md ancestors
   for candidate models, caches and source artifacts.
5. **Models:** absolute paths and compressed-file SHA256 digests of all three
   frozen artifacts, their training/calibration manifests, feature-order
   schemas, transformations and calibrators. Record the exact rich schema
   version/width rather than assume that a stated feature count identifies it.
   The same applies to original trajectory and logistic input projections.
6. **Executable environment:** SHA256 of evaluator, feature extraction,
   admission and metric code, relevant imported prediction code, dependency
   versions, fixed random seed 20260929 and 2,000 bootstrap resamples. Bind
   the selected snapshot of compressed match/timeline file paths. Hash each
   source file as part of its single logged read; store the observed compressed
   digest and reject identity/schema mismatches before that game is scored.

The binding is complete before labels are read. Model files are trusted only
after their hashes and the prediction implementation match the bound training
records. The evaluator may call prediction and fixed transformations only;
there is no fit, calibration, model selection or label-dependent adjustment.

## Access ledger and one-shot execution

Create an append-only confirmation access ledger before the first payload
read. Each entry records a monotonically increasing event number, UTC time,
run/binding digest, match ID, endpoint, intended operation, and status. Record
`read_started` **before** opening the compressed file, followed by its SHA256,
read success/failure, feature qualification and scoring completion. Winner
labels are accessed separately from feature construction. Seal the completed
ledger with a SHA256 and include it in the report. Unknown prior access or
unlogged payload access prevents an unexposed-confirmation claim.

Read the full admitted reservation once. Never silently replace missing,
corrupt or disqualified games with other games. Use the frozen extraction
rules from development; missing frames and source/schema failures receive
explicit reasons and counts. Unexpected extraction or integrity failures
stop the run. They are not repaired against exposed outcomes and rerun as
another independent confirmation. A pre-access failure that opened no payload
can be corrected without consuming the reservation; the ledger must prove it.

Use the existing one-thread, bounded, resource-gated launcher and a new output
directory. Never alter a frozen upstream run, stop another project's job or
read a source-project database for new outcomes. No training is part of this
job. There is no scheduled repeated confirmation or secondary candidate queue.

The eligible frame population is the frozen rich extractor's qualifying
nonterminal [3,40]-minute frames. All three models score **the identical
game/frame keys**. Existing baseline features must be reproduced exactly by
the declared original-feature projection. Report any original-versus-rich
qualification difference rather than comparing unequal populations. Do not
invent inputs for missing rich features; apply only the missingness semantics
already fixed during training. GAME_END, final duration, winner and final
participant totals cannot enter any input vector.

## Fixed metrics and decision rule

Primary comparison: rich trajectory minus original trajectory. Original
logistic is a mandatory reference, not a substitute if the primary fails.
Use one blue-team probability and one outcome per state, keeping all states
and perspectives of a game together. Give each eligible game equal weight,
then its qualifying frames equal weight, exactly as in development.

Use the existing frozen log-loss/Brier implementation and its numerical
probability clipping of `[1e-7, 1-1e-7]`. Compute each model's per-game mean
loss, then paired reference-minus-candidate differences. Positive means rich
trajectory improves. Resample whole games 2,000 times with seed 20260929;
report the 2.5th and 97.5th percentiles of the paired mean differences.
The relative log-loss reduction is paired mean log-loss improvement divided
by the reference's game-weighted mean log loss.

The **useful forecast confirmation gate** requires all of the following:

- Against original trajectory: at least **0.5% relative log-loss reduction**,
  with the paired **95% log-loss interval strictly above zero** and the paired
  **95% Brier-improvement interval strictly above zero**.
- Against original logistic: paired **95% log-loss and Brier-improvement
  intervals strictly above zero**. Logistic cannot rescue a failed trajectory
  comparison. This is a conjunctive gate, not a search for whichever metric
  or comparator happens to win.
- All source, feature, exposure, common-population and digest checks pass.
  Report every planned/loaded/qualified/excluded game and frame count and the
  final exact comparison denominator. A failed integrity check is invalid
  evidence, not an inconclusive statistical result.

Report point estimates and intervals even when the gate fails. Distinguish
failure of the useful-size criterion, an unresolved interval and an invalid
run. All conclusions apply to the eligible complete-data reservation and the
fixed fitted models; bootstrap intervals do not represent training-run
uncertainty or causal effects. No new sample-size threshold is selected from
the results, and excluded cases cannot be silently treated as missing at random.

Calibration diagnostics are mandatory overall, by patch and in the fixed
time bands [3,10), [10,20), [20,30), [30,40] minutes: ECE with 20 bins,
reliability observations, mean predicted/observed probability and game/frame
counts, alongside proper scores. Report a slice as unmeasured when fewer
than 30 games qualify. Disclose adverse or unresolved patch/time patterns;
do not claim universal transfer from a pooled gain. These diagnostics cannot
be used to recalibrate or select a favourable patch or band on confirmation.

Persist each scored frame's game key, timestamp, patch, outcome and all three
frozen predictions, the exact population/selection records, aggregate report,
artifact/source/code digests and the sealed access ledger. A read-only
independent arithmetic check may use those same persisted predictions. That
is verification of the one result, not another candidate evaluation or another
opening of the reservation. Never rewrite the original report after such a
check; append any correction with its reason and provenance.

## What passing would enable, and what still remains

A pass would establish a measured improvement in retrospective OMNI forecast
quality on a separately reserved WP population. It would justify retaining
the frozen rich-state judge and evaluating its use in a review workflow via
versioned artifacts. It would not establish a practically useful new decision
recommendation or authorize policy promotion, a live-POV claim or a causal
value label.

Practical product evidence still needs an explicit task and an appropriate
comparator: for example, a blinded review study testing whether the richer
forecast helps users locate independently judged actionable moments more
accurately or with less review time than the original/gold-time view. A
retrieval study must include the simple latest-window/time baseline exposed
by setback v1, and it cannot claim usefulness by beating only a weak trend
rule. Any proposed action-value intervention needs its own identification,
support and prospective validation plan; prediction alone does not supply it.

Keep the broader goal active after a forecast-only pass. Record exactly the
benefit established, preserve the frozen result, and pursue the missing
product evidence without repurposing this consumed reservation as a new
development set for another supposed final check.
