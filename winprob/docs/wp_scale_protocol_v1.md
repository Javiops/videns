# Fixed-recipe data-scale comparison for retrospective WP

2026-09-30. Declared before parsing any gameplay payload outside the original
clean6000-game pilot or viewing outcomes/predictions on its development
complement. Prior pilot gates and negative/uncertain results remain unchanged.
This study tests an increased admitted data budget and calibration budget;
it does not retune a failed feature or change an old result's gate.

## Scope and comparisons

Admissionv2 metadata contains11468train,2866calibration and2861development
games, totaling17195 requested games. Exclude all3282development_exposed and
1911confirmatory_reserved entries and the recomputed union of current/prior
external/global test/validation reservations. Require historical_validation
false and queue420 for every requested ID. Preserve original temporal roles.
No new collection. These are requested metadata counts, not a claim of raw
source availability or feature qualification.

Original pilot:4002FIT/1000calibration/998development games. The complement
contains7466train/1866calibration/1863development metadata IDs. Main new-model
evaluation is ONLY the newly evaluated development complement. It is outside
this project's pilot, not claimed globally untouched or confirmatory. Report
the old998-game development subset and the union separately as secondary;
do not pool them to rescue a failed primary result. All1911 confirmation games
remain closed throughout this scale study.

Freeze an explicit identity/role/patch/date scope and its digest before new
archive access. Hash each new compressed source before decompression and
publish source bindings before feature extraction. Reused pilot sources and
features retain exact previous digests. Log payload access and reject paths
escaping the admitted raw archive, quarantine ancestors, identity/patch/queue
mismatches and digest conflicts. No source DB access or sibling code imports.

## Data construction and exact pilot parity

Use the UNCHANGED `features.extract_game_states`, `features_rich` and
`features_current_combat_v1` definitions:19 base,61 rich and112 additional
current-combat features (full observed input173). No new feature or missing-
value rule. Use native public tag/static bindings already frozen. The original
extractor supplies qualifying complete recorded frames in[3,40]minutes and
excludes the GAME_END-containing frame and every later frame. No interpolation.

Pilot game indices0..5999 and their entire row order/values/roles/labels stay
unchanged. Append new qualifying games in(game_creation_ms,match_id) order;
never reindex old games. The original per-game/five-minute frame sampler,
including seed20260929, therefore chooses the identical original22554FIT rows.
New games add their own sampled frames. Keep all perspectives/frames per game
together. Equal game weights/equal selected frames within game, normalized to
mean1 for fitting/calibration as in the original code.

Missing archives and zero qualifying frames are explicit exclusions without
replacement, using the original cache rules. A malformed archive, metadata
mismatch, failed original pilot parity, or contradiction of a known end with
a qualifying frame aborts. Do not exclude on winner, model score, action type,
feature usefulness or an inconvenient output. Preserve an exclusion ledger
and role-specific requested/qualified denominators. No original pilot game
may be lost or have its frames changed.

The combined cache may reuse pilot arrays directly and stream new games into
versioned shards before bounded final packing. Verify exact pilot rich61,
combat112, timestamp, role, patch andY parity. Separate evaluation-only end
timing from all model matrices. Outcomes are extracted for supervised training,
calibration/evaluation as declared; final participant performance/roles are
never feature inputs. Health-positive/net-outflow remain proxies; observed
item/combat effects do not make a counterfactual item judge or live-safe model.

Apply the same current-combat quality thresholds to each full role:>=99%
observed health/wallet/valid health ratio,>=95% finite reported included combat
stats and>=90% participant series with varying health or healthMax. No quality
filtering of individual rows. Require at least11000 qualifiedFIT games,2700
calibration games and1700 development-complement games, and>=1000 complement
games in the at-least3-minute-to-end stratum, before scoring a useful signal.
These are declared scale/coverage minima, not statistical power guarantees.

## Fixed models and data-budget controls

Freeze the previous rich61 classifier and observed-combat173 classifier as
two unchanged reference pipelines, each trained on4002FIT games/calibrated on
1000 games. Reproduce their predictions on the original pilot exactly before
using them on the complement. Never initialize a newly fitted tree ensemble
from a reference model.

Create four explicitly declared deployment candidates:
1. rich_pilot_recalibrated: same frozen rich trees, Platt calibration refitted
   on all qualified full-pool calibration games;
2. combat_pilot_recalibrated: same frozen observed-combat trees, same enlarged
   calibration budget;
3. rich_full: fresh rich61 HGB on all qualified full-pool FIT frames plus the
   same enlarged calibration;
4. combat_full: fresh observed-combat173 HGB on exactly the same full-pool FIT
   frame sample plus the same enlarged calibration.

Tree recipe is unchanged:150 iterations,15 leaves,learning rate.06,L2=10,
min leaf100,early_stopping=False,seed20260929. Calibration is the existing
weighted Platt regression(C=1e6,max_iter500,seed20260929). No feature,
hyperparameter, horizon, sampler or checkpoint search. Pilot recalibration
copies frozen raw trees without mutating original artifacts. It isolates
calibration-data effects; full versus pilot-recalibrated within each feature
family isolates increased fitting data. Rich versus combat at equal budgets
tests the input representation. A total pipeline gain must not be described
as a pure training-data gain if only calibration explains it.

## Primary gates and deterministic selection

Compute equal-game/equal-within-game-frame log-loss/Brier/accuracy and2000
paired game bootstrap resamples(seed20260929), conditional on fitted models.
Recompute relative loss ratios within resamples. Main cohort is the qualified
development complement, with no previously piloted development game included.

For each of the four candidates require ALL:
- complete declared coverage/admission/code/feature/role/inference checks;
- >=2% relative calibrated log-loss reduction versus BOTH unchanged frozen
  references, with positive paired95% log-loss reduction lower bounds;
- positive paired95% Brier reduction lower bounds versus BOTH references;
- on complement frames at least180000ms before verified end:>=0.5% relative
  log-loss reduction versus BOTH references, positive paired95% log-loss lower
  bounds, nonnegative point Brier reductions and>=1000 represented games.

End-based strata are retrospective diagnostics, not live selection rules.
Main scoring retains all qualified complement frames, including unknown-end
games. Report frozen/recalibrated/full-budget contrasts, old-pilot and union
results, raw scores, ECE and fixed time/patch slices without choosing a
successful secondary endpoint or pooling cohorts after inspection.

Within each feature family, if the pilot-recalibrated candidate qualifies,
prefer it unless the full-FIT candidate also qualifies and improves log-loss
over it by>=0.5% with positive paired95% lower bound and nonnegative point
Brier reduction. If only the full candidate qualifies, retain it. Across the
retained families, prefer rich unless combat qualifies and improves over the
retained rich model by the same additional0.5%/positive-interval/nonnegative-
Brier rule. If only one family qualifies, choose it; if none, select none.
This four-candidate development screen is explicitly selection, not a fresh
confirmatory claim. Freeze any selected candidate before a separately declared
one-time confirmation/practical integration decision. No automatic promotion.

## Execution and artifact requirements

Work only in new own-project files/directories. Scope/source binding phase:
CPU1,900seconds,1.5GiB startup,.75GiB live floor; no gzip deserialization.
Streaming feature construction/packing: CPU1,1800seconds,2.5GiB startup,.75GiB
live floor. Model/complete evaluation stage:CPU1,1800seconds,same RAM gates.
No GPU. Use hidden durable one-shot jobs with no running game/competing
benchmark and preserve failed attempts. Record actual memory/runtime separately.

Bind original admission/reservation metadata, exact source files, old models,
source/code/protocol/static digests, role/sample ledgers, exclusions and output
hashes. Record which rows are original versus additional. Verify the full
original pilot subset exactly; independently verify all primary metrics and
prediction/model scope. No frozen original code/data/job, source dataset,
sealed/confirmation payload or informes/20260929/index.html modification.
No breakthrough/completion claim without demonstrated practical benefit.
