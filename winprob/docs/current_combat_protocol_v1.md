# Current observed combat availability for retrospective win probability

2026-09-30. Fixed before full extraction or fitting. The preceding schemas
were inspected on ten deterministic FIT games only:234 frames/2340 player
frames, with complete health/healthMax/wallet fields and220 health-zero
records. AbilityHaste was uniformly zero in that reconnaissance and is NOT
used. AttackSpeed is treated as a reported index, not attacks per second.
Power/powerMax mixes resource types and is also omitted.

## Question and representation

Does the current win model lose useful information by treating resources as
available independently of recorded health and champion type? The frozen61
features omit champion identity, currentGold and championStats explicitly.
A permutation of champion identities among teammates leaves those features
unchanged. Recorded temporary availability is also absent as an explicit
input. These are representational observations, not measured performance gains.

Use a permutation-invariant player transform followed by team aggregation,
conceptually rho(sum_i phi(player_i)). See Zaheer et al., *Deep Sets*,2017:
https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html.
Our phi is a FIXED transparent feature map and rho is the existing histogram
boosting recipe; this is not a full neural DeepSets implementation or a claim
of universal representation. Blue-minus-red and blue-plus-red summaries
preserve direction and absolute context.

Let g_i be totalGold, b_i currentGold, h_i health, H_i healthMax. Define
a_i=1[h_i>0] when health is observed and nonnegative, and r_i=h_i/H_i when
0<=h_i<=H_i and H_i>0. Use zero with explicit availability flags when invalid.
Useful candidate summaries include sum a_i*g_i, sum a_i*(g_i-b_i), and
sum r_i*(g_i-b_i). These are recorded-health/net-outflow PROXIES. Positive
health is not a complete active/dead status (forms, revives and timing may
differ); totalGold-currentGold is not certified current inventory value.

Public native champion tags assign weight1/number_of_tags to each associated
tag. Preserve the six fixed types Assassin/Fighter/Mage/Marksman/Support/Tank;
they are champion classes, NOT inferred player lanes/roles. Per tag aggregate
roster mass, gold excess relative to the team's mean, and health-weighted
net gold outflow. Per-champion IDs select public tags but are not model columns.

## Frozen sources, timestamps and purpose

Use exactly the clean original6000 games and138831 frames, rich cache SHA
65414f41331d4a82601ef38d9a95ff4b465bc90a758629730683593a93e729cb,
with4002FIT/1000calibration/998development games and22554 sampled fit frames.
No source/population expansion. Retain every qualified original frame and its
role, patch, timestamp and label. Existing original/RICH code and models stay
unchanged. The1911 confirmation games and current global/external reservations
remain closed. Validate current admission/reservations before source reads and
publication; reject quarantine and digest conflicts.

Full feature extraction reopens only the compressed match/timeline paths
already bound in clean W1v3's source manifest. Log each access before opening,
check exact expected path/byte length/SHA256/envelope identity, then deserialize.
Raw JSON includes final/outcome fields in memory; the feature extractor reads
only immutable roster participantId/teamId/championId, match identity/queue/
patch/creation metadata, and the EXACT cached timestamp's participantFrames.
It does not index wins, postmatch performance, final roles or target actions.
No future-frame interpolation, source DB reads or sibling code imports.

Extract separate evaluation-only game-ending timing from GAME_END.timestamp,
never winningTeam. Accept only one unique finite integer end timestamp later
than every cached qualifying frame; missing/nonunique/malformed timing is
unknown. A known unique end at/before a selected frame aborts as a contradiction
of the original qualification, without dropping that frame. This
timing is NEVER a model feature, training weight or frame-selection criterion.
No unavailable end time is guessed from the last cached frame or final outcome.

These new variables contain effects of items and combat. The resulting models
are explicitly OBSERVATIONAL RETROSPECTIVE WP predictors. They are not action-
neutral judges and must not be used to value a purchase by deleting an item or
changing its derived stats. Enemy information is OMNI; no live-safe claim.

## Fixed feature map and data quality

New team summaries are versioned in `features_current_combat_v1.py`:
- observed/positive health fractions; observed maximum-health and health-ratio
  fractions; mean health fraction; current/max health sums scaled by10000;
- observed level and positive-health level sums scaled by90;
- observed wallet fraction and wallet sum/10000; net gold outflow/50000;
- positive-health gold/50000 and its share of team gold; health-weighted gold
  share; positive-health and health-weighted net outflow/50000; the corresponding
  health/wallet joint availability fractions;
- for AD/AP/armor/MR/reported AS/movement speed, observed fraction and ordinary
  and positive-health sums divided by5 times scales500/1000/100/100/100/500;
- per public champion tag: roster mass/5, excess gold/50000 and health-weighted
  net outflow/50000. Gold excess is g_i minus its team's mean gold.

All team summaries emit diff and sum. Health-only uses precisely the first
two team summaries (four columns). Full combat uses the complete new block.
Missing/nonfinite values get zero contribution plus availability; no imputation
from another timestamp and no row removal. Negative finite wallet/resistance
values remain reported values; they are counted, not silently clipped. Gold
must be finite/nonnegative and recompute the original gold_diff and
rich_gold_total_sum EXACTLY after float32 conversion at every frame.

Require complete original coverage/native tags/finite final features; within
each original role require>=99% observed health, wallet and valid health ratio,
and>=95% finite observations for each included reported combat stat. At least
90% of participant time series in each role must vary in health or healthMax,
to catch accidentally broadcasting final values. Report quality and dynamics
without reading outcomes. Failure stops fitting; never loosen quality criteria
or remove failed rows after inspection. End timing coverage is separate.

## Two bounded candidates and predeclared choice

Keep the strongest published calibrated rich_trajectory classifier/probabilities
as the unchanged reference (log-loss0.4943414406 on the exposed998 dev games).
Fit two fresh models with identical old61 current inputs and fixed fit rows:
health_availability adds four health columns; combat_resources adds all new
current columns. Each uses CalibratedModel with150 HGB iterations,15 leaves,
learning rate.06,L2=10,min leaf100,no early stopping,seed20260929, equal-game
normalized FIT weights and the unchanged1000-game Platt calibration.
No hyperparameter, time-window, field or checkpoint search.

Both are declared development candidates before results. For each candidate,
require>=2% relative calibrated log-loss reduction against the frozen reference,
paired95% log-loss reduction lower bound>0 and paired95% Brier reduction lower
bound>0. To rule out a gain confined to the last moments, additionally require
>=0.5% relative log-loss reduction with positive paired95% lower bound on
frames at least180000ms before verified game end, with>=500 eligible dev games
and nonnegative point Brier reduction there. This is a hindsight-defined
retrospective lead-time stratum, not a live eligibility rule. Main evaluation
still uses all998 games/23366 frames, including unknown-end cases.

If only one candidate qualifies, select it. If both qualify, prefer the simpler
health model unless full combat improves log-loss over health by>=0.5% with
positive paired95% lower bound and nonnegative point Brier reduction. This
choice is fixed now. If neither qualifies, close the recipe. This is an
explicit two-candidate exploratory screen, not fresh confirmatory inference;
all primary decisions require a separately frozen confirmation before claiming
completion or promotion. No secondary window/patch result rescues a failure.

Use unchanged equal-game/equal-within-game-frame scores and2000 paired whole-
game bootstrap draws(seed20260929), conditional on fitted models. Recompute
relative ratios inside bootstrap resamples. Report raw/calibrated proper
scores, ECE20, time/patch slices and input-quality counts as diagnostics.

## Execution and required evidence

Feature preparation: CPU1,900seconds,2.0GiB startup,.75GiB live RAM floor.
Model/evaluation stage: CPU1,600seconds,same RAM gates,no GPU. Durable hidden
one-shot tasks wait for no game/competing benchmark; preserve all attempts.
Freeze source/code/protocol/static hashes before actual extraction/fitting.
Tests must prove permutation invariance, team-swap parity, identity-assignment
sensitivity, exact health-only projection, missing-value handling, outcome/
future-frame feature invariance and no end-time feature channel.

A passed development screen only opens a separately specified practical and
confirmatory comparison for its frozen selected candidate. It does not identify
causal purchase values or itself complete the user's breakthrough goal. No
source benchmark, sealed/confirmation payload, shared dataset or original
informes/20260929/index.html is changed; no automatic policy/site promotion.
