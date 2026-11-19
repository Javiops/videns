# Draft v1: replicate the iTero draft model on our archive

Declared 2026-09-30 before any outcome in this project was inspected or any
model was fitted. Status: exploratory development. Nothing here is advice.

## Question

iTero (itero.gg) predicts a draft's win probability from champion strength,
lane matchups and cross-lane counters/synergies, a gold@12 laning sub-model,
economy/snowball ratings, early/mid/late scaling, team damage mix and champion
mastery; recommendations rank each candidate for one lane with every unknown
slot set to an "average" champion. A probe of its public API on 2026-09-30
(Gold EUW) showed: the same 10-champion draft gives P(blue) 53.2-55.9% and
P(red) 49.6-55.6% depending on the queried lane (sums ~106%); "avg" slots bias
predictions low; unknown mastery is imputed as 2k.

1. Can the recipe be rebuilt from our games, and how much does it predict?
2. Does a single coherent (antisymmetric) match model lose anything against the
   per-perspective recipe that produces the incoherence?
3. For partial drafts, does marginalising unknown picks over their predicted
   distribution beat "average champion" imputation on real outcomes?
4. Draft turns: can pick order be recovered from our data? If not, what can be
   said about turn order from the model alone (explicitly model-based)?

## Data and isolation

Input: `data/admission/admission_v1.json` (metadata-only admission). It denies
the union of build_tracker split `test` folds (3,112), macro_coach reserved
tests (6) and winprob confirmatory games (1,911) before any payload is read;
46,888 match files admitted, 41,800 with timelines. Raw files are read only
through that manifest; per-file SHA-256 is recorded. PUUIDs are replaced by
salted hashes at extraction. Nothing in build_tracker, winprob or macro_coach
is modified or imported.

Games kept: queue 420, map 11, complete, not remade (early surrender flag or
<10 min excluded), five distinct roles per team. Population: KR + EUW games
sampled from ladder players (apex-heavy; exact rank per game unknown),
patches 16.17-16.19 pooled.

## Split

Chronological by `gameCreation`: first 70% fit, next 15% development, last 15%
confirmation. Confirmation games are not loaded by development code (the
loader refuses the role unless an explicit confirm flag is given) and are not
used in this protocol. Hyperparameters are fixed below and not tuned on
development.

## Features (the iTero recipe)

Tables are computed on fit games only; fit rows use 5-fold out-of-fold tables
(random game folds). Every rate is an empirical-Bayes shrunk residual (method-
of-moments prior per table):

- champion-role win rate (side-adjusted);
- same-lane matchup residual and 20 cross-lane counter residuals (my champion
  vs each enemy role), 20 ally synergy residuals, relative to a team base
  model built from champion-role deltas;
- laning sub-model: predicted lane gold difference at 12:00 = champion gold
  means + shrunk matchup gold residual (the "gold@12 first" stage);
- economy: residual win rate when behind / even / ahead in lane at 12:00;
- scaling: residual win rate in games <25, 25-32, >32 minutes;
- damage profile: physical share, damage dealt and taken per champion-role,
  team AD ratio with and without the player;
- experience proxy (mastery is not in match-v5 and no API calls are made):
  prior games of the same player on the champion inside the admitted archive,
  and prior games in total, strictly earlier start time; counts only.

## Models (fixed before fitting)

- R0: side only (fit blue win rate).
- R1: champion-role deltas only (logistic on the team sum difference).
- I: iTero-style perspective model. One row per player (10 per game): the
  player's lane features, counters, synergies, gold prediction, economy, team
  scaling/damage with and without the player, side, experience. Label: the
  player's team wins. Ensemble = mean of standardized logistic regression
  (C=1.0) and LightGBM (400 trees, lr 0.03, 15 leaves, min 200 per leaf,
  feature fraction 0.8, seed 7).
- C: coherent match model. One row per game, blue-minus-red aggregates of the
  same tables, trained on both orientations, prediction
  p = (f(x) + 1 - f(swap x)) / 2, so P(blue) + P(red) = 1 exactly and every
  lane query of a final draft reads the same number. Same ensemble recipe.
- Ablations of C: without experience features; plus assignment context
  (autofill/secondary counts, from `selectedRolePreferences`) reported
  separately because it is not a draft feature.

## Metrics and tests (development split)

Primary metric: log loss. Also Brier, accuracy, AUC, calibration intercept and
slope, 10-bin ECE. Game-level paired bootstrap, 2,000 resamples, 95%
percentile intervals. Every game counts once.

- T1 replication: I and C versus R0 and R1.
- T2 coherence: for I, the distribution over games of
  P(blue via lane L) + P(red via lane L) - 1 and the spread of P(blue) across
  lanes; log loss of I's single-lane (TOP) reading, of I averaged over the ten
  perspectives, and of C.
- T3 partial drafts: hide (a) the whole enemy team, (b) only the lane opponent
  of each lane in turn. Compare "average champion" imputation against Monte
  Carlo marginalisation (64 samples from fit pick rates per role, excluding
  bans and already-picked champions) on real outcomes. Report the mean
  predicted blue probability against the observed blue rate for each.
- T4 experience: C with and without the experience proxy; also missing
  experience imputed as zero versus left missing.

A difference is called a gain only if its 95% interval excludes zero. Results
of any size are published, including nulls.

## Turn order

match-v5 lists participants sorted by role and records only a ban `pickTurn`;
pick order within champion select is not stored, and no champion-select
capture exists in videns.gg. Any turn analysis here is a simulation with the
fitted model under the solo-queue sequence B1 R1 R2 B2 B3 R3 R4 B4 B5 R5 and
an explicit role-order assumption; it is not validated on outcomes and must be
labelled as such. Validating turn effects needs champion-select capture
(`/lol-champ-select/v1/session` actions) going forward.

## Limits declared up front

Our population is not iTero's (Gold web default; pro claims). Draft-only
solo-queue prediction is expected to be weak (a few points of accuracy above
the side prior); iTero's 70-73% refers to pro games with team strength.
Observational tables are associations, not causal pick values.

## Addendum v1b (exploratory, declared 2026-09-30 after reading v1 development results)

v1 showed (i) method-of-moments priors that zero out cross-lane counters and
synergies and zero lane matchups in 3 of 5 folds (inconsistent fit/dev feature
distributions), and (ii) C uses the experience proxy of all ten players while
I, like iTero, uses only the picker's. Three exploratory checks, development
split only, no selection claims: (a) I without experience vs C without
experience (structure only); (b) priors frozen from the full fit in every fold;
(c) light shrinkage k=50 on pair tables, closer to raw stats-site rates.
Gate: the v1 variant must reproduce run_v1's C and I_top log loss exactly.
Nothing from v1b is promoted; confirmation stays sealed.
