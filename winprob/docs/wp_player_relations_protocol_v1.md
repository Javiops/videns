# Joint player-state representations for retrospective win probability

2026-09-30. Declared before extracting this player tensor or fitting a player-
representation model. This is a new, explicitly exploratory development study.
The earlier scale result remains closed: combat_full improved log-loss0.7715%
over combat_frozen on1863games, below its declared2% gate. No old threshold is
changed and no failed recipe is reclassified as successful.

## Hypothesis and limits

The executable synthetic witness in
`data/reports/wp_spatial_health_collision_v1.json` constructs two states with
identical173 existing features but different joint health/position assignments.
It proves information loss in that representation, not different true win
probabilities, prevalence, causal action value or expected predictive benefit.

Test whether retaining the observed states of the ten players and learning
their interactions improves forecasts beyond the strongest measured full-pool
combat classifier. A team is a set: reordering its player records must not
change predictions. Blue/red side remains explicit; no artificial map-side
complement symmetry is imposed. This is retrospective OMNI and not live-safe.

Architectural sources: [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html),
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), and
[Relation Networks](https://arxiv.org/abs/1706.01427). Learned pooling represents
sets through shared element transforms. Attention lets elements interact
before pooling. Their relevance here is our hypothesis, not a published LoL
performance claim. We implement a small attention encoder inspired by the
Set Transformer, without claiming a verbatim reproduction of its experiments.

## Data, roles and features

Use exactly the17195 qualified games in the audited full-scale cache manifest
SHA256 `7a6852da971246e67051b0ad7622b27b1ff20fd7ce8cb43d9fd76ccf41d89c14`.
Keep its11468FIT,2866calibration,2861development games, chronological roles,
396667 qualifying timestamps and original game indices. Preserve the exact
64422 sampled FIT frames and use every calibration/development frame. No new
game or outcome-based exclusion, replacement or row filter is allowed.

Revalidate its scope/admission/current reservation union and compressed source
bindings before access. All1911 own confirmation games and all sibling/global
sealed payloads remain closed. The whole2861-game development set has already
been evaluated: it is exposed development, never fresh or confirmatory.

Read only bound admitted match/timeline envelopes. Match fields used for the
new tensor are immutable participantId/teamId/championId plus identity/queue/
patch/date checks. ChampionId selects previously bound public tags; it is NOT
a learned identity input. Timeline inputs are exact current participantFrames
at the old qualifying timestamps. No event/future frame, final performance,
final role, winner, item inventory, player identity or outcome enters tokens.
Labels/global173 matrices/end-time strata come unchanged from the old cache.

Each frame has10tokens,35float32 values each, ordered by(teamId,participantId)
only for storage. ParticipantId is not a feature. Features are:
- position x,y divided by14870 and a position-observed mask, using the frozen
  rich position validity rule; zero/out-of-map/nonfinite positions stay unknown;
- totalGold/10000; currentGold/2000 and observed mask; (totalGold-currentGold)/10000;
- health/2000 and observed mask; healthMax/2000 and observed mask; health/healthMax
  and observed mask; positive-health indicator;
- level/18 and observed mask;
- attackDamage/500,abilityPower/1000,armor/100,magicResist/100,reported attackSpeed/100,
  movementSpeed/500, each with its observed mask;
- six public champion-tag fractions (each tag receives1/number-of-tags);
- side indicator+1blue,-1red.

Use frozen current-combat `player_values` validity/units for all overlapping
values; absent values become0 with their mask0, without guessed imputation.
Reported attackSpeed is not attacks/second; positive health is not certified
alive status; net gold outflow is not inventory value or causal combat power.
No added per-player XP/damage/CS or champion embedding. Those aggregate inputs
remain in the existing173 global vector. This targets joint structure of
already admitted constituent measurements, not a broader source-data search.

Cache extraction compares all reproduced112 combat summary values to the old
shards exactly, preserves every timestamp and logs each raw read before access.
Write a versioned contiguous NPY token tensor plus offset/game manifest and
digests to avoid thousands of additional small files. Missing/mismatched old
sources or alignment is a hard failure, never an outcome-based exclusion.
Report position availability and retain all old rows, including missing tokens.

## Four fixed neural arms, three fixed seeds

Use the unchanged173 global features in ALL arms. Standardize globals and
tokens using only sampled FIT rows, equal game weights and equal player weight
within a frame; use standard deviation1 for constant columns. Missing values
and masks are retained. Calibration/development do not estimate preprocessing.

All arms have a global MLP head: concatenated global173 and a64-dimensional
set representation ->128GELU ->64GELU ->one logit. The four arms are:
1. global: the64 set inputs are zeros; no token encoder is fitted;
2. deepset: shared35->64GELU->64GELU token encoder, then mean across10tokens;
3. attention: shared35->64 projection; two self-attention blocks of width64,
   four heads, scaled dot-product attention, residuals and LayerNorm, and a
   row-wise64->128GELU->64 feedforward per block. One learned query pools the
   final10tokens with four-head attention and a residual/LayerNorm/feedforward
   block. No ordinal/participant positional encoding or dropout;
4. attention_position_shuffled: identical architecture/initialization/training
   order to attention, but the position x/y/mask bundle is reassigned among the
   five same-team players independently for every frame. The permutation is
   fixed from seed20260929,game_index,timestamp_ms,team, with no outcome input.
   Other token fields and all173 global inputs remain unchanged. Apply this
   same deterministic information-destruction rule to FIT/calibration/development.

The shuffle is a matched representation control, not a physically realizable
intervention. It retains team position marginals and global information while
breaking the extra token-level position association. It cannot attribute a
deepset improvement specifically to position because that arm has no matched
deepset shuffle. Attention/deepset parameter counts differ and are reported.

Fit each arm with seeds20260929,20260930,20261001, all three completed and retained.
Exactly30epochs, batch512, AdamW learning rate0.0003,weight_decay0.0001,
gradient norm clipping1.0, weighted binary cross-entropy, float32, no learning-
rate schedule, dropout, early stopping, checkpoint selection or tuning. Use the
same deterministic epoch permutations in all arms for each seed. Loss weights
give equal total mass to each FIT game, normalized to mean1 across frames.
Set CPU threads1, deterministic Torch settings, no TF32 or mixed precision.
No partial-arm or best-seed result substitutes for the12 completed models.

For each arm average its three raw sigmoid probabilities, then fit the SAME
weighted Platt recipe as prior work on all2866calibration games (C1e6,max_iter500,
seed20260929). Calibrate the ensemble, not separately selected seed predictions.
Save all models/preprocessors/calibrators and training receipts before reading
any new development predictions or comparison metrics. Report seed-specific
raw scores as stability diagnostics only; never choose a lucky seed.

## Primary comparison and fixed selection

Primary evaluation is the full exposed2861development games, equal game/equal
within-game qualifying frame weights. Report the original998pilot and1863
complement separately as secondary, without selecting a successful cohort.
Reference is the frozen full-pool combat model from
`data/runs/wp-scale-fit-20260930-v2/experiment/combat_full.joblib`, with report
SHA256 `6e10781c05ce85d706ef87f723c36bdabebbc75613624193bd2144a8fa2fe014`.
Require exact replay of its existing development predictions before comparisons.
The older weaker references do not substitute for this benchmark.

Use proper-score metrics and2000paired game bootstrap resamples with
seed20260929, including relative ratios recomputed within every resample.
For EACH of deepset and attention, useful-signal eligibility requires all:
- complete data/code/inference checks, all12models/30epochs and three-seed ensembles;
- >=2% relative calibrated log-loss reduction against frozen combat_full,
  positive paired95% log-loss AND Brier reduction lower bounds;
- >=0.5% relative log-loss reduction versus global neural control, positive
  log-loss reduction lower bound and nonnegative point Brier reduction;
- at least3minutes before verified game end:>=0.5% relative log-loss reduction
  against combat_full, positive log-loss lower bound, nonnegative point Brier
  reduction and at least1700represented games;
- at least two of its three seeds have positive raw log-loss reductions against
  the same reference's RAW forecasts on this fixed primary cohort.

Attention additionally requires>=0.5% calibrated relative log-loss reduction
against its position-shuffled control, positive log-loss lower bound and
nonnegative point Brier reduction. This limits a position-association claim.
If only one real candidate qualifies, select it. If both qualify, prefer
deepset unless attention adds>=0.5% resolved log-loss improvement with
nonnegative point Brier reduction. Neither control is a promotion candidate.
If neither qualifies, select none; do not rescue it with a weaker comparator,
secondary cohort, seed, patch, epoch, uncalibrated score or altered gate.

Report raw and calibrated proper scores/ECE, fixed3-10/10-20/20-30/30-40minute
slices, patches and lead strata, and permutation-invariance numeric checks.
Intervals condition on fitted models; seed diagnostics do not constitute
training-set uncertainty. This repeated-development selection is exploratory.
No confirmation payload is opened or deployment performed by this protocol.

## Practical contribution and execution

A positive forecast screen is still intermediate. The intended use is review
of local support/exposure around decision times, where separate team totals
can conceal who can participate. A later practical study must predeclare its
review-quality/decision criterion and compare against existing review behavior.
Attention weights, gradients and synthetic rearrangements are not action
effects, explanations of true causal influence, or validated prescriptions.
Keep observational forecast improvement and decision benefit separate.

Extraction: hidden durable task, CPU1,1800s cap,2.5GiB startup and0.75GiB live
RAM floor, no GPU. Fit/calibrate/evaluate: hidden durable GPU task, CPU1,
5400s cap,2.5GiB startup/0.75GiB live RAM floor,>=4GiB GPU memory available at
start. No running game or competing benchmark; stop own work if either starts.
No automatic resource-limit relaxation, partial publication or model promotion.
Bind protocol/code/inputs/scope/source hashes and record actual memory/runtime.
Independently audit source/row alignment, all prediction identities and primary
metrics/selection. Preserve all failed attempts. Do not modify frozen source
projects, prior models/caches, or informes/20260929/index.html.
