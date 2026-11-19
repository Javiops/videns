# Draft v1 result: iTero recipe replicated on our archive

2026-09-30. Protocol `docs/protocol_v1.md` (+ exploratory addendum v1b).
Development split only; the confirmation split (6,903 games) is still sealed.
Reports: `data/runs/v1/report.json`, `blind_report.json`, `turns_report.json`,
`turns_value_report.json`, `data/runs/v1b/report.json`.

## Data

46,888 admitted match files (5,029 sibling reservations denied before any
read) -> 46,019 valid ranked solo games -> chronological split fit 32,213 /
dev 6,903 / confirm 6,903. 510 fit games from patches before 16.17 dropped
(declared scope); fit = 31,703 games (16.17-16.19), dev = 6,903 (16.18: 5,524,
16.19: 1,379). KR + EUW ladder-sampled; rank per game unknown. Blue won 48.3%
of fit games and 49.5% of dev games.

Pick order: **not recoverable**. match-v5 lists participants sorted by role;
only bans carry `pickTurn`. No champion-select capture exists in videns.gg.

## T1 replication and T2 coherence (dev, 6,903 games)

| model | log loss | accuracy | AUC | cal. slope |
|---|---|---|---|---|
| R0 side only | 0.69334 | 50.5% | 0.500 | - |
| R1 champion-role win rates | 0.68713 | 54.5% | 0.566 | 1.31 |
| I iTero-style, TOP reading | 0.68561 | 55.2% | 0.571 | 1.25 |
| I averaged over 10 perspectives | 0.68575 | 54.7% | 0.571 | 1.32 |
| C coherent | 0.68048 | 56.5% | 0.591 | 1.14 |
| C without experience | 0.68618 | 54.1% | 0.567 | 1.12 |
| C + autofill/secondary context | 0.67918 | 57.0% | 0.596 | 1.16 |

Paired log-loss differences (95% game bootstrap):

- I_top - R0 -0.0077 [-0.0100, -0.0054]; I_top - R1 -0.0015 [-0.0027, -0.0004]:
  the recipe replicates and beats champion win rates alone.
- C - I_top -0.0051 [-0.0075, -0.0028], **but** C sees the experience proxy of
  all ten players while I (like iTero) sees only the picker's. Without any
  experience (v1b): C_noexp - I_noexp -0.0004 [-0.0015, +0.0006] (v1 priors),
  -0.0012 [-0.0024, -0.0001] (frozen priors). Coherence costs nothing and may
  help slightly; it is not the big lever.
- Experience is the largest single signal: C - C_noexp -0.0057 [-0.0079,
  -0.0035]; the picker's own experience in I: -0.0010 [-0.0017, -0.0003].
  Coefficients: prior games on the champion +, total prior games - (players
  off their usual champions lose more).
- Autofill/secondary context: -0.0013 [-0.0022, -0.0004] (not a draft feature).
- Unknown history as population mean + count instead of zero: +0.0000 (null).
- Calibration slopes > 1: every model is too timid on dev.

The replica reproduces iTero's incoherence: for the same final draft,
|P(blue via lane L) + P(red via lane L) - 1| averages 1.8 pp (p90 3.7, p99
5.6); 40% of games exceed 2 pp; P(blue) spreads 3.5 pp across the five lane
readings (p90 5.0). C is exact by construction (asserted).

## Pair effects at our volume

The empirical-Bayes prior finds no between-pair variance for cross-lane
counters and synergies (fully shrunk to zero) and needs k ~ 2,000 games per
lane matchup before trusting it. Light stats-site shrinkage (k=50) on pair
tables **hurts**: C +0.0008 [+0.0000, +0.0015], C_noexp +0.0012 [+0.0005,
+0.0020]. With ~32k games, raw counter/synergy rates are noise; iTero's claimed
volume (millions per patch) is what makes them usable.

## T3 partial drafts: average imputation vs marginalisation

| scenario | avg log loss | MC log loss | MC - avg |
|---|---|---|---|
| enemy team hidden | 0.68766 | 0.68749 | -0.0002 [-0.0005, +0.0002] |
| lane opponent hidden (10 variants) | 0.68200 | 0.68190 | -0.0001 [-0.0002, -0.0000] |
| only blue top known (C) | 0.69245 | 0.69239 | -0.0001 [-0.0005, +0.0004] |
| only blue top known (I) | 0.69232 | 0.69260 | +0.0003 [-0.0001, +0.0006] |

Marginalising unknown picks is not measurably better for outcome prediction
with a model this weak, and average imputation does not bias our replica (mean
P(blue) 0.486-0.488 in every variant vs 0.495 observed; the gap is the
fit-to-dev side drift, 48.3% -> 49.5%). The low bias seen in iTero's API is
therefore a property of its features, not of imputation as such.

## Turn order (MODEL-BASED, C without experience, 500-800 dev drafts)

Solo-queue sequence with a random role order per team; not outcome-validated.

- A pick made before its lane opponent loses 3.4-3.9 pp against the model's
  best reply from the 30 most-picked champions, but ~0-0.5 pp against
  pick-rate replies.
- Most of that "best reply" is simply picking a strong champion: the value of
  **seeing the lane opponent before choosing** (same pool, scored against the
  actual opponent) is +0.30 pp [0.29, 0.32] per lane decision (jungle 0.48,
  top 0.31, bottom 0.30, mid 0.23, support 0.18); the choice changes in 39%.
- Greedy-vs-average (iTero) and expected-reply rules pick the same champion
  84% of the time; maximin differs 33% of the time and gains ~0.2 pp in the
  worst case for ~0.15 pp against average replies. All within ~0.2 pp.

## Verdict

1. The iTero recipe replicates on our games at the expected solo-queue level
   (55-57% accuracy, AUC 0.57-0.60); its structure adds little beyond
   champion strength, gold@12 and experience at our volume.
2. Coherence is free: fixing the 106%-style incoherence does not cost accuracy.
3. The measurable levers are player-champion experience (largest) and
   role-assignment context, not counters/synergies (unmeasurable at 32k games)
   and not the imputation rule.
4. Turn effects cannot be validated without champion-select capture; in the
   model a counter-pick is worth ~0.3 pp per lane.

Next (not run): capture `/lol-champ-select/v1/session` in live_overlay to get
real pick order; pool more games (pairs need 10x volume); calibrate (slopes
> 1); confirmation of one frozen candidate (C_ctx or C) on the sealed split.
