# Profile v2: expected kills, kill participation and team dragons

Declared 2026-10-01 before evaluating. Same split as v1; fit tables,
development evaluation; confirmation sealed. Source: sidecar
`data/extract/match_stats_v1.jsonl.gz` (match payloads only).

- Expected kills of a player (whole game): shrunk mean kills of that champion in
  that role over fit games (prior weight 30 toward the role mean). Baseline:
  role mean. Metrics on development players: RMSE, R^2, game bootstrap.
- Expected kill participation: (kills + assists) / team kills, games with at
  least one team kill; same shrinkage, baseline and metrics.
- Expected team dragons (whole game): mu + b1 * sum of own champion-role
  dragon deviations + b2 * sum of enemy champion-role dragon deviations + b3 *
  side, where a deviation is the shrunk mean of (team dragons - mu) when the
  champion plays that role on the team (own) or on the opposing team (enemy).
  b1..b3 by least squares on fit rows with 5-fold out-of-fold tables (v1 folds);
  development uses full-fit tables. Baseline: fit mean per side. Metrics:
  RMSE, R^2, game bootstrap.

All are descriptive whole-game expectations for the draft, conditional on
nothing about the players. Published regardless of outcome.

## Amendment (2026-10-01, after the first KP numbers were printed; definitional, not chosen on accuracy)

`objectives.champion.kills` exceeds the sum of the team's player kills by 1-3 in
~8% of team-games. Kill participation now uses the sum of the team's player
kills (the in-game scoreboard total, the standard KP definition) as denominator.
