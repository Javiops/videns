# Gold v1: lane gold difference at 12:00 from the finished draft

Declared 2026-10-01 before fitting or inspecting any gold result of this model.
Same admission, extraction and split as protocol v1 (`data/split_v1.json`);
fit = in-scope fit games, evaluation = development; confirmation stays sealed.

## Target

For each lane L (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY): total gold of the blue
player in L minus the red player in L at the 12:00 timeline frame. Games
without a 12:00 frame (shorter than 12 minutes or no timeline) are excluded.
Team total = sum of the five lanes.

## Features (draft-time, champion information only)

Tables from fit games only; fit rows use the same 5 out-of-fold game folds as
v1 (seed 11):

- cross-role gold table G[L, X, c]: empirical-Bayes shrunk mean of lane L's
  gold difference, seen from champion c's team, when c plays role X (so the
  jungler's and support's effect on each lane is included; X = L is the
  champion's own lane);
- the existing same-lane matchup gold residual (v1 `mgold`), antisymmetrised;
- side.

Lane L row: side, five terms G[L, X, blue_X] - G[L, X, red_X], matchup
residual. Swapping the teams negates every term.

Variant "+exp" (reported separately, not used by the draft tool because enemy
experience is unknown in champion select): adds both lane players' and both
teams' experience proxies.

## Models

One per lane. Antisymmetric ensemble: ridge (alpha 1.0) and LightGBM
regressor (300 trees, lr 0.03, 15 leaves, min 200 per leaf, seed 7), each
fitted on both orientations (x, y) and (swap x, -y); prediction
(f(x) - f(swap x)) / 2, averaged over the two learners.

Baselines: B0 predicts 0; B1 side mean per lane (fit); B2 own-lane champion
gold means only (v1 `cgold[blue] - cgold[red]`).

## Metrics (development)

Per lane and team total: MAE, RMSE, R^2 (vs the development mean), Pearson r.
Paired game bootstrap (2,000) of squared-error differences, model vs B2 and
vs B0. A gain requires the 95% interval to exclude zero. All results are
published, including nulls. Expected: small R^2 (lane gold at 12:00 is mostly
decided by the players), so the tool must show uncertainty, not just a number.
