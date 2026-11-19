# Profile v3 result: kill participation and team dragons by phase

2026-10-01. Protocol `docs/protocol_profile_v3.md`. Report `data/runs/profile_v3/report.json`.
Gate: whole-game event totals equal the match payload in 99.9947% of 37,849 games.
Development split; confirmation sealed; games that reach each minute.

Kill participation by minute, R^2 vs role mean (all intervals exclude zero):

| minute | Top | Jungle | Mid | Bot | Support |
|---|---|---|---|---|---|
| 12:00 | 0.024 | 0.022 | 0.042 | 0.013 | 0.008 |
| 24:00 | 0.038 | 0.047 | 0.039 | 0.016 | 0.051 |
| 30:00 | 0.049 | 0.067 | 0.047 | 0.020 | 0.107 |

Team dragons by minute:

| minute | dev games | fit mean | RMSE (side means) | R^2 | model - side mean sq. error 95% CI |
|---|---|---|---|---|---|
| 12:00 | 6,254 | 0.53 | 0.572 (0.581) | 0.088 | [-0.0131, -0.0084] |
| 24:00 | 4,400 | 1.58 | 1.030 (1.049) | 0.079 | [-0.0503, -0.0299] |
| 30:00 | 1,926 | 2.09 | 1.095 (1.108) | 0.055 | [-0.0435, -0.0136] |

The tool shows both by the selected phase (early 12, mid 24, late 30); kills stay whole-game.
