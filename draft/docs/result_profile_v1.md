# Profile v1 result: expected CS and team damage composition

2026-10-01. Protocol `docs/protocol_profile_v1.md`. Report `data/runs/profile_v1/report.json`.
Development split; confirmation sealed. Gate: the CS sidecar's 12:00 values are identical to the
v1 extract.

## Team damage composition (6,903 development games)

Physical-share error per team: 4.8 points with champion profiles vs
8.9 with the fixed role-mean composition (difference
-4.1 points, 95% CI [-4.2, -3.9]); correlation with the actual share
0.83.

## Expected CS per player

Cells: R^2 / RMSE (role-mean RMSE), development players whose game reaches the minute.

| minute | Top | Jungle | Mid | Bot | Support |
|---|---|---|---|---|---|
| 12:00 | 0.093 / 15.7 (16.5) | 0.104 / 11.3 (11.9) | 0.127 / 13.7 (14.7) | 0.053 / 14.3 (14.7) | 0.094 / 5.2 (5.5) |
| 24:00 | 0.163 / 26.8 (29.3) | 0.185 / 24.5 (27.2) | 0.195 / 25.6 (28.5) | 0.111 / 25.6 (27.1) | 0.200 / 8.1 (9.0) |
| 30:00 | 0.227 / 32.3 (36.7) | 0.237 / 28.7 (32.9) | 0.226 / 30.7 (34.8) | 0.167 / 31.7 (34.7) | 0.249 / 9.6 (11.0) |

Every role and minute improves on the role mean (game-bootstrap intervals exclude zero). These are
"what this champion usually does in this role" expectations, not forecasts for a specific player.
