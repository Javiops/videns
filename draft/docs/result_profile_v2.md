# Profile v2 result: expected kills, kill participation, team dragons

2026-10-01. Protocol `docs/protocol_profile_v2.md` (KP denominator amendment). Report
`data/runs/profile_v2/report.json`. Development split; confirmation sealed.

| stat (R^2 / RMSE, role-mean RMSE) | Top | Jungle | Mid | Bot | Support |
|---|---|---|---|---|---|
| kills per game | 0.060 / 3.51 (3.62) | 0.074 / 4.26 (4.43) | 0.122 / 4.16 (4.44) | 0.031 / 4.66 (4.73) | 0.293 / 1.87 (2.22) |
| kill participation | 0.033 / 0.147 (0.150) | 0.043 / 0.144 (0.147) | 0.037 / 0.152 (0.155) | 0.016 / 0.156 (0.157) | 0.057 / 0.138 (0.142) |

Every role improves on the role mean (intervals exclude zero), but kill participation barely moves
(R^2 0.02-0.06): the champion explains little of it.

Team dragons: RMSE 1.215 vs 1.233 with side means, R^2 0.057, squared-error
difference -0.0438 [-0.0524, -0.0352]. Coefficients: own 0.72, enemy 0.65,
side -0.22 (red takes more dragons in this archive).
