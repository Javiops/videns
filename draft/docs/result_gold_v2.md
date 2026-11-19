# Gold v2 result: lane gold difference at 12 / 24 / 30 / 36 minutes

2026-10-01. Protocol `docs/protocol_gold_v2.md` (with the 30:00 amendment). Report
`data/runs/gold_v2/report.json`. Development split; confirmation sealed. Gates: the sidecar's
12:00 gold is identical to the v1 extract and the 12:00 model reproduces gold v1 exactly.
5 games whose timelines were downloaded after the v1 extraction are excluded.

Cells: R^2 / RMSE (gold), blue minus red, conditional on the game reaching that minute.

| minute | dev games reaching it | n dev | Top | Jungle | Mid | Bot | Support | Team (sum of lanes) |
|---|---|---|---|---|---|---|---|---|
| 12:00 | 90.6% | 6,254 | 0.090 / 1,200 | 0.103 / 1,039 | 0.073 / 1,072 | 0.049 / 1,549 | 0.097 / 668 | +0.074 / 3,035 |
| 24:00 | 63.7% | 4,400 | 0.106 / 2,538 | 0.125 / 2,484 | 0.073 / 2,379 | 0.067 / 2,982 | 0.149 / 1,256 | +0.032 / 6,681 |
| 30:00 | 27.9% | 1,926 | 0.138 / 2,846 | 0.186 / 2,798 | 0.090 / 2,732 | 0.096 / 3,381 | 0.196 / 1,323 | +0.021 / 5,896 |
| 36:00 | 6.0% | 417 | 0.114 / 3,228 | 0.222 / 2,918 | 0.123 / 3,120 | 0.112 / 3,757 | 0.163 / 1,523 | -0.077 / 4,879 |

- Every lane at every minute beats predicting zero (game-bootstrap intervals exclude zero).
  Lane R^2 grows with time (0.05-0.10 at 12:00, up to 0.12-0.22 at 30-36:00), consistent with
  champion scaling showing up in lane gold (not tested separately).
- The team total (sum of lanes) explains little (R^2 0.07 at 12:00, 0.02-0.03 at 24-30:00, negative
  at 36:00). Why is not established; one hypothesis is that lane differences partly cancel across
  lanes. At 36:00 only ~6% of games remain.
- Tool phases: early 12:00, mid 24:00, late 30:00 (owner decision after seeing that 36:00 is reached by
  ~5% of games). Predictions at 24 and 30 are conditional on the game lasting that long.
