# Gold v1 result: lane gold difference at 12:00 from the draft

2026-10-01. Protocol `docs/protocol_gold_v1.md`. Report `data/runs/gold_v1/report.json`.
Development split: 6,254 games with a 12:00 frame (fit 31,595).
Confirmation still sealed. Gold in units of gold; squared-error differences in (thousands of gold)^2.

| lane | SD of target | RMSE B0 (zero) | RMSE B2 (own-lane champion means) | RMSE model | R^2 model | r | model - B2 sq. error (95% CI) | R^2 +exp |
|---|---|---|---|---|---|---|---|---|
| Top | 1257 | 1257 | 1205 | 1200 | 0.090 | 0.30 | -0.0131 [-0.0224, -0.0037] | 0.104 |
| Jungle | 1097 | 1097 | 1045 | 1039 | 0.103 | 0.32 | -0.0128 [-0.0187, -0.0069] | 0.107 |
| Mid | 1113 | 1114 | 1077 | 1072 | 0.073 | 0.27 | -0.0104 [-0.0181, -0.0027] | 0.079 |
| Bot (ADC) | 1588 | 1588 | 1559 | 1549 | 0.049 | 0.22 | -0.0317 [-0.0477, -0.0157] | 0.053 |
| Support | 703 | 703 | 670 | 668 | 0.097 | 0.31 | -0.0030 [-0.0054, -0.0007] | 0.101 |

Team total (sum of lanes): SD 3154, RMSE model 3035 (R^2 0.074);
with experience 3027 (R^2 0.079).

## Reading

- The model (cross-role gold table + lane matchup residual + side) beats both baselines in every
  lane (all intervals exclude zero). Which component carries the gain was not ablated.
- The draft explains 5-10% of the lane gold variance at 12:00 (r 0.22-0.32). Typical error is
  700 (support) to 1,550 (bot) gold: a prediction of +300 means "slightly favoured", never a
  guarantee. The tool shows this error band next to every lane.
- Player experience adds a little more in every lane (+exp column; intervals exclude zero) but
  is not used by the tool, since enemy experience is unknown during champion select.

## Draft tool (`scripts/start_tool.ps1` -> http://127.0.0.1:8030)

Ranks OUR candidates for any slot by our probability of winning (frozen v1 C_noexp, champion
information only), not by how likely they are to be picked. Pick rates are used only to (a)
average over the enemy's still-unknown reply in the same lane ("Esperada"), (b) find the worst
reply ("Peor caso"), and (c) restrict candidates to champions with >=100 fit games in the role
(>=30 when the user's pool filter is on). Each of the top 12 shows the four feature groups that
move its probability most versus an average champion in the slot. The finished-draft panel
shows the predicted lane gold differences at 12:00 with their development RMSE.
