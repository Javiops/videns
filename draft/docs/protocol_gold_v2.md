# Gold v2: lane gold difference at 12, 24 and 36 minutes (early / mid / late)

Declared 2026-10-01 before fitting or inspecting any 24- or 36-minute result.
Same admission, v1 extract, split and folds as gold v1; development only;
confirmation sealed. The per-minute gold comes from a sidecar extraction
(`data/extract/gold_frames_v1.jsonl.gz`, frames 6/12/18/24/30/36) so the
frozen v1 extract, split and models are untouched.

Target at minute m (12, 24, 36): blue minus red total gold per lane at the m:00
frame. Only games that reach minute m count, so the 24 and 36 predictions are
**conditional on the game lasting that long** (the 36-minute sample is a
selected subset of long, usually closer games). The share of development
games reaching each minute is reported next to every number.

Features and model: exactly the gold v1 recipe (cross-role gold table for that
minute, lane matchup gold residual from the frozen v1 tables, side; antisymmetric
ridge + LightGBM per lane), refitted per minute. Team value = sum of the five
lanes.

Gate: the 12-minute model built from the sidecar must reproduce gold v1's
development predictions exactly (same data, same recipe).

Metrics per minute and lane: RMSE vs predicting 0, R^2, Pearson r; team total
RMSE and R^2; game bootstrap of model vs zero. Published regardless of outcome.

## Amendment (2026-10-01, after reading the first run)

Only 6.0% of development games reach 36:00 (median game 25.9 minutes), so a
36-minute "late" phase describes unusual games. Owner decision: late = 30:00
(24.7% of development games last longer than 30 minutes; 27.9% have a 30:00
frame, which includes the end frame of games finishing just before 30:00). The 30-minute model uses the identical
recipe and gates; 36:00 stays in the report for reference but is not shown in
the tool. This is a post-hoc choice of phase, not a selection on accuracy.
