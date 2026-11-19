# Profile v3: kill participation and team dragons by phase (12 / 24 / 30 min)

Declared 2026-10-01 before evaluating. Same split, folds and confirmation seal
as v1. Source: sidecar `data/extract/event_frames_v1.jsonl.gz` (timeline
CHAMPION_KILL and ELITE_MONSTER_KILL events, cumulative up to the m:00 frame;
v1 timeline population). Gate: whole-game totals from events must equal the
match payload's kills, assists and dragons for at least 99.5% of games.

For minute m in 12, 24, 30, over games that reach m:

- Kill participation by m: (kills + assists by m) / (team's player kills by
  m), team-games with at least one kill by m. Champion-role shrunk mean (prior
  30 games) vs role mean; RMSE, R^2, game bootstrap.
- Team dragons by m: the profile v2 additive own/enemy champion-role model
  refitted with dragons-by-m as the target (OOF tables on fit, least squares,
  full-fit tables for development). Baseline: fit side means at m.

Published regardless of outcome. Kills stay whole-game (profile v2).
