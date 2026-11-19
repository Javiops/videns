# Profile v1: expected CS and team damage composition from the draft

Declared 2026-10-01 before evaluating. Same split as v1; fit tables, development
evaluation; confirmation sealed.

- Expected CS of a player at minute m (12, 24, 30): shrunk mean CS (lane minions
  + jungle monsters, timeline frame m) of that champion in that role over fit
  games that reach m, prior weight 30 games toward the role mean. Baseline: the
  role mean. Metrics on development players whose game reaches m: RMSE, R^2.
- Team damage composition: shares of physical / magic / true damage to
  champions, from each champion-role's shrunk mean damage per minute summed over
  the five players. Baseline: the fixed role-mean composition. Metric: mean
  absolute error of the team physical share against the team's actual
  end-of-game share, and correlation.

These are descriptive expectations ("what this champion usually does in this
role"), not counterfactual predictions for a specific player. Published
regardless of outcome.
