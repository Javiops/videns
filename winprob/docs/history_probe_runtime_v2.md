# History probe operational runtime addendum v2

Declared2026-09-30 after the first synthetic resource preflight and BEFORE
any real neural/frequency model was fitted or any probe result evaluated.
This addendum supersedes ONLY the1800-second model-stage time ceiling in
`history_probe_protocol_v1.md`. All scientific/data specifications remain
frozen, including the selection, history, donor scheme, initialization,
three arms, two game folds,20epochs,92040updates, baselines, metrics, practical
gates and bootstrap recipe. Existing files and the failed attempt are kept.

Attempt `history-probe-20260930-v1` stopped at69.559seconds, before fitting.
Its measured synthetic step/gather times projected1843.383seconds remaining,
which failed the declared85%-of-remaining-time guard (remaining1730.455s).
That attempt is a failed resource preflight, not a failed predictive model.
No selected-game score was produced or inspected and no epoch was shortened.

The user's active research goal has no time/token budget and explicitly
authorizes use of available RAM/GPU. The prior30-minute cap was our own
operational estimate. Set a45-minute (2700second) model-stage hard limit for
the complete unchanged recipe, including validation, preload, mandatory
synthetic preflight, fitting, selected prediction/mutation checks, checkpoint
writing, frequency controls and paired metrics. This is an explicit resource
revision after measurement, not a claim that the original cap passed.

Retain the1.25 measured-work multiplier,180second allowance and85%-of-remaining
projection gate. Retain the0.75GiB live RAM floor, idle GPU/no-game/no-competing
job checks andCPU1. Preparation remains900seconds and its completed cache is
reused unchanged. A new mandatory preflight must pass before real fitting.
Do not shorten the recipe or relax statistical gates to fit the new budget.

Use NEW `scripts/train_history_probe_v2.py` and a new run directory. Bind this
addendum, the unchanged base protocol and the original preflight receipt at
entry and publication. Record both the old rejected resource projection and
the actual v2 runtime. A successful run does not establish the broader goal;
only its measured, scoped evidence can justify subsequent work.
