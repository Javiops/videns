# Pre-registration: frozen dev baseline on the independent temporal window

2026-09-22 ~01:00. Reviewer's proposed next step after closing the
training cycle. Run dir: `data/experiments/16.18-val-window-check-v1/`.

## What is checked, and what is not

ONE configuration, frozen BEFORE this read and bound by digest in
`data/experiments/16.18-basket-econ-finetune-v1/frozen_dev_baseline_v1.json`:
econ_ft_ep8 + frozen-parent economic domain (Bf_ep16 top-128) + FIT
normalization stats + ward-grouped logsumexp decoder ("no main purchase"
group). Evaluated once on the frozen VAL split of the generation
(1,601 games / 191,916 decisions, temporally later than TRAIN) under the
ward-free metric. NO comparisons between configurations on val, NO
selection, NO tuning from this read — the trained wardfree alternative is
NOT evaluated here precisely to avoid burning val on a choice. Test stays
sealed.

## Declared outputs

Ward-free first_action_exact and any-of-three with per-game bootstrap
CIs on val itself; slices single/multi/no_buy_death/ward_only; SAVE-group
rate; fallback count; per-decision rows persisted. Expectation declared:
val numbers may differ from the internal probe's (different population,
later window); the check is whether the configuration's behavior is
broadly stable, not whether it reproduces the probe's exact figures. No
promotion decision follows from this alone.
