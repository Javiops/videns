# Recipe fit: execution budget adjustment only

2026-10-01, before any actual purchase-target read or real coefficient fit.
The first consumer run completed all56288input states/164731native receipt
checks and zero-correction parity, but its timing projection failed.

Measured objective/gradient passes took0.03729-0.06451seconds. The unchanged
proxy formula from recipe_fit_runtime_v1.md projected1573.386seconds, exceeding
80% of the1735.375seconds remaining under the original1800second limit.
The run stopped before load_targets; real_purchase_fits=0. This is an execution
budget failure, not a measured purchase-accuracy failure or optimizer failure.

Increase ONLY the own worker budget from1800 to2700seconds (30to45minutes),
and its launcher monitoring window from31to46minutes. Keep startup RAM3.25GiB,
live RAM0.75GiB,CPU1,no concurrent source work, the same preflight formula and
its80%reserve. Register a new v2 run; preserve every v1 input lock, code snapshot,
preflight and failure log. The new run must reproduce the failed run's complete
input feature digests before any target read. No data-selection, descriptor,
standardization, solver, lambda, max-iteration, model, bootstrap or metric-gate
change. Native M2 is not rerun. No real target or accuracy informed this change.

Scientific protocol recipe_transition_probe_protocol_v1.md remains byte-identical
(SHA e87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e).
This note supersedes only the30minute execution bound stated there and the
hard worker seconds in recipe_fit_runtime_v1.md. User authorization to use
available RAM/GPU and pursue the active goal covers this runtime adjustment.
