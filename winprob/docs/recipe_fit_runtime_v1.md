# Runtime check for the fixed convex recipe experiment

2026-10-01, declared before any real correction fit. The statistical recipe,
sample, controls, lambda, solver settings and gates remain exactly those in
recipe_transition_probe_protocol_v1.md. Native M2 export is a separate job.

The own CPU1 worker first validates and freezes all input descriptors, before
opening actual purchase targets. It fits only the required training-fold feature
standardizers, checks zero-correction parity on all training states, and times
the actual objective/gradient geometry for each of the six fold/arm combinations.
The timing target is the synthetic constant category0, always supported by the
native first option. No purchase prediction metric is computed during preflight.

The operational projection is:

    1.5 * largest measured objective/gradient seconds
        * 300 iterations * 8 function evaluations/iteration * 6 fits
        + 180 seconds.

It must fit80% of the remaining1800-second worker budget. The factor8 is a
conservative working estimate, not a worst-case L-BFGS proof: the actual solver
retains its fixed maxls50. The1800-second hard worker limit and host resource
guard still apply. A failed resource/convergence check produces no partial
accuracy report and must not silently shorten the data or optimizer recipe.

All six final coefficient/scaler artifacts are frozen before any held-out
prediction, and the complete prediction artifact is saved before metric
evaluation. Every selected row enters fitting as its true displayed-candidate
category or OTHER; no shortlist-hit filtering is allowed. The audit independently
recomputes training-only means/scales, checks each saved optimum's objective/
gradient, replays all8700correction predictions exactly, verifies the2900native
baselines, and recomputes every summary and gate. It does not replay all56288
source M2 neural calls; that scope remains explicitly disclosed.

The fit/audit launchers are prepared but must NOT be registered while the native
M2 release task is running. Their registration guards enforce this, because a
second waiting task would otherwise look like a competing workload to the live
source launcher. The source release must complete and its task/process end first.
