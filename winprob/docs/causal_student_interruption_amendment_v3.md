# Recover the externally interrupted final control, preserving completed models

2026-10-04, before any student accuracy or held-out prediction. At20:01UTC,
the original process IDs11904/13160/28776 are absent and no process runs the
original training command. Its scheduled task is Ready, last result1073807364;
Windows last boot is09:51UTC, after the original23:47UTC previous-day start.
The old launcher JSON still says running and is stale. Do not rewrite it.
The precise cause/time of termination is not established by these observations.

Immutable evidence: artifacts/recovery/causal-student-interruption-20261004-v3.json,
SHAa0a6fb7073a3779562e0cf0c289ff777de43fb9fe5e4a1d136b58552a5aa83c8.
There are five final20epoch checkpoints: all three fold0 models and fold1 hard
and distilled. The trace records one completed epoch of fold1 shuffled_teacher,
221updates. There is no checkpoint for that arm, no models_lock, no prediction
directory and no complete training report. No purchase accuracy was inspected.

This is a resource-interruption exception to the original no-restart rule,
not model selection. Preserve the original files and all five final checkpoint
bytes. In one new version, verify original inputs, code/snapshots, folds, model
identities and trace; copy the five completed models and their20epoch receipts.
Recompute ONLY the unsaved fold1 shuffled_teacher model from its original seed
20261005 for the originally fixed20epochs/4420updates. Use the identical model,
loss, teacher permutation, minibatch order, optimizer and partial-batch scaling.
Do not warm start, reuse weights, change learning rate, select an epoch or tune
any parameter. The interrupted work is discarded, not an extra epoch appended.

Require exact reproduction of the lost epoch's order, state/update counts and
three loss aggregates, excluding elapsed time. Compare the AST of the actual
minibatch-update loop with the immutable original trainer before launching.
Run the original conservative resource preflight on the current machine.
If parity or a resource/integrity check fails, preserve diagnostics and stop.

After the sixth final model is saved, the effective family still has26460updates
and120kept epochs. Freeze all six before any held-out prediction. Use the already
verified scoped weights-only NumPy-string metadata loader on digest-bound
checkpoints. Release teacher/current-label storage before the original128-row
public-only CUDA inference, unchanged native M2 reference and native decoder.
The original two-candidate protocol, controls and all acceptance gates stand.

Record recovery lineage separately: five original checkpoint hashes, original
101-row trace, interruption evidence, lost-epoch replay proof, and the one arm
restarted. Report training_restarted=true, completed_models_retrained=false.
Training time for the six retained models excludes discarded work. At least221
updates/one42.8186013second epoch were discarded; possible later partial updates
are not measured. Expose this cost limitation rather than claiming exact total
compute. Original metadata-only prediction recovery v2 is not applicable to
this interruption and must not be run against a fabricated failed status.

CPU1; host2.5GiB startup/.75GiB live; at least3072MiB idle CUDA; no League,
capture, benchmark or other protected job. A hidden scheduled recovery worker
is allowed after verifying the old one stopped. No source runtime changes,
new collection, source/global sealed reads,1911confirmation reads or promotion.
