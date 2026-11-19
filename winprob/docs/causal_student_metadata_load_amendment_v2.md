# Checkpoint metadata loading, no training restart

2026-10-04. During the fixed v1training, review found that training_game_ids
are NumPy str_ scalars (sorted set of a Unicode NumPy array). torch.save retains
that metadata type. The trainer's later torch.load(weights_only=True) does not
allow NumPy scalar metadata by default. A synthetic roundtrip reproduced the
UnpicklingError; the completed hard/fold0checkpoint has the same metadata type.

The actual first model is valid and contains20epochs/4400updates/1757training
games. No weights, losses, training inputs or current predictions were changed.
The live six-model trainer must continue unchanged. Do not stop/restart it or
edit its consumed code/checkpoints. Its planned post-training load is expected
to fail before student prediction; that expectation is NOT evidence of exit.
Verify the actual terminal handle/status and six-model lock afterward.

A scoped weights-only loader for our own digest-bound checkpoints may allow
only numpy._core.multiarray.scalar, numpy.dtype and numpy.dtypes.StrDType to
read the string metadata. It keeps weights_only=True; no unrestricted pickle
load or global permanent allowlist change. Synthetic and actual completed
checkpoint loads with this scope succeeded. Normalize identifiers to Python
strings in memory only; preserve original checkpoint bytes and tensor hashes.

After actual training-phase completion, a separate new prediction worker may
copy the six byte-identical final checkpoints/locks/snapshots into a new version,
load them through that reader, and run the original public-only128-row inference
batches. It must validate the original20epoch/26460update accounting and fold
identities first. Use only public focal tensors at the original global inventory
padding width; no teacher/label GPU storage is needed. Verify the original M2
matrix and preserve all statistics, candidate roles and native decoder rules.

Report original training duration separately from prediction-recovery duration;
do not pretend the failed all-in-one worker produced a completed result. Both
candidate/teacher semantics and protocol SHAfc3168bffd7865eb9ab53a18450501d8732474cf8df8077f16e61ffac7abbd5e
stay unchanged. This amendment does not authorize partial-model scoring,
retraining, checkpoint selection, parameter changes or opening seals.
