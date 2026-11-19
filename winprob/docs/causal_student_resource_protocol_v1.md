# Synthetic resource pilot for the complete student

2026-10-04, before any real student training. Test only the implemented
causal_student_v1 architecture. No game, purchase, teacher-probability, outcome,
development or sealed payload is read. The pilot is not an accuracy experiment.

Fixed synthetic workload:128boards,10players,12inventory positions with unit
counts,37public globals, two rune indices;9333synthetic candidate baskets with
up to6total copies and padding, including empty-main group0. Synthetic labels
are group0 and teacher probabilities uniform. Seed20261004. The generated
catalogue is a workload fixture, not observed purchases or a real action prior.

Reserve an uninitialized CUDAfloat32 tensor[56288,9333] to account for the full
native-teacher storage without opening its file. The reservation is not model
data. Report allocation explicitly and preserve1GiBdevice headroom after peak.
Use Adamlr3e-4, two warmup forward/backward/step passes and four timed passes,
batch128/action chunk512, activation checkpointing for the deterministic deep
interaction only. All state/action embeddings are recomputed each update.
Store timings, parameter counts, peakCUDAallocated/reserved bytes, available
host memory and finite-gradient checks. No checkpoint is saved or reused.

CPU1, noTF32, deterministicCUDA/CUBLAS, no game/competing protected job,
startup2.5GiB/live0.75GiBhost RAM, GPU3072MiBfree, hidden one-shot15min worker.
Report failure rather than silently choosing another batch/architecture.
A successful pilot only supports feasibility at these synthetic shapes; a
separate real-data cost check and fixed statistical comparison remain required.
