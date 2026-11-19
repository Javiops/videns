# Experiment phase 2: whole-basket loss contrast (16.18-whole-basket-v2)

Pre-registered before execution, incorporating the external review's program
verbatim where we agree with it. Owner authorization: PENDING — this phase
does not run until explicitly ordered.

## Question

Does arm B's sampled-softmax objective (biased: dedup union, always-SAVE, no
logQ correction) cost accuracy and/or cause the observed basket-size
over-prediction, relative to the exact full-catalogue likelihood, at equal
architecture and budget?

## Design — two branches from identical clean state

Same d64 trunk as v1 (architecture change is a DIFFERENT experiment; one
variable at a time), same fit tensors (cached), same seed handling:

- **B-s**: the v1 sampled-softmax loss, unchanged.
- **B-f**: full-catalogue cross-entropy (7,938 classes), microbatch 16-32
  with gradient accumulation to effective batch 256 (chunking alone does not
  bound backward memory).

Both: optimizer re-initialized identically; FULL state saved (model,
optimizer, scheduler, RNG) with intermediate checkpoints at epochs 4, 8, 16;
maximum 16 epochs (pre-declared: if not converged within budget the result
is "inconclusive" — NO automatic extension to 48); full-catalogue loss logged
each epoch on a fixed 10k-row FIT subset for BOTH branches (comparable
scale); peak dedicated/shared GPU memory logged.

## New instrumentation (fills v1's measurement gaps; reported for BOTH
branches AND arm A on the same decisions)

- SAVE-fallback counter (no legal candidate in top-128).
- Population-level size stats: true vs predicted basket size distributions,
  over/under-fill rates, exactly-one-extra-item error rate.
- action_family_match; paired bootstrap CIs for any_of_three as well.
- Independent legality audit artifact: every displayed basket re-checked by
  the executor, result persisted.
- Per-row arm A results persisted for reuse (no re-evaluation of A).
- Decomposition denominators reported both ways (B-dependent and
  B-independent).

## Metrics & decision

Primary: first_action_exact on the probe, B-f vs B-s paired by game (400
bootstrap), plus each vs the persisted arm A. Diagnostics at 4/8/16 are for
convergence assessment only; the MAIN evaluation is the final checkpoint
(pre-declared to avoid checkpoint shopping). Guardrails as v1. No new
architecture, no hard negatives, no arm C, no size-factorized head in this
phase (the size head — P(|B|·x)·P(B·x,|B|) — is pre-registered as phase 3
ONLY if the size bias persists after this loss contrast). No validation or
test reads. Nothing promoted.

## Budget

B-s ≈ 25 min; B-f ≈ 60-100 min (measured in the first epoch; abort the
branch if epoch time exceeds 15 min — that would signal renewed spill);
evals ≈ 30 min using cached tensors and persisted arm A rows. Total ≤ 3 h,
one process, RTX 4060.
