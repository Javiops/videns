# Pre-registration: learning curve on the existing FIT games (v1)

2026-09-23 ~01:30. Owner's order ("haz 1-4"), item 1. Run dir:
`data/experiments/16.18-learning-curve-v1/`. Internal probe only; VAL and
test are NOT read.

## Question

Is the model data-limited at 2,000 games? Everything that added capacity
this week fitted better and generalised worse (d128, extension, weighted
loss, history), while the wins came from changing the task or combining
models. If the probe curve is still steep between 1,000 and 2,000 games,
more games is the lever and a new generation is worth its cost; if it is
flat, chasing data is the wrong investment.

## Design

- Recipe: the phase-2 Bf recipe unchanged, clean d64 init, full-catalogue
  CE, micro 32 / effective 256, Adam 1e-3 cosine T_max=16, 16 epochs,
  `torch.manual_seed(16)` per arm, order generator seed 16. Same epochs at
  every size, so optimizer steps scale with data (the standard learning
  curve); nothing is tuned per size.
- Subsets NESTED by game: the 2,000 FIT games shuffled once with a seeded
  generator; the first 500 and the first 1,000 are the two arms. Rows are
  selected by game membership from the frozen FIT cache.
- The 2,000-game point is NOT retrained: it is arm A of the capacity
  contrast (`16.18-capacity-d128-v1/A_ep16.pt`, clean d64, identical recipe
  and script family, first_action 0.4826), and its persisted probe rows are
  reused. Declared here so the reuse is a choice, not an accident; the
  digest of that rows file is recorded in the report.
- Evaluation: the ward-grouped decoder under the ward-free metric, exactly
  as the capacity contrast, on the same 58,781 probe decisions.

## Metrics and reading rule

- PRIMARY: ward-free first_action_exact at 500 / 1,000 / 2,000, with paired
  bootstrap by match for 1,000 vs 500 and 2,000 vs 1,000.
- Secondary: any-of-three; slices; FIT loss curves (reported, never read as
  a result: the FIT-loss trap has fired four times this week).
- Reading, declared now: if the 2,000-vs-1,000 gain is at least half the
  1,000-vs-500 gain, the curve is still steep and a new generation is
  justified. If it is a quarter or less, data is not the lever at this
  recipe. Between those, undecided and reported as such. No projection is
  made beyond the measured range.

## Budget and gates

Pilot gate before training: projected total hours must be below 3, else
abort. Free disk re-checked per epoch; per-epoch curves appended and
fsync'd. One training process at a time. Diagnostic only.
