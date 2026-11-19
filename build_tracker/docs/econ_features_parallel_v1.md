# Parallel `econ_features`: design and the equivalence gate

2026-09-22. Tool: `scripts/econ_features_parallel.py`. **Nothing produced
with the parallel path is admissible until `--verify` prints PASS.**

## Why this is safe to parallelize at all

`econ_features` (`scripts/basket_econ_cache.py:50`) is a pure function of
`(action, row, budget, dragon)`:

- no randomness;
- no global mutable state — the `counters` argument carries two
  order-independent diagnostic keys (`infeasible`, `replay_failed`), which
  merge by addition;
- the `lru_cache`s inside `app/shop_econ.py` and `app/conditional_policy.py`
  are per-process memoization: they change speed, never results;
- float summation order inside one call is fixed by the input, so the same
  input yields the same bits.

Sharding by row is therefore exact, and results are reassembled by index.

## Why it is nonetheless the dangerous kind of change

Windows spawns workers as **fresh interpreters**. Each one must
independently redo what `footwear_runtime.py` does for the parent:

1. `activate_for_manifest(manifest, review_path=<audited static review>)` —
   the activation that asserts the live footwear runtime equals the one the
   generation manifest binds;
2. `train_prefix.MAX_ITEMS / MAX_COPIES / MAX_BASKET_ITEMS /
   INVENTORY_ENCODING` from the split manifest's `input_schema`;
3. its own `DataDragon` for the generation's patch.

A worker that silently missed any of those would **not crash**. It would
return plausible, wrong features — this repository's stated failure mode
("not breakage — numbers that look right"). Note also that
`app/ddragon.py:load_json` falls back to HTTP when a cache file is absent;
several workers missing a file would each hit the network.

## The gate

`--verify` fails unless ALL of these hold:

| check | requirement |
|-------|-------------|
| worker configuration | every worker reports the parent's runtime declaration, patch, Data Dragon version, the four schema fields and the catalogue sha256 |
| numerical identity | `torch.equal(serial, parallel)` — **bit-identical, not `allclose`** |
| diagnostics | the merged `infeasible` / `replay_failed` counters equal the serial ones |

Both tensors' sha256 digests are recorded in the report so a later run can
be compared against this one.

## Benchmark honesty

The sample's candidate domains are the **real frozen top-128**
(`frozen_fit_full.pt` `top_idx`), not a random draw. A random basket is
infeasible far more often than a model's shortlist, and `econ_features`
returns early when it is — random ids would keep the equivalence gate valid
while making every timing optimistic. If that cache is missing the tool
falls back to a seeded draw and says so in the report.

## Expected payoff, stated before measuring

- Applies to economic-cache builds (`basket_econ_cache`, 4,528 s measured)
  and to VAL-style evaluations that compute features fresh
  (`basket_val_window_check` 6,237 s; `basket_mixture_val` longer).
- **Does not apply to training**: the loss contrast reads cached features.
- Amdahl: features are roughly 60-75% of a VAL run, so 8 workers give
  ~2.5-3x end to end, not 8x.
- On this machine RAM binds before cores do: 10 physical cores, but each
  worker carries its own DataDragon, and free RAM has been ~4 GB. Start at
  4 workers; `--bench` measures the real curve.
- `--profile` exists to test a separate hypothesis first: `econ_features`
  calls `purchase_sequence` per candidate and the decoder calls it again
  for the same `(row, action)` pairs. If that duplication is large, a
  per-row memo is a cheaper and completely risk-free win, and should be
  taken before any process pool.

## Usage

```
footwear_runtime.py --entry scripts\econ_features_parallel.py --review <review> -- \
    --profile --verify --bench --sample 512 --workers 4
```

Report: `data/reports/econ_features_parallel/equivalence_and_bench.json`.
Must not run while another training or evaluation job holds the machine.
