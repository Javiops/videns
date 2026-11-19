# IQL cache — independent integrity audit

2026-09-30, 08:38 UTC. **Passed.** The completed FIT cache matches every
source trajectory's action, reward, termination, timestamp, duration and
next-state edge. This establishes data/accounting integrity; no policy was
trained and no improvement or causal-value claim follows.

Evidence: `artifacts/audits/iql-cache-20260930-v2/audit.json`, SHA256
`8fba47e3016ecec961d2574eac90bd80925ff6be3277e955af24511da9b3ff3f`.
Reproducible auditor: `scripts/audit_iql_cache_v1.py`. The audit completed in
33.61 seconds using NumPy/memory-mapped arrays and one CPU thread; Torch was
not imported.

The audited cache is `data/runs/iql-cache-20260930-v1/cache`; its manifest
SHA256 is `1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea`.

- All **29 cache files**, **142 input/code bindings**, manifest sidecar and
  **25 array schemas** verified, then were checked again before completion.
- All **33,212 episodes / 392,448 decision transitions / 4,001 games** were
  checked against one complete streaming pass of the source FIT artifacts.
  The iterator exhausted successfully with its final verification flag set.
- Every episode has exactly one absorbing transition. The source terminal
  outcome reaches its last action once; earlier rewards are zero. Next-state
  pointers never cross an episode, player, game, fold or native patch.
- Fold assignment was independently recomputed from the declared game hash.
  Fold 0 has 2,000 games / 195,611 transitions; fold 1 has 2,001 games /
  196,837 transitions. Both terminal classes occur in both folds. Recomputed
  opposite-fold action coverage is **98.9172% / 98.8498%**.
- Full-action IDs and their main-group projection match the bound 13,166-
  action catalogue. All **39,498 native action/static vectors** across the
  three public patches were reproduced using the bound compiler. Inventory
  offsets/counts preserve 16,103,443 ragged tokens, with at most seven
  distinct items per player bag and no fixed-slot truncation.
- Exact feature parity was checked on the first 16 source episodes per
  native patch: **199 states on 16.17 and 180 on 16.18**, including every
  inventory bag. The 32 first-state mutation checks confirmed that changing
  ignored final-role, outcome and privileged-total-gold fields does not
  change the compiled public features.

The coverage denominator remains **36,029 validated source-complete player
chains before catalogue filtering**. The 33,212 encoded chains retain
**92.1813%**; feature-encoding exclusions are zero. This is the declared
conditional population, not all player decisions or every possible purchase.

No raw Riot archive, SQLite database, calibration/development game payload
or confirmation payload was opened. Raw-source truth was not re-audited.
Transition equality is exhaustive; state-feature parity is a fixed-sample
reproduction with the existing bound compiler, not an independent feature
implementation for all 392,448 states. Source/cache/compiler/protocol files
were not modified.
