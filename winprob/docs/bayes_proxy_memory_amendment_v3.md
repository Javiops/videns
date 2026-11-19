# Bounded state-matrix access, unchanged proxy experiment

2026-10-03. Forecast attemptv1 stopped during code provenance collection:
Torch exposed a virtual relative _ops.py filename that was not a physical
workspace file. No input lock, interpreted action labels or forecasts were
reached. Forecast v2 changed only the physical-file filter and preserved v1.

V2 then passed input validation and encoded the first memory bank, accessed
only its opposite-fold training labels and ran its64-row input-only timing
pilot. It reached the second bank before the launcher terminated it at the
fixed0.75GiB host RAM floor (terminal launcher21:36:26UTC). It had not read
recipient past evidence, updated posteriors, created candidate distributions
or scored purchases. Preserve all partial v1/v2artifacts and consumed code.

The public dense state arrays were still persistent memory mappings; repeated
full-bank passes can retain their pages. Other host consumers were not measured
as an exclusive cause. New bounded_npy_array_v1 reads at most256leading rows
with seek/read, supporting the original trailing tensor dimensions and exact
numeric dtype. It groups contiguous row runs but never maps the full matrix.
After original validation, replace and close only champion_indices,
inventory_static, visible_level and global_public mappings. A copied focal
champion vector remains available for retrieval. Retain the smaller ragged
inventory tables and all original episode/fold/time metadata unchanged.

Create bayes_proxy_data_v2 and forecast_bayes_proxy_v3, binding this amendment
and the reader in snapshots. Before closing each original dense map, compare
rows0,middle,last byte-for-byte against the bounded reader. Synthetic fixtures
cover scalar, contiguous, irregular and duplicate reads, exact trailing shapes,
dtype, bounds and truncated-file rejection. New run from original inputs;
do not reuse a partial bank. Feature values, batches, checkpoints, k32, metrics,
controls and protocol SHAa60292830d167824f8d14a608020c4f75da3a023b65ef85d5c42ad7f2b91837b
remain unchanged. Keep startup2.5GiB/live0.75GiB RAM, idleGPU and180minute limits.
This resource repair is not a predictive result or permission to change the model.
