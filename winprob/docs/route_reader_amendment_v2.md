# Route-prior bounded-read amendment, before target access

2026-10-03. The source full-prior release completed successfully:56288rows
replay exactly and saved rows verify, manifest SHA
35113430e6997ec1ef993da5d9dc3f9325941138e21db24fab300ea3b5f470ac.
The first route-fit task then stopped at the declared host0.75GiB RAM floor,
at38912/56288marginal-preparation rows (93.25s). Launcher terminal reason is
'Host RAM floor reached'; both worker handles disappeared. This is verified
termination, not an observation timeout. No input lock, scaler lock, preflight,
fit, model, target access or new accuracy was reached. Preserve the partial
route-probe-fit-20261003-v1 artifact and all consumed v1 code unchanged.

The memory-mapped2.1GBfull prior can accumulate resident pages despite a loop
that processes128rows at a time. Other host memory consumers were not measured
as the exclusive cause, so no sole-cause claim is made. New bounded_npy_rows_v1
uses header validation, seek/read and explicit maximum512-row allocations;
it never maps or materializes the complete prior. This changes resource access,
not any input bytes, arithmetic, population, feature, kernel, loss, optimizer,
regularizer, gate or displayed decoder in route_probe_protocol_v1.md (SHA
f6e0a8de1fdaffa76dc7775457cb1f618058c384858eafe71997cb4d5d23ec18).

Create route_probe_data_v2.py and fit_route_probe_v2.py from the preserved v1
implementations, changing only this prior reader and binding this amendment.
Record the reader/amendment in code snapshots and manifests. A new immutable
run starts from original source probabilities, no partial v1 cache reuse. Keep
startup2.5GiB/live0.75GiB and all original time/workload/game guards unchanged.
Downstream independent audit must use the same bounded input access to avoid
recreating the oversized mapping; its arithmetic checks remain unchanged.
Small file fixtures verify exact scalar, sliced, irregular and paired indexing,
row-order/count retention, dtype/shape/header/length rejection and batch bounds.
No source native inference rerun and no model/data/threshold tuning is allowed.
