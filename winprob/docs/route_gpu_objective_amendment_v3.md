# Exact-objective CUDA amendment, before target access

2026-10-03. The bounded-reader v2 completed all56288route marginals and fixed
zero-model parity, resolving the earlier oversized prior mapping. Its preflight
then stopped before targets/fits: sustained CPU objective times were
.69191486,.70226790,.74218842,.58266054seconds across the four comparisons.
The unchanged1.5*max*500*8*4+900projection was18712.5221seconds, above80% of
10658.7149remaining seconds. No actual fit or score was run. The source prior,
v1/v2attempts and consumed code remain immutable.

Use the same analytic likelihood/gradient in float64CUDA, with fixed log(m0)
and log(R(y)) cached on device. SciPy L-BFGS-B, zero initialization, ridge,
maxiter500,gtol1e-7,ftol0,maxls50, all data/features/kernels/folds and success
gates remain unchanged. No autograd, mixed precision, TF32, new parameter,
target, optimizer restart, hyperparameter tuning or support change.
The published statistical protocol remains route_probe_protocol_v1.md SHA
f6e0a8de1fdaffa76dc7775457cb1f618058c384858eafe71997cb4d5d23ec18, with this
explicit computational-backend exception before any real target access.

Configure deterministic torch and CUBLAS_WORKSPACE_CONFIG=:4096:8. CPU1,
startup2.5GiB/live0.75GiB host RAM, idleCUDA3072MiBfree, no League/competing
protected work, same180minute worker and unchanged conservative cost formula.
The bounded native-prior reader stays in use. Recompute original marginals in
a fresh v3run; do not reuse incomplete or outcome-bearing prior attempts.

Before real targets, compare GPU values AND analytic gradients to the original
CPU implementation on each fold/arm's16input-only pilot rows, dummygroup0
targets and fixed nonzero synthetic coefficients(seed20261003). Require both
absolute discrepancies<=2e-12. The full zero-fit parity check remains. Then
measure2warmups+5sustained full-objective calls for each fold/arm, including
device synchronization/host return; apply the original projection without
lowering its500iterations/eight evaluations/four fits/margin assumptions.

Small synthetic tests independently compare zero/nonzero/large-shift objective
values and gradients, finite differences, and fitted signal recovery. Prediction
and native decoding arithmetic do not change. The receiving audit verifies
real saved stationary points using the original CPU objective, finite
differences and exact original CPU prediction batches. GPU and CPU floating
point sums need not be bit-identical; declared2e-12receipts and1e-6stationarity
limits bound that distinction. No global-optimum or useful-prediction claim.
