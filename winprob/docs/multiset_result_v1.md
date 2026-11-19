# Exact multiset likelihood does not improve purchase accuracy

2026-10-01. The complete fixed experiment and functional audit finished.
The proposed marginal-likelihood model FAILS the purchase-accuracy gate.
No M2 comparison, confirmation read or promotion is justified by this result.

## Primary result

Population: the same 3,350 fixed focal decisions from distinct games, scored
out of fold within the original 4,001-game FIT cache. All states of a query
game are excluded from its model's training. The 651 unavailable focal episodes
remain disclosed; this is an already exposed internal population, not a fresh
test and not an M2 displayed-policy evaluation.

Main-basket first exactness:

- Exact marginal model: 1,481/3,350 = 44.2090%.
- Matched categorical control: 1,515/3,350 = 45.2239%.
- Canonical-order-loss control: 1,491/3,350 = 44.5075%.
- Stronger frozen memory reference: 1,584/3,350 = 47.2836%.
- Older state-only reference: 1,471/3,350 = 43.9104% (descriptive only).

Marginal minus memory is -3.0746pp, paired 95% game-bootstrap interval
[-4.4179, -1.7604]pp: 222 corrected and 325 broken. Marginal minus categorical
is -1.0149pp, interval [-2.3582, 0.2687]pp: 229 corrected and 263 broken.
Marginal minus canonical is -0.2985pp, interval [-1.4336, 0.8963]pp.
Neither the primary accuracy thresholds nor the top3 guard pass.

Main top3 is 69.4925% versus the memory reference's 71.8209%: -2.3284pp,
interval [-3.6418, -1.0448]pp. The small gain against the weaker old state-only
model does not rescue the failure against the declared stronger references.

## What did improve, and why it is not the requested result

Conditional-catalogue NLL improves to 2.133272, versus categorical 2.333686,
canonical 2.195544 and memory 2.282992. The paired differences versus categorical
and memory are -0.200414 [-0.269648, -0.132083] and -0.149720
[-0.222674, -0.075439] respectively. This is a better probabilistic score on
these decisions; it is not evidence of more correct first recommendations.
Calibration error was not measured and must not be inferred from NLL alone.

The marginal model's raw true-basket NLL is 2.142141. It retains mean 99.1570%
probability mass in the fixed catalogue (5th percentile 95.5314%, minimum
53.6603%). The canonical model retains mean 98.9787%. Raw and conditional
probabilities remain separate in the artifacts; no missing mass was hidden.

The user's success criterion is the actual purchase. Lower NLL therefore
cannot replace the failed exactness result or justify sending this model to
M2's 859-game development comparison.

## Execution and verification

All six models completed 20 epochs each: 120 model epochs, 92,040 optimizer
updates, all opposite-fold states used once per epoch. Initial encoders match
exactly within each fold, and marginal/canonical heads have identical initial
tensors. All 10,050 held-out target-mutation checks left complete predictions
unchanged. There was no checkpoint selection, extra epoch, weight sweep or
mid-run recipe change.

The audit independently verified model IDs, training sets, initial digests,
epoch permutations, Adam counts/settings, parameter counts, frozen reference
arrays, every aggregate/slice/paired interval and the failed gate. It reloaded
all six checkpoints and replayed all 10,050 neural query predictions on the
original GPU/batch/store path: zero top3-cell mismatches and exactly zero
conditional-logp, raw-logp and catalogue-logmass errors.

The 13 synthetic tests cover the mathematical likelihood/gradients, repeated
copies, STOP, normalization, shared-prefix computation and matched controls.
They validate the implementation; they do not imply predictive improvement.

Result: `data/runs/multiset-probe-20261001-v1/experiment/report.json`, SHA256
704000a609c45f6623ee7ad5c13eb4ee9c4852dd35aecfbd353375afe92f7770.
Audit: `data/runs/multiset-probe-audit-20261001-v1/audit/report.json`, SHA256
c0f627d003093da43161167df193fb39006ba933dc9244f3b21eb864932420ff.
Protocols: `multiset_likelihood_design_v1.md` and `multiset_probe_protocol_v1.md`.

## Costs and decision

Training/evaluation took 1,426.0s (23.77 minutes); functional audit 50.25s.
CPU1, RTX4060, peak Torch CUDA allocation 1,012,843,008 bytes. Lowest sampled
available host RAM was 1,893,617,664 bytes, above the fixed 0.75GiB floor.
No resource bound was breached and no source workload was stopped.

Each encoder has 164,224 parameters. The categorical head has 1,220,469;
each generative head 63,805. The shared encoder is controlled, but head capacity
differs: this result does not disprove exact marginalization in every model.
It rejects this tested recipe for the requested purchase-accuracy improvement.

Close the recipe and preserve all checkpoints, code snapshots and negative
results. Do not choose the canonical control, extend training or reinterpret
NLL as a breakthrough after seeing the scores. No new 859-game evaluation,
test/confirmation payload, outcome, raw collection, source-runtime import or
served-policy change occurred. All 1,911 own confirmation games remain closed;
the report index is unchanged. The broader goal is active and unachieved.
