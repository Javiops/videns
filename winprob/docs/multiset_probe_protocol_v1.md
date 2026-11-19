# Unordered purchase likelihood: fixed training comparison

Declared 2026-10-01 after the synthetic correctness/cost stage, before any
training or new prediction/target comparison. The mathematical construction
and reference papers are in `multiset_likelihood_design_v1.md`. The previous
fixed memory recipe remains closed; its frozen internal predictions are a
stronger reference here, not a candidate to retune.

## Data and comparison

Use the original validated own FIT cache: 4,001 games / 392,448 states and
the unchanged two whole-game folds. Score the same 3,350 fixed focal decisions
selected by the history protocol. Keep the 651 unavailable focal episodes in
the ledger and do not replace players/visits. Revalidate current reservations.
No reward, terminal-outcome, raw archive, development859 or sealed input.
Selection, field allowlist and per-game OOF membership are immutable.

Three newly trained arms, each with the existing PublicStateEncoder and
IDENTICAL initial encoder tensors within each scored fold:

1. `categorical`: state128 -> Linear128/ReLU -> 9,333-group categorical head;
   cross-entropy on the actual main-group label.
2. `marginal`: the tested MultisetTransitionHead, exact negative log probability
   of the observed unordered retained multiset, summing all distinct orders.
3. `canonical`: identical initial encoder/head and architecture to `marginal`,
   but training uses only the ascending-item-ID sequence (including repeated
   copies), followed by STOP. At evaluation it is interpreted as the same
   bounded generative model and uses the same exact unordered probability.

All use the public current-state feature compiler already audited for the
FIT cache. No past/future action input, target-derived wallet or inferred role.
The generative heads use the 196 item IDs of the already frozen main catalogue,
with STOP0 and maximum length six; no vocabulary selection from scored labels.
Train end-to-end from clean initialization, 20 epochs, batch256, Adam lr3e-4,
default betas/epsilon, weight_decay0, gradient norm cap10, float32/noTF32.
The per-fold seed is20261001+fold; a separately generated shared encoder state
is copied exactly. The two generative heads also copy identical initial state.
Use the same deterministic epoch row orders across arms (hash of
`multiset-probe-v1/order/{fold}/{epoch}` seeds a NumPy permutation). Every epoch
uses all opposite-fold states once. No early stopping/checkpoint shopping,
warm start, parameter search, data augmentation or continuation past20epochs.

The shared encoder is controlled; head parameter counts differ and must be
reported. This is a learning-objective/parameterization comparison, not an
equal-parameter architecture theorem. Record all losses and optimizer steps.

## Prediction and gates

Use exact shared-prefix dynamic programming to score every one of the same
9,333 catalogue main groups for both generative arms; no beam-search error or
candidate pruning. No held-out label enters prediction. Report raw retained
catalogue probability mass. Normalize over the catalogue only for the explicitly
labelled conditional-catalogue NLL comparison. Primary ranks do not change
under that normalization. Original categorical models have mass1 by definition.
Do not claim the generative model allocates all probability to the catalogue.

Primary: main-basket top1 exactness of `marginal` against BOTH the newly trained
categorical arm and the stronger frozen memory predictor from
`purchase-memory-20261001-v1` (47.2836% on these same 3,350 OOF decisions).
The original state-only reference (43.9104%) is descriptive only. A gain
against a weak new control must not repeat the mistaken inference that an
internal improvement necessarily helps M2.

Useful internal gate: at least+3pp and a positive paired95% lower bound versus
each primary reference; no mean top3 loss versus the frozen memory reference;
all integrity, finite-gradient, complete-training and membership gates pass.
The marginal/canonical comparison attributes the effect of summing orders and
is reported with its interval. It cannot rescue a failed primary.
Secondary: top3, conditional-catalogue NLL, raw true-basket NLL and retained
catalogue mass for generative arms, repeated-copy and basket-size slices.
Use2,000 paired whole-game bootstrap draws, seed20261001. No positive slice,
canonical arm or later epoch can replace the declared primary.

Integrity: initial-state digests, full epoch/order/optimizer counts, explicit
opposite-fold training IDs, no query game overlap, forbidden-target mutation
invariance of predictions, frozen artifact/code digests, and independent
observed-multiset/metric verification. Keep all queries, including errors and
unsupported results, in the denominator. No neural model is promoted.

Passing is only a reason to freeze a later direct M2 comparison; it is not
practical benefit or completion. These internal decisions are already exposed.
All1,911 own confirmation games and source/global seals remain closed.

## Resource bound and current status

The prototype passed eight synthetic correctness tests. Its head-only benchmark
on RTX4060 measured17.0ms per256-example optimization step and6.58ms per batch
of eight full-catalogue queries, peak217,949,696 CUDA-allocated bytes. These are
synthetic throughput measurements; the encoder/data-gather costs and single-shop
end-to-end latency are NOT included. No expected accuracy gain follows from them.

Before any real fit, measure end-to-end synthetic steps for all three arms and
full-catalogue inference with the actual tensor-store path. Require projected
complete work including hash/publication overhead to fit within80% of the
remaining fixed60-minute worker budget. Failure stops before training and is
reported as a resource feasibility failure, not a negative accuracy result.
One CPU thread, own idle CUDA runtime, startup RAM>=3.25GiB/live>=0.75GiB,
no League/source benchmark/capture workload. Durable hidden one-shot launcher;
new immutable run directory. Preserve code/data snapshots and failed runs.

Current status at declaration: mathematical implementation, eight synthetic
tests and head-only cost benchmark complete. Real training has NOT run. The
training harness and its full preflight still need implementation/verification.
