# Future completion supervision for immediate purchases

Protocol v1, 2026-09-30. Independently reviewed and frozen before preparation
of the auxiliary cache or any new model timing/fitting. Paired-evaluation
synthetic tests pass; cache/model implementation tests must also pass before
their real stages. This is a new internal development experiment. Previous
unsuccessful recipes remain closed.

## Question and mathematical mechanism

Can supervision about a later completed item improve the representation used
to predict the immediate main purchase, without requiring that future label
at prediction time? With public predecision state X, current main basket A,
and a later completion label G, fit

    L = CE(A, p_theta(A|X)) + lambda * CE(G, q_theta(G|X)).

Lambda is fixed at one for real and shuffled auxiliary supervision, zero for
the matched main-only baseline. The auxiliary gradient must reach the shared
state encoder and shared basket embedding used by the served main head.
This is a finite-sample representation/regularization hypothesis, not a
causal action-value estimator. A predicted goal that is a deterministic
function of X provides no additional observed information conditional on X:
I(A; predicted_G(X) | X) = 0. We are not recovering independently verified
private intentions. The future labels never enter the main forward inputs.

Training-time privileged information is a general motivation, discussed in
[Generalized Distillation](https://leon.bottou.org/publications/pdf/iclr-2016.pdf).
This experiment is direct auxiliary supervision, not an implementation of
that paper's teacher/student distillation algorithm or a claim of novelty.
The active M2 already uses compositional basket representations. The new
mechanism relative to that served path is future-supervision gradients into
the representation actually used for immediate choices; compositional item
sharing alone is not a new contribution. A legacy next-item head that freezes
the served trunk is not this mechanism.

## Frozen population, labels and separation

Use only the existing IQL cache, its public state/action arrays, the audited
history selection metadata, public static artifacts, and the audited future
goal census. Keep all392448 states,33212 qualified complete episodes and4001
FIT games across16.17/16.18. This is the previously exposed, source-qualified
population, not fresh confirmation or representative coverage of all shops.
Retain the exact two whole-game folds and3350 selected decisions from the
history protocol;651 missing focal episodes stay reported, with no replacement.
The model for held-out fold k learns only from the other fold. Never initialize
from an old learned checkpoint. The1911 confirmation games remain closed.

Required ancestor digests include:
- IQL manifest: 1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea.
- Admission v2: fd13d31892465d676d201bc069a1e099f852ddb8b24c940c80340601e9ab5881.
- Public static: 5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0.
- Full catalogue: 14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8.
- Goal definition v2: 4457a015b1d08a0893acfc19b8016b7a7104d98f220f2c893949d055144a9072.
- Census report: 367293d55cdeae9aba107055be2eca8af0b4bd03c95de75fcc19dd424624d566.

Use ONLY the census's strictly-later six-minute target. Its definition is the
first qualifying closed real purchase visit after the current source ordinal
whose end is no later than current_start+360000ms. Preserve the full counted
qualified-item subset of that visit, including multiple distinct items or
copies. Synthetic SAVE nodes do not define completion goals. The106 public
eligible item IDs are identical across the three admitted static patches.
The public definition fixes257 nonempty exact goal multisets. Map census
NONE_FULL_HORIZON and NONE_TERMINAL_HORIZON to ONE EMPTY class: diagnostic
codes0/1 become goal0, and codes2..258 become goal1..257. Do not expose the
terminal/full-horizon distinction to the model, loss, or evaluation slices.

The descriptive census found178490 strictly-later nonempty labels (178285
single-item,205 multiple) and213958 EMPTY labels. These are supervision counts,
not evidence of learnable signal or benefit. The goal is not necessarily what
the player intended at the current decision; intervening events can change it.

Revalidate all digests, sidecars, quarantine exclusions, current reservation
metadata, and complete game separation before array access and publication.
Conflicting digests for one resolved path fail closed. No new raw archive,
database, source payload,16.19 target, sealed or confirmation payload access.
No sibling code imports. The new adapter/trainer never open reward.npy.
Terminal, elapsed-time and next-state arrays are never loaded, indexed, or
used as model inputs. Only the explicitly bound census timing ancestors
(terminal and elapsed-time arrays) may be byte-hashed solely to revalidate
census provenance; this hashing opens bytes without interpreting array values.
No winning/outcome label is used.

## Fixed matched shuffle control

For a model holding out fold k, shuffle goal labels ONLY among its training
rows. Match exactly on (current main group ID, patch, EMPTY versus nonempty).
Exclude the recipient's entire game. Sort eligible donor global state indices
numerically and choose full-big-endian SHA256 of

    future-goal-v1/donor/<k>|<match_id>|<participant_id>|<ordinal>|<timestamp_ms>

modulo the eligible count; integers use ordinary decimal. The source ordinal
is the same zero-based complete-episode offset used by the history selection.
Copy the WHOLE goal multiset identity, not independently shuffled items.
If no eligible donor exists, retain the original goal and record the fallback;
never broaden a bucket, drop a state, or insert a fabricated goal. Held-out
rows have shuffled-label sentinel -1 and never receive a training donor.

This preserves current-action coarsening, patch and future-completion presence
while disrupting the association of nonempty goal identity with current state
beyond that matching. Report coverage, same-identity donors, changed-label
fractions, and no-donor rows by fold and EMPTY/nonempty status. Keeping the
original on missing donors dilutes the control; it does not justify selective
row removal. Real-versus-shuffled is a predictive comparison, not a causal
randomization or a direct information estimate. Auxiliary goal prediction
accuracy cannot substitute for improvement in immediate-purchase prediction.

## Same-capacity compositional model in all three arms

Freshly initialize the frozen own PublicStateEncoder architecture with output
h(X) of128 dimensions. Its item embedding has32 dimensions and is shared with
the new count-preserving basket encoder:

    e(b, patch) = Linear128(ReLU(Linear55([sum_item_embedding32, static23]))).
    main_logit(a) = h(X)^T [w_a + e(a, patch)] + b_a.
    goal_logit(g) = (W_G h(X))^T e(g, patch) + c_g.

Here w_a is a free128-dimensional vector for each of9333 main groups; W_G is
a128x128 linear map without bias. All9333 groups remain in the main softmax,
including classes absent in the training fold. All258 goal identities remain
in the auxiliary softmax. No legality mask, wallet override, predicted/oracle
goal routing, candidate pruning, or ward conditional is fitted. Item sums
preserve counts, with padding only and no bag truncation.

The native patch's own frozen PublicFeatureCompiler produces the23 static
coordinates. Main group0 uses the explicit no-main/SAVE flag1, preserving the
source metric's ward-only/no-buy grouping. Goal EMPTY has an empty item bag
and ALL23 static coordinates zero: absence of a completion over six minutes
is not an instantaneous SAVE action. A missing native static description for
an action class sets ONLY its compositional e(a) to zero AFTER the MLP,
preserving its free
w_a and b_a and its softmax mass. Record every such fallback by patch. Every
observed current target must have valid native attributes or fail preparation;
never drop it. Require every goal multiset valid on every admitted patch.

Train three separate copies with identical complete initialization within
each fold: mainonly (lambda0), future (real goal lambda1), and shuffled
(matched shuffled goal lambda1). Mainonly does not read goal-label tensors in
its loss path. Its goal-only parameters naturally have no Adam state/updates;
this is expected, not evidence of an incomplete fit. Shared/main parameters
must receive the same number of update opportunities in all arms. A synthetic
gradient test must prove the real auxiliary loss changes the served shared
state/item/basket parameters, while main inference accepts no goal labels.

## Optimization and cheap baseline

Seed20260930. Two original game folds,20 epochs per arm per fold, all opposite
fold states once per epoch, batch256 including final partial batches. Exactly
the same seeded epoch permutations and batch boundaries across the three
arms. Adam3e-4, no weight decay, gradient norm cap10. No early stopping,
schedule, dropout, outcome weighting, hyperparameter search, or selected
checkpoint. Record initialization, fresh optimizer state, epoch-order hashes,
step counts, arm-specific loss components, and six final checkpoints.

The identical fixed champion/inventory frequency baseline uses global add1
over9333 groups, champion smoothing100, exact(champion, counted inventory)
smoothing20, fitted separately on each training fold. Its already frozen
predictions may be reused only with explicit complete binding and selection/
fold/target equality verification; otherwise refit the same recipe. It cannot
rescue a failed future-versus-mainonly/shuffled comparison.

## Primary evaluation, uncertainty and gate

Evaluate all3350 original selected decisions, one per available focal game,
with complete OOF main predictions and no legality filtering. Score immediate
main-basket NLL/top1/top3 over all9333 classes, using stable full logsumexp and
lowest canonical class ID for exact ties. Export target log probability, top3
IDs and row hashes. Do not compare these percentages directly to the859-game
16.19 native displayed-policy accuracy measured in the earlier failed trial.

Use2000 paired whole-game bootstrap resamples, seed20260930, the identical
resamples for all arms. Recompute relative NLL reduction from aggregate losses
inside each resample. These are conditional intervals for the fitted models
on exposed internal cross-validation, without retraining uncertainty.

All of the following must hold:
- At least3000 selected games, complete4001-game ledger,100% prediction coverage.
- Future main NLL reduces at least2% relative to matched mainonly, paired95%
  lower bound strictly positive.
- Future main top1 improves at least2 absolute percentage points over mainonly,
  paired95% lower bound strictly positive; test the point threshold with counts.
- Future main NLL beats shuffled auxiliary supervision with paired95% lower
  bound strictly positive.
- Every admission, provenance, full-class, initialization/order, auxiliary
  gradient-path and no-future-input check passes.

Mutating held-out current/future/goal target arrays must leave ALL selected
main9333 logits bitwise identical for each arm. Keep training donor targets
untouched during this inference-invariance check. Synthetic tests additionally
must make the separation explicit; no goal-label parameter exists in main
forward. Report existing explicit no-buy/ward-only/purchase diagnostic slices,
with metrics only at n>=30. Auxiliary accuracy/loss and slice metrics are
secondary and never replace the primary gate. No results-driven revision of
lambda, horizon, shuffle, targets, sample, epochs, or architecture is permitted.

## Resource budget and consequences

New own scripts and immutable artifacts only. CPU1. Preparation budget900s;
model-stage hard budget3600s (60minutes), fixed before any new timing or fit.
Include validation, preload, synthetic timing, all fitting/inference/mutation
checks, checkpoint writing and metrics. Preflight measures per-arm optimizer
step/gather work; require1.25 times projected remaining work plus180s overhead
to fit within85% of remaining stage time. If it cannot, stop before fitting;
do not shorten arms or epochs. Save the preflight and all resource failures.

Use a hidden, durable one-shot Windows task. No running game or competing
benchmark; start training only with at least3.25GiB host RAM and2GiB GPU RAM
free, stop own worker if host RAM falls below0.75GiB. Preparation needs1.5GiB
at launch. Do not close unrelated apps. The own review preview may be paused
and restored. Outer launcher hard wall90minutes, task95minutes, includes
resource waiting; an elapsed limit is not a model result. No source serving
change, automatic promotion, or report index edit.

A pass supports a separate frozen practical trial against M2 on aligned
decisions with independently known wallets; it does not prove win improvement
or complete the user's breakthrough objective. A failure closes this recipe
without retuning or promoting a successful secondary endpoint. All prior
frozen code, source exports, models and reports remain unchanged.
