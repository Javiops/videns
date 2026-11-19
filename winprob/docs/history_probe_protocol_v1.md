# History information for immediate purchases

Protocol v1, 2026-09-30. Status: root and independently reviewed; frozen before
real data preparation or model fitting. No result has been computed for this
recipe. Causal-history, selection, donor and paired-metric synthetic reviews
passed before this freeze. This is a new internal development experiment, not a repair
or reopening of the failed ordinary BC20 reranker trial.

## Decision and scientific claim

Test whether the previous two completed own purchase visits improve prediction
of the current retained main basket beyond the current public predecision
state. The current neural encoder aggregates inventories but does not retain
their purchase order. Past observations could clarify state or intention.
Neither explanation is identified solely by a predictive gain.

With observed current state O, available history H and current action A, the
optimal conditional categorical log-loss difference is

    H(A|O) - H(A|O,H) = I(A;H|O).

This identity motivates a discriminating experiment; a finite fitted-model
gain is not a direct mutual-information estimate or a policy-value theorem.
Belief-state reasoning is described by
[Kaelbling, Littman and Cassandra](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf).
Goal-conditioned imitation offers empirical motivation in
[RvS](https://arxiv.org/abs/2112.10751), but no future realized goal enters this
prototype, and neither paper establishes a League-specific benefit.

The primary endpoints are **unfiltered main-basket negative log likelihood
and first-choice exactness**, not legal displayed-policy accuracy. The
completed859-game diagnostic found a reconstructed-budget obstruction on108
catalogue hits, while the live adapter reads reported currentGold. Do not
alter wallets, mask logits by reconstructed legality, rerank M2, analyze wins,
or claim a useful deployed recommendation from this internal prediction test.

## Frozen data and game separation

Consume only the already versioned own cache
`data/runs/iql-cache-20260930-v1/cache/manifest.json` and its bound metadata,
public-state arrays, action catalogue/static arrays and retained action IDs.
Its immutable population is392448 transitions,33212 complete modelled player
episodes and4001 admitted FIT games across16.17/16.18. Keep all perspectives
and visits of a game in its original fold. Folds remain the first eight
big-endian bytes of SHA256(`iql-v1-fold:`+match_id), modulo2. Every learned
model evaluated on foldk is fitted from scratch on fold1-k only.

Revalidate the original admissionv2, all current global reservations and1911
own confirmation IDs before any new array/source payload access and before
publication. Verify cache/source manifests, sidecars, all consumed file hashes,
code/static bindings and prior admission ancestors. No raw archive, source
database, new game collection, development19 or sealed/confirmation payload
is part of this recipe. Do not import sibling-project code. No frozen old
IQL/IPO/ordinary model, compiler, source app or protocol is edited.

The cache previously selected complete encodable retained-action episodes;
its population is conditional on that original qualification. It is not a
representative sample of every human shop decision. Cross-fitting here does
not erase prior research exposure of this FIT population. Do not initialize
an encoder, head, embedding, optimizer or feature transform from BC20 or any
all-FIT-trained checkpoint. Fixed public static vocabularies may be shared;
learned representations may not.

The old reward/terminal outcome arrays are unnecessary and must not be opened
by the new dataset adapter/trainer. Source trajectory JSON records contain
terminal fields and may deserialize them while reading visit metadata; those
fields must not be indexed, used, summarized or exported by the new adapter.
State that boundary honestly rather than claiming JSON deserialization never
encounters an outcome. No outcome enters selection, model features or loss.

## Evaluation selection before target/history payloads

First publish a4001-game selection ledger using only bound game/episode,
timestamp and identity metadata. A game has one fixed focal participant:

    1 + big_endian(SHA256('history-probe-v1/focal/' + match_id)[:8]) % 10

If its cached episode is absent, record `no_eligible_focal_episode`; never choose
another participant, game or visit. The metadata census with this exact salt
found3350 available focal episodes and651 absent ones. This count is verified,
not a permission to select a replacement if metadata disagrees.

Within an available episode choose the smallest full SHA256 digest of

    'history-probe-v1/visit/' + mid + '|' + pid + '|' + ordinal + '|' + ts

where ordinal is the zero-based index in the complete source decisions list
(equal to cached episode offset), and integer fields use ordinary decimal.
Tie-break by(timestamp_ms,ordinal). Ordinal is not an invented source event
index. Keep first/second visits with missing history. Do not select on action,
support, wallet, target type, history usefulness, time window or model score.
Publish the complete ledger and SHA256 sidecar before opening retained-action
arrays or streaming source trajectory payloads. Verify every selected key at
each later stage. Evaluate one selected decision per available focal game,
equal weight per game. Report all651 unavailable games separately; they have
no selected focal-action endpoint and do not receive invented zero labels.

## Causal history and closure sidecar

Use the two immediately preceding **real purchase visits** of the same player
episode, newest first. Remove every explicit synthetic SAVE node (full action
ID0) BEFORE selecting prior-visit adjacency. Keep ward-only purchase variants
(nonzero full-action IDs) and their counts. Read visit_end_ms from the already exported FIT trajectories;
the existing cache omitted it. Stream those versioned files without raw or
database access and align every node by game, participant, source ordinal,
timestamp and existing complete-episode metadata. Revalidate original source
manifest scope before reading each shard. No per-state remapping or bridging
across missing/excluded episodes is allowed.

A history slot is available only if the corresponding previous purchase exists
and its retained basket is finalized strictly before current_start:
previous.visit_end_ms < current.timestamp_ms. A timestamp equality masks that
slot; do not replace it with an older purchase. Verify the source retained-basket
definition uses only events in that closed visit. Preserve the complete
counted purchase basket including wards.

This pre-fit revision is necessary because source SAVE markers are stamped at
death+15seconds but their existence checks no purchases through death+90seconds
and a later endgame condition. Their exported visit_end_ms is not a causal
availability time. Simply masking their content could still leak their
future-qualified existence by displacing an older history node. Removing
synthetic SAVE nodes before adjacency prevents that pathway. Require a
synthetic SAVE-insertion/deletion invariance test for the histories of the
same actual current decision. The current prediction target still preserves
the original source SAVE class: it represents a future no-buy-window label,
not a directly observed instantaneous action. Report its separate slice and
do not reinterpret this internal target as an actual live choice.

Age = (current_start - previous_visit_end_ms)/360000, float32, no clipping.
Absent slots have zero age, zero presence and zero content. Current state is
the existing frozen public feature allowlist; inferred final roles, outcomes,
future states and other-player private wallets do not enter. Mutating the
current and later action labels must leave current history features and its
predictions unchanged. These mutations may correctly affect later histories.

## Three fixed arms and shuffled-content control

All arms have exactly the same architecture and initial parameters within a
fold; they differ only in history input. Fit separate fresh models:

1. State: both history slots zero, including age/presence.
2. History: real available previous purchase baskets and closure ages/presence.
3. Shuffled: recipient ages/presence unchanged, but replace basket content
   using one deterministic donor context from another training-fold game.

Match a donor on current focal champion, patch, floor(current_ts/180000), and
the exact two-slot history-presence pattern. Donors always come from the
current model's training fold, including when the recipient is held out.
Exclude the recipient's entire game. Sort eligible donor state indices in
ascending numeric order and choose full-big-endian SHA256 of
`history-probe-v1/donor/`+heldoutfold+`|`+mid+`|`+pid+`|`+ordinal+`|`+ts,
modulo the eligible donor count. All integer fields use ordinary decimal.
Eligible lists may be cached per matching bucket and recipient game; this
does not change selection. Copy donor
previous baskets, not its current action or times. One donor supplies the two
slots. If no donor exists, use zero basket content while preserving recipient
timing/presence and record the fallback; never broaden the matching bins.

This control isolates the contribution of real basket content beyond timing
and additional capacity approximately. Coarse donor matching is not a causal
randomization of human intention. If history beats state but fails to beat
the shuffled-content control, do not claim evidence for useful basket memory.
Report donor coverage and history/tie-mask distributions by fold.

## Model and fixed optimization

Reuse the architecture of the frozen own public-state encoder, freshly
initialized: champion embedding16 and item-sum embedding32,22 inventory static
coordinates and visible level; shared player encoder64; focal/ally mean/max/
enemy mean/max plus37 public globals;357→256→128 with the existing activations
and LayerNorm. Reuse its item embedding to encode each history basket by a
count-preserving sum32. Add the native-patch23 public action coordinates,
closure age1 and presence1: each history slot has57 coordinates.

Concatenate state128 and two history slots114; Linear242→128, ReLU,
Linear128→9333. Every retained main group from the frozen catalogue is a class.
Main labels remove only ward IDs2055/772043; group0 preserves the source
SAVE/ward-only main-basket convention. Report typed no-buy/ward-only separately.
Do not learn a within-group ward conditional or call this a full-basket policy.

Seed20260930. Exactly20 epochs per arm per fold, all opposite-fold cached
transitions sampled once per epoch in one fixed seeded permutation. Reuse the
exact permutation order and batch256 boundaries across arms within a fold.
The last partial batch is retained. Adam learning rate3e-4, no weight decay,
gradient norm cap10, no early stopping, schedule, hyperparameter/checkpoint
search, reward weighting, dropout, or label-dependent transition filtering.
Loss is ordinary categorical cross entropy over all9333 groups. Save initial
parameter hashes, epoch-order hashes, step counts, losses, final checkpoints
and fixed selected-game predictions. Held-out target classes absent from a
training fold remain scored with finite full categorical mass.

## Fixed cheap baseline

Fit frequency counts on each opposite training fold only. There areG=9333
classes. Global p0(a)=(n_a+1)/(N+G). Champion p_c(a)=(n_ca+100*p0(a))/(N_c+100).
Exact(champion,current inventory multiset) p_ci(a)=(n_cia+20*p_c(a))/(N_ci+20).
Keep item multiplicities; unseen contexts naturally back off. No history,
outcome, inferred final role or learned encoder enters this baseline. Report
its main NLL/top1/top3 alongside all three neural arms; it cannot rescue a
failed history-vs-state/shuffled gate.

## Metrics, uncertainty and fixed gate

Use complete selected-game OOF predictions only. Missing predictions or
nonfinite losses invalidate publication; never drop hard selected rows.
Lowest canonical group ID breaks an exact class-logit tie. Compute NLL with
stable full logsumexp, never log of a rounded/underflowed exported probability.
Export target log probability and predicted top3 IDs, plus bound prediction
row hashes. Probability diagnostics use float64 normalization if exported.

Report mean NLL, main top1 and top3, per-fold metrics, and paired differences.
Use2000 paired whole-game bootstrap resamples, seed20260930, reusing exactly
the same sampled selected games for all arms. For relative NLL improvement,
recompute (state_loss-history_loss)/state_loss within each resample rather
than averaging per-game ratios. These intervals are conditional on fitted
cross-fitted models and this exposed population; they do not include model
retraining uncertainty or establish fresh confirmatory generalization.

A useful internal history-information signal requires all of:
- At least3000 selected games;100% prediction coverage; complete4001-game ledger.
- History reduces mean NLL by at least2% relative to state; paired95% lower
  bound of relative reduction is strictly positive.
- History improves main top1 by at least2 absolute percentage points versus
  state; paired95% lower bound is strictly positive. Use integer count
  arithmetic for the point threshold, not a rounding-sensitive comparison.
- History has lower NLL than shuffled content, paired95% lower bound positive.
- Every admission, strict-history, current/future mutation, initialization,
  training-order, full-class and provenance check passes.

Report purchases excluding explicit no-buy, explicit no-buy, and ward-only
counts/results; estimate subgroup metrics only with at least30 selected games.
Keep these exploratory slices visible without replacing the primary gate.
No after-results choice of epoch, mixing coefficient, donor scheme, history
length, cohort, target definition or successful secondary is permitted.

## Execution, artifacts and practical next step

Run new own-project scripts only. Data preparation streams existing artifacts
with CPU1 and a900-second limit. Model work uses the idle RTX4060,CPU1,
bounded batches and an1800-second hard stage limit including preload,
preflight, fitting, selected-game inference, checkpoint writing and metrics.
Before any full fit, measure optimizer steps including history gathering and
project1.25 times the measured per-arm update workload plus180seconds for
remaining work. Require that projection to fit within85% of the remaining
stage time. Save measured marks and reject before fitting if it cannot fit;
do not shorten arms or epochs. Preparation has its own declared900-second
budget and is reported separately. This30-minute model budget replaces the
earlier informal20-minute proposal before any new fitting or measurement.

Use a durable one-shot task; no long process attached to the agent shell.
Respect no-game/no-competing-benchmark guards, a measured startup RAM gate,
and a live host-memory stop at0.75GiB. Do not stop unrelated processes. Own
temporary review preview may be paused and restored if necessary. Store
immutable/versioned manifests, all consumed hashes, source/feature/model
snapshots, runtime versions, coverage, fallbacks and explicit failure state.
No report-index edit, source serving-policy change or automatic promotion.

A positive result justifies a separate frozen candidate and evaluation of
actual recommendations with independently known wallet and aligned decision
timing. It is predictive evidence toward the user's objective, not itself
the requested practical breakthrough. A negative result closes this recipe;
do not convert timing-only or post-hoc subgroup gains into success.
