# Complete causal learner: fixed two-candidate comparison

2026-10-04, declared before real student fitting or any student purchase score.
This trains a complete predictor with the public board/action architecture in
causal_student_design_v1.md. Predictions do not consume M2 probabilities.
The observed retained purchase remains the ground truth in every arm.

## Frozen inputs and game separation

Use causal-student-inputs-20261004-v1/inputs/manifest.json,
SHA4b45bb0f8e68d68645d1ac40c3f1a6c06780fe8e09667f9366146e966d5b8c2a:
56288states,3518FIT16.17games, original2900focals and618unavailable games.
All game perspectives/states remain together in the original two folds.
The feature stage reports25530selected player-games, constant selected rune
pairs,0missing keystones,2missing secondary styles and0unknown nonzero IDs.
Keep all states; missing rune index0is the declared unknown category.

Reuse the own public-state cache through explicit selected state indices;
stream/bound host reads. Native teacher priors are the immutable9333-class
float32 release route-prior-20261003-v1, manifest SHA
35113430e6997ec1ef993da5d9dc3f9325941138e21db24fab300ea3b5f470ac,
matrix SHAa2c4b14975027a7aafafe5a4a879322c2ea74f2116f9fd021d3128be4641b678.
Teacher package trained on6354source16.18games is disjoint from these16.17games.
Verify original admission and all reservations before payload access.

Teacher rows may physically reside in a shared fixed GPU input table, but
each model's loss may gather ONLY the opposite fold's rows. Held-out teacher
rows are not student inputs, pseudo-targets, early-stopping data or model
selection signals. Current teacher probabilities are used only for the frozen
M2 reference during later paired scoring. No final role is a student input.
Teacher privileged information is confined to training supervision, not passed
through a runtime multiplier. No859/1911/source/global sealed payload reads.

## Model, loss and controls

Use exactly CausalStudent from causal_student_v1.py: eleven tokens including
the global query, two64-wide four-head attention layers, FF128,dropout.1;
shared count-aware item embeddings and nonlinear64-dimensional action encoder;
bilinear plus192->256->64->1 joint interaction, candidate chunks512, deep
activation checkpointing. Public catalogue has9333main baskets and padded
distinct-item width5, with total copies preserved and no truncation. Empty-main
group0is not equated with full no-buy. No source runtime imports or weight reuse.

Three arms per fold, six freshly initialized fits:

- hard: observed-purchase CE. This is a declared candidate in its own right.
- distilled: optimize.5*(observed-purchase CE + KL(native teacher || student)),
  temperature1. This is the second declared candidate. Record both loss terms.
- shuffled_teacher: identical regularized loss and architecture, but permute
  teacher rows within TRAIN-fold focal-champion and floor(timestamp/180000)
  bins. Seed20261004+heldoutfold, sorted bins and ascending ordinal rows. It is
  a diagnostic control, never eligible for promotion. Report singleton/self
  mappings and preserve row confidence/entropy; no label-dependent shuffle.

Within a fold, all arms start from the exact same initial state dict and use
the same batch order and dropout seed sequence. Model seed20261004+fold.
Epoch order is generated independently by SHA256 of
'causal-student-v1/order/'+fold+'|'+epoch, first8big-endian bytes as NumPy seed,
permuting ascending TRAIN ordinals. Reset Torch seed to20261004+fold immediately
before each arm. Teacher shuffling uses a separate NumPy generator.

Adam(lr3e-4,betas.9/.999,eps1e-8,weight_decay0), batch128,20epochs, no scheduler,
no mixed precision/TF32. DeterministicCUDA/CUBLAS. Final epoch20only, no restart,
architecture/weight/temperature sweep, checkpoint selection or validation-driven
stopping. Every selected TRAIN state is used each epoch. Six fits total26460
updates (220batches for scoredfold0 and221for scoredfold1 per epoch, including
last partial batches). Multiply each mean minibatch objective by batch_rows/128
before backpropagation, so the final partial batch does not upweight its rows.
The input selection has16states per game. Average epoch loss accounting uses
exact row counts, not equally weighted batch means.

## Inference and fair native comparison

Freeze all six models before any held-out prediction. Generate full9333-class
probabilities from public state/runes/catalogue only, fixed128-row GPU batches
in ascending focal ordinal within each fold. Float32 logits are normalized in
float64. Mutating or withholding held-out purchase labels and teacher rows
must leave student predictions unchanged. Preserve all2900query decisions.

Export all three learned distributions and originalfloat32M2 prior. Decode
through unchanged native MixturePolicy.options: same native16.18 world,CUDA
top128, wallet, recursive feasibility, up to3distinct main actions and original
SAVE fallback. Require exact M2 replay and forbidden-field mutation invariance.
Audit every arm's options again in a clean native16.17process. No post-audit
filter/replacement or source/frozen policy edit.

Primary native comparison is each declared candidate (hard,distilled) versusM2
on displayed main first exactness. Also report full/typed first, top3, nonempty
main first, full-catalogue NLL, corrected/broken decisions, neutral/padding
semantics, costs and all controls. Use2000paired whole-game bootstrap draws,
seed20261004. Report ordinary95%intervals plus97.5%two-sided intervals
(quantiles.0125/.9875) for the TWO candidate-minus-M2 primary comparisons as
conservative two-candidate multiplicity handling, conditional on fitted models.
This does not correct the broader exposed research history or training uncertainty.

A candidate qualifies only with>=3pp first gain, adjusted primary lower>0,
positive ordinary95%nonempty lower, no decrease in mean top3, no increase in
mean catalogue NLL, zero native17/18legality failures and all audits passing.
If both qualify, select higher displayed first, then lower NLL, then hard on an
exact tie. If neither qualifies, close the fixed candidate family; do not rescue
shuffled_teacher or a slice. Distillation itself is credited only if its paired
contrast against hard and shuffled_teacher supports that claim; architecture
benefit and teacher benefit must not be conflated. Neither an internal positive
nor compression/latency improvement alone completes the practical goal.

## Resource and reproduction requirements

CPU1, idleCUDA3072MiBfree at startup, host2.5GiB startup/.75GiB live, no League
or competing protected workload, hidden worker6hours. Preload only the selected
state/rune/catalogue tensors and fixed teacher matrix on GPU, using bounded
host transfers; no complete teacher host array or full-cache preload.

Before actual labels, use a throwaway model on real training INPUTS, dummy
group0labels and their training-only teacher probabilities for a cost check:
two warmups/four timed updates, samebatch128/chunk512/checkpointing. Require
1.5*max_timed_step*26460+900seconds<=80%remaining worker time and at least1GiB
device headroom using peak reservation and observed free memory. Reset model/
optimizer/RNG for real fitting afterward. No labels/checkpoint is selected by
that pilot. Stop and preserve explicit diagnostics on resource/schema failure.

Save model state dicts, train identities, orders, input/code hashes, loss terms,
epoch/update counts and predictions. Independent audit rechecks fold/teacher
permissions, checkpoint lineage, exact original inference batches, public
permutation/count invariants, native receipts and every metric/interval/gate.
Student inference contains no teacher or current-label parameter. Original
index, source data, sealed games and served policies remain unchanged.
