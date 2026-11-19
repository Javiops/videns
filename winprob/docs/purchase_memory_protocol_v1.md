# Purchase memory v1: fixed internal screen

2026-10-01. User criterion: the recorded purchase is the correct label
(`m2_imitation_objective_v1.md`). This is an internal mechanism screen, not
M2 confirmation, a new win-probability experiment, or permission to promote.

## Hypothesis and mathematical intervention

A neural classifier may recognize similar purchase contexts better than it
compresses their exact basket frequencies into its output weights. Retain an
external memory of training contexts and their observed main-purchase labels.
This adapts the semi-parametric interpolation in Khandelwal et al.,
[Generalization through Memorization](https://arxiv.org/abs/1911.00172).
Their language-model evidence motivates a test; it establishes no League gain.

For a query x, use the L2-normalized 128-dimensional penultimate activation
h(x) of the already fitted state-only history-probe classifier. Search only
opposite-fold training states with the SAME focal champion. Rank by squared
Euclidean distance, breaking ties by immutable state index. Retain the nearest
state from each distinct training game, then at most k=32 games. A game cannot
supply multiple votes. No query purchase enters retrieval or distance.

For retained neighbors i, let d_i be distance, t=max(d_last/4, 1e-6), and
w_i = exp(-(d_i-d_first)/t) / sum_j exp(-(d_j-d_first)/t).
The memory distribution is R(b|x)=sum_i w_i * 1[B_i=b]. The single candidate
is P_memory = 0.5 P_state + 0.5 R. Empty retrieval falls back exactly to
P_state. All constants are fixed here; no sweep, fitted gate or retraining.

Controls: unchanged P_state; the same mixture using training labels shuffled
within focal champion, patch and three-minute decision-time bin (seed20261001,
fold-specific deterministic permutation). It keeps neighbor identities and
weights fixed, breaking detailed state/purchase association inside each bin.
Report unchanged shuffle labels and small/singleton bins; this is an imperfect
negative control, not independent evidence about every possible confounder.
Report pure-memory top1/top3 as secondary (no finite-NLL claim for zero support).

## Locked artifacts and partitions

Use ONLY the existing own-project IQL FIT cache (4001 games,392448 states),
its validated history sidecar and the fixed3350 focal decisions already chosen
without labels for the history probe. Reuse the two state-only checkpoints
from `data/runs/history-probe-20260930-v3/experiment`:

- fold0: e050c340298e5495fb966378d33cd133b558cf6302f673db5d8a14188a84364a
- fold1: 5381e72e1606f773817963c317652ecb2d6b568316ae189a63261df4565a1440

History manifest SHA256:
1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b.
Base manifest SHA256:
1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea.
Frozen original predictions SHA256:
27eb094eb1dcfc1551b375446684b0b423d86c03b79077b2ccb5772c8c0968f5.

For each scored fold, encoder training IDs AND memory IDs must equal the
opposite-fold game set, with zero overlap with queries. Revalidate current
reservations before array access and publication. Reuse the own validated
HistoryDataset interface; it opens no rewards, terminal outcomes, raw archives,
859-game development payloads, or confirmation/test payloads. No cross-project
runtime import. The already exposed3350 OOF decisions give exploratory evidence
conditional on these fitted encoders; do not call them a fresh test.

## Evaluation and integrity

Main top1 exactness is the primary metric; preserve counts and existing ward
marginalization. Top3 exactness and target NLL are secondary, with paired2000
whole-game bootstrap draws, seed20261001. Every selected decision stays in the
denominator. All651 unavailable focal episodes remain disclosed in the full
4001-game selection ledger; do not replace them with another player.

Before interpreting candidate scores, require exact original state top1/top3
reproduction and maximum absolute target-log-probability error <=1e-5 against
the frozen original file (bounded floating-point allowance for streamed batch
shape; no changed weights). Persist both the error and the observed mismatch
counts. Require source/model/code digest stability, verified fold exclusion,
at most one neighbor per game and target-independent neighbor construction.
Store neighbor IDs/distances/weights and all compact query predictions.

A promising internal signal requires at least+3 percentage points top1 over
state, paired95% lower bound>0, a positive top1 paired lower bound over the
shuffled-memory control, and no increase in mean NLL versus state. Report
every result if the gate fails. No retuning this recipe on these scores.

This screen has NO legal displayed-policy decoder, and its state baseline is
not M2. Passing only justifies freezing a subsequent source-native comparison
against M2, including candidate legality and costs. It cannot establish
practical M2 benefit, full-basket/ward correctness, win benefit or completion.

## Resources

One CPU thread, bounded streaming CUDA inference (batch256), no optimizer or
resident full-state GPU cache. Host startup floor2.5GiB; live floor0.75GiB.
No run during League or source benchmark/capture workloads. Hidden durable
one-shot task,20-minute worker/30-minute launcher wall limits, new immutable
output directory. Preserve old source/artifacts/index. Abort on integrity,
resource or reproduction failure; do not narrow the scored population.
