# Outcome weighting lost to matched imitation on the revision challenge

Independent audit, 2026-09-30. The fixed IQL candidate failed its declared
usefulness gate. On the 158 already exposed development revision pairs from
144 games on patch 16.19, its weighted preference accuracy was **53.2051%**,
versus **59.6154%** for ordinary imitation with the same actor, warmstart,
optimizer state and subsequent batch order. The paired difference was
**−6.4103 percentage points**, with 95% game-bootstrap interval
**[−12.1795, −1.2577] pp**. IQL improved five pair predictions and worsened
15 relative to this control. This is resolved lower agreement with retained
expert purchases in this particular diagnostic; it is not measured win-rate
harm or a general refutation of IQL.

## What was actually tested

The [fixed protocol](iql_experiment_protocol_v1.md) trained on 392,448 complete
retained-basket transitions from 33,212 player-game chains and 4,001 admitted
FIT games on patches 16.17/16.18. All 13,166 full catalogue actions remain
distinct, including SAVE and ward copies. The actor received public-board
features and its own estimated wallet; it did not receive win probability,
other players' wallets, total gold, future outcomes, or legacy final roles.

Two fresh, whole-game cross-fitted critic/value sets received terminal team
reward once per chain. Each was trained for 20 epochs on the opposite fold.
The ordinary actor warmed for ten epochs; both continuations then received
ten further epochs from identical actor parameters and Adam state. The
candidate used globally normalized out-of-fold advantage weights; the
control used unit weights. Thus this comparison isolates the specified
weighting procedure conditional on this fitted representation and seed.

The challenge predicts whether a model prefers retained B over rejected A
in certified buy–undo–buy revisions. Its weights are 1/(10 × eligible pairs
for that player-game), conditioned on an eligible revision. Main-group odds
sum over every full-action variant before comparison with M2. The receipt
certificate proves both items feasible at B's time, not necessarily at the
pre-A model timestamp. These are preference labels, not causal win effects.

## Results and uncertainty

All intervals below use the same 2,000 paired whole-game bootstrap draws,
seed 20260929, with fitted policies held fixed.

- Frozen M2: **54.4872%**. IQL−M2: **−1.2821 pp**, interval
  **[−8.0784, +5.5241]**. There is no resolved advantage over M2.
- Frozen ten-epoch warmstart: **52.5641%**. IQL−warmstart: **+0.6410 pp**,
  interval **[−4.4304, +5.2989]**.
- Always favor the more expensive item: **57.2115%**; ordinary FIT
  champion/item-group frequency: **48.3974%**; cheaper item: **42.7885%**.
  Ties receive half credit. IQL did not beat the stronger price rule.
- **Exploratory control finding:** BC20−M2 was **+5.1282 pp**, interval
  **[−0.6330, +10.5599]**; 14 pair predictions improved and six worsened.
  BC20−dearer was **+2.4038 pp**, interval **[−7.1192, +12.7499]**.
  Neither establishes superiority. BC20−warmstart was +7.0513 pp,
  interval [+1.8987, +12.4203]; this was not the declared IQL success test.

BC20 was a required control, not the preregistered improvement candidate.
It must not be relabelled a successful confirmed method after viewing the
results. The challenge was already exposed in the IPO experiment; this run
is a cross-signal development diagnostic. Intervals cover game sampling
conditional on these checkpoints, not training-seed uncertainty, population
shift, or all possible model-selection decisions.

## What the weighting diagnostic says

Opposite-fold full-action support was 98.8834% overall: 388,066 supported
transitions and 4,382 raw-unit-weight fallbacks. Every saved weight was
independently reconstructed from the saved Q1/Q2/V values using the fixed
temperature, cap and one global normalization. The mean raw weight was
1.4326977056. Effective weight count was 70,854.84, **18.0546%** of the
392,448 transitions. Only **2.1794%** of transitions hit the raw cap, but
they carried **30.4237%** of the final actor objective's weight.

This is a substantial redistribution of training emphasis, despite high
catalogue support. It identifies a plausible sensitivity of this particular
outcome-weighting recipe; it does not prove that concentration caused the
loss. Nor does broad action occurrence establish local causal overlap.
Separate weighted and unweighted training objectives cannot be compared as
if a lower numerical loss implied better policy quality. Cross-fitting
prevents direct same-game critic fitting; it does not remove unobserved
skill, strategy, state aliasing, or long-horizon credit-assignment error.

The warranted decision is to retain the matched imitation control and
reject promotion of this IQL checkpoint. There is no reason here to tune
temperature, clipping, shaping or epochs repeatedly on these 158 pairs
until the sign changes.

## Independent verification

The [machine audit](../data/reports/iql_result_v1_audit.json) and its
[reproducible audit code](../data/reports/iql_result_v1_audit.py) checked:

- All **183** recorded input bindings, all **16** saved code/protocol files,
  five checkpoint digests, the current admission/reservation metadata and
  FIT/development separation. No original raw archive or sealed payload was
  opened. Qualified FIT export files were only streamed for their hashes.
- All 392,448 OOF support flags and raw/normalized weights; exact whole-game
  fold assignment; all **60** independently regenerated epoch permutations.
  The persisted epoch logs cover 20 epochs per critic, ten warmup epochs and
  ten matched continuation epochs.
- Warmstart actor and Adam hashes agree with both continuation starts.
  Saved Adam counters are 15,330 at warmup and 30,660 for each final actor;
  critic/value folds have 15,380 and 15,300 updates. The training lock records
  checkpoint freezing before preference metrics were inspected.
- All **158 predictions for each of the three actors** were recomputed on
  CPU from saved checkpoints and the previously exported challenge contexts.
  Every main-group ranking agrees; maximum CPU/GPU log-odds difference was
  0.000005723. A deterministic 64 supported transitions per fold reproduced
  saved OOF Q/V values within 0.000001029. This is a sampled checkpoint
  inference check, not a re-execution of every training update.
- Pair identities, actions, player weights and M2 masses match their source
  artifact. Weighted accuracies and paired bootstrap intervals were rebuilt
  independently without the experiment's metric helper.

The actual run took **1,784.00 seconds (29.73 minutes)**, with peak CUDA
allocation 905,992,704 bytes. The recorded minimum available host RAM was
847,499,264 bytes, above its unchanged 805,306,368-byte abort floor. No
resource gate, epoch count or statistical threshold was relaxed.

Exact result identities:

- [Run report](../data/runs/iql-outcome-20260930-v1/experiment/report.json):
  SHA256 `c812ef7d19909f9366354e82f79e936e3b1127b95a4b045f446c07ff7a146a4d`.
- [Predictions](../data/runs/iql-outcome-20260930-v1/experiment/challenge_predictions.npz):
  SHA256 `dce9fa2bc0c01afe0ce11d6a566ba9c1478fc97a8c135d9c047be2d180da0bbb`.
- [Saved code manifest](../data/runs/iql-outcome-20260930-v1/experiment/code_snapshot/manifest.json):
  SHA256 `81818704b699a7bf0ce24cec50157e94a31cf1fb1cb840dc9628e0c70dd13043`.

The audit JSON also binds the training lock, OOF weights, source admission,
dataset, reference manifest, each checkpoint and the audit code itself.

## Useful next direction

The concrete remaining candidate is the frozen public-board **BC20** actor,
not further outcome-weight tuning. Its better revision agreement is a useful
lead, but still uncertain against both M2 and the stronger cheap control.
Before any new fit, qualify this exact checkpoint through the source's full
catalogue and native legality/display path. Then test its ordinary-purchase
coverage and recommendations against M2 on one independently admitted,
predeclared evaluation cohort that also contains decisions without undo.
Freeze the checkpoint, population, comparators and success rule before
reading that evaluation. Keep the original IQL result visible.

This would answer the practical question currently left open: whether the
public inventory representation yields better usable recommendations across
normal decisions, instead of only a favorable point estimate on a selected
revision subset. Passing a genuinely untouched confirmation and a native
serving qualification could justify a product candidate; actual coaching or
win benefit would still require independent user/prospective evidence.
The broader breakthrough goal is **not established by this run**.
