# Bounded full-basket IQL experiment v1

2026-09-30. Fixed before IQL fitting or inspection of its results. This tests
whether outcome-weighted learning improves a purchase policy over matched
ordinary imitation. It is not another WP forecasting experiment, an assertion
of causal identification, or a new confirmation. The 16.19 revision challenge
is already exposed through IPO and will be labelled a cross-signal diagnostic.

## Action unit and reference

Use the source-defined complete **modelled retained-purchase multiset**,
including ward copies. This is not a primitive buy/sell/undo policy, nor a
claim to model every consumable or manual no-buy. The fixed source catalogue
contains 13,166 full actions and maps them to 9,333 main-purchase groups.
Full SAVE is distinct from buying one or two wards, although their main group
is the same. Map the observed no-buy-death sentinel to the catalogue's explicit
SAVE action; never invent a manual no-buy label.

Catalogue source: versioned `full-action-catalogue-20260930-v1/catalogue.json`,
SHA256 `14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8`.
Copy it across projects only as a bound artifact. Keep full-action ordering,
canonical multisets, action kinds and `full_to_group` unchanged.

M2's main probabilities alone are not a full-basket policy. A separately
available reference lift is

```text
nu_A(b|s,g) = q_A(b|s) / sum_{c:group(c)=g} q_A(c|s)
pi_joint_ref(b|s) = pi_M2(group(b)|s) * nu_A(b|s,group(b)).
```

This joint reference has M2's exact main marginal but is not native M2's
whole displayed policy. It is optional evaluation material, not a training
propensity. The first experiment uses original IQL advantage-weighted full
categorical cloning, **not an M2 residual actor**. No dense M2 distribution
is required for every training transition: 400,000 by 13,166 float32 values
would be approximately 21.1 GB before metadata. Full categorical training
logits exist only for the current minibatch.
If the optional joint reference is exported, compute its within-group
conditional directly from corrected A logits with groupwise logsumexp;
do not divide global float32 masses that may already have underflowed to zero.

## Frozen population and eligibility before fitting

The authority remains admission v2 and the context population: 4,002 original
fit games, 1,000 calibration games and all 859 development games on 16.19.
Read terminal outcomes only for the qualified FIT trajectories used for
learning. Calibration is not fitted; the preference challenge needs no new
calibration/development terminal-outcome access. The 3,118 global sealed IDs
and 1,911 own-confirmation IDs remain excluded before source access.

Source-owned trajectory exports must retain every modelled decision and the
absorbing terminal. Use `trajectory_contract.compile_transitions`: terminal
reward belongs to the preceding transition. Never stitch over a source gap.
If any action in a player-game lies outside the fixed full catalogue, exclude
the **whole player-game**. Apply the same whole-chain rule to unencodable
state, unknown static item/champion identity, malformed timestamps and
uncertain source versions. Do not truncate inventories, remap unknown actions
to SAVE, discard an intermediate step, or condition away unknown reference
mass. Catalogue restriction is a population limitation, not repaired coverage.

Before training require all of:

- At least 1,000 distinct FIT games, 5,000 complete player-game trajectories
  and 100,000 transitions after every eligibility filter.
- At least 70% of otherwise source-certified complete FIT player-games survive
  catalogue and feature encoding. Publish exclusions by game, player, patch
  and reason, including before/after transition counts.
- Each critic-training fold contains at least 450 games and 40,000 transitions.
  At least 70% of each scored fold's transitions have a full action observed
  in its opposite critic-training fold.
- The frozen preference evaluation retains at least 50 development games and
  100 pairs after common encoding/support exclusions.

These are engineering/support minima, not power or positivity guarantees.
A failed gate stops this recipe; no threshold relaxation or catalogue expansion
is permitted within this run.

## Exact POV tensor boundary

Use the source-certified pre-purchase state after permitted opening sales.
Preserve the source's visit clock semantics; do not claim an exact primitive
transaction timestamp. Read only own champion, inventory, `gold_est` certified
as `prequential-v3`, level, CS and K/D/A; the existing public-board contract's
other champion/team, item and level fields; and public objective counts.
Exclude `total_gold`, `others[].gold`, positions, WP, outcome fields, target
fields, final roles, all role/quest/footwear flags, `can_complete`,
`gold_after_complete`, and other derived completion/slot helpers. The frozen
M2 comparator's legacy inferred roles remain separately disclosed.

Vocabularies are fixed by the bound public static metadata across 16.17,
16.18 and 16.19, not by labels. Champion/item embeddings start at zero; index
zero is fixed zero. Thus identities not encountered during that model's own
training keep zero embeddings, while known native-patch item attributes remain
available. An identity missing from this static union is an explicit encoding
failure. Report unseen-in-training identities separately from unknown-static
identities. No test-dependent scaler or vocabulary fitting is allowed.

Each player has champion embedding 16, an inventory **sum** embedding 32,
22 public inventory coordinates and level/18: 71 inputs. Inventory bags are
ragged and preserve every copy; no arbitrary slot/token truncation. The 22
coordinates are the existing IPO action coordinates excluding SAVE: total
item price/3000; HP/1000; mana/1000; AD/100; AP/100; armor/100; MR/100;
movement speed/100; attack-speed fraction/.5; crit fraction/.5; lifesteal
fraction/.3; ten tag counts/6 in the frozen IPO order; item-count/6.
Use the bound native-patch public item definitions for these state attributes.
An absent stat key means zero recorded stat; an absent item definition fails.

The shared player encoder is Linear(71,64), ReLU, Linear(64,64), ReLU.
Concatenate focal token, ally mean, ally max, enemy mean and enemy max:
320 values. Pool exactly four allies and five enemies; within-team permutation
must not affect features. Add these **37 globals in this exact order**:

1. `ts/2400000`, blue indicator, `gold_est/3000`, `cs/500`, `kills/20`,
   `deaths/20`, `assists/20` (7).
2. Ally then enemy objective counts, each ordered towers, dragons, barons,
   heralds, voidgrubs and divided by 10 (10), then their matching presence
   flags in the same order (10). Missing counts use zero plus presence zero.
3. Own team including focal, then enemy team: sum of held-item total values
   /30000, maximum player share, top-two share, positive-total flag (8).
4. Focal held-item value / own-team held-item total, and its availability flag
   (2). Fractions with zero denominator use zero plus availability zero.

Allocation uses item values only, not totalGold or others' wallets. Own
estimated wallet is the separate third global. No final-duration or future
income normalization is used. Machine-readable names/order must match the
Torch model module constants before a real tensor cache is accepted.

## Fixed network capacity

Each network has its own state encoder: Linear(357,256), ReLU,
Linear(256,128), ReLU, LayerNorm(128), following its shared-player encoder.

- Actor: Linear(128,13166), full categorical softmax. It learns all full-action
  logits and therefore the within-main-group ward conditional; it does not
  freeze `nu_A`. Never train by normalizing only over a preference pair.
- Q1 and Q2: independent complete networks. Encode the observed full action
  using an item sum embedding 32 plus the 23 fixed public action coordinates
  including SAVE, followed by Linear(55,64), ReLU, Linear(64,64), ReLU.
  Concatenate state128/action64; apply Linear(192,256), ReLU,
  Linear(256,256), ReLU, Linear(256,1), sigmoid. Item embeddings may be shared
  between state and action within the same Q network, never between Q twins.
- V: its own state encoder followed by Linear(128,256), ReLU,
  Linear(256,256), ReLU, Linear(256,1), sigmoid.

Use native-patch public item coordinates for the observed critic action;
canonical full-action IDs/order stay fixed. Sigmoid value heads are this
application's bounded parametrization for undiscounted binary return, not a
claim of calibration. Linear layers use the model module's fixed seeded
initialization; persist the initial-state hash. No model-width/depth sweep.

## Rewards, cross-fitting and optimization

Gamma=1. Every nonterminal reward is zero; the final transition receives its
focal team's Y once, with bootstrap multiplier zero. Variable shop intervals
are recorded but not arbitrarily discounted per visit. No shaping is used in
the primary recipe. A later shaping comparison must keep terminal potential
zero and preserve total return; it cannot be advertised as new action data.

Using detached targets, the updates are

```text
Qbar = min(Q1_target, Q2_target)
L_V  = mean[ |.7 - 1[Qbar(s,a)-V(s)<0]| * (Qbar(s,a)-V(s))^2 ]
y_Q  = r + (1-done) * stop_gradient(V(next_s))
L_Qj = mean[(Qj(s,a)-y_Q)^2], j=1,2
Qj_target <- .995*Qj_target + .005*Qj_online after each critic update.
```

Fit the expectile to action-conditioned Q, not realized wins or individual
WP increments. This distinction and observed-action critic updates follow
[Kostrikov, Nair and Levine, IQL, Eqs. 5–7](https://arxiv.org/pdf/2110.06169).

Assign whole FIT games to two folds by SHA256(`iql-v1-fold:` + match_id),
interpreting the first eight digest bytes as a big-endian integer modulo two.
All ten players, all times and outcomes from one game stay together. Train
fresh Q1/Q2/V on the opposite fold and freeze them before scoring that game's
actor weights. Critic encoders are **not** initialized from an all-FIT actor.
Fixed public static preprocessing is shared; learned model state is not.

Critic minibatches sample observed transitions uniformly. No inverse episode-
length weighting is introduced into Bellman regression. Such a weight depends
on future trajectory length and can change the conditional transition measure.
Evaluation uncertainty still clusters by game.

For each FIT transition compute the out-of-fold raw weight
`omega=min(20, exp((Qbar(s,a)-V(s))/.1))`. If its full action was absent from
that critic's training fold, use omega=1 and record the fallback; do not query
arbitrary alternative actions to choose a weight. Compute
`c=mean_FIT(omega)` once and persist `w=omega/c`. Actor minibatch loss is
`-mean_batch(w*log pi(a|s))`; never renormalize advantage weights within a
minibatch. This global scalar rescale is a declared optimization choice.
The cap applies before rescaling; report c, raw cap fraction, fallback fraction,
weight quantiles and `(sum w)^2/sum(w^2)`. Require this effective weight count
to be at least 10% of FIT transitions before actor continuation.

Fixed seed 20260930; actor/critic seed derivations and batch permutations are
persisted. Adam, learning rate 3e-4, betas(.9,.999), epsilon1e-8, no weight
decay, batch256, global gradient-norm cap10. No dropout, target-policy
sampling, early stopping or calibration-selected epochs. Train each critic
fold for 20 complete epochs; no scoring of unobserved candidate Q values.

Train one uniform full-categorical BC actor for 10 FIT epochs. Copy its exact
parameters and Adam state into two actors. Continue each for 10 further epochs
on identical transitions and identical batch order: weights w for IQL, weights
one for the ordinary-BC control. Preserve the frozen 10-epoch warmstart as
another reported reference. Additional critic compute is reported separately;
the claim is matched actor updates, not equal total method compute.

## Evaluation and stop rule

Fit uses only allowed FIT outcomes/actions. Calibration contexts may receive
label-free coverage/drift diagnostics but never change this recipe. Do not
read development terminal targets for this trial. Freeze every actor before
opening its preference prediction metrics.

The primary comparison uses the existing qualified 16.19 revision artifact,
the same common pair set, existing focal-player weights and 2,000 paired
whole-game resamples. Sum actor full-action probabilities through the exact
`full_to_group` map, or use grouped logsumexp. Compare **total main-group log
odds B minus A** with frozen M2 main odds. Do not compare a single actor ward-
free variant against M2's marginal. An optional full w0 singleton comparison
must instead use the separately exported M2-times-nu joint reference and be
named as a different diagnostic.

Report IQL, 20-epoch ordinary BC, frozen warmstart BC, frozen M2, and the fixed
price/frequency rules. The useful cross-signal gate requires at least +5 pp
weighted pair accuracy over both 20-epoch BC and M2, with each paired interval
above zero; IQL must also improve on frozen warmstart BC and the cheap controls
with positive paired lower bounds. All contrasts remain visible. This is an
already-exposed development challenge, not fresh confirmation. It can establish
agreement with audited revisions, not that IQL increases wins.

Report action/group coverage, feature OOVs, whole-chain exclusions, patch and
price-direction slices where at least 30 games exist, full versus main
prediction differences, critic diagnostics, actor entropy and policy drift.
Critic values/Bellman fit and the optimized actor's own estimated value are
not independent evidence of policy improvement. Native affordability/display
validation and a separate human/prospective study remain necessary before
serving or claiming practical win benefit. No promotion is part of this run.

## Resource and integrity bound

Use one idle CUDA device and one CPU thread, with host memory-mapped feature
arrays and ragged inventory bags. Do not materialize all JSON trajectories as
Python dictionaries or store dense per-transition reference/policy matrices.
Store next-state references instead of duplicating full next-state tensors.
No GPU work during an actual League game; the durable launcher owns this gate.

One fixed training run has a 90-minute wall deadline after source/cache
preparation. Use a warm synthetic throughput/resource smoke before launch;
if the fixed recipe cannot fit the budget, stop and report that limitation,
not a shortened candidate versus a completed control. Abort on nonfinite
loss/gradient/weight, reservation or digest drift, missing complete epochs,
insufficient coverage, resource pressure or mismatched warmstart/batch order.
Bind source, catalogue, statics, fold map, encoder schema, initializations,
optimizer states, OOF weight file and final checkpoints by digest.
