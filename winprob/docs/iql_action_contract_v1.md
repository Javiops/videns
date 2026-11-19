# IQL action contract v1 — complete retained baskets and an explicit M2 lift

2026-09-30. Recommendation: use **complete retained purchase multisets as
IQL actions**, including control-ward counts, and keep M2's main groups as a
separate projection for comparison. Train ordinary categorical IQL/AWR and
an equal-budget uniform-BC control. Use the frozen full-basket lift below as
an evaluation reference; no dense M2 distribution over every training
transition is necessary. This defines a testable sequential learning
experiment without new collection or an unspecified ward intervention.

## What the source actually represents

`build_tracker/app/mixture_policy.py::probabilities` returns 9,333 main-group
masses. A group is an entire basket with items 2055 and 772043 removed; it
is not one largest item or a next completed legendary. Group zero combines
no-main-purchase cases. `options` walks the highest 128 group scores, checks
`purchase_sequence` on the stripped group at the original `gold_est`, and
returns up to three distinct legal main options. It does not append a ward.

The older ward-free protocol describes a future optional control-ward side
suggestion, but explicitly leaves its probability model out of that
experiment. Free trinket use is the player's decision. Neither a documented
intention to add an optional ward nor an observed factual ward is an
implemented completion policy inside the frozen M2 adapter.

The new trajectory producer preserves `label_counts` with multiplicities
and wards. These are **purchase-backed retained net gains**, not the complete
primitive transaction sequence. Intermediate components, canceled purchases,
sales and grants cannot silently acquire the same intervention meaning as a
retained target. Initial IQL admits only complete observed modeled decision
chains; boundary/projection gaps invalidate the whole player-game.

Define a complete action as either:

- `purchase`: a canonical sorted item/count multiset, retaining wards;
- `observed_no_buy_death`: the explicit source SAVE label, stored as an empty
  catalogue action with a separate semantic tag, never as physical item
  999999 and never as an invented label for unobserved manual no-buys.

For a purchase (b), write (b=g\uplus w\), where (g) is its non-ward
multiset and (w) is its ward multiset. Preserve the tag for empty purchases.
This decomposition is lossless on the declared source action vocabulary.

## The existing package already supports a concrete full-action reference

Read-only catalogue audit of `mixture-v4-m2-v1/catalogues.json`, SHA256
`56ab14f4a478489a88c6c2c33fc479b0ce7e3137676cb1fcbd292d953c93fa5f`:

- 13,166 distinct complete actions: SAVE plus 13,165 `baskets_A` entries.
- 9,333 distinct main groups; every group has at least one full-A variant.
- 6,282 groups have one variant, 2,269 have two, and 782 have three.
- 8,742 groups have a zero-ward variant; 591 multi-item groups do not.
  Every singleton main group has a zero-ward full action.
- Group zero has three distinct complete actions: SAVE at full-action ID 0,
  one 2055 at ID 2, and two 2055 at ID 20. No 772043 action occurs in this
  catalogue. Do not silently convert that item to 2055.

`basket_econ_finetune.TunedNet.corrected_scores` returns scores over **all**
13,166 actions. The frozen top-128 domain determines where an economic
correction is added; it is not a hard support mask. Let those logits be

\[
\ell_A(s,b),\quad q_A(b\mid s)=\operatorname{softmax}_b\ell_A(s,b),\quad
q_A^G(g\mid s)=\sum_{b:G(b)=g}q_A(b\mid s).
\]

The source computes

\[
p_{M2}(g\mid s)=\tfrac12 q_A^G(g\mid s)+\tfrac12 q_B(g\mid s).
\]

Recover a frozen completion reference, with stable within-group logsumexp:

\[
\nu_A(b\mid s,g)=
\frac{q_A(b\mid s)}{q_A^G(g\mid s)},\qquad G(b)=g,
\]

and define the explicitly named joint prior

\[
\boxed{p_0(b\mid s)=p_{M2}(G(b)\mid s)\,\nu_A(b\mid s,G(b)).}
\]

It is normalized and has exactly the M2 main marginal in real arithmetic:

\[
\sum_{b:G(b)=g}p_0(b\mid s)=p_{M2}(g\mid s).
\]

Equivalently (p_0=\frac12q_A+\frac12(q_B\otimes\nu_A)\). This uses
existing frozen components and adds no learned ward head. Call it
**M2⊗νA**, not the native M2 full policy: M2 never supplied a complete
ward decision. A source-owned exporter can capture the existing method's
full-A logits while returning its original values unchanged, derive this
lift, and verify the grouped marginal against untouched native M2 outputs.
No frozen source file needs modification.

Finite-precision evidence must distinguish mathematical equality from
bitwise equality: retain original float32 M2 group masses as authoritative,
report the maximum aggregation/normalization residual for the lifted full
array, and use the exact original group array for a zero-residual native
decoder check. Do not silently renormalize the frozen reference.

**Marginal equality does not imply decoder equality.** In general,

\[
G(\arg\max_b p_0(b\mid s))\ne\arg\max_g p_{M2}(g\mid s).
\]

A group whose mass is divided across ward variants can lose the full-action
argmax. Likewise a legal main basket does not certify a legal basket after
adding wards. The 591 missing zero-ward variants make silent completion
fallback particularly inappropriate. Actual full-action extraction must
apply `purchase_sequence` to the **whole** multiset. Main-advice parity and
complete-policy legality require separate, explicitly named checks.

## Concrete first IQL configuration

Export a versioned static full catalogue in the frozen A order, plus its
full-action→main-group map and source hashes. A logged action matches by
its complete canonical multiset/tag, not by its stripped group. Mark an
out-of-catalogue action explicitly. For the first complete-trajectory
experiment, exclude the entire affected player-game rather than jump over
that action or silently change it. Measure this coverage before training.

Use a full-action critic (Q(s,b)\), value (V(s)\), and categorical actor
with 13,166 outputs. Vanilla IQL's value expectile is over **observed full
actions**, its Bellman regression uses the next actual state, and its actor
loss is weighted full-action behavioral cloning:

\[
L_V=\mathbb E_D[|\tau-\mathbf1(u<0)|u^2],\quad u=\bar Q(s,b)-V(s),
\]
\[
L_Q=\mathbb E_D[(r+(1-d)V(s')-Q(s,b))^2],\quad
L_\pi=-\mathbb E_D[\min(c,e^{A(s,b)/T})\log\pi_\theta(b\mid o)].
\]

Here γ=1, (A=\bar Q-V\), and terminal outcome is paid exactly once.
Fix temperature, clipping, expectile and compute budget before results;
compare against the same actor with uniform cloning weights. The actor
learns both main and within-group choices; **νA is a reference prior, not
a fixed restriction on its future ward behavior**. Otherwise the ordinary
full-action IQL Bellman target optimizes a policy class the actor cannot use.

Whole-game cross-fitting and clustering remain necessary. A WP potential
may be a separately controlled shaping ablation with terminal potential
zero; its telescoping reward does not repair action or confounding issues.
No logging propensity is required by these IQL optimization losses. That
does not make observational value estimates causal or solve hidden intent,
partial observability, interaction between players, or missing decision
opportunities.

Dense teacher storage on every trajectory is unnecessary for this first
recipe. A float32 400,000×13,166 matrix would be about 21.1 GB decimal.
The matched uniform-BC and advantage-weighted actors need the static action
map, their input states and logged labels. Export full M2⊗νA distributions
only for bounded reference/challenge contexts initially. A later KL-anchored
actor would need its own compute/storage declaration.

## Comparison on the fixed revision challenge

Keep all current source roles and the 16.19 development population fixed.
Both receipt candidates have zero wards and are strict singleton retained
actions, so their full catalogue IDs exist whenever their singleton M2
groups exist. Report unsupported candidates rather than inventing support.

Report both projections on the identical usable pairs, preserving all
coverage and random-focal weights:

1. **Main revision ranking:** aggregate the learned full actor over each
   main group and compare its B-versus-A ranking against exact native M2
   main masses. This keeps the established strong M2 control intact.
2. **Complete zero-ward ranking:** compare actor probabilities for the exact
   singleton B and A against M2⊗νA and the matched uniform-BC actor. This is
   a separately named full-action comparison, not a substitute for item 1.

Do not report a win merely because the lifted reference is weaker than
native M2 on the same pairs. Price/frequency controls and no-preference
roundtrips remain separate diagnostic controls from the revision protocol.
These tests measure agreement with observed revisions, not improved wins.

## Why the simpler alternatives are not the primary contract

Holding each row's **factual** wards fixed when replacing its main action is
not a policy defined from pre-action information. It conditions completion
on the observed choice, can create an unaffordable joint basket, and does
not specify what to do at a new state. Deleting ward-containing rows also
breaks trajectories; selecting entire ward-free games conditions on future
behavior and targets a narrower population rather than the general task.

A partial-action estimand is viable if an explicit stationary completion
kernel ν is part of the intervention: (\pi(g,w\mid s)=\pi_G(g\mid s)\nu(w\mid s,g)\).
Then the reduced transition kernel integrates over ν; with common ν, joint
likelihood ratios and KL terms reduce to their group counterparts. However,
ordinary group-level Bellman regression on logged transitions corresponds
to the historical completion mechanism, not an arbitrary new νA. Using a
different completion requires explicit correction or a full-action critic.
The full-action design above avoids assuming that the source's ward
completion mechanism already equals the frozen lift.

Reference contexts retain legacy post-match inferred roles solely for exact
M2 compatibility. New actor features must exclude those roles and privileged
total-gold fields; their presence in an exported reference projection is not
permission to consume them. A reference-anchored implementation requiring
those fields at inference would not yet be a certified POV actor. Full net
actions also remain a declared source abstraction, not a proven model of
every shop interaction. These limits qualify the theory without changing
the preserved data or weakening the comparison baseline.
