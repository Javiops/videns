# Recipe-transition residual: mathematical prototype and feasibility

Declared2026-10-01 after the negative scoreboard result and before any new
purchase-data fitting. The previous turn made progress by completing and
auditing that result. The objective remains improved prediction of the player's
recorded retained purchase, ultimately against unchanged native M2.

## Verified motivation and scope

M2 already uses nonlinear item/count basket embeddings, a board transformer,
bilinear/deep state-basket interactions, and ten execution-economic features.
It is not an independent-item model and does not ignore recipes. Inspection of
`train_basket_ab.py`, `train_prefix.py` and `basket_econ_cache.py` establishes
this. Its economic block includes cost, remaining gold, consumed item count/
value and slot delta, but no vector of the attributes of the consumed items.
Our recent categorical probes used a different state representation and9333
separate output classes. Their failures cannot establish that every M2-relative
correction would fail. No claim that the native representation is deficient
has been measured.

Hypothesis: a small correction can share evidence across purchases through
their NET inventory-attribute changes, while retaining M2's existing ranking
as the prior. The same completed item adds different attributes when different
components are already owned. Explicitly subtracting consumed components may
make this easier to learn with fewer examples. It may also add nothing: M2's
inputs and nonlinear scorer can already learn some of these interactions.

This is not the previous IPO experiment: no undo-derived preferences or rejected
buys. It is not another independently trained actor blended with M2. It is not
a GNN, a new WP kernel, a simulator of combat or a causal action-value estimate.
The fixed memory, multiset and scoreboard recipes remain closed.

## Exact mathematical construction

Let D(x) be the original up-to-three distinct legal main baskets displayed by
M2, in their original order. Never insert the true purchase into D. Let p_j be
the frozen full-catalogue probability of candidate j, and
p_other=1-sum_{j in D}p_j. OTHER represents every purchase outside D, including
an observed basket absent from the fixed catalogue. It is a training category,
not a newly displayed action. Every selected training decision remains present.

For a source-native hypothetical execution T(I,b), define

    delta(I,b) = attributes(T(I,b)) - attributes(I)
               = attributes(b) - attributes(consumed components).

The equality is multiset conservation, not an estimated counterfactual outcome.
Use ten existing declared numeric Data Dragon attributes. Their sums exclude
passives, stacking rules, multiplicative interactions, health and damage. These
are item-descriptor deltas, not actual effective champion/combat stat changes.
Source-owned receipt export must call unchanged native purchase_sequence and
try_purchase; the own consumer verifies conservation without importing source
runtime. Hypothetical post-purchase inventories come only from the candidate,
not the player's observed post-purchase board or subsequent events.

Candidate descriptors f(x,b) have26coordinates: empty-main flag, execution cost,
remaining gold, purchased copies, consumed copies, item-count change, ten raw
purchased attributes and ten net attributes. Common economic/count scales are
3000gold and6items; numeric attribute scales are ACTION_STATS from ipo_residual.
Context h(x) has26coordinates: intercept,time/40min,level/18,wallet/3000, ten
owned-item attributes, six own-champion tags and six opponent tag proportions.
No inferred final role, outcome, target, recency or other-player gold enters
the residual. The frozen M2 reference retains its declared legacy-role inputs;
adding a residual does not retroactively certify those inputs as live-safe.

    r_theta(x,b) = h(x)^T theta f(x,b)
    q_theta(j|x) = p_j exp(r_theta(x,j)) / Z_theta(x)
    q_theta(OTHER|x) = p_other / Z_theta(x)
    Z_theta(x) = p_other + sum_{j in D}p_j exp(r_theta(x,j)).

The target is the actual main multiset's index in D, or OTHER. Exact coarsened
negative log likelihood uses every selected row, including shortlist misses.
Conditioning the training set on a target hit would answer a different question
and is prohibited. The likelihood is exact for this coarsened four-category
experiment, not a claim to model individual purchases inside OTHER.

For fixed inputs, minimize game-weighted mean negative log likelihood plus
lambda*||theta||_F^2. This is convex: the Hessian is a weighted sum of feature
covariances plus2*lambda*I. With lambda>0 it is strictly convex and the optimum
is unique. theta has676parameters. Zero theta returns the unmodified reference
masses and original displayed order. Prediction accepts no target argument.

Only reorder the original legal candidates by log(p_j)+r_theta. Never display
OTHER, drop an original option or replace a main basket. Top3 coverage and
legality are therefore invariant; any claimed improvement must be in first
exactness and must include all evaluation decisions, including OTHER misses.
The existing859-game development top3 minus first gap is137/859=15.95pp;
this is an oracle ceiling, not an expected or attainable improvement.

Related primary literature: [Wong and Farooq, ResLogit](https://arxiv.org/abs/1912.10058)
studies residual choice models; [Wang, Mo and Zhao, TB-ResNet](https://arxiv.org/abs/2010.11644)
combines structured and learned choice utilities. Neither validates this
particular construction, League purchases or a forecasted improvement. Our
linear offset likelihood and conservation descriptors are the concrete proposal.

## Stage zero, before any real fitting

Implement receipt-conservation checks, features, the normalized coarsened
likelihood, analytic gradient/Hessian-vector product, bounded convex solver and
stable legal-option reranking. Test zero-residual identity, repeated copies,
component subtraction, malformed receipts, OTHER-label contribution, normalized
mass, finite-difference gradients, positive curvature, target-free inference,
and equality to a full-catalogue exponential tilt after grouping its untouched
tail. Synthetic recovery must show the estimator can recover an identifiable
known correction. Synthetic success is not purchase improvement.

Measure source-native synthetic M2/recipe replay cost in a clean process before
exporting a substantial FIT sample. Existing ordinary CPU reference took552.93s
for859decisions including its full audit workload; do not multiply that figure
as if it were pure inference throughput. Use idle CUDA only after RAM/process
guards, oneCPUthread, and a durable task for any potentially long benchmark.
No data/model installation and no source freeze changes.

Only after numerical and resource gates pass, freeze a separate real-data
protocol: label-blind FIT selection, exact game partitions, native-world receipt
semantics, fixed lambda, matched raw-only/net/shuffled-net controls, reference
integrity and a paired native-M2 purchase-accuracy threshold. The same learned
correction must beat both M2 and its matched raw-only control. No hyperparameter
or shortlist sweep, no859-game fitting, no reopened failed recipe. All seals
and1911confirmation games stay closed. This design itself establishes no gain.
