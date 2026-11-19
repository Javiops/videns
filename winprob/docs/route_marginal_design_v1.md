# Route-marginal probability update: mathematical feasibility stage

2026-10-03, after closing the personal kill-graph comparison. No new purchase
fit, held-out score, source inference or game payload is used in this stage.
The intended endpoint remains improved prediction of the player's actual main
purchase, ultimately compared with unchanged native M2 and independently audited.

## What differs from the closed experiments

The prior future-completion experiment used CE(current basket)+CE(strictly later
completion) on a shared representation. Its inference path did not use a goal
distribution to produce current-basket probabilities. This proposal instead
defines an explicit joint distribution over complete baskets and construction
routes, then changes a route marginal and recomputes basket probabilities.
It operates over the complete fixed catalogue, not only the old three cards.
It cannot create actions outside that catalogue or outside prior support.

This is not a claim that a future completion is the player's private intention.
Intervening events can change what gets completed. The old strictly-later label
also excludes a completion in the CURRENT purchase; it cannot silently be
renamed the current active route. The existing definition separately records
including-current and strictly-later semantics. Any future data/model stage
must fix which observable target, missingness and multi-item treatment it uses.
Nothing in this mathematical stage authorizes a policy to observe those labels.

Hierarchical/latent-plan models offer motivation, e.g.
[Learning Latent Plans from Play](https://arxiv.org/abs/1903.01973). That work
uses goal-specified robotics tasks. Here the route belief must be inferred
from predecision observations or closed past purchases; supplying the observed
future completion at inference would be leakage. Neither that paper nor the
closed auxiliary-supervision result demonstrates a League gain for this proposal.

## Exact minimum-divergence bridge

Let p0(a|x) be a normalized frozen M2 distribution over all counted main baskets.
Let R(z|a,x) be a specified ROW-STOCHASTIC route attribution: every action's
route weights sum to1, including a background/unspecified route if needed.
Overlapping item recipes are therefore not counted as separate extra mass.
Define the prior joint J0(a,z|x)=p0(a|x)R(z|a,x), and

    m0(z|x) = sum_a J0(a,z|x).

For a candidate route belief alpha(z|x), supported wherever m0 is positive,

    J1(a,z|x) = J0(a,z|x) * alpha(z|x)/m0(z|x)
    p1(a|x)   = p0(a|x) * sum_z R(z|a,x)*alpha(z|x)/m0(z|x).

This is a single probability-kinematics update on an explicit joint model.
For any joint Q with the desired route marginal alpha,

    KL(Q || J0) = KL(alpha || m0)
               + sum_z alpha(z) KL(Q(a|z) || J0(a|z)).

Thus J1 is the unique minimum-KL supported joint with that marginal. It retains
the original action conditional within each route. It is normalized, preserves
zero action support, and alpha=m0 returns EXACT p0. By marginalization,
KL(p1||p0)<=KL(alpha||m0). These are mathematical properties, not accuracy gains.

Probability kinematics and its limitations are discussed by
[Grunwald and Halpern](https://www.cs.cornell.edu/home/halpern/papers/uai02.pdf).
We do not infer that an estimated alpha is valid new evidence merely because
the update is coherent. A route forecast derived from the same x adds no new
observed information conditional on x. Any gain would come from a useful
structured estimator and must be measured. No commutativity of repeated
updates is assumed, and repeated use of the same evidence is not permitted.

The bound also exposes a limitation: if two actions have identical R rows,
their probability ratio is unchanged for every possible alpha. A route model
cannot fix all component/count errors by knowing a route alone. The kernel
must report these equivalences and never label an oracle route as a policy.

## Count-preserving public recipe compatibility

As a feasibility object only, test exact matching of a recorded basket to a
frontier of a public item-recipe forest. A node can match one copy of itself,
or its child recipes can contribute matches, but the same branch cannot count
both the parent and its ingredients. Repeated recipe children and repeated
purchased copies remain distinct capacities. Missing IDs/cycles fail closed.

For a basket count cap b, recursively enumerate only submultisets <=b:

    F(node) = {empty} union {one node if present in b}
              union capped_multiset_sum(F(child) for each recipe child).

Combine goal roots in the same way. The maximum matched count provides a
bounded compatibility score, NOT proof of actual token lineage, intent, slot
legality, affordability or optimality. Side purchases and multiple simultaneous
construction routes mean a hard recipe-only action mask is not justified.
No real routing table R, route predictor alpha or new training objective is
selected solely by this mathematical prototype.

## Current work and next gates

Implement the batched normalized bridge without materializing an N x A x Z
tensor; optionally expose one-row joints for verification. Test normalization,
zero-support rejection, neutral identity, route marginal conservation, KL
decomposition/contraction, identical-row odds invariance and count-preserving
recipe matching against independent exhaustive frontiers. No GPU or real model
fit is needed. Any small public-static census must bind its catalogue/statics
and may not read purchase trajectories, future labels,851/859development or seals.

Before any real predictive experiment, separately fix and validate R, a causal
alpha estimator, target semantics and a matched control. Verify against source
history that it is not a renamed closed recipe. Then predeclare actual M2
first/top3/full endpoints, uncertainty, scope and success thresholds. No route-
prediction accuracy, lower training loss, oracle future-goal score or synthetic
proof alone can complete the objective. The original index, frozen source code,
all1911confirmation games and source/global seals remain untouched.
