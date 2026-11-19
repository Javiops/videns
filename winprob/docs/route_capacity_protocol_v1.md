# Route capacity and static audit, 2026-10-03

Public-only follow-up to route-static-census-20261003-v1. No prediction fit,
game reads, source forward pass, target/future-label reads or policy selection.
Bind that report SHA08b426d97260521b52c9271c235f705a60c3b899ac9d449fddf67a1a8c951def
and verify every public-input, matrix and consumed-code digest it records.

Independently recompute all matrix count/equivalence summaries. Check public
recipe compatibility with a different algorithm: unfold recipe-node occurrences,
then maximize assigned purchased copies subject to an antichain constraint.
A chosen ancestor and descendant cannot both match; duplicate node occurrences
are separate capacities. Check512pairs from seed20261003 plus the first128
baskets' first maximizing forest and128 zero-score pairs, without claiming this
sample independently verifies every one of the2,407,914matrix entries.

The bridge is exactly a convex mixture of conditional experts:

    E(a,z)=p0(a)R(z|a)/m0(z),  p1=E alpha.

Ignore unsupported routes. Therefore the min/max of p1(a)-p1(b) are the
min/max over columns of E(a,z)-E(b,z). A pairwise certificate is not sufficient
for simultaneously outranking all rivals. Maximum KL(p1||p0) and total
variation over this family are attained at a column, by convexity. This
extends the original conservation/KL tests with sharp capacity statements.

Reproduce the census's SYNTHETIC softmax(fraction) routing and UNIFORM prior.
Report its expert-column KL/TV maximum, largest probability amplification,
and largest routing-row ratio per column. These are hypothetical geometry,
not M2 capacity, action frequencies, achievable accuracy or expected gain.
This still does not fix a routing kernel for real training.

An implemented optional residual parameterization is

    alpha(z|x) = m0(z|x) exp(u_theta(x,z)) / sum_k m0(k|x) exp(u_theta(x,k)).

Zero logits preserve M2 exactly. A current observed purchase y supplies the
relative main-basket loss

    log sum_z m0_z exp(u_z) - log sum_z R(y,z) exp(u_z).

Add -log p0(y) to report actual NLL. For linear u=x theta, the analytic gradient
is x^T[alpha-beta_y], where beta_y is normalized R(y,z)exp(u_z), plus the
declared regularizer's gradient. This loss is generally NONCONVEX; the prior
net-recipe convexity/uniqueness guarantees do not carry over. Tests compare
finite differences and actual bridge probabilities, including zero support.

This parameterization optimizes observed purchases. It does not predict a
future-item label or establish a latent variable as the player's private plan.
Before any real experiment, fix the kernel, causal input features, fitting
recipe and control; require native full-distribution marginals. Existing
three-plus-OTHER exports alone are insufficient to reconstruct those marginals.
Do not quietly substitute the three-option distribution or an oracle goal.

CPU1, no game/competing protected task, startup RAM2.5GiB/live0.75GiB;
10minute hidden worker limit. Existing consumed files stay immutable; original
index and all859development/1911confirmation/global/source seals untouched.
