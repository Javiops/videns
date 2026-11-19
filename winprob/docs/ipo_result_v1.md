# IPO residual v1: independently reproduced null result

The fixed small IPO adapter did not improve the audited revision challenge.
On **158 development pairs from 144 games, all patch 16.19**, weighted pair
accuracy was **53.5256%** for IPO and winner-only imitation, versus
**54.4872%** for frozen M2. IPO minus M2 was **-0.9615 percentage points
[-3.9396, +1.9231]**, using the declared paired whole-game bootstrap.
The predeclared +5 pp useful-improvement gate is false. This result tests
one feature map, regularization, adapter class and selected revision
population; it does not dismiss preference learning or IPO in general.

Primary evidence is
[report.json](../data/runs/ipo-revisions-20260930-v1/experiment/report.json),
[development_pair_predictions.npz](../data/runs/ipo-revisions-20260930-v1/experiment/development_pair_predictions.npz)
and [residual_models.npz](../data/runs/ipo-revisions-20260930-v1/experiment/residual_models.npz).
All accuracy/contrast measurements in this document use that same development
population and its recorded random-focal-player pair weights unless a slice
is explicitly identified.

## Independent checks

No raw match/timeline or sealed payload was opened for this review. Streaming
SHA256 checks reproduced **all 25 bound input/code/protocol hashes**, the
model digest and the development-prediction digest. The current metadata-only
reservation verifier passed. Every reference row's source-projection digest,
admission role and patch were checked, and fit/calibration/development game
sets are disjoint.

Source reference qualification produced **863 fit pairs / 775 games**,
**200 calibration pairs / 178 games**, and **158 development pairs / 144
games**, for 1,221 aligned rows. The 144 challenge IDs have **zero overlap**
with M2's **6,354 frozen training IDs**, checked directly against the source
v4 split metadata. Its sorted training-ID digest exactly equals the package
manifest's `aa306612342a5b5966b395dba381c939c044bb55dabbad9d9e3c7b2afb5a6321`.
This verifies the needed M2-training exclusion; it does not turn exploratory
development into a universally unexposed or confirmatory dataset.

An independent calculation reconstructed the 22-coordinate public feature
vector from the source-qualified context and static champion tags, applied
the saved fit-only standardization, and recomputed both bilinear residuals
from the saved coefficient matrices. Adding the exported full-catalogue M2
pair log odds reproduced every stored total log-odds prediction within 2e-14.
Pair identity, game, A/B actions and weight matched the canonical source
context on every development row.

Without importing the evaluator, the audit recomputed weighted accuracy by
aggregating each whole game's numerator and denominator. It reproduced the
2,000-sample paired multinomial game bootstrap, seed 20260929, and both main
confidence intervals within numerical tolerance. No pair-IID interval was
substituted for the declared game-cluster analysis.

## Why the IPO-versus-imitation interval is exactly zero

The two fitted coefficient matrices are **different**: their L2 distance is
**2.88537**. Their development log odds also differ, by as much as **0.89633**.
Nevertheless their B-versus-A classifications agree on **all 158 pairs**, with
no ties. Their weighted accuracy contribution is therefore identical for
every pair and every resampled game. The observed contrast and every bootstrap
replicate equal zero, correctly giving **0 pp [0,0]** for this discrete metric.
That is not a claim of identical probability policies or population-level
equivalence.

Each fitted arm changed the same **six** pair classifications relative to
M2: **two improvements and four regressions**. The parameters, predictions
and comparisons are not accidental duplicates. The reported full-policy
diagnostics also differ: IPO's weighted mean KL to M2 is **0.01210** versus
**0.03812** for imitation, and their canonical full-group top-one change
fractions are **5.1282%** and **9.2949%**. These are canonical probability
diagnostics, not displayed-policy legality or win-rate measurements.

Both optimizers completed their fixed recipe. IPO's 506-parameter design has
rank **409**, with maximum stationarity residual **1.30e-17**. Winner-only
full-9,333-group cross-entropy converged in **13 iterations**, maximum gradient
**3.64e-8**. These training diagnostics rule out an unfinished control; their
losses are different objectives and are not compared as performance metrics.

## Controls and interpretation

The more-expensive-item rule scored **57.2115%**, the cheaper-item rule
**42.7885%**, and the fit-derived retained-item-frequency control **53.3654%**.
IPO's contrasts against these controls did not pass the additional usefulness
guard. The frequency heuristic used the latest available fit patch for 157
challenge rows and global fit counts for one; no patch-16.19 challenge labels
were fitted into that baseline. It is a revision-derived frequency control,
not an ordinary-shop-frequency estimate.

The declared price slices remain descriptive: B cheaper has **55 pairs / 53
games**, B dearer **78 / 76**, and equal price **25 / 25**. The last slice is
correctly marked unmeasured under the 30-game rule. Do not select a favourable
slice or a different temperature from this result and call the next pass a
confirmation.

There is no established preference benefit, coaching benefit or causal win
improvement here. The source/reference bridge, zero-residual identity checks
and usable revision challenge are reusable infrastructure. A later flexible
critic/actor experiment is a separate hypothesis with its own full-action,
training and evaluation contract; this result does not justify changing the
served model or opening reserved confirmation games.

## Artifact binding

Original report SHA256:
`098891a709e9a7e70f4eb0f6c1599919517c6103b2a199aa537780a56b6772e7`.
Development predictions SHA256:
`09a274bc173564dc2f88c267b36f0439685bfe8680bfd2beb78b40ba307cc5c0`.
Fitted models SHA256:
`bfafe22ce564beb86d90db682983656f50b4f11daf213bb6eb24cd4d0bdcd50e`.
The independent review report is this file; original experiment files were
not changed.
