# Purchase-supervised full-catalogue route correction, v1

Declared2026-10-03 while the input-only native-prior export is running, before
any real route marginal, target-dependent route fit or route score is computed.
Primary question: does sharing a correction through public recipe structure
improve the actual displayed first purchase over frozen native M2?

## Fixed population and provenance

Use the already frozen3518FIT16.17games/56288states (16per game), same whole-game
folds, and original2900distinct-game focal queries (1449/1451);618games have no
eligible original focal. Training folds contain28112/28176states respectively.
M2's6354TRAIN16.18games remain disjoint. This is an exposed retrospective
internal screen with native16.18 M2, not fresh forward confirmation or live
evidence. All859ordinary-development and1911confirmation games stay closed.
No new population selection, future targets, outcomes, raw timelines or database.

Reuse the old recipe selection/context/native releases only as immutable input
artifacts, not as a refitted net-recipe model. The complete native prior must
come from route-prior-20261003-v1, schema route-prior-native-m2-v1, with all56288
original probability hashes and saved rows verified exactly. Require completed
manifest, bind its digest and every consumed file/code/protocol before labels.
Revalidate admission and all source/global reservations before payloads.

## Fixed structural kernels and control

Use the audited public fraction_units matrix[9333,258],
SHA d5157ae14c1b7267306686dc3bdcca32fd2f40b1442375cd4736db6d402cf572.
Real R(a,z)=softmax_z(fraction_units[a,z]/60), with temperature exactly1.
No kernel sweep, new forests or side-purchase masking. This positive kernel
allows every route attribution and does not delete main-action prior support.
Its old106major-root ontology excludes boots; equivalent rows preserve odds.

Matched control: permute complete matrix ROWS among main baskets of equal
total purchased-copy count, preserving the empty group and each size bin's
multiset of route vectors. NumPy default_rng(20261003), bins0..6 ascending,
ascending group IDs within each bin, exactly one permutation per bin. Both
arms retain the same26input features,258routes,6708parameters, optimizer and
regularizer. Neither permutation nor kernels access targets. Freeze the actual
permutation and both kernel hashes before fitting. Route-column permutation
would merely rename the latent variable and is explicitly NOT a control.

## Model and purchase likelihood

Context x is EXACTLY the existing26role-free context_features: intercept,
time/level/reconstructed wallet,10summed owned-item attributes,6own champion
tags and6enemy-tag fractions. No KDA/kill graph, future completion, final roles,
damage, inferred intent or additional history. Standardize only the training
fold with equal-game weights; leave intercept mean0/scale1. No clipping.

For each arm, m0(x)=p0(x)R over the FULL9333-class normalized prior. Set

    u=x theta; alpha_z=m0_z exp(u_z)/sum_k m0_k exp(u_k)
    p_theta(a)=p0(a) sum_z R(a,z)exp(u_z)/sum_z m0_z exp(u_z).

Zero theta preserves normalized p0 exactly. A common route-logit shift changes
nothing. Training minimizes equal-game mean relative main-basket NLL

    logsumexp(log m0+u) - logsumexp(log R(y,:)+u)
      + (.01/258)*sum(theta^2).

The actual label y is the observed counted main purchase. The regularizer is
.01 times the sum across input coordinates of mean squared route coefficients,
so its scale does not grow merely with the number of routes. Add -log p0(y)
to report actual full-catalogue NLL. Keep every selected training state. If an
observed target has zero native full prior, stop before fitting rather than
drop it, inject epsilon support or publish an incomplete comparison.

Four fits (two arms times two folds), float64CPU1, zero initialization,
L-BFGS-B maxiter500,gtol1e-7,ftol0,maxls50; no restarts or checkpoint selection.
Mini-batches512 only accumulate the EXACT full-batch objective/gradient.
Require optimizer success and max absolute gradient<1e-6 for all four models
before scoring; otherwise numerical failure is recorded before held-out scores.
This mixture loss is generally NONCONVEX. A gradient receipt is not a claim
of a global or unique optimum. Freeze all models before predictions.

## Prediction, native decoding and evaluation

Predict all9333classes on2900focals using only predecision inputs. Export
original float32 M2 priors, both float64 corrected distributions, neutral flags,
query identities, kernel/model/protocol hashes, and a prediction lock before
reading any decoded-prediction score. All labels stay outside prediction APIs.

Decode each distribution through unchanged native MixturePolicy.options:
top128, recursive purchase_sequence, same wallet, at most3distinct options,
same SAVE fallback. The source decoder uses CUDA topk as in the original
native-prior release; this is a separate source-owned stage, not GPU fitting.
M2 must reproduce every old option exactly. If a route row
is mathematically neutral, the decoder uses its ORIGINAL float32 M2 row to
preserve native tie behavior; this is declared, not a tuned fallback. Native
zero/forbidden-field mutation checks must pass. Source runtime remains in a
source-owned process, with only digested artifacts crossing project boundaries.
Audit all returned main actions in native16.18 and actual-cohort16.17 worlds;
report both including baseline, with no filtering/replacement after the audit.

Primary: displayed MAIN first exactness, real-M2. Secondary: displayed top3,
full-purchase first, typed-main first, nonempty-main first, corrected/broken,
changed option sets, full-catalogue NLL and incremental costs. Wards and true
post-death no-buy remain distinct; do not equate a ward-only label with SAVE.
Paired2000whole-game bootstrap, seed20261003, same game draws for all contrasts.

Useful internal gate requires real-M2 first gain>=3percentage points, paired
95%lower>0, real-control first95%lower>0, nonempty-main real-M295%lower>0,
no decrease in mean displayed top3, no increase in mean catalogue NLL, zero
native17/native18 legality failures for all arms, and all integrity audits pass.
Publish all arms when primary fails. Do not promote a successful control or
retune kernels/ridge/features/windows after scores. A passing internal screen
still requires a separately frozen forward trial and practical integration;
it does not complete the broader goal.

## Resources and mathematical motivation

Do not run this worker while the native prior release or another protected job
is live. Fitting: hidden task, CPU1, no GPU, startup2.5GiB RAM/live.75GiB, streaming/mmap
full priors and route marginals; no complete2.1GBprior allocation. Worker180min.
Before labels, time2warmup+5sustained objective calls per arm/fold with a dummy
group0target only. Require1.5*max_mean*500*8*4+900seconds fit80% of remaining
time. This is a declared conservative projection, not an upper bound on line
search cost; the worker hard deadline is authoritative. No partial-score result.
Native decoding: a separate hidden task after fitting, CPU1/idle CUDA with
3072MiBfree, same host RAM/game/workload guards. Native16.17 legality runs in
a clean CPU process afterward so world activation cannot cross-contaminate.

The conditional-expert factorization follows the mixture-of-experts idea in
[Jacobs et al., Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf),
but experts here are fixed conditionals of M2 under a specified recipe kernel.
The likelihood and neutral bridge are derived/tested locally. Neither the paper
nor public compatibility proves a League gain, private intent or causal value.
Original index, frozen code, source data and served policies stay unchanged.
