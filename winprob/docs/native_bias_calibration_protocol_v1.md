# Native M2 class-bias calibration: fixed prospective comparison

2026-10-04. Declared before any real calibration fit, class-residual inspection
or new calibrated purchase score. The complete-student family remains closed.
This study preserves M2's full probability vector and changes only one global
log-probability offset per counted main basket. No new neural learner is fitted.

## Hypothesis, references and exclusions

Hypothesis: stable class-wide probability biases in native M2 may affect its
displayed purchase ranking. A correction estimated on separate FIT games could
reduce those biases. Whether M2 has such biases, and whether correcting them
improves first-purchase accuracy, is unmeasured. No expected gain is asserted.

[Kull et al., Dirichlet calibration](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)
gives the map softmax(W log p + b). This proposal is its restricted W=I
subfamily, with a fixed Gaussian penalty. It is not the full method, a new
theorem about League, or a claim of full multiclass calibration. A scalar
temperature preserves rankings; class offsets can change them.

[Lipton et al., BBSE](https://proceedings.mlr.press/v80/lipton18a.html) and
[Alexandari et al., maximum-likelihood label-shift adaptation](https://proceedings.mlr.press/v119/alexandari20a.html)
require assumptions about stable class-conditional distributions; these have
not been verified across our patches. The proposed fit is supervised
recalibration, not an unsupervised estimate of a new patch's class priors.
It reads neither sealed features nor labels for adaptation.

Source history contains a proposed frequency offset and cardinality diagnostics,
but the inspected files do not establish this class-bias experiment as executed.
The September21 dossier withdrew its claim that aggregate/group calibration
established irreducibility. We do not revive that claim or a cardinality fix.
The previous route correction was restricted by shared route vectors; these
9333 offsets can distinguish baskets with identical route representations.
That extra flexibility is a hypothesis, not evidence of improvement.

## Mathematical contract

Let p_i be M2's normalized complete 9333-class vector. For frozen offsets b:

    q_i(a; b) = p_i(a) exp(b_a) / sum_c p_i(c) exp(b_c).

Each target y_i remains the actual retained purchase. Define n_a as the number
of TRAIN targets in class a. Fit the unique minimizer of

    F(b) = sum_i log(sum_a p_i(a) exp(b_a)) - sum_a n_a b_a
           + (1/2) ||b||^2.

This is summed observed-purchase NLL relative to fixed M2, plus a fixed unit
ridge penalty, equivalently a N(0,1) prior on each offset. It does not balance
or reweight rare target classes. The omitted baseline NLL is constant; verify
positive p_i(y_i) before aggregation. No clipping or invented target support.

The gradient is sum_i q_i - n + b. The Hessian is I plus the sum of categorical
covariance matrices, hence F is 1-strongly convex. Its minimizer is unique and
has sum(b)=0. The numerical residual gives the bounds

    F(b)-F(b*) <= ||gradient F(b)||^2 / 2,
    ||b-b*|| <= ||gradient F(b)||.

Report these as analytic bounds evaluated with float64 gradients, not rigorous
interval-arithmetic certificates. They certify optimization, not out-of-sample
accuracy. Penalization does not enforce exact aggregate calibration.

For this objective, permuting TRAIN labels while preserving their counts leaves
the fitted offsets unchanged. Such a permutation is NOT a negative control.
The declared diagnostic instead permutes the learned class offsets within
public main-basket total-count strata (0 through6), seed20261004, using ascending
class indices and sorted strata. It preserves coefficient magnitudes and basket
size strata while destroying class identity. It is never an eligible candidate.

## Inputs, folds and fixed fit

Use the existing 56288 selected states from3518 admitted FIT16.17 games and
their fixed two whole-game folds. Each game contributes16 selected states;
summed row loss therefore gives each TRAIN game equal total weight. Original
2900 focals are scored;618 unavailable original focals stay disclosed.

Use the exact native full-prior artifact:
../build_tracker/data/exports/winprob-route-prior/route-prior-20261003-v1/manifest.json,
SHA35113430e6997ec1ef993da5d9dc3f9325941138e21db24fab300ea3b5f470ac;
full_probability.npy SHAa2c4b14975027a7aafafe5a4a879322c2ea74f2116f9fd021d3128be4641b678.
Selection SHA97b7842e8811b3b4ed3baf4cc2918185c98061d8997589b52ba11459213b4191.
Catalogue is unchanged; preserve all item quantities and empty-main semantics.
No source runtime import, raw timeline scan, new collection,859 training,
1911 confirmation, or source/global sealed payload access.

Validate admission/reservations and all bindings before payload access. Freeze
inputs, class-control permutation and code before real labels. For scored fold f,
only opposite-fold targets form n, and only opposite-fold priors enter F.
This is a per-model restriction; the two cross-fits collectively use both FIT
folds. Save exact training identities/counts and enforce fold permissions.
No target, fitted count vector or label lookup belongs in prediction.

Two actual fits, one per fold; initialize b=0. Fixed ridge1, L-BFGS-B,
maxiter500, at most2000 optimizer evaluations plus one final gradient recheck,
maxls20, maxcor20, gtol1e-6, ftol0. Require solver success and max absolute final
gradient below1e-5. No retry, ridge/temperature search, feature addition or
alternate optimizer selected from scores. Preserve diagnostics and stop if
either fit fails; no held-out prediction until both maps are frozen.

At inference export original float32 M2 and two full float64 maps: calibrated
and class-permuted. Stable normalization retains exact zero support. The
all-zero bias case returns the exact float64-normalized prior and requests
the original float32 native path, preventing a spurious baseline tie change.

## Resources and verification before outcomes

CPU1, host2.5GiB startup/.75GiB live, at least3072MiB idle CUDA, no League,
capture, reconstruction, benchmark or competing protected task. Hidden worker
at most3hours. Bounded host readers; only the current TRAIN fold's float64
log-priors reside on GPU, with bounded256-row objective scratch space. No full
host copy of the teacher matrix and no full 9333-square Hessian.

Before labels, compare CPU/GPU values and gradients on a bounded real-prior
sample with dummy group0 counts at zero and a fixed synthetic nonzero bias;
tolerance2e-10 for values and2e-10 max absolute gradient error. Then time two
warmups/five full TRAIN objective evaluations per fold, still with dummy counts.
Require at least1GiB conservative device headroom and
1.5 * max_timed_call * 4002 + 900seconds <=80% remaining worker time. If this
fails, stop without opening real targets or silently shrinking the cohort.

Save b, class permutation, counts, fold identities, code/input hashes, optimizer
receipts, gradient/gap bounds and all predictions. Audit TRAIN count support,
fold restrictions, objective derivatives/residuals and exact prediction replay
before current-target scoring. A synthetic counterexample already shows that
lower NLL can reduce argmax accuracy from3/5 to0/5; NLL is not a success proxy.

## Native comparison and decision

Decode all three arms through unchanged native MixturePolicy.options: CUDA
top128, original wallet, recursive legality, up to3 distinct main actions and
original SAVE fallback. Require exact M2 replay and8700 forbidden-field
mutation checks. Audit every returned option again in a separate native16.17
process. No post-audit filtering or replacement.

The sole candidate is calibrated versus M2. Main first exactness must gain at
least3 percentage points with positive paired95% lower bound; nonempty-main
95% lower bound must also be positive. Require no mean top3 decrease, no mean
catalogue NLL increase, zero native17/18 legality failures across all arms and
all audits passing. Report candidate-control contrast and whether its95% first
lower bound is positive, separately from the M2 benefit gate. Never select the
control. Bootstrap2000 whole games, seed20261004, paired on the original2900
one-per-game focals; M2 must reproduce1483first/1959top3.

Report full/typed exactness, corrected/broken decisions, change counts, all
intervals, numerical optimization bounds and costs. A final independent metric
audit recomputes all counts/intervals/gates. If the candidate fails, close the
fixed family without retuning or a forward trial. Internal success would still
require a separate prospective forward evaluation and practical integration.
This exposed cohort, conditional intervals, prior research multiplicity and
the native reference's legacy-role limitation preclude a breakthrough claim
from calibration quality or this internal comparison alone.
