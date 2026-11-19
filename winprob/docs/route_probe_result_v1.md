# Full-catalogue recipe routes: two extra hits, no useful M2 gain

2026-10-03. The declared route-mixture correction, unchanged native decoding,
two-world legality check and result audit completed. The useful-signal gate
fails. Close this fixed recipe without a forward trial, retuning or promotion.

## Same observed purchases and frozen baseline

Four cross-fitted models use56288selected states from3518originalFIT16.17games,
disjoint from M2 TRAIN. The primary population remains2900original focal
decisions, one per game;618games without an eligible original focal remain
disclosed. The native reference is frozen16.18 M2 on older16.17 states. This is
exposed retrospective development, not forward/live or confirmatory evidence.
No859ordinary-development or1911confirmation payload is opened in this study.

Displayed main-purchase first exactness, preserving item quantities:

- M2:1483/2900,51.1379%.
- Real recipe routes:1485/2900,51.2069%.
- Count-stratified permuted routes:1480/2900,51.0345%.

Real-M2 is+0.0690percentage points, paired95% interval[-0.3448,+0.4828]pp:
20decisions corrected,18broken. Real-control is+0.1724pp[-0.3103,+0.6897],
31corrected/26broken. Neither comparison is statistically resolved or reaches
the declared3pp primary requirement. On2736nonempty-main purchases, real-M2
is+0.0365pp[-0.3682,+0.4429]; that positive-lower-bound guard also fails.

Real displayed top3 covers1963/2900 (67.6897%) versus M2's1959 (67.5517%):
+0.1379pp[-0.1034,+0.4138], unresolved. Permuted covers1955. Real changes the
first recommendation on94queries and the option set on183; its near-null result
cannot be attributed to being restricted to M2's original three cards. This
does not establish that all possible candidate-expansion methods are useless.

Complete-purchase first is1107/2900 (38.1724%) versus1106M2 (38.1379%):
+0.0345pp[-0.3448,+0.4138]. Typed-main correctness stays1383for both real/M2.
Only declared ward IDs are removed for the main endpoint. Ward-only is not
credited as a complete SAVE prediction, and counts remain intact.

Full9333-class NLL is1.534942877real,1.536004128M2,1.536564856control.
Real-M2 is-0.001061251,95% interval[-0.004799384,+0.002724951]. Lower is better,
but the apparent reduction is unresolved. These full-catalogue likelihoods
must not be compared directly with the old three-plus-OTHER graph/net losses.

All intervals use2000paired whole-game resamples, seed20261003, conditional on
the fitted models. They omit training-set uncertainty and multiplicity
correction across the broader research program; the cohort is already exposed.
No secondary endpoint or slice rescues the failed primary.

## Formula, implementation and independent checks

The real row-stochastic kernel is softmax(public recipe-frontier matched
fraction), temperature1, over258forests and9333main actions. It uses the
audited public matrix;106major-item roots exclude boots. The control permutes
whole action rows within purchased-copy-count bins with the fixed seed.
Both arms use the same26current role-free context features and6708parameters.

For normalized M2 prior p0 and m0=p0R, fit route logits u=x theta and use

    alpha_z = m0_z exp(u_z) / sum_k m0_k exp(u_k)
    p_theta(a) = p0(a) sum_z R(a,z)exp(u_z) / sum_z m0_z exp(u_z).

This is an explicit conditional-expert mixture. The model optimizes the actual
observed main purchase, not a future goal or guessed private intention. Zero
correction preserves the prior; identical kernel rows preserve action odds.
No new support is invented. The mixture-of-experts motivation is discussed in
[Jacobs et al.](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf); its results
do not establish a League benefit for this implementation.

All selected targets retain positive native support. Training-only equal-game
standardization precedes target access. All four zero-initialized L-BFGS-B fits
complete under the fixed500iteration/gtol1e-7/ftol0/maxls50 recipe and ridge
.01/258. They take279/359/178/280iterations,1096total, with1166function evaluations.
Max gradients range6.773e-8 to9.774e-8. The loss is nonconvex; stationarity is
not a global-optimum certificate. All models freeze before predictions.

Full native probabilities were recovered exactly from the previous row hashes.
M2's original recommendations replay exactly under the unchangedCUDA native
decoder. Every query/arm passes forbidden-field mutation checks (8700queries).
All8488options per arm pass native16.18 and separately activated reconstructible
16.17 legality, with zero failures in both worlds. No option was removed or
replaced after either audit. Legal recommendations alone do not imply accuracy.

The receiving auditor verifies all56288route marginals with zero discrepancy,
independent training-only moments (max discrepancy1.709e-13), fitted IDs and
solver receipts, and real-solution directional derivatives (max error3.294e-14).
It reproduces all5800corrected full distributions and2900original focal priors
exactly. An alternative direct formula differs by at most1.333e-15. Every
metric/interval/gate rechecks, plus independent Python exactness counts. This
audit reuses bound native neural/decoder receipts; it does not rerun those
source stages. Its scope does not include live parity or causal purchase value.

## Resource repairs, cost and retained artifacts

The first own attempt hit the0.75GiB host RAM floor during marginal preparation;
its whole-prior mmap could retain pages. A new bounded seek/read implementation
then completed preparation. The second attempt stopped at its CPU cost gate,
before target access or fitting. Both attempts remain preserved; neither
supplied scores or selected a model. Startup/live RAM limits stayed unchanged.

The third attempt uses the same analytic objective/gradient in deterministic
CUDAfloat64. On16input-only dummy-target rows per fold/arm, GPU/CPU value error
is at most5.552e-17 and gradient error3.470e-18. Objective times decrease from
.58-.74s in CPU preflight to.0177-.0200s in GPU preflight. Those timings are
objective-call measurements, not measured recommendation latency or buy gains.
The independent result audit checks real optima again with the CPU objective.

Native prior retention:3243.985s,2,101,343,616payload bytes. Successful own fit
worker:211.950s, including22.762s across four optimizations. Native decode:
68.668s; native17 audit:12.007s; scoring:72.826s; result audit:153.085s.
Frozen prediction arrays occupy541,320,312bytes on disk. No new Riot collection.

Primary artifacts:

- Protocol: docs/route_probe_protocol_v1.md,
  SHA f6e0a8de1fdaffa76dc7775457cb1f618058c384858eafe71997cb4d5d23ec18.
- Bounded-reader amendment: docs/route_reader_amendment_v2.md,
  SHA0f0b940e528dfc11f3d63f0e1cfc094b6970968a10521694385e1f48c3096506.
- CUDA amendment: docs/route_gpu_objective_amendment_v3.md,
  SHAe22716ced993861ae726a4201890b5b475ec9feed08aba4be977640e0896df73.
- Fit: data/runs/route-probe-fit-20261003-v3/experiment/report.json,
  SHA838056d4849ebbe3d5db8e293023352a5168f88cfb2782ccdcf2ddfecbe345a3.
- Evaluation: data/runs/route-probe-evaluation-20261003-v1/evaluation/report.json,
  SHA10a88830ce17bfc064c4ab4e4301e4ed7017e4413bc9a1b3b1d587546c4718af.
- Audit: data/runs/route-probe-evaluation-20261003-v1/audit/report.json,
  SHA751dcd8301695f7a5a2023c043c50a4738445421887a0adb1dba9c76f7daef6f.

## Decision and limits

Close this fixed public-frontier kernel and26-feature linear route correction.
No temperature/ridge/ontology/feature sweep, restart, control promotion or
forward859trial follows the failed primary. The two net extra correct first
purchases do not demonstrate the requested practical breakthrough.

This rejects the tested estimator, not every hierarchical, sequential or
revealed-preference model. It introduces no newly observed information beyond
the current state; mathematical coherence does not establish predictive value.
The result also does not establish causal value, optimal purchases or a live
advantage. The legacy role fields in the frozen M2 reference remain uncertified
as decision-time/live-safe; new correction features do not use them.

All own jobs completed/unregistered. No serving change or promotion. All1911
confirmation games and source/global seals remain closed. The original index
SHA94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46 is unchanged.
The broader objective remains ACTIVE and UNACHIEVED.
