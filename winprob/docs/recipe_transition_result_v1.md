# Net recipe attributes did not improve native M2 purchase prediction

2026-10-03. The completed correction experiment fails its declared useful-gain
gate. The numerical optimizer now reaches the required precision and every
prediction replays exactly. This is a verified negative result, not a remaining
implementation failure or a practical breakthrough. Close this fixed recipe.

## Population and measured result

3518 original admitted16.17 FIT games supplied56288label-blind selected states,
16per game. All games are disjoint from M2's6354training games. Two unchanged
whole-game folds train on opposite-fold states. Primary evaluation uses2900
original fixed focal decisions, one per game;618original unavailable focal
games remain disclosed and were not replaced. This is an already exposed,
retrospective internal screen: M2 was trained on16.18 and its native16.18 engine
scores these16.17 states. It is not the859-game forward development trial.

Main-purchase first exactness, retaining quantities and stripping only the two
declared ward IDs:

- Unchanged M2:1483/2900,51.1379%.
- Net-attribute correction:1481/2900,51.0690%.
- Matched raw-attribute control:1489/2900,51.3448%.
- Shuffled-net control:1475/2900,50.8621%.

Net minus M2 is-0.0690percentage points, paired95% game-bootstrap interval
[-0.6552,+0.5172]pp:37corrected and39broken. It does not establish either an
improvement or a statistically resolved worsening. Net minus raw is-0.2759pp
[-0.6897,+0.1724]; net minus shuffled is+0.2069pp[-0.3103,+0.7241].

On2736 nonempty-main purchases, net-M2 is exactly0.0000pp, interval
[-0.5900,+0.5883]pp. The remaining164focals are119ward-only purchases and45
observed no-buy-death cases. The positive-lower-bound nonempty guard fails.

All arms preserve the same original option sets and top3 coverage:
1959/2900,67.5517%. Full-basket first exactness is1104/2900,38.0690%, for net
versus1106/2900,38.1379%, for M2. Ward-only is not counted as a correct full
SAVE. The coarsened four-category NLL is0.887774 for net versus0.883664 for M2;
difference+0.004111,95% interval[-0.000750,+0.008935]. This is not full-catalogue
NLL and must not be compared with earlier full-catalogue likelihood figures.

The primary required at least3pp versus M2 with a positive95% lower bound,
a positive lower bound versus the matched raw control, a positive nonempty-main
lower bound and unchanged options/top3/legality. It fails. All intervals use
the same2000paired whole-game draws, seed20261001, and are conditional on the
fitted models; they do not include training-set uncertainty.

## Construction and verification

The model keeps the native M2 prior and its original up-to-three legal main
baskets. Its676-parameter logit correction uses public context and native
hypothetical inventory transitions. Purchased static attributes minus consumed
item attributes produce the net descriptors. OTHER retains every observed
purchase outside the shortlist in the likelihood. These ten static descriptors
exclude passives and do not simulate actual champion combat stats or action value.

The source pipeline verified3518complete indexed-game receipts, prepared all
56288contexts with targets in a separate file, and scored unchanged native M2.
The200-state real pilot verifies exact input/prediction/option parity after
ignored-target/future mutation. Own consumption independently verified all
164731native candidate receipts, query identities, static attributes and
catalogue mappings before opening targets. Actual feature digests match across
all runtime versions; neither the selected data nor the model equation changed.

Six convex fits completed from zero initialization with lambda.01. Their final
iterations are88-125, totaling619iterations and671function evaluations. Every
fit ends on its requested projected-gradient criterion; maximum absolute
gradients range6.057e-8 to8.422e-8, below both gtol1e-7 and the1e-6 acceptance
gate. All coefficients/scalers were frozen before held-out prediction and the
prediction artifact was frozen before scoring.

The auditor independently recomputed training-only weighted means/scales
(maximum discrepancy1.709e-13), checked train IDs and convex objective/gradient
receipts, and replayed all8700correction queries plus2900native baseline queries.
Probabilities, stable log probabilities and option orders match exactly.
It recomputed every summary, paired interval, slice and failed gate. It did not
rerun all56288source M2 neural forwards; source recommendations are bound
artifacts with the separately documented200-state mutation check.

## Numerical history, costs and evidence

Fit v1/v2 stopped before any target read because an instantaneous timing maximum
failed the operational projection. Runtime v3 uses sustained20-pass blocks and
the45minute hard limit; no statistical setting changed in those retries.
Its first real raw-control fit then stopped at relative loss tolerance1e-12
after79iterations, with gradient2.11555e-6. A diagnostic reproduced the cause
and verified the gradient with directional finite differences (errors<=6.512e-11).
The final v4 run changes only that function-change tolerance to0, keeping
gtol1e-7,maxiter300,maxls50 and the exact same strictly convex objective.
All earlier runs and the unqualified diagnostic coefficients are preserved;
none was used as a warm start or evaluated for accuracy. The attempted diagnostic
on October1 was not executed because automatic approval review reported a usage
limit. It was revalidated and executed normally on October3.

Source role recovery340.57s; separated context preparation84.41s; complete
native M2 release3286.42s (54.77minutes), peak Torch allocation164065792bytes.
The successful own fit/evaluation took125.38s, including25.54s across the six
optimizations; independent audit74.12s. CPU1 throughout; CUDA only for native
M2. Stopped-attempt overhead is retained in its run logs and not included in
those successful-stage timings. Total wall time across retries is not aggregated.

Final experiment: `data/runs/recipe-transition-fit-20261003-v4/experiment/report.json`,
SHA256 0ed980c46e99c31f342b983935e99fc4feb7492242374410302b8583cac2c2d8.
Audit: `data/runs/recipe-transition-audit-20261003-v4/audit/report.json`, SHA256
a629f732b26d96ed7b98fbfd7a5746156fbf750d69e21708ecb774c1244e33a7.
Native M2 release: `build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2/manifest.json`,
SHA256 f0bf5e91a06a04151654c2293364bfe47e46d8b9acbfe96eb96822c3020c9abe.
The fixed protocol and numerical/runtime amendments are in
`recipe_transition_probe_protocol_v1.md`, `recipe_fit_runtime_v2.md`,
`recipe_fit_runtime_v3.md` and `recipe_solver_precision_v2.md`.

## Decision and remaining limits

Do not select the raw control after the failed primary, tune lambda on these
results, expand this recipe or run its859-game forward trial. No model is
promoted. The result rejects this fixed descriptor/linear-correction recipe;
it does not prove that every recipe representation or every additional input
must fail. Legacy M2 roles remain uncertified as predecision/live-safe. The
original complete/catalogue-supported episode restriction limits the population.
No causal benefit, actual live accuracy or better win probability is established.

All1911confirmation games and source/global seals stay closed. No original
report-index edit occurred; SHA256
94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46 remains unchanged.
All recipe tasks/workers completed; unrelated services were preserved. The
broader objective remains ACTIVE and UNACHIEVED.
