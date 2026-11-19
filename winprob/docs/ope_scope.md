# Identification gate before observational policy evaluation

This module is a bounded synthetic falsification harness. It reads no game
data, fits no models, and uses exact rational arithmetic. Passing it establishes
that these algebraic examples behave as declared. It does not establish any
real-data propensity, exchangeability, overlap, confidence-interval coverage,
or benefit from a recommendation. It cannot complete the research goal.

Run from this project with its own or a system Python:

```powershell
python -m unittest discover -s tests -p test_ope.py
python scripts/check_identification.py --output data/reports/identification_v1.json
```

The script exits nonzero on a failed algebraic check. A real W4 runner must
require this gate plus a separately approved identification and evaluation
protocol; a passing JSON report alone is never permission to make causal claims.

## The counterexample

The population contains equal numbers of strong and weak players. Strong
players choose action A with probability 0.8 and win with probability 0.8;
weak players choose A with probability 0.2 and win with probability 0.2.
Action B is the alternative. Both actions have the same potential-outcome
mean within skill, so their true causal difference is zero.

A constant win-probability judge of 0.5 is perfectly calibrated marginally:
the population win rate is exactly 0.5. Nevertheless, A buyers win with
probability 0.68 and B buyers with probability 0.32. Mean terminal outcome
minus the judge is therefore +0.18 for A and -0.18 for B. That is +18 and
-18 percentage points of spurious action value, with no sampling error.

With skill observed, doubly robust adjustment recovers the exact null when
either the propensity or outcome model is correct. When both are wrong, the
example reports a false A-minus-B gain of 0.36. With skill hidden, even the
correct *observational* propensity and conditional outcome model yield 0.36:
double robustness does not repair failed causal identification.

A positive control gives A a true 0.20 advantage in both skill strata. Either
correct nuisance model recovers this nonzero truth, preventing an estimator
that always returns zero from passing the harness. All values are synthetic
population expectations across eight atoms; zero real games are measured.

## The estimand and paired score

For one decision, target policy pi1 and reference pi0, outcome regression q,
and historical propensity g, the directly paired score is:

```text
sum_a (pi1(a|x) - pi0(a|x)) * q(x,a)
+ (pi1(A|x) - pi0(A|x)) / g(A|x) * (Y - q(x,A)).
```

The implementation evaluates both policies on the same observation. Identical
policies give exactly zero per observation; reversing policies reverses every
contribution. No weight clipping is performed. A real analysis must quantify
the bias of any clipping and its effect on the target population.

Interpreting this as a causal contrast requires consistency, measured
pre-treatment covariates sufficient for conditional exchangeability, and
historical support for the evaluated policies. With terminal reward after a
single purchase, the intervention changes that purchase and then allows
natural continuation. It does not estimate following a new policy at every
future purchase. Team and opponent reactions belong to that continuation;
joint interventions require a separately specified estimand.

M2 is a learned imitation model, not a known historical randomization policy.
Normalized scores or top-three membership do not prove positive historical
support. The harness rejects a target action with zero supplied propensity,
even when it belongs to a three-action shortlist. Positive supplied values do
not prove that a fitted real-data model is accurate or that overlap holds.

## Further gates required on real data

- Define the exact decision, legal action space, pre-action state, population,
  and intervention before outcome inspection. Never interpolate from a future
  minute frame or select eligibility by the observed treatment or outcome.
- Fit and calibrate historical propensities separately from the fixed target
  policy. Account for missing and out-of-catalogue actions explicitly.
- Keep all states from one game together. Select a policy independently of its
  evaluation cohort and cross-fit nuisance estimation within the declared
  development design. A previously exposed cohort is not a fresh holdout.
- Report the supported population, effective sample size, weight tails,
  paired match uncertainty, and sensitivity to hidden confounding. Correct
  synthetic expectations do not validate a bootstrap or its coverage.
- Use the known 0/1 result as the absorbing value for a game ending before a
  reward horizon. Do not drop early-ending games or ask a nonterminal judge
  to invent terminal truth. The small `absorbing_value` helper implements
  only that rule; it does not validate time alignment or construct horizons.
- Require a preregistered useful improvement on independent games before any
  policy claim. W1 prediction quality and W2/W3 agreement do not establish it.

## Primary methodological sources

- [Dudik, Langford and Li: Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)
  develops the contextual-bandit estimator combining reward and historical
  policy models; its robustness is not robustness to arbitrary confounding.
- [Jiang and Li: Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://proceedings.mlr.press/v48/jiang16.html)
  treats sequential evaluation, which is a different problem from pooling
  single-decision contextual scores.
- [Chernozhukov et al.: Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060)
  describes orthogonal scores and cross-fitting to control nuisance-estimation
  bias under the applicable identification and regularity conditions.
- [Kallus and Zhou: Confounding-Robust Policy Improvement](https://arxiv.org/abs/1805.08593)
  studies sensitivity to unobserved confounding through uncertainty sets and
  improvement relative to a baseline; no such real-data sensitivity bound is
  implemented or established by this synthetic harness.
