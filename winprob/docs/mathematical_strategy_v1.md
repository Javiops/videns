# Mathematical strategy for decision learning

2026-09-30. A proposed adaptation of established methods, not a new theorem
or an established improvement in wins. Mathematical kernels are implemented
in `winprob/decision_math.py`; no IQL/IPO actor or critic has been trained.

## One reusable policy form

For a fixed legal action set and frozen normalized M2 reference, use

`pi_new(a|x) = pi_M2(a|x) exp(f(x,a)) / sum_b pi_M2(b|x) exp(f(x,b))`.

Zero correction reproduces the same conditioned reference. The normalizer
cancels in pairwise log-odds. With `f=A/T`, this is the exact finite-action
solution of maximizing expected estimated advantage minus `T*KL(pi||pi_M2)`.
This is not a claim that M2 is the historical human logging propensity.
Legal masking and empirical data support are distinct questions.

## Outcome-directed route: implicit Q-learning

[Kostrikov, Nair and Levine, IQL](https://arxiv.org/abs/2110.06169)
learns a state value by asymmetric squared loss on observed action values,
then a Q function through Bellman regression. Policy extraction weights
observed actions by exponentiated estimated advantage. Our small-correction
M2 formulation is an adaptation, not an IQL theorem about this project.

Use `A=Q-V`. The practical learning question is whether conditioning on the
action improves expected future outcome, rather than whether one realized
trajectory happened to finish well. Complete player-game decision sequences
and well-defined basket/action semantics are required. Full state/history,
support and confounding assumptions remain material in observational LoL data.

Win probability can shape rewards without changing their total ordering:
nonterminal rewards are `w(next)-w(now)` and the final reward is `Y-w(last)`.
Their sum is exactly `Y-w(initial)`, so terminal victory is paid once.
[Ng, Harada and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
provide the potential-based shaping foundation. A state-only potential shift
also shifts Q and V equally, leaving their difference unchanged. Shaping alone
cannot create action information; compare shaped and unshaped IQL at equal
budget, rather than calling the reward rewrite a breakthrough.

## Preference-directed route: IPO

For audited retained/rejected actions under the same context, define
`h = log(pi_new(a+)/pi_M2(a+)) - log(pi_new(a-)/pi_M2(a-))`.
The residual form makes `h=f(a+)-f(a-)` exactly. IPO fits the finite target
`L = tau * (h - 1/(2*tau))^2`.

[Azar et al., IPO](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html)
is attractive for sparse revision pairs because its target margin is finite.
Keep [DPO](https://arxiv.org/abs/2305.18290), with loss
`-log sigmoid(beta*h)`, as a fixed comparison. Finite margins do not by
themselves prove robustness to mislabeled corrections. Undo/replacement
records are selected, dependent observations; original sampling guarantees
do not automatically apply. This learns revealed preference, not win effects.

## Initial implementation and evidence boundary

Eight synthetic properties check exact reference recovery, support retention,
the KL variational optimum, normalizer cancellation, reward conservation,
expectile weights and finite IPO targets. They validate algebra only.

The next measured comparison needs an audited transition or preference dataset,
frozen M2 probabilities on the identical action unit, matched baselines and
untouched game partitions. Keep two separate hypotheses: outcome advantage
through IQL, and prediction of player revisions through IPO. Do not combine
their scores under a generic 'value' name without a new validated objective.
