# Decision-learning direction: preference first, outcome learning alongside it

2026-09-30. Recommendation from code, existing result documents and primary
papers. No game payload, new prediction result or held-out outcome was read.

**Keep the small IPO residual as the nearest practical experiment. Make
allocation-aware IQL the complementary outcome-directed experiment.** More
generic WP feature searches are currently a weaker use of effort. Neither
route has yet established better decisions or wins.

## What the existing evidence actually supports

The recorded feature ablation favors allocation: log-loss reduction 0.003403
[0.000059, 0.006890] on 998 development games; Brier remains unresolved.
Absolute gold alone and geometry do not resolve gains, and combined does not
resolve an improvement over allocation. Source:
`data/runs/rich-ablation-20260929-v1/experiment/report.json`, summarized in
`docs/research_progress_20260929.md`. This motivates representing **who holds
resources**, not calling that representation an advantage estimator.

Spending does not itself change cumulative `totalGold`. A WP gradient with
respect to gold concentration therefore is not the value of buying an item.
Allocation belongs in state representation; action quality must be learned
from an explicitly action-conditioned target.

## IPO: plausible, but its gain must survive cheap explanations

The existing design is attractive because the zero-residual policy exactly
recovers M2 and its IPO arm is a small weighted ridge problem. IPO's finite
relative-log-odds target comes from
[Gheshlaghi Azar et al., AISTATS 2024, Eq. 17](https://proceedings.mlr.press/v238/gheshlaghi-azar24a/gheshlaghi-azar24a.pdf).
Using transactional revisions as preference labels remains our adaptation;
finite margins do not remove selection, changed information or accidental clicks.

The recorded fit set has 866 strict pairs in 777 games before M2 export
exclusions, against at most 768 adapter parameters. Support minima do not
establish power. Lock action-feature units/scaling as well as context scaling:
unscaled gold prices and binary tags otherwise receive very different effective
ridge penalties. Report design rank rather than treating every coefficient
as independently supported.

In addition to M2 and winner-only imitation, keep price/frequency heuristics
visible and test a fixed **state-independent residual** if not yet locked.
That distinguishes contextual revision learning from a global item preference.
If a cheap declared rule beats the learned arms, a win over weaker controls
does not establish a useful learned method. Use total policy odds, not the
residual alone. The primary residual remains POV; an OMNI advantage is not
a deployable improvement over POV M2. The current design already preserves
the B-time certificate versus pre-A context distinction.

## Concrete complementary experiment

Use qualified full shop-to-next-shop trajectories and canonical action
multisets, not minute frames pretending to be decisions. Terminal reward is
the focal team's outcome once. Keep actual elapsed times and a terminal mask.
For the undiscounted win objective use gamma=1; arbitrary per-shop discounting
would change the objective when the number of shop visits changes.

The IQL equations are:

```text
rho_tau(u) = |tau - 1[u < 0]| * u^2
L_V = E[rho_tau(Q_target(s,a) - V(s))]
L_Q = E[(r + (1-done)*V(next_s) - Q(s,a))^2]
L_actor = -E[min(C, exp((Q_target(s,a)-V(s))/T)) * log pi(a|s)]
```

They use observed actions in critic updates. Crucially, the expectile is over
action-conditioned Q values; applying an upper expectile directly to realized
wins or WP changes favors lucky transitions. This separation is explicit in
[Kostrikov, Nair and Levine, IQL, Eqs. 5–7](https://arxiv.org/pdf/2110.06169).

Our proposed first contrast is a small IQL-weighted residual actor versus an
equal-budget uniform-imitation residual, both starting from the same frozen
M2, with identical actor features and ordinary training decisions. Include
frozen M2 as a third reference. Fit/calibrate only on their declared games;
use game-disjoint critic predictions to construct actor weights. Freeze one
recipe instead of sweeping expectiles/temperatures on the challenge.

Use observable resource-allocation proxies for any deployable actor. An OMNI
critic or potential is a separately declared privileged-training variant,
not an unnoticed actor-input expansion. Other-player behavior, hidden intent
and state insufficiency remain limitations of interpreting learned Q causally.

WP shaping can later be an equal-budget optimization contrast. With terminal
potential zero, `r' = r + Phi(next)-Phi(now)` sums to `Y-Phi(initial)`; at the
exact solution Q and V shift together, so their difference gains no new
information. The foundation is
[Ng, Harada and Russell, ICML 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
Do not make shaped rewards the claimed source of a new decision signal.

## Validation that can justify the next step

Evaluate the outcome-trained actor on the frozen 16.19 revision challenge
without training it on those preference labels. Compare its total pair odds
against M2, ordinary imitation, IPO and the fixed cheap rules, with mapping
coverage, native legality and paired whole-game uncertainty. This tests whether
outcome learning transfers to an independently defined correction task.

A successful result would justify a frozen confirmation candidate and blinded
human assessment of the changed recommendations. Critic score, Bellman fit,
WP accuracy and the actor's own estimated value cannot independently validate
the policy that optimizes them. A win-rate claim needs a prospective outcome
comparison or a separately justified causal design; preference prediction alone
does not supply it.
