# IQL numerical contract — no trained critic

`winprob/iql_kernels.py` is a NumPy reference for targets, losses, parameter
updates and fold provenance. It uses no real trajectories and imports no
source-project code. It is not an alternative toy model to the proposed
flexible critic. An actual Torch learner must implement the same operations
with explicit stop-gradient boundaries and independently validate its outputs
against these tests after the own-project runtime is available.

## Full action and episode boundaries

The source-owned contract is a complete retained purchase multiset. Keep
control wards and multiplicities; never collapse it to a ward-stripped main
group for the critic or transition model. Data-admission reports an exact
reference lift over 13,166 full actions from M2's 9,333 main groups using the
frozen A-arm within-group conditional. That is a reference prior, not a
logging propensity, and must be bound as a separate artifact before actor
training. The kernels do not construct or assume this artifact.

Empty SAVE is a distinct possible full action, not the same action as buying
one or two control wards. `full_basket_key` maps the sole source sentinel
999999 only with its explicit observed_no_buy_death tag to empty SAVE; it
never treats that sentinel as a physical item. It does not add SAVE nodes absent from
the source trajectory. `trajectory_contract.compile_transitions` remains
the authority for source episode qualification, gaps and exactly-once terminal
reward. No player chain is silently stitched through a rejected decision.

The primary reward is zero on nonterminal transitions and Y on the transition
arriving at the absorbing terminal; gamma=1. The terminal node has no action
and no state features. Targets never evaluate V on a fabricated terminal
state or multiply NaN by zero. A framework adapter must query only existing
nonterminal next-state references.

## Fixed update conventions

Use two flexible online critics Q1/Q2 and their target copies. Initialize each
target identically to its online network. The numerical reference accepts
named parameter arrays and returns independent copies, with no in-place
mutation. The proposed order is:

1. Evaluate min(target Q1,target Q2) on each recorded state/full action only.
   Fit V to the .7 upper expectile of these **Q predictions**, not to an
   upper expectile of the realized binary outcome.
2. Form the detached target `r + (1-done)*V(next)` using the updated V. Fit
   each online Q by ordinary squared error on its observed full action.
   There is no next-action maximization or candidate-action enumeration.
3. Update target parameters after the Q optimizer step with
   `.995 * old_target + .005 * updated_online`.

For actor fitting, use frozen critic/value predictions from a learner that
did not see the scored game. All players and states of a game stay in the
same one of two deterministic folds. Exposure provenance must include
training, calibration and data-dependent model-selection IDs, not only the
last optimizer's rows. The fixed advantage weights are
`min(exp((minQtarget - V)/.1), 20)`. The denominator .1 is a temperature;
the original JAX implementation names the inverse multiplier temperature.
Do not confuse these conventions. The generic raw-weight kernel normalizes
sampling mass, not advantage weights. The application recipe uses uniform
transition sampling and one separately persisted global FIT rescale:
`normalized_weight = raw_clipped_weight / mean(raw_clipped_weight_on_FIT)`.
This mean is computed once from OOF predictions, never within each minibatch;
cap20 applies before this rescale. Record the constant, raw cap fraction and
final effective sample size. This is an optimizer-comparability choice.

These conventions follow the structure of the
[IQL paper](https://arxiv.org/abs/2110.06169) and the authors'
[critic](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/critic.py),
[update order](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/learner.py)
and [actor](https://raw.githubusercontent.com/ikostrikov/implicit_q_learning/master/actor.py)
implementations. Gamma, expectile, clipping, bounded heads if selected, and
the finite basket actor are application choices, not transferred guarantees.

## Shaping and gradient boundaries

Optional potential shaping uses `r + gamma*(1-done)*Phi(next) - Phi(current)`.
Terminal Phi is zero because Y is already paid by r. At gamma=1 this changes
the episode return by `-Phi(initial)` and supplies no new action information.
Compare shaped/unshaped learners at equal budget if shaping is studied.

The NumPy losses return scalar/per-row losses and analytic gradients with
respect to online V, online Q or observed-action log probability. Target
arrays, advantages and sampling weights have no trainable role in this API.
This is not automatic differentiation: a Torch implementation must detach
the appropriate tensors and test that target networks receive no gradients.
It must also match observed-only Q queries and terminal masking.

## Dependencies and remaining implementation

Numerical tests need only this project's NumPy and standard library. The
planned flexible state/full-basket critic needs the root-managed own-project
Torch installation; do not import another project's runtime or replace the
critic with a small linear model just to obtain a completed run. Torch model
architecture, categorical full-action reference, source/schema binding,
resource-gated training, loss/convergence diagnostics and evaluation remain
separate work. The Torch architecture now lives in `iql_models.py`: shared
71->64->64 per-player encoding, symmetric own/ally/enemy pooling, 357->256->128
ReLU/LayerNorm state representation, independent twin Q encoders and V,
count-preserving full-basket encoder, and an ordinary 13,166-way actor.
Public static-union vocab embeddings start at zero, so rows unseen by a fold
remain zero; unknown IDs outside the union fail upstream. Ragged inventories
are padded only to the actual batch maximum, never truncated. Machine-readable
global37/inventory22 enums are imported from `iql_features.py`.

No data payload, real model fitting, policy improvement or causal outcome
evidence is established by these modules or their synthetic tests.
