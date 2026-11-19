# A complete causal state/action learner with optional teacher supervision

2026-10-04. Reassessment after closing the Bayesian proxy-ratio experiment.
No new purchase scores or model fit have been run for this design.

## Why change the learner

The own state/history proxies use per-player encoding followed by ally/enemy
mean/max pooling and a flat9333-class output head. Native M2 instead uses a
board transformer and a counted-basket encoder with joint state/action scoring.
These are verified implementation differences, not an explanation established
by a controlled comparison. Changing the weaker proxies' probabilities has
not produced a useful native M2 improvement.

The source history contains positive measured data-scaling results and mostly
negative gate/capacity/transfer variants. A metadata-only check also establishes
that source16.19 TRAIN contains2770games, exactly859own development plus1911
own confirmation. Those games cannot silently become new training data.
No source16.19payload or confirmation seal was opened for this check.

The next candidate is a complete learner, not another runtime M2 multiplier.
It retains the ten individual public player tokens, exchanges information by
attention, and scores counted item multisets through shared item representations.
This is an architecture/data-efficiency hypothesis; no improvement is assumed.

## Causal forward interface and prototype

Use the existing validated public board tensors (champion IDs, counted
inventories, public inventory attributes, levels, own public globals). Add
only own selected keystone and secondary rune-tree categories if a separate
versioned input stage verifies those fields. They are pregame selections;
initial/dynamic summoner-spell semantics are not added in this prototype.
No final role, outcome, future build, teacher prediction, current label or
other-player private wallet belongs in the student's prediction interface.

Prototype: champion16 and shared item32 embeddings (normal0,1 initialization,
padding row0 fixed at zero); player71->64 projection;
buyer/ally/enemy type embeddings, no player-position embeddings. Add a query
token from37public globals and two8-dimensional rune embeddings. Two transformer
layers,d64,4heads,FF128,dropout.1. Query and buyer outputs project to64.
The action encoder shares the counted item embedding, concatenates23public
action attributes and maps55->64->64. Main group0has a learned empty-main vector;
its meaning includes ward-only and is not renamed an observed no-buy.

Score each action by a bilinear term plus a192->256->64->1 GELU interaction
network over state, action and their product. Evaluate action chunks of512;
checkpoint only the deterministic deep interaction during training to bound
activation memory. No inventory or action count truncation. Tests must verify
player permutation symmetry, quantity preservation, chunk/checkpoint gradient
equivalence and no label/teacher argument in prediction. This prototype is not
the frozen M2 implementation and does not import sibling runtime code.

## Actual-purchase objective and teacher regularizer

The recorded retained main basket y remains the supervised ground truth.
An optional training-only regularizer uses the frozen M2 distribution t:

    L_hard = -log q(y|x)
    L_distill = -log q(y|x) + KL(t || q(.|x)).

Temperature1 and equal unit coefficients are a prospective fixed comparison,
not tuned on new scores. The derivative with respect to a logit is q-e_y for
hard supervision and2q-e_y-t for the regularized loss. If t=e_y, the latter is
twice the same observed-label loss; teacher supervision does not redefine which
purchase is correct. Store hard loss and teacher regularization separately.

In any real cross-fit, only the training fold's teacher rows may regularize
the learner. No teacher call/probability or hidden role appears at inference.
Teacher metadata that is not certified at decision time is privileged TRAIN
information, never a new student feature. The student's actual forward pass
must remain invariant when held-out targets/teacher/roles change.

[Generalized distillation](https://arxiv.org/abs/1511.03643) studies learning
across representations including privileged information.
[Born Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html)
reports settings where a student exceeds its teacher. Neither establishes a
League gain or guarantees that this student will beat M2. This is different
from the closed win-probability distillation study: the objective here is the
actual counted purchase and the comparator is native displayed M2.

## Next gates, not yet a predictive experiment

Implement the prototype and its math/invariance tests. Verify input availability
and GPU/host memory with a bounded synthetic forward/backward pilot before a
real long fit. Preserve the already admitted3518FIT16.17games/56288selected
states and2900focals unless a separate prospective protocol establishes a
different admissible population. Existing complete M2 priors can be reused;
no new source neural export or collection is necessary for these states.

Before real fitting, separately freeze candidate/control roles, training recipe,
teacher-row permissions, input artifacts, source decoding, paired uncertainty,
success rules and whether student-only inference meets the practical gate.
No teacher loss, model capacity, synthetic success or weak-proxy improvement
can stand in for beating unchanged M2. No sealed data, forward859trial,
promotion or index modification is authorized by this prototype stage.
