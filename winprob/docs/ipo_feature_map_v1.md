# Frozen small IPO feature map and implementation contract

Declared before real residual fitting. This implements the design in
`ipo_experiment_design_v1.md`; it does not waive canonical context/action,
reference exposure, support, size or actual zero-residual identity gates.
The current receipt file alone is not a training artifact.

The actor context has **22 coordinates**: intercept, pre-A time in minutes,
blue-side indicator, own pre-A level, and six champion-tag coordinates each
for own champion, four allies and five enemies. Tag order is Assassin,
Fighter, Mage, Marksman, Support, Tank. Own tags are indicators; ally/enemy
tags are counts. Champion tags use the bound patch's immutable roster static
metadata. No inferred lane, omniscient team gold, damage, position, WP, wallet,
refund, event order, winner or post-A information enters the adapter.

Use the source-qualified pre-A `source_projection` only: champion_id,
team_id, level and the nine `others` champion/team IDs, plus the independently
bound outer context_timestamp_ms. Missing inputs or incorrect 4/5 team counts
fail. Unknown champion tags are not filled with plausible values.

Standardize the 21 nonintercept context coordinates with fit-only weighted
mean and population standard deviation; constant coordinates use scale one.
Keep the intercept exactly one. Freeze these parameters before any other
partition is transformed. Do not fit means to calibration/development data.

Action coordinates have fixed, state-independent physical units, defined
from **M2's pinned 16.18 item world** and exact canonical group multisets:
total price/3000; HP/1000; mana/1000; AD/100; AP/100; armor/100; MR/100;
movement speed/100; attack-speed fraction/.5; crit fraction/.5; lifesteal
fraction/.3; then ten tag counts/6 (AttackSpeed, Armor, SpellBlock, Health,
Damage, SpellDamage, Boots, CooldownReduction, LifeSteal, MagicPenetration);
item-count/6; and SAVE indicator. Raw stat keys are explicitly enumerated in
code. This gives **23 action coordinates**, **506 bilinear parameters**.
Missing *declared stat keys* mean zero recorded stat; a missing item definition
is an error. Unrepresented passives and identical feature vectors are model
limitations; report feature collisions rather than invent effects.

The fixed action scaling avoids penalizing a 1000-gold unit and a binary tag
as if their numerical units were comparable. No action scaling is estimated
from preferred/rejected orientation. The same fixed action vectors serve all
source patches, matching the frozen reference's world; source-patch mechanics
are not silently substituted into M2's action feature dictionary.

Compute `D_i = vec(c_i outer (v(B_i)-v(A_i)))` only after these transforms.
**Never center D.** There is no learned standalone intercept in a pair margin.
Margins must obey h(A,B)=-h(B,A) and h(A,A)=0. Globally swapping A/B and
flipping the target sign must leave the fitted coefficients unchanged.

For normalized positive weights w, tau=.5 and lambda=.01, solve
`(tau D' diag(w) D + lambda I) theta = .5 D' (w * target_sign)`.
The genuine retained-item imitation baseline receives the same context,
action vectors, full 9,333 reference masses and B labels, but never A. Its
normalizer includes the entire canonical action space, not the observed pair.
Use deterministic L-BFGS with at most 250 iterations and convergence checks.
The generic implementation is tested only on synthetic arrays until the
source-owned export and identity gates are available.

Probability input is validated at the source's 2e-6 row-sum tolerance. A zero
W returns the **original supplied reference array unchanged**, preserving
dtype and exact bytes, with no log/softmax/renormalization roundtrip. The CLI
requires this identity on every row before fitting; it composes with the
source's actual native-decoder parity on the exact same float32 masses.
Internal winner-only CE and nonzero softmax computation use float64 canonical
normalization, preserving zeros. M2 is a policy regularizer, never an asserted
historical logging propensity.

The real loader must bind manifest schema `winprob-m2-preference-v1`, full
probabilities tensor, exact group mapping, ordered canonical contexts,
certificate IDs/roles, static export and each file's SHA256. Reject a partial
probability export, failed native zero-residual equality, unknown/unmapped
pair action, zero reference mass on either pair member, or mismatched row
identity. Fit only after postmapping fit/challenge size gates pass. No real
training command is authorized by synthetic kernel tests.

`scripts/fit_ipo_residual.py` defaults to training only after the source loader's
full identity and postmapping population gates pass. An explicit
`--evaluate-development` freezes both fitted artifacts before one development
comparison. The original +5pp-over-M2-and-imitation gate is reported unchanged.
An additional usefulness guard requires positive paired evidence over both
cheaper-item, more-expensive-item and fit-derived retained-item-frequency
rules, avoiding the weak-baseline problem
found in the setback experiment. No state-dependent superiority is claimed
solely from this diagnostic; the modest actor may learn global item bias.

The frequency control counts retained B on the same fit revision rows by
own champion and patch, with additive .5 smoothing. For a new patch use that
champion's latest fit-patch counts; if that champion has no fit cell, use
global fit counts. Never fit a 16.19 frequency on challenge labels. This is
explicitly a revision-derived frequency control, not an unprovided ordinary-
shop-frequency artifact. Report fallback counts, pair-design rank, action-
feature rank, correction/KL drift, and pinned-price direction slices. Slices
under 30 distinct games are marked unmeasured. Nonconverged winner-only
optimization writes its gradient/message diagnostics and aborts evaluation.
