# M2 purchase prediction: explicit user criterion

Recorded 2026-10-01, following the user's direct instruction:

> la respuesta correcta a todas estas compras es las que se compra el jugador

The observed historical purchase is the supervised ground truth. The practical
objective is to improve M2's prediction of that purchase from the information
available before the decision. Independent subjective purchase ratings are not
a dependency. This supersedes the provisional quality-rubric direction and
its unresolved-reference blocker in `research_progress_20260929.md`.

## Target and success

For a decision state x, let B be the recorded retained purchase multiset,
including item quantities. Learn P(B | x) from admitted training games and
compare displayed predictions with B on separate games. Never replace B with
the model's prediction, a hypothetical better buy, a future completed build,
or a purchase reconstructed from the target itself as an input feature.

Preserve the established M2 main-purchase endpoint: exact agreement after
stripping only wards 2055 and 772043, with all other counts retained. Report
complete-basket exactness separately, including wards. A ward-only purchase
can agree with an empty MAIN basket; that is not a correct full no-buy. An
explicit recorded no-buy remains distinct. These semantics already exist in
`winprob/ordinary_policy_evaluation.py`; no scoring implementation or historical
result is changed by this clarification.

Primary evidence is a paired improvement over frozen M2 on the same decisions,
using the actual first displayed legal recommendation. Also report whether the
actual main purchase appears among the three displayed options. Catalogue-only
scores, related item families and generic win-probability loss do not substitute
for those endpoints. Report misses rather than removing inconvenient targets.

Use disjoint games for fitting, calibration, selection and confirmation. The
859-game ordinary cohort is exposed development data, not fresh confirmation or
training data. All 1,911 own confirmation games and all source/global reserved
games remain closed. A new candidate needs a fixed experiment before evaluation;
this clarification neither opens a seal nor changes any completed trial's gate.

## Consequences for the next work

- Retire the eight-case subjective review as a required step. No human ratings
  were received or ingested. The user's statement defines the target; it must
  not be converted into fabricated ratings of the displayed alternatives.
- Reuse the existing observed targets and multiset evaluator. No additional
  annotation pipeline or new data collection is required for the criterion.
- Keep the rejected WP/kernel line shelved. Improvements must directly address
  purchase prediction and ultimately beat M2 in displayed-policy evaluation.
- Preserve the frozen M2 package, source code/data, experiment artifacts and
  report index. Work in this sibling project and exchange versioned artifacts.

The latest verified ordinary baseline is 454/859 (52.85%) main-basket top-1
exactness and 591/859 (68.80%) main-basket top-3 coverage. Complete-basket top-1
exactness is 354/859 (41.21%); the exported M2 options contain main purchases,
so that last number is not an evaluation of an additional ward recommender.
Source: `data/runs/ordinary-evaluation-20260930-v1/experiment/report.json` and
`data/reports/ordinary_evaluation_audit_v1.json`. These are existing development
results, not a new evaluation or an improvement from this clarification.

The breakthrough remains unachieved. The missing-criterion dependency is now
resolved; demonstrating a useful gain remains the work to do.
