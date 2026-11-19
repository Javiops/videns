# Winprob

Independent research project for retrospective win probability and decision
review in videns.gg. The objective is a demonstrably useful improvement, not a
high-looking training score or an unsupported claim that an item causes wins.

The starting proposal is
`../informes/20260929/wpa_dossier_for_external_review_20260929.md`.
The active protocol is `docs/protocol_v1.md`. This project does not import
build_tracker or macro_coach code, modify their databases, or serve live advice.
Raw archive reads must pass the explicit source-admission manifest. Every
derived data release and model records its inputs and SHA-256 digests.

Current user direction (2026-10-01): improve M2's prediction of the player's
actual recorded purchase. The observed purchase multiset is the correct label;
subjective ratings are not required. `docs/m2_imitation_objective_v1.md` records
the criterion and supersedes the provisional quality-review detour. The
WP/kernel direction remains shelved. The missing-criterion blocker is resolved;
a demonstrated improvement over M2 is still outstanding.

Purchase-memory experiments (2026-10-01) now provide a positive internal
mechanism screen but no verified M2 improvement. On3350 fixed OOF decisions,
state+memory improves main top1 by3.37pp [2.33,4.57] over the state-only model.
The subsequent fixed reranking of M2 on859 exposed development games improves
by1.16pp [-0.35,2.56], below the3pp gate and statistically unresolved. Nothing
is promoted. See `docs/purchase_memory_result_v1.md` for both populations,
controls, costs, audit scope and remaining limits.

The full-catalogue follow-up also failed: the same memory distribution decoded
through unchanged M2 source logic still gets 464/859 first purchases right,
while top-3 coverage drops from 591 to 582. All 2,516 memory options pass both
economic-world audits. The fixed recipe is closed without promotion; its
internal gain does not establish a useful M2 replacement.

The exact multiset-likelihood experiment also completed and passed functional
verification. All six models trained their fixed20epochs. On3350 OOF decisions,
marginal top1 is44.2090%, versus45.2239% categorical and47.2836% frozen memory.
Its-3.0746pp versus memory has95% interval[-4.4179,-1.7604]pp. NLL improves,
but actual purchase accuracy fails the declared gate. All10050 neural
predictions replay exactly. The fixed recipe is closed without an M2 trial
or promotion; see `docs/multiset_result_v1.md`.

The other-player public-KDA experiment completed and was audited. On the same
3350 fixed OOF decisions, real-score top1 is45.2239%, zero-score44.7761% and
frozen memory47.2836%. The real-zero gain is0.4478pp [-0.6567,+1.6418]; real
scores lose2.0597pp [-3.4634,-0.7164] to memory. The nonempty-purchase guard
also fails. All10050 neural predictions replay exactly. This fixed recipe is
closed without an M2 trial or promotion; see `docs/scoreboard_probe_result_v1.md`.
The v1 audit finished after RAM became available; the proposed v2 startup-gate
adjustment was NOT applied. No scoreboard training or audit job remains active.

The native-M2 net-recipe correction completed and was audited on2026-10-03.
All six676-parameter convex fits reached their gradient criterion. On2900fixed
internal focal decisions, net correction gets1481right (51.0690%) versus M2's
1483 (51.1379%): -0.0690pp, paired95% interval[-0.6552,+0.5172]pp. Nonempty-main
gain is exactly0; top3 is unchanged. All8700correction predictions and2900native
baseline queries verify exactly. The failed fixed recipe is closed without a
forward859trial or promotion. See `docs/recipe_transition_result_v1.md` for the
population, numerical repair, controls, cost, evidence and limits. No recipe
task remains active; the broader improvement goal is still unachieved.

The personal kill-graph comparison also completed and was audited. Four
272-parameter corrections preserve M2 exactly on zero-information decisions.
On2900fixed internal focals, real graph gets1491right versus1483M2 and1484null:
+0.2759pp versus M2,95% interval[-0.1034,+0.6552]pp. The useful-gain/nonempty
gates fail and coarsened NLL worsens. All5800correction and2900baseline queries
replay exactly. This fixed graph recipe is closed without a forward859trial
or promotion; see `docs/personal_kill_graph_result_v1.md`. Its verified feature
availability does not establish the broader goal, which remains unachieved.

The full-catalogue route comparison also completed and passed its reproduction
audit. Four6708-parameter models use exact complete M2 priors and a public
recipe kernel, compared with a count-stratified permuted kernel. On2900fixed
internal focals, real routes get1485right (51.2069%) versus1483M2 (51.1379%):
+0.0690pp, paired95% interval[-0.3448,+0.4828]. Top3 adds4hits; neither first,
nonempty, control contrast nor catalogue NLL has a resolved useful gain.
All5800corrected predictions and2900baseline priors replay exactly; all options
are legal in16.17/16.18. Close this fixed recipe without a forward trial or
promotion. `docs/route_probe_result_v1.md` records the result, resource-only
reader/CUDA repairs, cost, artifacts and limits. The broader goal remains
unachieved; mathematical correctness and faster fitting are not buy gains.

The closed-history Bayesian proxy-ratio comparison also completed with both
audits passing and a negative primary result. On2900fixed internal focals it
gets1469first purchases right versus1483M2: -0.4828pp, paired95% interval
[-0.9310,-0.0690]. Nonempty accuracy also falls; mean catalogue NLL worsens.
The287empty histories preserve M2 exactly. All392448bank embeddings,18473proxy
forecasts and8700corrected distributions replay exactly, with zero native17/18
legality failures. Close the fixed recipe without retuning or a forward trial;
see `docs/bayes_proxy_result_v1.md`. The broader goal is still unachieved.

The complete public-state student study also finished with a negative result.
On 2,900 fixed internal decisions, native M2 gets 1,483 first purchases right
(51.1379%), the hard-label student 1,003 (34.5862%) and the distilled student
1,032 (35.5862%). Distilled minus M2 is -15.5517 percentage points, adjusted
97.5% interval [-17.5172, -13.6897]. Top-3, nonempty accuracy and NLL also worsen.
Distillation helps the weaker student by 1.00 point; that is not an M2 benefit.
All 8,700 student predictions replay exactly; native legality and final metric
audits pass. Recovery preserved five completed models and exactly reproduced
the first epoch of the interrupted sixth model before finishing it. Close this
fixed family without retuning, a forward trial or promotion. No student task
remains active. See `docs/causal_student_result_v1.md` for costs, recovery,
artifacts and limits. The practical improvement goal is still unachieved.

The next prospective study is identity-anchored class-bias calibration of the
complete native M2 probabilities. Its NumPy reference implementation and five
synthetic math/control tests pass. The fixed protocol is
`docs/native_bias_calibration_protocol_v1.md`; no real calibration fit or new
purchase score has run. This is supervised recalibration, without assuming
identified label shift between patches. Displayed purchase accuracy remains
the primary success criterion; calibration or optimizer convergence alone is
not a benefit. The bounded-data/GPU worker, native decoder adapter, scoring and
both audits are now prepared and compile; three additional protocol tests pass.
The synthetic CPU/GPU parity test and real runs await an idle machine. At the
latest check a League process was active, so the guard prevented GPU use and no
calibration task was registered. Verify current process/resource state first.

Status: active research. W1 trained on 4,002 games, calibrated on 1,000, and
evaluated on 998 later development games. Its snapshot model improves paired
log loss over the logistic baseline by 2.05%; the history extension adds only
0.23%, below the declared useful-signal threshold. These are developmental
prediction results, not better-purchase or coaching evidence.

The short-horizon retrieval model loses to selecting the latest two eligible
windows. Rich resource-allocation features show a small forecast signal but
do not clear the joint prediction gate. Current work moves to decisions:
audited purchase/undo/replacement pairs for a reference-policy IPO adapter,
and complete retained-basket trajectories for IQL. The fixed IPO recipe was
fitted and independently audited: 53.53% preference accuracy versus 54.49%
for M2 on 158 pairs from 144 development games; no useful gain. The complete
IQL/BC GPU experiment completed on392,448 transitions from4,001FIT games.
On the158-pair/144-game exposed development challenge,IQL scored53.21%,M2
54.49%,and ordinary20-epochBC59.62%. IQL fails its gate; BC's exploratory
+5.13pp overM2 has95% interval[-0.63,+10.56],so it is not a confirmed gain.
See `docs/ipo_result_v1.md`, `docs/iql_experiment_protocol_v1.md`
and `docs/research_progress_20260929.md`. The broader goal remains unachieved.

The subsequent frozen ordinary-shop trial completed on859 development games.
Reranking unchanged native M2 candidates by BC20 reduced main-basket top1
agreement from52.85% to49.13%: -3.73pp, paired95% interval[-6.17,-1.28]. The
primary gate fails. Direct BC is47.85%; the fixed geometric secondary is53.78%
but its+0.93pp interval[-0.82,+2.68] is unresolved. No candidate is promoted.
See `data/runs/ordinary-evaluation-20260930-v1/experiment/report.json` and
`docs/bc20_ordinary_eval_protocol_v1.md`. All1,911 confirmation games remain
reserved. The completed history-information probe is described below.

Independent audits confirmed the ordinary-policy result and its economic
diagnosis. All108 BC catalogue-top3 target hits removed by the native decoder
become feasible when only the supplied reconstructed wallet increases; both
patches give identical thresholds, median69.5 additional gold. This identifies
an offline state/evaluation limitation, not actual historical wallet error or
an improved policy. The live adapter already reads reported currentGold.
See `data/reports/ordinary_evaluation_audit_v1.json` and
`data/reports/ordinary_wallet_diagnostic_audit_v1.json`. No wallet override or
policy promotion is justified by this diagnosis.

The proposed history experiment has now been implemented, run and audited.
Across3350 fixed whole-game OOF decisions, state-only accuracy is43.91%, true
purchase history44.24%, and timing-preserving shuffled history44.27%.
History adds0.33pp versus state (paired95% interval[-0.87,+1.52]) and reduces
NLL by1.42% (interval[-0.11,+2.90]); it fails the fixed useful-signal gate.
All six20-epoch models and92040 updates completed. See
`data/runs/history-probe-20260930-v3/experiment/report.json`,
`data/reports/history_probe_v3_audit.json`, and `docs/history_probe_protocol_v1.md`.
This closes that recipe without a recommendation or win-benefit claim.
The legal M2/BC candidate union was also screened: its oracle covers606/859
ordinary targets versus591/859 for M2, so even a perfect selector adds only
1.75pp. A fixed-three shortlist from that union cannot meet a3pp gain; that
route is rejected before fitting or scoring a new policy.

The audited future-completion census covers all392448 FIT states:178490 have
a qualifying strictly-later completion within six minutes. The subsequent
training-only auxiliary-supervision experiment is complete and negative.
Across3350 fixed OOF decisions, matched main-only accuracy is44.3881%, real
future supervision43.3134%, and shuffled future supervision44.1194%.
Real supervision loses1.0746pp versus main-only (paired95% interval
[-2.3881,+0.1493]pp), and increases NLL by2.3730% (interval[0.9544,3.7443]%).
All six20-epoch models/92040 updates completed; the fixed useful-signal gate
fails. Full independent original-batch GPU replay exactly matched all10050
neural predictions after diagnosing small CPU/batch floating-point differences.
See `docs/future_goal_protocol_v1.md`,
`data/runs/future-goal-probe-20260930-v1/experiment/report.json`, and
`data/runs/future-goal-result-audit-20260930-v2/audit.json`.
This recipe is closed; no policy is promoted and all confirmation games remain
reserved. Future labels supplied during training did not establish useful
immediate-purchase information or the requested breakthrough.

A subsequent outcome-blind temporal census found307 pure focal first-component
initiations, below its fixed500-case minimum; per-fold MR support also failed.
There are301 recorded traces to defensive boots and272 to other items, so a
generic component cannot simply be treated as an observed boot intention.
No new boot-value model was fitted. See `docs/commitment_census_protocol_v1.md`
and `data/runs/commitment-census-20260930-v1/census/report.json`.

Privileged future-state distillation for WIN probability also completed and
was independently audited. Eight FIT-only cross-fitted teachers and three
current-only students preserve the original4002/1000/998 game partitions.
The future-trained student scores log-loss0.495306 versus0.494341 for the
strong frozen rich classifier. Its relative reduction is-0.1952%, paired95%
interval[-0.7368,+0.3585]%; the useful-signal gate fails. A2.3265% log-loss
gain over the weaker hard-target regressor does not beat the existing model.
See `docs/wp_distillation_protocol_v1.md`,
`data/runs/wp-distill-20260930-v1/experiment/report.json` and
`data/reports/wp_distillation_audit_v1.json`. No candidate is promoted.

The current-combat screen also completed and was audited. Complete current
health, wallet and reported-stat features were extracted for the SAME6000
games/138831 frames. Health-only log-loss is0.494480; fuller combat/resource
pooling reaches0.489674 versus0.494341 for the reference. Its0.9442% relative
reduction has paired95% interval[-0.3339,+2.2575]%, below the fixed2% gate and
unresolved. The>=3minute-to-end stratum is also unresolved. No candidate is
selected. See `docs/current_combat_protocol_v1.md`,
`data/runs/current-combat-fit-20260930-v1/experiment/report.json`, and
`data/reports/current_combat_audit_v1.json`.

The fixed full-pool scale study completed and passed its independent audit.
It uses11468FIT/2866calibration games and evaluates primarily on1863development
games outside the old pilot. The full observed-combat model lowers log-loss
from0.482556 to0.478833 against the stronger frozen combat reference:0.7715%
relative reduction, paired95% interval[0.0704,1.4621]%. Brier also improves,
but the declared2% gate fails; no model is selected or promoted. Recalibration
alone does not explain the gain. The result is a small developmental forecast
improvement, not a breakthrough, confirmation or practical decision benefit.
See `docs/wp_scale_protocol_v1.md`,
`data/runs/wp-scale-fit-20260930-v2/experiment/report.json` and
`data/reports/wp_scale_result_audit_v2.json`. The v1 execution failed before
fitting/comparison because its array-stride-sensitive copy identity check was
incorrect; the preserved v2 fix changes no scientific recipe or gate.

The joint-player representation study also completed and was independently
audited. All12models (four arms,three seeds,30epochs each) used the same11468FIT
games and2866calibration games. On2861previously exposed development games,
the strong full-combat reference scores log-loss0.482955; global neural0.491003,
learned set pooling0.491621, attention0.510100 and position-shuffled attention
0.508116. Set pooling increases log-loss1.7942% (paired95% interval
[0.7993,2.6826]%); attention increases it5.6206% ([4.3063,6.8753]%). Real
positions do not beat the matched shuffle. No candidate is selected; this
fixed neural recipe supplies no useful gain. It does not prove that joint
player information is useless under every representation or learner.
See `docs/wp_player_relations_protocol_v1.md`,
`data/runs/wp-players-fit-20260930-v1/experiment/report.json`, and
`data/runs/wp-players-result-audit-20260930-v2/audit/report.json`. The auditor
functionally reproduced1597716calibration/development probabilities exactly,
recomputed all four calibrators and verified47summaries/30paired contrasts.
All confirmation games remain closed. A Gaussian proximity-field prototype
has synthetic tests only; no subsequent kernel-model experiment has run.

Evidence lives under `data/runs/w1-pilot-20260929-v3/` and
`data/runs/setback-20260929-v1/`. W1 v2 is quarantined; do not use its metrics.

## Development

Use this project's own `.venv`. Run `python -m unittest discover -s tests`.
Only bounded synthetic tests run in an agent shell. Data extraction and fitting
run through the resource-gated, one-shot launcher after existing jobs finish.

The original report index and frozen project code are outside the write scope.
Versioned source-owned exporters in build_tracker provide research artifacts.
The authorized macro_coach review integration consumes copied curve artifacts;
it does not import this project's runtime or access its raw data.
