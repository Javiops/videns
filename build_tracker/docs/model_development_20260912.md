# Model development for September 12, 2026

## Owner authorization and time boundary

At approximately 16:17 Europe/Madrid the owner asked Codex to take on
development for the rest of the day and keep them updated on decisions.
The owner explicitly selected **Improve model behavior**. This authorizes
evidence-driven diagnostics, code changes, tests and isolated model/decoder
experiments. It does not authorize model promotion, final-test access, commits
or pushes, external messages, paid services or shutdown.

Work ends by 23:59 Europe/Madrid on September 12 (21:59 UTC). The heartbeat
`build-tracker-model-development-today` continues this same task every 30 minutes
within that day. Its last useful wakeup must summarize and pause; at or after
23:30 local, finish documentation and verification instead of starting another
expensive experiment. Do not forcibly terminate an already authorized safe
job solely for the clock; report any job still running at handoff.

Source checkout for new work:
`C:/Users/Javier/build_tracker/.worktrees/model-behavior-20260912`.
It was based on today's completed benchmark implementation, with data and
venv junctions to the existing local directories. Keep the completed
`.worktrees/benchmark-16-18` checkout and its code/report bindings unchanged.
Main-checkout source is also unchanged by this new work so far.

## Starting evidence

The completed `16.18-reconstructible-development-v1` dataset has 722 training
games / 86,488 rows and 151 reused development-validation games / 18,041 rows.
Its export fingerprint is
`fef1eb63d0d524f33fb85292ee302da64cb7502df09f41dc01cbf42594709f6e`.
The 1,308 later qualified games remain outside this original temporal benchmark.
The final test remains sealed and is not materialized here.

Candidate SHA256:
`35bc20c781cb1f49903bd39bb98bd5d6f21f5757121cb313f37bcce9482121a9`.
The candidate is permanently development-only under its recorded provenance;
new evidence must not relabel it as release eligible.

The transformer beat a train-only conditional policy on this identical
validation population: `first_action_exact` 0.3304140569 versus 0.1997117676,
n=18,041; `action_family_match` 0.5999249390 versus 0.3856883718, n=15,987
applicable purchases. Both paired differences were positive with their
predeclared intervals above zero. Source for those metrics:
`data/experiments/16.18-reconstructible-development-v1/policy_eval_val_candidate_vs_conditional_policy.json`.

The one `count-tail-075-v1` copy-rule trial failed: validation exactness fell
from 0.3304140569 to 0.3291391830 on the same n=18,041, while excess additional
copies fell 559 to 445 and missing additional copies rose 697 to 758.
Source: that run's `decoder_eval_val.json`. Keep `argmax-v1` as the default.
Do not continue searching copy thresholds after this negative result.

All original collection/recovery stages completed at 13:26 UTC, including all
4,432 selected non-test refetch pairs. This is source recovery, not permission
to add those games to any frozen export. Check actual jobs again before work;
do not race or stop Riot collection and preserve its dated sample follow-up.

## Decisions

### D1 — Focus on item selection; retain original quantity decoding

The completed exploratory analysis shows many overlapping missing and extra
item errors. Such flags do not establish simple underspending. The first
follow-up targets how item identities are selected, rather than further
tuning quantity thresholds. Persist the action-error partition and secondary
save/ward slices in
`data/reports/model_day_20260912/baseline_error_attribution.json`, bound to the
original report and paired-outcome hashes. These slices are exploratory and
do not receive invented confidence or causal interpretations.

### D2 — First controlled experiment: target-head purchase filter

The displayed decoder normally admits purchases from the recipes of the four
highest-ranked predicted final items, except for high-scoring purchase-head
overrides. This auxiliary target head can therefore veto the direct purchase
head. Test that concrete mechanism with one fixed-checkpoint ablation.

Planned name: `16.18-target-filter-ablation-v1`.
Control policy: `top4-gate-v1`.
Experimental policy: `direct-purchase-v1`, removing only that target-recipe
purchase restriction. Retain target arrows, purchase ranking and thresholds,
original count decoding, all save gates/fallbacks and every affordability,
recipe, slot, stack, boot-family and footwear rule. Keep the same checkpoint,
dataset and all 151 validation games. No retraining or parameter sweep is part
of this comparison.

The new evaluator must reproduce the control's exact per-decision outputs and
primary metrics from the completed original baseline report, and fail if its
population differs. Bind the parent completion, trained checkpoint/config,
generation, report/outcomes and original code root separately from the new
experiment code. Never rewrite the checkpoint's old training plan/provenance
to fit a new experiment directory.

Use the two existing primary metrics and populations:
`first_action_exact` on all decisions and `action_family_match` on purchases
with a known next final. Use paired game bootstrap, 400 resamples and seed 16.
Record save precision/recall and ward recall as secondary tradeoffs. Zero
illegal or unaffordable displayed options is mandatory. A provisional supported
development win requires favorable paired intervals on both primaries; report
secondary regressions explicitly before choosing whether the change merits
keeping. Negative and inconclusive results end this specific experiment without
threshold shopping. Reused validation is not confirmatory or live evidence.

### D3 — Instrument hold reasons alongside the comparison

Current policy contains both a learned lead-save gate and an unconditional
save fallback when its purchase decoder produces no basket. Existing outcome
captures cannot distinguish these causes. Add optional diagnostic traces for
target-gate rejections, internal no-buy score and learned-save versus fallback
selection. Trace collection must not change any player-facing output or expose
ranking scores as confidence. All training labels remain stripped before
inference; traces explain the decision rather than supply information to it.

After this result, choose the next bounded experiment from the measured cause
and remaining time. A save-fallback singleton is a possible later hypothesis,
not an approved result or a second simultaneous change to this experiment.

### D2 result — Retain the original target filter

The target-filter comparison completed at 15:05 UTC and its completion,
parent/code/output bindings were independently verified in that run's
`final_verification.json`. Removing the filter raised `first_action_exact`
from 0.3304140569 to 0.3323540824, paired difference 0.0019400255, CI95
[0.0006254620, 0.0033793142], development-validation n=18,041/151 games.
`action_family_match` changed from 0.5999249390 to 0.6001751423, paired
difference 0.0002502033, CI95 [-0.0013647643, 0.0018574701], n=15,987 applicable
validation purchases. It did not satisfy the predeclared two-primary win rule.
Post-death no-buy recall fell by 0.0122699387 (n=326), and ward-only recall
fell by 0.0055325035 (n=723). Source for all these metrics:
`data/experiments/16.18-target-filter-ablation-v1/target_filter_eval_val.json`.
Keep the original default filter; preserve the positive exactness finding and
secondary tradeoffs without promoting the ablation as an overall winner.

The original policy's trace attributes 1,021 lead holds to unconditional
fallback (130 match recorded no-buys) and 116 to the learned gate (31 match),
over the same n=18,041 validation decisions. Source: that run's
`target_filter_trace_summary.json` and bound outcome capture. The additional
`data/reports/model_day_20260912/fallback_scope.json` shows that only a narrow
subset of fallback decisions has a legal target-filtered proposal above the
existing floor. A narrower retry could exchange some correct no-buys for
purchase matches; it would not explain the bulk of the remaining error.

### D4 — Measure mechanical target feasibility before a larger learning change

The owner asked why exact accuracy is low. The next diagnostic measures which
recorded baskets are representable by the current label vocabulary, quantity
and basket caps, buy-only purchase rules, and reconstructed decision wallet.
This is an oracle using labels only for offline diagnosis; it must never feed
labels into a model or manufacture different targets to improve the headline.

Use the same fixed 18,041 validation decisions and original checkpoint metadata.
Partition all decisions explicitly into no-buy targets, missing vocabulary,
unsupported quantity/basket capacity, no legal buy-only sequence even with an
unlimited diagnostic budget, wallet-limited sequences, and mechanically
representable targets. Preserve overlapping reasons separately. Use the actual
recipe/slot/footwear rules and search legal orderings of each target multiset.
An unlimited-wallet witness is not automatically a minimum-cost sequence.
Every previously exact original-model prediction must classify as representable;
otherwise the diagnostic itself must fail before making any claim.

Report a mechanical representability bound, not a Bayes-accuracy ceiling or a
measurement of gold-estimator error. A basket outside the current budget may
depend on unmodeled sales or income during the visit; it does not prove that
the causal estimator is wrong. No new inference or training is part of this
audit. Join the existing outcome/hold traces to locate remaining errors among
representable and unrepresentable targets, then select the next learning change
from that evidence. Do not perform another small fallback replay first.

Implementation checkout:
`C:/Users/Javier/build_tracker/.worktrees/model-feasibility-20260912`.
Planned run: `data/experiments/16.18-action-feasibility-v1`, with a 30-minute
wall budget, 4 GB child memory cap, 40 GB disk reserve and the same day boundary.
The completed model-behavior checkout remains unchanged.

### D4 result — A substantial compatibility limit and substantial remaining errors

The diagnostic completed at 15:53 UTC and passed independent final verification.
Across validation n=18,041 / 151 games, 4,544 recorded baskets require more than
the supplied `gold_est` and 626 have no legal sequence under the current
buy-only action rules even with an unlimited diagnostic wallet. Vocabulary and
quantity-capacity exclusions are both zero. The remaining 12,871 actions
(12,545 purchases plus 326 no-buys) are mechanically representable, a loose
71.3431% envelope. The original model still makes 6,910 errors within that
envelope. Source for every count and derived fraction:
`data/experiments/16.18-action-feasibility-v1/action_feasibility_val.json`.
The original exact-prediction invariant passes with zero contradictions.

This explains part of the low exact-match score without changing that score
or its denominator. It does not identify exact gold errors: sales and income
during a shop visit are outside this buy-only snapshot replay. It also does
not prove every feasible mismatch can be learned from the available inputs.

### D5 — Matched purchase/count head learning

The next experiment targets the remaining feasible item-set errors through
learning. Keep the original checkpoint, encoder representation, save head,
target head, inference configuration, item vocabulary and every decoder rule
fixed. Update only `head.*` and `count_head.*` parameters.

Use all original TRAIN rows for the supervised anchor in both child models.
Mine additional comparison baskets only from TRAIN rows whose recorded target
is mechanically representable. Use the original real `predict_options` on
label-stripped strict-wallet state, retaining up to three distinct legal
alternatives that differ from the recorded physical basket. A SAVE alternative
is the empty physical basket for ranking. Infeasible targets are excluded only
from this additional ranking term; they retain their original supervision.
No validation negatives or validation-based threshold choices are allowed.

Cache deterministic encoder features from the actual parent inference path.
Prove that attaching labels for loss targets/masks does not change encoder
inputs. Bind the complete all-TRAIN row population, features, targets, loss-only
legal masks, negative baskets and frequency-derived weights into an immutable
cache. The original legal mask permits 500 gold slack and restores positives;
this applies to the loss only, never to feature extraction or displayed buys.

Two child recipes start from the identical parent and use identical row order,
three epochs, batch size 256, seed 16, Adam learning rate 1e-4, zero weight decay
and gradient clipping at 1.0. The supervised control uses the original weighted
per-label mean BCE plus 0.25 positive-count cross-entropy. Its matched candidate
adds 0.1 times a per-row basket-ranking loss, averaged over valid mined
alternatives and then all batch rows. The basket score uses physical-item
presence logits plus conditional count log probabilities; common absent-item
terms cancel in each comparison. These are internal ranking scores, not
calibrated player-facing confidence.

Keep the encoder in deterministic evaluation mode. Prove every tensor outside
the purchase/count heads remains identical to the parent in both artifacts.
Inherited inference configuration remains byte-identical; new provenance records
the actual child training plan, parent, cache, optimizer, epochs and arm.
Do not relabel the children as the old graft or overwrite its training history.

Compare original, supervised-control and contrastive-candidate policies on the
same full 18,041 validation decisions. Verify exact original parent replay.
Report game-paired original-versus-each-child and contrastive-versus-supervised
effects on both primary metrics and all save/ward tradeoffs, with zero illegal
or unaffordable outputs. A child development win requires favorable intervals
on both primaries against the original; a contrastive-specific claim additionally
requires both primaries to improve against its matched supervised control.
Keep mixed/negative results and do not sweep epochs, loss weights or thresholds
after seeing this comparison.

Implementation checkout:
`C:/Users/Javier/build_tracker/.worktrees/model-basket-learning-20260912`.
Planned run: `data/experiments/16.18-basket-head-learning-v1`.
Stages are tests, TRAIN mining, supervised heads, contrastive heads, three-arm
validation and summary. Budget: 3 hours, 10 GB child memory, 40 GB disk reserve,
one expensive child at a time, with enough time to finish before 21:59 UTC.
No real mining, new training or model result is claimed at this planning entry.

### D5 launch

At 16:39 UTC, hidden task `BuildTracker 16.18 Basket Heads V1` started from the
basket-learning checkout. The fixed plan SHA256 is
`ba4a72f976c7fc6e1f91e65ec9e97caf731e20a7eab4b678a2e4a66a999409f0`.
Focused miner/loss/cache/training, evaluator and supervisor tests passed, and
independent review found no remaining blocker. The task runs the complete
assert suite before actual TRAIN mining and both head-training arms. Both head
arms explicitly use one CPU thread for deterministic small-matrix training.

Read `data/experiments/16.18-basket-head-learning-v1/status.json` for actual
progress; no new trained model or comparison result is claimed at launch.
Do not edit or add code in this checkout while its full code inventory is
pinned. Mining retains all 86,488 TRAIN rows for the supervised anchor and
records the observed mining rate and comparison eligibility counts. Earlier
experimental checkouts and all source/held-out evidence remain unchanged.

At 16:44 UTC, all **59/59** assert scripts had passed and TRAIN mining was
advancing. The inspected checkpoint reported 7,000 of 86,488 rows processed;
this is preparation progress, not a training or model-quality result. Follow
the run's live status for subsequent stages and observed rate.

### D5 result — Neither head variant is adopted

The matched learning run completed at 17:41 UTC. All cache, stage, checkpoint,
frozen-tensor, configuration and report bindings passed independent verification
in that run's `final_verification.json`. The original model remains the
reference. On validation n=18,041 / 151 games, exact-action agreement is
0.3304140569 for the original, 0.3274208747 for supervised head fine-tuning,
and 0.3319660773 for the contrastive variant. Contrastive-minus-original is
0.0015520204, CI95 [-0.0024560201, 0.0051597870]. On n=15,987 applicable
purchases, family match is 0.5999249390, 0.5961718897 and 0.5797835742,
respectively; contrastive-minus-original is -0.0201413649, CI95
[-0.0232587778, -0.0172413793]. Neither child meets the predeclared win rule.
Source for all metrics:
`data/experiments/16.18-basket-head-learning-v1/basket_heads_eval_val.json`.

Contrastive training improves exactness against its matched supervised control
but loses family match. Purchase precision rises while recall falls, consistent
with more selective recommendations under unchanged thresholds. This is not
evidence that further loss-weight or epoch tuning will help. Keep both negative
results and the original configuration; no model is promoted.

The subsequent read-only postmortem reconciles 547 lost and 225 gained family
hits on the same 15,987 applicable validation decisions. The losses partition
into 339 replacement/mixed baskets, 165 nonempty deletion-only baskets and
43 fallback SAVEs; the last two groups together are 208 deletion-only cases.
Of all lost hits, 348 have fewer lead copies and 521 previously matched via
a nonfinal family member. Only 17 coincide with increased missing true wards.
Source: `data/reports/model_day_20260912/d5_postmortem.json` and `.md`, with
original report/outcome hashes verified before and after. These descriptive
transitions do not identify why training changed particular scores.

The temporary Windows task was removed after confirming successful exit. Its
frozen wrapper's default task name differed from the registered name, so manual
cleanup was necessary; the wrapper and completed evidence were not rewritten.
Future launchers must receive their exact registered task name explicitly.

### D6 — Source accounting for the incompatible labels

The next diagnostic inspects the 5,170 validation targets classified by D4 as
wallet-limited or incompatible with buy-only rules. It will bind and join
their original frozen corpus events and raw timeline references, using only
the known validation IDs and exact decision keys. It does not score a model.

Record observed sales, undos, recipe destruction, role/support/automatic
transforms, consumable events and ward events separately. An inventory's
negative delta is not a sale label; a destroyed item is not automatically a
freed slot. Event flags are observations, not sufficient causal explanations.

Use deliberately narrow conditional sell-first checks: only ordinary observed
sales of items present at the beginning, with no undo, no sold-item/target-ID
overlap and no ambiguous role/support/effective-ID transformation. Never cap
sold counts silently or credit newly acquired items. Preserve starting role,
quest and footwear state. Remove eligible sold items, recalculate modeled and
hidden-slot inventory from the full source state, then search for the unchanged
target multiset with an unlimited diagnostic wallet, the original gold_est,
and that same gold_est plus reviewed sale credits. Preserve every target copy.

A positive result proves a conditional sell-first route for the retained
target basket, not the observed transaction order, necessity, every original
transaction, exact starting gold or a new recommendation quality result.
Undo, transformation and income cases outside these restrictions remain
explicitly unresolved. Do not use a future-frame wallet or future transactions
as model inputs, and do not modify the original reconstructed examples.

Implementation checkout:
`C:/Users/Javier/build_tracker/.worktrees/model-source-accounting-20260912`.
Planned run: `data/experiments/16.18-source-accounting-v1`, with 30-minute/4 GB/
40 GB limits and the same day boundary. Source flags and conditional rescues
retain the full n=18,041 validation denominator as well as subset denominators.

### D6 launch

At 18:36 UTC, hidden task `BuildTracker 16.18 Source Accounting V1` started
from the source-accounting checkout. The frozen plan SHA256 is
`116ad23864767f5a1c076092923ae9950547c62e095cae818a616ab151918a65`.
Twelve focused worker tests and six supervisor fault tests passed, and
independent review found no blocker. The scheduled sequence validates its
input lineage, runs the entire assert suite, then performs the diagnostic.
Full-suite completion and source-accounting findings are not yet claimed.

Actual progress: `data/experiments/16.18-source-accounting-v1/status.json`.
Keep this checkout's complete application/script inventory frozen. The exact
registered task name was passed explicitly to the wrapper for cleanup.

### D6 result — Sales explain a limited, conditional subset

The diagnostic completed at 18:40:52 UTC, after all 61 assert scripts passed
in 112.7 seconds. The completed supervisor's no-op verification rechecked
historical/current source code, source bytes, stage receipts and output hashes;
the validation generation and independent row/witness arithmetic also passed.
Final receipt: `data/experiments/16.18-source-accounting-v1/final_verification.json`.
Report: `data/experiments/16.18-source-accounting-v1/source_accounting_val.json`,
SHA256 `03055cc767730332836c285433b0e380f228ac3fcd1c1be09248da60f9c25e52`.

On validation, the full denominator remains 18,041 decisions / 151 games;
exactly 5,170 D4 failed targets are diagnosed. Of these, 399 pass the narrow
ordinary-sale screen: 131 have a conditional route at unchanged gold_est,
162 more have a route after the reviewed sale credit, and 106 remain
wallet-limited. The other 4,771 are not assessed under these conservative
restrictions. Of the 626 original buy-only-rule failures, the respective
counts are 131, 76, 41 and 378; among 4,544 supplied-wallet failures, they
are 0, 86, 65 and 4,393. Every failed key and target count reconciles.

Observed sales appear in 971 of the 5,170 failed visits; undos in 262,
role-quest events in 458 and recipe-related destruction in 2,823. These
overlapping flags are observations, not proof that any one event caused a
failure. Successful routes establish conditional sell-first feasibility
only. The diagnostic supplies neither exact starting gold nor new accuracy.
The original inputs and all earlier evidence remain unchanged.

Decision: do not broaden source reconstruction or add an automatic sale
policy based on this result alone. Its proven conditional routes cover a
limited subset and leave most failures unresolved. Assess the existing
displayed alternatives next: determine whether ranking them better has
measurable headroom before committing to another learning experiment. This
assessment reads saved development outcomes only; no new model run has
been selected at this checkpoint. Preserve D5's negative findings and avoid
further loss-weight, epoch or threshold tuning.

### D7 — Rank the original BUY alternatives without changing their contents

Decision at 18:49 UTC: implement one fixed-option ranking experiment. The
D5 postmortem (`data/reports/model_day_20260912/d5_postmortem.json` and `.md`)
shows a reduction in recommended copies and loss of component-family hits.
Saved original outcomes also show opportunities where the observed basket
is among the displayed alternatives but is not first. That is oracle
headroom, not evidence that a learned ranker will improve validation.

The bound headroom report is
`data/reports/model_day_20260912/option_ranking_headroom.json` and `.md`.
On the original full validation (18,041 decisions / 151 games), the proposed
scope covers 11,656 decisions. It contains 1,427 potential exact purchase
rescues (purchase denominator 17,715) and 1,652 potential family rescues
(family denominator 15,987). These are separate oracle opportunities, not
a jointly achievable gain. There are also 3,279 currently exact eligible
purchase choices and 4,164 currently family-correct decisions with a
nonfamily BUY alternative; reranking can lose existing correct choices.

New isolated checkout:
`C:/Users/Javier/build_tracker/.worktrees/model-option-ranking-20260912`.
Planned run: `data/experiments/16.18-option-ranking-v1`.
No real mining, training or evaluation is launched at this decision entry.

Question: can TRAIN-only listwise learning select a better first BUY basket
from the original policy's existing alternatives, improving both displayed
exact action agreement and family match? Preserve the original parent
checkpoint, encoder, configuration, export, static mechanics, target gate,
count decoder, wallet rules and every emitted card's complete contents.
Only when the original first card is BUY and at least two cards are BUY may
BUY cards be permuted among their existing positions. Every SAVE card stays
at its original position; all other decisions use the identity order.
This preserves SAVE behavior and the entire candidate set per decision.

Before freezing or any data execution, review of `web/live.html` found that
the `tier` field is positional: it renders `best` as "First line". The
permutation contract therefore has one presentation-only exception: each
moved BUY card receives the original destination slot's tier. Preserve all
other fields, quantities, prices, item order and target arrows from the
selected source card, and preserve every SAVE card exactly. Use copied BUY
objects so the original candidate list remains unchanged. Evaluation must
check non-tier BUY content and positional tiers separately. This prevents a
moved alternative from retaining a contradictory rank label; it does not
change model features, physical candidates or the learning/evaluation rule.

Use a small fixed scorer: Linear(input,64), ReLU, Linear(64,1), with a zero
final layer so initial ties preserve original order. Candidate inputs are
the frozen 256-dimensional parent context, physical item counts in the
parent vocabulary divided by max_copies, original card index divided by 2,
candidate cost divided by max(1,gold_est), and total copies, distinct IDs and
ward copies divided by max_basket_items. All are available from the original
label-stripped input and actual emitted candidates. Scorer logits are
ranking values, never player probabilities.

Regenerate complete original candidates for ALL 86,488 TRAIN rows, before
attaching labels. D5's negative cache cannot reconstruct these options:
it omitted truth-equal baskets and did not generate options for every row.
Reuse only its frozen context cache after exact row-index/hash alignment and
bitwise comparison with fresh actual parent context for every TRAIN row.
Store the complete new candidate/rank/cost records and immutable bindings;
never append the observed target as a candidate or manufacture candidates
from labels. Validation and test cannot reach the mining entry point.

Train only informative eligible TRAIN decisions: original primary BUY,
at least two BUY alternatives, at least one exact observed target and at
least one incorrect alternative. Record the complete TRAIN population and
every eligibility/exclusion count. Use loss logsumexp(all BUY scores) minus
logsumexp(exact BUY scores), preserving multiple exact alternatives. Fixed
seed 16, three epochs, Adam 1e-3, weight decay 0, batch 256, one CPU thread,
gradient clip 1, no schedule, threshold, epoch or hyperparameter selection.
The original model remains entirely frozen; ranker weights live separately.

Evaluate on the unchanged full validation: first_action_exact on all 18,041
decisions / 151 games and action_family_match on 15,987 applicable purchases.
Require exact replay of the original captured policy and game-paired
intervals favoring the ranker on BOTH primaries for a development win. Report
all tradeoffs. Assert zero illegal/unaffordable baskets, identical non-tier
BUY card content/quantities, preserved positional tiers, unchanged SAVE cards,
positions and primary action kind,
and unchanged any-three and SAVE metrics. A ranker cannot gain coverage by
changing the candidates. Reused validation remains development evidence.

Scheduled sequence: tests, TRAIN mining, ranker training, paired validation,
summary/completion. Fixed 7,200-second / 10 GB / 40 GB resource limits, one
expensive child at a time, deadline 21:59 UTC with a 60-second completion
margin; refuse to launch if the entire bound cannot fit. Use hidden task
`BuildTracker 16.18 Option Ranking V1`, passing that exact name explicitly.
Bind all historical experiment/code evidence including D6, D5's complete
cache, the unchanged source generation and the new complete code inventory.
No final-test reads, promotion, parent mutation or serving changes.

## Execution discipline

### First experiment launch

The implementation passed focused predictor parity, evaluator tamper/replay,
error-attribution and supervisor fault tests. Hidden task
`BuildTracker 16.18 Target Filter V1` now runs from the day checkout. Its
authority is `data/experiments/16.18-target-filter-ablation-v1/status.json`;
`evaluation_plan.json` freezes the parent, new code, two policies and budgets.
The sequence runs the full assert suite, then the paired evaluation and trace
report. There is no training stage. A 90-minute wall budget, 10 GB memory cap,
40 GB disk reserve and same-day finish condition apply.

Do not edit or add application/script files in this checkout while it runs:
the supervisor binds the complete code inventory as well as file hashes.
Use read-only analysis or a separate new checkout for independent preparation.
Keep completed stages and parent artifacts immutable; inspect existing status
before any attempted launch. The expected outputs are
`target_filter_eval_val.json`, `target_filter_outcomes_val.jsonl`,
`target_filter_trace_summary.json` and `completion.json` in the new run.
No result is claimed at launch.

At 14:48 UTC, the complete assert suite passed **54/54** and the paired
evaluation started. Evaluation-plan SHA256:
`d247b437a1a4a860c6b421fce3c64f248d0d957affb9abcaf9989cd15c95c6ad`.
No model-quality result is claimed at this checkpoint. The next decision should
use the final report and trace summary after their bindings are verified.

Each experiment needs a fresh immutable evaluation plan before scoring, with
exact code/input hashes, question, controls, policies, populations, metrics,
stopping rule and budgets. Write stage logs/status and final evidence; preserve
failed attempts. Run expensive stages through hidden durable Windows scheduled
tasks, one expensive model/experiment stage at a time, with at most 10 GB child
memory, at least 40 GB disk reserve, and a wall budget bounded by the remaining
day. Never load the whole corpus into Python dictionaries.

Run meaningful targeted tests and required repository checks. Once passing,
repeat only for new changes, failures or unresolved risks. Keep separate code
checkouts/snapshots when a running job pins source. Avoid parallel GPU training
or duplicated validation runs. Check report and artifact bindings independently
before reporting a result, always with metric, split, n and report path.

Update this ledger when choosing a new experiment, learning a result or
changing direction. Tell the owner what was chosen, why, and what evidence
would change the decision. Heartbeats should continue useful implementation
between jobs and remain quiet on unchanged polling.

## End-of-day handoff

### D7 result — Exact agreement gains, family match loses

D7 completed at 20:24:29 UTC. All 64 assert scripts passed. The entire 86,488
TRAIN rows were freshly mined with all original card objects recorded; their
captured parent contexts matched the frozen D5 cache bitwise for every row.
The fixed ranker trained on 29,540 informative eligible TRAIN rows. The
completed supervisor's no-op verification, explicit train/val generation and
cache/ranker checks, and independent recorded-outcome verification all passed.
Final receipt: `data/experiments/16.18-option-ranking-v1/final_verification.json`.

On the unchanged validation, first_action_exact is 0.3403359016 for reranked
versus 0.3304140569 original, n=18,041 / 151 games; paired difference
+0.0099218447, game-bootstrap CI95 [+0.0070098576,+0.0124685491].
action_family_match is 0.5892287484 versus 0.5999249390 on n=15,987 / 151
games; difference -0.0106961907, CI95 [-0.0144798679,-0.0078991913].
Ward-only recall falls from 0.7939142462 to 0.7441217151 on n=723.
Source for all values: `data/experiments/16.18-option-ranking-v1/option_ranking_eval_val.json`,
SHA256 `77304fe043bc3ac33b2e91b8d0523e6d7eb7392c4d7d93e7e8b84f45308b5040`.

The independent verifier reproduced all paired intervals and checked 18,041
unique decisions, 151 games, all card/permutation/tier and SAVE invariants,
unchanged any-three coverage, and zero displayed-wallet violations. It found
2,257 changed first-card orders among 11,656 eligible decisions. Receipt:
`data/reports/model_day_20260912/option_ranking_independent_verification.json`.

Decision: the exact gain is real within this reused development validation,
but the family loss is also clear and the ward-only tradeoff is material.
Reject D7 under the frozen both-primary rule; keep the original candidate.
There will be no ranker retraining, threshold/epoch sweep, test evaluation
or promotion today. The owner's wrap-up request supersedes further experiments.
The current served deployment remains unbound. The closeout is
`docs/model_day_closeout_20260912.md`. The day heartbeat was deleted after
the final evidence and closeout were verified; the two-night collection
follow-up remains separate.

At 19:17 UTC the owner asked to start wrapping up. This supersedes further
open-ended development today: finish and verify the already running D7,
summarize its result, and pause the day heartbeat. Do not select another
experiment. D7 launched at 19:15 UTC with plan SHA256
`c92c3f5bbebae56eb9775da25cde7e3354223efc52b96fbe6e13450cbec5ffae`;
its full suite reports 64/64 passed in 106.4 seconds. Follow the run's actual
status for subsequent stages. The provisional closeout is
`docs/model_day_closeout_20260912.md`; it must remain explicit about the
pending D7 result until completion and independent verification.

Record completed changes, tests, supported development results, negative
findings, any running jobs, limitations and the next concrete step. Pause the
day-limited heartbeat; do not create another day of work without authorization.
