# The complete student remains substantially worse than native M2

2026-10-04. Six final models, native decoding, both legality worlds and both
independent audits completed. Neither declared candidate qualifies. Close this
fixed family without retuning, a forward859trial, control promotion or serving.
The practical improvement goal remains unachieved.

## Matched observed-purchase result

Population:2900original distinct-game focal decisions from3518FIT16.17games;
618original focals unavailable. This is an exposed retrospective internal
comparison with frozen native16.18M2, not fresh or forward confirmation.
The correct answer is the recorded retained purchase multiset. Quantities are
preserved; main scoring strips only the declared control wards.

Displayed main first exactness, same decisions and unchanged native decoder:

- M2:1483/2900,51.1379%.
- Hard-label student:1003/2900,34.5862%.
- Distilled student:1032/2900,35.5862%.
- Shuffled-teacher diagnostic control:909/2900,31.3448%.

Distilled minus M2 is-15.5517percentage points, adjusted two-candidate97.5%
interval[-17.5172,-13.6897]pp. It corrects116M2errors and breaks567M2successes:
451fewer correct first purchases. Hard minus M2 is-16.5517pp, adjusted interval
[-18.4836,-14.6207];116corrected and596broken,480fewer successes.

Main top3:1959/2900M2 (67.5517%),1569hard (54.1034%),1596distilled (55.0345%)
and1451control (50.0345%). Distilled loses12.5172pp to M2 on this endpoint,
ordinary95% interval[-13.8966,-11.2759]. Full-purchase first counts are1106M2,
726hard,748distilled,655control; typed-main counts1383/904/932/808respectively.
Empty-main is not renamed full no-buy: it also contains real ward-only targets.

On2736nonempty-main purchases, distilled loses16.3743pp to M2,95% interval
[-18.0958,-14.7099]. Hard loses17.4708pp[-19.2257,-15.7084]. This is not a
result driven only by the empty-main class.

Catalogue NLL (lower is better):1.536004128M2,2.803743168hard,
2.625575244distilled and3.133880872control. Distilled-M2 is+1.089571116,
95% interval[1.032701429,1.146565331]. First, nonempty, top3 and NLL all fail
the predeclared benefit/no-worsening criteria; native legality passes.

## What the teacher comparison does and does not establish

Distilled exceeds hard by1.0000pp on first exactness,95% interval
[0.0690,1.9319], and shuffled_teacher by4.2414pp[2.9310,5.5517]. The within-
student teacher diagnostic is positive, conditional on these fitted models
and this exposed cohort. It does not establish a useful improvement over M2:
the best student remains15.55pp behind the reference.

The architecture used public ten-player attention, shared count-aware item
embeddings and nonlinear state/action interactions, with196545parameters per
model. Hard CE used the actual purchase. Distilled and shuffled arms optimized
.5*(actual-purchase CE + KL(teacher||student)), temperature1. The teacher was
training supervision only; no teacher probability or current label entered
student prediction. Shuffling stayed within TRAIN champion/three-minute bins.
No final role, outcome or future build was a student inference input.

Two game folds,20epochs,Adam3e-4,batch128,identical per-fold initialization/order
and dropout seeds, and fixed final-epoch checkpoints were declared before
scores. The56288selected states supply28112or28176training states per fold.
All six retained fits total26460updates. No checkpoint, learning rate, loss
weight or architecture was chosen after viewing this result.

This comparison does not isolate why the student loses. Training volume,
patch, features, architecture and optimization differ from native M2's training
regime. It does not prove that attention or distillation generally fails, nor
that more data or more epochs would close the gap. Those explanations and
remedies were not tested here. The verified architecture difference from older
own-project proxies was never an established explanation for their M2 gap.

## Reproduction, legality and interruption recovery

The model audit verifies all8700student forecasts exactly using the original
128-row batches and public-only inputs. It verifies2900baseline priors,
26460training-batch permissions,56288rejected held-out supervision requests,
seeds,folds,orders,optimizer receipts and checkpoint hashes. It does not perform
a second complete26460-update training run.

Unchanged native M2 options replay exactly, with11600forbidden-field mutation
checks. Both16.18and separate reconstructible16.17 audits find zero infeasible
options:8488M2,8486hard,8499distilled,8460control. No option is filtered or
replaced after an audit. All models use the same9333-class main catalogue.

The original process stopped after five complete models and one epoch of the
last control. On later inspection its handles were absent and Windows had
rebooted; exact termination cause/time is unknown. Its stale running JSON was
preserved. Recovery v3 preserved all five finished checkpoint files byte for
byte and recomputed only the unsaved fold1control from its original seed.
The minibatch-update AST matched the original. The lost epoch's three losses,
order,counts and221updates reproduced exactly; elapsed time alone differed.
All six models froze before any held-out prediction or purchase score.

The scoped NumPy metadata loader retained weights_only=True and normalized
string metadata only in memory. No unrestricted checkpoint loading or weight
editing occurred. The earlier metadata-only prediction recovery v2 was not run.
See causal_student_interruption_amendment_v3.md for the declared exception.

After the model and native stages, the final metric audit recomputed every
metric,interval,gate and selection, and independently counted all displayed
first/top3/full/typed matches. Both audits pass; the scientific success gate
fails and selected_candidate is null.

## Cost, artifacts and limits

Retained-model fit time:5128.079seconds (85.468minutes). At least221real updates
and42.819seconds from the interrupted control were discarded; possible later
partial updates are unknown. Do not describe retained time as exact total
compute. Recovery worker time987.888seconds includes just one recomputed model,
preserving the other five and generating predictions, not six fresh fits.

Model replay69.457s; native18decode105.193s; native17audit15.673s;
scoring78.348s; final metric audit78.240s. CPU1, host2.5GiB startup/.75GiB live,
and idle CUDA gates remained in force. Assessment reported complete at20:34:17UTC;
later inspection found no remaining student task or worker.
Mean legal first-option incremental gold cost:696.07M2,634.01hard,621.32distilled,
528.87control in both worlds. These costs do not establish purchase quality.

Key immutable artifacts under this project unless prefixed with ../build_tracker:

- Protocol docs/causal_student_probe_protocol_v1.md,
  SHAfc3168bffd7865eb9ab53a18450501d8732474cf8df8077f16e61ffac7abbd5e.
- Training/predictions data/runs/causal-student-recovery-20261004-v3/experiment/report.json,
  SHA0f869f5947d0103e02e5df9c28109b80fa2f14e6f5ae97283a5e806146136716.
- Distribution manifest SHA237fe8b1bfd9ce1d9c86f5a3c8a2d6406cd1d79bd2a27e6d80bfbce0e65dd8af.
- Evaluation data/runs/causal-student-assessment-20261004-v1/evaluation/report.json,
  SHAf369133d8ef2d3d22972679beca271e51d4a0e36035636854077610ff00cb00d.
- Model audit data/runs/causal-student-assessment-20261004-v1/model_audit/report.json,
  SHAbbfe4b0f96279579ad739e8bd6d51d8675dd5d443404e7b5d2347f1aa06b55a7.
- Final audit data/runs/causal-student-assessment-20261004-v1/audit/report.json,
  SHA294f40aef5099ad3bda14e7ae999430ad7ea51bc234b769bb1c234368f5d4bd8.
- Native manifests under ../build_tracker/data/exports/winprob-causal-student/
  causal-student-assessment-20261004-v1: decoded SHA4b9c7fe74d01c4a2a1a19543b8544fa9a84ff5df274b9724072c18bfd8e91d92,
  native17 SHAa728f2cdbd0052a664de288015c69ef06ffc921feabf8fd34d35a719ebe8099b.

Intervals use2000paired whole-game resamples,seed20261004; one original focal
per game. Primary candidate-M2 comparisons use97.5% intervals for the two
candidates; other intervals are95%. They are conditional on fitted models and
do not include training uncertainty or correct the full prior research history.
M2 retains its original offline-role/live-input limitation. The sample is
conditioned on the existing complete, catalogue-supported episode cache.
No forward859trial,1911confirmation/source/global sealed read, new collection,
source database query or policy promotion. Original20260929index SHA
94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46unchanged.

Decision: close this fixed student family. Preserve M2 as the stronger measured
reference. Do not count teacher-specific learning, successful recovery or
reproducibility as the requested practical breakthrough.
