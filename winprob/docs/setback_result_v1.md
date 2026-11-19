# Setback v1: reproduced correctly; no useful breakthrough established

Independent review of the completed exploratory development run. No raw
archive, sealed test, new game payload or quarantined run was read. The
experiment and its protocol remain unchanged.

The original positive against the selected cheap baseline is real, but the
history model does not resolve an advantage over logistic risk. More
decisively, an exploratory audit finds that simply selecting the latest two
eligible windows beats every fitted retrieval method. This closes the claim
that v1 established a useful new retrieval method; it does not invalidate the
underlying probability infrastructure or manufacture a new confirmatory result.

## Evidence and independently reproduced metrics

All aggregate metrics below use the **development** population of **998
games / 6,178 complete windows**, with **1,688 positive windows in 721 games**.
Each method selects at most two windows per game: **1,994 selections**. Macro
recall averages within-game recall over the 721 positive games; precision
uses all 1,994 selected windows, including games with no positive windows.
The primary report is
[report.json](../data/runs/setback-20260929-v1/experiment/report.json), and the
row-level evidence is
[development_windows.npz](../data/runs/setback-20260929-v1/experiment/development_windows.npz).

An independent NumPy calculation, without importing the experiment's metric
helpers, reproduced every persisted label from its gold/kill changes, every
top-two selection from its score and timestamp, and the reported aggregate
recall/precision values. It also reproduced the following paired 95% intervals
using 2,000 whole-game resamples, seed 20260929, to numerical tolerance below
1e-14. Stored start/end windows were checked to be nonoverlapping within game.

- Direct history: recall **35.4785%**, precision **29.7392%**, **593** positive
  windows retrieved. Direct snapshot: recall **34.4475%**, precision
  **29.4383%**, **587** retrieved.
- Logistic risk: recall **34.0800%**, precision **29.9398%**, **597** retrieved.
  History's larger macro recall despite fewer total hits is a consequence of
  giving each positive game equal recall weight, not an arithmetic error.
- Calibration-selected cheap reference, worst gold slope: recall **31.7684%**,
  precision **27.8837%**, **556** retrieved. History minus this reference is
  **+3.7101 pp [1.0316, 6.3878]**, or **+11.6787% relative recall**. This passes
  the declared cheap-reference gate.
- History minus logistic: **+1.3985 pp [-0.5796, 3.3876]**. The difference is
  unresolved and the relative gain is only **4.1036%**.
- History minus the same-capacity snapshot: **+1.0310 pp [0.00435, 2.1037]**,
  only **2.9929% relative recall**. The very small positive lower limit does
  not meet the declared 10% useful-gain threshold. The history-specific gate
  is therefore false.
- W1 disagreement recall is **34.1447%**, precision **28.8365%**; its primary
  declared gate is false. All of these are development results, with several
  prespecified exploratory comparisons, not independent confirmation.

Baseline selection was correctly locked on calibration: worst-slope macro
recall **32.3195%** versus low-current-WP **27.2972%**, on **1,000 calibration
games**. The source is
[selection_lock.json](../data/runs/setback-20260929-v1/experiment/selection_lock.json).

## Missing strong baseline found by the audit

An explicitly **post-result exploratory diagnostic** ranks the same eligible
windows solely by start time, latest first, taking two per game. It uses no
outcome labels to construct scores, but, like every original method, selects
from the completed game's available window pool. Its measurements are newly
reported in this review and computed from the persisted development rows above.

On the same 998 games / 721 positive games / 1,994 selected windows, this rule
has **44.1586% macro recall**, **36.4594% precision**, **727** positive windows
retrieved, and finds a positive in **513** games. Direct history finds one in
444 games. History minus time-only recall is **-8.6801 pp [-11.0483, -6.2464]**,
from a separate paired 2,000-resample whole-game bootstrap with the same seed.
This diagnostic was not preregistered and must not be described as a newly
validated product; it exposes an omitted baseline.

The deficit appears on both recorded development patches. Patch **16.18**:
**695 games / 503 positive games**, time-only recall **44.8873%**, history
minus time-only **-9.7051 pp [-12.4868, -6.7308]**. Patch **16.19**:
**303 games / 218 positive games**, time-only recall **42.4771%**, difference
**-6.3150 pp [-10.3621, -2.0690]**. History minus logistic is unresolved on
each patch: **+1.2558 pp [-1.1081, 3.7266]** and **+1.7278 pp [-2.0549, 5.5948]**
respectively. Patch-specific results do not support a transfer breakthrough.

## What the audit does and does not establish

No label-derived feature or endpoint leakage was found in the reviewed
start-row scoring code. Endpoint gold and kill changes construct labels;
direct models consume start-only columns, and W1 outputs are complemented for
the actual focal team rather than evaluated on artificially mirrored inputs.
The audit verified persisted computations, not raw-event truth or independent
re-extraction of every feature.

There are two material evaluation limitations. First, absolute gold/kill
setbacks and their variance change with game time. The time-only diagnostic
shows that state sophistication has not beaten this simple source of ranking
information. Second, top-two selection from an entire completed replay uses
future knowledge of which candidate windows exist. That is legitimate for
the declared retrospective task, but not an online alert experiment. A fully
hindsight replay selector could also rank the actual realized losses directly;
beating weak predictive heuristics alone does not establish better replay
review or human time savings.

Coverage is **6,178 of 10,978 planned clock slots (56.2762%)**, with **974
missing endpoint bins** and **3,826 missing start bins**; these counts include
unavailable clock slots and cannot be called a percentage of actual played
time. No missing endpoint was inferred to be a terminal event. Observed
horizons are **180.004–180.202 seconds**, median **180.052 seconds**. These
coverage measurements come from the original report; their raw reasons were
not re-audited here.

## Decision for the next experiment

Retain v1 as a reproducible negative/limited result. Do not retune the setback
threshold or horizon on these labels. Rich position and team-resource
allocation features should first face the fixed win-probability proper-score
comparison against existing gold/time and snapshot/history controls. Any
repeat of the same retrospective setback metric must show the time-only
baseline alongside logistic and history risk.

A future short-horizon forecasting protocol can evaluate fixed clock strata
or a chronological alert rule with calibration-fixed thresholds, so that
knowing the end of the candidate pool cannot drive selection. Its missing
follow-up and terminal handling must be declared before inspection. It needs
fresh confirmation after a candidate is frozen. Predictive improvement would
still not identify preventable mistakes, a better action, direct win gains,
or coaching benefit. None of those claims is made here.

## Artifact binding

Development windows SHA256:
`b6f68e53ad32c9ac8ebb922259fff87b2934ac89429b169d975a0bf3bd246f76`.
Original report SHA256:
`2725c6a52acfbe779ac573832b8ed52a9e5cda9c14a6a7197763c0c35e21dda7`.
The independently computed time-only measurements have this document as their
report path: `winprob/docs/setback_result_v1.md`.
