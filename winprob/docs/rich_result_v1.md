# Rich state v1 — independent result and integrity audit

2026-09-29. **Rich trajectory has a small resolved log-loss gain, but the
declared full WP gate does not pass because Brier improvement is unresolved.
Setback retrieval also fails its useful-signal gate.** This is development
evidence, not confirmation or a demonstrated coaching benefit.

Source: `data/runs/rich-20260929-v1/experiment/report.json`, SHA256
`f5e566b4187f02858c7fecdf7fb9ff64d689336461235d5080102d12d4f4808d`.
Protocol: `docs/rich_protocol_v1.md`. All intervals below are 95% paired
whole-game bootstrap intervals, 2,000 resamples, conditional on the fitted
models. Positive score reductions mean improvement. The cohort has already
supported earlier hypotheses, so these comparisons remain exploratory.

## Main forecast result

The run uses 6,000 matched games / 138,831 frames: fit 4,002 / 92,294;
calibration 1,000 / 23,171; development 998 / 23,366. Model fitting uses the
same 22,554 hash-sampled fit frames as W1. Rich state adds 42 columns to the
original 19, for 61 total. Recorded run time is 184.2 seconds.

On the 998 development games, original trajectory log loss is 0.497963 and
rich trajectory is 0.494341. Reduction is **0.003621 [0.000176, 0.007357]**,
or **0.727% relative**. This clears the declared 0.5% relative reduction and
positive log-loss interval conditions. Brier falls from 0.164116 to 0.162921,
but its reduction **0.001194 [−0.000103, 0.002553]** includes zero; the
conjunction of all WP gate conditions is therefore false.

ECE20 decreases from 2.077% to 1.859%; the paired reduction is 0.218
percentage points [−0.330, 0.683], also unresolved. ECE is descriptive and
bin-sensitive. Rich snapshot alone reduces log loss by 0.002579
[−0.001120, 0.006478]. Rich trajectory versus rich snapshot reduces log loss
by 0.001042 [0.000204, 0.001880], but its Brier interval still includes zero.

## Clock and patch checks

Rich trajectory versus original trajectory, fixed clock ranges:

- 3–10 minutes, 998 games: log-loss reduction 0.003201
  [−0.000060, 0.006472], relative 0.509%.
- 10–20 minutes, 998 games: 0.003082 [−0.002426, 0.008453], relative 0.616%.
- 20–30 minutes, 817 games: 0.006807 [0.001242, 0.012886], relative 1.833%.
- 30–40 minutes, 254 games: −0.000370 [−0.011318, 0.010413], relative −0.091%.

Every corresponding Brier interval includes zero. The 20–30 minute
log-loss result is a descriptive location of signal, not a selected
subpopulation for a new positive claim. Rich snapshot has an adverse late
result: 30–40 minute log-loss reduction −0.014602 [−0.028198, −0.001216].

On patch 16.18, 695 development games, rich trajectory log-loss reduction is
0.003443 [−0.000986, 0.007862], relative 0.700%. On patch 16.19, 303 games,
it is 0.004030 [−0.002176, 0.010205], relative 0.788%. Both point estimates
are favorable; neither patch independently resolves the improvement, and
both patch Brier intervals include zero.

## Retrieval result

The matched development population is 6,178 complete three-minute windows
in 998 games, containing 1,688 positive windows in 721 games. Every method
retrieves 1,994 windows under the same at-most-two-per-game budget. Macro
recall averages the fraction of positives retrieved within each positive
game; precision counts positives among all selected windows.

- Rich risk: recall 35.821%, precision 29.589%; 590 positives retrieved.
- Original history risk: recall 35.479%, precision 29.739%; 593 positives.
- Original logistic risk: recall 34.080%, precision 29.940%; 597 positives.
- Latest-two time ranking: recall 44.159%, precision 36.459%; 727 positives.

Rich risk's recall gain versus original history is only 0.342 percentage
points [−1.282, 1.951], or 0.964% relative. Versus logistic it is 1.741
points [−0.449, 3.829], or 5.108% relative. Both fail the declared 10%
relative-recall threshold and positive-interval condition. Against latest-two,
rich recall is **8.338 points worse [−10.730, −6.041]**, with precision
**6.871 points worse [−8.576, −5.263]**. The overall retrieval gate is false.

The patch direction is also uneven: rich versus original history recall is
+1.902 points [−0.010, 3.814] on 16.18 and −3.257 points
[−6.568, −0.182] on 16.19. Latest-two beats rich on both patches. Within
fixed clock ranges, direct-risk log-loss reduction versus original history
resolves only at 20–30 minutes: +0.007867 [0.000411, 0.014996], n=628 games;
all four fixed-clock Brier intervals remain unresolved.

Latest-two was identified after the previous risk experiment and declared
before this rich run. These are retrospective complete replay pools; later
eligible-window membership has hindsight. None of these results establishes
an early warning, preventable mistake, useful recommendation or live benefit.

## Independent integrity findings

Read-only post-run checks passed. The original and enriched arrays have
exactly the same 6,000 game identities and 138,831 rows. All **19 original
float32 feature columns match exactly by feature name**, plus all seven
metadata arrays: label, numeric game index, role, timestamp, match start,
split timestamp and patch. Every pairwise role intersection is zero.

Current reservations were rechecked from 71 build split metadata files and
the external reservation export: 3,112 build test IDs, six external test
IDs, 9,179 historical validation IDs and 1,911 prior confirmatory IDs.
**The matched 6,000-game set intersects none of these four sets.**

The report sidecar, cache metadata sidecar, original/enriched cache hashes,
source-manifest hash, admission sidecar, external/prior bindings, all 17
frozen report input/protocol bindings, all three rich model artifact hashes,
all three original cache code hashes, and both saved prediction hashes
validate. Persisted old risk scores and window identities exactly match the
original risk experiment. WP point metrics and every retrieval point metric
reproduce from the saved predictions; latest-two scores equal start times.

Original cache SHA256:
`7b25b48a8582d76019bd4c4ddbc4ebc8e74295325a927c2018da311a133a9670`.
Rich cache SHA256:
`65414f41331d4a82601ef38d9a95ff4b465bc90a758629730683593a93e729cb`.

The source manifest binds 12,000 compressed files across 6,000 games. This
independent audit **did not reopen raw archives**; it verified the bound
source metadata and cache parity. The original enrichment run performed
the compressed-file checks before decoding. No model was refitted, no
confirmation payload opened, and no existing implementation changed.

## Consequence

There is a useful hypothesis to investigate in WP state representation, but
not yet a passing forecast result. The retrieval branch has no demonstrated
benefit at this budget. Feature-family ablation can identify whether the WP
gain comes from spatial structure or allocation without changing the sealed
confirmation population. Any subsequent scale experiment must be declared
separately and retrain original and rich models on identical admitted games;
these exploratory results do not authorize a successful-result claim.
