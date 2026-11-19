# Model development closeout — September 12, 2026

Status at 20:27 UTC: today's model work is complete. The final ranking
experiment finished and passed independent evidence checks. At the owner's
wrap-up request, no further experiment will be selected today.

## Verified findings

The reconstructed 16.18 development benchmark is complete. On validation
n=18,041 decisions / 151 games, the original candidate's exact-action agreement
is 0.3304140569 versus 0.1997117676 for the conditional baseline. Family match
is 0.5999249390 versus 0.3856883718 on n=15,987 applicable purchases. Both paired
intervals favor the candidate; all displayed baskets pass the offline-wallet
legality check. Source:
`data/experiments/16.18-reconstructible-development-v1/policy_eval_val_candidate_vs_conditional_policy.json`.

The additional-copy rule, target-filter ablation and matched head-learning
variants did not establish the required two-primary improvement. Keep the
original candidate as the reference. The contrastive head variant's small
exact-action change is uncertain and its family match is worse. Detailed
controls, intervals and reports remain in `docs/model_development_20260912.md`.

Source accounting is complete and independently verified. Among 5,170 failed
validation targets within the same full n=18,041 / 151 games, strict
conditional sell-first scenarios find 131 routes at unchanged gold_est and
162 more with reviewed sale credits. Another 106 screened cases remain
wallet-limited; 4,771 are outside this conservative scenario. This establishes
conditional feasibility only, not exact gold or improved model accuracy.
Source: `data/experiments/16.18-source-accounting-v1/source_accounting_val.json`.

## Final ranking result

D7 trained a separate small scorer of the original BUY alternatives while
keeping the original model and candidates fixed. All 86,488 TRAIN rows had
fresh options and bitwise-matched parent contexts; 29,540 informative rows
trained the ranker for the fixed three epochs. All 64 assert scripts passed.
The completed supervisor, frozen cache/ranker and independent outcome checks
all passed. The one-shot scheduled task removed itself.

Run: `data/experiments/16.18-option-ranking-v1`.
Task: `BuildTracker 16.18 Option Ranking V1`, launched at 19:15 UTC.
Plan SHA256: `c92c3f5bbebae56eb9775da25cde7e3354223efc52b96fbe6e13450cbec5ffae`.
On the same development validation, exact-action agreement rises from
0.3304140569 to 0.3403359016 (n=18,041 / 151 games; paired difference
+0.0099218447, CI95 [+0.0070098576,+0.0124685491]). Family match falls
from 0.5999249390 to 0.5892287484 (n=15,987 / 151 games; difference
-0.0106961907, CI95 [-0.0144798679,-0.0078991913]). Ward-only recall
falls by 0.0497925311 on n=723. Candidate coverage, SAVE behavior and all
displayed basket contents stay fixed; zero options breach the offline wallet.
Report: `data/experiments/16.18-option-ranking-v1/option_ranking_eval_val.json`.
Final and independent checks:
`data/experiments/16.18-option-ranking-v1/final_verification.json` and
`data/reports/model_day_20260912/option_ranking_independent_verification.json`.

The ranker fails the declared rule because family match declines. Preserve
it as a development-only negative result and retain the original candidate.
Do not tune another variant on this validation. No further experiment runs
today.

## Boundaries and next decision

No final-test evaluation, promotion, serving change, commit/push or shutdown
occurred. The served deployment remains unbound. All original source plans,
datasets, completed experiments and failed attempts remain preserved.

The completed day heartbeat was deleted. The separately authorized
two-night collection/acceptance follow-up remains responsible for new dated
samples. Any later model work should start from this verified record and a
fresh recorded question; it is not authorized as another autonomous day.
