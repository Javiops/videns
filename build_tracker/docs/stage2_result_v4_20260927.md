# Stage 2 on the v4 generation — result (2026-09-27 03:09)

Owner GO 2026-09-25 23:30 (`docs/experiment_stage2_econ_wardfree_v1.md`,
conditions: recipes/checkpoints/weights/bounds fixed; new Bf frozen as the
reference; every candidate compared directly with it; full publication incl.
costs; no confirmation between segments after the smoke). Chain
`scripts/run_stage2_v4.py`; reports under `data/experiments/16.18-stage2-v4/`
(`stage2_report_full.json`, per-segment reports, per-decision rows of every
arm). Nothing promoted; test sealed.

## Headline

Cohort: **146,597 decisions / 1,209 games** (2026-09-20/21, later than every
training game; the generation's `val` split). Metric: ward-free
first_action_exact, grouped decoder, legality at `gold_est`; paired by game
(seed 16; 400 + 2,000 draws). R1 = the new Bf frozen after stage 1
(50.737%).

| arm | trained on | first-exact | top-3 | Δ vs R1 [95% CI] | net decisions | beats R1 |
|---|---|---|---|---|---|---|
| **M2 = 0.5·econ_ft_new (grouped) + 0.5·wardfree_new** | — (declared rule, weights untuned) | **51.53%** | **66.09%** | **+0.794 [+0.691, +0.889]** (2,000 draws [+0.698, +0.894]) | **+1,164** | **yes** |
| econ_ft_new (ep8, real features) | v4 FIT, from R1 | 50.98% | 65.68% | +0.246 [+0.160, +0.324] | +361 | yes |
| control_ft_new (ep8, zeroed features) | v4 FIT, from R1 | 50.71% | 65.56% | −0.025 [−0.078, +0.032] | −37 | no |
| wardfree_new (ep16, clean init) | v4 FIT | 50.65% | 65.59% | −0.086 [−0.216, +0.050] | −126 | no |
| R1 new Bf (reference) | v4 FIT | 50.74% | 65.60% | — | — | — |
| old mixture `mixture-50-50-v1` (previous best) | v3 FIT | 49.20% | 64.20% | −1.540 | −2,257 | — |

- **Attribution**: econ_ft_new vs its twin control **+0.271 [+0.202, +0.335]**
  (net +398; top-3 +0.12 [+0.08, +0.17]). Fine-tuning without the features
  neither helps nor hurts (control vs R1 −0.03, CI crosses 0).
- **M2 vs its parents**: +0.55 [+0.45, +0.64] vs econ_ft_new (net +803);
  +0.88 [+0.78, +1.00] vs wardfree_new (net +1,290). Both parents right → M2
  wrong: **0**; both wrong → M2 right: 97; M2 novel first recommendation in
  618 decisions (0.4%). Oracle union of the parents 54.69%.
- **M2 vs the old mixture**: **+2.33 [+2.18, +2.48]**, net **+3,421** of
  146,597 — the product bar moves from 49.20% to 51.53% across the two
  blocks of this week (expansion + stage 2).
- **Slices, M2 vs R1**: single +0.81 [+0.69, +0.93]; multi +1.05 [+0.83,
  +1.26]; no_buy_death −0.11 [−0.83, +0.61] (ns); **ward-only −0.77 [−1.21,
  −0.34]** (real cost; econ_ft_new −0.85, wardfree_new −0.78 carry the same
  sign). econ_ft_new vs R1: multi +0.97 [+0.80, +1.12]; single +0.08 (ns);
  SAVE −0.99 [−1.54, −0.53]; ward-only −0.85 [−1.18, −0.54].

**Decision (rules declared before the run)**: econ_ft_new beats control and
R1; wardfree_new does not beat R1 (null, published); **M2 beats R1 with the
CI excluding zero → M2 is the practical reference (R2)**:
`data/experiments/16.18-stage2-v4/frozen_reference_stage2.json` (component
digests: econ_ft_new_ep8 670dd7bd…, wardfree_new_ep16 ffbfc40a…, domain
parent R1 c923d6ea…, econ_stats). R1 stays the single-model reference.

## Verification

Independent recompute from the persisted rows: identity (match_id, ts,
champion) identical across eight row sets (R1, old Bf, old mixture,
wardfree_new, control_ft_new, econ_ft_new, M2, and the mixture stage's
re-decodes of both parents); every aggregate, paired delta, movement count
and slice reproduces; CIs to bootstrap noise (e.g. M2 vs R1 [+0.698, +0.891]
vs report [+0.691, +0.889]). Integrity gates: R1 reproduced its stage-1
rows 0/0 (`gate` phase, GPU); the mixture stage reproduced econ_ft_new's and
wardfree_new's segment rows 0/0; step-0 correction exactly 0 and the
parent's fixed-subset CE (1.7395) matched the stage-1 checkpoint.

## Costs, as measured (every declared bound respected)

| segment | pilot / bench | measured | peak job commit |
|---|---|---|---|
| A · domains (CPU) | — | 3 h 20 (pathological CPU forward ≈ 64 rows/s; run on the GPU next time) | 8.4 GB |
| A · features (CPU, 10 econ features × 128 candidates × 904,572 decisions) | sequential bench 16.9 ms/decision → 4.24 h (bound 6 h) | parallel: 8 shards / 4 workers ≈ 62 min + cohort 41.5 min (+ 49 min recomputing two pieces lost at a memory-starved start); bit-identical to sequential (asserted on the smoke) | 3.9 GB |
| gate (GPU) | — | 23.5 min | 7.1 GB |
| C · wardfree (GPU, 16 epochs) | 0.517 s/batch → 6.81 h (bound 8 h) | 6.97 h train + 5.5 min eval | 9.7 GB |
| B · twin fine-tune (GPU, 8 epochs × 2 arms) | 0.610 s/batch → 8.03 h (bound 12 h) | 4.50 h + 4.47 h + 13 min eval = 9.31 h | 15.1 GB |
| D · mixture (GPU) | — | 34 min | 8.0 GB |

≈ 23 h of machine; wall 2026-09-26 00:33 → 2026-09-27 03:09 with four
operational incidents, all published in `docs/history.md`: the owner's pause
(chain made game-aware: CPU segments re-prioritised, GPU segments suspended
during a game), a false bound trip of the features bench (cyclic GC over the
resident cohort rows + Windows maintenance; fixed with `gc.freeze()` and
re-measured at 14–17 ms), a silent loss of two parallel workers plus a
PowerShell exit-code misread (5 idle hours), and a CUDA "out of memory" with
4.8 GB of VRAM free caused by the segment job's 12 GB commit ceiling.

## Limits

- **Selection on the cohort**: eight configurations have now been scored
  on it; adopting M2 is a selection. The sealed test (3,112 games) remains
  for one final confirmation of whatever is adopted — not before.
- **Not a package**: M2 is two checkpoints, a frozen domain and a rule.
  Promotion needs the versioned package (as `mixture-50-50-v1`), the
  displayed-policy report with zero unaffordable baskets, corpus
  independence, and a measured online cost (v1 package ≈ 40 ms/decision
  offline estimate; M2 adds a ward-free forward — unmeasured).
- **Ward-only decisions lose 0.77 points** (5,884 of 146,597): every new
  piece predicts a main purchase where the truth was ward-only. The product
  rule already keeps wards outside the basket; the metric cost is real.
- Estimated wallet (`prequential-v3`); volume/recency entanglement and the
  catalogue caveats of stage 1 still apply; 50/50 weights untuned by
  design; ep8/ep16 fixed in advance (FIT-loss curves still descending, no
  extension).

## What it means for the product

On the same forward cohort, the best available configuration went from
49.20% (frozen mixture of the v3 models) to **51.53%** — +2.33 points,
**+3,421 net decisions of 146,597** — through two measured steps: more and
fresher data for the single model (+1.54), then the economic features and
the ward-free partner re-learned on that data and combined (+0.79). The
learning-curve prediction and the v3 "pair beats either piece" lesson both
held on the new generation. The next decisions are the owner's: package M2
for the promotion path, and whether to grow the data again.

## Legality accounting on the cohort (2026-09-27, `data/reports/cohort_legality_v4_20260927/legality.json`)

Of the 138,073 cohort decisions with a main purchase, **103,716 (75.12%) are legal at the
estimated wallet**; 33,065 (23.95%) are budget-blocked and 1,292 (0.94%)
structurally blocked (the reviewer's v3 probe figures were 23.91% / 0.9%). Every arm's hits fall on legal
decisions (0 hits on blocked ones), so the conditioned rates are hits / legal mains:

| arm | exact 1st, all decisions | exact 1st, main purchases | **exact 1st on LEGAL mains** | top-3 on legal mains |
|---|---|---|---|---|
| M2 | 51.53% | 49.75% | **66.23%** | 87.98% |
| econ_ft_new | 50.98% | 49.19% | 65.49% | 87.44% |
| new Bf (R1) | 50.74% | 48.87% | 65.06% | 87.33% |
| wardfree_new | 50.65% | 48.82% | 64.99% | 87.31% |
| old mixture | 49.20% | 47.36% | 63.05% | 85.46% |
| old Bf | 48.17% | 46.24% | 61.56% | 84.45% |

Conditioned rate, not a forecast: it describes what the policy does where the truth was reachable at the
estimated wallet; the 24% budget-blocked share is the exact-wallet problem, unchanged by this block.

## Package (2026-09-27 10:29) — `mixture-v4-m2-v1`, identity `ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9`

Built by `scripts/build_mixture_package_v2.py` under the pinned runtime from the v4 generation (one offline read of
TRAIN), same contract as `mixture-50-50-v1` so `app/mixture_policy.py` loads it unchanged: domain_parent = R1,
econ_tuned = econ_ft_new_ep8, wardfree = wardfree_new_ep16 (parameters only), tensors (group ids, identity
permutation B→A, catalogue tensors, econ stats from the fine-tune), vocab, catalogues (13,165 baskets / 9,333
groups); every input verified against `frozen_reference_stage2.json` (4/4) before packaging; provenance block
filled with the generation manifest digest, export fingerprint 6c78398c…, eval_version 7, status
`experimental_reference` (the cohort is exposed; not a validation selection).

Displayed-policy harness through the adapter (`scripts/mixture_displayed_policy_v2.py`), smoke on the first
2,000 cohort decisions: equivalence with the experiment's M2 rows {'identity': 0, 'pred_first': 0, 'first_action_exact': 0, 'any_of_three_exact': 0}
(identity, pred_first, both exactness flags); legality of every displayed option: 0 unaffordable of
5,846 options checked; corpus independence: package scored with dataset_generations/
experiments/corpora unreadable (self-test blocked 3 probes, package 0 blocked reads). The full 146,597-row run is
in progress. Adapter latency (`scripts/mixture_package_latency.py`, 400 warm requests on a fixed v4 FIT sample,
RTX 4060): warm median **39.0 ms** (p90 48.9, p99 65.9; v1 package 35.3 / 49.0 / 65.7),
cold load median 81 ms. Not a promotion artifact: the promotion path still needs the owner's gate on a
frozen validation.

## Next investment prepared, not launched — prior-patch transfer (`docs/experiment_prior_patch_transfer_v1.md`)

Inventory: 20,859 patch-16.17 games in the DB, 17,904 with raw pairs (2026-08-25 → 09-10). The existing
`16.17-development-v2` generation was built with the older pipeline (reconstruction decision-start-footwear-v3,
prequential-v2 gold, prototype inventory) → not comparable; 16.17 must be re-reconstructed with the current
pipeline and its own static data (16.17.1 seeded). Arms with equal total budget: T (target-only, 16 + N' epochs)
vs H (N epochs on 16.17, then R1's 16-epoch adaptation on 16.18), scored on the same cohort, PRIMARY H vs T,
SAVE slice reported with its CI (where the antecedent failed). Cost anchors: ≈ 3 h reconstruction + ≈ 30 h GPU
for two arms at N = 16; pilots and the owner's GO before anything runs.
