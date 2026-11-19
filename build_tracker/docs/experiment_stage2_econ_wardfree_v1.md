# Stage 2 on the v4 generation — economic features + ward-free partner (protocol v1, budget first)

2026-09-25. Reviewer's step 3 after the v4 stage-1 result, accepted by the
owner: "prepare the protocol and measure the budget; the machine-day is not
committed until step 2's result and the projected cost of every segment
are on the table; the comparison must keep BOTH the new Bf and the best
frozen combination." Nothing in this document is launched by itself.

## Question

On the new generation (TRAIN 6,354 games / 757,975 visits), do the two
pieces that carried the old mixture — economic features (twin fine-tune)
and a ward-free partner — still add accuracy on top of the new Bf, and does
their 50/50 mixture beat the best frozen combination we already have?
Their earlier gains (+0.33 econ vs twin control; mixture +1.03 over its
parent) do NOT transfer automatically; each is re-measured here.

## References (fixed before any segment runs)

- **R1 — new Bf** (`frozen_reference_v4.json`): 50.737% ward-free
  first-exact on the cohort (146,597 decisions / 1,209 games).
- **R2 — best frozen combination**: M (new Bf ⊕ old mixture, 50/50) if
  step 2 adopts it by its declared rule; otherwise R2 = R1.
- Every candidate below is scored on EXACTLY the cohort rows of stage 1,
  paired by game against R1 and R2 (400 + 2,000 draws, seed 16), with
  slices (single / multi / no_buy_death / ward_only), movement and the
  integrity gate "R1 and R2 reproduce their persisted rows with 0
  mismatches".

## Segments (order fixed; each has its own measured pilot and bound)

**A. Economic feature cache on the new generation** (CPU). Parent = new
Bf; domain = its frozen top-128 per decision; 10 econ features per
candidate with the activated engine (`purchase_sequence` at `gold_est`),
for FIT (757,975) and the cohort (146,597). v3 anchor: 4,528 s for 297,240
rows = 15.2 ms/decision on one core. Extrapolated ≈ 3.8 h; the builder's
own 200-row bench projects before building and refuses above its bound
(declared here: **6 h**; the parallel builder `econ_features_parallel.py`
may be used ONLY after its bit-identity gate passes on a sample).

**B. Twin economic fine-tune from new Bf_ep16** (GPU). Recipe of
`basket_econ_finetune.py` unchanged: both arms fine-tune the full net plus
a zero-init correction head on the fixed domain; candidate gets real
features, control gets zeros; 8 epochs; micro 32 / effective 256; same
batch order per arm; ep8 fixed. v3 anchor: 7,638 s for 2 arms × 8 epochs
on 238,459 rows (477 s per epoch-arm, 7,937-basket catalogue). Extrapolated
by rows (×3.18) ≈ 6.8 h; by rows × catalogue (13,165) up to ≈ 11 h. Pilot:
2 untimed warm-up + 12 timed effective batches → bound **12 h**.

**C. Ward-free partner** (GPU). Phase-2 Bf recipe, clean init, on the
ward-stripped catalogue of the new FIT (9,333 groups incl. SAVE); 16
epochs, ep16 fixed. v3 anchor: 4,637 s for 16 epochs on 238,459 rows
(5,692 groups). Extrapolated ≈ 4.1 h by rows, up to ≈ 6.7 h with the
catalogue growth; the measured stage-1 epoch (1,734 s at 13,165 baskets)
bounds it from above at ≈ 7.7 h. Pilot as in B → bound **8 h**.

**D. Mixtures (declared, no weight search)**: M2 = 0.5 · P(econ_ft_new,
grouped) + 0.5 · P(wardfree_new), the old package's rule on the new
generation. No other weights, no other pairs. If step 2 adopted M, the
comparison keeps M as R2; M is not re-mixed.

**E. Evaluation** (CPU-bound by econ features unless cached): with the
cohort features cached in A, scoring is minutes on GPU; without, 34 rows/s
≈ 72 min per package-style configuration.

Total (extrapolated, before pilots): A 3.8 h CPU ∥ C 4–8 h GPU, then B
7–12 h GPU, then D/E ≈ 1 h → 12–20 h wall with A and C overlapped.
**The pilots replace these numbers before anything is committed.**

## Engineering before any segment can run (not machine time)

The v3 scripts assert the v3 frozen membership digests and hard-code v3
directories (`basket_econ_cache.py`, `basket_econ_finetune.py`,
`basket_wardfree.py`, `build_mixture_package.py`). Stage 2 needs
new-generation variants, as `basket_bf_newgen.py` was for the Bf recipe:
same recipes, generation-agnostic inputs, fingerprinted caches, pilots
built in, and a common cohort scorer that keeps R1 and R2 in every report.
Estimated one working day; smoke on the v4 generation before launch.

## Budget pilot (runs BEFORE the owner decides; ~10 min, no training)

`scripts/stage2_budget_pilot.py` (allowlisted) measures on the v4
generation, GPU idle: (A) ms/decision of the econ features on 200 real FIT
rows with the new Bf's top-128 domain → projected hours for 904,572
decisions; (B) s/effective batch of the twin fine-tune step (TunedNet on
new Bf + correction head, backward included) → hours for 8 × 2 epochs;
(C) s/effective batch of the ward-free recipe on the 9,333-group catalogue
→ hours for 16 epochs. Two untimed warm-up batches precede every timed
series (the cold-start lesson of 2026-09-24). Output:
`data/experiments/16.18-stage2-budget-v1/budget_pilot.json`.

## Decision rules (declared)

- A segment whose pilot projects above its bound is not run; the owner
  decides (raise the bound or drop the segment).
- econ_ft_new is "positive" only vs its twin control (attribution) AND
  vs R1; wardfree_new is measured vs R1; M2 is adopted as the new
  practical reference only if it beats BOTH R1 and R2 with the 95% CI
  excluding zero. Anything else is published as null/negative.
- No epoch choice after the fact, no weight search, no VAL/test read of
  earlier generations, no promotion. The cohort's exposure count grows by
  the declared configurations only (econ_ft_new, control_ft_new,
  wardfree_new, M2).

## Budget pilot RESULT (2026-09-25 22:51, `data/experiments/16.18-stage2-budget-v1/budget_pilot.json`)

Measured on the v4 generation, GPU otherwise idle, one CPU core busy with
the step-2 run (so A is, if anything, pessimistic):

| segment | measured | projection | declared bound | within |
|---|---|---|---|---|
| A · econ feature cache (CPU, one core) | 14.37 ms/decision (0.1123 ms/candidate; v3 anchor 15.2) | **3.61 h** for 904,572 decisions | 6 h | yes |
| B · twin econ fine-tune (GPU, 8 epochs × 2 arms) | 0.65 s/effective batch | **8.56 h** | 12 h | yes |
| C · ward-free partner (GPU, 16 epochs, 9,333 groups) | 0.495 s/effective batch | **6.51 h** | 8 h | yes |

GPU total B + C = **15.07 h**; A overlaps C on the CPU, so wall ≈
**15.07 h** of machine plus minutes of evaluation once A caches the cohort
features (72 min per configuration without it). Engineering before launch
(new-generation variants of the three scripts + common scorer + smoke) is
separate from machine time. Every segment is within its bound; nothing is
launched by this result — the owner decides after step 2.

---

## RESULT (2026-09-27 03:09) — M2 beats the frozen reference: +0.79 [+0.69, +0.89]; econ features carry it; ward-free alone is null

Cohort 146,597 decisions / 1,209 games (== generation `val`), ward-free
first_action_exact, grouped decoder, paired by game (seed 16; 400 + 2,000
draws). Every number below was recomputed independently from the persisted
rows (identity identical across 8 row sets; the mixture stage reproduced both
parents' segment rows with 0 mismatches). Full write-up:
`docs/stage2_result_v4_20260927.md`.

| arm | first-exact | top-3 | vs R1 (new Bf 50.737%) | net | beats R1 |
|---|---|---|---|---|---|
| **M2 = 0.5·econ_ft_new (grouped) + 0.5·wardfree_new** | **51.53%** | **66.09%** | **+0.794 [+0.691, +0.889]** (2,000 draws [+0.698, +0.894]) | **+1,164** (3,383 − 2,219) | **yes** |
| econ_ft_new (ep8) | 50.98% | 65.68% | +0.246 [+0.160, +0.324] | +361 | yes |
| control_ft_new (ep8, twin) | 50.71% | 65.56% | −0.025 [−0.078, +0.032] | −37 | no |
| wardfree_new (ep16) | 50.65% | 65.59% | −0.086 [−0.216, +0.050] | −126 | no |
| old mixture (mixture-50-50-v1) | 49.20% | 64.20% | −1.540 | −2,257 | — |

- Attribution: econ_ft_new vs its twin control +0.271 [+0.202, +0.335] (net +398; top-3 +0.12).
- M2 vs parents: +0.55 [+0.45, +0.64] vs econ_ft_new; +0.88 [+0.78, +1.00] vs wardfree_new;
  both parents right -> M2 wrong 0; both wrong -> M2 right 97; novel 618 (0.4%); oracle union 54.69%.
- M2 vs the old mixture: +2.33 [+2.18, +2.48], net +3,421.
- Slices M2 vs R1: single +0.81 [+0.69, +0.93]; multi +1.05 [+0.83, +1.26]; no_buy_death -0.11 (ns);
  ward-only -0.77 [-1.21, -0.34] (cost). Top-3 M2 +0.49 [+0.40, +0.57].

Decision by the declared rules: M2 beats R1 with the CI excluding zero -> M2 is the practical
reference R2 (`frozen_reference_stage2.json`); R1 stays the single-model reference. Nothing
promoted; test sealed. Costs (all bounds respected): wardfree 6.97 h train (pilot 6.81); twin
fine-tune 8.97 h train + 13 min eval (pilot 8.03 for both arms); features parallel ~1.7 h wall
(sequential bench 16.9 ms/decision -> 4.24 h); domains 3 h 20 on CPU (avoid); gate 23.5 min;
mixture 34 min; ~23 h of machine, 26.5 h of wall with four incidents (docs/history.md).

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
