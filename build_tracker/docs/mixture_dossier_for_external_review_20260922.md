# Mixture dossier — probe, VAL, latency, loss contrast, and what's queued

2026-09-22, full day. For the external reviewer. Everything below is on the
frozen generation `16.18-reconstructible-development-v3`; per-decision rows
are persisted and independently recomputable for every number. **Test
sealed. Nothing promoted.**

---

## 0. Where the day started and where it ended

**Started**: best observed configuration was the frozen dev baseline
(`econ_ft_ep8` + ward-grouped decoder), 0.48487 ward-free first_action_exact
on the internal probe. The previous night had produced 7 pre-registered runs
→ 2 negatives, 1 unresolved, 0 promotions.

**Ended**: best observed configuration is a **fixed 50/50 probability
mixture of the two models we already had**, 0.49514 on probe and 0.49190 on
VAL, beating both of its parents in both populations. It cost no training
and ~3.4 min of GPU. Serving it costs a measured +7.78 ms/decision.

One new hypothesis was isolated, tested and closed as a null for the
product. One more is prepared and not run.

---

## 1. Frozen 50/50 mixture — the result

Protocol `docs/experiment_frozen_mixture_v1.md` (weights declared at 0.5
before the run; no sweep, no weight adjusted after seeing any number).
Config frozen by sha256 in
`data/experiments/16.18-frozen-mixture-v1/frozen_mixture_v1.json`.

Rule: `P_mix = 0.5·P_A + 0.5·P_B` over the shared 5,692 main-purchase
actions, where `P_A` is the softmax of the per-group logsumexp of the
economic model's full-catalogue scores and `P_B` the softmax of the
ward-free model's catalogue scores. Normalized probabilities, mixed before
the common decode and the usual legality check.

### Probe (58,781 decisions / 500 games)

| arm | first | any-3 | vs A | vs B |
|---|---|---|---|---|
| A grouped `econ_ft_ep8` | 0.48487 | 0.63819 | — | −0.09 [−0.32, +0.17] |
| B `wardfree_ep16` | 0.48574 | 0.63808 | — | — |
| **M mixture** | **0.49514** | **0.64519** | **+1.03 [+0.85, +1.19]** | **+0.94 [+0.78, +1.12]** |

### VAL (191,916 decisions / 1,601 games)

| arm | first | CI95 | any-3 | vs A | vs B |
|---|---|---|---|---|---|
| A | 0.48275 | [0.4805, 0.4853] | 0.63380 | — | +0.08 [−0.06, +0.22] |
| B | 0.48198 | [0.4796, 0.4847] | 0.63394 | — | — |
| **M** | **0.49190** | [0.4895, 0.4945] | **0.64110** | **+0.92 [+0.82, +1.01]** | **+0.99 [+0.87, +1.09]** |

**This VAL window was already exposed** — read once on 2026-09-22 ~01:00 for
configuration A. It is a check in a different population, not a blind
confirmation. Stated in the protocol before the pass.

A-vs-B stays unresolved in **both** populations, so M is not inheriting one
parent's edge; it beats both.

### Movement (VAL)

- corrects 5,555 / breaks 3,798 vs A (net +1,757)
- **breaks 0 of 191,916 cases both parents got right**; wins 218 where both
  failed; 1,166 first recommendations match neither parent
- oracle union rate 0.5321 (what a truth-knowing selector over the two first
  options would reach — **not** a hurdle, an expectation, or a ceiling for a
  probability mixture, which can promote a shared runner-up)

### Costs, named

Only one slice has a CI excluding zero against a parent: **`no_buy_death`
vs B, −0.73 [−1.44, −0.06] on 3,564 VAL decisions.** Against A, M wins all
four slices.

### Integrity

- The two catalogues hold the **same** 5,692 actions but order them
  differently — **only 31 positions coincide**. Alignment is an explicit
  permutation by canonical multiset key with the bijection asserted; the
  31 reproduces the reviewer's own audit independently.
- Re-decoding A in-process reproduced the previous VAL run's **191,916**
  rows with **0 mismatches** in `pred_first`, `first_action_exact` and
  `any_of_three_exact`. On probe, both parents reproduced their persisted
  rows byte-identically.

Reports: `16.18-frozen-mixture-v1/mixture_report_full.json`,
`16.18-mixture-val-v1/mixture_val_report_full.json`, independent recompute
in `data/reports/frozen_mixture_review_20260922/`.

---

## 2. Latency — measured, after a first attempt that was not a measurement

The figure computed inside the VAL run was **arithmetic, not a
measurement**: mixture total minus the ward-free forward, still carrying the
blend and the decode. It never ran circuit A on its own. Withdrawn.

`scripts/basket_latency_bench.py`: both circuits end to end, per decision,
400 decisions on a fixed FIT sample, order alternated.

| | median | mean | p90 | p99 |
|---|---|---|---|---|
| circuit A | 75.53 ms | 91.31 | 97.53 | 923.62 |
| circuit M | 83.83 ms | 94.06 | 108.37 | 146.92 |
| **paired delta M−A** | **+7.78 ms** | +2.75 | +22.24 | +54.34 |

**Overhead of the mixture: +11.0% median.** Order does not bias it (A: 75.43
first vs 75.53 second; M: 84.13 vs 83.74).

**Open, unexplained**: the circuit measured **second** in each pair carries a
~900 ms p99 (A-second 928 vs A-first 122; M-second 895 vs M-first 136).
Alternating the order does **not** prove the effect is symmetric or that the
paired median delta is unaffected. Defensible statement: *observed paired
delta +7.78 ms; high tails when a circuit runs second; cause and effect on
the comparison undetermined.*

**Correction to an earlier statement of mine**: I said the per-decision
timings were already persisted for this run. They are NOT. The persistence
(`latency_bench_raw_*.jsonl`, with per-measurement timestamps and position
in the pair) was added to the script AFTER this benchmark had already run,
so the only surviving artifact is the aggregate report. **The tail cannot
be analysed from existing data** — it needs a re-run, which needs the GPU,
which is reserved until the history pilot finishes. The instrumentation is
in place for that re-run.

---

## 3. Ward-weight loss contrast — works on the loss, null for the product

Motivated by a diagnostic: the CE decomposes exactly by chain rule into
`main + ward|main`, and on a fixed 10,000-row FIT subset the ward term is
**18%** of the loss (0.30976 of 1.70479 at `Bf_ep16`) with
`cos(g_main, g_ward)` = **−0.287** / **−0.122** on shared parameters —
the objectives interfere. Decomposition residual 5e−09.

Twin arms from `Bf_ep16`, real econ features in **both**, zero-init head,
same seeds/batches/optimizer/8 epochs; only λ differs (1.00 vs 0.25).
Result read at the final checkpoint, no epoch chosen.

| comparison | Δ | CI95 |
|---|---|---|
| **PRIMARY 1** mix(w025) vs mix(w1) | **+0.048** | [+0.002, +0.099]; 2000 draws [**+0.000**, +0.097] |
| **PRIMARY 2** mix(w025) vs frozen mixture | **+0.044** | [−0.002, +0.095] **NOT RESOLVED** |
| arm alone w025 vs w1 | +0.073 | [+0.012, +0.141] |
| control mix(w1) vs frozen mixture | −0.003 | [−0.014, +0.007] |
| reproduction w1 vs `econ_ft_ep8` | −0.003 | [−0.025, +0.019] |

**The intervention did what it was designed to do, in the objective**: at
epoch 8, w025 has the better main term (1.3644 vs 1.3707) and the worse ward
term (0.3135 vs 0.3029), with worse total CE. *Precision*: those per-epoch
figures are **running means collected while training**, with the weights
changing inside the epoch — not a decomposition evaluated on the frozen
final checkpoints. They indicate the mechanism's direction; they do not
measure it on the delivered model.

**Reading, by the rule declared before the run**: improvement over the twin
control mixture, **no demonstrated improvement over the frozen mixture**.
The alternative explanation is excluded by measurement, not assumption — the
control did **not** regress. The arm gain is simply small (43 net decisions
of 58,781) and does not add to the mixture. Context: +0.048 on top of a
mixture already worth +1.03.

**Cost, with its exact meaning**: `ward_only` loses in all three comparisons
with CIs excluding zero (−0.28 / −0.24 / −0.32 on 2,487 decisions). Under
the ward-free metric that means **recommending a main purchase when the
player only bought wards**. It does **not** measure whether the optional
pinkward suggestion is right — that probability remains uncalibrated and out
of scope.

**Complementarity did not move**, which was the stated risk: oracle union
0.53510 with w025 vs 0.53475 with `econ_ft`; each-only-correct 2,902/2,912
vs 2,881/2,932. A slightly better economic arm was not a less complementary
one — and brought nothing either.

**λ=0.25 not adopted. Published as it stands, no weight sweep**, as
pre-registered.

Gates all passed: λ=1 ≡ conventional CE (|ΔL| 9.5e−07, |Δgrad|max 4.7e−09
against grad scale 2.1e−02, global and CUDA RNG asserted untouched); step 0
identical in both arms at the parent's 1.7047866682052613; row identity
keyed on `(match_id, ts, champion)` across all six row lists; **zero**
reproduction mismatches against the persisted frozen rows.

---

## 4. Claims withdrawn today

Listed because a reviewer should know what has been corrected, and because
they share one failure mode: **inferences stated with the confidence of
measurements.**

1. *"The catalogues align by construction."* True of the **set**, false of
   the index order — only 31 of 5,692 positions coincide.
2. *"`w1 − econ_ft_ep8` is the noise floor; a smaller effect is
   uninterpretable."* Withdrawn. It is a **reproduction diagnostic**, not an
   estimate of training variability, and nothing is declared uninterpretable
   by its size.
3. *"The latency p99 came from the contrast starting."* Timestamps disprove
   it (benchmark written 18:46:38; contrast started after; the launcher runs
   them sequentially with `-Wait`). Cause still unknown.
4. *"GPU at 75% ⇒ 25% waiting on Python, 4× ceiling, ~1.5× from a better
   card."* Not demonstrated. `nvidia-smi utilization.gpu` is the fraction of
   sampled time with at least one kernel resident; it says nothing about
   occupancy, the cause of gaps, or another device. A CPU/CUDA timeline
   profile is what would answer it. **What is measured**: 75% utilization
   and 3.0/8 GB VRAM while training; **0% GPU with one core saturated**
   during feature-heavy evaluation (i5-14400F 10c/16t, 15.8 GB) — opposite
   bottlenecks.
5. *"Exact finish 20:57 ±5 min."* An extrapolation from one 441.2 s epoch;
   the interval was not measured. (The run did finish 20:57.)

---

## 5. Traps recorded in `docs/history.md`

- **Same action universe ≠ same index order** (the 31/5,692 above). Mixing
  column *i* with column *i* would have averaged probabilities of different
  purchases while looking healthy.
- **PowerShell return channel**: a function doing `Write-Output "msg"` then
  `return $p.ExitCode` returns `Object[] = (msg, code)`; `$code -ne 0`
  filters the array and fires — **a successful run announced as failed**.
  Observed verbatim twice. Use `Write-Host`, return `[int]`, type the
  receiver `[int]`.
- **`Start-Transcript` buffers a native command's stdout until it exits**:
  the 2h56 VAL run wrote 900 bytes and zero progress lines and was
  unobservable throughout. Long runs now use
  `Start-Process -RedirectStandardOutput` plus per-epoch fsync'd curves.
- **Cache fingerprints**: `ShopDataset.from_cache` used to validate nothing
  while the basket line loads caches by fixed filename. See §6.

---

## 6. Prepared, NOT run: the purchase-history contrast

Protocol `docs/experiment_history_contrast_v1.md`. Question: do prior
purchases separate decisions that the inventory snapshot and the economic
features do not? The inventory keeps the state; only history keeps the
route (components are consumed on completion, consumables vanish).

Design: frozen `econ_ft_ep8` + frozen `wardfree_ep16`, a small zero-init
scoring correction, twin arms where the only difference is whether the
correction receives the history block or the same block zeroed. CS, assists
and inter-visit timings deliberately excluded. Both arms scored alone and
mixed 50/50; primaries compare mixtures.

### Blocker 1 — the cache, verified and fixed

`hist` in `cache_fit_full_v2.pt` is **2,861,508 entries, every one zero**,
and `from_cache` validated nothing. Setting `PREFIX_HISTORY=1` and reloading
it would have produced two identical arms, silently.

Fixed durably: `save_cache` now writes a `__meta__` fingerprint and
`from_cache(..., expect={...})` refuses a mismatching cache and refuses a
legacy cache outright when validation is asked for. `build_history_cache.py`
builds the history caches and asserts the real invariant — **identical to
the standard cache in every tensor except `hist`**, compared by prefix.

**Corrected after your review.** The figures first quoted here came from
the smoke sample, and "window full" was not the same as "history
truncated". Your full-TRAIN audit
(`data/reports/history_design_review_20260922/raw_history_lengths.json`),
SAVE excluded before accumulating, is the admissible measurement:
**27.46%** of FIT decisions and **26.95%** of probe decisions have MORE
than twelve prior units and are genuinely truncated; a further ~4.7% have
exactly twelve, filling the window without losing anything.

`HIST_LEN` stays at 12 for this pilot: a fixed limitation of the
representation, not a between-arms confound — both arms see the same
window.

### Blocker 2 — live, answered and it bounds the result

`prefix_items` appears **nowhere in `app/`**. `snapshot_to_row` does not
produce it and the Live Client API returns a state snapshot, never a
purchase log. A session could only accumulate history itself from inventory
and wallet deltas — the reconstruction problem again. And on a **cold start**
(the app attaching mid-game) everything earlier is unknown, while "unknown"
and "has bought nothing yet" are the **same all-zero tensor**.

Consequence declared in advance: **an offline gain is an upper bound on live
value.** The experiment measures the degradation instead of assuming it —
a declared secondary re-scores the candidate with history hidden for the
earliest half of each game's decisions.

### Three bugs the smoke caught, one of them a design error

1. The first builder set `USE_HISTORY` alone and left extras/gold/runes at
   module defaults — a different QUERY regime, exactly what `apply_config`'s
   docstring warns about. Now it reads the backbone's own config and flips
   one flag.
2. **Design error**: the frozen backbone was being fed populated history.
   **Measured: its own CE moves 1.628 → 3.865.** It was trained with those
   12 tokens always padded, so its `hist_pos` embedding was never updated
   and activating them drops untrained vectors into the attention input.
   That is not adding information, it is corrupting the frozen model. Fixed:
   **every frozen model sees `hist` zeroed; history reaches the correction
   head only.**
3. The same error in two more places (the frozen reference and the ward-free
   forward), both fixed.

After the fixes: training loss equals the parent's CE exactly (1.6275),
zero-init correction asserted, reproduction 0/1000 for the frozen mixture.

### Three further corrections from your review, applied

1. **SAVE was entering the window.** `with_history` extended the buffer
   with every `label_id`, sentinel 999999 included; it only became zero
   later, at embedding lookup, after already taking one of the twelve slots
   and possibly evicting a real earlier purchase. Measured on 60,000 TRAIN
   rows: **2.1% carry it**. Now filtered before accumulating; caches
   rebuilt.
2. **The head is a SET encoder and is now described as one.** The position
   embedding is added before a mean pool, so `mean(e_i + p_i)` over
   occupied slots `= mean(e) + mean(p over occupied)`: swapping two items
   between occupied positions is identical. Positions can inform how many
   slots are filled, never which item came first. The pilot therefore tests
   whether the **set of the last twelve purchased units** carries extra
   signal — nothing about ordered sequence, and the click-order/canonical-
   order risk no longer applies since order is not read at all.
3. **The hiding probe is not a late-attach simulation.** After the cut it
   returned the full history, including purchases before the supposed join,
   so it measures **sensitivity to hiding history** and nothing else.
   Renamed and rescoped accordingly.

Two more of my framings withdrawn: the full-history result is a
**diagnostic under ideal information**, not a mathematical upper bound on
live accuracy; and **zeroing the history block does not return the
parent** — the correction keeps its learned biases and still receives the
state and basket vector, which is why `mix_frozen` is the only true parent
reference.

The reproduction gate now runs **before either arm trains**, with
normalization from the full FIT econ cache in every mode. It passes with
**0 mismatches on both frozen references** — the previous "expected 1/1000
smoke artifact" is gone, because a gate with a known false positive is not
a gate.

A negative would rule out *this correction on this set representation over
this frozen backbone* — not purchase history, not a from-scratch model with
history, not inter-visit timings, not CS/assists.

---

## 7. Your answers, recorded

- **Mixture promotion path: yes.** Prepare the integration and the report of
  the policy actually displayed, with a legality check on every option. The
  results justify the work; they do not amount to a promotion.
- **Latency tail: investigate before general deployment.** The median does
  not explain the near-second episodes. Proceeds independently of the
  history work; per-decision timings with timestamps are already persisted.
- **History pilot: go**, as an offline diagnostic, after the SAVE filter,
  the correct encoder description and the withdrawal of the late-attach
  reading. All three applied and re-smoked.
- **`HIST_LEN=12`: no change** for this first contrast.

## 8. Original open questions (superseded by §7)

1. **Does the mixture warrant the promotion path?** It needs the
   displayed-policy report with zero unaffordable baskets and
   `promote_served_model.py` binding the digest — a different artifact from
   these diagnostics. The VAL evidence is from an already-exposed window.
2. **The latency tail.** Is the second-position effect worth chasing before
   anything ships, or is the median delta enough for a product decision?
3. **History contrast: go or hold?** It is prepared and validated but not
   run. Given blocker 2, a positive result would still need an explicit
   known/unknown indicator and training-time truncation augmentation before
   it could reach live — that follow-up is out of the current scope.
4. **Is `HIST_LEN=12` a confound worth fixing first?** 27% of decisions
   saturate it.

The stated next lever remains **data**: 23.91% of main purchases are still
budget-blocked by the estimated wallet, and the adopted
estimated-vs-measured wallet contrast is blocked on the League client
(test download `EUW1_7978990508`).
