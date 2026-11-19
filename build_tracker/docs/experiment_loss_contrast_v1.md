# Pre-registration: ward-weight loss contrast, twin arms (v1)

2026-09-22 afternoon. Owner's assignment, recorded BEFORE the run. Run
dir: `data/experiments/16.18-loss-contrast-v1/`. Internal probe only; VAL
and test are NOT read by this experiment.

## Question

The current loss decomposes exactly into `main + ward|main`. On the fixed
FIT subset the ward term is 18% of the loss and its gradient opposes the
main term on shared parameters (`cos = −0.287` at `Bf_ep16`, `−0.122` at
`econ_ft_ep8`; `docs/experiment_ward_loss_diagnostic_v1.md`). Does
downweighting it to 0.25 produce a better MAIN-PURCHASE model — and, the
question that decides anything for the product, a better MIXTURE?

## Arms — identical except one scalar

Both arms fine-tune the full `Bf_ep16` plus a zero-init correction head,
with **real economic features in both** (this is not the control/econ
contrast; that question is already answered). Everything is cloned from
the recipe that worked (`docs/experiment_basket_econ_finetune_v1.md`):
D(x) = the frozen parent's cached top-128, never a mask; full-catalogue CE
recomputed with gradients; microbatch 32 → effective 256; Adam lr 1e-4,
cosine T_max = 8; 8 epochs; `torch.manual_seed(16)` per arm and a shared
`Generator().manual_seed(16)` batch order, so both arms see the same
examples in the same order.

| arm | loss |
|-----|------|
| `w1` (control) | `(logZ − logZ_g*) + 1.00 · (logZ_g* − s_b*)` |
| `w025` (candidate) | `(logZ − logZ_g*) + 0.25 · (logZ_g* − s_b*)` |

`w1` is arithmetically the current CE, written through the decomposed code
path so that the ONLY difference between arms is the scalar.

**Result is read at the final checkpoint (epoch 8). No epoch is chosen.**

## Reproduction diagnostic (NOT a noise floor)

`w1` is arithmetically the current CE and should reproduce `econ_ft_ep8`
(same recipe, same seeds, same batch order — only the numerical path
through two logsumexps differs). `w1 − econ_ft_ep8` is therefore published
as a **reproduction diagnostic**: it measures agreement between two code
paths that should agree.

It is **not** an estimate of training variability and does not bound what
counts as interpretable. Nothing is declared uninterpretable on the basis
of its size. A large discrepancy is a reason to investigate the
implementation before reading anything else; a small one says the two
paths agree. Either way, the questions are answered by the paired
comparisons against the twin control and against the frozen mixture.

## Evaluation — six scorings, two primaries

Each arm is scored alone (grouped ward-marginalized decoder) AND mixed
50/50 with the frozen `wardfree_ep16` under the frozen mixture rule
(`frozen_mixture_v1.json`: canonical-key permutation, bijection asserted,
normalized probabilities, weights never tuned).

- **PRIMARY 1 — attribution**: `mix(w025)` vs `mix(w1)`, ward-free
  `first_action_exact`, paired, bootstrapped by match.
- **PRIMARY 2 — practical progress**: `mix(w025)` vs the current frozen
  mixture `M = mix(econ_ft_ep8)`.
- Secondary: `w025` vs `w1` alone; each arm vs `econ_ft_ep8` alone;
  any-of-three; slices single / multi / no_buy_death / ward_only.

**Declared reading rule.** Both primaries compare MIXTURES, so both are
read as statements about mixtures and nothing else.

A positive PRIMARY 1 with an unresolved PRIMARY 2 means: **improvement
over the twin control mixture, with no demonstrated improvement over the
frozen mixture.** That is all it means. In particular it does NOT
establish that the gain overlapped with what the ward-free model already
supplied — the same pattern arises if `mix(w1)` is simply worse than the
frozen mixture, i.e. if the control arm regressed. Which of those it is
must be read off `mix_w1 vs mix_frozen`, reported alongside.

Whether the loss change helps the economic model ITSELF is a different
comparison (`w025` vs `w1` alone, and each vs `econ_ft_ep8` alone) and is
reported separately. A mixture result is never read as an arm result, and
an arm result is never read as a product result.

Whatever the outcome, it is published as it stands and not retried with a
different weight.

## Complementarity is measured, not assumed

The owner's warning is the mechanism this experiment must watch: a better
economic arm can be a LESS complementary one, and today's `+1.03` may not
survive its own parent improving. For every arm (`econ_ft_ep8`, `w1`,
`w025`) against `wardfree_ep16` we report, on the same probe rows:
disagreement in first recommendation, disagreement in correctness,
each-only-correct counts, and the oracle-union rate. **No future gain is
added to the `+1.03`**; only the measured mixture numbers count.

## Pre-training gates (the run aborts if any fails)

1. **λ=1 equals the conventional CE**, in value and in gradient w.r.t. the
   logits, within tolerance, checked on synthetic logits through a local
   `torch.Generator` so the global RNG state is untouched (asserted before
   and after).
2. **The frozen references must be present**, and in full mode their
   re-scored rows must match the persisted ones with **zero** mismatches —
   `econ_ft_ep8` alone against the grouped decoder's rows and `mix_frozen`
   against the frozen mixture's rows. A nonzero count aborts the run
   instead of producing a report that merely mentions it.
3. **Row identity includes the champion**: every paired comparison and
   every reproduction check keys on `(match_id, ts, champion)`, not
   `(match_id, ts)` — the ten perspectives of a game share timestamps, and
   this project has already been bitten by that exact collision.
4. **Free disk** is checked before the run and again after every epoch.

## Budget and stopping rule

~430 s/epoch/arm measured on the parent recipe (the decomposed loss added
no measurable cost in the smoke: 429 s equivalent) → ~2.1 h training plus
scoring; abort if the first epoch exceeds 700 s or free disk drops below
40 GB, checked after every epoch. One training process at a time on this
machine: this run may not start while `BuildTracker Mixture VAL` holds the
GPU. **Loss curves are appended to `progress_full.jsonl` as each epoch
ends**, not only written into the final report — a run that dies at epoch
6 must still leave its curves behind (this has already cost us curves
twice). Diagnostic only — no promotion, no epoch selection,
no weight sweep. If the result is negative or unresolved it is published
unchanged.
