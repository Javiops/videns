# Pre-registration: purchase-history contrast, residual twin arms (v1)

2026-09-22 night. Owner's assignment, recorded BEFORE anything is built.
Run dir: `data/experiments/16.18-history-contrast-v1/`. Internal probe
only; VAL and test are NOT read.

## Question

Do the player's PRIOR PURCHASES separate decisions that the current
inventory and the economic features do not separate well enough?

The motivation is concrete: the inventory is a snapshot, so the path is
gone. Components are consumed when an item completes, consumables vanish
when used, and two players holding the same three items can have arrived
there by different routes. History is the only input that keeps the route.

## The two blockers, resolved before designing anything

### 1. The cache — verified, and the trap was real

`ShopDataset.from_cache` used to validate NOTHING, and the basket line
loads caches by fixed filename (`cache_fit_full_v2.pt`). Setting
`PREFIX_HISTORY=1` and reloading that file would have produced an all-zero
`hist` and two IDENTICAL arms, silently. Measured on the real cache:
`hist` is 238,459 x 12 = **2,861,508 entries, every one zero**.

Fixed durably rather than locally: `save_cache` now writes a `__meta__`
fingerprint (history/extras/gold flags, HIST_LEN, schema, rows, and
`hist_nonzero`), and `from_cache(..., expect={...})` refuses a cache whose
fingerprint disagrees AND refuses a legacy cache outright when validation
is requested. Verified on four cases, including that the project's own
`cache_fit_full_v2.pt` is now rejected when history is demanded. This
experiment builds its own caches with `scripts/build_history_cache.py` and
loads them with `expect={"history": True}`, asserting `hist_nonzero > 0`.

### 2. Live — history is NOT available today, and that bounds the result

`prefix_items` appears NOWHERE in `app/`. `app/live.py:snapshot_to_row`
does not produce it, and the Live Client API returns a STATE SNAPSHOT
(current inventory, gold, scores), never a purchase log. So:

- A live session could only accumulate history itself, from inventory and
  wallet deltas observed across polls — which is the reconstruction
  problem again (components are consumed into completions).
- **Cold start**: when the app attaches mid-game, everything earlier is
  unknown. And in the current representation "unknown history" and "has
  bought nothing yet" are the SAME all-zero tensor. The model cannot tell
  them apart.

Consequence, declared in advance: a result here is a **diagnostic under
ideal information** — NOT a mathematical upper bound on live accuracy,
which is a different quantity this experiment does not measure. If the
result is positive, the follow-up work is an explicit known/unknown
indicator plus training-time truncation augmentation, and a real
late-attach evaluation — out of scope here, and named so a positive result
is not mistaken for a shippable one.

**Zeroing the history block does not return the parent.** The control arm
keeps the correction's own learned biases and still receives the state and
the basket vector. `mix_control` is the twin control for exactly that
reason; `mix_frozen` is the only true parent reference.

## Design — a residual correction, not two full trainings

Cheapest structure that can answer the question, mirroring the residual
economic experiment that worked:

1. **Frozen**: `econ_ft_ep8` and its ward-free mixture partner
   `wardfree_ep16`, both bound by sha256 from `frozen_mixture_v1.json`.
   Nothing about them is updated.
2. **Correction**: a small scoring head, **zero-initialised so step 0
   reproduces the parent exactly** (asserted: `max|correction| == 0` and
   the fixed-subset CE equals the parent's).
3. **Twin arms, identical in everything else** — same seeds, same batch
   order, same optimizer, same epochs, same domain:
   - `hist`: the correction receives the history block.
   - `control`: it receives that same block **zeroed**.
   CS, assists and inter-visit timings are deliberately LEFT OUT so the
   signal being tested is purchase history alone.

## History representation, and what it does not claim

- **Causal**: only visits strictly EARLIER than the decision, and only
  closed ones. `with_history` appends after the yield, so the property
  holds; it is re-asserted here.
- **Multiset**: duplicates preserved (`expanded_labels` repeats by count).
- **SAVE is not an item — and it WAS entering the window.** `with_history`
  extended the buffer with every `label_id`, sentinel 999999 included. It
  only became zero later, at embedding lookup, by which point it had
  already taken one of the HIST_LEN slots and could have evicted a real
  earlier purchase. Measured on 60,000 TRAIN rows: **2.1% carry it**. Now
  filtered BEFORE accumulating, and the caches were rebuilt.
- **This is a SET, not a sequence, and is tested as one.** The head adds
  the position embedding before a mean pool, so
  `mean(e_i + p_i)` over occupied slots `= mean(e) + mean(p over occupied)`:
  swapping two items between occupied positions gives an identical vector.
  Positions can inform HOW MANY slots are filled, never WHICH item came
  first. The pilot therefore asks whether the **set of the last 12
  purchased units** carries extra signal. It says nothing about the value
  of an ordered sequence, and no claim about click order or canonical
  order applies — order is not read at all.
- Identity is keyed on `(match_id, champion_id)`, which is a true
  participant key in this corpus — verified: 297,240 rows, 0 missing
  champion ids, max 10 distinct champions per game, 0 time regressions.
  It would break silently on mirror picks, so `(match_id, team_id, role)`
  is the safer key for any future corpus.

## Metrics

Both arms are scored alone AND mixed 50/50 with frozen `wardfree_ep16`
under the frozen mixture rule (canonical-key permutation, bijection
asserted, weights never tuned).

- **PRIMARY 1 — attribution**: `mix(hist)` vs `mix(control)`, ward-free
  `first_action_exact`, paired, bootstrapped by match.
- **PRIMARY 2 — practical progress**: `mix(hist)` vs the frozen mixture.
- **History-hiding probe (declared secondary, and NOT a late-attach
  simulation)**: the frozen `hist` candidate re-scored with history hidden
  for the earliest half of each game's decisions. After the cut it returns
  the FULL history, including purchases made before the supposed join, so
  it measures **sensitivity to hiding history** and nothing else. A real
  late-attach simulation would fix a join instant and accumulate only
  purchases observed after it, never recovering earlier ones. Kept under
  its true name and scope.
- Secondary: arms alone; any-of-three; slices single / multi /
  no_buy_death / ward_only; and the complementarity of each arm against
  `wardfree_ep16` (disagreement, each-only-correct, oracle union), since a
  better arm is not automatically a more complementary one.

**Reading rules, written before the numbers.** Both primaries compare
MIXTURES and are read as statements about mixtures only. PRIMARY 1
positive with PRIMARY 2 unresolved means improvement over the twin control
mixture with no demonstrated improvement over the frozen mixture — it does
NOT by itself establish overlap with the ward-free model, since a
regressed control produces the same pattern; `mix(control)` vs the frozen
mixture is reported alongside to separate the two. An arm-alone result is
never read as a product result.

A negative result rules out **this correction on this representation with
this frozen backbone**. It does not rule out purchase history — in
particular not a trained-from-scratch model with history, nor inter-visit
timings, nor CS and assists, each of which is a separate experiment.

## Budget and gates

Residual correction over a frozen backbone, so far cheaper than the
2.2 h twin fine-tune; the cache build is the main new cost and is measured
before the run rather than predicted.

Gates, in order: the FULL featurization fingerprint on both caches (not
just the history flag) plus `hist_nonzero > 0`; the caches must be
identical to the standard ones in every tensor except `hist`;
**the frozen-reference reproduction check runs BEFORE either arm trains**,
with normalization from the FULL FIT econ cache in every mode so the gate
has no built-in expected mismatch — two arms cost real GPU time and a
failed gate invalidates every comparison they would produce; zero-init
correction asserted; row identity keyed on `(match_id, ts, champion)`;
per-epoch curves appended and fsync'd; free disk re-checked each epoch.
One training process at a time.

**HIST_LEN stays at 12 for this pilot** (owner's call): it is a fixed
limitation of the representation, not a confound between arms — both arms
see the same window. Reviewer's full-TRAIN audit, SAVE excluded before
accumulating (`data/reports/history_design_review_20260922/`): **27.46%**
of FIT decisions and **26.95%** of probe decisions have MORE than twelve
prior units, i.e. are genuinely truncated; a further ~4.7% have exactly
twelve, which fills the window without losing anything.
Diagnostic only — no promotion, no epoch selection, no weight sweep.
