# Whole-basket A/B experiment — complete analysis for external review

Written 2026-09-21 for a second-opinion review by an external model. This
document is self-contained: it states the system, the data contract, both
arms, every headline number, the failure decomposition, the validity caveats
we know about, and the concrete questions we want challenged. Nothing here is
hidden context; if a claim lacks a number, treat it as unmeasured.

## 1. What the product is and what is being predicted

Build Tracker is a behavioural-imitation shop advisor for League of Legends:
at a shop visit, show what high-elo (Challenger-ladder) players historically
bought in a similar reconstructed state. It does NOT optimise outcomes and
never claims "optimal builds" — the target is agreement with what the human
actually bought. A "visit" is a burst of shop events (10 s idle window) with
reconstructed pre-decision state: own inventory, wallet estimate, level, CS,
KDA, runes, and a 10-player board snapshot (champions, positions, visible
items, levels). Labels are complete purchase multisets ("baskets"), duplicates
preserved. A post-death no-buy is an explicit SAVE action; ward-only baskets
are ordinary baskets.

Hard project rules relevant here: features must be observable from the
buyer's point of view (POV — no spectator-only knowledge; the board gold
totals of other players were removed after an ablation measured their
removal at −0.0005 first_action_exact, i.e. zero); training-log metrics are
never quotable (a historic 0.63 "accuracy" from train logs against canonical
labels caused a public retraction — the honest displayed-policy number for
the production model is 0.3811 on its validation split); the test split is
sealed; nothing is promoted without a separate human-gated evaluation.

## 2. Motivating diagnostics (frozen, independently reproduced)

On the training corpus (291,819 purchases / 2,500 games):

- Only **8,721 distinct baskets** exist; **837 cover 90%** of purchases;
  singleton baskets are 3,922 types = 3,922 rows (1.3%).
- Chronological probe (earliest 2,000 games = "fit", latest 500 = "probe";
  memberships frozen by SHA-256 digests): **98.5%** of probe baskets already
  appear in the fit catalogue; of probe targets that are affordable+legal at
  the estimated wallet, **99.1%** are in the catalogue.
- Only **33.0%** of probe purchases have an exact (champion, role, inventory)
  precedent — global coverage is high but exact-context coverage is low, so
  a selector must transfer across related contexts.
- **26.6%** of sampled probe targets are ILLEGAL at the estimated wallet
  (`gold_est`, a causal reconstruction; participant wallet frames are 60 s
  stale) — i.e. the player's true basket could not have been recommended
  under our budget estimate. Median measured shortfall in earlier audits:
  ~74 gold.

These diagnostics were produced by one assistant, then ported and reproduced
EXACTLY by another (byte-identical sample file, all compared fields equal),
so treat them as solid.

## 3. The hypothesis under test

The incumbent model scores ITEMS independently (sigmoid per item + a copy-
count head) and assembles a basket with greedy rules under budget/legality.
Independent thresholds can assemble combinations nobody buys (players buy AB,
AC, or BC; every item scores high; the assembly recommends ABC). The
challenger learns a joint score over COMPLETE baskets from the fit catalogue.

Pre-registered protocol (before any tensor): arms from clean initialization
on the same 2,000 fit games; the 500 probe games are internal development
only (the production checkpoint has trained on them, so it is NOT a valid
unseen comparator and does not appear here); primary metric
first_action_exact; decision hurdle **B − A ≥ +5.0 points** with a
game-paired bootstrap CI excluding zero, zero illegal displayed baskets,
save/ward guardrails within 2 points of arm A; an arm may not be declared
failed without a convergence check (loss plateau + no metric improvement
across ≥3 checkpoints).

## 4. Arms (both POV inputs, both the same corrected wallet contract)

Shared featurization (identical tensors): 10-player board tokens (champion +
side + mixed item embeddings) through a 4-layer transformer (d=128, ff=256,
8 heads), a query token carrying wallet/affordability/level/objectives/rune
features, POV-observable gold diffs (visible item value ± own wallet — same
resolved budget as the legality mask; a bug where these two disagreed was
fixed before tensor construction).

- **Arm A (incumbent formulation):** the production architecture and losses
  (per-item BCE with rarity-scheduled pos_weight 8:24, copy-count CE 0.25,
  then a frozen-trunk graft of save/plan heads: 16 + 6 epochs — the exact
  production recipe), trained on fit only, saved as a standard artifact and
  decoded through the REAL production path (`predict_options`: plan probe,
  recipe-consistency filters, greedy budget-constrained assembly, save card),
  exactly as the production evaluator scores it.
- **Arm B (whole-basket scorer):** the SAME trunk architecture from clean
  init + a permutation-invariant basket encoder (Deep Sets: per-element MLP
  over item+count embeddings, sum-pool, post-pool MLP) over the 7,937-basket
  fit catalogue, with SAVE as an explicit catalogue action (learned vector).
  Score = bilinear(state, basket) + deep MLP on [state, basket, state⊙basket].
  Training: sampled-softmax NLL (candidates = batch truths + SAVE + 2,048
  uniform negatives), 16 epochs, Adam 1e-3 cosine. Evaluation: score the FULL
  catalogue (chunked), walk the ranking, keep the top 3 candidates that pass
  the real recursive-economy executor at the estimated budget (SAVE always
  legal). Displayed illegality is therefore zero by construction.

Ops note that changed one thing mid-experiment: the first full run trained B
with FULL-catalogue softmax; it was numerically healthy (loss 4.09→2.05 over
9 epochs, still descending) but materialized [256×7938×384] intermediates
that spilled ~7 GB into shared system memory on the 8 GB GPU (~75 min/epoch;
killed by the owner at epoch 9/16 after 11 h). The rerun switched to the
sampled-softmax objective above (which is what the written protocol
specified) plus chunked eval scoring; total rerun wall time 85 min. So: B's
reported result is the sampled-objective run; the full-softmax run produced
no evaluation.

## 5. Headline results (probe: 58,781 visits, 57,734 purchases, 500 games)

| Metric | Arm A | Arm B | Δ (B−A) |
|---|---|---|---|
| first_action_exact (all visits) | 0.3626 | **0.3969** | **+3.43** [3.13, 3.74], paired bootstrap 400, 500 games |
| any_of_three_exact (purchases) | 0.4513 | **0.5774** | **+12.61** |
| no_buy_death_recall (n=1,047) | 0.4527 | **0.5263** | +7.4 (guardrail: pass) |
| ward_only first-action (n=2,487) | 0.6060 | **0.6224** | +1.6 (guardrail: pass) |
| illegal displayed baskets | 0 | 0 | — |

**Hurdle verdict: NOT met on the primary** (+3.43 < +5.0, CI excludes zero).
Secondary any_of_three massively exceeded. Guardrails passed (B better on
both). Note arm A's 0.3626 here is not comparable to the production model's
0.3811 (different training set size — 2,000 vs 2,500 games — and different
evaluation population).

## 6. Failure decomposition of arm B (57,734 probe purchases)

| Bucket | n | % | Meaning |
|---|---|---|---|
| hit (first choice exact) | 22,780 | 39.5% | |
| ranking_miss | 18,518 | 32.1% | truth in catalogue, in top-128, legal — B ranked something else first |
| legality_block | 15,195 | 26.3% | truth basket ILLEGAL at `gold_est` — unwinnable for any selector under this budget |
| catalogue_miss | 851 | 1.5% | truth basket absent from the fit catalogue |
| shortlist_miss | 390 | 0.7% | in catalogue but outside B's top-128 |

On winnable purchases (excluding legality/catalogue/shortlist losses), B's
first-choice exactness is **55.2%** (22,780 / 41,298).

Per basket size (n / B first-exact): size 1: 35,097 / 50.9%; size 2:
17,287 / 24.0%; size 3: 4,720 / 15.3%; size 4: 604 / 9.6%; size 5: 26 / 0%.
Duplicate-copy baskets (n=3,900): 28.6% exact (the incumbent's exact-on-
duplicates diagnostic historically ran ~0.23 on its own logs).

The legality_block share (26.3%) independently reproduces the frozen
diagnostic's 26.6% — the budget estimate, not the selector, caps roughly a
quarter of the metric. Exact wallets measured from replays exist for 40
games (~9,700 wallets, verified to ≤10 gold against ground truth), but those
games post-date this corpus window (join verified = 0 rows here); they
become usable as an evaluation stratum on the next data generation.

## 7. Qualitative pattern from the 50 sampled disagreements (25 B-wins /
25 A-wins, drawn where exactly one arm was right)

B wins are dominated by COMPOSITION: A predicts the lead item alone (or with
the wrong companion) while B completes the real basket — e.g. Rell 11'
truth {control ward 2055 + Plated Steelcaps 3050}: A said {3190+2055}, B
exact; Orianna 32' truth {2055 + Rabadon 3089}: A said {3089} alone, B
exact; Shen 16' truth {1029 + 3748}: A said {3748} alone, B exact.

A wins are dominated by B OVER-FILLING single-item truths: Neeko 8' truth
{3145}: B added a Control Ward-adjacent extra {2022+3145}; Yunara 16' truth
{3085}: B said {1036+3085}. B has a basket-size over-prediction bias on
1-item visits (consistent with size-1 exactness being its weakest ratio
relative to volume).

## 8. Validity caveats we already know (challenge these first)

1. **Convergence unchecked**: B's per-epoch losses from the counted run were
   lost to a transcript-buffering quirk; the earlier full-softmax run was
   still descending at its kill point. The protocol forbids declaring an arm
   failed without a convergence check; a warm-started extension (epochs are
   ~90 s now) has not yet been run. The +3.43 could be an undertrained lower
   bound.
2. **Objective/eval mismatch**: B trains on sampled softmax but is evaluated
   by full-catalogue ranking. Standard, but the 2,048-uniform-negative choice
   (no frequency weighting, no hard negatives) is unexamined.
3. **Probe is development, not validation**: repeated looks at these 500
   games will overfit decisions to them. The real confirmation is one paired
   run on the frozen validation split, which has NOT been touched by this
   experiment.
4. **Arm A's decode is production's, arm B's decode is new**: B's "top-3" is
   rank-then-filter; A's is a product policy (lead + lead-banned replan +
   save card). any_of_three comparisons across different option-construction
   policies favour whichever policy diversifies better — B's +12.6 is real
   but partly a decode-policy difference, not purely representation.
5. **Single seed, single split**: no seed variance estimate.
6. **legality_block is measured at `gold_est`**: some of those 26.3% are
   budget-estimation artifacts, not true unaffordability; the measured-wallet
   eval stratum (next generation) is the instrument to split that.

## 9. Levers on the table (with their measured ceilings)

- Budget/legality: 26.3% of purchases capped by the wallet estimate; an
  oracle-exact-wallet audit on the production model measured +3-6 points of
  first-action headroom from budget alone (quantile-gold modeling is the
  planned attack; replay-measured wallets cover too little training mass to
  train on, 0.4% effective).
- Ranking (32.1%): more epochs, hard negatives, item-semantic features in
  the basket encoder (the planned arm C), size-calibration to fix the
  over-fill bias.
- Catalogue/shortlist (2.2% combined): near-exhausted; not the bottleneck.

## 10. Questions we want the reviewer to answer

1. Given +3.43 [3.13, 3.74] vs a pre-registered +5.0 hurdle, with +12.61 on
   any_of_three and both guardrails improved: does the evidence justify
   building arm C (item-semantic basket features), or does discipline demand
   stopping at the hurdle? What would you pre-register for C?
2. Is the convergence extension (16→32/48 epochs, warm start) the right next
   spend, and what stopping rule would you set on it?
3. How would you correct B's basket-size over-prediction on single-item
   visits without a hand-tuned size prior (options: length-normalized
   scoring, explicit size head, calibration on fit)?
4. Does the decode-policy asymmetry (§8.4) invalidate the any_of_three
   comparison, and should arm B adopt a lead-banned-replan second option to
   make it apples-to-apples?
5. Negative sampling: would frequency-weighted or hard (same-champion,
   affordable) negatives be expected to move ranking_miss materially at this
   data scale?
6. The 26.3% legality_block: argue for or against evaluating/reporting a
   companion metric at an interval-upper-bound budget (mask reaching the
   wallet interval ceiling) as the honest way to stop punishing the selector
   for the estimator's errors.
7. Any methodological flaw you can find in the decomposition or the paired
   bootstrap (by-game resampling, 400 draws) that would change the verdict.

## Appendix: provenance

Experiment id 16.18-whole-basket-v1; protocol docs/experiment_whole_basket_
v1.md; generation 16.18-reconstructible-development-v3 (export fingerprint
e4ea6431…); fit/probe membership digests d534b9f7… / f53db64a…; frozen
diagnostics data/reports/strategy_analysis_20260920/ (legality sample sha
4806447777d5…); report data/experiments/16.18-whole-basket-v1/ab_report_
full.json; examples ab_examples_full.json; commits c6e2ec7 (protocol+probes+
pilot), d83f0cb (harness), 5fa0f24 (VRAM fix + sampled objective). Test
split untouched; nothing promoted.

---

## CORRECTIONS (2026-09-21, after external review — the reviewer was right)

Verified against artifacts and the harness; the original text above is left
intact so the correction is auditable:

1. **Architecture**: both arms trained at **d64, 2 layers, FF128, 4 heads**
   (module defaults — the harness set feature-flag envs but not the
   architecture envs). NOT d128/4L/FF256/8H as §4 claims, and arm A is
   therefore NOT the canonical production size. The internal A/B contrast
   stands (shared trunk); absolute numbers and any comparison to the
   production candidate do not.
2. **Arm B scorer inputs**: state, basket embedding, and their elementwise
   product ONLY. Recursive purchase cost / leftover wallet were proposed but
   are NOT implemented; economy only filters post-hoc.
3. **Arm B decode**: walks the top-128 only; if none legal, SAVE is counted
   as the first action. That fallback's frequency is unmeasured in v1.
4. **Not computed in v1**: action_family_match; an independent legality
   audit artifact of displayed options; paired CI for any_of_three.
5. **Arm B checkpoint**: weights only (no optimizer/scheduler/RNG/
   intermediate checkpoints) — any continuation is a new tuning phase.
6. **§6 "winnable" figure**: 55.2% conditions on B's own shortlist; the
   B-independent denominator (excluding only legality_block and
   catalogue_miss) gives 22,780/41,688 = **54.6%**.
7. **Measured wallets wording**: 7,557 exact + 2,220 intervals; the ≤10-gold
   anchor validation certifies the resolution process, not that all ~9,700
   are exact.
8. **Sampled-softmax bias**: the training loss is CE over the deduplicated
   union of batch truths + SAVE + 2,048 uniform negatives, with no
   inclusion-probability (logQ) correction; SAVE is always present and
   frequent baskets are over-represented via batch truths. This is a valid
   contrastive loss but not the catalogue likelihood, and it is the leading
   suspect for the size-over-prediction bias. The chunked scorer does NOT
   reduce backward memory (autograd retains all chunks' activations); a
   full-CE contrast requires microbatching + gradient accumulation.

**v1 verdict as recorded**: improvement observed (+3.43 [3.13, 3.74]
first_action_exact; +12.61 any_of_three without paired CI); +5.0 investment
hurdle NOT met; convergence check pending; arm C not authorized by this
result.
