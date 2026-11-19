# Pre-registration: per-candidate economic features via residual reranker (v1)

2026-09-21 evening. Next phase agreed with the external reviewer after the
extension's published negative result. Run dir:
`data/experiments/16.18-basket-residual-econ-v1/`.

## Binding condition (verbatim from the review loop)

Control AND candidate both start from the same frozen Bf_ep16 and receive
IDENTICAL additional tuning; ONLY the economic features differ. Any
difference then attributes to informing the scorer, not to another training
continuation.

## Motivation (measured, not assumed)

`ranking_miss` = 18,161 of 57,734 probe purchases
(phase2_report_full.json, Bf decomposition). These are historical baskets
that were in the catalogue, in the top-128, and legal at the observed
budget, yet not ranked first. They mix quantity/size mistakes, ambiguous
states and wrong-item choices — NOT 31.5 points of guaranteed recoverable
accuracy. The scorer (train_basket_ab.py `BasketNet.scores`) sees only
state/basket embeddings; it never receives what executing a candidate DOES
(cost after owned components, leftover gold, slots). This experiment
measures the value of that information, isolated.

## Design

1. **Frozen backbone.** Bf_ep16 (16.18-whole-basket-v2/Bf_ep16.pt) is never
   updated. new_score(c) = frozen_score(c) + correction(c), with
   correction defined ONLY on the parent's per-decision top-128 shortlist
   and identically zero elsewhere (that is the model class, so the
   normalizer below is exact, not an approximation).
2. **Correction network** (same for both arms, seed 16, identical init):
   MLP [state 128 | basket_vec 128 | econ 10] -> 128 -> 32 -> 1, GELU,
   final layer zero-initialized so both arms reproduce Bf_ep16 exactly at
   step 0. Frozen state and basket vectors come from the cache; gradients
   flow only through the correction.
3. **Arms.** Control: econ block replaced by a zero vector. Econ arm: real
   features, standardized with FIT feasible-buy statistics. Same parent,
   candidates, wallet, legality, batches, epochs, seed, LR schedule.
4. **Loss: exact full-catalogue CE preserved.** Since non-shortlist scores
   are frozen constants, per decision
   `Z = logsumexp([s_i + c_i for i in top128] + [tail_lse])` where
   `tail_lse` = frozen logsumexp of the remaining catalogue (cached).
   Target inside the shortlist -> corrected logit; target outside ->
   its frozen score, unchanged — never dropped, never inserted into
   candidates. No sampled softmax anywhere.
5. **Training budget (both arms identical, pre-declared):** Adam lr=1e-3,
   cosine T_max=8, 8 epochs, batch 4096 decisions over all FIT decisions.
   No extension, no best-picking: the final epoch is the result.

## Economic features (10 dims, pinned engine)

Computed with the SAME activated 16.18 economic engine the decoder uses
(`purchase_sequence` / `try_purchase`), preserving duplicates and purchase
ordering, at the decision's observed `gold_est` budget:

0. is_save (SAVE candidate flag; all other dims zero for SAVE)
1. feasible (purchase_sequence found a legal order at budget)
2. total recursive cost after consuming owned components (/1000)
3. gold remaining after execution (/1000)
4. fraction of wallet spent (clipped [0, 1])
5. number of purchase steps
6. number of PRE-OWNED components consumed (replay: consumed multiset
   intersected with pre-decision inventory)
7. value of those pre-owned components consumed (from-empty combine cost,
   /1000)
8. inventory slot-count delta (elements after − before)
9. completed items/upgrades produced (purchased ids with a recipe)

Infeasible buy candidates: feasible=0, rest zero. No hardcoded preference
for spending more or completing items — the features only describe.

## Cache (built once, gate first)

`scripts/basket_econ_cache.py` stores per decision: frozen state (fp16),
top-128 ids + frozen scores, tail_lse, target shortlist position + frozen
target score, econ features [128 x 10]; plus the frozen basket-vector
matrix once, probe tail top-64 (ids + frozen scores) for decoding, and a
row-alignment digest (sha256 over match_id|ts in stream order).
**Gate (reviewer's requirement):** CPU feature throughput is benchmarked on
a real sample of decisions BEFORE the full build and the measured
ms/decision (= the real added inference cost for 128 candidates) is
reported. If the projected cache build exceeds 8 h, stop and report
instead of building.

## Decoding and evaluation

Merged ranking per probe decision: corrected top-128 plus frozen tail
top-64 (a tail candidate can overtake a down-weighted shortlist one);
walk in score order, same legality (cached feasible flags for the 128,
lazy purchase_sequence for reached tail candidates), first 3 legal options,
SAVE always legal. At zero correction the first action is identical to the
parent's by construction (parent fallback_save_n = 0).

Report, on the same full probe (58,781 decisions):

- Previously wrong decisions corrected AND previously correct decisions
  broken vs the parent rows (join on match_id|ts) — net, not just gains.
- Paired-by-game first_action_exact CIs: econ vs control (PRIMARY),
  econ vs parent, control vs parent.
- Single-item vs multi-item exactness separately; family, SAVE recall,
  ward-only, duplicate-basket exactness; decomposition; measured latency
  (feature ms/decision + correction forward).
- 30 inspectable examples where the econ arm changes its first action,
  with candidate costs, remaining gold and inventory changes.

## Success criteria (pre-declared)

Primary: econ − control > 0 on first_action_exact with 95% paired CI
excluding zero. Secondary: econ − parent > 0. A null or negative result is
published as-is. This is a feature-value diagnostic: NO promotion, test
stays sealed, the +5.0 promotion hurdle is not in play here. If economics
helps, end-to-end fine-tuning may be proposed as a SEPARATE pre-registered
step; if not, remaining disagreements guide one targeted intervention.

## Why this is not the failed reranker again

The earlier reranker could only reorder 3 already-displayed cards. Here up
to 128 complete baskets compete under exact full-catalogue CE, and the
scorer receives information it demonstrably lacks today.
