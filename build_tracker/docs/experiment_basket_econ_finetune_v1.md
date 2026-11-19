# Pre-registration: twin economic fine-tune on a fixed feature domain (v1)

2026-09-21 night. Approved design from the review loop after the residual
contrast (+0.20/+0.21 across two seeds) and its audit. Run dir:
`data/experiments/16.18-basket-econ-finetune-v1/`.

**Question (delimited by the reviewer):** does adapting the model WITH
access to economics on a fixed domain improve over the SAME adaptation
without that information?

## The six binding conditions (reviewer's protocol, fixed before launch)

1. **Frozen feature DOMAIN, not frozen scores.** D(x) = top-128 of frozen
   Bf_ep16 for each decision x — the already-cached, seed-independent
   shortlists from `16.18-basket-residual-econ-v1/`. Identical in both
   arms, never refreshed, never widened with the true purchase. The cached
   econ features stay valid because snapshot, budget, basket and economic
   engine are unchanged even as neural representations drift.
2. **Shared membership indicator.** score = current_base_score +
   member_of_D x correction. SAME correction net in both arms (input:
   current state repr | current basket vec | econ block); control receives
   the econ block zeroed, candidate receives real standardized features.
   Outside D the correction is zero in both arms, so membership
   information is symmetric — we measure economics, not a parent-priority
   side channel. Correction output zero-initialized; **verified before the
   first update that both arms reproduce the parent** (fixed-subset
   full-CE equal to the parent's, and max |correction| = 0).
3. **Full denominator recomputed with gradients.** Full-catalogue CE per
   decision over all n_cat scores from the CURRENT model (phase-2
   microbatching: 32 -> effective 256). No reuse of frozen tail_lse,
   embeddings or scores. Targets outside D are NOT dropped — they
   participate through their updated base score.
4. **Feature domain != action domain.** Baskets outside D remain
   recommendable (they just lack the correction). Decoding rule and
   legality identical in both arms: full corrected scores -> top-128 ->
   walk to 3 legal options, SAVE always legal. D is never a hard mask.
5. **Inference contract reproduced and measured.** Online, a new state
   requires the FROZEN parent forward to obtain D(x) (using the tuned
   model's own top-128 would change the contract). Latency is reported as
   parent forward + economic features + tuned forward, measured on a real
   sample and labeled an offline estimate, not an online p95.
6. **Domain-limitation diagnostics (only these two, no refresh in this
   contrast):** proportion of final first options / top-3 inside D; mean
   probability mass the tuned model assigns outside D.

## Training budget (identical in both arms, pre-declared)

Both arms start from the same Bf_ep16 full state. Fresh Adam lr=1e-4
(declared: a NEW segment; the extension already showed fresh 5e-4 hurts
this parent), cosine T_max=8, 8 epochs, seed 16, same batch order.
Full-state checkpoints at +4/+8; **the final (+8) checkpoint is the
result** — no best-picking, no extension. Fixed-subset full-CE (the
phase-2 10k subset) logged per epoch for both arms, correction included.

## Report (both arms + parent in everything)

Same full probe, common-evaluator records per decision, join key
(match_id, ts, champion): first_action_exact aggregate + paired CIs
candidate-vs-control (PRIMARY), each-vs-parent; corrected/broken/net;
cost slices single, multi, SAVE/no_buy_death, ward_only; family;
decomposition; the two domain diagnostics; latency components; 30
inspectable changed-decision examples. Null or negative -> published.

No promotion; test sealed; the +5.0 hurdle is not in play. The product
trade-off (single/SAVE/wards vs composition) stays an open decision and
is not settled by this experiment.
