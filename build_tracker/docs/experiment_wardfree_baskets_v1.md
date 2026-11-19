# Pre-registration: ward-free basket vocabulary (v1)

2026-09-21 night. Owner's product decision, recorded verbatim before this
experiment: free (trinket) wards are the user's own decision and are never
suggested; control wards ("pinkwards", items 2055 / 772043) become an
OPTIONAL side suggestion — the exact basket is shown WITHOUT the pinkward,
and a pinkward is offered separately when legal, together with its
purchase probability. This experiment measures the modeling half of that
decision. Run dir: `data/experiments/16.18-wardfree-v1/`.

## Question

If wards are removed from the TASK ITSELF (catalogue, targets, decode) —
not just forgiven post-hoc in the metric — does a model trained on the
ward-free vocabulary beat the current models scored under the same
ward-free metric?

## Design

1. **Stripping rule** (the only change): remove items 2055 and 772043 from
   every basket key and every target multiset. A basket that becomes empty
   (ward-only purchase) maps to SAVE (catalogue id 0). Inventories, gold,
   candidates' economics and everything else untouched.
2. **Vocabulary**: rebuilt from FIT purchases after stripping (same
   frozen fit/probe membership, digests asserted). Input tensors are
   unchanged, so the v1 caches are reused; only targets and catalogue are
   rebuilt.
3. **Training**: the phase-2 Bf recipe unchanged — clean-init BasketNet,
   exact full-catalogue CE, microbatch 32 -> effective 256, Adam 1e-3
   cosine T_max=16, 16 epochs, seed 16, checkpoints 4/8/16, fixed-subset
   full-CE per epoch. Final (ep16) checkpoint is the result.
4. **Evaluation** (full probe, per-decision rows persisted): common
   decode — score stripped catalogue, top-128, walk to 3 legal options
   (purchase_sequence at gold_est on the stripped action), SAVE always
   legal. Ward-free metric: first_action_exact = predicted SAVE when the
   stripped truth is empty (no-buy AND ward-only rows), else exact
   multiset match of the stripped baskets.
5. **Baselines under the SAME metric, paired by game**: parent Bf_ep16,
   control fine-tune and econ fine-tune, their persisted enriched rows
   re-scored by stripping wards from their predicted options (save ->
   empty). PRIMARY: wardfree model vs econ_ft_ep8 (best observed) on
   ward-free first_action_exact. Secondary: vs parent; any-of-three;
   slices single/multi (stripped sizes), no_buy_death, ward_only
   (reported separately — their truth is now SAVE).

## Declared expectations and caveats

- The ~7.8-point gap between canonical and ward-insensitive scoring is
  NOT the expected gain: the baselines are already credited for it by
  re-scoring them under the same metric. The measurable question is
  whether removing ward noise from training and decoding helps beyond
  post-hoc forgiveness (vocabulary shrinks, per-size distributions
  shift, SAVE gains ward-only mass ~4.2% of decisions).
- The pinkward-probability product feature is NOT part of this
  experiment; if this vocabulary is adopted it will need a separate
  calibrated source (e.g. ward-containing basket mass from a model that
  still sees wards, or a dedicated head) — deliberately out of scope.
- Diagnostic, not promotion: internal probe only, test sealed, +5.0
  hurdle untouched. Null/negative published.
