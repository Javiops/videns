# Experiment: POV-observable board gold (16.18-pov-gold-v1)

Registered before execution, per the run-experiment protocol. Owner
authorization: 2026-09-20 — the owner ruled that training inputs must be
limited to the buyer's point of view ("recuerda los POVs") and instructed
execution ("la key está viva, dale").

## Question

Does replacing the two truth-layer board gold features — `team_gold_diff` and
`opp_gold_diff`, both computed from Match-V5 `totalGold` of all ten players,
which no live player can see — with the POV-observable estimate (item value of
the ten visible builds plus the buyer's own wallet, `PREFIX_POV_GOLD=1`)
preserve a fixed-recipe candidate's advantage over the conditional lookup on
the 16.18-reconstructible-development-v3 generation?

This measures the COST of the owner's POV rule. The rule itself is not
conditional on the result: spectator-only knowledge must not featurize. If the
cost is large, that is a finding about how much omniscient gold was carrying,
not a reason to keep it.

## Design

- Generation: `16.18-reconstructible-development-v3` (fingerprint
  `e4ea6431...`), UNCHANGED. Same export, same train/val windows as Phase D.
- Recipe: the fixed development recipe (trunk 16 epochs, frozen-trunk graft 6
  epochs), identical to Phase D except `PREFIX_POV_GOLD=1`. One knob moved.
- Run directory: `data/experiments/16.18-pov-gold-v1/` (isolated, new).
- Primary metrics: `first_action_exact` (all scored validation decisions) and
  `action_family_match` (validation purchases with a known next final).
- Evaluation: exactly one paired comparison against this generation's own
  conditional lookup, full validation (`--games 0`), seed 16, bootstrap 400.

## Pre-registered expectations (written before any result)

- The candidate must beat the conditional lookup with paired intervals
  excluding zero on both primary metrics, and zero unaffordable displayed
  options.
- Phase D's margin over the same lookup was **+0.1512** first_action_exact.
  Expected POV margin: **>= +0.10**. The difference between the two margins is
  the POV cost estimate — same generation, same baseline, but NOT a paired
  A-B; quote it as suggestive.
- If the POV margin exceeds +0.1512, treat with suspicion and audit before
  celebrating (removing information should not help; a gain suggests the
  truth-layer features were injecting noise or the run differs somewhere else).

## Stopping rule

One fixed trunk+graft training and exactly one paired conditional-baseline
evaluation; stop on any failed gate; no sweeps, no promotion, no test reads,
no cross-export claims.

## Explicitly out of scope

- `PREFIX_GOLD_MEASURED` (macro_coach measured wallets): the v3 export's games
  all pre-date the measured 40 (Sept 17-19), so the import would join zero
  rows. Measured-wallet training/eval needs the next generation whose window
  includes those games; the plumbing is in place and plan-bound when used.
- Promotion or serving changes of any kind.

## Resource budget

max 1 training process, 10 GB RAM, 40 GB free disk floor, 16 h wall.
