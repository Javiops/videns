# Pre-registration: learned gate over the two frozen models (v1)

2026-09-23 ~01:30. Owner's order ("haz 1-4"), item 4. Run dir:
`data/experiments/16.18-learned-gate-v1/`. Internal probe only; VAL and
test are NOT read.

## Question

The two frozen models disagree on 13,189 probe first recommendations; a
truth-knowing selector over their first options would reach 53.5% while the
fixed 50/50 mixture reaches 49.5%. Can a small gate that chooses HOW MUCH to
trust each model per decision, from information available at decision time,
beat the fixed mixture?

## Design

- Frozen: econ_ft_ep8 (grouped) and wardfree_ep16, the frozen mixture's own
  components, bound by digest. The gate never touches them.
- Gate input (8 features, all available at decision time): max and entropy
  of each model's action distribution, whether their top-1 agree,
  gold_est/1000, level/18, n_inventory/8. Output: w in (0,1);
  P = w*P_A + (1-w)*P_B. MLP 8 -> 32 -> 1, sigmoid.
- Loss: -log(w*P_A(y) + (1-w)*P_B(y)) on decisions whose true main purchase
  is in the action space; Adam 1e-2, 300 full-batch steps. Fixed here, not
  tuned.
- CONTROL: identical gate with all inputs zeroed -- it can only learn a
  constant w. Separates "a per-decision gate helps" from "a different
  constant than 0.5 helps".
- Reference: the fixed 50/50 mixture's persisted rows (rows_M_full.jsonl).
- Where it is trained, stated plainly: the parents were trained on FIT, so
  their FIT probabilities are overconfident and a gate trained there would
  learn which parent overfits more. The gate is therefore trained
  OUT-OF-FOLD ON THE PROBE: 5 folds by match, each decision scored by a gate
  that never saw its game. The probe is also the selection set; the result
  is an out-of-fold estimate on it, and this caveat travels with the number.

## Metrics and reading rule

- PRIMARY: ward-free first_action_exact, gate vs fixed mixture and gate vs
  control, paired, bootstrapped by match.
- Secondary: any-of-three; slices; movement; the learned w distribution and
  the control's constant per fold.
- Gate over control positive with gate over fixed unresolved reads as "a
  per-decision gate beats a learned constant, not the fixed mixture" -- and
  control vs fixed is reported alongside so a regressed control cannot be
  mistaken for overlap. No hyperparameter is changed after seeing a number;
  a null is published as one.
