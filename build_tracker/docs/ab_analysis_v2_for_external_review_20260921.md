# Whole-basket experiment, dossier v2 — phase-2 results for external review

Written 2026-09-21 (afternoon). Companion to `ab_analysis_for_external_
review_20260921.md` (v1 + its CORRECTIONS section). Context you already
have: your review of v1 found five implementation/dossier discrepancies (all
were verified true and published as corrections), and your assignment was:
correct the dossier, build a common evaluator with persisted per-decision
results, then run a bounded loss contrast (current sampled loss vs full-
catalogue CE) at fixed architecture and decoder, and only then choose among
learned cardinality, per-candidate economic features, or arm C.

That assignment has been executed in full. This dossier reports what came
out, including one result that inverts a belief we both held.

## 1. What was implemented from your review (all verified in code)

- Corrections published (architecture d64/2L/FF128/4H both arms; scorer
  inputs; top-128 fallback; missing metrics; weights-only checkpoint; 54.6%
  vs 55.2%; measured-wallet wording; sampled-softmax bias; inadmissible
  duplicate-rate comparison). Original text left intact for auditability.
- Common evaluator with canonical definitions taken from the production
  evaluator: `is_no_buy` = `save_kind == "no_buy_death"` (verified: on this
  probe, empty-basket rows and no_buy_death rows coincide 1:1, n=1,047);
  `action_family_match` = any first-basket item in {next completed final} ∪
  its components, on purchases with a known next final.
- Per-decision JSONL persisted for ALL policies on the same 58,781 probe
  decisions (`rows_A_full.jsonl`, `rows_Bs_full.jsonl`, `rows_Bf_full.jsonl`
  under `data/experiments/16.18-whole-basket-v2/`) — every interval and
  decomposition below is independently recomputable, and arm A's expensive
  real-predictor evaluation is persisted for reuse.
- Legality split you demanded: the executor is retried at 10,000,000 gold to
  separate `budget_blocked` from `structurally_blocked`.
- SAVE-fallback counter added (decode walks top-128; fallback = no legal
  candidate found).
- Full-state checkpoints (model/optimizer/scheduler/RNG) at epochs 4/8/16
  for both branches; per-epoch full-catalogue CE on a FIXED 10k FIT subset
  logged for both; wall time and peak GPU memory logged (peaks 2.27-2.69 GB
  — the v1 shared-memory spill did not recur).
- Your microbatching point was adopted verbatim: B-f trains with microbatch
  32 + gradient accumulation to effective 256 (not chunking).

Not changed, deliberately: architecture stays d64/2L (one variable per
phase — the loss), decoder unchanged, single seed (16) still a limitation,
max 16 epochs with no auto-extension as pre-registered.

## 2. Phase-2 design as run (16.18-whole-basket-v2)

Branches from IDENTICAL clean initialization (same seed, same initial
weights, optimizers configured identically): **B-s** = v1 sampled-softmax
(batch truths + SAVE + 2,048 uniform negatives, dedup union, no logQ);
**B-f** = exact full-catalogue cross-entropy (7,938 classes). Same fit
tensors (cached), same probe, same decode (top-128 walk, executor filter,
SAVE always candidate). Arm A unchanged from v1 (canonical two-stage recipe
at d64, real `predict_options` decode) — re-evaluated once under the common
evaluator. Total wall time 181 min on the RTX 4060.

## 3. Results (probe: 58,781 visits / 57,734 purchases / 500 games)

| Metric | A | B-s | B-f |
|---|---|---|---|
| first_action_exact | 0.3626 | 0.3968 | **0.4040** |
| any_of_three_exact | 0.4513 | 0.5770 | **0.5820** |
| action_family_match (n=51,815) | **0.6338** | 0.6124 | 0.6064 |
| mean pred size (buying) vs true 1.50 | 1.49 | 1.43 | 1.34 |
| overfill / underfill rate | .153 / .176 | .116 / .191 | .078 / **.230** |
| SAVE-fallback fired | 0 | 0 | 0 |

Paired by game (400 bootstrap):

- **B-f vs A: +4.15 points** [3.84, 4.51] — the new best; still under the
  +5.0 hurdle.
- B-f vs B-s: **+0.72** [0.48, 1.02] — your sampling-bias suspicion is
  CONFIRMED and now quantified: the biased objective costs about 0.7 points.
- B-s vs A: +3.42 [3.12, 3.71] (replicates v1's +3.43 under the common
  evaluator — good internal consistency).
- any_of_three, paired: B-s vs A +12.57 [12.27, 12.90]; B-f vs B-s +0.50
  [0.29, 0.72].

Convergence: both fixed-subset full-CE curves are still descending at the
epoch-16 cap (B-s 3.07→1.77; B-f 3.06→1.71; no plateau). Per the
pre-registration this phase is **inconclusive on convergence** and did not
auto-extend. Costs: B-s 90 s/epoch, B-f 425 s/epoch.

## 4. The inverted finding: the size bias is UNDERfill, not overfill

v1's 50 curated disagreements suggested B over-fills single-item truths.
You warned that curated examples cannot estimate the population bias. You
were right, and the population statistics reverse the sign:

- Arm A is nearly unbiased on size (pred 1.49 vs true 1.50).
- B-s under-predicts (1.43; underfill 19.1% vs overfill 11.6%).
- B-f under-predicts MORE (1.34; underfill 23.0% vs overfill 7.8%).

Mechanism, visible per size: moving from B-s to B-f gains +1,075 size-1
hits (17,831→18,906 of 35,097) but LOSES 557 size-2 hits (4,142→3,585 of
17,287) and 121 size-3. The exact likelihood concentrates probability on
frequent — hence small — baskets. So the loss fix that removed the sampling
bias also strengthened a frequency-concentration bias against multi-item
baskets. The exactly-one-extra rate falls A .132 → B-s .104 → B-f .069
(over-filling was never the dominant error at population level).

A second cost surfaced by the family metric: A (0.6338) beats both B
branches (B-s 0.6124, B-f 0.6064) on build-direction agreement. Arm A's
plan/target head supervision — which B lacks — appears to buy directional
alignment even when the exact basket is wrong.

## 5. Decomposition v2 (B-f; B-s in parentheses; denominator = 57,734)

| Bucket | n | % |
|---|---|---|
| hit | 23,166 (22,765) | 40.1% |
| ranking_miss | 18,161 (18,528) | 31.5% |
| budget_blocked | 14,701 (14,691) | **25.5%** |
| structurally_blocked | 507 (506) | 0.9% |
| catalogue_miss | 851 (851) | 1.5% |
| shortlist_miss | 348 (393) | 0.6% |

The split you demanded validates the original wallet attribution: ~96.6% of
the old `legality_block` is budget, 3.4% structural (slots/recipes). The
B-independent "winnable" first-exact for B-f: 23,166 / 41,675 = 55.6%.

## 6. Status vs the pre-registered rules

Improvement observed and grown (+4.15 best); +5.0 hurdle NOT met;
convergence INCONCLUSIVE (curves descending at cap); arm C still not
authorized; guardrails (save/ward) held in v1 and no_buy_death recall is
computed per-policy in the persisted rows; single-seed limitation stands;
the eventual confirmation split (frozen v3 validation) remains untouched by
this experiment but its prior exposure to Phase D and the POV run is
acknowledged — a fully independent confirmation needs a future window.

## 7. The phase-3 menu, now with measured motivation, and our questions

Mechanisms still failing, by size: ranking (31.5%), budget (25.5%, outside
the architecture), size-underfill (−557 size-2 hits in B-f vs B-s), family
direction (−2.7 pts vs A). Candidate levers: (a) your learned cardinality
factorization P(|B| | x)·P(B | x,|B|) — originally proposed against
overfill, now motivated by UNDERfill; (b) a target-family auxiliary head on
B (transplanting arm A's plan supervision); (c) hard negatives aimed at
ranking_miss; (d) more epochs (both curves still descending) as its own
pre-registered extension; (e) arm C item-semantic features.

Questions:

1. **Phase-3 composition.** With B-f > B-s settled and both unconverged, how
   would you structure phase 3 to combine "train longer" with ONE mechanism
   change without confounding them — sequential (extend B-f to convergence
   first, then contrast the mechanism at matched convergence) or factorial
   (2×2 at fixed epochs)? Budget: single 8 GB GPU, B-f ≈ 7 min/epoch.
2. **Cardinality under frequency concentration.** Given the underfill is
   driven by the exact likelihood concentrating on frequent small baskets,
   is the size-factorized form still your recommendation, or is a
   frequency-debiased objective (e.g., logit adjustment / per-basket prior
   offset) the cleaner fix? The two make different predictions about size-2
   recovery — which would you pre-register?
3. **Family auxiliary.** Is adding arm A's next-final target supervision as
   an auxiliary loss on B's trunk principled here, or does it contaminate
   the A-vs-B contrast (A's advantage partly BEING that supervision)?
4. **Hurdle jurisprudence.** +4.15 [3.84, 4.51] with convergence open: does
   the protocol's spirit permit a pre-registered convergence-extension phase
   whose result may cross +5.0, or does that constitute moving the
   goalposts? State the stopping rule you would write.
5. **Computations on the persisted rows.** You cannot run code here; we can.
   Name any statistic over `rows_{A,Bs,Bf}_full.jsonl` you want next
   (they contain per-decision: match_id, ts, champion, save_kind, true/pred
   size, first prediction, exact/any3/family outcomes, decomposition bucket,
   fallback flag).

## Appendix: provenance

Report `data/experiments/16.18-whole-basket-v2/phase2_report_full.json`
(elapsed 10,878 s); checkpoints Bs_ep{4,8,16}.pt / Bf_ep{4,8,16}.pt with
optimizer/scheduler/RNG; per-decision rows as listed above; protocol
`docs/experiment_whole_basket_v2_loss_contrast.md`; harness
`scripts/train_basket_phase2.py` (commit b4d8e1b; corrections commit
8c9d942; v1 chain c6e2ec7 → d83f0cb → 5fa0f24). Same generation
(e4ea6431…), same frozen fit/probe digests (d534b9f7… / f53db64a…).
Test split untouched; nothing promoted.
