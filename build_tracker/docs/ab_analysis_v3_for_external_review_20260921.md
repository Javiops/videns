# Whole-basket dossier v3 — prework + size diagnostics, per your precisions

2026-09-21 afternoon. Companion to v1 (+corrections) and v2. Your last two
messages asked us to (1) not infer the cardinality intervention from the
identity or from aggregate mass calibration, (2) wait for the prework, and
(3) run the two diagnostics with your precisions: context strata (not just
confidence bins), oracle-size labelled as privileged, calibration
conditioned on buying with SAVE separated, and pre- vs post-economic-filter
masses. All executed; per-decision rows and reports persisted.

## 1. Checkpoint generalization curves (probe, common evaluator)

| ckpt | exact | any3 | multi-item exact | family | mean pred size |
|---|---|---|---|---|---|
| Bs ep4 | 0.3685 | 0.5504 | 0.1800 | 0.6039 | 1.404 |
| Bs ep8 | 0.3850 | 0.5660 | 0.1931 | 0.5976 | 1.393 |
| Bs ep16 | 0.3968 | 0.5770 | 0.2180 | 0.6124 | 1.430 |
| Bf ep4 | 0.3734 | 0.5522 | 0.1391 | 0.5713 | 1.270 |
| Bf ep8 | 0.3947 | 0.5730 | 0.1684 | 0.5981 | 1.309 |
| Bf ep16 | 0.4040 | 0.5820 | 0.1882 | 0.6064 | 1.339 |

Every metric still rising at the cap for both branches — generalization, not
just FIT loss. Bs leads multi-item at every checkpoint; Bf leads exact.

## 2. Size-mass calibration, conditioned on buying (SAVE separated)

Aggregate (probe purchases, n=57,734): Bf mean masses k1/k2/k3 =
0.599/0.294/0.080 vs empirical 0.597/0.294/0.080 — essentially exact.
Bs = 0.514/0.354/0.105 — over-weights multi. Argmax frequencies: Bf
0.697/0.228 (mode-concentrated), Bs 0.606/0.299 (matches empirical).

**Per-context (your point that aggregate proves nothing):** eleven
pre-declared strata (role x game phase x wallet staleness), gap =
observed − predicted P(multi | buy), per-game bootstrap CI:

- **Bf: every stratum gap within ±0.013**; 9 of 11 CIs contain zero (the
  two wallet strata are significant at −0.013/+0.010 — negligible
  magnitude). ECE 0.026.
- **Bs: every stratum over-predicts multi by 0.076-0.095, all 11 CIs
  exclude zero.** ECE 0.088. Its multi-item advantage is a systematically
  miscalibrated bet toward multi, paying global exactness for coverage.

## 3. Oracle-size (privileged information, as you framed it)

Argmax within the TRUE-size catalogue slice, legality at the ORIGINAL
budget, truth never inserted: Bs 0.4894 global / 0.6737 feasible;
Bf 0.4918 / 0.6770 (feasible n=41,938, your corrected denominator).
Knowing the size would add ~+12 points on feasible for either model.

## 4. Pre- vs post-economic-filter masses

Bf k1 mass 0.609 pre-filter → 0.675 post-filter (legal top-128,
renormalized; partial, declared). Part of the small-basket tilt at decision
time is the WALLET filter removing legal multi options — your warning about
misreading wallet errors as cardinality errors, confirmed in data.

## 5. Reading against your decision tree

Bf lands in branch (c): size probabilities well-calibrated per context AND
per confidence; the argmax small-basket preference is decision-theoretic
behavior under exact-match utility, plus a wallet-filter contribution. With
the oracle gain large but the per-context calibration clean, the residual
size uncertainty looks mostly irreducible with current inputs. Bs is the
complementary case: a miscalibrated multi-bet that happens to serve
composition coverage.

**Decisions taken accordingly:** the control-vs-cardinality contrast is NOT
executed — its justification (a size-assignment defect) is not supported;
recorded as such. The product-utility question (optimize per-decision
exactness vs composition coverage — which would be a DECODER/display
choice, e.g. size-diverse top-3, not a loss change) is escalated to the
owner. What IS launched, pre-registered: a closed 8-epoch training
extension of both branches from their full-state ep16 checkpoints
(motivated by §1's rising curves; fresh Adam lr=5e-4 cosine T_max=8 —
the parent scheduler ended at ~0 so the new segment is declared; final
checkpoint is the result, no best-picking, no further extension; paired
vs parents and vs A on persisted rows). One honest smoke observation
pre-declared: the first continuation epoch RAISES fixed-subset CE (fresh
optimizer loses the parent's moments); if the final lands worse than the
parent, that is the reported result.

## 6. Questions for this round

1. Any objection to the branch-(c) reading for Bf, or residual test you'd
   demand before accepting "mostly irreducible with current inputs"?
2. If the owner chooses composition coverage as a product goal: is a
   size-diverse top-3 decoder (e.g., best overall + best multi-item legal)
   the right mechanism, and how would you evaluate it honestly (any3 is
   already 0.582 for Bf)?
3. Ranking-within-size remains the largest recoverable bucket (31.5%).
   Between hard negatives (same-lead perturbations) and per-candidate
   economic features (recursive cost, leftover wallet) — which first, and
   what would you pre-register?
4. The extension's LR continuation choice (fresh Adam 5e-4 cosine-8):
   would you have pre-registered differently, and does the smoke
   observation change expectations?

## Provenance

Reports: `data/experiments/16.18-whole-basket-v3-prework/
{prework_report_full.json, size_diagnostics_full.json}` (reading rules
embedded); enriched rows for both finals with all three options + truth;
extension protocol in `scripts/basket_extension.py` docstring (commit
pending alongside this dossier); prior chain c6e2ec7→d83f0cb→5fa0f24→
8c9d942→b4d8e1b→98990a0→8fc39e3. Same generation, same frozen digests.
Test sealed; nothing promoted; cardinality contrast not run.

---

## PRECISIONS ACCEPTED (round 5, 2026-09-21 ~15:00 — reviewer verified right)

1. **Branch (c) downgraded to the reviewer's wording, now the official
   verdict**: "no large mean cardinality mismatch in the examined groups;
   this weakens the justification for automatic size correction, but does
   not establish irreducibility or optimality." Verified from our own
   report: Bf's top confidence bins are overconfident (bin 0.8-0.9:
   predicted 0.8515 vs observed 0.7940, n=4,165; bin 0.9-1.0: 0.9534 vs
   0.9094, n=5,709); the strata are marginals, not intersections; and
   calibration does not equal exploiting all available information.
   "Mostly irreducible" is withdrawn.
2. **Dossier figure corrections**: the 0.599/0.294/0.080 masses were over
   all 58,781 decisions including SAVE (conditioned Bf ≈ 0.609/0.299/0.081
   — aggregate closeness stands); the oracle's within-size top-128 changes
   candidate access relative to the global top-128 (not a pure cardinality
   isolation); the 0.609→0.675 post-filter shift bundles truncation +
   all economic restrictions + renormalization (686 purchases with no
   legal buy candidate contribute zeros) and does not isolate wallet error
   specifically.
3. **ranking_miss premise corrected**: the 31.5% includes size errors; it
   is not a within-size-only measurement nor proven recoverable.
4. **Extension framing**: this run measures "an additional phase with an
   optimizer restart", not "8 more epochs unchanged"; the smoke CE rise has
   at least two candidate causes (lost Adam moments OR the LR jump vs the
   annealed parent) and neither is established. Design unchanged mid-run;
   the final checkpoint will be published against the parent even if it
   loses. The reviewer would have considered warm-kept moments at 1e-4 —
   recorded as an alternative continuation recipe, not evidence.
5. **Agreed next steps after the extension**: (a) pre-registered
   per-candidate ECONOMIC-FEATURES contrast (recursive execution cost,
   leftover gold, inventory/slot deltas as scorer inputs; same parent,
   candidates, budget, decoder; control without; nothing else changed;
   inference cost measured on a real sample first — no forward-only
   extrapolations); (b) the decode-only size-diverse top-3 as a PRODUCT
   variant measured paired (first option identical by construction; third
   slot replaced by best legal multi-item when top-3 has none), with no
   claim that it improves any_of_three; (c) cardinality stays parked;
   B-s kept as a reference with different behavior, not dismissed.

## EXTENSION RESULT (published even though it loses, as pre-registered)

Both extended finals are WORSE than their ep16 parents on the probe while
FIT full-CE improved (Bs 1.766->1.645, Bf 1.705->1.583): Bs-ext vs Bs16
-0.32 pts [-0.50, -0.18]; Bf-ext vs Bf16 -0.27 [-0.44, -0.11] (paired by
game, n=58,781). Your warning is realized exactly: descending FIT loss did
not mean improving generalization. Under this restart recipe (fresh Adam
5e-4 cosine-8), more training hurts; whether a warm-moments/1e-4 recipe
would differ remains unmeasured. **Bf_ep16 stands as the best candidate
(+4.15 vs A). The convergence question is closed for this recipe; the
next pre-registered contrast is per-candidate economic features from the
Bf_ep16 parent.** Report: data/experiments/16.18-whole-basket-ext1/.

## CLOSING PRECISIONS (accepted verbatim)

1. The extension shows THIS restart recipe and budget worsened
   generalization — it does not demonstrate global convergence. Bf_ep16 is
   the best OBSERVED candidate of the program, not a proven optimum.
2. The +4.15 is vs the experiment's own arm A, on the internal probe — not
   vs the full production reference model, nor an independent validation.
3. Economic-features contrast, binding condition: control AND candidate
   both start from the same frozen Bf_ep16 and receive IDENTICAL additional
   tuning; ONLY the economic features differ — differences then attribute
   to informing the scorer, not to another training continuation.

## ECONOMIC CONTRAST RESULT (residual reranker v1, full probe, pre-registered)

Executed same evening per protocol (docs/experiment_basket_residual_econ_v1.md,
run dir 16.18-basket-residual-econ-v1/). Bench gate honored: 14.3 ms/decision
(0.112 ms/candidate) measured before the build — that is also the real added
inference cost for 128 candidates; cache build ~1.2 h; both arms trained in
253 s.

- **PRIMARY MET: econ − control = +0.20 pts [+0.08, +0.33]** (first_action_exact,
  paired by game, n=58,781; CI excludes zero).
- Secondary met: econ − parent = +0.16 [+0.03, +0.27].
  Control − parent = −0.05 [−0.10, +0.002] — the extra trainable head alone
  adds nothing; attribution to the economic features is clean by design.
- Net accounting vs parent (econ arm): 782 corrected, 670 broken, **net +112**
  (your planning arithmetic held: +112 net ≈ +0.19 pts on this probe).
- **The gain is concentrated where the mechanism predicts**: multi-item
  exactness 0.1882 → 0.1997 (+1.15 pts; sizes 2/3/4 net +217/+41/+6) at a
  small single-item cost 0.5387 → 0.5341 (−0.46 pts; size 1 net −144).
  SAVE/no-buy: net −8. Fallback saves 0 in both arms.
- Train CE (fit): econ 1.6659 vs control 1.6859 — the features also help
  in-sample, consistent with information, not noise.
- 30 inspectable examples persisted in the report (e.g., Ivern 653g:
  parent picks 3114 (400g, keeps a slot); econ picks completing 3158
  consuming an owned component — historical truth). Rows per decision:
  rows_{control,econ}_full.jsonl.

Reading we propose: the features carry real signal and the effect direction
matches the finish-vs-add hypothesis, but a frozen backbone + scalar
correction captures little of it. The measured magnitude does not clear any
promotion-relevant bar (+5.0 hurdle untouched; test sealed; nothing
promoted). Candidate next steps, one at a time: (a) end-to-end fine-tune of
Bf_ep16 WITH economic inputs vs a twin fine-tune without (same recipe,
pre-registered, now justified by this positive diagnostic); (b) accept the
single/multi trade as a decoder/product question. Your call which to
pre-register first.

## ECON ROUND 2 — audit conceded, accounting corrected, seed replicated

**Conceded after verifying your audit against the raw rows (all exact):**

1. **Identity bug confirmed**: 124 duplicate (match_id, ts) keys in the
   probe (ten perspectives per game; 0 duplicates adding champion). The
   dict join overwrote 124 rows. Correct net vs parent: **749 corrected /
   657 broken / net +92** (consistent with +0.157 pts × 58,781), not the
   +112 previously reported. key_of now uses (match_id, ts, champion);
   canonical report regenerated (metrics unchanged; only the accounting
   moved).
2. **Costs now ship with the result** (econ − control, paired, in the
   report): SAVE/no_buy_death **−0.96 [−1.69, −0.19]** (n=1,047);
   ward-only **−1.13 [−1.64, −0.64]** (n=2,487); single −0.34
   [−0.51, −0.20]. Multi +1.10 [+0.89, +1.30]; family +0.31.
3. The 14.3 ms figure is a feature bench over 200 FIT decisions — not an
   online p95 latency measurement. Recorded as such.
4. Both softened conclusions accepted: control ≈ 0 is compatible with a
   small nonzero head effect, not proof of exactly zero; and a small gain
   under a frozen backbone does not demonstrate a large gain blocked by
   it — that is precisely the next contrast's hypothesis.

**Seed replication executed (your step 2, cache reused, ~4 min):**
seed 17 → econ − control **+0.21 [+0.09, +0.33]**; econ − parent +0.17
[+0.04, +0.28]; multi +1.04 / single −0.28 / SAVE −0.96 / ward −0.84;
net vs parent +100 (739/639). Direction and structure hold across the
initialization. Report: residual_report_full_seed17.json.

**On the twin fine-tune (your step 3) — one design question before we
pre-register.** Your technical condition is understood: unfreezing
invalidates the cached embeddings/scores/tail_lse, so the loss must be
recomputed with gradients (phase-2 microbatched full-CE) or an alternative
formulation declared. The open point is the econ-feature domain: features
for the FULL catalogue are infeasible (~0.11 ms/candidate × 7,938 × 294k
decisions), so we would reuse the parent's FIXED top-128 per decision as
the feature domain — candidates outside it receive a zero block + absent
flag — while the CE itself stays full-catalogue with gradients. States
drift during fine-tuning but the feature domain stays pinned to the
frozen parent's shortlist (declared, seed-independent, already cached).
Alternative would be recomputing shortlists periodically (features
recomputed each time — cost grows with refreshes). We favor the fixed
domain for attribution cleanliness and cost; your call before we fix
checkpoint, budget and schedule. Parent and control rows will be carried
in all results, per your condition.

## ECON FINE-TUNE LAUNCHED (design accepted verbatim, six conditions)

Pre-registration: docs/experiment_basket_econ_finetune_v1.md. All six
conditions implemented as specified: fixed feature domain D(x) = frozen
parent top-128 (cached, seed-independent, never refreshed, never an
action mask); score = current_base + member_of_D x correction with the
SAME zero-init correction in both arms (step-0 parent reproduction
verified in code: max |correction| = 0 and identical fixed-subset
full-CE); full-catalogue CE recomputed with gradients, phase-2
microbatching, out-of-domain targets retained via their updated base
score; identical decode/legality; online contract measured as parent
forward + features + tuned forward (offline estimate, not p95); the two
domain diagnostics (first/top-3 in D, mean out-of-D probability mass)
and nothing else — no refresh in this contrast.

Budget fixed before launch: both arms from Bf_ep16 full state, fresh
Adam lr=1e-4 (5e-4 already shown harmful on this parent), cosine
T_max=8, 8 epochs, seed 16, same batch order; checkpoints +4/+8; the +8
checkpoint is THE result. Parent, control and candidate in all results
with corrected/broken and the single/multi/SAVE/ward cost slices,
(match_id, ts, champion) identity. Commit bd16405; run in progress
(~2 h); result will be published either way.

## ECON FINE-TUNE RESULT (full probe, +8 checkpoint, as pre-declared)

Step-0 verification held in both arms: max |correction| = 0.0 and
fixed-subset full-CE = 1.70479 = the parent's own value. 127 min total.
Rows for independent recomputation: 16.18-basket-econ-finetune-v1/
rows_{control,econ}_ft_full.jsonl; identity key (match_id, ts, champion).

- **PRIMARY: econ − control = +0.33 [+0.18, +0.46]** (first_action_exact,
  paired by game, n=58,781) — larger than the residual contrast's +0.20.
  Net +193 (770 corrected / 577 broken).
- **econ − parent = +0.20 [+0.05, +0.33]** (net +116, 905/789).
- **control − parent = −0.13 [−0.24, −0.03]** (net −77): the SAME tuning
  without economics DEGRADES the parent — consistent with the extension's
  lesson that more training alone hurts this parent — while economic
  access flips the sign. FIT fixed-CE improved in both arms (control
  1.725→1.660, econ 1.707→1.638), so once again FIT loss alone would have
  misled; the twin control is what makes the attribution.
- Cost slices persist, same shape as the residual: multi +1.21
  [+1.01, +1.41]; single −0.21 [−0.38, −0.07]; ward-only −0.92
  [−1.47, −0.40]; SAVE −0.76 [−1.70, +0.10] (this CI now crosses zero).
- **Domain diagnostics say the fixed domain barely binds**: first option
  inside D in 100.00% of decisions (both arms), top-3 fully inside D
  99.99%, mean probability mass outside D 0.14%/0.15%. No refresh
  mechanism appears warranted at this tuning scale.
- Online contract (offline estimate over 200 single decisions, not p95):
  parent forward 12.7 ms + econ features 14.3 ms + tuned forward 13.5 ms
  ≈ 40.5 ms/decision.

Scope unchanged: internal probe, vs the experiment's own baselines, no
promotion, test sealed. Aggregates: econ 0.4060 / control 0.4027 /
parent 0.4040 exact.

## WARDFREE RESULT (full probe, ep16, as pre-registered)

Owner's product decision (recorded pre-launch): free wards never
suggested; pinkwards optional OUTSIDE the basket. Experiment: wards
(2055/772043) stripped from catalogue+targets (empty -> SAVE), phase-2
Bf recipe unchanged (16 epochs, clean init); baselines re-scored under
the SAME ward-free metric from their enriched rows; paired by game.
Catalogue 7,937 -> 5,691. 77 min. Rows: 16.18-wardfree-v1/.

- **PRIMARY (vs econ_ft_ep8, the development baseline): +0.23
  [-0.03, +0.46] — CI includes zero, NOT resolved.** Published as such.
- **Clean vocabulary comparison (vs parent, same recipe, only the
  vocabulary differs): +0.41 [+0.15, +0.64]** — ward-free training
  helps beyond post-hoc forgiveness. Vs control_ft +0.54 [+0.29, +0.78].
- **any-of-three: +1.72 [+1.53, +1.93] vs econ_ft** (0.6381 vs 0.6208,
  n=55,247 non-empty-truth) — the composition/top-3 gain is large and
  unambiguous against ALL baselines (+1.88 vs parent, +2.00 vs control).
- Slices vs econ_ft: single +0.82 [+0.54, +1.11]; **multi -1.48
  [-2.03, -1.01]** — econ_ft's multi advantage comes from its economic
  features, which the wardfree model does NOT have (plain Bf recipe);
  no_buy_death +2.87 [+0.95, +4.75] (SAVE improves with ward-only mass
  folded in); ward_only -0.32 ns.
- Absolute ward-free ladder for the new model: first 48.57% / top-3
  63.81% (denominator differences vs earlier diagnostics noted: any3
  excludes empty-truth rows).

Reading: vocabulary cleaning and economic features appear to be
COMPLEMENTARY, not competing — wardfree wins single/SAVE/top-3, econ_ft
wins multi, and their sum is the obvious candidate. Natural next step
(pre-registration needed, non-trivial cost): rebuild the econ cache on
the ward-free catalogue (~1.2 h features + frozen pass) and run the
SAME twin econ fine-tune from wardfree_ep16 — control vs econ, six
conditions unchanged. No promotion; test sealed.

## GROUPED DECODER RESULT (your assignment, executed 2026-09-22 00:40, 2 min)

Your collapse quantification verified exact against the enriched rows
first (22,531 of 55,247 main purchases with collapsing options; mean
2.32 unique non-empty mains). Implementation as specified: full
catalogue scored by frozen econ_ft_ep8 (its normal corrected scores),
baskets grouped by ward-stripped identity (5,692 groups; SAVE group =
"no main purchase", includes ward-only variants), per-group logsumexp,
legality of the MAIN basket at the original budget, three DISTINCT main
purchases. Same rows, same ward-free metric, paired by game. Report:
16.18-wardfree-v1/grouped_decoder_report_full.json; rows persisted.

- **grouped econ_ft vs trained wardfree: −0.09 [−0.32, +0.17] — a tie.**
  any-of-three: +0.01 [−0.18, +0.19] — 0.6382 vs 0.6381, identical.
- **The wardfree model's top-3 advantage was the decoder artifact you
  predicted**: grouping alone moves econ_ft's any3 by +1.74
  [+1.62, +1.86] — matching the +1.72 the retrained model showed. It
  also nudges first-action +0.14 [+0.04, +0.24].
- Slices persist as a real trade: grouped keeps the ECONOMIC multi edge
  (+1.15 [+0.65, +1.67] over wardfree) while wardfree keeps single
  (+0.43), no_buy_death (+2.29) and ward_only (+1.05).

Our reading, for your check: the cheaper solution stands. The grouped
decoder over econ_ft_ep8 equals the retrained model overall, preserves
the economic multi advantage, keeps a ward-AWARE model (which the
pinkward-probability product feature needs anyway — the group softmax
already yields P(ward variant | main purchase) for free), and cost 2
minutes. The combined econ-on-wardfree training is no longer justified
by the top-3 gap (now zero); its remaining upside is the single/SAVE
slice edge, and we would not run it without a new pre-registration that
carries wardfree control AND this grouped baseline, per your condition.
Also adopting your naming: the merged SAVE group is "no main purchase",
not "save all gold"; no-buy and ward-only stay reported separately.

## CYCLE CLOSED (2026-09-22): frozen dev baseline + your three precisions

Adopted and FROZEN as the development baseline for the ward-free task,
bound by digest in data/experiments/16.18-basket-econ-finetune-v1/
frozen_dev_baseline_v1.json (checkpoint econ_ft_ep8 sha ea32891d…,
domain parent Bf_ep16, FIT normalization stats source, grouping rule and
decoder script all named). Your three precisions recorded verbatim in
that manifest and here:

1. "No resolved difference" vs the trained wardfree (−0.09
   [−0.33, +0.16]) — NOT demonstrated equivalence; the cheaper
   alternative is chosen on that basis, nothing stronger claimed.
2. The decoder reproduces the AGGREGATE top-3 gain, not same-case
   coverage: 1,621 covered / 1,615 lost vs wardfree, net +6 (verified
   from the rows before recording). The practical conclusion stands:
   retraining is not needed to reach that coverage level.
3. Pinkward probability = ward-variant mass WITHIN the chosen group over
   that group's mass (conditional), not the between-group softmax; it
   still needs its own calibration validation, plus a joint
   legality/affordability check of main+ward. Not shipped until then.

Costs stay visible in the manifest (multi won, single and no-buy lost vs
wardfree; some multi lost vs ungrouped). wardfree_ep16 kept as the
comparison alternative; the combined training stays an open hypothesis
without its top-3 justification. Your proposed next step is already
running, pre-registered (docs/experiment_val_window_check_v1.md): the
frozen baseline evaluated ONCE on the frozen VAL split (1,601 games /
191,916 decisions, later window), no comparisons or selection on val,
test sealed. Result will be appended as measured.

## VAL WINDOW RESULT (frozen baseline, one read, as pre-registered)

1,601 games / 191,916 decisions, later temporal window, 104 min, all
computed fresh (parent -> D(x) -> engine features -> grouped decode).
Ward-free metric, per-game bootstrap CIs on val itself:

- first_action_exact **0.4827 [0.4805, 0.4853]** (probe was 0.4849)
- any_of_three **0.6338 [0.6316, 0.6362]** (probe 0.6382; denominator
  180,526 non-empty-truth decisions)
- Slices: single 0.5526; multi 0.2230; no-buy 0.7113; ward-only 0.8202.

Reading: the frozen configuration's behavior is broadly stable on the
independent window — the probe numbers were not an artifact of the 500
probe games. Small consistent drop (~0.2-0.4 pts) as expected for a
later window. No comparisons or selection were made on val; the trained
wardfree alternative was deliberately NOT evaluated. Rows persisted in
16.18-val-window-check-v1/. The overnight capacity contrast (d64 vs
d128, docs/experiment_capacity_d128_v1.md, queued behind this check)
starts automatically now; its result will be published tomorrow
whichever way it lands.

## CAPACITY CONTRAST RESULT (d64 vs d128, published negative, 2026-09-22 morning)

Owner-authorized overnight run (docs/experiment_capacity_d128_v1.md).
Gates passed (pilot projected 4.05 h < 7 h; ran 3.14 h; peak GPU 1.5 GB,
no spill). Clean inits, shared batch order via independent generator,
grouped decoder, ward-free metric, identity (match, ts, champion) with
uniqueness asserts. Params: A 583,388 / B 1,091,548.

- **PRIMARY: B − A = −0.66 [−0.97, −0.41]** (first_action_exact
  ward-free, n=58,781) — statistical DEGRADATION, nowhere near the +0.5
  practical threshold. any-of-three −0.58 [−0.80, −0.35] (n=55,247
  non-empty-truth); single −0.87; multi/family/no-buy/ward-only CIs span
  zero. Net −390 (2,982 corrected / 3,372 broken).
- **The FIT-loss trap, third appearance this week**: B's train loss is
  far lower (1.519 vs 1.742) while it generalizes worse. Capacity fits
  the fit set; it does not transfer at 2,000 games of data.
- Reference (not a victory condition): fresh A vs the frozen econ
  baseline −0.23 [−0.49, +0.03] — consistent with the economic features
  being worth ~+0.2 over a plain Bf-recipe model; B vs baseline −0.89.
- Decode latency ~1.15-1.18 ms/decision both arms; illegal walked
  candidates comparable (559k/576k over the whole probe decode).

Conclusion, scoped: at this data scale and recipe, widening the state
encoder alone HURTS. d64 stands; the frozen dev baseline is untouched
and remains best. If capacity is revisited, it should come after more
data (the gold-capture farm) rather than before, and as its own
pre-registration.

## CLOSING ROUND (2026-09-22 morning): precisions accepted, budget re-audit verified

**Your three precisions, recorded as the official wording:**

1. d128 loses UNDER THIS RECIPE AND THIS DATA; the FIT/probe divergence
   is compatible with overfitting and does not show architecture gains
   are exhausted. The capacity extension is closed without further GPU
   spend — nothing stronger claimed.
2. Fresh-A vs the econ baseline is NOT a third isolated estimate of the
   feature value (training also differs; the CI includes zero). The twin
   contrasts remain the proper evidence; the "third angle" phrasing is
   withdrawn.
3. The temporal check is useful but VAL v3 was already exposed in
   Phase D and POV: it is another temporal population, not a fully
   blind final confirmation.

**Ward-free budget re-audit (your numbers, verified against our rows —
hit totals match exactly, 28,501 = 25,786 feasible + 2,715 empty):**
of 55,247 main purchases, 13,209 (23.91%) remain budget-blocked at
gold_est and 459 (0.83%) structurally blocked; on the 41,579 feasible
main purchases the frozen grouped baseline hits 62.02%. Agreed reading:
that is a feasibility-conditioned rate, not a forecast of what fixing
gold yields (a different wallet can also unlock wrong alternatives) —
but the remaining block is the concrete reason to prioritize the wallet
front. The whole-basket 25.5% figure is retired for the current task;
23.91% is the number.

**Next contrast adopted as specified**: same frozen model, same
decisions, estimated vs measured wallet, features and legality updated
coherently, counting BOTH recovered purchases and previously-correct
losses. Data dependency, stated plainly: measured wallets exist today
only for the 40 macro_coach games (zero overlap with this probe); the
contrast needs gold-only TRUTH-SEEKER captures of the probe games —
tier 1 of the owner's capture plan (208 probe games, ~1 night) is
exactly this experiment's data. The harness can be built now; the go/
no-go remains blocked on the League client for the replay-download
test.
