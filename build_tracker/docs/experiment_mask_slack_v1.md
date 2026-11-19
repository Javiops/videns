# Pre-registration: mask-slack selection (phases 1 and 2) — frozen 2026-09-16

Written and committed BEFORE the first arm ran. Nothing below was edited after
a result was visible; corrections, if any, go in a dated section at the end.

Owner authorization: 2026-09-16, "go, write the protocol and set up the sweep,
only proceed with phase 1 and 2, this is the task for tonight".

## The recorded question

With the model held fixed, which decode-time budget `gold_est + delta`
maximises exactly-correct displayed choices on TRAIN, and what does each delta
cost in displayed baskets the player could not actually afford?

This is a **selection** run. It picks an arm. It does not measure how much that
arm is worth — that requires the paired validation comparison in phase 3, which
this document does **not** authorise.

## Why this experiment exists

`data/reports/gold_error_shape_v1/` measured the gold estimator against exact
frame `currentGold` on 653,030 TRAIN windows:

- signed error `gold_est − currentGold`, no-purchase stratum: median **−75 g**,
  IQR **212 g**, p10..p90 **−317..+109**, over-credit rate **29.5%**;
- the bias is a stable rate, **≈ −0.5 g/s**, at anchor ages of 60, 120 and 180 s;
- the spread scales with staleness too — IQR 192 → 337 → 450 over those same
  anchor ages — and with game phase (IQR 182 → 218 → 247 by minute bucket);
- kill/bounty events do not explain the spread (quiet windows IQR 203 on 83% of
  the mass), which is why the Phase A income variants could not help;
- **26.41%** of recorded purchases (77,073) have a basket unaffordable at
  `gold_est`, median shortfall **83 g**.

Two consequences put the pipeline on both sides of one line:

- `scripts/train_prefix.py` masks legality at `gold + GOLD_DRIFT` (500). At that
  slack **97.4%** of states are over-credited, median excess 431 g.
- `scripts/eval_policy.py:320` decodes at `budget_slack=0`, where 26.4% of
  recorded baskets are unreachable.

Neither number was chosen from the error distribution. The model is therefore
trained against a candidate set far wider than the one it is scored on, and the
sweep asks which decode point is actually best for the model as it exists.

## The arms (frozen; no others in this pass)

Twelve arms, all scoring the **same** visits in the same order so every
comparison is paired.

| family | arms | delta |
|---|---|---|
| phase 1, constant | `const:0` `const:50` `const:75` `const:100` `const:150` `const:200` `const:300` `const:500` | flat, in gold |
| phase 2, rate | `rate:1` `rate:2` `rate:4` `rate:6` | `k · dt_seconds` |

`const:0` is the reference arm and is mandatory. `const:500` is included
because it is what the trainer already uses. The rate arms produce mean deltas
of roughly 28 / 56 / 112 / 168 g, which brackets the constants and lets the two
families be compared at matched mean delta.

**`dt` is approximated as `ts mod 60000`.** The export carries no
`gold_gap.frame_ts`, so the true anchor gap is not available on the rows.
Measured against the real gap on 60 TRAIN games / 7,350 shop visits: **99.55%**
of visits land within 1 s of the truth; the remaining 0.45% are off by about a
full 60 s cadence. Stated so the approximation is never mistaken for the real
thing; an exact `dt` requires adding `frame_ts` to the export, which is a
rebuild and is not in scope tonight.

## Mechanics

`scripts/mask_slack_sweep_v1.py`, TRAIN split only, run through the bound 16.18
economics runtime via `footwear_runtime.activate_for_manifest`.

- Model: the Phase D candidate `candidate.pt` in
  `data/experiments/16.18-reconstructible-development-v3`, unchanged. **No
  retraining in this experiment.**
- Metrics: `eval_policy.score_visit` is imported and used **verbatim**, so every
  metric is computed by the same code that produced the Phase D report. The only
  difference between arms is the budget the displayed policy may spend.
- `eval_policy.py` is **not modified** — it is digest-pinned in
  `scripts/run_development_1618_v3.py` and editing it would invalidate frozen
  plans. `score_visit` hardcodes `budget_slack=0`, so the slack is injected by a
  documented proxy around the predictor (`SlackProxy`).
- The price tag is measured against the **honest** wallet: over-budget displayed
  baskets are judged against `gold_est`, never against `gold_est + delta`, so an
  arm cannot define its own cost away.
- Intervals: game-level bootstrap, and every arm reported both absolutely and as
  `paired_vs_const0` on identical games and visits.
- Sample: 500 TRAIN games (≈ 59,500 decisions), seed 16, bootstrap 400.

## Pre-registered predictions

Written before the first arm ran.

1. **The optimum is interior**, with the best arm at a mean delta between 50 and
   200 g. A curve that is still rising at `const:500` is a red flag, not a win.
2. **The best arm beats `const:0` on `first_action_exact` by +0.02 to +0.09.**
   The arithmetic behind that range: at delta = 100, 56.4% of the 26.4% blocked
   basket mass becomes affordable (14.9 points of all purchases); if those
   convert at the oracle ceiling's masked-row rate of 0.6079, gross is ≈ +9.0
   points, against an unmeasured displacement loss on the feasible 73.6%.
3. **At least one `rate:` arm beats the best `const:` arm at equal or lower mean
   delta.** The error's bias and spread both scale with `dt`, so a flat constant
   is the wrong functional form.
4. The unaffordable-displayed count rises monotonically with mean delta, and
   `const:500` is the worst on that axis.

## Audit triggers — investigate, do not celebrate

- Any arm gaining **more than +0.10** exact over `const:0`: audit for a leak or
  a scoring defect before reporting the number anywhere.
- `const:500` not clearly worst on the price tag: the slack is not doing what
  this document assumes.
- `no_buy_death_recall` or `ward_only_recall` rising with delta: a budget slack
  has no mechanism to improve save or ward behaviour, so a rise means the metric
  is coupled to something unintended.
- Prediction 1 failing (monotone to 500): stop and re-derive before proposing a
  delta.

## Bounds, stated in advance

- **TRAIN rates are memorisation-favoured.** The level will not transfer. Only
  the *ordering* of arms is being selected here. No number from this run may be
  compared with any validation figure, including Phase D's 0.3811.
- **This cannot promote anything.** Any delta > 0 displays baskets that are
  unaffordable at `gold_est`, which fails the promotion gate's "zero unaffordable
  displayed baskets" as that gate is currently written. Restating the gate is an
  owner decision and is not part of this experiment.
- A delta buys exact-match honestly — the recorded basket was by construction
  affordable at the player's real wallet, and a wrong expensive item still scores
  as a miss — but it is optimising the symptom, not the estimator.

## What this document does NOT authorise

- Any validation or test read. No ledger comparison is spent by this run.
- Any retraining, and any change to `GOLD_DRIFT`, `eval_policy.py` or the served
  configuration.
- Promotion of anything, on any result.
- Deriving a wallet from the label. The oracle-budget ceiling remains a
  diagnostic; raising `gold_est` toward a recorded purchase is the banned
  `gold_left + net_spend` pattern (`docs/history.md`).

---

## Addendum, 2026-09-17 morning — phase 2b (quantile arms) and the miss profile

Written after phases 1 and 2 reported and before 2b ran. Nothing above was
edited; the original pre-registration stands as written and its predictions
were met (see `data/reports/mask_slack_sweep_v1/mask_slack_sweep.json`).

Owner authorization: 2026-09-17, "its already morning i will have running for as
long as it takes", plus the request for a miss breakdown by gold distance and
item class.

**Why the arm family changes.** The `rate:k` arms assume the needed correction
is linear in staleness. It is not. Fitting the measured correction
`c = currentGold − gold_est` against anchor age gives

    c_q(dt) = b·dt + k_q·√dt,   b = 0.5153 g/s

— bias linear in dt, spread as √dt — with worst per-quantile fit error **2–5%**
against **26–30%** for a straight line through the origin
(`data/reports/gold_posterior_v1/posterior.json`, `scripts/gold_posterior_fit_v1.py`).
Phase 2's winner `rate:2` turns out to approximate quantile 0.60–0.70, drifting
with dt. Phase 2b replaces the hand-picked slopes with actual quantiles of that
posterior, which differ most at short dt — where most real visits sit (median
≈ 29 s) and where no direct measurement exists.

**Censored coverage check (recorded before the sweep).** The 77,073 budget-short
purchases each say `true wallet ≥ basket cost`, at a real visit timestamp. The
fit does not use them — they are label-selected — so they serve as a one-sided
validation. Result: covered share sits below its quantile at every level and
both reference staleness values (q=0.90 → 0.77 at dt=29 s, 0.88 at dt=60 s),
which is the direction a selected sample must show if the tail is not too wide.
A covered share ABOVE q would have falsified the fit; none did.

**Phase 2b arms:** `const:0` (reference), `quant:0.50` … `quant:0.90`, plus
`rate:2` carried forward so phase 2 and 2b sit on one paired axis. Same 500
TRAIN games, seed 16, bootstrap 400, same candidate, no retraining.

**Pre-registered for 2b:** the best `quant:` arm beats `rate:2` on
`first_action_exact`, and the optimum sits at q between 0.60 and 0.75. If the
optimum lands at q ≥ 0.90 — i.e. near coverage — that contradicts phase 1, where
the coverage-shaped `const:300` lost 5.9 points, and is an audit trigger.

**Miss profile (`scripts/miss_profile_v1.py`, `data/reports/miss_profile_v1/`).**
A diagnostic, not an arm and not a selection criterion: for every missed
purchase it records the gold distance between the displayed basket and the
recorded one (signed and absolute, priced by `purchase_sequence` from the row
state) and the item class of both leads — pink ward / basic / component /
completed / boots / support item, taken from `DataDragon.classify`. Reported at
`const:0` and `quant:0.65`. It changes no decision by itself; it says which
misses are 75 g apart and which are 3,000 g apart.

**Everything the original document forbids still applies**, unchanged: no
validation or test read, no promotion, no retraining under this protocol, no
label in any feature.
