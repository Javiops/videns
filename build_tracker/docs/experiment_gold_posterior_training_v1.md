# Pre-registration: training under an uncertain wallet — frozen 2026-09-17

Written and committed BEFORE the first training step. Nothing below is edited
after a result is visible; corrections go in a dated section at the end.

Owner authorization: 2026-09-17 — "what I would like you to do is train this
model with the uncertainty that the gold is in an estimated range to see how it
will mimic once it goes to a live game", then "i want to retrain the model then
run my metrics", then "its already morning i will have running for as long as it
takes".

## The recorded question

Does training the shop model on a wallet drawn from the measured error
distribution — instead of on a single wrong number — produce a candidate that
beats the point-estimate candidate on its own validation window, and does it
behave better at the exact-wallet end where live serving actually sits?

This run produces a **candidate**. It promotes nothing.

## Why

Offline we never know the player's gold. The estimator's error was measured
against exact frame `currentGold` on 653,030 unselected TRAIN windows
(`data/reports/gold_error_shape_v1/`) and fitted as

    c_q(dt) = a_q·dt + k_q·√dt

— drift plus diffusion, per quantile, worst absolute fit error 1.8–6.4 gold
across quantiles spanning −153 to +268 g
(`data/reports/gold_posterior_v1/posterior.json`). The wallet is a *range* and
we now know its shape.

The trainer did not use that. It used `gold_est` as a point, masked everything
else at a flat `gold + GOLD_DRIFT` (500 g) — a slack at which **97.4%** of states
are over-credited by a median 431 g — and then force-unmasked the label
(`train_prefix.py:765`), so the model was taught to rank recorded items that its
own budget said were unaffordable. Decode then masks at `budget_slack=0`, where
26.4% of recorded baskets are unreachable. Phase 1 measured what that costs as a
decode setting: `const:500` scores **−0.1424** exact against `const:0`.

Live is the opposite regime: `app/live.py` supplies the real wallet
(`live-exact-v2`, `gold_exact`). A model trained on one wrong number and served
an exact one is out of distribution in a direction nobody has measured.

## What changes (env-gated; off by default)

`PREFIX_GOLD_POSTERIOR=<posterior.json>`, `PREFIX_GOLD_MASK_Q=0.65`:

1. **The wallet is drawn, not assumed.** Per row, `gold = gold_est + c_u(dt)`
   with `u` a seeded per-row uniform draw (hash of match/participant/ts, so the
   cached tensors are reproducible) and `dt = ts mod 60000`. The budget feature,
   the affordability block and the legality mask are all built from that same
   draw, so they cannot disagree with each other.
2. **The mask is the standing rule, not the draw.** `allowed = gold_est +
   c_0.65(dt)` — the posterior's quantile above the causal estimate, never above
   the sampled wallet, so a lucky draw cannot widen its own candidate set.
   q = 0.65 comes from phase 2, whose winner `rate:2` maps to q 0.60–0.70.
3. **One appended feature:** the posterior's width at this row's staleness.
   `QUERY_DIM` 26 → 27 when the flag is on, 26 when it is off. Live sets width 0,
   and because a visit one second after a frame already has a width of ~7 g,
   **the exact-wallet end live serves is inside the training range rather than
   outside it.** That is the transfer argument and the reason the feature exists.

   *Pre-run correction, 2026-09-17, before any training step:* a second feature,
   a `gold_exact` flag, was in the first draft of this document and was removed
   on inspection. Only live sets that flag, so it would have been 0 on every
   training row and serving would have presented a combination the model had
   never seen — defeating the purpose it was added for. The width carries the
   same information continuously. Recorded rather than silently edited.
4. Cache key includes the posterior's content digest; provenance records the
   source, its sha256, the mask quantile and `labels_used: false`.

**No label enters any of it.** Drawing a wallet that falls below what the player
bought is intended — that is the uncertainty we actually have. Truncating the
draw at the recorded basket cost would be `gold_left + net_spend` in new
clothing (`docs/history.md` row 1) and would teach the model that the budget
constraint goes soft exactly where the answer is expensive, which it cannot know
at inference.

## Mechanics

- Generation `16.18-reconstructible-development-v3`, unchanged. No re-export,
  no re-reconstruction, no new data.
- Stages: trunk → graft, same geometry and hyperparameters as the Phase D
  candidate except `QUERY_DIM` and the posterior flags, so the wallet treatment
  is the only deliberate difference.
- Baseline for comparison: the existing Phase D candidate `ed1fda4f…`, scored on
  the same rows. Both are evaluated by `eval_policy.score_visit` verbatim.
- Suite must be 51/51 before training starts; it is a stage of the chain.

## Pre-registered predictions

1. **TRAIN-side, decode at `const:0`:** the posterior candidate beats the point
   candidate on `first_action_exact`. Expected +0.01 to +0.05 — smaller than the
   decode-only sweep's +0.0297, because part of that gain is already available
   without retraining.
2. **The posterior candidate is flatter across decode slack.** Its best-to-worst
   spread over `const:0 … const:500` is narrower than the point candidate's
   0.4353→0.2929 range, because it trained across wallet error instead of at one
   point. If it is *not* flatter, the uncertainty training did not take.
3. **Its optimum decode quantile sits closer to q=0.50** than the point
   candidate's ~0.65: a model that already expects wallet error needs less slack
   handed to it at decode.
4. Guardrails (`no_buy_death_recall`, `ward_only_recall`) do not fall more than
   0.02 against the point candidate.

## Audit triggers — investigate, do not celebrate

- Any TRAIN gain above **+0.10** exact: audit for a leak before reporting it.
- The posterior candidate winning at `const:500` as its best decode: the mask
  alignment did not take effect.
- `gold_posterior` absent or `labels_used` not false in the artifact provenance:
  stop, the run is unattributable.
- Validation exact above **0.45**: audit rather than celebrate — Phase D's
  comparable figure is 0.3811 and a jump that size is more likely a defect.

## Bounds, stated in advance

- TRAIN numbers are memorisation-favoured and settle only the *ordering*. The
  claim needs the paired validation comparison, which spends a ledger line on
  this generation's window.
- **The live-behaviour claim is not measured by this run.** Setting width 0 at
  serving makes the model in-distribution at the exact-wallet end; whether that
  helps a real player is unmeasurable until shop telemetry exists, which is
  still blocked on the personal Riot API key. Direction argued, magnitude
  unmeasured — do not upgrade that to a result.
- The `dt` proxy is `ts mod 60000`: 99.55% of visits within 1 s of the true
  anchor gap, the remaining 0.45% off by about one 60 s cadence.

## What this document does NOT authorise

- Promotion. A candidate needs a matching frozen-validation displayed-policy
  report with zero unaffordable baskets, and any decode slack > 0 fails that gate
  as written.
- Any test-split read.
- Changing `GOLD_DRIFT`, `eval_policy.py`, or the served configuration.
- More than one paired validation comparison on this generation's window.
