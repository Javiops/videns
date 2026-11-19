# Defensive boots: outcome-blind support audit before value estimation

Stage 0 is counts only. It uses the exact clean 6,000-game W1 v3 source set,
whose fit/calibration/development roles stay fixed. It reads no new source ID,
own confirmation game or global sealed test. Archive envelopes contain full
match DTOs, so deserialization necessarily includes outcome-bearing fields;
the extractor never accesses, copies, joins, prints or analyzes match `win`,
winner, final performance or cache `y`. This is outcome-blind analysis, not a
claim that compressed source bytes contain no outcomes.

## Fixed action question and selection

The proposed comparison is first defensive boot choice: Plated Steelcaps
(3047) versus Mercury's Treads (3111). Keep **all** first tier-two boot
categories in the counts, including choices outside this pair and ambiguous
transactions. Do not select an action family or subgroup because its outcome
looks favourable. The full pinned static objects for this pair were verified
equal across 16.17.1/16.18.1/16.19.1 during read-only feasibility: total prices
1200 and 1250, with ordinary Boots (1001, total 300) as a common component.
This does not establish equivalent affordability, patch-invariant champion
balance, or an unconfounded comparison.

From immutable roster champion IDs and patch-pinned Data Dragon tags, identify
players tagged Fighter or Tank. Select the player with the smallest
SHA256(match_id + ':boots-focal:' + participant_id). This fixes at most one
focal player per game independently of choices and outcomes. Tags describe
champion classes, not actual role, damage mix or lane assignment. Do not use
postmatch-inferred lane/role or final build to select the player.

Group that player's ITEM_PURCHASED / ITEM_SOLD / ITEM_UNDO events into visits
separated by more than 10 seconds of inactivity. Locate the visit containing
the first explicit tier-two boot purchase. The decision cutoff is the
**first event of that visit**, not the final boot-completion timestamp.
All target-visit component purchases and spends are excluded from covariates,
inventory and wallet reconstruction. Otherwise the chosen recipe leaks into
the proposed pre-decision state. Persist both cutoff and boot event time.

Tier-two categories are purchasable Summoner's Rift direct descendants of
ordinary Boots in the frozen static catalogue. Explicit later boot upgrades
before any tier-two event, multiple tier-two purchases in a visit, any undo
in that visit, and free role-quest ambiguity are counted separately; the
extractor does not silently relabel them as a clean choice. It does not infer
magical-footwear grant timing or implement automatic quest upgrades.

The target-resolution audit also examines all focal item events from visit
start through ten seconds after its last shop event, including destruction
events that did not create the shop grouping. Quest-item destruction
(1201/1202), a later boot upgrade, or tier-two boot destruction during this
interval excludes the visit from reliable support and is counted separately.
These post-cutoff audit fields never become pre-decision covariates.

## Pre-decision covariates and budget evidence

Take the latest timeline participant frame strictly before the visit cutoff.
It must match an actual row of the clean W1 cache; an earlier-than-cache,
missing or otherwise unqualified state is explicitly unmatched, not repaired
by interpolation. Frames within one second of the cutoff are marked as a
timestamp-boundary ambiguity and excluded from reliable support counts,
because existing project evidence documents near-boundary frame/event lag.
This margin is a conservative diagnostic rule, not proof of perfect timing.

Export only an allowlist: own champion, side, ally/enemy champion IDs; existing
cache feature vector at that preceding frame; focal frame gold/XP/level;
basic/magical footwear evidence from earlier events; magical-footwear rune
selection; pre-visit quest/undo ambiguities; frame age. No feature may depend
on purchases in the target visit. Post-visit transaction audit fields live
under a separate label-audit object and are never training inputs.

The primary conservative wallet lower bound is:

```text
prior currentGold
- sum(full static total price of every focal purchase
      after that frame and strictly before the visit cutoff).
```

Do not add positive income, refunds or future frame values. Do not recover
arrival money from the chosen purchase's price. This is a lower bound **under
the assumption that every spend is visible and the preceding wallet is truly
pre-visit**. Unknown item costs, nonfinite wallet measurements, an undo in the
interval, or timing ambiguity make confidence unknown. A negative conservative
bound can be floored at zero. Full-price dual affordability requires a trusted
bound at least 1250. Report it as monetary support, never full slot/recipe
legality. Slot legality remains unknown.

Minimal explicit inventory tracking can establish that ordinary Boots were
already held before the visit. Any unresolved undo, boot transformation or
magical-footwear selection invalidates that proof. A separate sensitivity
count may subtract the proven common ordinary-Boots value 300 from both full
prices; it is labelled separately and never substituted for the primary full-
price support count. No arbitrary-instant wallet is called exact.

Count mixed baskets: purchases outside the final boot and its static recursive
recipe, sales, or other main purchases in the same visit are potential
co-treatments. An association involving that visit cannot automatically be
attributed to its boot alone. Keep these counts visible rather than inventing
a clean single-action intervention.

## Stage-0 artifacts, gates and bounds

Inputs: original clean cache, its source manifest, and a versioned static
export `winprob-boots-static-v1` with `.sha256`. Verify cache metadata and
compressed source hashes; bind the original admission manifest/sidecar;
revalidate current global/external/prior reservations and each source role.
Read only paths already bound in that clean source manifest. Reject quarantine
ancestors. Bind and recheck static source files, extractor and protocol digests.
Log each raw access before opening it. Nothing imports sibling project code.

Outputs: a JSONL record per source game including explicit exclusion status;
outcome-free choice rows; source/static/code digests; an access ledger; and
counts by role, patch, boot category, ambiguity reason, mixed basket, and
conservative dual-affordability status. Report support per own champion for
the two actions, without collapsing rare cells after seeing outcomes.

Counts-only readiness requires at least 100 records per defensive action in
fit, 30 per action in calibration, and 50 per action in development **within
the predeclared reliable, full-price-dual-affordable, unmixed population**.
These are plumbing/support minima, not a claim of adequate power for one-
percentage-point effects. Report the stricter common-champion counts too.
Failing support is a useful result; do not force a positive by switching
champions, endpoints or items after outcome inspection. A count-blind expansion
to other already eligible training/development IDs needs a separate manifest
and must preserve all reservations. It is not part of this 6,000-game run.

One CPU thread, BelowNormal priority, no GPU, at most the bound source set,
deadline/memory checks, durable launcher after root review. No inline raw-data
run is authorized by this script's synthetic tests.

## Later outcome study, only after counts are locked

The primary target would be terminal team outcome. The first result is an
observed-context standardized **association**, not the effect of switching
boots. Conditioning on having chosen defensive boots, unobserved player
skill/intent, prior component commitment, timing and co-purchases all limit
causal identification. A cohort selected by the realized action family is
not a general first-boot recommendation population.

Fit separate historical action propensity and outcome regressions using
regularized tabular models and game-level cross-fitting; M2 is not a known
logging propensity. Keep the action pair and model recipes fixed. Show raw
win rates, initial WP distribution and balance, outcome regression, IPW and
directly paired doubly robust standardized contrasts with match intervals.
Declare support/weight/ESS diagnostics before accepting a contrast. Unclipped
and clipped estimates must be distinguished; clipping is not a free fix.

Before those regressions, export audited **pre-visit component and finished-
item inventory**, including Cloth Armor (1029) versus Null-Magic Mantle (1033).
Earlier recipe commitment predicts the chosen boot and the surrounding game
context; gold and basic-boot evidence alone do not adjust for it. Stage 0's
minimal inventory evidence is not a complete inventory reconstruction and
must not be passed off as sufficient confounder adjustment. Components bought
inside the target visit remain excluded from pre-decision covariates.

A selection reversal is evidence that the raw ranking depends on adjustment,
not proof that the adjusted ranking is causal. Report sensitivity to nuisance
model specification, patch and assumed hidden-confounding bias. A simple
additive bias/tipping analysis must state that its bias bound is an assumption;
a propensity-odds sensitivity bound must use a reviewed estimator, not an
invented causal interval. A six-minute WP change is secondary only, with a
judge that did not fit the scored game's outcome and known terminal outcomes
handled as absorbing values. Do not start an expensive Q trunk.

At this sample size terminal contrasts may remain several percentage points
wide even when counts look substantial. No effect, success, reversal, power or
causal identification has been measured at the protocol-writing stage.
