# Win probability added — dossier for the external reviewer (proposal; nothing run)

2026-09-29, evening. Source under review: xPetu, *Win probability estimation for
strategic decision-making in esports*, MSc thesis (Mathematics and Operations
Research), 26 August 2024, 38 pages, CC BY-NC-SA 4.0. Local copy
`Downloads/Thesis_xPetu.pdf`; text extracted with `pdftotext -layout` and read in
full. **Nothing has been trained, scored or changed.** The only new numbers in
this document are inventory counts measured tonight on `tracker.db` and the raw
archive, each marked *measured* with the source. Everything else is either quoted
from the thesis (*thesis*), taken from the project's own records (*recorded*), or
my projection (*projection*). Test sealed, chain freeze on `app/*.py` respected:
this work touches nothing in build_tracker.

---

## 0. Headline

1. **The thesis is a recipe, not a result.** Two statistics per action in a
   context: the mean *initial* win probability (when players choose it) and the
   mean *win probability added* (what happened after). Its Shen numbers on patch
   14.15 transfer nothing. Its estimator carries **no uncertainty at all**, and
   at our scale its bucketed means are underpowered (§2.5). What transfers is the
   framing: the **value axis build_tracker explicitly lacks** — the README states
   "behavioural imitation, not counterfactual optimisation… no validated
   causal-outcome claim". No file in the three projects mentions win probability
   (grep, 0 hits, *measured*).
2. **It is cheap for us, for three reasons we did not have to build.**
   (a) `data/raw_match_v5/` already holds **43,874 archived timelines** with
   per-minute frames — position, xp, gold, damage, champion stats for all ten —
   a strict superset of the thesis's feature table (§3, *measured*). No new
   collection. (b) Win-probability features are patch-stable, so the model can
   pool **39,968 games across 16.17/16.18/16.19**, which the item model cannot.
   (c) **M2 is a behaviour policy** P(a | x). The thesis has only the outcome
   half of a doubly-robust estimator; we would have both halves.
3. **Proposed: five bounded experiments plus one macro_coach diagnostic** (§4),
   in a sibling project (`videns.gg/winprob`) that reads the archive read-only
   and exports versioned artifacts with digests — the buys_export_v1 pattern.
   Order W1 → W2 → W3 → W4 (→ W5 only on a resolved positive); M1 in parallel.
4. **Honest ceiling**: at challenger level item choices are compressed. Expect
   most per-cell effects under one point with intervals wider than that. The
   deliverable is a value axis with honest uncertainty and a policy-evaluation
   metric, not a table of best items.
5. **One correction to what I told the owner earlier tonight** (§2.2): the
   thesis's statistic does *not* simply collapse to "win rate minus expected win
   rate"; its own tables show the two differ, and the thesis does not document
   why. Withdrawn and replaced by the precise objection.

Questions for you are in §9.

---

## 1. What the thesis does (faithful summary, its numbers)

### 1.1 Formalism (Chapter 4)

- Zero-sum, no draws. `w(x) ≈ Pr(Y = 1 | x)` from team T's view; mirrored
  state gives `1 − w(x)`. Calibration is first-class: ECE with M bins,
  reliability diagrams.
- Decision problem: given state x and finite action set A = {a₁…a_K}, pick
  `argmax_a Pr(Y = 1 | x, a)`. Continuous actions (movement, aim) excluded by
  construction. Analysis is *retrospective/premeditated*, explicitly not
  in-game automation.
- Data set of actions `D = {(a, x, z)}`: x = state where the action took
  effect, z = last state affected by it (item sold / destroyed / upgraded);
  default z = terminal state.
- Two per-datum quantities: `W₀(d) = w(x)` (initial WP) and
  `ΔW(d) = w(z) − w(x)` (WP added).
- Two statistics per action a and state set S ⊆ X:
  `W̄₀_S(a)` = mean W₀ over data where the action is a and x ∈ S (measures
  selection bias: > 0.5 → chosen when ahead);
  `ΔW̄_S(a)` = mean ΔW over the same set (measures value).
- Justification: LLN plus S → {x} gives
  `ΔW̄ → Pr(Y=1 | x, a) − Pr(Y=1 | x)`; rule: `argmax_a ΔW̄_S(a)` for S
  similar to x. The thesis names the choice of S ("specificity vs sample size")
  as its open problem.
- Four challenges it lists: state representation, complex/delayed actions
  (Mejai's), variance/external factors (one player among ten), **action
  selection bias** (actions chosen when ahead or in desperation; community
  meta over-use).

### 1.2 The win-probability model (§5.1) — *thesis*

| item | value |
|---|---|
| data | 400,000 ranked matches, patch 14.15; 350k train / 50k test |
| states | Match-V5: one frame per minute + every event; **one state sampled uniformly per 5-min interval per match** for training (decorrelation) |
| features (Table 5) | per player: position (x, y), level, cumulative gold, damage dealt, damage taken, kills, assists, deaths; per team: voidgrubs, dragons, barons, turrets, inhibitors; time; average rank (only pre-game feature). **Items deliberately excluded** ("60 categorical variables… imprudent") |
| model | DNN "following Kim et al. 2020"; architecture not described |
| test set | 50,000 matches / 5,653,687 states |
| accuracy | 75.9 % aggregate; by 4-min bin 55.8 % at [0,4] rising to 84.8 % at [24,28], declining in the last three bins |
| ECE (20 bins) | 0.90 % aggregate; by 10-min window 0.67 / 1.19 / 1.59 / 1.49 % |
| benchmark (Table 6) | Silva 2018 RNN 63.9–83.5 % (7,621 pro games); Kim 2020 baseline DNN 73.0 % / ECE 4.47 %, calibrated 73.8 % / 0.57 % (83,875 games) |

### 1.3 The case study (§5.2) — *thesis*

Shen's **first legendary item**, four most popular candidates, 8,206
purchases. Quantities in percent; n = purchases.

| item | ΔW̄ | W̄₀ | win rate | n |
|---|---|---|---|---|
| Titanic Hydra | +1.46 | 51.41 | 51.94 | 2,189 |
| Heartsteel | +0.66 | 53.66 | 54.15 | 1,411 |
| Sunfire Aegis | −0.51 | 50.53 | 49.10 | 3,004 |
| Hollow Radiance | −0.84 | 52.76 | 51.25 | 1,602 |

Split by lane opponent's damage type: physical n = 5,994 (Titanic +1.94,
Heartsteel +0.23, Sunfire −0.68, Hollow −0.71); magic n = 2,212 (Heartsteel
+2.24 on n = 304, Sunfire +1.75 on n = 213, Titanic −0.21 on n = 493, Hollow
−0.88 on n = 1,202). The thesis's reading: win rate is biased (Heartsteel is
bought when ahead, W̄₀ 53.66); Titanic is the value pick vs physical; Hollow
Radiance, the popular pick vs magic, "is also the worst alternative"; the magic
tables are "tentative".

### 1.4 What in the thesis is worth keeping verbatim

- Calibration reported by time window, not only aggregate.
- Uniform sampling of states per match to break within-match correlation.
- The pair (W̄₀, ΔW̄) shown together: selection bias next to value.
- Action space ≠ state space: the WP model used to *judge* an action must not
  have that action in its inputs, or the effect is "priced in" at x.

---

## 2. Critique — what not to copy

### 2.1 No uncertainty (the central weakness)

The thesis reports no standard error, interval or test for any ΔW̄. My
estimate (*projection*): a per-purchase ΔW lies in [−1, 1]; if its window
resolves near the terminal state its SD is ≈ 0.5, if windows end mid-game it is
lower, say 0.3. Then:

| cell | n | SE range (pp) | reported ΔW̄ | z range |
|---|---|---|---|---|
| Titanic Hydra, all | 2,189 | 0.64 – 1.07 | +1.46 | 1.4 – 2.3 |
| Heartsteel vs magic | 304 | 1.7 – 2.9 | +2.24 | 0.8 – 1.3 |
| Sunfire vs magic | 213 | 2.1 – 3.4 | +1.75 | 0.5 – 0.8 |

The headline (Titanic best) is borderline; the versus-magic ranking is not
established; "Hollow Radiance is the worst" is a point estimate two SE wide.
Nothing in the thesis is wrong, it is simply unquantified. We would cluster-
bootstrap by match for every number, as we already do.

### 2.2 Window semantics are undocumented — and they change what the statistic means

If every window ran to the terminal state and the model saturated there,
`ΔW̄` would equal `win rate − W̄₀`. The thesis's own Table 7 shows it does
not: Titanic +1.46 vs (51.94 − 51.41) = +0.53; Sunfire −0.51 vs −1.43; Hollow
−0.84 vs −1.51; Heartsteel +0.66 vs +0.49. So windows end early (sale, upgrade)
in an unreported share of cases and/or the DNN is not saturated at GAME_END.
The thesis never reports the share of terminal windows, the horizon
distribution, or the model's output at the terminal frame. The number's
meaning depends on all three. **Correction of my earlier verbal claim** that
it "collapses to an adjusted win rate": it would only under conditions the
thesis does not establish. We must report the horizon distribution and the
terminal-frame calibration explicitly (W1 gate, W2 design).

### 2.3 It adjusts for the observed state only

Conditioning on w(x) removes selection on what the WP model sees. Player
identity (one-tricks), team composition, intent, and the reason a player
chose the item remain. It is an outcome-model-only estimator with no
propensity term. A doubly-robust design (W4) is the standard remedy and we
own the missing half.

### 2.4 Hand-made contexts, first legendary only, no cost or timing

S is chosen by hand (physical vs magic opponent). Only the first legendary
item; no basket, no sequence, no gold efficiency, no purchase timing. The
thesis lists "optimal choice of S" as future work. A model of
`Pr(Y = 1 | x, a)` with a as an input (W3) is the direct answer, and our
board encoder already tokenises items — the "60 categorical variables"
objection does not apply to us.

### 2.5 Scale: the thesis's method verbatim is not feasible here

400,000 games gave n = 2,189 for one champion's most popular item. We hold
39,968 challenger/GM games, about a tenth (*measured*), so a comparable cell is
n ≈ 200 → SE ≈ 2.5–3.5 pp (*projection*). Bucketed means will be unresolved
for almost every (champion, item) cell. Feasible instead: a model-based
estimator with cross-fitting (W3) and hierarchical shrinkage across champions
and items (W2 second estimator), validated on the few cells that are powered.

### 2.6 The WP model itself is standard

Kim et al. 2020 had already trained and calibrated a comparable DNN. The
thesis's model is bigger-data, not new. Fine for us: a gradient-boosted model
on the same features is likely to reach the same regime in minutes of CPU
(*projection*, to be measured in W1).

---

## 3. Inventory — what we already have

All *measured* tonight unless marked. Read-only queries on `tracker.db`
(`mode=ro`); one long scan of `game_events` was **stopped** on purpose to keep
the disk quiet for the running 16.19 benchmark chain, so the event-type
breakdown is not measured.

| fact | value | source |
|---|---|---|
| unified games (`games`) | **39,968**, all queue 420: 16.17 = 20,859; 16.18 = 12,990; 16.19 = 6,119 | sqlite, `select patch, count(*) from games` |
| perspectives (`matches`) | 282,970 rows (per match × puuid) | sqlite |
| shop visits (`shop_visits`) | 3,394,921 | sqlite |
| narrator events (`game_events`) | 4,801,050 rows, all types (per-type count not measured) | sqlite |
| raw archive | `data/raw_match_v5/`: **43,874 timeline files**, 48,743 match files (`*.json.gz`, envelope schema 1 with `payload`); per-game coverage vs `games` not cross-checked | filesystem + `app/riot.py` RawMatchArchive |
| frame content (one sampled timeline) | `frameInterval` 60,000 ms; per participant: `position{x,y}`, `xp`, `level`, `totalGold`, `currentGold`, `goldPerSecond`, `minionsKilled`, `jungleMinionsKilled`, `timeEnemySpentControlled`, `damageStats` (12 fields incl. damage to champions and taken), `championStats` (25 fields) | gzip read of one file |
| event types in that timeline | CHAMPION_KILL, CHAMPION_SPECIAL_KILL, BUILDING_KILL, TURRET_PLATE_DESTROYED, ELITE_MONSTER_KILL, DRAGON_SOUL_GIVEN, OBJECTIVE_BOUNTY_PRESTART, ITEM_PURCHASED/SOLD/DESTROYED/UNDO, LEVEL_UP, SKILL_LEVEL_UP, WARD_PLACED/KILL, PAUSE_END, GAME_END | same |
| shop-event payload | ten-player `board` (items, `totalGold`, level, K/D/A, lane, `is_self`), objectives `score`, `win` | read from `app/reconstruct.py::_board_snapshot` (code, not a sampled row) |
| behaviour policy | M2 = frozen 0.5·econ_ft_new(grouped) + 0.5·wardfree_new; package `mixture-v4-m2-v1`; grouped decoder yields P(main-purchase group | x); 51.53 % first-exact on the v4 cohort | *recorded* (stage-2 dossier, HANDOFF) |
| in flight | 16.19 benchmark: generation `16.19-development-v1` (train 2,770 g / cohort 1,342 g); R1 = 50.943 % on the cohort; M2 scoring | *recorded* (HANDOFF 20:25) |
| macro_coach | 41 corpus games, each with `mv5_timeline.json` and `decisions.jsonl`; 25-token lexicon; snapshot at t₀ − 1 s with the ten players' state; rule: no reportable number under 30 games | filesystem + `LEXICON.md`, `docs/design.md` |
| refetch path if a timeline is missing | `app.ingest` has a re-fetch option for timelines | `app/ingest.py` help text |

Consequences: the thesis's Table 5 is covered **except Rank**, which is
constant for us (ladder pull). We additionally have xp, CS split, crowd-control
time, damage split by type, and champion stats per minute. The
thesis-style training set would be ≈ 1.1–1.2 M frames, or ≈ 230 k states under
its 1-per-5-min sampling (*projection* from ~28-minute games).

Constraints we inherit (*recorded*): chain freeze on `app/*.py` and pinned
scripts while a chain runs; build_tracker ↔ macro_coach share no code or data
except versioned artifacts with digests; POV discipline (`totalGold` is
spectator-only; the live estimate is items + own wallet); never `git add -A`.

---

## 4. Program — six bounded pieces (every budget is a projection)

Naming: **OMNI judge** = WP model on the full omniscient frame, items
excluded from inputs (offline scoring only). **POV WP** = same target on
observable inputs (live-safe). **Q** = P(win | board before purchase,
candidate action), items included by design. The three are never mixed.

### W1 — win-probability model as shared infrastructure

- **Data**: archived timelines, all three patches pooled; frames + events;
  one state per 5-min interval per game (thesis) and, as a second cache, all
  frames with match-clustered inference.
- **Split**: by game, temporal (train up to a date, test after; the 16.19
  window as the cross-patch transfer check). No game in both.
- **Features**: thesis Table 5 minus Rank, plus the extra frame fields; team
  differences as explicit features. Two models: OMNI judge and POV WP.
- **Baseline**: logistic on gold diff, kill diff, tower diff, dragon diff,
  time.
- **Primary**: ECE (20 bins) overall and per 10-min window; accuracy per
  4-min bin; reliability diagrams; all vs baseline; paired by game.
- **Gates**: no feature from the future (no GAME_END, `game_duration`,
  terminal-frame values); leakage check by outcome permutation within time
  bins; **terminal-frame calibration reported** (how saturated w is at
  GAME_END — needed by W2's window semantics).
- **Budget**: GBDT on ~1 M frames, CPU minutes to an hour; MLP < 1 h GPU.
- **Deliverable**: `winprob_v1` artifact (sha256) + calibration report.

### W2 — thesis-faithful value with uncertainty (items)

- **Unit**: a main-purchase group at a shop visit (unified events). x = OMNI
  judge at the visit's frame-interpolated state; z at horizons h ∈ {3, 6, 10
  min}, at sale/upgrade, and terminal.
- **Statistics**: W̄₀ and ΔW̄ per (champion, group, S) with S from the context
  taxonomy the product already uses (lane opponent, damage type, gold bracket);
  **cluster bootstrap by match**, 2,000 resamples, every number with its
  interval; cells reported only with n ≥ 100.
- **Second estimator**: empirical-Bayes shrinkage across champions and items;
  shrunk and raw shown side by side.
- **Pre-declared sanity**: replicate the thesis's design on one heavily played
  tank top laner. Expectation stated now: most intervals include zero.
- **Budget**: minutes once W1 exists. No product exposure.

### W3 — Q(x, a), the model-based estimator

- **Input**: the board before the purchase (same encoding family as the
  imitation trunk, trained from scratch on the d64 recipe, not the frozen
  trunk) + candidate main-purchase group. **Output**: P(win) (primary) and ΔW
  at the W4 horizon (secondary).
- **Cross-fitting**: 5 folds by match; no game is scored by a model that saw
  it. Advantage `A(x, a) = Q(x, a) − Σ_a' π_b(a' | x) Q(x, a')` with π_b = M2.
- **Validation**: calibration of Q on held-out; agreement with W2 in its
  powered cells (sign agreement, rank correlation) — the thesis-style
  statistics become the unit test.
- **Budget**: same order as one Bf training; reference epoch 1,734 s on
  757,975 visits × 16 epochs ≈ 7.7 h GPU (*recorded* for Bf; *projection*
  for Q). On the frozen v4 generation so rows are auditable.

### W4 — doubly-robust off-policy evaluation of a value-reranked policy

- **Policy class**: M2's top-3 reranked by A (or by W2 value). Overlap is
  guaranteed by construction: the policy never leaves the imitation
  shortlist.
- **Estimator**: DR with π_b = M2's grouped probabilities (cross-fitted), Q̂
  from W3, reward = **terminal outcome (primary, model-free)** and ΔW at h =
  6 min (secondary, lower variance, judge-dependent). Weight clipping and
  effective sample size declared and reported.
- **Primary**: value(π_e) − value(M2), paired by decision, cluster-bootstrap
  CI by match. **Cost reported**: exactness vs humans (expected to fall).
- **Population**: a cohort not used to train M2 or Q. The v4 and 16.19
  cohorts are exposed to scoring (*recorded*); declare that, or cut a fresh
  temporal cohort.
- **Budget**: inference only, hours of CPU.

### W5 — advantage-weighted imitation (only if W4 resolves positive)

Twin arms from one init, same recipe and batch order: loss weights
`exp(A/β)` vs uniform. Primary = W4's DR value; secondary exactness. One
Bf-scale training per arm.

### M1 — macro_coach diagnostic (parallel, cheap)

- WP curve with decision markers and per-decision ΔW in `review_app` /
  the web review, from the 41 timelines through the OMNI judge (frames
  interpolated to t₀ − 1 s). Diagnostic only: **no reportable per-token
  number** (41 games; per-token cells are tiny).
- W̄₀ per token as the selection-bias reading (e.g. whether invades happen
  only when ahead).
- Later proposal, not now: coarse jungle-pathing actions from frame positions
  (lane proximity per minute) across the 39,968 games — the thesis's framework
  fits decisions with short windows better than items.

---

## 5. Design decisions declared now

1. Two WP models, two purposes: OMNI judge (no items in inputs) for offline
   value; POV WP for anything live. Q includes items. Never mixed.
2. Windows: horizons + sale/upgrade + terminal, with the horizon distribution
   and terminal calibration always reported; one primary horizon fixed in the
   W2/W4 pre-registration.
3. Action unit = main-purchase group (grouped decoder), not the full basket
   — for overlap and for cell size.
4. Uncertainty: cluster bootstrap by match everywhere; nothing reported
   without an interval; n ≥ 100 for bucketed cells.
5. Data: pooled patches for W1; per patch for W2–W4 (item semantics change
   between patches).
6. Isolation: sibling project `videns.gg/winprob`; inputs = the raw archive
   read-only + versioned exports; outputs = versioned artifacts with sha256;
   nothing in `app/`; independent of the chain freeze.
7. Product line, if it ever ships: "Challengers buy X (p %) · value +a [lo,
   hi] pp at 6 min", abstaining when the interval includes zero.

---

## 6. Limits, stated with the number

- **Effect sizes**: thesis best case +1.46 pp at n = 2,189 across all ranks.
  Challenger choices are compressed; expect most |ΔW̄| < 1 pp, so W2 cells at
  n ≈ 200 (SE 2.5–3.5 pp) stay unresolved. W3/W4 exist for this reason.
- **Confounding**: only what x contains is adjusted. `puuid` is available for
  within-player contrasts as a robustness check, but sparse.
- **Horizon bias**: short horizons miss delayed payoffs (the thesis's Mejai's
  example); terminal windows carry the whole game's variance. Hence the
  curve, never one number.
- **Overlap**: DR is valid near π_b only. A policy that leaves the top-3 gets
  no estimate; that is a product constraint, not a bug.
- **Late game**: thesis ECE rises to ≈ 1.5 % after 20 min with 350 k games;
  we have fewer late states. Report per window.
- **Exposed cohorts**: v4 and 16.19 cohorts have been scored by R1/M2
  (*recorded*); any OPE on them is declared exposed.
- **Live**: a POV WP is an overlay feature, not a decision tool; the thesis
  itself excludes in-game automation.
- **Coverage**: 43,874 timelines vs 39,968 unified games not cross-checked
  per game; the refetch path exists.

---

## 7. Verified vs inferred

| claim | status | how |
|---|---|---|
| thesis content, numbers, tables | verified | full text read; numbers copied from Tables 5–9, §5.1 |
| SE ranges in §2.1 | inferred | SD bounds [0.3, 0.5] assumed; thesis gives no variance |
| window semantics differ from win rate − W̄₀ | verified from thesis tables | arithmetic on Table 7 |
| DB counts | measured | read-only sqlite tonight |
| raw archive counts and frame fields | measured | filesystem glob; one file opened |
| shop payload contents | verified from code | `_board_snapshot` in `app/reconstruct.py`; no row sampled |
| M2/R1/16.19 status | recorded | HANDOFF.md, stage-2 dossier |
| all budgets | projection | from recorded Bf epoch times or order-of-magnitude |
| "no win-probability work exists" | measured | grep over videns.gg, 0 hits |

Claims I am **not** making: any causal claim; any product change; any
number about our data beyond the inventory.

---

## 8. Costs (projection)

| piece | data | compute | engineering | owner |
|---|---|---|---|---|
| W1 | archive only | CPU minutes–1 h (GBDT); < 1 h GPU (MLP) | 1–2 days: frame loader, feature spec, calibration report, artifact | none |
| W2 | W1 + unified shop events | minutes | 1 day: horizon logic, bootstrap, shrinkage | reading |
| W3 | frozen v4 generation | ≈ 8 h GPU (Bf order) | 1–2 days: Q head, cross-fitting, agreement test | GO |
| W4 | W3 + M2 rows + a clean cohort | hours CPU | 1 day: DR, ESS, clipping, report | GO + reading |
| W5 | W3 + W4 | 2 × ≈ 8 h GPU | 1 day | GO |
| M1 | 41 timelines + W1 | minutes | 0.5 day in review_app | none |

Machine policy as today: GPU segments never during a game; CPU pieces run
BelowNormal during play.

---

## 9. Questions for the reviewer

1. **Primary horizon** for ΔW in W2 and as W4's secondary reward: 6 min,
   "until the player's next shop visit", or terminal? I propose 6 min with the
   full curve reported.
2. **W4 reward**: terminal outcome as primary (model-free, high variance) and
   judge-based ΔW as secondary — or the reverse? I propose outcome primary.
3. **W3 trunk**: train Q from scratch on the d64 recipe (clean attribution,
   ≈ 8 h) vs reuse the frozen imitation trunk (cheaper, entangled). I propose
   from scratch.
4. **Policy class for the product**: reranking within M2's top-3 is what
   guarantees overlap. Is that the right class, or should we also estimate
   "top-3 ∪ any legal item with π_b ≥ ε"?
5. **Order**: W1 → W2 → W3 → W4 as proposed, or skip W2's bucketed
   statistics entirely and use W3's agreement test only where W2 is powered?
6. **Population for W4**: declare the exposed v4 cohort, or cut a fresh
   temporal cohort from 16.19 (which is also being scored today)?
7. **Rank**: constant for us. Do you want a within-ladder proxy (LP band) as
   a feature, or is its absence acceptable for a challenger-only judge?

Recommended answer set if you want a default: (1) 6 min, (2) outcome
primary, (3) from scratch, (4) top-3 reranking, (5) W1 → W2 → W3 → W4, (6)
fresh 16.19 temporal cohort, (7) no proxy in v1.
