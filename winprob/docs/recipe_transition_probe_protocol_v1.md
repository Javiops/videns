# Native-M2 recipe-transition correction: fixed internal screen

Declared2026-10-01 after the synthetic numerical/native feasibility checks,
before new real-data prediction or fitting. Mathematical definition and source
inspection: `recipe_transition_design_v1.md`. The user's observed retained
purchase is the only supervised answer. No causal/win-value claim is tested.

## Population, separation and selection

Use only the3518 patch16.17 games already in the original4001-game FIT cache
`data/runs/iql-cache-20260930-v1/cache/manifest.json`, SHA256
1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea.
The483 patch16.18 FIT games are excluded from this screen because M2 trained
on16.18. M2's package declares16.18-only fitting; verify source training-ID
metadata has no intersection with the selected16.17 IDs before source reads.
This must not import a source model into the own project.

The original fold assignment is unchanged:1761 patch16.17 games in fold0 and
1757 in fold1. Fit each correction only on the opposite whole-game fold.
No game, player perspective or other selected state from its evaluation fold
can train or standardize that correction. M2 is the same frozen prior in both.
It was trained on a later patch than this retrospective internal screen; this
is NOT a chronological transfer result or a confirmatory deployment test.

Primary evaluation is the2900 original focal decisions with patch16.17 and
status=selected in the existing history selection manifest. The618 original
no-eligible-focal-episode games remain explicitly unavailable, never replaced.
Selection SHA256 fcc372f7598ea87d0d1e6e9456ea5cd143b21198f870234ac55306a2b353e3f7
at `data/runs/history-cache-20260930-v1/cache/selection.json`.
These focal decisions are already exposed research development, not fresh data.

Build a new metadata-only selection of at most16 original cached decision states
per admitted16.17 game. Include the original focal state when available; fill
remaining slots by smallest SHA256 of
`recipe-transition-v1/sample/{match_id}|{participant_id}|{source_event_index}`.
Without a focal state select the smallest16. Keep fewer when a game has fewer
eligible original states, report every count, and never sample with replacement.
Use state indices, episode/game metadata and existing source-event indices only
for selection. Persist IDs and digests before opening source trajectory rows or
current-action arrays. Do not select by M2 correctness, target support, win,
purchase size, affordability or predicted option quality. Maximum56288 states.

Revalidate all own/global/source reservations before payload reads. Source
`build_tracker/data/exports/winprob-fit-trajectories/16.17-iql-trajectories-20260930-v1/manifest.json`
and its `trajectories.jsonl.gz` (SHA256
1b5a999c5ac41bb4e86088e1c6653763e4c847e725a8d76dd3f27b722aee9e48)
are existing certified FIT-only artifacts. Stream one episode at a time; match
selected states to exact identity/visit receipts. No raw archive, mixed training
export, calibration,859-development or sealed payload read.
Terminal metadata is necessarily deserialized in these existing episode
envelopes, but never analyzed or used as an input, label or selection criterion.
The original complete/catalogue-supported episode restriction is inherited and
must be disclosed; this screen does not repair its population coverage.

## Source-owned release

Add a NEW source-owned adapter. Inspection of encode_record in the FIT v2
exporter established that it deliberately removed reference_compatibility;
do not assume those legacy fields are present in the trajectory artifact.
Recover only M2's legacy roles in a separate source-owned stage from the3518
admitted games in the original16.17 qualified SQLite database, read-only using
an indexed query per game. Revalidate each whole indexed game/event digest
against its original certified game_bindings receipt before using any role.
Follow baseline.py exactly: focal event.team_position or empty string, other
board player's team_position or empty string. Join other players by verified
team/champion identity, retain the original prestate order, and require full
roster parity. Append roles only to a separately tagged reference projection;
all other prestate fields come unchanged from the certified FIT artifact.
No new reconstruction, raw reads, full-table scan or new game admission. Hashing
the indexed payload necessarily deserializes targets/outcomes; neither may
enter role projection or residual features. Publish this sidecar before M2
scoring, which runs in a separate clean process without database access.
Freeze package digest
ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9.
Run clean M2 native16.18 world, batch16, CPU1 and deterministic CUDA with
CUBLAS_WORKSPACE_CONFIG=:4096:8, noTF32. Use constant dummy labels for the
native dataset adapter. Call its original probabilities/options and replay
each returned main basket with native purchase_sequence/try_purchase. No
consumer-runtime imports and no frozen source/model/database edits.

Serialize original options in their original order, canonical group IDs,
selected full-catalogue probabilities and the sum of all other probabilities,
the public residual context, and step-by-step hypothetical native receipts.
Retain multiplicities, costs and after inventories; bind native static item
attributes/world. M2's legacy final-role fields remain explicitly uncertified
as live-safe and never enter new residual features. These are16.18-engine
recommendations applied retrospectively to16.17 states, not a16.17 simulator.

Write actual retained targets to a SEPARATE file keyed by selected identity.
The target must not modify candidate membership or descriptors. Map a target
outside the native options to OTHER; never discard it. The exporter checks
input-tensor and exact prediction/option equality after mutation of ignored
target/future fields on the200-state real throughput pilot. Own code then
checks complete receipt conservation and computes its26+26 fixed descriptors.
Consumer files cross the project boundary only as versioned digested releases.

Before the full export, score the200 selected states with the smallest sample
hashes (without reading their actual targets). Their states may be inspected
only after full selection has been saved. Record actual source throughput and
peak RAM/VRAM. Project1.5*mean_cost*selected_states+900seconds, after warmup,
and require it fits80% of the remaining180-minute worker budget. If it fails,
stop without shrinking data or changing the statistical comparison. The pilot
predictions can be reused unchanged in the release. No accuracy is inspected
until the complete export and all correction fits/predictions are frozen.

## Fixed corrections

Three arms have the same26x26 coefficient matrix and start at0:

- `raw`: the ten net-attribute coordinates are zero after standardization.
- `net`: actual candidate-specific net attributes from native receipts.
- `shuffled_net`: permute only the ten net-attribute coordinates between the
  original options of each state; leave common/raw/economic attributes fixed.
  Preserve the complete ten-vector. Permutation comes from the first8 bytes
  of SHA256 `recipe-transition-v1/shuffle/{match_id}|{participant_id}|{source_event_index}`
  interpreted big-endian as the seed of NumPy default_rng. No cross-game donors,
  labels or metric selection. Single-option states have the identity permutation.

Fit each fold's context and candidate standardizers on its training states
only. Context intercept remains1. Other coordinates get weighted mean/std,
with scale1 for variance<1e-12. Each game has equal weight, its selected states
equal weight within game; each present candidate equal weight within state for
candidate standardization. Padding stays exactly0. Compute these statistics
from the true descriptors once per fold and share between arms. Shuffle net
vectors before transformation; zero the raw arm's net block after transformation.

Minimize the exact four-category coarsened likelihood including OTHER, with
game-balanced row weights and ridge lambda=.01. L-BFGS-B, max300iterations,
gtol1e-7,ftol1e-12,maxls50; no sweep or validation-based stopping. Require
successful optimization and max absolute gradient<1e-6. Exactly six fits.
No source prior tuning, candidate expansion or extra model training. Freeze
coefficients and standardizers before held-out prediction. Prediction receives
no true label, and a zero correction must return the original option order.

## Metrics, uncertainty and fixed decision

The primary is net main-basket first exactness over ALL2900 focal decisions,
including targets outside D. Preserve quantities; strip only2055 and772043.
Report counts/rates for all arms and unchanged M2, paired2000-resample game
bootstrap95% intervals with seed20261001, corrected/broken cases and NLL of
the coarsened category (not full-catalogue NLL). Report complete-basket first
exactness separately, without calling ward-only a correct full no-buy.

Proceed only if net beats M2 by at least3percentage points with positive95%
lower bound AND beats the matched raw control with positive95% lower bound.
Also require a positive lower bound versus M2 on nonempty-main true purchases,
all2900decisions scored, all integrity checks, and unchanged native option sets/
top3 coverage/legality. Real versus shuffled net is mechanism attribution;
an unresolved comparison weakens that attribution and must be reported. It
cannot rescue a failed primary. No metric-dependent exclusion or slice selection.

Failure closes this fixed recipe. Passing justifies a separately frozen
859-game forward development comparison using its already exported original
M2 probabilities/options and native16.19 feasibility audit. It does NOT open
confirmation or establish the goal. Do not add the hypothetical internal gain
to M2's historical gain or extrapolate it to live sessions.

## Resource and artifact rules

Hidden durable source task, CPU1, startup RAM>=3.25GiB,live>=0.75GiB,freeGPU>=3GiB,
no League or concurrent source benchmark/capture.180-minute worker and210-minute
launcher bound. Own correction is a small CPU1 job; measure its actual projected
time before fitting and bound the complete six fits/evaluation to30minutes.
Save complete source selection/code/static/package hashes, input/output digests,
costs, controls, full predictions and an independent final audit. Preserve all
versions and any stopped run. No promotion, new Riot collection or index edit.
