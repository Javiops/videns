# Public scoreboard information: fixed prediction test

Declared 2026-10-01 after complete feature recovery and validation, before
any model fitting or new accuracy comparison. The actual retained purchase is
the target. The multiset and fixed-memory candidate recipes remain closed.

## Hypothesis and fixed inputs

Other players' cumulative kills/deaths/assists may contain purchase-relevant
information not represented by their champion, inventory and level. Existing
M2 and own current-state features omit these scores. This is a test of omitted
observable information, not a claim that KDA must be predictive or that the
current model is broken.

Source export: `build_tracker/data/exports/winprob-scoreboard/scoreboard-export-20261001-v1/manifest.json`,
SHA256 fee70f9d7be28235e07b22b409471833bd4674d33a10abfc9939ae046fa2eeb2.
Own prepared cache: `data/runs/scoreboard-features-20261001-v1/cache/manifest.json`,
SHA256 c3a0f735523efac4eb040ca138f720196cb8a323194f3a29666119d9173ae786.
It covers all392448 original FIT states /4001 games, preserves actor order and
uses only original stored cumulative KDA. Counts are divided by20, matching
the established own-KDA scale; the buyer's entire new channel is zero.
No recency, exact death age, hidden enemy gold, final roles or damage data.

Revalidate all current reservations and source/base/cache digests before reads.
Use original whole-game folds and the same3350 fixed focal OOF decisions.
The651 missing focal episodes stay disclosed. No new player/visit selection,
859-game development payload, confirmation/test read or win/outcome input.

## Three matched arms

All three share the same expanded architecture, copied initial tensors and
the ordinary categorical main-basket loss. The common body remains the existing
PublicStateEncoder, with one addition at the first per-player linear layer:

    z_j = W_base u_j + b_base + W_score k_j.

Here k_j has three normalized KDA channels. W_score is3->64 with no bias and
starts at exactly zero. The original per-player nonlinear layers, team
mean/max pooling, board encoder and128->128/ReLU->9333 categorical head remain.
This adds192 parameters, equally present in every arm. It must reproduce the
base prediction at initialization with scores masked/zero projection. The
common encoder/head initialization uses seed20261001+fold and is copied;
models must not share mutable parameter storage.

- `zero`: all new KDA channels zero throughout training and inference.
- `scoreboard`: real cumulative KDA of the nine other players; focal zero.
- `shuffled`: the same per-player KDA triples permuted within the four allies
  and five enemies separately, using the frozen prepared permutations. This
  preserves team score multisets/totals while breaking assignment to champion
  and inventory. No donor comes from a different game or training partition.

The permutation is fixed by SHA256 of
`scoreboard-v1/shuffle/{match_id}|{participant_id}|{event_index}`. It is never
chosen using a purchase label. All model inputs remain past observable state.

Train from clean initialization for20epochs, batch256, Adam3e-4 with default
betas/epsilon and weight_decay0, gradient norm cap10, float32/noTF32. Use the
same epoch order in all arms (SHA256 of `scoreboard-probe-v1/order/{fold}/{epoch}`
seeds a NumPy permutation of opposite-fold state IDs). No checkpoint selection,
extra epochs, hyperparameter search, warm start or metric-dependent filtering.
Report initial hashes, update/epoch counts and every finished model.

## Primary decision rule

First main-basket exactness of `scoreboard` must beat BOTH `zero` and the
stronger frozen internal memory reference (1584/3350=47.2836%) by at least3pp,
with positive paired95% lower bounds. Mean main top3 coverage must not fall
below the memory reference. Require all feature/identity/training integrity
gates and complete coverage. A gain against a weak zero control alone is
insufficient to justify an M2 trial.

Additional construction-bias guard: on decisions with a NONEMPTY recorded
main purchase (main_group!=0), scoreboard-zero first-exact difference must
have a positive paired95% lower bound. This prevents a gain confined to the
synthetic no-buy/ward-only group from being presented as improved main buys.
These targets define an evaluation slice only; they cannot enter features,
model selection or sample replacement.

Report scoreboard versus shuffled with paired intervals to test whether
assignment to individual champions matters beyond team aggregates. This is
mechanism attribution, not an alternative candidate or rescue gate. Report
main top1/top3, NLL, corrected/broken cases, patch and purchase-size slices,
plus observed no-buy/ward-only behavior. Small slices carry their counts and
are descriptive; no selected slice substitutes for the primary.
Bootstrap2000 paired whole-game draws, seed20261001. All internal decisions
are already exposed; intervals are conditional on this fitted recipe.

Passing justifies a separately frozen direct M2 comparison with source-native
legal recommendations. It does not establish M2 benefit, causality, live API
parity or goal completion. Failing closes this fixed recipe without tuning it
on these outcomes. No previous failed model is rescued by a secondary metric.

## Execution and current status

CPU1, own idle CUDA, startup hostRAM>=3.25GiB/live>=0.75GiB, no League or source
benchmark/capture. Hidden durable job, fixed60-minute worker. Before real
fitting, measure synthetic end-to-end training and actual-store inference;
project the complete six-model work with1.25margin+180s overhead and require
it to fit80% of the remaining bound. No silently truncated experiment.

The source exporter passed seven projection/mutation tests; own normalization
and control code passed five tests. The complete source and own feature caches
exist and passed identity/focal parity checks. The matched model implementation,
full preflight, real training and performance evaluation have NOT run yet.
