# Frozen BC20 on ordinary shop decisions

Protocol v1, declared 2026-09-30 before selection payload reads or scoring for
this endpoint. Status: root-reviewed development protocol, fixed before payload
selection execution or ordinary-policy scoring. This is
an inference-only follow-up to the already exposed revision challenge. The
BC20 checkpoint is selected from that earlier result; neither the model nor
this population is fresh confirmation. No terminal outcome is a target of
this experiment. Qualified source replay internally reconstructs final match
fields already present in its DTOs; disclose that access rather than claiming
the source deserializer or replay never encounters them. No win target is
exported, analyzed or used for selection, features, scoring or model choice.

## Question and fixed decision

Does frozen ordinary 20-epoch BC improve the first recommendation by reordering
the existing frozen M2 displayed candidates on ordinary source-admitted shop
decisions? The primary candidate is **BC20 reranking M2's unchanged top-three
legal option set**, with an explicit M2 fallback when BC cannot encode a state.
This tests a small concrete change to the existing recommendation interface.
It does not test whether imitation raises win rate or identifies a better item.

There is one primary arm and one primary accuracy endpoint. Report a fixed
half-M2/half-BC geometric pool and BC20 through the existing displayed decoder
as secondary arms, and unfiltered BC group argmax as a catalogue diagnostic.
Do not select a mixing coefficient,
checkpoint, decoder, subgroup, or winner after seeing these labels. A positive
secondary result cannot rescue a failed primary gate.

## Population and selection before payloads

The identity universe is exactly the 859 patch-16.19 games marked development
in the current frozen admission v2. Validate this identity equality and the
union of global reservations, all 1,911 confirmation IDs, external project
reservations and admission ancestors before any corpus or raw payload read.
No calibration, FIT, sealed test or confirmation game is added. These games
have already been exposed through development work and revision discovery;
ordinary retained-basket labels form a different endpoint, not a new holdout.

Use native source `decision_admission.json` metadata, whose admitted entries
contain event index, participant ID and timestamp. Freeze one focal player
and at most one decision per game before opening events, targets or scores:

1. Focal participant is `1 + big_endian(SHA256("bc20-ordinary-v1/focal/" +
   match_id)[0:8]) % 10`. Do not choose a champion, role, side or player with
   more decisions, and do not replace an empty focal player's selection.
2. Among existing admitted entries of that focal player, select the smallest
   SHA256 digest of `"bc20-ordinary-v1/visit/" + match_id + "|" + pid + "|" +
   event_index + "|" + timestamp_ms`, with integer fields in ordinary decimal.
   Break a digest tie by `(timestamp_ms, event_index)`. Duplicate keys fail.
   Use all admitted timestamps; do not add a target-dependent time filter.
3. A focal player without an admitted entry produces `no_admitted_decision`.
   Preserve that game in the 859-game selection ledger; no alternate player,
   visit or game is sampled. Such a game has no ordinary-decision accuracy
   endpoint. Report its count and the conditional nature of the estimand.
4. Publish the complete selection JSON plus SHA256 sidecar before payload
   reads. Bind admission, native decision mask, qualification and split
   metadata, sampler code and this protocol. No revision-pair membership,
   target kind, basket availability, policy score or source outcome influences
   this selection.

The estimand gives equal mass to each selected decision-bearing game: a
deterministically sampled focal player and admitted decision, conditional on
that focal player having a source-admitted decision. It is neither all shop
visits nor all opportunities to recall. The existing admission mask itself
limits the population to decisions the source can reconstruct.

## Source reconstruction and target boundary

The source project owns native 16.19 reconstruction, indexed reads, replay
proof, source qualification and economic semantics. Reproduce the selected
entry and its original retained target without importing source code into
winprob. Selection identity mismatch, failed replay, changed metadata or
reservation overlap invalidates the run rather than selecting a replacement.
An explicit unavailable source row is retained in the selection ledger; it
cannot silently disappear from counts or become a different visit.

Export the canonical predecision state and full count-preserving retained
basket, with separate target kind: observed no-buy after death, ward-only
purchase, mixed purchase including wards, or other purchase. A no-buy is only
the source's explicit death-with-no-buy observation. Do not invent a manual
recall/no-purchase label. Keep targets outside policy input objects. Preserve
target multisets missing from either frozen catalogue; they are misses for
that action-space metric, not an eligibility filter. A full basket absent
from the full catalogue can still have a represented main basket; report both
support indicators and score each endpoint in its own declared action space.

BC20 consumes only the existing public IQL feature allowlist and static
vocabulary, with native16.19 attributes. No new feature fitting, standardizer,
role inference, future frames, targets, gold-arrival receipts, total gold or
outcome metadata enters the new actor. The frozen M2 projection retains its
legacy inferred-role exposure, explicitly disclosed. One predecision
`prequential-v3` estimated wallet supplies M2 scoring and its decoder; never
add observed spend to a previsit wallet.

## Policy API, candidate set and economic worlds

The verified source API is `build_tracker/app/mixture_policy.py`:
`MixturePolicy.probabilities(rows)` returns all 9,333 grouped masses;
`MixturePolicy.options(rows)` sorts the frozen shortlist, calls
`purchase_sequence`, and returns up to three distinct options, with SAVE
fallback if needed. Group IDs map to multisets through `group_main` and
`action_of`. The package identity is
`ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9`.
Do not treat its learned masses as human logging propensities.

Runtime activation pins M2's economic world to16.18. The source prestate is
native16.19. Record both; do not silently call the pinned decoder a native19
legality proof. Use the exact unmodified source `options` output as the common
ordered candidate set C(x), including cases with fewer than three options.
Export the full group masses, candidate objects and per-row hashes. Replaying
the same native decoder with exported probabilities must reproduce every
complete option object and its order. Mutating/removing target and future
metadata must leave all masses and options unchanged.

Audit every candidate separately in the source's native16.19 economic runtime
on the same canonical predecision state and estimated wallet. Preserve the
purchase sequence and its actual incremental cost, or explicit infeasible/
unknown status. Do not filter, replace or expand C(x) after this audit. The
audit therefore exposes cross-world failures instead of changing the primary
policy. SAVE has cost zero. Both the pinned-world and native-world legality
checks are conditional on reconstructed wallet/slot/quest inputs; they do not
prove exact live affordability. Inferred roles in the legacy economic engine
remain a stated limitation.

The direct-BC secondary uses the same source decoder with the exported BC
group distribution substituted for `probabilities`, through the existing
source probability-override pattern. This requires no new legality algorithm
and no new M2 forward pass. Its own returned candidates receive the same
separate native19 audit. They do not replace or expand the primary M2 set.

## Frozen scores and arms

Let l_b(x) be the frozen BC20 full-action logit for each of13,166 complete
retained-basket actions. Compute its main-group log mass up to one common
normalizer by L_g(x)=logsumexp_{b:full_to_group[b]=g} l_b(x). Keep every ward
variant in this sum; comparing one ward-free full action with M2's marginal
would change the action unit. The mapping and checkpoint are digest-bound.

- M2 control: original first option and complete original C(x).
- Primary BC20 reranker: order C(x) by descending L_g(x); exact ties keep M2
  order. Return the same complete option objects, not rebuilt baskets. If the
  frozen feature compiler cannot encode the context, return original M2 order
  and record `bc_feature_fallback`; do not remove that decision.
- Cheap control: order the same C(x) by descending native19 incremental
  purchase-sequence cost, ties in M2 order. An infeasible/unknown native audit
  makes this control unverified and prevents the success gate; no guessed
  price or full-total-price substitute is used.
- Fixed secondary pool: order C(x) by 0.5*log P_M2(g|x)+0.5*L_g(x), ties in
  M2 order. Exact zero M2 mass is negative infinity; no tuned epsilon. If all
  candidate masses are zero or BC features unavailable, use original M2
  order and record the fallback. This is a fixed geometric score, not a
  newly calibrated probability.
- Direct BC20 secondary: normalize the complete grouped BC masses and export
  all9,333 float32 probabilities in the exact M2 group order. The source
  decoder consumes that distribution through its probability override and
  retains the original shortlist size, option limit, fallback, pinned world
  and economic checks. Reuse unmodified option objects and separately audit
  all returned candidates in native19. BC feature failures use original M2.
  This arm may have a different candidate set; it remains secondary whatever
  its score and cannot satisfy the primary reranker gate.
- Catalogue diagnostic: BC20's unfiltered grouped argmax/top3 over all9,333
  groups. Report prediction and target support, but do not call it a displayed
  or legal recommendation. No second legality implementation is built for v1.

Direct BC diagnostic scores are useful for distinguishing weak candidate
recall from a weak BC ranking. They are not a prerequisite for interpreting
the primary reranker, and cannot replace its metric if they look better.

## Primary endpoint and explicit ward-only semantics

Primary endpoint is **ward-stripped main-basket first-recommendation exactness** on
all selected decision-bearing games, with equal weight per game. Remove only
control-ward IDs2055 and772043 from a purchase target, preserving every other
item and count. Compare the entire resulting main multiset with the displayed
option's main multiset. Both an explicit observed no-buy and a ward-only target
have an empty main multiset; group0 therefore receives the same credit that
the original M2 compatibility metric gives it. Preserve this strong baseline:
do not remove correct M2 ward-only main predictions before testing BC. This
credit means correct *main-purchase agreement*, not a correct SAVE decision
or a correct ward quantity. Mixed purchases also measure main-basket
agreement only. Main targets
absent from the catalogue remain zero, including in candidate-recall metrics.

The same metric denominator must be used for point estimates, contrasts and
intervals. A BC fallback remains scored using its M2 recommendation. If a
selected source row cannot be reconstructed, keep it as an explicit unavailable
row and report both the decision-bearing selected denominator and completed
prediction coverage; count it as zero for all arms in the conservative primary
ledger, and show the available-row result as a diagnostic. Identity/provenance
failures invalidate the run rather than becoming ordinary zero-credit rows.

Also report, without substituting for the primary endpoint:

- Typed main-basket accuracy: group0 is correct only for explicit observed
  no-buy; ward-only targets receive zero because these main-only cards cannot
  express a ward purchase. This separately named diagnostic cannot replace
  the primary compatibility endpoint. Report strict complete-basket exactness,
  which preserves wards and counts, and paired contrasts as well.
- Fixed M2 candidate-set main top3 recall/ceiling, oracle first-choice accuracy,
  and the gap between first-choice accuracy and this ceiling. Reranking cannot
  improve top3 set recall; assert equality of candidate membership for M2,
  the primary reranker, spend control and geometric pool. Report direct BC's
  different candidate-set recall as secondary, never as a reranking gain.
- No-buy, ward-only, mixed and other-purchase counts; candidate cardinality;
  full/main catalogue support; BC feature fallbacks; zero-mass fallbacks;
  M2/BC disagreements, BC-only wins, M2-only wins, and both-wrong cases.
- Primary purchase-only and explicit no-buy slices when each has at least30
  games. Other small slices report counts and `not_measured`, not unstable
  apparent wins. Keep every declared arm and slice visible.

## Uncertainty and fixed success gate

Use2,000 paired whole-game bootstrap resamples, seed20260930; one observation
per selected decision-bearing game. Reuse exactly the same sampled games for
all arms. Report absolute percentage-point changes and paired percentile95%
intervals; independent arm intervals are not a test of their difference.

A useful development reranking signal requires all of the following:

1. At least700 selected decision-bearing games, at least95% with complete
   source reconstruction and M2 predictions, and a fully accounted859-game
   selection ledger. All mapping failures remain in the accuracy denominator.
2. Primary BC20 reranker gain over M2 is at least+3 percentage points and its
   paired95% lower bound is strictly above zero.
3. Primary gain over the fixed incremental-spend control has a paired95%
   lower bound strictly above zero. No subgroup or secondary pool can supply
   either missing condition.
4. Zero changed/added candidates in the primary reranker/pool/control, zero target-mutation or native-decoder
   parity failures, and zero infeasible/unknown returned candidates in either
   declared economic world. Report direct-BC legality independently; a failing
   secondary cannot be claimed useful. Report failures even if accuracy rises.

These are conjunction gates, not evidence of independent confirmation. With
859 observations, a small gain can remain unresolved: for paired credit
difference D in{-1,0,1}, its standard error depends on discordance as
sqrt((Pr[D!=0]-E[D]^2)/n). Sample size alone does not establish power; report
observed discordance and interval width. Failure to clear the gate is not
proof the frozen actor has no signal. Success establishes a broader exposed
development recommendation-agreement signal, not win benefit or permission
to deploy. Fresh confirmation and a source-native displayed-policy review
would require a separate declared step; this trial keeps those games closed.

## Artifacts, resource bound and stop conditions

Cross-project exchange is versioned files only. Source-owned artifacts bind
selection, per-game indexed/replayed source evidence, native predecision
contexts, target records, full M2 group masses, exact candidate objects,
native19 candidate economic audits and all code/static/runtime digests.
The direct secondary adds a versioned winprob-owned BC group-distribution
artifact consumed by the source-owned unchanged decoder. Winprob binds the
frozen BC20 checkpoint and IQL feature compiler, catalogue,
input hashes, predictions and report. Consumers revalidate reservations and
all artifacts before scoring and before final publication. Every859 identity
has one terminal ledger status; selected keys cannot change between stages.

No training, refitting, critic query, terminal-outcome target analysis, broad event-table
scan, new collection, source implementation edit, serving change or report
index edit is part of this trial. Source inference and native-world audit run
in separate clean processes because runtime activation changes module globals.
Use CPU1 streaming source work and small batches; the frozen actor may use one
idle GPU under the existing no-game guard. No dense training matrix is needed:
at most859 by9,333 M2 masses, plus bounded BC inference batches. Durable stages
have a combined60-minute ceiling; stop rather than expand scope or drop hard
rows if resource, integrity or coverage gates fail. Root reviews this protocol
and the metadata selection interface before payload execution.

## Verified code basis

Read-only inspection for this design covered source `app/mixture_policy.py`
(full masses, pinned runtime, exact options), `app/conditional_policy.py`
(`purchase_sequence` preserves counts and returns incremental costs),
`scripts/eval_policy.py` (label removal and paired population rules),
`scripts/basket_wardfree.py` (legacy SAVE/ward-only collapse),
`scripts/export_winprob_revision_contexts_v3.py` (native metadata/runtime and
reservation boundaries), and winprob's frozen IQL feature/training protocol.
No new ordinary-decision payload, prediction or target was read for this design.
