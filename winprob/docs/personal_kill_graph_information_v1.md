# Personal kill relationships: information recovery before a new policy test

Declared2026-10-03 before reading any new raw timeline payload for this study.
The previous net-recipe comparison is closed. This hypothesis adds observed
relationships omitted by M2; it does not retune that failed descriptor recipe.
No improvement or breakthrough is established by this proposal.

## Verified distinction and public availability

Inspected build_tracker/app/live.py, scripts/train_prefix.py and the frozen
reconstruction. Current M2 inputs do not include a directed killer-victim graph.
The previous scoreboard experiment tested cumulative player KDA under a
different learner, not buyer-specific edges conditional on those totals.
Two histories can have identical K/D/A for all ten players but different
opponents killing the buyer. Synthetic examples must demonstrate this exact
distinction. A kill graph records last-hit attribution, not damage, pressure,
skill, an optimal counter-item or causal purchase value.

[Riot's official sample events](https://static.developer.riotgames.com/docs/lol/liveclientdata_events.json)
show ChampionKill with KillerName, VictimName and Assisters. The
[official API documentation](https://developer.riotgames.com/docs/lol?from=20423&from_column=20423)
exposes event history and player scores. Existing local source code/measurements
also document untagged names and alias ambiguity. No current game API, capture,
or live corpus is opened in this stage. A later live adapter must resolve only
unique observed identities and verify KDA parity; missing or ambiguous events
must produce unavailable features, not guessed identities or invented zeros.
The archived graph's predictive value and live parity are separate questions.

## Causal alignment through an observable invariant

Frozen reconstruction processes frames/events in ORIGINAL ARCHIVE ORDER,
without a new timestamp sort. Every valid CHAMPION_KILL increments exactly
one player's deaths, including an environment kill with killerId0. Therefore
the sum of the ten OBSERVED prestate death counts uniquely identifies the
number of processed champion-kill events in that stored snapshot.

For each selected state, take that exact prefix, reconstruct all ten K/D/A
triples, and require exact equality with the existing source scoreboard sidecar.
Also require every included event timestamp<=the recorded decision timestamp.
Assists enter parity checking only; the initial graph features use killer-victim
counts. Reject invalid participant identifiers or unsupported same-team kills.
Do not guess a nearest prefix or silently alter the selected population.

This matters for synthetic no-buy rows: the source takes a death snapshot and
timestamps its decision15seconds later. A naive all-events-before-decision
rule could add later interactions absent from the original state. Matching
the observed all-player KDA preserves the source observation, including ties.
No event recency, last-death age, last-kill timestamp, SAVE flag, damage recap,
position, future outcome or post-purchase inventory is a model feature.
Event times and prefix ordinals exist only for provenance/causal validation.

## Degree-conditioned mathematical feature

Let C[j,i] count enemy j killing buyer i in the matched prefix. Let K[j] be
j's total champion kills, T the opposing team's total kills, and d[i] the
buyer's deaths to enemy champions (environment deaths are separate).
For T>0 the random-mixing expectation conditional on killer/victim degrees is

    E[C[j,i] | degrees] = d[i] * K[j] / T.

The proposed excess relation is

    r[j,i] = (C[j,i] - d[i]*K[j]/T) / (d[i]+5).

Set it exactly0 when T=0 or d[i]=0. The5 is fixed shrinkage, not a fitted
window or inferred confidence level. This is a centered descriptive feature,
not a future-hazard probability. Compute its numerator with integers:
T*C[j,i]-d[i]*K[j], whose sum over opponents is exactly0.

For each enemy, psi[j] contains the six existing champion tags, ten declared
numeric owned-item attributes and visible level/18 (17coordinates). Aggregate
the excess-weighted profile g[i]=sum_j r[j,i]*psi[j]. Subtract a fixed opponent's
profile inside that sum to cancel common profile components exactly, using
the zero-sum integer numerator. No learned embeddings, target or legacy final
role enter this projection. A later edge-only correction can use

    q(j|x) proportional to p_M2(j|x) * exp(g(x)^T theta f(x,j)),

with original native candidates plus OTHER, and the original16common/raw
candidate descriptors. That would have272parameters. With no intercept and
noncentering RMS scales, a zero excess profile leaves M2 exactly unchanged.
This is a proposed continuation, not authorization to fit/evaluate in this
information-only stage. A separate fixed real experiment must precede fitting.

The control samples a directed bipartite configuration graph: repeat killer
IDs by row degree and victim IDs by column degree, randomly permute the victim
stubs within each opposing-team direction, then count pairings. It preserves
every player's total kills and champion-attributed deaths EXACTLY; environment
deaths and original assists stay unchanged. It tests associations beyond node
statistics. It is not a temporally valid alternative match or a causal intervention.
Seed from the first8big-endian bytes of SHA256
`personal-kill-graph-v1/null/{match_id}|{participant_id}|{source_event_index}`.
Never use labels or select among shuffles.

Degree correction is motivated by
[Karrer and Newman, Stochastic blockmodels and community structure in networks](https://arxiv.org/abs/1008.3926).
We do not fit their blockmodel or claim that their community-detection results
predict League purchases. The conditional stub expectation and application
above are our construction; any improvement must be measured against M2.

## Fixed source admission and first deliverable

Reuse EXACTLY the3518original16.17FITgames/56288states/2900focals selected in
data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json,
SHA979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6.
No new game/visit selection. The618original missing focal episodes remain.
All current source/global/own reservations must be revalidated before payloads.

Source-owned access only: use each selected game's already certified timeline
path/bytes/SHA in the16.17 FIT trajectory manifest's game_bindings.raw_sources.
Read only its timeline envelope, not the match payload, database, mixed export
or live captures. Require canonical raw path, envelope identity and exact
compressed digest. Whole JSON deserialization necessarily includes other event
fields, but only timestamp/killerId/victimId/assistingParticipantIds of needed
CHAMPION_KILL events are projected. No purchase/outcome analysis or collection.

The existing scoreboard export supplies int32 all-player KDA and participant
order for original state indices. Validate its full admitted-FIT file digests
(the old sidecar also contains483unselected16.18FITgames), then index only the
56288 selected states. Recover public participant/team/champion rosters from
the already published source role receipts, ignoring all role strings. No
cross-project Python import. Exchanges are versioned digested artifacts.

Publish a sanitized per-game kill-prefix ledger, public roster, exact selected
state query identities and source KDA witnesses. Truncate each game's exported
kill stream at the maximum prefix needed by its16selected states. The source
and own independent consumer both verify all KDA/prefix/temporal relationships.
Prototype checks must cover same-KDA/different-edges, future suffix mutation,
same-time ordering, stale no-buy observation, environment kills, count
conservation, roster permutation and exact marginal preservation by the null.

Report availability and the number of focal states with a NONZERO exact integer
edge-excess numerator. For an edge-only correction the fraction of such states
is an outcome-blind upper bound on possible first-exact gain: if fewer than87
of2900focals can change, a3pp improvement is impossible and this route stops
before fitting. This bound is only necessary, never evidence of attainable gain.
No target file, new prediction comparison, fitted model or accuracy is read in
this stage. Existing native M2 inference artifacts can be reused later unchanged.

CPU1, startup RAM>=3.25GiB/live>=0.75GiB, no concurrent source benchmark/capture
or League, hidden durable source job,30minute worker/45minute launcher. No GPU
is needed for extraction. Bound memory to one raw game plus compact selected
metadata; preserve failures and every consumed version. No edits to frozen
apps/models/data/index and no sealed/confirmation payload reads.
