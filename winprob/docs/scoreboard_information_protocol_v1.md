# Recover omitted public scoreboard information

2026-10-01. Previous multiset experiment completed and failed its first-choice
gate. This stage investigates an input-information hypothesis, not another
objective/architecture tweak or a rescue of that failed recipe.

## Evidence and scope

Read the frozen source paths: `footwear_reconstruction._board_snapshot` stores
kills/deaths/assists for every participant from `start_kda`; `baseline._others`
drops those fields, `train_prefix.ShopDataset` does not consume other-player
KDA, and the current live adapter also omits it. Own PublicFeatureCompiler uses
focal KDA but no other-player KDA. Champion identities, inventory, levels and
item-based damage traits ARE already present; do not call those missing.

[Riot's Live Client API documentation](https://developer.riotgames.com/docs/lol?from=20423&from_column=20423)
exposes player scores, including kills, deaths and assists. The existing live
adapter already obtains the complete snapshot. No new collection is necessary
to test the historical signal. Live schema documentation establishes availability,
not parity with every local client capture or proof that this improves M2.

Exact time since death is EXCLUDED. Source no-buy examples are synthesized at
death+15,000ms after a future90-second no-purchase qualification window, using
the earlier death snapshot. A recency feature could reveal this sampling rule.
Use only the cumulative scores of the EXISTING stored board snapshot: purchases
use their start snapshot, no-buy uses its already frozen death snapshot. No new
observation time, participant frame, reconstructed visit, timestamp feature,
future window or actual purchase label is added to the inputs.

## Fixed source-owned export

Read only the 4,001 original FIT games already in the immutable IQL cache,
covering all392,448 existing states in their original order. Source-owned
read-only indexed queries go to the two qualified frozen native corpora, with
an explicit allowed-ID set checked against current tests/validation/confirmation
reservations before SQL. Require an index search, no whole-table scan.

Match each game's full indexed game/event digest to the previous certified FIT
trajectory export. Match every state by game, participant, source event index,
timestamp and episode; do not select another event or player. Read no raw
archive and run no new reconstruction. The indexed payloads necessarily contain
purchase fields, and the original identity hash includes outcome metadata;
these are deserialized/hashed solely for exact source identity. They are not
used or analyzed as feature values or training targets in this export.

Extract only integer K/D/A counts from the ten-player board, ordered exactly
like the own cache: buyer first, four allies sorted by champion ID, five enemies
sorted by champion ID. Verify all champion indices and the buyer's three scores
against the existing compiled public globals, with zero discrepancies. Preserve
the raw buyer scores for this audit; a later information experiment must zero
that focal channel because focal KDA is already available to the baseline.

No inferred roles, player names/identifiers as features, total gold, winner,
post-visit inventory, purchase counts, kill timing, damage breakdown, positions
or hidden enemy stats. Mutation tests must show that changing target/outcome/
role/economic fields cannot change the extracted scoreboard array. Missing or
invalid fields fail publication; never silently drop hard rows or invent zeros.

Write a versioned source artifact with player_kda.npy [392448,10,3] int32,
participant_ids.npy [392448,10] uint8, identity/coverage receipts, input/code
digests and manifest. Complete coverage, row identity, original indexed hashes,
roster mapping, focal-score parity, source immutability and reservation gates
must all pass before any consumer can admit it for a later experiment.

This stage measures availability/coverage only. No accuracy, outcome benefit,
new model or M2 recommendation is inferred. A later fixed comparison should
test real KDA against matched zeros and within-team shuffled assignments,
using separate whole-game folds and actual purchase exactness. It must also
beat the stronger frozen internal reference before spending M2 development
evaluation. Do not choose thresholds or recipients from evaluation labels.

## Resources and preservation

One CPU thread, streaming one game at a time, SQLite query_only plus immutable
read mode, small SQLite cache, startup hostRAM>=2.5GiB/live>=0.75GiB. Hidden
durable one-shot job,20-minute worker/30-minute launcher bound. No GPU or
competing benchmark/capture. No source runtime/model/database edits, policy
promotion, new API collection, raw reads, sealed payloads or index edits.
New source adapter code and versioned outputs are the only source additions.
