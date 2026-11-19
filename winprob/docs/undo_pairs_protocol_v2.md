# Expanded, M2-untrained revision challenge — counts protocol v2

This version changes source scope only. It reuses the frozen
`undo_pairs_support.analyze_game` certificate, controls, feature cutoff and
weights from `undo_pairs_protocol_v1.md` without modifying either v1 file.
The source is the outcome-free `winprob-revision-context-v1` cache: **5,861
games**, consisting of **4,002 fit**, **1,000 calibration**, and **859
development games, all development on patch 16.19**. This is not another run
claimed to use the original 6,000-game population.

The source producer preserves 5,305 existing contexts and adds the remaining
556 admitted 16.19 development games. The 695 original 16.18 development games
are absent from the primary challenge because M2 had trained on that source
population. All original own-confirmation and global-test reservations remain
closed. Admission and current reservation verification run before any raw
read, during batches and at completion. Every raw path and compressed digest
must be bound in this new cache's source manifest.

The consumer requires the new schema explicitly and refuses an outcome array,
an incorrect role count, a non-16.19 development game, duplicate source
identity, a quarantine ancestor or a cache/source/admission digest mismatch.
It does not access match outcomes, cache y, or another source ID. As in v1,
raw DTO deserialization contains outcome-bearing fields but the analyzer does
not extract those fields. No winner field is added to the output.

All ten players are counted. The exact v1 eligibility remains: isolated
A-purchase / explicit matched positive-full-price undo / B-purchase;
strict timestamps within ten seconds; no recipe destruction, extra focal
transaction, material recorded context change or retention censoring; primary
equipment; no nested recipe pair; and a matched cached feature frame strictly
before A. Same-A roundtrip controls remain controls. Physical feasibility is
conditional and applies at B time, not retroactively at A. No threshold,
certificate, item family or label is relaxed to enlarge this challenge.

Outputs are `pairs.jsonl`, `report.json`, binding/access ledgers and digests.
Every row preserves v1 fields and adds a stable `pair_id`:
`match_id|participant_id|A_time|undo_time|B_time|A_item|B_item`.
The game-local candidate index and output line number support exact source
producer alignment; neither becomes a model feature. Eligible pair weights
remain `1/(10 * eligible revision count for the player in that game)`.
Use whole-game uncertainty in later experiments.

The counts gate remains at least 100 fit games and 50 development games with
strict revisions, plus 100 development pairs. Canonical M2 action mapping and
pre-A-context qualification may reduce those counts later, so this is only a
size gate. It authorizes neither a benefit claim nor fitting unqualified
rows. No pair scores, win rates, parameter selection or real training occurs
in this extraction.

Run through the root-reviewed durable launcher, one CPU thread, BelowNormal,
with resource/deadline checks. V1 artifacts are preserved. This source
expansion was chosen to complete the predetermined M2-untrained 16.19
development cohort, not because any pair-prediction outcome was favourable.
