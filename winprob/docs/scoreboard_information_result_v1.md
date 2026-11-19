# Public scoreboard inputs recovered; predictive benefit not measured

2026-10-01. Read-only code inspection established that the source board retains
K/D/A for all ten players, while the current training projection and live
adapter omit it for the other nine. M2 already has champions, levels,
inventories and item-based damage traits; these are not new features.
The [official Live Client API documentation](https://developer.riotgames.com/docs/lol?from=20423&from_column=20423)
exposes player scores. Actual live-capture parity remains to be verified before
any eventual deployment.

The source-owned export recovered all 392,448 original FIT states from 4,001
games. It used only indexed read-only queries against two frozen qualified
corpora, reproducing every game's previous indexed digest. Query plans use
`idx_events_match`, with per-game ordering; no whole-table scan or raw archive
was used. Source databases remained unchanged.

All ten-player rosters match the cached champion order. The buyer's KDA exactly
matches the original public feature columns for every state: zero mismatches.
Target/outcome/role/economic field mutations leave the projection unchanged
on one example from each of the 4,001 games. The source payloads do contain
labels/outcome metadata needed for their original identity hash; these fields
were not used or analyzed as inputs or targets in this stage.

Other-player scores are nonzero in 363,576 states. Observed column maxima are
34 kills, 21 deaths and 49 assists. These are coverage facts, not evidence
that the scores provide conditional predictive information.

The own consumer independently verified admission, complete episode identities,
participant ranges and every buyer-score value. It wrote normalized features
with the focal channel explicitly zeroed, because that information already
exists in the baseline. Frozen within-team permutations preserve complete KDA
triples and change 362,823 rows. No target array was read during preparation.

Exact death age/recency is deliberately excluded: source no-buy labels use a
fixed death+15s timestamp and a future90s no-purchase qualification window.
The new inputs use the same stored snapshot as existing inputs. A later
comparison also requires a positive nonempty-main-purchase slice, so a gain
confined to this synthetic/ward-only class cannot pass as better main buys.

Artifacts:

- Source `build_tracker/data/exports/winprob-scoreboard/scoreboard-export-20261001-v1/manifest.json`,
  SHA256 fee70f9d7be28235e07b22b409471833bd4674d33a10abfc9939ae046fa2eeb2.
- Own `data/runs/scoreboard-features-20261001-v1/cache/manifest.json`,
  SHA256 c3a0f735523efac4eb040ca138f720196cb8a323194f3a29666119d9173ae786.

Source extraction took464.9s; own preparation26.0s, CPU1/no GPU. Seven source
tests and five own tests pass. No source runtime was imported into winprob,
no source database/model/serving path changed, and all seals remain closed.

`scoreboard_probe_protocol_v1.md` fixes a subsequent matched real/zero/shuffled
comparison, with both accuracy primaries and a nonempty-purchase guard. No
new model, accuracy gain or breakthrough has yet been demonstrated here.
