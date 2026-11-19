# Magical Footwear recovery

## Completed normal timer/slot recovery measurement

After the owner requested the less restrictive ordinary path, an isolated
event-driven replay was tested over all 11,555 selected 16.17 footwear games.
It schedules eligibility from earlier takedowns, grants when physical space
permits, and leaves full inventories pending. The distinct 2422 component keeps
its correct upgrade, sale and undo economics. A separate conditional correction
handles reviewed elixirs bought with six occupied regular slots, preserving
their paid gold and avoiding a nonexistent seventh carried item.

9,706 games pass the recovery source-consistency checks: 9,048 from the training
cohort and 658 from development validation. They contribute 1,078,515 and 80,230
candidate rows respectively after the narrow boundary filter. The game sets are
disjoint from the 1,013-game no-footwear pilot, so prospective combined coverage
is 10,719 games. This is not an already-published training dataset.

The boundary filter removes 4,567 of 1,163,312 model rows in the passing games
(0.393%). Thus the normal path retains most decisions in those games; it does
not wait until every wearer upgrades their boots.

Of 9,863 replayed games, 9,778 pass the independent boot-specific checks. A
further whole-game screen retains the 9,706 above. Another 1,692 source games
were excluded before replay for existing Viego/opaque-undo/explicit-grant
handler limitations; 157 replayed games remain unresolved. Reasons overlap.

Final evidence and exact IDs:
`data/acceptance/footwear_repair_20260911/recovery_summary.json` and its bound
`timer_full_source_parallel.json`. Four disjoint workers verified every selected
game exactly once. No final-test examples were read, no existing generation was
replaced and no model was trained on these recovered candidates.

The one-second grant-boundary filter is a declared sensitivity choice, not a
verified upper bound on server delivery. All replayed games receive endpoint,
transaction, intermediate inventory and final-outcome invariance checks. The
full run does not repeat every sample's future-prefix perturbation; those remain
covered by the fixed sample and synthetic regression tests. Wider admission
requires explicit footwear fields in the actual pipeline and a new qualification
and export generation.

## Earlier conservative prototype

The first 16.17 experiment excluded 11,555 selected source games containing
rune 8304 because its unobserved boots were missing from reconstructed inventory.
The following prototype aims to recover reliable decision intervals. It has not
changed an accepted corpus, dataset generation, live policy or trained model.

## What the sources establish

Pinned Riot 16.17.1 Data Dragon describes eligibility at 12 minutes, reduced by
45 seconds per takedown, and prohibits buying boots before the grant. Item 2422
is not purchasable; it has a 300-gold component value and a 90-gold resale value.
Recipes name ordinary boots 1001, which sell for 210. Preserve the physical
2422 identity through pricing, consumption and undo. Primary payloads, URLs and
hashes are in `data/acceptance/footwear_repair_20260911/sources/`.

A bounded raw audit covered 100 non-test 16.17 games and 201 rune users.
None had an `ITEM_PURCHASED 2422` grant event. The postgame rune counters fit a
reduced deadline rather than a reliable delivery timestamp. In 38 cases a
takedown occurs after that displayed deadline, so using it as an earlier grant
would leak the future. Four additional cases have traces consistent with delayed
delivery from a full inventory. One includes purchases 58 and 125 milliseconds
after eligibility. No source-backed universal grant-tick bound was established.
See `raw_footwear_audit.json` and `raw_findings.json` in the same directory.

## Conservative state model

Only fixed rune selections and earlier timeline events determine these states:

1. **Before eligibility:** no free boots have been established. The current
   inventory remains usable, and the selected rune blocks boot purchases.
2. **Eligible, grant unobserved:** delivery and occupancy are uncertain. Omit
   any decision whose ten-player input board includes an unresolved wearer.
3. **Resolved ownership:** an observed 2422 sale, or its destruction paired with
   a compatible boot upgrade, establishes the missing component at that
   transaction. Account for the real 300-gold recipe credit or 90-gold sale.
   Only strictly later decisions can use the resulting resolved inventory.

The resolving transaction must never supply its own earlier decision input.
Omit its entire shop visit, including a visit beginning before eligibility and
continuing through that receipt. Maintain distinct physical 2422 components so
an undo restores the same boot and the correct paid amount. A grant adds no
purchase label, shop visit, earned gold or fabricated expenditure.

This does not require guessing a delivery timestamp or inferring a slot opening
from a later frame. Games without a receipt can still contribute trustworthy
early decisions; their later board states remain unresolved. Viego, opaque undo
and other unsupported grants retain their independent exclusions.

## Implemented prototype and evidence

- `scripts/magical_footwear.py`: strict-prefix state classification and early
  boot-purchase blocking; 19 synthetic tests passed.
- `scripts/footwear_economics.py`: distinct 2422 context, compatible recipes,
  physical component consumption, sale value and equivalent affordability
  features; 12 tests passed on pinned 16.17.1 data.
- `scripts/replay_footwear_windows.py`: isolated replay using a copied
  reconstructor under `data/experiments/footwear-prototype-v1/`. Its receipt hook
  materializes the missing component only for current transaction accounting;
  all affected decision windows are omitted.

The first 20 sampled games include two with separate Viego/opaque-undo blockers.
Among the remainder, 17 contribute partially screened windows: 738 training-cohort candidate
rows and 501 validation-cohort candidate rows. One additional game remains
withheld because a truncated label window makes one prefix check inconclusive.
These are candidate rows, not admitted training examples. The screen checks
retained inventory/label constraints, known endpoint inventories, invariance to
changed postgame outcome fields and representative future-prefix mutations.
It is not a replacement for the full semantic audit.

Authoritative first replay evidence:
`data/acceptance/footwear_repair_20260911/receipt_replay_prototype_final.json`.

Independent review reproduced two boundary failures and the driver now rejects
both: a receipt tied with any shopper's decision timestamp, and a receipt whose
destruction and purchase are split across separate contiguous source batches.
The latter can otherwise yield a correct endpoint but a 300-gold pricing error.
`scripts/test_footwear_receipt_boundaries.py` exercises these gates, including
death-snapshot anchors. Contradictory histories are distinct from unresolved
delivery windows and fail the prototype screen. The copied reconstructor is
not safe to use without its driver's exclusion gates.

Before broader training, integrate the explicit footwear fields into both
policy paths and live handling, audit every retained decision, bind the
decision-exclusion masks and code version, and publish a new immutable
generation preserving the declared train/validation membership. Do not compare
an expanded export's headline against the smaller pilot as a model improvement;
compare candidates on the same expanded validation population.
