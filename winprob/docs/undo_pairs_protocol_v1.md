# Purchase reversals as revealed revision pairs — counts protocol v1

This is an outcome-blind support experiment on the same clean 6,000 source-
bound games. It is not a win-effect estimator. No confirmation or global test
IDs are admitted. The first run counts sequences before anyone chooses a
training loss or looks for favourable win-rate contrasts.

## Signal and its exact meaning

A player buys A, undoes that purchase, and buys B during one short shop visit.
When the undo restores the prior inventory and refund, both purchases have
been demonstrated executable from that inventory. Final B is a **revealed
revision**, not proof A was wrong, B causes wins, or the initial click was an
intentional recommendation. Accidental clicks, changing plans and information
from teammates remain label noise. A -> undo -> A is an explicit no-preference
roundtrip control, not a negative-quality label for A.

Local reconstruction code establishes the event interpretation: `beforeId`
identifies the undone purchased item; `afterId` may name a returned component
but does not describe every silently restored component. `goldGain` is the
signed transaction delta, and can be negative for an undone sale. Existing
code uses the matched purchase's component stack to restore combined items.
V1 therefore does not infer a full inventory from `afterId`.

## Strict primary certificate

Inspect all ten participants, without selecting by outcome, role or champion.
Group each player's purchases/sales/undos into visits with a 10-second idle
boundary. Record all purchase-undo-purchase patterns, and a reason for every
excluded pattern. A primary certificate requires:

- The entire visit has exactly A purchase, undo, B purchase, in that order;
  timestamps are strictly increasing and A-to-B is at most 10 seconds.
- Undo `beforeId` equals A, `afterId` is zero, and finite positive `goldGain`
  equals A's full patch-pinned static total price. Both items are purchasable
  Summoner's Rift equipment with positive prices; consumables, trinkets,
  support-gold/special-recipe items and explicit champion restrictions are
  excluded from the primary population.
- There is no focal ITEM_DESTROYED, sale, extra purchase or extra undo from
  A's timestamp through visit closure (10 seconds after the final shop event).
  This deliberately excludes combined purchases and free transformations
  rather than inventing their inventory restoration. Partial refunds and
  nonzero afterId remain visible as excluded candidates.
- No recorded champion kill, building/epic-monster kill, dragon-soul event,
  level-up, skill-level-up or game-end event occurs between A and B. This is a
  recorded-context guard, not proof that all information or champion state
  was identical. Nonfocal item changes are counted and separately flagged.
- B is not undone, sold or destroyed before visit closure; the source covers
  that closure. “Retained” means through this window, not forever.
- There is a clean cached feature frame strictly before A, at least one
  second earlier. No frame from between A and the undo can enter X. Recipes
  nested within each other (component versus its upgrade) are counted but
  excluded from the primary nontrivial revision challenge.

The certificate uses transaction-inverse semantics and the absence of
intervening inventory changes; it does not claim a complete reconstruction of
the ten inventories. Let W be money before A, p its paid/refunded price and
g nonnegative intervening income. Successful A implies W >= p, and after a
full refund money is W+g >= p. Successful B establishes its own feasibility.
Thus both are affordable **at B time**, conditional on no unobserved gold
drain, unchanged restrictions and a faithful undo event. No wallet value is
reconstructed, and purchase/refund amounts are certificate fields, never X.

The decision timestamp is B; the common feature cutoff is before A, with
ages reported. More expensive B may only have become affordable after A.
Do not claim both were affordable at A, or that the complete game state was
identical. A separate price-dominance flag can certify earlier affordability;
do not select only cheaper-B pairs, which would make that baseline trivial.

## Population and outputs

Only the original clean cache's compressed match/timeline paths may be opened.
Verify original admission/current reservations, cache/source/static digests,
quarantine guards and source IDs before access, and repeat at completion.
Deserialize source DTOs but never access match winner/final-performance fields
or cache y. Existing pre-A cache X, immutable champion/team IDs and focal
pre-frame wallet/level/XP are allowed. A/B/undo/certificate fields are separate
from those inputs. No sibling-project runtime code is imported or changed.

Output outcome-free `pairs.jsonl`, aggregate counts by role/patch/pair class
and rejection reason, binding digests and an access ledger. Record different-
item revisions, same-item roundtrips, compound/refund-ambiguous patterns,
recipe-nested pairs, missing contexts and retention failures separately.

For future evaluation the sampling estimand is one uniformly chosen player
from each game, then one uniformly chosen eligible revision for that player.
Each eligible pair gets weight 1/(10 * eligible-pair-count-for-that-player).
Players with no eligible revision abstain; report that coverage. Retaining
all players integrates over this random focal selection rather than drawing
one noisy subsample. It is not counterfactual outcome estimation. Bootstrap
whole games, and never treat all pairs as independent observations. A second
equal-game/eligible-player weighting may be reported only as a sensitivity.

Stage-0 readiness requires at least 100 fit and 50 development **games with
strict different-item pairs**, and 100 development pairs. These are minimum
challenge-size checks, not evidence of utility or a promise of power. Keep
all source roles fixed. Do not relax the certificate after inspecting outcomes
or pair-prediction results. Same-item controls must never acquire a different-
item preference label, and ordinary adjacent purchases without an undo must
never become rejected-choice pairs.

## What would make this useful next

If support is adequate, freeze a challenge asking models to rank retained B
above reversed A using only the pre-A context and candidate action identities.
Report coverage, absolute accuracy and paired game uncertainty. Strong
comparators are frozen M2 on the same independently reconstructed inventory,
champion/patch purchase frequency fitted on ordinary training decisions, and
cheaper-item/more-expensive-item rules. Neither event order nor refund metadata
is a prediction input. The comparison must beat these controls on the fixed
development role before claiming a useful revealed-revision signal.

Only then consider a bounded contrastive-imitation experiment: these reversed
actions offer more grounded negative examples than arbitrary sampled items.
An equal-budget ordinary-imitation arm remains necessary. Such a result could
improve agreement with high-elo revisions; it would not establish causal item
value, preference in every state, or higher win rate. No OPE/DR win-effect
claim is part of this experiment.

One CPU thread, BelowNormal priority, the bound 6,000 games, and a durable
resource-gated launcher after review. Synthetic tests authorize no raw run.
