# A small residual IPO experiment on revealed purchase revisions

Design only: no real training, probability export, preference measurement or
raw-data read was performed for this document. It uses the strict, outcome-
blind revision labels specified in `undo_pairs_protocol_v1.md`. Retained B is
a revealed revision, not a causal win-effect label. Same-A roundtrips remain
controls and never enter the preference loss.

## The exact M2 reference exists, but its probability export does not yet

Read-only package/API inspection verified `mixture-v4-m2-v1`, identity
`ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9`.
Its catalogue contains **9,333 canonical main-purchase groups**, including
**196 single-item groups**. Actions are multisets; a group is not its largest
item. The manifest sets `history:false`, so this package does not require an
active prefix-history feature. It does require its complete board/inventory,
prequential gold/version, rune/summoner, role, objective and footwear/slot
inputs. The undo counts artifact does not contain enough information to
fabricate those rows.

Concrete source API, verified in `build_tracker/app/mixture_policy.py`:

```text
activate_runtime(package_directory, pinned_16_18_static_review)  # FIRST
load_package('mixture-v4-m2-v1', device='cpu')
policy.probabilities(audited_rows)  # [batch, 9333], normalized mixture masses
policy.action_of(group_id)         # exact ward-stripped multiset
policy.options(audited_rows)       # native legal displayed recommendations
```

`probabilities` precedes the native legality filter. Existing displayed-policy
and frozen-generation CLI reports persist options/exactness, not the full
probability tensor. No usable real M2 probability artifact is assumed here.
A new **source-owned exporter** must run inside build_tracker's own process
and publish a versioned artifact. Winprob consumes that artifact; it neither
imports source-project inference code nor reimplements M2.

Source producer requirements are concrete: query only explicitly admitted
IDs through indexed source lookups, after the current global-test/confirmation
denylist. Do not use source iterators that read all game payloads before
filtering. Reconstruct an explicit pre-A snapshot using the source game's
qualified patch runtime; do not silently substitute an existing net-visit
or pre-B row. Serialize it, then score it in a **separate clean process**
with M2's own pinned 16.18 economics. Missing audited inputs are an exclusion,
never padding disguised as a valid complete state.

The reference export must bind: source IDs/roles and membership audit;
pair/context keys and exact timestamps; canonical context JSON digests;
producer/runtime/static/model/catalogue digests; full group probability
vectors, exact action mapping and native displayed options; unknown-item,
unmapped-group, numeric-zero and context-exclusion counts. Keep the physical
pair certificate at B time separate from the pre-A feature cutoff. Do not
replace `gold_est` with a label-derived refund or purchase price to make a
candidate pass the native estimated-wallet legality check.

## A development split genuinely unseen by both compared learners

Own-WP development is not automatically held out for M2. The package manifest
records 6,354 source training games, and the metadata audit found overlap
with own-WP calibration/development. Those 16.18 games cannot support an
end-to-end held-out-M2 claim, although they may support the adapter's training
or clearly labelled exposed diagnostics.

The primary challenge is the admitted **16.19 own-development population**
not used to train M2: 859 IDs are recorded in the broader metadata population,
303 occur in the current clean pilot. Freeze the exact eligible ID list and
verify its intersections with M2 fit, IPO fit, calibration, global tests and
own confirmation before any pair prediction. These counts describe source
metadata, not the number of usable revision pairs. All 1,911 own-confirmation
IDs remain closed. The existing 16.19 benchmark may be a separately declared
exposed-evaluation extension later; it is not silently pooled into a fresh
challenge. No macro historical exposure is read to enlarge the cohort.

Use earlier permitted fit games for both fitted arms, and calibration games
only for diagnostics. No candidate or hyperparameter is selected on the
16.19 challenge. Before fitting require at least 100 fit games with exact
M2-mappable strict revisions, 50 challenge games and 100 challenge pairs after
all mapping/context exclusions. If these minima fail, stop this recipe as
underpowered; do not replace M2 with a weaker classifier or call a 16.18
reference-exposed cohort clean to produce a positive result.

## One small model, two losses, one frozen reference

Use the full exported canonical group space, preserving reference zeros.
Define a shared residual action score `f_W(x,a) = c(x)^T W v(a)` with at most
32 standardized start-context coordinates and 24 fixed action coordinates.
The context vector contains an intercept and a frozen allowlist of pre-A
numeric state/actor features; means/scales are fitted on fit only. Action
coordinates are fixed, patch-bound recipe-independent item attributes/tags
aggregated by multiset counts, plus basket-size/SAVE indicators. They cannot
contain retained/rejected position, undo/refund metadata, future information
or pair identity. The exact feature list/scaling must be locked before fit;
this is at most 768 parameters, not a new board transformer.

The policy is `pi_W(a|x) proportional to pi_M2(a|x) * exp(f_W(x,a))`.
For a pair, `h = f_W(x,B)-f_W(x,A)` is exactly the change in log odds relative
to M2; the normalization cancels. Fix tau=.5 and lambda=.01:

```text
IPO: normalized weighted mean[tau * (h - 1/(2*tau))**2]
     + lambda * ||W||_F**2.
```

Flatten W and set each design row to `c(x) outer (v(B)-v(A))`. The objective
is weighted ridge regression with a finite target, so solve its linear system
directly and verify the gradient residual. This makes the IPO arm cheap and
deterministic. It does not make noisy preferences correct. The finite target
follows [Azar et al.](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html);
the bilinear adapter and revision data are this project's adaptation.

The **winner-only imitation** arm uses identical W, features, zero init,
weights and lambda, minimizing `-log pi_W(B|x)` over **all 9,333 groups**.
It must not receive A as an input or restrict normalization to the observed
pair: a two-action cross-entropy would already use the rejected negative and
would not be a winner-only control. Use a bounded convex optimizer, at most
250 iterations, with convergence diagnostics. Stop on failure to converge
within the declared CPU/resource budget rather than compare an unfinished
control with the closed-form arm. Freeze M2 itself as the third arm.

Normalize the existing `1/(10 * eligible-pairs-for-player-game)` weights over
fit observations. This integrates the planned uniform-focal-player choice;
all uncertainty remains clustered by game. Train both arms on the identical
rows and labels. No DPO arm, hyperparameter search, full-trunk fitting or
outcome-weighted loss is added to this first bounded comparison.

## Mandatory zero-residual end-to-end identity

Before fitting, run the actual export-to-adapter-to-decoder path with W=0.
It must reproduce the exact source group mapping, support zeros, pair odds,
rank order and native displayed options on the identical contexts. Require
zero identity/action mismatches. The W=0 adapter must return the source's
exact float32 probability array unchanged, preserving dtype and bytes. Use
an explicit zero-residual fast path: even a nominally harmless log/softmax or
renormalization can alter finite-precision ties. The all-row byte-identity
check composes with the source's native decoder parity on that exact array.
No observed-pair conditioning, new legal mask or altered wallet may hide a
failure. A synthetic algebra test alone does not satisfy this real-data gate.

Mutation tests must show that changing target labels, undo/refund metadata or
future events cannot change context features or exported M2 masses. Scoring
A and B must use the same canonical context. The physical certificate does
not establish both actions were affordable at the earlier pre-A instant, nor
that M2's estimated-wallet decoder accepts them. Report that distinction and
any decoder disagreement; the experiment's primary endpoint is canonical
finite-action preference ranking, not a served-policy win claim.

## Held-out metric and honest success/stop rule

Evaluate **total** log odds: `log pi_M2(B)-log pi_M2(A)+h`. Counting a positive
residual h as a correct prediction would falsely call a correction successful
when M2's stronger prior still ranks A above B. Accuracy awards 1 for B>A,
0 for B<A and .5 for ties. Use the declared focal-player weights and 2,000
paired whole-game bootstrap resamples, seed 20260929. Report reference support,
mapping/context coverage, pair/game counts, price-direction slices, patch,
same-item control handling and correction size/KL drift.

Proposed useful-preference gate: IPO improves weighted pair accuracy by at
least **5 percentage points over both frozen M2 and winner-only imitation**,
with a paired 95% interval above zero in each contrast. Both comparisons are
required; do not select the weaker baseline after observing results. Compare
cheap price and champion/patch-frequency heuristics too, with frequency fit
on permitted training data, and disclose if either beats the learned models.
Same-item controls must never be scored as directional preferences. Report
every arm even on failure; no rerun on the same challenge is called confirmation.

A pass would show improvement in predicting audited high-elo revisions in
this selected challenge. It would justify a subsequent independently
confirmed contrastive-imitation experiment. It would not establish causal
item value, superior play, higher win rate or broad coaching benefit, and
does not by itself complete the overarching goal.
