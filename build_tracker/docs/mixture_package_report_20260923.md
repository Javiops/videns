# Mixture package report — equivalence, legality, corpus independence, latency

2026-09-23, 00:00-01:00. Package `mixture-50-50-v1`, identity
**`56e80a576b2cedce9e953112307016b977c47854c09535667ca89ece8a949761`**
(sha256 of the canonical manifest, which binds every component by its own
sha256). Provenance status: **`experimental_reference`** — unchanged, and
deliberately so.

**Scope, stated first.** The 58,781 decisions are the **internal
chronological TRAIN probe**, already exposed and already used for model
selection. This pass verifies PACKAGING, EQUIVALENCE and LEGALITY in that
population. **It does not replace the VAL displayed-policy report that a
future promotion would require**, and it is not a promotion artifact:
`promote_served_model.py` expects a single-artifact schema and cannot
consume this report.

---

## 1. Equivalence — 1 differing row in 58,781, fully explained

The package reproduces the frozen mixture's persisted rows **while computing
its economic features from scratch**, touching no experiment cache.

| | |
|---|---|
| rows compared | 58,781 |
| identity mismatches | **0** |
| first-option mismatches | **1** |

The single difference is row 23267 — Ornn, `EUW1_7982410557` @ 442600 ms,
wallet 486 gold. The package's first legal option is `3076`; the frozen
reference's is `3082`.

**Cause, measured rather than assumed.** Those two actions are separated by
**1.21 × 10⁻⁵ in absolute probability (7.1 × 10⁻⁵ relative)**: `3076`
P=0.17131269 vs `3082` P=0.17130059. The experiment cache stores the
economic features in **`float16`**; the package computes them at full
precision. A tie that close flips on that quantization alone. (The
top-scoring group at that decision, `3047` with P=0.4486, is illegal at 486
gold and the decode correctly skipped it.)

So the package is not bit-identical to the frozen reference, and the reason
is that the reference is the **quantized** one.

**And the package loses that decision.** The true purchase was `3082`, so
the flip costs one hit: **49.5126% vs 49.5143%** over the 58,781 probe
decisions. Confirmed by the reviewer with a controlled float32 -> float16 ->
float32 round-trip on that single row, which reproduces the change and
reverts it (`data/reports/mixture_package_external_review_20260923/
single_row_precision_check.json`). Higher precision is not automatically
better here; it is simply different, and on this one decision it is worse.

Architecture verified on the BUILT models, not on the request:
`{d_model 64, layers 2, ff_dim 128, heads 4}` — layer count, real
`self_attn.num_heads`, `linear1.out_features` and embedding dimension all
re-inspected. Runtime activated from a clean process, its declaration
matched against the manifest, and the live economic symbols verified **by
object identity** (`baseline.inventory_state is footwear_economics.
inventory_state`), because a file digest says a file is unchanged, not that
the process is using it.

---

## 2. Legality — every displayed option, zero unaffordable

All three options are persisted per decision
(`data/reports/mixture_displayed_policy/rows_mixture-50-50-v1_probe.jsonl`)
and each was re-checked independently against `purchase_sequence` at the
decision's declared wallet.

| | count |
|---|---|
| options checked | 171,539 |
| legal purchases | 154,229 |
| SAVE (always legal) | 17,310 |
| **unaffordable** | **0** |

By position: 52,631 / 50,554 / 51,044 legal at positions 1 / 2 / 3.

One wallet throughout: `row["gold_est"]`, used by the state, the frozen
domain, the economic features, the blend and the legality filter alike. The
adapter no longer accepts a budget override — one would have reached only
the filter, leaving the scorer pricing against a different wallet.

Note what this budget is: the **offline estimate**, not an exact live
wallet.

---

## 3. Corpus independence — three deliberate accesses refused, zero during scoring

Serving must answer from the package alone. This is enforced and measured,
not asserted: a guard denies every read under `data/dataset_generations`,
`data/experiments` and `data/corpora`, and records the attempts.

**The guard is self-tested first**, because "0 blocked reads" means nothing
if the guard cannot fire — a broken guard and a clean package look identical
from outside. It deliberately attempts `visits_train.jsonl`,
`econ_fit_full.pt` and a `torch.load` of the latter, and requires all three
to be refused.

> **Three deliberate forbidden accesses refused, and zero forbidden
> accesses during scoring.**

With the guard active the package loaded and scored 64 decisions, every
first option legal.

The earlier `assert_reads_no_train()` — which returned `True` without
checking anything — was deleted. A method that cannot fail reads like
evidence and is worse than no check at all.

---

## 4. Latency — the adapter of this digest, cold separated from warm

The earlier figure measured the experimental circuits assembled inline.
That answers "what does the mixture add over a single model"; it does not
answer "what does serving this package cost". Both are reported, labelled
for what they are.

### 4a. The package adapter (`app/mixture_policy.py`), this digest

| | median | mean | p90 | p99 |
|---|---|---|---|---|
| **cold load** (per process) | 70.6 ms | 101.2 | 80.9 | — |
| **warm request** (per decision) | **35.3 ms** | 37.6 | 49.0 | 65.7 |

Cold load is digest verification of every component, vocabularies,
catalogues, tensors, three `state_dicts`, architecture re-inspection and
basket-vector precomputation. **Paid once per process, never per request.**
n=5 cold loads (first 248 ms, then 51-81 ms), 400 warm requests after 20
warm-ups, batch size 1.

### 4b. Circuit comparison A vs M (the other question)

| | median | mean | p90 | p99 |
|---|---|---|---|---|
| circuit A (single model) | 40.18 ms | 48.30 | 50.39 | 491.23 |
| circuit M (mixture) | 44.33 ms | 49.09 | 53.94 | 79.77 |
| **paired delta M−A** | **+3.70 ms** | +0.79 | +9.42 | +30.04 |

**Mixture overhead: +10.3% median** — stable against the +11.0% measured
earlier, although the absolute figures are much lower now (40/44 ms vs
75/84 ms), because that earlier run was taken on a loaded machine.

### 4c. The tail — the previous hypothesis does NOT reproduce

Per-decision records now exist and were verified before the process closed:
840 rows (40 warm-up + 800 measured), sha256 recorded, and the published
aggregates **recomputed from the file** and compared field by field.

With those records, the earlier "the circuit measured second carries the
tail" pattern **does not hold**:

- p95 threshold 56.4 ms, 41 slow measurements: **20 in position 1, 21 in
  position 2** — an even split.
- The 11 episodes above 400 ms: circuit A 7, circuit M 4; position 1 four,
  position 2 seven; and **never both legs of the same pair**.

So the episodes are sporadic, roughly evenly spread across circuit and
order, ~1.4% of measurements. **The cause remains unidentified**, but it is
not a property of the mixture and not a property of measurement order. The
earlier pattern was a coincidence of that run, which is exactly what could
not be checked when the records were never written.

---

## What this does and does not establish

**Established**: the package is self-contained and reproduces the frozen
mixture up to one float16-induced tie; every displayed option is legal at
the declared wallet with zero unaffordable; the package needs neither the
training corpus nor the experiment caches; serving it costs ~35 ms per
request warm, with a ~71 ms one-off load, and the mixture's overhead over a
single model is ~10%.

**Not established**: anything about VAL under this package; any promotion
eligibility. The three checkpoints still carry no provenance and no config
of their own, `manifest_sha256`, `export_fingerprint` and `eval_version`
remain `not_established`, and the selection was probe-internal rather than
validation. `experimental_reference` stands.

**Two limits on the claims above, recorded for promotion.** Both narrow what
sections 3 and 4a actually demonstrate:

1. **The corpus guard covers reload and scoring with the runtime ALREADY
   initialised.** Process startup still reads the static review under
   `data/corpora`. So "independent of the corpus" holds for the serving
   loop, not yet for a cold process start.
2. **The 70.6 ms cold figure is repeated construction inside one process**,
   not a full new-process start: the interpreter, torch and the pinned
   runtime are already warm. A real process start is larger and is not
   measured here. The 35.3 ms warm request, over 400 FIT decisions, does
   correspond to the scope claimed for it.

**Correction (reviewer, 2026-09-23)**: of the 13,668 probe purchases excluded from the "legal" denominator, **13,209 are budget-blocked and 459 structurally blocked** — not all are gold. The 63.38% / 85.73% figures describe the 41,579 main purchases feasible at the ESTIMATED wallet, not live performance with an exact wallet.

**Still open**: the ~500 ms episodes — **cause unidentified; the earlier
order pattern did not reproduce** — and the VAL displayed-policy report.
