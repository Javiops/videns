# Mixture promotion prework — what the path requires, and what is missing

2026-09-22 night. Owner approved preparing the integration and the
displayed-policy report while the history pilot trains. **Nothing here
promotes anything**, and no file the pilot uses was modified.

## What passes already

**Every displayed FIRST option is legal at its own budget.** Re-checked on
CPU over the mixture's 191,916 persisted VAL rows by re-running
`purchase_sequence`: 171,593 purchase recommendations, **0 illegal**,
20,323 SAVE. Report:
`data/reports/mixture_promotion_prework_20260922/displayed_first_option_legality.json`.

Partial by construction: options 2 and 3 are not persisted in the row
files, so the full report must re-decode and check all three. That run
needs the GPU and is reserved until training finishes.

Note the budget: this is `gold_est`, the offline estimate. The live path
scores one exact-wallet snapshot, and the repo already records that
over-budget is measured against offline estimates, not exact live wallets.

## CORRECTION (2026-09-22 23:00) — three of my claims here were wrong

The reviewer checked this document against the workspace. Three things it
asserted do not hold, and one deliverable it described does not exist.

1. **"Every artifact in the basket line carries `probe_only`" — FALSE.**
   Only `arm_a_full.pt` does. The three checkpoints the mixture actually
   uses — `Bf_ep16`, `econ_ft_ep8`, `wardfree_ep16` — are raw
   `save_checkpoint` blobs holding `state_dict`, `optimizer`, `scheduler`,
   `epoch` and RNG state, with **no `provenance` and no `config` at all**.
   They are not *flagged* ineligible; they are *incomplete* for service,
   which is a different and more tractable problem.
2. **"A hereditary prohibition on the architecture" — FALSE.**
   `app/deployment.py` rejects an artifact whose OWN
   `provenance.status` is `probe_only` or `legacy_unverified`. It says
   nothing about a recipe developed through experiments. Producing a new,
   properly provenanced candidate with this recipe is not forbidden. This
   is emphatically **not** licence to strip labels or invent provenance.
3. **"Distillation resolves the latency for free" — WITHDRAWN.** A student
   may keep part of the economic circuit, or lose quality by dropping it.
   Its saving must be measured, and it does not explain the observed tail.
   Also: `app/predictor.py` builds `tp.PrefixModel`, not `BasketNet`, so a
   distilled basket `d64` would not drop into serving either — it would
   need its own adapter, and returning to the earlier architecture would be
   a separate contrast.
4. **`scripts/mixture_displayed_policy.py` did not exist.** This document
   described it as written and reserved. It was not written. That was my
   error, not a path problem — the file was nowhere in the workspace. It is
   being built now, as part of the package work below.

## What actually blocks promotion, restated

- The three mixture checkpoints carry **no provenance and no config**.
  `REQUIRED_PROVENANCE` needs six keys (`manifest_sha256`,
  `export_fingerprint`, `eval_version`, `gold_input`, `training_split`,
  `selection_split`); these blobs carry none of them.
- The deployment contract is **single-artifact**: one `prefix_model.pt`,
  one sha256, one `Predictor`. The mixture is a configuration — two
  scoring models, a frozen-parent domain, normalization statistics, a
  canonical-key permutation and the ward-grouped decoder.
- The report schema is artifact-shaped (`artifact_sha256.A`,
  `metrics.A.unaffordable`).
- Everything the decoder needs is currently **rebuilt from TRAIN** at
  startup (`first_pass` over `visits_train.jsonl`). Serving must not do
  that.

## Decision taken by the owner

**Extend the contract with a versioned mixture package.** Distillation
becomes a later experiment, not a prerequisite for moving toward service:
the mixture has favourable evidence, a student adds a new quality
uncertainty, and it would not remove the integration work anyway.

Work, in the owner's terms:

1. A new contract for this mixture, **leaving the previous path intact**.
   The package holds weights, configs, catalogues, correspondences,
   normalization and the decoder, plus the economic engine versions.
   **Service must never read TRAIN to reconstruct any of it.**
2. A single identity: the digest of the canonical manifest binds every
   component. That digest is the report's candidate "A" — several models do
   not prevent identifying one evaluated policy.
3. **One implementation shared between evaluation and serving.** The
   adapter must produce exactly the options the harness produces, keep the
   legality checks, and score only a shop session the player started
   explicitly.
4. A formal candidate with **demonstrable** provenance: real data, training
   subset, recipe and selection recorded. The current checkpoints are
   experimental references — packaging them does not make them eligible.
   Any final run required must be defined before it is launched.
5. Evaluate **the exact package intended for service**, persisting all
   three options and checking the legality of each.

Promotion of that specific digest is a decision that comes after, not part
of this work.

## Reserved for after training

The latency benchmark must be re-run — per-decision timings were never
written for the existing run, so the tail cannot be analysed from what we
have. On that re-run, **verify the raw timings and their timestamps are
actually flushed to disk before the process exits**, rather than assuming
the instrumentation ran. That is the same class of error as (4) above.
