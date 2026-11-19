# DeepSeek project review (deepseek-flash)

Source-only review. 36 files, 501567 chars sent. No data, split or run history was included, so nothing here is a verified finding about this project's numbers.

---

I only have the source. No data, no run history, no ability to execute. Everything below is code-reading; where a claim needs a measurement, I say so instead of guessing.

## Strengths that are real

- **The provenance/deployment chain is the best thing here.** `app/deployment.py` fail-closes on missing manifest, digest mismatch, legacy/probe status, non-validation split, or nonzero unaffordable count. `scripts/promote_served_model.py` binds the served bytes to a schema-v1 validation report. `app/dataset_generations.py` is obsessive about content hashes and frozen admission evidence. Confidence: high.
- **`scripts/eval_policy.py` scores the policy the user sees, not a trainer diagnostic.** It strips labels before they reach the Predictor, feeds one budget to both the feature and decoder, resamples by game, keeps point estimates and CIs on the same population, and refuses test reads without `--confirm-final`. Confidence: high.
- **The causal-gold design is disciplined.** `app/reconstruct.py::_attach_gold_estimates` uses only pre-decision frames/events, tags estimator versions, and has a no-future unit test claimed in `scripts/test_gold_causal.py`. I cannot confirm the test passes, but the code structure is the right one. Confidence: medium-high.
- **Multiset targets are handled seriously.** `app/purchase_multiset.py`, `item-counts-v1`, count head, refusal to truncate capacity. Confidence: high.
- **Streaming and atomic artifact publication are taken seriously.** `baseline.py`/`train_prefix.py` stream; `app/dataset_artifacts.py` writes partials then renames. Confidence: high.

## Weaknesses, ranked by damage

### 1. The promotion/serving gate has holes that can serve a non-promotable artifact

**Mechanism:** `scripts/promote_served_model.py` checks schema, `split == "val"`, artifact digest, `artifact_provenance.A.status == "verified"`, `unaffordable == 0`, and `visits > 0`. It does **not** check `report["promotion_allowed"]`, `report["production_hold_cleared"]`, or `report["purpose"]`. A development report written by `eval_policy.py` with `promotion_allowed=False` can still be fed to the promoter.

**Why it matters:** This directly violates the stated hard rule that development never clears production hold. Worse: a footwear-aware development artifact can be promoted, and `app/live.py::snapshot_to_row` emits no `footwear_state_version`, `magical_footwear_selected`, or `footwear_granted`. `app/footwear_policy.validate_footwear_state` then raises when `Predictor` featurizes the live row. So the failure mode is either an unpromotable model being served, or live sessions crashing.

**Confidence:** high from code. Damage: critical.

**Also in the same hole:** `app/deployment.py::serving_eligibility` verifies the manifest fields but never re-hashes the report file or checks `evaluation_report_sha256`. The manifest is trusted after promotion. The “binding to the report” is therefore a one-time promotion-time claim, not a serve-time guarantee.

### 2. The promoter does not require the full validation set

**Mechanism:** `eval_policy.py` defaults `--games=300`. `promote_served_model.py` only checks `report["visits"] > 0` and `unaffordable == 0`. Nothing forces `games=0` or full-validation.

**Why it matters:** The exact-budget hard stop can be satisfied on a 300-game sample. The manifest then says “evaluation_split: val” as if the whole validation split passed.

**Confidence:** high. Damage: high.

### 3. Live serving has no patch guard and several regime mismatches

**Mechanism:** `app/live.py::snapshot_to_row` never reads the live game version. `app/main.py::start_live_shop_session` shapes the live row with `predictor.dragon`, i.e. the artifact’s static-data version. If the live game is on a newer patch, new items/champions are classified against old Data Dragon; unknown items are dropped by `_player_items` because `classify(...).get("skip")` is true for missing data.

**Why it matters:** On patch day the advisor can silently omit inventory, misprice recipes, or emit missing-item advice. There is no “artifact patch == live patch” refusal.

**Confidence:** high from code. Damage: high.

**Related live mismatches:**
- `app/live.py` sets `total_gold: None` and other players’ `gold: None`, so `team_gold_diff` and `opp_gold_diff` features are zeroed in live while training rows have nonzero frame-derived values. The README claims “measured neutral”; I cannot verify that from source. Confidence: medium; needs measurement.
- `train_prefix.py::_fill_row` sets `gold_mask_extra` from the posterior whenever a posterior exists, even when `row["gold_exact"]` is true. So live exact gold still gets posterior mask slack in the model’s legal mask. The decoder later filters with exact budget, but the ranking/mask is looser than exact. Confidence: high from code.

### 4. Telemetry purchase labels are contaminated by automatic grants

**Mechanism:** `app/telemetry.py::ShopDecisionSessions.observe` treats any net inventory addition as `purchase_observed`. Support quest upgrades, Magical Footwear grants, and automatic transformations are all inventory additions that are not purchases.

**Why it matters:** This is the one prospective dataset meant to fix the “Match-V5 cannot see manual no-buys” problem. If its positive purchase labels include automatic grants, the prospective labels are not trustworthy either. The code does distinguish removals as `inventory_changed_unattributed`, but not additions.

**Confidence:** high from code. Damage: high.

### 5. Performance: reconstruction and featurization are both repeatedly rebuilt

**Mechanism:** `app/ladder.py::_ingest_match_perspectives` and `app/backfill.py::_fill_match` call `reconstruct_game(...)` once, then call `reconstruct_visits(...)` for each perspective. `reconstruct_visits` calls `reconstruct_game` again internally. That is up to 10–11 full timeline replays per game instead of one replay plus `perspective_from_game`.

**Why it matters:** Backfill and ladder ingest scale with perspectives times the full reconstruction cost. This is exactly the kind of thing that turns a nightly pull into a multi-hour job.

**Confidence:** high from code. Damage: high for operational cost.

**Second performance bug:** `app/predictor.py::_probs` and `_target_probs` construct a fresh `tp.ShopDataset([row])` per call. `ShopDataset._init_meta` walks the label set and item set, and `tp.apply_config` re-reads the gold-posterior file when configured. `eval_policy.py` calls this per visit per model. For any posterior-backed artifact, policy evaluation will do two dataset constructions and two posterior file reads per scored visit. Confidence: high; damage: high for eval throughput.

### 6. Default ingest writes legacy tables only; training reads `game_events`

**Mechanism:** `app/ingest.py::ingest_faker` uses `insert_match_and_visits`, which writes `matches` and `shop_visits`. It does not write `games` or `game_events`. `app/db.py::init_db` runs `_unify_perspectives_into_games` only once, gated by a `meta` key. After that, default-ingested games are never migrated into `game_events`. The training exporter reads `game_events`.

**Why it matters:** Anyone using the documented default `python -m app.ingest` path can pull games that are invisible to training. The UI may see them through legacy tables; the trainer will not.

**Confidence:** high from code. Damage: high for anyone not using `--ladder`/`--backfill`.

### 7. Purchase legality is incomplete

**Mechanism:** `app/shop_econ.py::is_blocked` handles only completed-item uniqueness and boots. `eval_policy.py::check_displayed_purchases` checks recipe cost, wallet, regular slots, ward stacks, unique boots, and purchase blocks. It does not enforce the support-item line restriction, jungle-item uniqueness, or other unique-item restrictions that the game client enforces.

**Why it matters:** The advisor can display a basket that the game will refuse, or that the model learned from states where such purchases never occurred. The “legal alternatives” claim is weaker than it sounds.

**Confidence:** high that the code does not check these; the exact set of missing restrictions needs a static-data/game-rules audit. Damage: high if it happens in practice.

### 8. `decision_inventory or inventory_before` treats a legitimate empty inventory as missing

**Mechanism:** `scripts/baseline.py::_example` does:
`inventory_payload = event.get("decision_inventory") or event.get("inventory_before") or []`.
If a player sells everything before buying, `decision_inventory == []` is falsy, so the exporter falls back to the pre-sale inventory.

**Why it matters:** The model sees items the player no longer owns at decision time. That can make an unaffordable/illegal line look owned and can mislead the legality mask. It is a small population but a real state bug.

**Confidence:** high. Damage: medium.

### 9. Support-item chain is internally suspicious

**Mechanism:** `app/inventory.py::InventoryReplay.apply_batch` handles `ITEM_DESTROYED` on the support line with `upgrade = {3865: 3866, 3866: 3867, 3867: 3867}.get(item)`. A destroy-only upgrade from Bounty cannot reach a finished support item through that map. The comment in `CLAUDE.md`/`reconstruct.py` says Match-V5 never emits `ITEM_PURCHASED` for the support line, while the code has a branch for `ITEM_PURCHASED` of `SUPPORT_FINISHED`.

**Why it matters:** If the comment is right, finished support items are never reconstructed from destroy-only events. If the purchase branch is right, the comment is wrong. Either way, support inventories are a known special case and this is not clean.

**Confidence:** medium; need raw timeline event order to confirm. Damage: medium-high for support data.

### 10. `paired_diff` is not a paired comparison for prediction-dependent metrics

**Mechanism:** `scripts/eval_policy.py::paired_diff` raises if the scored game/visit sets differ, then builds `shared = set(pa) | set(pb)` and computes `mean_a - mean_b` over each policy’s own defined population. For `save_card_precision`, the population is “visits where a save card was shown”, which depends on the policy. So the point estimate compares different visit sets, and the bootstrap resamples a union where one policy may contribute zero visits in a resample.

**Why it matters:** The CLI labels this “the comparison that counts”. For save-card precision it is not a paired difference. The significance call can be misleading.

**Confidence:** high from code. Damage: medium.

### 11. `scripts/status.py` looks for the wrong manifest keys

**Mechanism:** `show_served_artifact` prints keys `artifact`, `report`, `split`. `promote_served_model.py` writes `artifact_name`, `evaluation_report`, `evaluation_split`. The status command therefore omits the artifact/report/split lines and dumps the real keys as “other keys”.

**Why it matters:** `CLAUDE.md` says `scripts/status.py` is the authority on what is running/served. It misreports the served artifact.

**Confidence:** high. Damage: low-medium operationally.

### 12. `scripts/baseline.py` is documented as the export path but its `main()` is disabled

**Mechanism:** `baseline.py::main` begins with `require_pipeline_clear("export")` then `raise SystemExit('In-place exports are retired...')`. Everything below is unreachable. README/CLAUDE still tell you to run `scripts/baseline.py` to export.

**Why it matters:** Onboarding and agent instructions are stale. The real path is `scripts/publish_patch_dataset.py`, which is not in the provided source. Confidence: high. Damage: medium-high for humans/agents following docs.

### 13. `hash()` for fallback champion IDs is not reproducible

**Mechanism:** `scripts/train_prefix.py::fix_champ_ids` and `_fill_row` use `abs(hash(player.get("champion") or "")) % ...` for missing champion IDs. Python string hashing is salted per process unless `PYTHONHASHSEED` is fixed. The code calls this “Deterministic-ish”.

**Why it matters:** Vocabularies and board tokens can change between runs, invalidating tensor caches and making training non-reproducible for any row with a missing champion ID.

**Confidence:** high. Damage: medium.

### 14. Save labels are built from a stale frame and a fixed 15 s post-death timestamp

**Mechanism:** `app/reconstruct.py` creates save events from `death_snaps`, which store `last_frames` before the death. It then stamps the shop event at `ts + 15_000` and uses that stale frame’s `currentGold`. It checks no purchase within 90 s and gold ≥ 400. It does not verify respawn or fountain.

**Why it matters:** At higher levels, 15 s after death may still be dead time. The save head can be trained on “could not buy” states as if they were deliberate no-buys. This is a label-quality issue, not a leak.

**Confidence:** high from code; the magnitude needs data. Damage: medium.

### 15. `/api/sync` bypasses the collector lease

**Mechanism:** `app/ingest.py` CLI acquires `CollectorLease`. `app/main.py::_sync_stream` calls `ingest_faker` directly with no lease. A web-triggered sync can run beside the scheduled task or another collector.

**Why it matters:** The project’s own rules forbid racing Riot sequences. The API path does not enforce it.

**Confidence:** high. Damage: medium.

### 16. Smaller but real

- `app/main.py::_get_predictor` raises `FileNotFoundError` if `data/ml/prefix_model.pt` is missing; `POST /api/live/session` returns a 500 instead of a clear 409. Confidence: high; damage: low-medium.
- `app/telemetry.py::_safe_row` names a field `gold_exact` but stores the gold amount, not a boolean. Confidence: high; damage: low.
- `app/config.py::selected_patches` raises on an unreviewed patch, so the “16-day fallback” warning in `app/ladder.py` is unreachable. Confidence: high; damage: low.
- `app/db.py::build_for_champion` queries a `decisions` table that `init_db` does not create. Confidence: high; damage: low if dead code.

## What I would need to run or measure

- Whether the promotion gate has actually been used to promote a development-only artifact, or whether the manifest keys are always correct in practice. I can only see the missing checks.
- Whether live serving has already been run against a patch different from the artifact’s Data Dragon version. Code shows no guard; actual severity depends on how often patches change while beta users play.
- Whether telemetry `purchase_observed` records contain support/footwear grants. That needs the JSONL and a join to game events.
- Whether incomplete legality actually produces displayed illegal baskets. Run `eval_policy.py` on a candidate and inspect `purchase_legality` failures and manual game-rule checks.
- The support-chain event order in real Match-V5 timelines. The code and the comment disagree; only raw timelines settle it.
- Live performance and transfer. The repo itself says these are unmeasured. Nothing in the source changes that.

The strongest impression is that this project has learned the right paranoid habits around provenance and leakage, but the last mile — promotion, live regime, and telemetry labels — has gaps that are exactly the kind the project’s own history warns about: no crash, no traceback, just plausible numbers or a convenient artifact that should not have been served.