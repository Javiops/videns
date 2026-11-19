# CLAUDE.md

Guidance for AI assistants in this repository. `AGENTS.md` points here.

## What this is

A behavioural-imitation macro advisor for League of Legends junglers: collect
challenger junglers' revealed decisions with full state snapshots, train a
model that proposes what a challenger would do in a given state, and let the
owner audit it. It does **not** optimise outcomes. Never call its output
"the correct play" — it is what challengers did in similar states.

Sibling of `../build_tracker` (same owner, same Riot key, same engineering
culture). **No shared code or data** between the two projects — reimplement,
don't import across.

Spanish is the project language for owner-facing docs (`README.md`,
`LEXICON.md`, `docs/`); this file is English by convention.

## Rules that override your defaults

1. **Unmeasured is "not measured".** A number without
   `<metric, split, n, report path>` is inadmissible.
2. **No invented labels for what the sources cannot observe.** Visibility truth
   comes ONLY from the fog passes; never approximate it from omniscient data.
   `WARD_PLACED` has no coordinates — wards are out of scope entirely (owner
   decision), not to be re-introduced via proxies.
2b. **Every CV measurement needs a negative control, not just a positive one.**
   A positive control (own team is always visible) only catches false
   negatives. Dead champions are not drawn at all: that control exposed an
   80-87% false-positive rate that the positive control called healthy. Any new
   detector claim without both controls is inadmissible.
3. **Verify in code before asserting.** Docs go stale; read the file.
4. **No reported metrics under 30 games** (`MIN_GAMES_REPORT` in
   `app/config.py`). `--smoke` runs are plumbing checks and must say so.
5. **The lexicon is versioned.** Changing a token or a detection rule bumps
   `LEXICON_V` and regenerates every `decisions.jsonl`; never mix versions in
   one dataset. Same for `FEATURES_V` (feature vector) and `SNAPSHOT_V`
   (snapshot schema) — `dataset.export()` refuses rows from another version.
6. **Retrospective truth is not an input.** The snapshot carries two layers and
   they must never be confused: `players[*]` holds what actually happened
   (from the no-fog pass — for labelling and for the owner's review), while
   `players[*].observed` holds what the POV team could know at that instant.
   **Features may only read `observed`** (`app/dataset.py`). An enemy in fog
   contributes his last confirmed sighting and its age, or nothing at all —
   never his real position. Same for time: anything feeding the model uses
   `trajectory.pos_at_causal`, never `pos_at`, which happily returns a sample
   from *after* the decision. `scripts/check_leakage.py` asserts both over the
   real corpus (it was 312 of 315 hidden enemies leaking their region).
7. **Unknown is `None`, never a plausible number.** If the detector's
   sensitivity for a champion is below `VIS_MIN_SENSITIVITY`, his
   non-detection proves nothing and his visibility is unknown — not "hidden".
   A suspicious pass invalidates that game's visibility outright. An aggregate
   frequency correction (`visible_frac_corrected`) is for auditing, never for
   inventing instant-by-instant visibility.

## Hard prohibitions

- No Live Client `eventdata` — it duplicates events across passes (audited).
  `playerlist` is fine (state snapshot).
- No test-set reads outside one final confirmatory check.
- Never commit anything under `data/` beyond the tracked assets
  (map, portraits, region mask).
- **Never run a capture session as a child of the agent shell.** Background
  commands get reaped mid-run (it happened twice, losing a pass each time).
  Launch detached and tail the log:
  `Start-Process .venv\Scripts\python.exe -ArgumentList "-u","scripts\capture_queue.py" -RedirectStandardOutput data\logs\capture_<stamp>.log -RedirectStandardError data\logs\capture_<stamp>.err -WindowStyle Hidden`
  The queue is resumable: finished games are skipped, so a kill costs at most
  the pass in flight (delete pass dirs without `samples.jsonl` before resuming).
- Do not touch mouse/keyboard expectations during a capture pass; if the game
  loses focus the frames capture another window.
- **Never process while capturing.** `process_all --jobs 8` saturating the 16
  cores starves the game's renderer: the replay clock keeps advancing while the
  screen does not redraw, and ~97% of frames come out byte-identical. The
  capture loop now detects and skips duplicates (`dup_frac` in the pass meta)
  and aborts after `DUP_STREAK_ABORT` in a row, but the rule stands.
- **Capture speed is 32x, and the limit is density in GAME seconds, not speed.**
  Above ~3.5 s between samples, visibility sensitivity collapses (81.7% → 50%),
  so 64x is out (4.5 s). 32x holds 8x's density (2.9 s) in a third of the time —
  but only with the async PNG writer and the persistent Replay-API connection;
  remove either and the loop floor pushes density past 3.5 s.
  **Nothing new may add work per frame.** At 32x a loop turn has ~75 ms of wall
  clock and a screen grab costs 10-15 ms of it. One grab per turn: the minimap
  crop and the scoreboard crop both come out of the same `pantalla`. Anything
  that must be read off the screen is stored as pixels and interpreted
  afterwards (that is why `app/ocr.py` runs offline, never in the loop).
  Whether the screen is legible at 32x is a measurement, not an assumption:
  `read_gold.py` reports `frac_legible` next to the pass `speed`.
- Dev Riot keys expire every 24h; 401/403 must abort ingest loudly.

## Commands

Windows; call the venv interpreter directly, no activation needed.

| Task                    | Command                                                        |
| ----------------------- | -------------------------------------------------------------- |
| Current state           | `.venv\Scripts\python.exe scripts\status.py`                   |
| Run tests               | `.venv\Scripts\python.exe scripts\run_tests.py`                |
| Queue pro jungle games  | `... scripts\collect_pro_games.py --games 40 --fill-challenger` |
| Queue generic games     | `... scripts\collect_matches.py --patch 16.18 --games 40`      |
| Download all replays    | `... scripts\download_queue.py` (client open; patch window!)   |
| Capture run             | `... scripts\capture_queue.py` (machine untouched, ~4 min/game at 32x) |
| Set spectator minimap   | `... scripts\set_minimap.py` (game closed; no args = show)     |
| Preflight before a batch| `... scripts\preflight_gold.py <ID>` (gold at 32x vs Match-V5) |
| Learn gold digits       | `... scripts\learn_digits.py [--check-corpus]`                  |
| Read gold from crops    | `... scripts\read_gold.py` (writes `gold.jsonl` + report)      |
| Resolve gold at buys    | `... scripts\resolve_buys.py` (writes `buys.jsonl` + report)   |
| Audit fog (also a gate inside process_all) | `... scripts\audit_fog.py`          |
| Visibility quality      | `... scripts\vis_quality.py` (sensitivity distribution)        |
| Leakage check           | `... scripts\check_leakage.py` (must report 0 leaks)          |
| Process all captured    | `... scripts\process_all.py --jobs 8` (audits first; refuses a bad batch) |
| One game: fetch/capture/process | `fetch_replay.py <ID> --watch` / `capture_game.py <ID>` / `process_game.py <ID>` |
| Build frozen dataset    | `... scripts\build_dataset.py` (`--rebuild-split` is explicit) |
| Train baseline          | `... scripts\train_baseline.py`                                |
| Review UI (owner)       | `... scripts\review_app.py` then http://127.0.0.1:8010         |

`RIOT_API_KEY` lives in `.env` (see `.env.example`). Replay capture needs the
League client on the game's patch, `EnableReplayApi=1` in
`C:\Riot Games\League of Legends\Config\game.cfg`, and an undisturbed machine.

## Architecture

Riot API + LCU replays → 3 capture passes → detection/visibility →
segments → lexicon decisions → JSONL corpus → frozen split → sklearn baseline
→ owner review loop.

**The three passes and what each is for** (`capture.ROLES`):

| Pass | Fog | Captures | Exists for |
| ---- | --- | -------- | ---------- |
| `omni` **TRUTH-SEEKER** | Todos | minimap + **scoreboard in gold view** + playerlist | ground truth: positions, and the **only** source of instant gold |
| `blue` **REAL-BLUE** | Azul | minimap | what blue could SEE |
| `red` **REAL-RED** | Rojo | minimap | what red could SEE |

The spectator does **not** limit a player's card to what their team sees — the
scoreboard hands out all ten builds under any fog (owner-confirmed, 2026-09-19).
So the fog passes are **not** the source of builds; they are the source of
**visibility**, which nothing else can give. "What blue saw" is *constructed*:
exact build from `playerlist` gated on those visibility intervals. Never read a
build off a fog pass and call it that team's view.

| File                | Role                                                                    |
| ------------------- | ----------------------------------------------------------------------- |
| `app/config.py`     | Paths, constants (map size, confidence horizon, thresholds), .env       |
| `app/riot.py`       | Rate-limited httpx client (1.3s gap, 95 req/2min), gzip cache, 404→None |
| `app/lcu.py`        | League client lockfile auth; replay download/watch                      |
| `app/regions.py`    | 13-region mask (`data/region_labels.npy`), camps, quadrants             |
| `app/lexicon.py`    | Canonical decision tokens (`LEXICON.md` is the contract)                |
| `app/inventory.py`  | 10-player build reconstruction from timeline item events                |
| `app/capture.py`    | 3-pass replay capture (omni + blue fog + red fog) with fog validation   |
| `app/calibrate.py`  | Minimap geometry by multi-scale correlation of `data/map_background.png`; Match-V5 positions verify it |
| `app/detect.py`     | Champion detection on minimap frames (ddragon portraits, size-scaled)  |
| `app/icons.py`      | Per-champion minimap icons **learned from the omni pass**; the only templates discriminative enough for visibility |
| `app/trajectory.py` | Trajectory cleaning + Match-V5 60s anchors fusion                       |
| `app/camps.py`      | Camp diamond series → alive/dead intervals                              |
| `app/visibility.py` | Per-team visibility intervals + confidence metric (0–100)               |
| `app/segments.py`   | The audited segmentation core (ported from prototype `macro_v5.py`)     |
| `app/snapshot.py`   | Decision-point snapshots: builds, objectives, vision, minimap refs      |
| `app/ocr.py`        | Reads `actual(total)` gold for the ten off the scoreboard; unknown is `None` |
| `app/buys.py`       | Exact gold **at each purchase** for the ten: Match-V5 buy instants × OCR/Match-V5 wallet anchors + passive; unlocalizable income → interval, never invented |
| `app/dataset.py`    | Feature vector `f2` (**observable only**), split validation, JSONL export |

## Domain gotchas

- `WARD_PLACED` has no position in Match-V5. Wards are out of scope — the fog
  passes capture vision's net effect instead.
- Item events with `participantId <= 0` exist in real data; skip them.
- Support item line (3865→3866→3867) upgrades via ITEM_DESTROYED, and
  `ITEM_UNDO` returns components silently — `app/inventory.py` documents the
  approximations; playerlist samples + match `item0..item6` are the checks.
- `participantFrames` are 60s stale by design: snapshot gold is
  `*_prev_frame`, never interpolate `currentGold` across purchases. **The HUD
  fixes this**: with the gold action on (owner's mouse3), the scoreboard shows
  `current(total)` for the ten, and it matches Match-V5 to 1-2 gold at the same
  instant (measured, `EUW1_7986655074` t=900). It is the only source of instant
  gold — the Live Client API exposes none in replay. Crops live in the
  TRUTH-SEEKER pass at the same instants as the playerlist; `app/ocr.py` reads
  them offline. `ensure_gold_view()` now verifies **by reading the ten cells**,
  not by how colourful the bar is: that indirect check is why all 40 first games
  were captured in items view without anything complaining.
- Everything except gold comes exactly from `playerlist` (items, level, CS,
  dead, respawn, KDA, ward score). Don't build a CV path for what the API
  already gives exactly. Runes and summoner spells are read into `meta.roster`
  once per pass (they don't change), defensively with `.get`: **whether replay
  mode actually fills `runes` is unverified** — the next capture settles it, and
  an empty `runas` in the roster is the tell.
- Replays: same region as the client (EUW), same patch, ~2 week window. The
  LCU refuses to download a replay whose local metadata does not exist yet —
  create it first (`lcu.create_replay_metadata`, done by `download_replay`).
- **Fog of war is controlled by the replay HUD dropdown, not the API.**
  `render.fogOfWar` gives no team perspective and `selectionName` is read-only.
  `capture.set_fog()` locates the "Niebla de guerra" label on screen, verifies
  the list actually opened, and picks the row **by its text colour** — the list
  is positioned relative to the current value, so fixed offsets silently chose
  the wrong team. That produced passes with the same fog twice and omni passes
  carrying the previous game's fog, and **capture-time validation cannot see
  either**: run `scripts/audit_fog.py` over every batch before processing.
- **Minimap geometry is calibrated, never assumed** (`app/calibrate.py`, by
  multi-scale correlation of `data/map_background.png`). In replays it renders
  centred, not bottom-right as in phase 1. Frames are stored already cropped to
  the minimap (`format: minimap-crop-v2`); phase-1 passes keep the old format
  and their `geom`/`geom_switches`.
- Detector template sizes scale with the calibrated minimap
  (`detect.sizes_for`); fixed phase-1 sizes miss ~2/10 champions.
- **The spectator minimap is set from `PersistedSettings.json`, not `game.cfg`.**
  The in-game slider writes `game.cfg`; the client's per-account store
  overrides it on next launch, which is why the owner's minimap "resets after
  the replay" (they had 5.0/-0.3448 in `game.cfg` and 1.0/-0.0203 winning from
  PersistedSettings). `scripts/set_minimap.py` writes the store that wins —
  **with the game closed**, or it gets rewritten on exit. Current setting:
  scale 5.0, offsets -0.3448/-0.2269 → 508px at (726,299), which clears the
  gold strip (y≥820), both team-frame columns (x≤95 and x≥1835 — the corner
  position covered the red support's death timer) and the top bar.
- **Geometry guards use a tolerance, never `==`.** Minimap calibration jitters
  ±1-2px between captures of the same size (the corpus has 508 and 507 mixed).
  A strict-equality guard failed real games in the gold batch on a 1px diff;
  the guards now allow `GEOM_SIZE_TOL` (8px) — enough to catch a real scale
  change (508 vs 255 = 253px) and ignore the noise. General rule: an
  exact-equality check on a measured, jittering quantity is a false-positive
  factory.
- **Match-V5 frames can already reflect an event its own stream timestamps
  slightly later.** A participant frame at t=540.0 showed the wallet *after* a
  purchase the event list dates at 540.2 (`EUW1_7986562911`). `app/buys.py`
  gold-at-purchase is correct through this (it brackets on the OCR wallets and
  passive); only the anchor *validation* was fooled — it now skips anchors
  within `LAG_HUD_S` of an event. Worst anchor error across 40 games dropped
  from 1597 to 10 gold. Don't validate a resolved value against an anchor that
  straddles the very event being resolved.
- **The three passes of a game must share one minimap geometry.** Icons are
  learned on the omni pass and searched for in the fog passes, so different
  sizes break the match — and it does **not** fail, it inflates visibility
  toward "always visible", the one direction that teaches the model nobody ever
  hides. Measured on `EUW1_7984516268`: all three at 508px gave blue-sees-red
  0.747/0.137/0.563/0.88/0.585; recapturing **only** the omni at 255px turned
  that into 0.909/0.327/0.968/1.0/1.0 and left the red pass with no sensitivity
  at all. `suspect_no_fog` caught the red side; the blue side would have passed
  with invented numbers. `build_visibility` now refuses a size mismatch. So
  re-capturing one pass of an existing game is only safe if the owner's HUD is
  back to the same minimap size — otherwise recapture all three.
- Scuttle has no camp diamond; it is inferred from jungle CS in river
  segments (rule added in phase 2, pending audit).
- The prototype game `EUW1_7983109023` (patch 16.18, Evelynn ORDER jungle) is
  the regression fixture ONLY — see `EXCLUDED_MATCHES` in `app/config.py`. It
  is the owner's own soloq game, not a challenger jungler's, it has no fog
  passes (`vision.available=false`) and uses the phase-1 capture format, so it
  must never enter the training corpus. `dataset.corpus_games()`,
  `process_all.py` and `audit_fog.py` all skip it; the regression test reaches
  it by path.

## Where the rest lives

- **`HANDOFF.md`** — where everything stands right now: corpus, split, model
  numbers, the operational findings that make capture work, known limitations
  and what to do next. Read this first when picking the project back up.
- **`LEXICON.md`** — the decision vocabulary contract. Read before touching
  labels.
- **`docs/design.md`** — phase-2 architecture and collection plan.
- **`docs/history.md`** — ledger of withdrawn beliefs. Add to it whenever an
  approach is disproved; read it before proposing one.
