# CLAUDE.md

Guidance for AI assistants in this repository. `AGENTS.md` points here.

## What this is

An in-game overlay for League of Legends: two small translucent always-on-top
windows that show, from the League client's local Live Client Data API only,
Baron/dragon respawns and Baron/Elder buffs with their holders, and team item
gold as two columns (BLUE, RED) of five rows, each row a rectangle of the
champion's splash chosen by the owner in `/faces` (the same for both teams),
fading into black towards the number (BLUE: number right; RED: number left,
fade mirrored). A FastAPI server on port 8020 shapes
the data and caches images; a pywebview launcher draws the windows.

It has **no model, no Riot API key, no database**. It never reads game memory
and never injects input: the Live Client Data API is the sanctioned local
endpoint (the one Blitz and Porofessor use), and the only key it touches is
read with `GetAsyncKeyState` to make the windows opaque while Tab is held.

Sibling of `../build_tracker` (shop advisor) and `../macro_coach` (jungle
macro). Same owner, same engineering culture. **No shared code or data between
the three projects** — reimplement, don't import across. Born inside
build_tracker on 2026-09-29 and moved out the same day so that nothing here
can touch the shop model or its operational envelope.

Spanish is the project language for owner-facing docs (`README.md`, `docs/`);
this file is English by convention. Code comments and docstrings are Spanish.
The UI says BLUE / RED (the owner's words), the API payload keeps ORDER /
CHAOS and team ids 100 / 200.

## Rules that override your defaults

1. **Only what the API observes.** `allgamedata` gives items, gold of the
   active player, level, death state, runes, summoner spells and the event list
   (epic monster kills, champion kills, buildings). It gives no positions, no
   buffs, no cooldowns, no damage, no regular jungle camps. A feature that
   needs any of those is either derived (say how, and show its uncertainty in
   the UI) or out of scope (see `docs/design.md`). Never fake it from screen
   pixels or memory.
2. **Constants carry provenance.** Every duration in `app/tracker.py` says
   whether it was checked against captured games or copied from the wiki.
   Buff durations cannot be measured through this API; the page says so.
3. **`price` is the combine cost.** Measured, not assumed (see `docs/design.md`).
   Item values come from Data Dragon `gold.total`; the API field is only a
   flagged fallback for ids missing from the catalog.
4. **Event names are untagged.** `KillerName` / `Assisters` are
   `riotIdGameName` when the player has one; hidden-name players arrive as
   riotIdGameName "" / riotId "#" / summonerName == champion, and events use
   that summonerName. The tagged `riotId` never matches. A name shared by two
   players is not guessed: assisters decide, else the team is "?".
5. **Derived state must degrade honestly.** Holders unknown when the overlay
   opened after the kill; `≈` when the catalog lacks an id or is not loaded;
   `?` for a dragon spawn that may be Elder; "Summoner's Rift only" outside
   map 11. Never a plausible number for something not observed.
6. **Never block a poll on the network.** The catalog loads in a background
   thread; images are fetched once and cached; Data Dragon failures back off.
7. **Verify in code before asserting.** Docs go stale; read the file.

## Hard prohibitions

- No import from `../build_tracker` or `../macro_coach`, and no reading their
  `data/` from code. `scripts/replay.py` takes any directory as an explicit
  argument; this project records its own corpus with `scripts/record.py`.
- No memory reading, no screen scraping, no input injection. Reading the Tab
  key state is the only keyboard access, and it stays read-only.
- Never commit anything under `data/`.
- No long jobs inside an agent session; `scripts/record.py` runs in the
  owner's own terminal while they play.
- When launching the overlay from an agent session, create the process
  through WMI (`Win32_Process.Create`) — a child of the agent's shell dies
  with the tool call.

## Commands

Windows; call the venv interpreter directly, no activation needed.

| Task            | Command                                                                  |
| --------------- | ------------------------------------------------------------------------ |
| Run tests       | `.venv\Scripts\python.exe scripts\run_tests.py`                           |
| Server only     | `.venv\Scripts\python.exe -m uvicorn app.main:app --port 8020`            |
| Overlay windows | `start_overlay.bat` (or `.venv\Scripts\pythonw.exe -m app.overlay`)       |
| Stop everything | `stop_overlay.bat` (closing the windows leaves the server up for `/faces`) |
| Record a game   | `.venv\Scripts\python.exe scripts\record.py`                              |
| Capture a game for the landing | `start_capture.bat` (`scripts\capture.py`; `--selftest` checks it): champ-select sessions from the client's local API (lockfile), allgamedata every 2 s, JPEG screenshots (2 s in champ select, 5 s in game), F9 marks a moment with 5 shots. Output in `data/captures/<start>/`. Runs in the owner's terminal. |
| Replay a game   | `.venv\Scripts\python.exe scripts\replay.py --dir data\snapshots`         |

League must be in Borderless or Windowed display mode for the windows to draw
over it. Launcher errors go to `data/overlay.log`, server stderr to
`data/server.err`, window positions to `data/layout.json`.

## Architecture

| File                  | Role                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `app/live_client.py`  | One GET to `https://127.0.0.1:2999/liveclientdata/allgamedata`; `None` without a game.   |
| `app/ddragon.py`      | `Catalog`: item id → `gold.total`, champion raw/display name → image id; cached per version with atomic writes, offline fallback to older caches, corrupt files discarded. |
| `app/tracker.py`      | **The core.** `OverlayTracker.observe(snap, catalog)` → panel payload. State: seen events, buff holders (by player identity), last poll's deaths. Detects a new game by roster, clock rewind or EventID stream conflict. |
| `app/main.py`         | FastAPI: `/` (panel), `/faces` (head marks), `/draft` (champion-select widget preview), `/videns-mark.svg` (the V. mark, copy of `../brand/v-mark.svg`), `/rows.css` + `/rows.js` (shared row geometry), `/api/overlay`, `/api/faces` (GET/PUT/DELETE, `config/faces.json`), `/api/health`, cached image proxies `/img/champion/{id}.png` (round icons for buff holders), `/img/splash/{id}.jpg` (the rows: an owner-provided `config/splashes/{id}.jpg` first, then CommunityDragon's current client art, then Data Dragon's unversioned and sometimes pre-rework splash), `/img/objective/{name}.png`. Catalog loads in a background thread. |
| `web/rows.js`         | `Rows.mark(entry)` / `Rows.rect(entry)` / `Rows.artStyle(entry)`: the splash rectangle (centre + width, row proportion) a 150×34 row shows; old head-only marks are converted; used by the panel and by `/faces`. |
| `app/overlay.py`      | Two pywebview windows; Win32 layered alpha (translucent / opaque on Tab), rounded corners, single instance, layout persistence, log. `transparent=True` and colour keys do not work with WebView2 — uniform alpha does. |
| `web/overlay.html`    | The panel; `?panel=objectives|gold&overlay=1` are the widgets, `/` shows both. Countdowns interpolate between 1 Hz polls with a monotonic clock. |
| `web/game.html`       | `/game`: design preview of three in-game windows in the `/draft` style (each with the VIDENS. wordmark): OBJECTIVES, ITEM GOLD (the /faces rows) and CS / MIN (yours vs the drafting tool's expected CS for your champion and role at that minute), over a blurred mock game screen. The CS in the fixture comes from that game's match-v5 timeline. Data: `web/preview/game-fixture.json`, one real moment (21:01) of EUW1_7982999189 taken from videns-site's published review-data, plus that second's minimap (`web/preview/minimap.webp`), served by `/preview/{name}`. The live windows still use `web/overlay.html`. |
| `web/draft.html`      | `/draft`: design preview of the champion-select overlay: ONE window headed by the VIDENS. wordmark (Inter, `/inter-var-latin.woff2`), sections WIN, PICK (top 3; Karthus hidden), LANE (gold, CS/min and KP, you · enemy laner), TEAMS (gold, drakes, damage) and BAN in the ban phase, left of a mock champ select, shown at your last pick (`?step=N` for other moments of the scripted draft). Numbers come over HTTP from the drafting tool (`../draft`, `scripts/tool_server.py` on :8030; `?draft_api=` overrides); no code shared. Not wired to the client's champ-select session yet. Never shows the worst reply (owner). |
| `scripts/record.py`   | Dumps live snapshots to `data/snapshots/` (own corpus, millisecond names).              |
| `scripts/replay.py`   | Serves the panel replaying a snapshot directory in real time from the first poll.        |

## Where the rest lives

- **`docs/design.md`** — what the API gives, what each feature derives from
  it, the measurements behind the constants, image sources, and the features
  judged out of scope with the reason.
- **`docs/history.md`** — the ledger of withdrawn beliefs and traps.
- **`HANDOFF.md`** — current state and queue.
