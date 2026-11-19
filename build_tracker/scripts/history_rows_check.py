"""Row-content check of a validated 16.17 history export against its frozen corpus (second review, 2026-09-27).

`history_1617.validate_history` binds the row file to the manifest's own digest, so a consistent two-file
forgery (relabelled rows + re-hashed manifest) passed. `history_1617.py` is pinned build code of the completed
reconstruction runs (the exporter refuses to run if it drifts), so this hardening lives in its own module and is
called AFTER validate_history by the exporter, the cache builder and the trainer:

  * every row of a deterministic sample of games (all games when few) must carry its own event's state
    (gold_est, level, gold_left, kills, deaths, assists, cs, champion_id, team_id) and a label the event actually
    bought (save rows: the event is a save with no purchase); the anchor is the digest-pinned corpus database;
  * the reviewed row-builder declaration recorded by the exporter (`reviewed_row_builder`) must still match the
    live files, so `corpus_rows.py` / `build_recovered_patch_corpus.py` / `train_prefix.py` are re-checked at
    training time as well as at export time.
"""
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVE_SENTINEL = 999999            # the no-buy label item id (app.shop_econ.SAVE_ITEM)
SPOT_CHECK_GAMES = 300            # sampled games whose rows are compared with the frozen corpus events
ROW_EVENT_FIELDS = ("gold_est", "level", "gold_left", "kills", "deaths", "assists", "cs", "champion_id", "team_id")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    raise ValueError("history rows rejected: " + msg)


def spot_check_games(games):
    """A deterministic sample of exported games (all of them when few)."""
    ordered = sorted(games)
    step = max(1, len(ordered) // SPOT_CHECK_GAMES)
    return ordered[::step][:SPOT_CHECK_GAMES] if len(ordered) > SPOT_CHECK_GAMES else ordered


def check_rows_against_corpus(corpus, rows_by_game):
    """Every sampled row must carry its own event's state and a label the event actually bought."""
    uri = (Path(corpus) / "tracker.db").as_uri() + "?mode=ro&immutable=1"
    checked = 0
    with sqlite3.connect(uri, uri=True) as conn:
        for mid, rows in rows_by_game.items():
            events = {}
            for (payload,) in conn.execute("SELECT payload_json FROM game_events WHERE match_id=?", (mid,)):
                e = json.loads(payload)
                events[(e.get("participant_id"), e.get("ts"))] = e
            for row in rows:
                key = (row["participant_id"], row["ts"])
                e = events.get(key)
                if e is None:
                    fail(f"row {(mid,) + key} has no event in the frozen corpus")
                for f in ROW_EVENT_FIELDS:
                    if f in row and f in e and row[f] != e[f]:
                        fail(f"row {(mid,) + key} field {f} differs from its event ({row[f]} vs {e[f]})")
                bought = {int(b["item_id"]) for b in e.get("label_bought", []) if not b.get("target_skip", b.get("skip"))}
                labels = {int(k) for k in row.get("label_counts", {})}
                positive = {i for i in labels if i > 0}
                if positive == {SAVE_SENTINEL}:                  # a no-buy decision: its event must be a save with no purchase
                    if not e.get("is_save") or bought:
                        fail(f"row {(mid,) + key} is a save row but its event bought {sorted(bought)}")
                elif positive and not positive <= bought:
                    fail(f"row {(mid,) + key} labels {sorted(positive)} are not among its event's purchases {sorted(bought)}")
                label_id = row.get("label_id") or 0
                if label_id > 0 and label_id != SAVE_SENTINEL and label_id not in bought:
                    fail(f"row {(mid,) + key} label_id {label_id} is not among its event's purchases")
                checked += 1
    return checked


def validate_rows(directory, manifest):
    """Call after history_1617.validate_history(directory) returned `manifest`. Returns the number of rows checked."""
    directory = Path(directory).resolve()
    reviewed = manifest.get("reviewed_row_builder")
    if not reviewed:
        fail("the manifest carries no reviewed row-builder declaration (export predates the second review)")
    drifted = [name for name, digest in reviewed.items() if sha256_file(ROOT / name) != digest]
    if drifted:
        fail(f"reviewed row-builder code changed since export: {drifted}")
    games = set(manifest["games"]["exported"])
    sampled = set(spot_check_games(games))
    rows_by_game = defaultdict(list)
    with open(directory / manifest["rows"]["file"], encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["match_id"] in sampled:
                rows_by_game[row["match_id"]].append(row)
    if set(rows_by_game) != sampled:
        fail("sampled games are missing rows")
    return check_rows_against_corpus(Path(manifest["corpus"]["path"]), rows_by_game)
