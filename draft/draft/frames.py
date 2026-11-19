"""Sidecar extraction: per-participant total gold at fixed minutes, for the admitted games of
the v1 extract. Kept separate so the frozen v1 extract, split and models stay byte-identical."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

from .admission import RAW, check_sidecar
from .extract import ROLES, read_envelope

MINUTES = (6, 12, 18, 24, 30, 36)


def frame_gold(timeline: dict, match: dict) -> list | None:
    """[[side][role]] -> list of totalGold at MINUTES (None once the game has ended)."""
    frames = timeline["info"]["frames"]
    slot = {}
    for p in match["info"]["participants"]:
        slot[str(p["participantId"])] = (0 if p["teamId"] == 100 else 1, ROLES.index(p["teamPosition"]))
    out = [[[None] * len(MINUTES) for _ in range(5)] for _ in range(2)]
    for k, m in enumerate(MINUTES):
        if len(frames) <= m:
            continue
        for pid, pf in frames[m]["participantFrames"].items():
            s, r = slot[pid]
            out[s][r][k] = pf["totalGold"]
    return out


def extract_frames(admission_path: Path, games_path: Path, out_path: Path, log=print) -> dict:
    check_sidecar(admission_path)
    wanted = []
    with gzip.open(games_path, "rt", encoding="utf-8") as f:
        for line in f:
            wanted.append(line[7:line.index('"', 7)])
    tmp = out_path.with_suffix(".partial")
    start, written, missing = time.time(), 0, 0
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for i, mid in enumerate(wanted):
            tl_path = RAW / "timeline" / f"{mid}.json.gz"
            if not tl_path.exists():
                missing += 1
                continue
            match, _ = read_envelope(RAW / "match" / f"{mid}.json.gz")
            timeline, _ = read_envelope(tl_path)
            out.write(json.dumps({"id": mid, "g": frame_gold(timeline, match)}, separators=(",", ":")) + "\n")
            written += 1
            if i % 5000 == 0:
                log(f"{i}/{len(wanted)} written={written} missing_timeline={missing} {time.time() - start:.0f}s")
    tmp.replace(out_path)
    report = {"minutes": MINUTES, "games": len(wanted), "written": written, "missing_timeline": missing,
              "games_v1_sha256": hashlib.sha256(games_path.read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
              "seconds": round(time.time() - start, 1)}
    out_path.with_name("frames_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def frame_cs(timeline: dict, match: dict) -> list:
    """[[side][role]] -> creep score (lane minions + jungle monsters) at MINUTES (None once ended)."""
    frames = timeline["info"]["frames"]
    slot = {str(p["participantId"]): (0 if p["teamId"] == 100 else 1, ROLES.index(p["teamPosition"]))
            for p in match["info"]["participants"]}
    out = [[[None] * len(MINUTES) for _ in range(5)] for _ in range(2)]
    for k, m in enumerate(MINUTES):
        if len(frames) <= m:
            continue
        for pid, pf in frames[m]["participantFrames"].items():
            s, r = slot[pid]
            out[s][r][k] = pf["minionsKilled"] + pf["jungleMinionsKilled"]
    return out


def extract_cs_frames(games_path: Path, out_path: Path, log=print) -> dict:
    """Same games and the same timeline files as the gold sidecar (v1 timeline set only)."""
    wanted, tl = [], {}
    with gzip.open(games_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            wanted.append(r["id"])
            tl[r["id"]] = r["tl"]
    tmp = out_path.with_suffix(".partial")
    start, written = time.time(), 0
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for i, mid in enumerate(wanted):
            if not tl[mid]:          # keep exactly the v1 timeline population
                continue
            match, _ = read_envelope(RAW / "match" / f"{mid}.json.gz")
            timeline, _ = read_envelope(RAW / "timeline" / f"{mid}.json.gz")
            out.write(json.dumps({"id": mid, "cs": frame_cs(timeline, match)}, separators=(",", ":")) + "\n")
            written += 1
            if written % 5000 == 0:
                log(f"{i}/{len(wanted)} written={written} {time.time() - start:.0f}s")
    tmp.replace(out_path)
    report = {"minutes": MINUTES, "games": len(wanted), "written": written,
              "games_v1_sha256": hashlib.sha256(games_path.read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
              "seconds": round(time.time() - start, 1)}
    out_path.with_name("cs_frames_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def extract_match_stats(games_path: Path, out_path: Path, log=print) -> dict:
    """Per participant kills/deaths/assists and per team dragons/kills, from match payloads only."""
    wanted = []
    with gzip.open(games_path, "rt", encoding="utf-8") as f:
        for line in f:
            wanted.append(line[7:line.index('"', 7)])
    tmp = out_path.with_suffix(".partial")
    start = time.time()
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for i, mid in enumerate(wanted):
            match, _ = read_envelope(RAW / "match" / f"{mid}.json.gz")
            info = match["info"]
            kda = [[None] * 5 for _ in range(2)]
            for p in info["participants"]:
                s, r = (0 if p["teamId"] == 100 else 1), ROLES.index(p["teamPosition"])
                kda[s][r] = [p["kills"], p["deaths"], p["assists"]]
            teams = [None, None]
            for t in info["teams"]:
                o = t.get("objectives", {})
                teams[0 if t["teamId"] == 100 else 1] = {"dragons": o.get("dragon", {}).get("kills"),
                                                         "kills": o.get("champion", {}).get("kills")}
            out.write(json.dumps({"id": mid, "kda": kda, "teams": teams}, separators=(",", ":")) + "\n")
            if i % 5000 == 0:
                log(f"{i}/{len(wanted)} {time.time() - start:.0f}s")
    tmp.replace(out_path)
    report = {"games": len(wanted), "games_v1_sha256": hashlib.sha256(games_path.read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(), "seconds": round(time.time() - start, 1)}
    out_path.with_name("match_stats_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def frame_events(timeline: dict, match: dict) -> dict:
    """Cumulative kills/assists per participant and dragons per team at each MINUTE (None when the
    game did not reach it), plus whole-game totals for the integrity check."""
    frames = timeline["info"]["frames"]
    slot = {p["participantId"]: (0 if p["teamId"] == 100 else 1, ROLES.index(p["teamPosition"]))
            for p in match["info"]["participants"]}
    kills = [[0] * 5 for _ in range(2)]
    assists = [[0] * 5 for _ in range(2)]
    dragons = [0, 0]
    at = {}
    cut = {m: frames[m]["timestamp"] for m in MINUTES if len(frames) > m}
    snapshots = sorted(cut.items(), key=lambda kv: kv[1])
    events = sorted((ev for fr in frames for ev in fr["events"]), key=lambda ev: ev["timestamp"])
    i = 0
    for m, ts in snapshots + [(None, float("inf"))]:
        while i < len(events) and events[i]["timestamp"] <= ts:
            ev = events[i]
            if ev["type"] == "CHAMPION_KILL":
                if ev.get("killerId") in slot:
                    s, r = slot[ev["killerId"]]
                    kills[s][r] += 1
                for a in ev.get("assistingParticipantIds") or []:
                    if a in slot:
                        s, r = slot[a]
                        assists[s][r] += 1
            elif ev["type"] == "ELITE_MONSTER_KILL" and ev.get("monsterType") == "DRAGON" and ev.get("killerTeamId") in (100, 200):
                dragons[0 if ev["killerTeamId"] == 100 else 1] += 1
            i += 1
        snap = {"k": [row[:] for row in kills], "a": [row[:] for row in assists], "d": dragons[:]}
        if m is None:
            at["final"] = snap
        else:
            at[str(m)] = snap
    return at


def extract_event_frames(games_path: Path, out_path: Path, log=print) -> dict:
    wanted, tl = [], {}
    with gzip.open(games_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            wanted.append(r["id"])
            tl[r["id"]] = r["tl"]
    tmp = out_path.with_suffix(".partial")
    start, written = time.time(), 0
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for i, mid in enumerate(wanted):
            if not tl[mid]:          # the v1 timeline population, as for gold and CS
                continue
            match, _ = read_envelope(RAW / "match" / f"{mid}.json.gz")
            timeline, _ = read_envelope(RAW / "timeline" / f"{mid}.json.gz")
            out.write(json.dumps({"id": mid, "at": frame_events(timeline, match)}, separators=(",", ":")) + "\n")
            written += 1
            if written % 5000 == 0:
                log(f"{i}/{len(wanted)} written={written} {time.time() - start:.0f}s")
    tmp.replace(out_path)
    report = {"minutes": MINUTES, "written": written,
              "games_v1_sha256": hashlib.sha256(games_path.read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(), "seconds": round(time.time() - start, 1)}
    out_path.with_name("event_frames_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
