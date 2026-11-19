"""Stream admitted raw envelopes into compact per-game draft records.

Reads only IDs listed in a verified admission manifest. Player identities are
salted BLAKE2b hashes; raw PUUIDs never leave this function.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import secrets
import time
from pathlib import Path

from .admission import RAW, check_sidecar

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
FRAME_MIN = 12


def load_salt(path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(16), encoding="ascii")
    return bytes.fromhex(path.read_text(encoding="ascii").strip())


def player_hash(puuid: str, salt: bytes) -> str:
    return hashlib.blake2b(puuid.encode(), key=salt, digest_size=8).hexdigest()


def assigned(value: str | None) -> str | None:
    """selectedRolePreferences 'TOP.SECONDARY.MIDDLE.TOP' -> 'SECONDARY' (PRIMARY/SECONDARY/AUTOFILL)."""
    if not value or "." not in value:
        return None
    return value.split(".")[1]


def read_envelope(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    env = json.loads(gzip.decompress(raw))
    if env.get("schema") != 1 or "payload" not in env:
        raise ValueError(f"unknown envelope {path.name}")
    return env["payload"], hashlib.sha256(raw).hexdigest()


def game_record(mid: str, match: dict, timeline: dict | None, salt: bytes) -> tuple[dict | None, str]:
    info = match["info"]
    if info.get("queueId") != 420 or info.get("mapId") != 11:
        return None, "queue"
    if info.get("endOfGameResult") not in (None, "GameComplete"):
        return None, "incomplete"
    parts = info["participants"]
    if len(parts) != 10:
        return None, "participants"
    if any(p.get("gameEndedInEarlySurrender") for p in parts) or info["gameDuration"] < 600:
        return None, "remake"
    slots = {}
    for p in parts:
        key = (0 if p["teamId"] == 100 else 1, p.get("teamPosition"))
        if key[1] not in ROLES or key in slots:
            return None, "roles"
        slots[key] = p
    wins = {t["teamId"]: bool(t["win"]) for t in info["teams"]}
    if wins.get(100) == wins.get(200):
        return None, "winner"
    frame = None
    if timeline is not None:
        frames = timeline["info"]["frames"]
        if len(frames) > FRAME_MIN:
            frame = frames[FRAME_MIN]["participantFrames"]
    bans = [[], []]
    for t in info["teams"]:
        side = 0 if t["teamId"] == 100 else 1
        bans[side] = [b["championId"] for b in sorted(t.get("bans", []), key=lambda b: b["pickTurn"])]
    players = []
    for side in (0, 1):
        for role in ROLES:
            p = slots[(side, role)]
            pf = frame.get(str(p["participantId"])) if frame else None
            players.append({
                "side": side, "role": role, "champ": p["championId"],
                "pl": player_hash(p["puuid"], salt),
                "asg": assigned(p.get("selectedRolePreferences")),
                "pdmg": p["physicalDamageDealtToChampions"], "mdmg": p["magicDamageDealtToChampions"],
                "tdmg": p["trueDamageDealtToChampions"], "taken": p["totalDamageTaken"],
                "g12": pf["totalGold"] if pf else None, "xp12": pf["xp"] if pf else None,
                "cs12": (pf["minionsKilled"] + pf["jungleMinionsKilled"]) if pf else None,
            })
    version = info["gameVersion"].split(".")
    return {
        "id": mid, "platform": info["platformId"], "t": info["gameCreation"],
        "patch": f"{version[0]}.{version[1]}", "dur": info["gameDuration"],
        "blue_win": int(wins[100]), "bans": bans, "p": players, "tl": frame is not None,
    }, "ok"


def extract(admission_path: Path, out_path: Path, salt_path: Path, log=print) -> dict:
    digest = check_sidecar(admission_path)
    manifest = json.loads(admission_path.read_text(encoding="utf-8"))
    salt = load_salt(salt_path)
    counts: dict[str, int] = {}
    sources = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial")
    start = time.time()
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for i, mid in enumerate(manifest["admitted"]):
            match, msha = read_envelope(RAW / "match" / f"{mid}.json.gz")
            tl_path = RAW / "timeline" / f"{mid}.json.gz"
            timeline, tsha = (read_envelope(tl_path) if tl_path.exists() else (None, None))
            record, status = game_record(mid, match, timeline, salt)
            counts[status] = counts.get(status, 0) + 1
            sources.append([mid, msha, tsha])
            if record is not None:
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
            if i % 5000 == 0:
                log(f"{i}/{len(manifest['admitted'])} {counts} {time.time() - start:.0f}s")
    tmp.replace(out_path)
    src = out_path.with_name(out_path.stem.replace(".jsonl", "") + "_sources.json")
    src.write_text(json.dumps(sources), encoding="utf-8")
    report = {"admission_sha256": digest, "counts": counts, "seconds": round(time.time() - start, 1),
              "output": str(out_path), "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
              "sources_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    out_path.with_name("extract_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
