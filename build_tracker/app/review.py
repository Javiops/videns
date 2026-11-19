"""Read-only Match-V5 review. No model scoring or outcome-derived advice.

Positions and CS are minute-sampled Riot timeline observations, not replay
tracking. Inventory between observations is reconstructed from item events;
the final board is taken from the match DTO. Uncertain undo paths are marked.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, defaultdict

from app.ddragon import DDRAGON_VERSIONS, DataDragon, load_json
from app.config import CACHE_DIR, PATCH_DATA_VERSIONS
from app.inventory import InventoryReplay, SUPPORT_ATLAS
from app.profile import ProfileError, REGIONS, _client_lock, _get_client, _runes, _summoner_spells
from app.riot import RiotError
from app.shop_econ import combine_cost

MATCH_ID = re.compile(r"^[A-Z0-9]+_[0-9]{6,15}$")
ITEM_TYPES = {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO"}
ROLE_ORDER = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "UTILITY": 4}
# Nominal Rift structure positions, checked against Match-V5 BUILDING_KILL
# positions in KR_8386011226, KR_8385951341 and KR_8385882749. Nexus
# centres are map-layout approximations; Riot's GAME_END has no position.
STRUCTURE_LAYOUT = (
    (100, "tower", "TOP", "OUTER", 981, 10441),
    (100, "tower", "TOP", "INNER", 1512, 6699),
    (100, "tower", "TOP", "BASE", 1169, 4287),
    (100, "tower", "MIDDLE", "OUTER", 5846, 6396),
    (100, "tower", "MIDDLE", "INNER", 5048, 4812),
    (100, "tower", "MIDDLE", "BASE", 3651, 3696),
    (100, "tower", "BOTTOM", "OUTER", 10504, 1029),
    (100, "tower", "BOTTOM", "INNER", 6919, 1483),
    (100, "tower", "BOTTOM", "BASE", 4281, 1253),
    (100, "tower", "TOP", "NEXUS", 1748, 2270),
    (100, "tower", "BOTTOM", "NEXUS", 2177, 1807),
    (200, "tower", "TOP", "OUTER", 4318, 13875),
    (200, "tower", "TOP", "INNER", 7943, 13411),
    (200, "tower", "TOP", "BASE", 10481, 13650),
    (200, "tower", "MIDDLE", "OUTER", 8955, 8510),
    (200, "tower", "MIDDLE", "INNER", 9767, 10113),
    (200, "tower", "MIDDLE", "BASE", 11134, 11207),
    (200, "tower", "BOTTOM", "OUTER", 13866, 4505),
    (200, "tower", "BOTTOM", "INNER", 13327, 8226),
    (200, "tower", "BOTTOM", "BASE", 13624, 10572),
    (200, "tower", "TOP", "NEXUS", 12611, 13084),
    (200, "tower", "BOTTOM", "NEXUS", 13052, 12612),
    (100, "inhibitor", "TOP", "", 1171, 3571),
    (100, "inhibitor", "MIDDLE", "", 3203, 3208),
    (100, "inhibitor", "BOTTOM", "", 3452, 1236),
    (200, "inhibitor", "TOP", "", 11261, 13676),
    (200, "inhibitor", "MIDDLE", "", 11598, 11667),
    (200, "inhibitor", "BOTTOM", "", 13604, 11316),
    (100, "nexus", "BASE", "", 1300, 1300),
    (200, "nexus", "BASE", "", 13600, 13600),
)
_cache: dict[tuple[str, int], tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def _version_for(game_version: str) -> str:
    patch = ".".join(game_version.split(".")[:2])
    if patch in PATCH_DATA_VERSIONS:
        return PATCH_DATA_VERSIONS[patch]
    versions = load_json(CACHE_DIR / "versions.json", DDRAGON_VERSIONS)
    version = next((v for v in versions if v.startswith(patch + ".")), None)
    if version is None:
        raise ProfileError(f"No matching Data Dragon assets for patch {patch}.", 422)
    return version


def _item_rows(ids: list[int], version: str, dragon: DataDragon) -> list[dict]:
    return [
        {"id": item, "name": dragon.item_name(item),
         "icon": f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{item}.png"}
        for item in ids if item
    ]


def _structures(raw_events: list[dict], participants: list[dict], duration_ms: int) -> list[dict]:
    structures = [
        {"key": f"{team}:{kind}:{lane}:{tier}", "team_id": team, "kind": kind,
         "lane": lane, "tier": tier, "position": {"x": x, "y": y},
         "destroyed_at": None, "down_intervals": []}
        for team, kind, lane, tier, x, y in STRUCTURE_LAYOUT
    ]
    for event in raw_events:
        if event.get("type") != "BUILDING_KILL":
            continue
        team = int(event.get("teamId") or 0)
        kind = "inhibitor" if event.get("buildingType") == "INHIBITOR_BUILDING" else "tower"
        lane = {"TOP": "TOP", "MID": "MIDDLE", "BOT": "BOTTOM"}.get(
            (event.get("laneType") or "").replace("_LANE", ""), "")
        tier = (event.get("towerType") or "").replace("_TURRET", "")
        candidates = [s for s in structures if s["team_id"] == team and s["kind"] == kind
                      and (s["lane"] == lane or tier == "NEXUS")
                      and (kind == "inhibitor" or s["tier"] == tier)
                      and (kind == "inhibitor" or s["destroyed_at"] is None)]
        point = event.get("position") or {}
        if not candidates or "x" not in point or "y" not in point:
            continue
        chosen = min(candidates, key=lambda s: (
            (s["position"]["x"] - point["x"]) ** 2 +
            (s["position"]["y"] - point["y"]) ** 2))
        ts = int(event.get("timestamp") or 0)
        if kind == "inhibitor":
            chosen["down_intervals"].append({"from": ts, "to": ts + 300_000})
        else:
            chosen["destroyed_at"] = ts
    if not any(p.get("gameEndedInSurrender") or p.get("gameEndedInEarlySurrender")
               for p in participants):
        winner = next((int(p.get("teamId") or 0) for p in participants if p.get("win")), 0)
        for structure in structures:
            if winner in (100, 200) and structure["kind"] == "nexus" and structure["team_id"] != winner:
                structure["destroyed_at"] = duration_ms
    return structures


def _event_row(raw: dict, by_pid: dict[int, dict]) -> dict | None:
    kind = raw.get("type")
    ts = int(raw.get("timestamp") or 0)
    point = raw.get("position") or {}
    position = {"x": point["x"], "y": point["y"]} if "x" in point and "y" in point else None
    if kind == "CHAMPION_KILL":
        victim = by_pid.get(int(raw.get("victimId") or 0))
        killer = by_pid.get(int(raw.get("killerId") or 0))
        if not victim:
            return None
        return {"ts": ts, "kind": "kill", "title": "Champion kill",
                "detail": f"{killer['champion'] if killer else 'Execution'} → {victim['champion']}",
                "team_id": killer["team_id"] if killer else 0,
                "actors": [int(raw.get("killerId") or 0), int(raw.get("victimId") or 0)],
                "assists": [int(i) for i in raw.get("assistingParticipantIds") or []],
                "position": position}
    if kind == "BUILDING_KILL":
        building = raw.get("buildingType") or ""
        lane = (raw.get("laneType") or "").replace("_LANE", "").title()
        tower = (raw.get("towerType") or "").replace("_TURRET", "").replace("_", " ").title()
        name = "Inhibitor" if building == "INHIBITOR_BUILDING" else "Tower"
        detail = " · ".join(part for part in (lane, tower) if part)
        return {"ts": ts, "kind": "base", "title": f"{name} destroyed",
                "detail": detail, "team_id": 200 if raw.get("teamId") == 100 else 100,
                "destroyed_team_id": int(raw.get("teamId") or 0),
                "position": position, "actors": [int(raw.get("killerId") or 0)]}
    if kind == "ELITE_MONSTER_KILL":
        monster = (raw.get("monsterType") or "Objective").replace("_", " ").title()
        subtype = (raw.get("monsterSubType") or "").replace("_", " ").title()
        return {"ts": ts, "kind": "objective", "title": monster,
                "detail": subtype, "team_id": int(raw.get("killerTeamId") or 0),
                "position": position, "actors": [int(raw.get("killerId") or 0)]}
    if kind == "GAME_END":
        return {"ts": ts, "kind": "base", "title": "Game ended",
                "detail": "Nexus destroyed / surrender", "team_id": int(raw.get("winningTeam") or 0),
                "position": None, "actors": []}
    return None


def shape_review(match: dict, timeline: dict, focus_participant_id: int | None = None) -> dict:
    info = match.get("info") or {}
    participants = info.get("participants") or []
    frames = (timeline.get("info") or {}).get("frames") or []
    match_id = (match.get("metadata") or {}).get("matchId")
    if (not match_id or (timeline.get("metadata") or {}).get("matchId") != match_id
            or int(info.get("mapId") or 0) != 11 or len(participants) != 10 or not frames):
        raise ProfileError("Review is available for complete Summoner's Rift matches only.", 422)
    version = _version_for(info.get("gameVersion") or "")
    dragon = DataDragon(version)
    spells = _summoner_spells(version)
    runes = _runes(version)
    by_pid = {int(p["participantId"]): p for p in participants}
    focus = int(focus_participant_id or 0)
    if focus and focus not in by_pid:
        raise ProfileError("Participant not found in this match.", 404)
    order = sorted(by_pid, key=lambda pid: (
        by_pid[pid].get("teamId") != 100,
        ROLE_ORDER.get(by_pid[pid].get("teamPosition") or "", 9), pid))
    players = []
    for pid in order:
        p = by_pid[pid]
        cid = int(p.get("championId") or 0)
        art_id = dragon.champion_id_name(cid)
        styles = (p.get("perks") or {}).get("styles") or []
        primary = (styles[0].get("selections") or []) if styles else []
        keystone_id = int(primary[0].get("perk") or 0) if primary else 0
        secondary_id = int(styles[1].get("style") or 0) if len(styles) > 1 else 0
        players.append({
            "participant_id": pid, "team_id": int(p.get("teamId") or 0),
            "name": p.get("riotIdGameName") or p.get("summonerName") or dragon.champion_name(cid),
            "tag": p.get("riotIdTagline") or "", "role": p.get("teamPosition") or "",
            "champion": dragon.champion_name(cid),
            "icon": f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{art_id}.png",
            "spells": [spells.get(int(p.get(f"summoner{n}Id") or 0)) for n in (1, 2)],
            "keystone": runes.get(keystone_id), "secondary_tree": runes.get(secondary_id),
            "final_items": _item_rows([int(p.get(f"item{i}") or 0) for i in range(7)], version, dragon),
            "win": bool(p.get("win")),
        })

    raw_events = sorted(
        (e for frame in frames for e in frame.get("events") or []),
        key=lambda e: int(e.get("timestamp") or 0),
    )
    actor_index = {
        pid: {"champion": dragon.champion_name(int(p.get("championId") or 0)),
              "team_id": int(p.get("teamId") or 0)} for pid, p in by_pid.items()
    }
    events = [r for e in raw_events if (r := _event_row(e, actor_index))]
    duration_ms = max(int(info.get("gameDuration") or 0) * 1000,
                      int(frames[-1].get("timestamp") or 0))
    if not any(event["title"] == "Game ended" for event in events):
        winner = next((int(p.get("teamId") or 0) for p in participants if p.get("win")), 0)
        events.append({"ts": duration_ms, "kind": "base", "title": "Game ended",
                       "detail": "Result from match record", "team_id": winner,
                       "position": None, "actors": []})
    structures = _structures(raw_events, participants, duration_ms)
    decisions = []
    decisions_status = "unavailable"
    patch = ".".join((info.get("gameVersion") or "").split(".")[:2])
    if patch in PATCH_DATA_VERSIONS:
        try:
            from app.reconstruct import reconstruct_game

            _game, reconstructed = reconstruct_game(match, timeline, dragon)
            for decision in reconstructed:
                if decision.get("type") != "shop":
                    continue
                decisions.append({
                    "ts": int(decision["ts"]),
                    "ts_end": int(decision.get("ts_end") or decision["ts"]),
                    "participant_id": int(decision["participant_id"]),
                    "champion_id": int(decision["champion_id"]),
                    "role": decision.get("team_position") or "",
                    "gold_est": decision.get("gold_est"),
                    "gold_est_version": decision.get("gold_est_version"),
                    "bot_quest_complete": decision.get("bot_quest_complete"),
                    "level": decision.get("level"),
                    "inventory": [int(i["item_id"]) for i in decision.get("decision_inventory") or []],
                    "bought": [int(i["item_id"]) for i in decision.get("bought") or []],
                    "modeled_bought": [int(i["item_id"]) for i in decision.get("label_bought") or []],
                })
            decisions_status = "reconstructed"
        except (KeyError, ValueError, TypeError):
            # Review still shows source events; it must not fabricate a shop
            # decision when a patch-specific reconstruction gate rejects it.
            decisions = []
    observations = []
    for frame in frames:
        state = {}
        for pid_s, value in (frame.get("participantFrames") or {}).items():
            pid = int(pid_s)
            if pid not in by_pid:
                continue
            pos = value.get("position") or {}
            stats = value.get("championStats") or {}
            health = stats.get("health")
            health_max = stats.get("healthMax")
            state[str(pid)] = {
                "position": {"x": pos["x"], "y": pos["y"]} if "x" in pos and "y" in pos else None,
                # Missing health is unknown. Zero is an observed dead/down
                # frame and must survive serialization as zero.
                "health": health if isinstance(health, (int, float)) else None,
                "health_max": health_max if isinstance(health_max, (int, float)) else None,
                "level": int(value.get("level") or 0),
                "cs": int(value.get("minionsKilled") or 0) + int(value.get("jungleMinionsKilled") or 0),
            }
        observations.append({"ts": int(frame.get("timestamp") or 0), "players": state})

    inventories = {pid: ([SUPPORT_ATLAS] if by_pid[pid].get("teamPosition") == "UTILITY" else [])
                   for pid in by_pid}
    replay = InventoryReplay(inventories, dragon)
    inventory_states = [{"ts": 0, "players": {str(pid): list(items) for pid, items in inventories.items()}}]
    item_batches: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for raw in raw_events:
        pid = int(raw.get("participantId") or 0)
        if raw.get("type") in ITEM_TYPES and pid in by_pid:
            item_batches[(int(raw.get("timestamp") or 0), pid)].append(raw)
        elif raw.get("type") == "WARD_PLACED" and raw.get("wardType") == "CONTROL_WARD":
            owner = int(raw.get("creatorId") or 0)
            if owner in by_pid:
                item_batches[(int(raw.get("timestamp") or 0), owner)].append(raw)
    uncertain_from = None
    purchase_stack: dict[int, dict[int, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    shops = []
    for (ts, pid), batch in sorted(item_batches.items()):
        before = list(inventories[pid])
        modeled = Counter(before)
        undo_restores = {}
        for item_event in batch:
            kind = item_event.get("type")
            item_id = int(item_event.get("itemId") or 0)
            if kind == "ITEM_PURCHASED" and item_id:
                pre_purchase = modeled.copy()
                combine_cost(item_id, modeled, dragon, consume=True)
                consumed = list((pre_purchase - modeled).elements())
                purchase_stack[pid][item_id].append(consumed)
                modeled[item_id] += 1
            elif kind == "ITEM_SOLD" and modeled[item_id] > 0:
                modeled[item_id] -= 1
            elif kind == "ITEM_UNDO":
                before_id = int(item_event.get("beforeId") or 0)
                after_id = int(item_event.get("afterId") or 0)
                stack = purchase_stack[pid][before_id]
                matched = bool(stack)
                restored = list(stack.pop()) if matched else []
                if after_id and after_id not in restored:
                    restored.append(after_id)
                undo_restores[id(item_event)] = restored
                if before_id and not matched and not after_id:
                    uncertain_from = ts if uncertain_from is None else uncertain_from
                if modeled[before_id] > 0:
                    modeled[before_id] -= 1
                modeled.update(restored)
            # ITEM_DESTROYED can precede its same-batch purchase. Recipe-aware
            # combine_cost consumes the pre-batch components; don't double-remove.
        replay.apply_batch(pid, ts, batch, undo_restores)
        for event in batch:
            if event.get("type") == "WARD_PLACED" and event.get("wardType") == "CONTROL_WARD":
                replay.consume_control_ward(pid)
        after = list(inventories[pid])
        inventory_states.append({"ts": ts, "players": {str(i): list(v) for i, v in inventories.items()}})
        bought = [int(e.get("itemId") or 0) for e in batch if e.get("type") == "ITEM_PURCHASED"]
        if bought:
            shops.append({"ts": ts, "participant_id": pid,
                          "before": before, "after": after, "bought": bought})
    final = {str(pid): [int(by_pid[pid].get(f"item{i}") or 0) for i in range(7)
                        if int(by_pid[pid].get(f"item{i}") or 0)] for pid in by_pid}
    inventory_states.append({"ts": duration_ms, "players": final, "source": "match_final"})
    inventory_states.sort(key=lambda state: state["ts"])
    used_items = {item for state in inventory_states for items in state["players"].values() for item in items}
    used_items.update(item for shop in shops for item in shop["bought"])
    return {
        "match_id": match_id, "duration_ms": duration_ms, "queue_id": int(info.get("queueId") or 0),
        "patch": ".".join((info.get("gameVersion") or "").split(".")[:2]),
        "ddragon_version": version, "focus_participant_id": focus or order[0],
        "players": players, "events": events, "observations": observations,
        "inventory_states": inventory_states, "shops": shops,
        "decisions": decisions, "decisions_status": decisions_status,
        "structures": structures,
        "map_bounds": {"min_x": -120, "min_y": -120, "max_x": 14870, "max_y": 14980},
        "inventory_uncertain_from_ms": uncertain_from,
        # Data Dragon's map11 asset is not published under every current patch.
        # This verified stable map artwork is presentation only; positions and
        # structure state still come from the reviewed match timeline.
        "map_image": "https://ddragon.leagueoflegends.com/cdn/10.18.1/img/map/map11.png",
        "item_icon_base": f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/",
        "item_names": {str(item): dragon.item_name(item) for item in used_items},
        "item_meta": {str(item): {
            "stacks": max(1, int((dragon.item(item) or {}).get("stacks") or 1)),
            "trinket": "Trinket" in (dragon.item(item) or {}).get("tags", []),
        } for item in used_items},
        "build_comparison": {
            "status": "unavailable",
            "message": "No promoted comparison policy is attached to on-demand reviews.",
        },
        "sources": {"positions": "Match-V5 participant frames (about once per minute)",
                    "life_state": "Champion-kill events and observed frame health; no inferred respawn timer",
                    "items": "Match-V5 item events; final inventory from match DTO",
                    "events": "Match-V5 timeline"},
    }


def compile_review(match_id: str, region_key: str = "euw", focus_participant_id: int | None = None) -> dict:
    if not MATCH_ID.fullmatch(match_id or ""):
        raise ProfileError("Invalid match ID.")
    region = REGIONS.get((region_key or "").lower())
    if not region or not match_id.startswith(region["platform"].upper() + "_"):
        raise ProfileError("Match ID does not belong to the selected region.")
    focus = int(focus_participant_id or 0)
    key = (match_id, focus)
    with _cache_lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < 180:
            return cached[1]
    with _client_lock:
        try:
            client = _get_client()
            match = client.match(region["match"], match_id)
            timeline = client.timeline(region["match"], match_id)
        except RiotError as exc:
            raise ProfileError(str(exc), exc.status or 502) from exc
    if not match or not timeline:
        raise ProfileError("Match or timeline not found.", 404)
    result = shape_review(match, timeline, focus)
    with _cache_lock:
        if len(_cache) >= 128:
            oldest = min(_cache, key=lambda existing: _cache[existing][0])
            _cache.pop(oldest, None)
        _cache[key] = (time.monotonic(), result)
    return result
