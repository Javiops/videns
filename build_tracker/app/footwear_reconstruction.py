"""Opt-in, patch-pinned replay for the recovered 16.17 development corpus.

This module is never imported by production reconstruction. Activate it only
through scripts/footwear_runtime.py in an isolated process. Its timer is a
source-consistent delivery hypothesis; admission additionally requires the
independent source checks and the frozen grant-boundary decision mask.
"""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

from app.config import SHOP_IDLE_MS
from app.ddragon import DataDragon, dragon_for_patch
from app.config import PATCH_DATA_VERSIONS
from app.inventory import SUPPORT_ATLAS, SUPPORT_BOUNTY, SUPPORT_LINE
from footwear_economics import FootwearDataDragon, combine_cost, net_spent
from footwear_timer_inventory import FootwearTimerInventoryReplay, INVENTORY_VERSION
from footwear_timer import TimerFootwearScheduler, physical_slot_state
from magical_footwear import _rune_selected

SHOP_ACTIONS = {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_UNDO"}
ITEM_EVENTS = SHOP_ACTIONS | {"ITEM_DESTROYED"}

# A death is an observed fountain visit: if the player respawns holding real
# gold and buys nothing within this window, that is a deliberate save decision
# — the action the purchase-only dataset can never show otherwise.
SAVE_WINDOW_MS = 90_000
SAVE_MIN_GOLD = 400
SAVE_ENDGAME_MS = 120_000

# Causal (prequential) gold estimate at a shop visit. The stored frame gold is
# up to 60s stale, so the offline budget sits in a different regime from the
# exact gold the live client reports; `gold_est` narrows that gap using ONLY
# information that existed before the decision.
#
# It is an APPROXIMATION, never "exact arrival gold". Match-V5 does not contain
# a player's gold at an arbitrary timestamp, and every reconstruction that
# recovers it exactly must read either the visit's own spend (label leak) or a
# later frame (temporal leak). Version "leaky-v1" (2026-09-08, superseded) did
# the latter: it prorated a residual computed from the frame AFTER the visit,
# making the feature depend on post-purchase income. Do not reintroduce it.
GOLD_EST_VERSION = "prequential-v3"
RECONSTRUCTION_VERSION = "decision-start-footwear-v4"
FOOTWEAR_STATE_VERSION = "footwear-timer-slots-v1"
SLOT_STATE_VERSION = "role-slots-v1"
PASSIVE_GOLD_PER_S = 2.04
PASSIVE_START_MS = 110_000
# The last seconds before a shop visit are spent walking or recalling, not
# earning; the residual forecast is not credited for that stretch.
SHOP_DEAD_TIME_MS = 12_000

# Match-v5 never PURCHASES the support gold line. It is granted at start, then
# only ITEM_DESTROYED fires as it upgrades: Atlas → Compass → Bounty → finished.


def arrival_from_parts(leftover, before, bought, consumed, dragon):
    return max(0, int(leftover or 0) + net_spent(before, bought, consumed, dragon))


def patch_from_version(game_version: str) -> str:
    parts = (game_version or "").split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return game_version or "unknown"


def perk_fields(p: dict) -> dict:
    """Keystone rune, secondary tree, and summoner spells from a match-v5
    participant DTO. Zeros when the DTO predates rune storage (backfillable)."""
    styles = ((p.get("perks") or {}).get("styles")) or []
    keystone = 0
    sub_style = 0
    if styles:
        selections = styles[0].get("selections") or []
        keystone = int((selections[0] or {}).get("perk") or 0) if selections else 0
        if len(styles) > 1:
            sub_style = int(styles[1].get("style") or 0)
    return {
        "keystone_id": keystone,
        "sub_style": sub_style,
        "summ1": int(p.get("summoner1Id") or 0),
        "summ2": int(p.get("summoner2Id") or 0),
    }


def reconstruct_game(
    match: dict, timeline: dict, dragon: DataDragon, gold_debug: dict | None = None
) -> tuple[dict, list[dict]]:
    info = match["info"]
    patch = patch_from_version(info.get("gameVersion") or "")
    if patch != "16.17":
        raise ValueError("Footwear reconstruction is reviewed only for patch 16.17")
    if any(e.get("type") == "ITEM_PURCHASED" and e.get("itemId") == 2422
           for f in timeline.get("info", {}).get("frames", []) for e in f.get("events", [])):
        raise ValueError("Explicit Magical Footwear grant requires its own reviewed handler")
    expected_version = PATCH_DATA_VERSIONS.get(patch)
    if expected_version is None:
        raise ValueError(f"Cannot reconstruct unreviewed patch {patch!r}")
    if dragon.version != expected_version:
        dragon = dragon_for_patch(patch)
    if not isinstance(dragon, FootwearDataDragon):
        dragon = FootwearDataDragon(dragon)
    participants = []
    by_pid: dict[int, dict] = {}
    win_team = 100
    for p in info["participants"]:
        row = {
            "participant_id": p["participantId"],
            "puuid": p["puuid"],
            "champion_id": p["championId"],
            "champion_name": p.get("championName") or dragon.champion_id_name(p["championId"]),
            "team_id": p["teamId"],
            "team_position": p.get("teamPosition") or p.get("individualPosition") or "",
            "riot_id": f"{p.get('riotIdGameName') or ''}#{p.get('riotIdTagline') or ''}".strip("#"),
            "win": bool(p.get("win")),
            "kills": p.get("kills", 0),
            "deaths": p.get("deaths", 0),
            "assists": p.get("assists", 0),
            **perk_fields(p),
        }
        participants.append(row)
        by_pid[row["participant_id"]] = row
        if row["win"]:
            win_team = row["team_id"]

    game = {
        "match_id": match["metadata"]["matchId"],
        "game_creation": info.get("gameCreation") or 0,
        "game_duration": info.get("gameDuration") or 0,
        "game_version": info.get("gameVersion") or "",
        "patch": patch_from_version(info.get("gameVersion") or ""),
        "queue_id": info.get("queueId") or 0,
        "win_team": win_team,
        "participants_json": json.dumps(participants),
        "source": "reconstructed",
    }

    pid_by_key = {str(p["participant_id"]): p for p in participants}
    inventories: dict[int, list[int]] = {p["participant_id"]: [] for p in participants}
    for p in info["participants"]:
        pid = p["participantId"]
        role = p.get("teamPosition") or p.get("individualPosition") or ""
        if role == "UTILITY":
            inventories[pid].append(SUPPORT_ATLAS)
    inventory_replay = FootwearTimerInventoryReplay(
        inventories, dragon, role_for=lambda owner: by_pid[owner]["team_position"])
    footwear_batches = []
    footwear_inventory_failures = []
    footwear_inventory_checks = 0
    footwear_timer = None
    # Record only observed completion markers. InventoryReplay may already have
    # applied a same-timestamp marker when another event starts a visit, but
    # that marker is not strictly before the decision and cannot unlock a slot.
    bot_quest_completed_at: dict[int, int] = {}
    kda = {p["participant_id"]: {"kills": 0, "deaths": 0, "assists": 0} for p in participants}
    objectives = {
        100: {"towers": 0, "dragons": 0, "barons": 0, "heralds": 0, "voidgrubs": 0},
        200: {"towers": 0, "dragons": 0, "barons": 0, "heralds": 0, "voidgrubs": 0},
    }
    last_frames: dict[int, dict] = {}
    events: list[dict] = []
    open_visits: dict[int, dict] = {}
    purchases_by_pid: dict[int, list[int]] = {}
    death_snaps: list[dict] = []
    # forward gold walk state: per-pid frame checkpoints (ts, current, total),
    # timed gold events (bounties in, spends out), and paid prices for undos
    frame_series: dict[int, list[tuple[int, int, int]]] = {}
    gold_events: dict[int, list[tuple[int, float, str]]] = {}
    paid_stack: dict[int, dict[int, list[tuple[int, list[int]]]]] = {}

    def record_batch_spend(owner: int, ts: int, batch: list[dict]) -> dict:
        """Gold leaving (or refunding to) the wallet, priced against the
        inventory at BATCH start. A combine's ITEM_DESTROYED events can precede
        its ITEM_PURCHASED inside the batch, so pricing per-event after applying
        destroys loses the combine discount (measured: wallet-identity residual
        +232 mean); instead combine_cost(consume=True) models the consumption
        itself and destroy events are ignored for pricing."""
        inv_c = Counter(inventories.get(owner) or [])
        undo_restores = {}
        for event in batch:
            etype = event.get("type")
            if etype == "ITEM_PURCHASED":
                item_id = event.get("itemId") or 0
                if not item_id or item_id in SUPPORT_LINE:
                    continue
                before_combine = inv_c.copy()
                paid = combine_cost(item_id, inv_c, dragon, consume=True)
                components = list((before_combine - inv_c).elements())
                inv_c[item_id] += 1
                paid_stack.setdefault(owner, {}).setdefault(item_id, []).append((paid, components))
                gold_events.setdefault(owner, []).append((ts, float(paid), "spend"))
            elif etype == "ITEM_SOLD":
                item_id = event.get("itemId") or 0
                sell = dragon.gold_block(item_id)["sell"] if item_id else 0
                if inv_c[item_id] > 0:
                    inv_c[item_id] -= 1
                gold_events.setdefault(owner, []).append((ts, -float(sell), "spend"))
            elif etype == "ITEM_UNDO":
                before = event.get("beforeId") or 0
                after = event.get("afterId") or 0
                # The API reports the transaction delta, including negative
                # goldGain for an undone sale. Recipe prices cannot recover it.
                reported_gain = event.get("goldGain")
                stack = (paid_stack.get(owner) or {}).get(before) or []
                transaction = stack.pop() if before and stack else None
                restored = list(transaction[1]) if transaction else []
                if after and after not in restored:
                    restored.append(after)
                undo_restores[id(event)] = restored
                held_before = before if inv_c[before] else inventory_replay.effective_purchase(owner, before)
                if before and inv_c[held_before] > 0:
                    inv_c[held_before] -= 1
                inv_c.update(restored)
                if isinstance(reported_gain, (int, float)):
                    gold_events.setdefault(owner, []).append((ts, -float(reported_gain), "spend"))
                    continue
                if before:
                    refund = transaction[0] if transaction else dragon.gold_block(before)["base"]
                    gold_events.setdefault(owner, []).append((ts, -float(refund), "spend"))
        return undo_restores

    def close_visit(pid: int) -> None:
        ov = open_visits.pop(pid, None)
        if not ov:
            return
        before = ov["inventory_before"]
        after = ov["inventory_after"]
        before_build = [i for i in before if not dragon.classify(i)["skip"]]
        after_build = [i for i in after if not dragon.classify(i)["skip"]]
        if before_build == after_build:
            return
        bought, consumed = _diff(before, after)
        # Net gains can include granted support upgrades or an undone sale.
        # Only gains backed by purchases in this visit are imitation targets.
        purchased = Counter(inventory_replay.effective_purchase(pid, i) for i in ov["purchased"].elements())
        label_bought = list((Counter(bought) & purchased).elements())
        owner = by_pid[pid]
        stats = ov["start_frames"].get(pid) or {}
        leftover = int(stats.get("currentGold") or 0)
        # Free system grants are context changes, never purchase expenditure.
        arrival = arrival_from_parts(leftover, before, [i for i in bought if i != 2422], consumed, dragon)
        start_kda = ov["start_kda"][pid]
        # Decision state: opening sales (strictly before the first purchase)
        # free their slot and credit their proceeds. Admissible because the
        # quantity is measurable from the event stream truncated strictly
        # before the first ITEM_PURCHASED, obtainable from live client state,
        # and not a function of the purchase multiset. Gold comes from the
        # priced spend stream (net, so an undone sale cancels itself).
        first_buy = ov.get("first_purchase_ts")
        opening = +ov.get("opening_sales", Counter())
        opening_sale_gold = 0.0
        decision_inventory = list(before)
        applied_sales: list[int] = []
        if first_buy is not None and opening and ov.get("opening_sales_certain", True):
            # By construction only whole pre-purchase batches land here, so a
            # positive amount can only be an undone sale taking its refund
            # back; the net sum cancels the pair. If accounting nets to zero
            # or negative while the item counter is non-empty, the two
            # disagree — fail safe to the pre-sale state on both fields.
            proceeds = -sum(amount for ets, amount, kind in (gold_events.get(pid) or [])
                            if kind == "spend" and ov["ts_start"] <= ets < first_buy)
            if proceeds > 0:
                opening_sale_gold = proceeds
                remaining = Counter(opening)
                decision_inventory = []
                for item in before:
                    if remaining.get(item, 0) > 0:
                        remaining[item] -= 1
                        applied_sales.append(item)
                        continue
                    decision_inventory.append(item)
        events.append(
            {
                "type": "shop",
                "reconstruction_version": RECONSTRUCTION_VERSION,
                "ts": ov["ts_start"],
                "ts_end": ov["ts_end"],
                "puuid": owner["puuid"],
                "participant_id": pid,
                "champion_name": owner["champion_name"],
                "champion_id": owner["champion_id"],
                "team_id": owner["team_id"],
                "team_position": owner["team_position"],
                "bot_quest_complete": ov["start_bot_quest_complete"][pid],
                "riot_id": owner["riot_id"],
                "gold": arrival,
                "gold_left": leftover,
                "level": stats.get("level"),
                "cs": (stats.get("minionsKilled") or 0) + (stats.get("jungleMinionsKilled") or 0),
                "kills": start_kda["kills"],
                "deaths": start_kda["deaths"],
                "assists": start_kda["assists"],
                "inventory_before": _items_payload(before, dragon),
                "inventory_after": _items_payload(after, dragon),
                "decision_inventory": _items_payload(decision_inventory, dragon),
                "opening_sales": sorted(applied_sales),
                "opening_sale_gold": int(round(opening_sale_gold)),
                "bought": _items_payload(bought, dragon),
                "label_bought": _items_payload(label_bought, dragon),
                "label_source_version": "purchase-net-v1",
                "purchase_actions": sorted(purchased.elements()),
                "consumed": _items_payload(consumed, dragon),
                "board": _board_snapshot(
                    participants,
                    ov["start_inventories"],
                    ov["start_frames"],
                    ov["start_kda"],
                    dragon,
                    pid,
                    ov["start_bot_quest_complete"],
                ),
                "score": deepcopy(ov["start_objectives"]),
            }
        )

    def maybe_close(ts: int) -> None:
        for pid, ov in list(open_visits.items()):
            if ts - ov["last_shop_ts"] > SHOP_IDLE_MS:
                close_visit(pid)

    def stamp_visit(pid: int, ts: int) -> None:
        ov = open_visits.get(pid)
        if not ov:
            return
        ov["ts_end"] = ts
        ov["inventory_after"] = list(inventories[pid])

    def slot_snapshot(ts: int) -> dict[int, bool]:
        return {pid: bot_quest_completed_at.get(pid, ts) < ts for pid in inventories}

    def start_visit(pid: int, ts: int) -> None:
        open_visits[pid] = {
            "ts_start": ts,
            "ts_end": ts,
            "last_shop_ts": ts,
            "purchased": Counter(),
            # Opening sales: ITEM_SOLD events strictly before the visit's first
            # ITEM_PURCHASED. They free a slot and credit gold the player
            # demonstrably had when deciding what to buy — the live client is
            # already in that post-sale state. ITEM_DESTROYED never counts: a
            # combine's destroys precede its purchase inside the same batch,
            # and counting them would reveal the label through the state.
            "first_purchase_ts": None,
            "opening_sales": Counter(),
            "opening_sales_certain": True,
            "inventory_before": list(inventories[pid]),
            "inventory_after": list(inventories[pid]),
            "start_inventories": {k: list(v) for k, v in inventories.items()},
            "start_frames": dict(last_frames),
            "start_kda": {k: dict(v) for k, v in kda.items()},
            "start_objectives": deepcopy(objectives),
            "start_bot_quest_complete": slot_snapshot(ts),
        }

    def handle_item_batch(owner: int, ts: int, batch: list[dict]) -> None:
        if any(e.get('itemId') == 2422 or e.get('beforeId') == 2422 or e.get('afterId') == 2422 for e in batch):
            footwear_batches.append({'participant_id':owner,'ts':ts,'before':list(inventories[owner]),
                                     'batch':deepcopy(batch)})
        if any(e.get("type") == "ITEM_DESTROYED" and e.get("itemId") == 1202 for e in batch):
            bot_quest_completed_at.setdefault(owner, ts)
        is_shop = any(e.get("type") in SHOP_ACTIONS for e in batch)
        if is_shop:
            if owner not in open_visits:
                start_visit(owner, ts)
            ov = open_visits[owner]
            if ov["first_purchase_ts"] is None:
                # Whole batches only: a batch containing the first purchase is
                # already inside the decision, so its sales are NOT opening
                # sales (this keeps the item counter and the [ts_start,
                # first_purchase_ts) gold window describing the same events).
                if any(e.get("type") == "ITEM_PURCHASED" and e.get("itemId") for e in batch):
                    ov["first_purchase_ts"] = ts
                else:
                    for event in batch:
                        etype = event.get("type")
                        if etype == "ITEM_SOLD" and event.get("itemId"):
                            ov["opening_sales"][int(event["itemId"])] += 1
                        elif etype == "ITEM_UNDO":
                            after = int(event.get("afterId") or 0)
                            if after and ov["opening_sales"][after] > 0:
                                ov["opening_sales"][after] -= 1
                            elif ov["opening_sales"]:
                                # An undo we cannot attribute: fail safe to the
                                # pre-sale state rather than free an uncertain slot.
                                ov["opening_sales_certain"] = False
            restores = record_batch_spend(owner, ts, batch)
            open_visits[owner]["purchased"].update(inventory_replay.apply_batch(owner, ts, batch, restores))
            open_visits[owner]["last_shop_ts"] = ts
            stamp_visit(owner, ts)
            return
        inventory_replay.apply_batch(owner, ts, batch)
        if owner in open_visits:
            stamp_visit(owner, ts)

    def check_physical_inventory(owner, ts, cause):
        nonlocal footwear_inventory_checks
        footwear_inventory_checks += 1
        shape = physical_slot_state(inventories[owner], dragon,
            role=by_pid[owner]['team_position'], bot_quest_complete=owner in inventory_replay.bot_complete)
        ward_overflow = any(inventories[owner].count(ward) > int((dragon.item(ward) or {}).get('stacks') or 1)
                            for ward in (2055,772043))
        if not shape['known'] or shape['regular_slots'] > 6 or shape['boot_copies'] > 1 or ward_overflow:
            footwear_inventory_failures.append({'participant_id':owner,'ts':ts,'cause':cause,
                'shape':shape,'items':list(inventories[owner]),'ward_overflow':ward_overflow})

    def grant_footwear(owner, ts, metadata):
        # Close stale shop windows at the grant's own time before updating
        # inventory. This never creates a shop or a purchase action.
        maybe_close(ts)
        inventories[owner].append(2422)
        if owner in open_visits:
            stamp_visit(owner, ts)
        check_physical_inventory(owner, ts, 'modeled_grant')

    footwear_timer = TimerFootwearScheduler(
        info['participants'], dragon, lambda owner: inventories[owner],
        lambda owner: by_pid[owner]['team_position'],
        lambda owner: owner in inventory_replay.bot_complete, grant_footwear)
    frames = timeline.get("info", {}).get("frames") or []
    precise_end = next((e['timestamp'] for f in frames for e in f.get('events',[]) if e.get('type') == 'GAME_END'),
                       int(info.get('gameDuration',0))*1000)
    explicit_ward_destroys = {(e.get("participantId"), e.get("timestamp")) for f in frames for e in f.get("events", [])
                             if e.get("type") == "ITEM_DESTROYED" and e.get("itemId") in (2055, 772043)}
    for frame in frames:
        raw_events = frame.get("events") or []
        i = 0
        while i < len(raw_events):
            event = raw_events[i]
            etype = event.get("type")
            ts = event.get("timestamp") or 0
            footwear_timer.advance_before(ts)
            maybe_close(ts)

            if etype in ITEM_EVENTS:
                owner = event.get("participantId")
                batch = [event]
                while i + 1 < len(raw_events):
                    nxt = raw_events[i + 1]
                    if (nxt.get("timestamp") or 0) != ts:
                        break
                    if nxt.get("participantId") != owner:
                        break
                    if nxt.get("type") not in ITEM_EVENTS:
                        break
                    batch.append(nxt)
                    i += 1
                if owner:
                    for entry in batch:
                        if entry.get("type") == "ITEM_PURCHASED":
                            purchases_by_pid.setdefault(owner, []).append(ts)
                    handle_item_batch(owner, ts, batch)
                    check_physical_inventory(owner, ts, 'raw_item_batch')
                    footwear_timer.observe_batch(ts, batch)
                i += 1
                continue

            if etype == "WARD_PLACED" and event.get("wardType") == "CONTROL_WARD":
                owner = event.get("creatorId")
                if (owner, ts) not in explicit_ward_destroys:
                    inventory_replay.consume_control_ward(owner)
                    if owner in open_visits:
                        stamp_visit(owner, ts)
                    check_physical_inventory(owner, ts, 'ward_consumption')
            elif etype == "CHAMPION_KILL":
                killer = event.get("killerId") or 0
                victim = event.get("victimId") or 0
                if killer in kda:
                    kda[killer]["kills"] += 1
                    bounty = float(event.get("bounty") or 0) + float(event.get("shutdownBounty") or 0)
                    if bounty > 0:
                        gold_events.setdefault(killer, []).append((ts, bounty, "bounty"))
                if victim in kda:
                    kda[victim]["deaths"] += 1
                for aid in event.get("assistingParticipantIds") or []:
                    if aid in kda:
                        kda[aid]["assists"] += 1
                if victim in kda:
                    death_snaps.append(
                        {
                            "pid": victim,
                            "ts": ts,
                            "inventories": {k: list(v) for k, v in inventories.items()},
                            "frames": dict(last_frames),
                            "kda": {k: dict(v) for k, v in kda.items()},
                            "objectives": deepcopy(objectives),
                            "bot_quest_complete": slot_snapshot(ts),
                        }
                    )

            elif etype == "BUILDING_KILL" and event.get("buildingType") == "TOWER_BUILDING":
                team = event.get("teamId")
                if team == 100:
                    objectives[200]["towers"] += 1
                elif team == 200:
                    objectives[100]["towers"] += 1

            elif etype == "ELITE_MONSTER_KILL":
                killer_id = event.get("killerId") or 0
                killer_team = pid_by_key.get(str(killer_id), {}).get("team_id")
                if not killer_team:
                    killer_team = event.get("killerTeamId")
                monster = (event.get("monsterType") or "").upper()
                sub = (event.get("monsterSubType") or "").upper()
                if killer_team in objectives:
                    if monster == "DRAGON":
                        objectives[killer_team]["dragons"] += 1
                    elif monster == "BARON_NASHOR":
                        objectives[killer_team]["barons"] += 1
                    elif monster == "RIFTHERALD":
                        objectives[killer_team]["heralds"] += 1
                    elif "HORDE" in monster or "GRUB" in monster or "VOID" in sub:
                        objectives[killer_team]["voidgrubs"] += 1

            footwear_timer.observe_batch(ts, [event])
            i += 1

        frame_ts = int(frame.get("timestamp") or 0)
        # Empty periods still advance the timer, never beyond the recorded end.
        clock_ts = min(frame_ts, precise_end)
        footwear_timer.advance_before(clock_ts)
        for key, stats in (frame.get("participantFrames") or {}).items():
            pid = int(stats.get("participantId") or key)
            last_frames[pid] = stats
            frame_series.setdefault(pid, []).append(
                (frame_ts, int(stats.get("currentGold") or 0), int(stats.get("totalGold") or 0))
            )

    for pid in list(open_visits):
        close_visit(pid)

    game_end_ms = (info.get("gameDuration") or 0) * 1000
    for snap in death_snaps:
        pid = snap["pid"]
        ts = snap["ts"]
        if ts + SAVE_WINDOW_MS > game_end_ms - SAVE_ENDGAME_MS:
            continue
        if any(ts < p <= ts + SAVE_WINDOW_MS for p in purchases_by_pid.get(pid, [])):
            continue
        stats = snap["frames"].get(pid) or {}
        gold = int(stats.get("currentGold") or 0)
        if gold < SAVE_MIN_GOLD:
            continue
        owner = by_pid[pid]
        inv_payload = _items_payload(snap["inventories"].get(pid) or [], dragon)
        end_kda = snap["kda"][pid]
        events.append(
            {
                "type": "shop",
                "is_save": True,
                "reconstruction_version": RECONSTRUCTION_VERSION,
                "ts": ts + 15_000,
                "ts_end": ts + 15_000,
                "puuid": owner["puuid"],
                "participant_id": pid,
                "champion_name": owner["champion_name"],
                "champion_id": owner["champion_id"],
                "team_id": owner["team_id"],
                "team_position": owner["team_position"],
                # Match the death snapshot used for the other no-buy inputs;
                # later completion during the outcome window is not an input.
                "bot_quest_complete": snap["bot_quest_complete"][pid],
                "riot_id": owner["riot_id"],
                "gold": gold,
                "gold_left": gold,
                "level": stats.get("level"),
                "cs": (stats.get("minionsKilled") or 0) + (stats.get("jungleMinionsKilled") or 0),
                "kills": end_kda["kills"],
                "deaths": end_kda["deaths"],
                "assists": end_kda["assists"],
                "inventory_before": inv_payload,
                "inventory_after": inv_payload,
                "bought": [],
                "label_bought": [],
                "label_source_version": "purchase-net-v1",
                "purchase_actions": [],
                "consumed": [],
                "board": _board_snapshot(
                    participants, snap["inventories"], snap["frames"], snap["kda"], dragon, pid,
                    snap["bot_quest_complete"],
                ),
                "score": snap["objectives"],
            }
        )

    _attach_gold_estimates(events, frame_series, gold_events)
    # A zero-ID reversal has no recoverable inventory identity. Keep the game
    # as source evidence, but no perspective may teach from an uncertain board.
    opaque_undos = [e for f in frames for e in f.get("events", [])
                   if e.get("type") == "ITEM_UNDO" and not e.get("beforeId") and not e.get("afterId")]
    footwear_selected = {p["participantId"]: _rune_selected(p) for p in info["participants"]}
    footwear_grant_times = {g["participant_id"]: g["timestamp"] for g in footwear_timer.grants}
    for event in events:
        # Save inputs were frozen at death, fifteen seconds before their
        # synthetic shop timestamp. Never attach the eventual grant outcome.
        anchor = event["ts"] - 15_000 if event.get("is_save") else event["ts"]

        def footwear_state(owner):
            grant_at = footwear_grant_times.get(owner)
            return {
                "footwear_state_version": FOOTWEAR_STATE_VERSION,
                "magical_footwear_selected": footwear_selected[owner],
                "footwear_granted": grant_at is not None and grant_at < anchor,
            }

        event.update(footwear_state(event["participant_id"]))
        for player in event.get("board", []):
            player.update(footwear_state(player["participant_id"]))
        event["patch"] = patch
        event["ddragon_version"] = dragon.version
        event["static_data_sha256"] = dragon.signature
        event["inventory_version"] = INVENTORY_VERSION
        event["slot_state_version"] = SLOT_STATE_VERSION
        event["training_eligible"] = not opaque_undos
        event["source_quality_issues"] = ["undo_without_item_ids"] if opaque_undos else []
    if gold_debug is not None:
        gold_debug["frame_series"] = frame_series
        gold_debug["gold_events"] = gold_events
        gold_debug["final_inventories"] = {p:list(inv) for p,inv in inventories.items()}
        gold_debug['footwear_grants'] = list(footwear_timer.grants)
        gold_debug['footwear_boundaries'] = list(footwear_timer.boundaries)
        gold_debug['footwear_diagnostics'] = list(footwear_timer.diagnostics)
        gold_debug['footwear_state'] = footwear_timer.snapshot()
        gold_debug['footwear_batches'] = footwear_batches
        gold_debug['footwear_inventory_failures'] = footwear_inventory_failures
        gold_debug['footwear_inventory_checks'] = footwear_inventory_checks
        gold_debug['elixir_auto_consumptions'] = getattr(inventory_replay, 'auto_consumption_records', lambda: [])()
    events.sort(key=lambda e: (e.get("ts") or 0, e.get("ts_end") or 0))
    for idx, event in enumerate(events, start=1):
        event["event_index"] = idx
    return game, events


def _passive_gold(a_ms: int, b_ms: int) -> float:
    a = max(a_ms, PASSIVE_START_MS)
    b = max(b_ms, PASSIVE_START_MS)
    return max(0.0, (b - a) / 1000.0 * PASSIVE_GOLD_PER_S)


def _attach_gold_estimates(
    events: list[dict],
    frame_series: dict[int, list[tuple[int, int, int]]],
    gold_events: dict[int, list[tuple[int, float, str]]],
) -> None:
    """Stamp each shop event with `gold_est`: a CAUSAL estimate of the gold the
    player held on arriving at the shop at time t.

    Strictly pre-decision. The only legal inputs are:

      * `f0`, the last participant frame strictly before t (`currentGold`);
      * deterministic passive income from f0 to t;
      * kill/shutdown bounties with timestamp < t;
      * purchases/sells/undos with timestamp < t — the visit's own buys share t
        exactly and are never counted;
      * a residual-income forecast extrapolated from the PREVIOUS complete frame
        window (fprev, f0): the farm, assists and objective gold the player was
        already earning before deciding.

    Reading `series[i + 1]` (the frame after the visit), its `totalGold`, or any
    event after t is forbidden — that is how "leaky-v1" let post-purchase income
    flow into the feature. The no-future property is covered by a unit test in
    scripts/test_gold_causal.py; keep it passing.

    The result is deliberately approximate. It is not exact arrival gold and
    must never be described as such: offline, exact gold at an arbitrary
    timestamp does not exist in Match-V5. Only live rows carry exact gold, and
    they are tagged `live-exact-v2` instead.
    """
    from bisect import bisect_left

    for ev in events:
        if ev.get("type") != "shop":
            continue
        pid = ev.get("participant_id")
        t = int(ev.get("ts") or 0)
        series = frame_series.get(pid) or []
        if not series:
            continue
        i = bisect_left([f[0] for f in series], t) - 1
        if i < 0:
            continue
        f0_ts, f0_cur, f0_tot = series[i]
        timed = gold_events.get(pid) or []
        bounty_to_t = sum(a for ets, a, k in timed if k == "bounty" and f0_ts < ets < t)
        spend_to_t = sum(a for ets, a, k in timed if k == "spend" and f0_ts < ets < t)
        est = f0_cur + _passive_gold(f0_ts, t) + bounty_to_t - spend_to_t
        # Opening-sale proceeds: sales strictly before the visit's first
        # purchase, priced by the spend stream and stamped by close_visit.
        # Window predicates above stay strict — relaxing `< t` to `<= t` for
        # spends would debit the visit's own first purchase, which is priced
        # from the label. gold_gap["spent"] stays 0 for the visit's own buys.
        opening_sale_gold = float(ev.get("opening_sale_gold") or 0)
        est += opening_sale_gold

        # Forecast the non-passive, non-bounty income (farm/assists/objectives)
        # at the rate observed over the previous complete window. totalGold is
        # cumulative income, so spending never enters this term.
        #
        # Asymmetry, on purpose: the rate is per unit of WALL-CLOCK time in the
        # previous window (prior_residual / prior_window, dead time included,
        # because there is no reason to think that window contained a shop
        # stop), while it is credited only over (t - f0) minus the dead time the
        # player spends walking or recalling. So the forecast under-credits by
        # roughly one dead-time's worth of income — order 60-120g at mid-game
        # farm rates. This reduces the forecast; it does not guarantee that the
        # estimate lies below the player's true wallet. Subtracting dead time from the
        # denominator as well would assume the prior window ended in a shop stop
        # too, which is the less defensible of the two claims.
        prior_residual = None
        if i > 0:
            fp_ts, _fp_cur, fp_tot = series[i - 1]
            bounty_prior = sum(a for ets, a, k in timed if k == "bounty" and fp_ts < ets <= f0_ts)
            prior_residual = max(
                0.0, (f0_tot - fp_tot) - _passive_gold(fp_ts, f0_ts) - bounty_prior
            )
            prior_window = max(1, f0_ts - fp_ts)
            active_ms = max(0.0, (t - f0_ts) - SHOP_DEAD_TIME_MS)
            est += prior_residual * (active_ms / prior_window)

        ev["gold_est"] = max(0, int(round(est)))
        ev["gold_est_version"] = GOLD_EST_VERSION
        ev["gold_gap"] = {
            "frame_ts": f0_ts,
            "bounty": int(bounty_to_t),
            "spent": int(spend_to_t),
            "opening_sale_gold": int(opening_sale_gold),
            "prior_residual": int(prior_residual) if prior_residual is not None else None,
            "version": GOLD_EST_VERSION,
        }


def reconstruct_visits(
    match: dict,
    timeline: dict,
    puuid: str,
    dragon: DataDragon,
) -> tuple[dict, list[dict]]:
    game, events = reconstruct_game(match, timeline, dragon)
    return perspective_from_game(game, events, puuid)


def perspective_from_game(game: dict, events: list[dict], puuid: str) -> tuple[dict, list[dict]]:
    """Shape one perspective from an already reconstructed full lobby."""
    participants = json.loads(game["participants_json"])
    target = next((p for p in participants if p["puuid"] == puuid), None)
    if not target:
        raise ValueError("Target player not in this match")
    match_row = {
        "match_id": game["match_id"],
        "puuid": puuid,
        "game_creation": game["game_creation"],
        "game_duration": game["game_duration"],
        "game_version": game["game_version"],
        "patch": game["patch"],
        "queue_id": game["queue_id"],
        "champion_id": target["champion_id"],
        "champion_name": target["champion_name"],
        "team_position": target["team_position"],
        "team_id": target["team_id"],
        "win": int(target["win"]),
        "kills": target["kills"],
        "deaths": target["deaths"],
        "assists": target["assists"],
        "participants_json": game["participants_json"],
    }
    visits = []
    for event in events:
        if event["type"] != "shop" or event["puuid"] != puuid:
            continue
        visits.append(_shop_event_to_visit(event, game["match_id"], puuid, target["team_id"]))
    for i, visit in enumerate(visits, start=1):
        visit["visit_index"] = i
    return match_row, visits


def _shop_event_to_visit(event: dict, match_id: str, puuid: str, team_id: int) -> dict:
    score = event.get("score") or {}
    return {
        "match_id": match_id,
        "puuid": puuid,
        "visit_index": event.get("event_index") or 0,
        "ts_start": event["ts"],
        "ts_end": event.get("ts_end") or event["ts"],
        "gold": event.get("gold"),
        "gold_left": event.get("gold_left"),
        "level": event.get("level"),
        "cs": event.get("cs"),
        "kills": event.get("kills") or 0,
        "deaths": event.get("deaths") or 0,
        "assists": event.get("assists") or 0,
        "inventory_before_json": json.dumps(event.get("inventory_before") or []),
        "inventory_after_json": json.dumps(event.get("inventory_after") or []),
        "bought_json": json.dumps(event.get("bought") or []),
        "consumed_json": json.dumps(event.get("consumed") or []),
        "board_json": json.dumps(event.get("board") or []),
        "allies_json": json.dumps([]),
        "enemies_json": json.dumps([]),
        "objectives_json": json.dumps(
            {
                "ally": score.get(team_id) or {},
                "enemy": score.get(100 if team_id == 200 else 200) or {},
            }
        ),
    }


def reconstruct_decisions(match: dict, timeline: dict, puuid: str, dragon: DataDragon):
    return reconstruct_visits(match, timeline, puuid, dragon)



def _diff(before: list[int], after: list[int]) -> tuple[list[int], list[int]]:
    bought = list((Counter(after) - Counter(before)).elements())
    consumed = list((Counter(before) - Counter(after)).elements())
    return bought, consumed


def _items_payload(ids: list[int], dragon: DataDragon) -> list[dict]:
    out = []
    for item_id in ids:
        cls = dragon.classify(item_id)
        out.append(
            {
                "item_id": item_id,
                "item_name": cls["name"],
                "skip": bool(cls["skip"]),
                "target_skip": bool(cls['target_skip']),
                **({"inference": "support-tier-proxy-v1"} if item_id == SUPPORT_BOUNTY else {}),
            }
        )
    return out




def _board_snapshot(
    participants: list[dict],
    inventories: dict[int, list[int]],
    frames: dict[int, dict],
    kda: dict[int, dict],
    dragon: DataDragon,
    self_pid: int,
    bot_quest_complete: dict[int, bool] | None = None,
) -> list[dict]:
    out = []
    for p in participants:
        items = inventories.get(p["participant_id"], [])
        stats = frames.get(p["participant_id"]) or {}
        out.append(
            {
                "participant_id": p["participant_id"],
                "champion_id": p["champion_id"],
                "champion_name": p["champion_name"],
                "team_id": p["team_id"],
                "team_position": p["team_position"],
                "is_self": p["participant_id"] == self_pid,
                "bot_quest_complete": (bot_quest_complete or {}).get(p["participant_id"]),
                "slot_state_version": SLOT_STATE_VERSION if bot_quest_complete is not None else None,
                "items": _items_payload(items, dragon),
                "gold": stats.get("totalGold"),
                "level": stats.get("level"),
                **kda[p["participant_id"]],
            }
        )
    out.sort(key=lambda row: (row["team_id"], row["participant_id"]))
    return out
