"""Independent QA for timer-prototype outputs; never infers a delivery time.

These checks consume a candidate grant trace, source events and pinned static
data. They do not import the timer engine, reconstructor, inventory replay or
prototype economics. Final rune counters are deliberately QA-only. A pass is
consistency with observed evidence, not proof of an exact server grant tick.
"""
from __future__ import annotations

from collections import Counter, defaultdict

RUNE, FOOTWEAR, ORDINARY_BOOTS = 8304, 2422, 1001
BASE_MS, TAKEDOWN_MS = 720_000, 45_000
ORACLE_VERSION = "footwear-timer-independent-qa-v1"


def source_events(timeline):
    return sorted((event for frame in timeline["info"]["frames"]
                   for event in frame.get("events", [])),
                  key=lambda event: event["timestamp"])


def selected_wearers(match):
    return {p["participantId"]: p for p in match["info"]["participants"]
            if any(r.get("perk") == RUNE for s in p.get("perks", {}).get("styles", [])
                   for r in s.get("selections", []))}


def grant_rune_qa(match, timeline, grants):
    """Check each inferred grant against causal eligibility and outcome QA.

All credited takedowns through a pending-capacity grant count toward the final
displayed deadline. Counting only before first eligibility incorrectly rejects
the three documented full-inventory examples. A takedown at the grant's own
timestamp is permitted to trigger it; it cannot make an earlier grant causal.
The m:ss interpretation is an empirical hypothesis, not a Riot API guarantee.
"""
    teams = {p["participantId"]: p["teamId"] for p in match["info"]["participants"]}
    wearers = selected_wearers(match)
    by_owner = defaultdict(list)
    for grant in grants:
        by_owner[grant["participant_id"]].append(grant)
    issues, records = [], []
    for owner in sorted(set(by_owner) - set(wearers)):
        issues.append({"participant_id": owner, "code": "grant_without_fixed_rune"})
    kills = []
    seen = set()
    for event in source_events(timeline):
        if event.get("type") != "CHAMPION_KILL":
            continue
        identity = event["timestamp"], event.get("victimId"), event.get("killerId")
        if identity in seen:
            continue
        seen.add(identity)
        victim = event.get("victimId")
        if victim not in teams:
            continue
        credit = {event.get("killerId"), *(event.get("assistingParticipantIds") or [])}
        kills.append((event["timestamp"], {p for p in credit if p in teams
                                         and p != victim and teams[p] != teams[victim]}))
    for owner, participant in sorted(wearers.items()):
        selection = next(r for s in participant["perks"]["styles"]
                         for r in s["selections"] if r.get("perk") == RUNE)
        counters = [selection.get("var" + str(i)) for i in (1, 2, 3)]
        observed = counters[0] * 60 + counters[1] * 10 + counters[2] if all(
            type(value) is int and value >= 0 for value in counters) else None
        candidates = by_owner.get(owner, [])
        row = {"participant_id": owner, "postgame_deadline_seconds_qa_only": observed,
               "grant_count": len(candidates)}
        if len(candidates) > 1:
            issues.append({"participant_id": owner, "code": "duplicate_system_grant"})
        if not candidates:
            row.update(expected_postgame_deadline_seconds=0,
                       postgame_deadline_matches=observed == 0)
        else:
            grant = candidates[0]
            stamp = grant["timestamp"]
            prior = [ts for ts, credit in kills if owner in credit and ts <= stamp]
            expected_ms = BASE_MS - TAKEDOWN_MS * len(prior)
            row.update(grant_timestamp=stamp, credited_takedowns_through_grant_ms=prior,
                       expected_postgame_deadline_seconds=expected_ms // 1000,
                       postgame_deadline_matches=observed * 1000 == expected_ms if observed is not None else False,
                       causal_eligibility_satisfied=stamp >= expected_ms)
            if type(stamp) is not int or stamp < 0 or stamp < expected_ms:
                issues.append({"participant_id": owner, "code": "grant_precedes_causal_eligibility"})
            if any(grant.get(field) != required for field, required in
                   (("gold_delta", 0), ("purchase_cost", 0), ("purchase_label", False))):
                issues.append({"participant_id": owner, "code": "invalid_system_grant_economics"})
        if observed is None:
            issues.append({"participant_id": owner, "code": "missing_postgame_rune_qa"})
        elif not row["postgame_deadline_matches"]:
            issues.append({"participant_id": owner, "code": "postgame_reduced_deadline_disagrees"})
        records.append(row)
    return {"version": ORACLE_VERSION, "passed": not issues,
            "wearers": records, "issues": issues,
            "limit": "Outcome counters validate a reduced-deadline hypothesis; no grant timestamp is independently observed."}


def source_recipe_price(item_id, consumed, dragon):
    """Independently price the observed physical component multiset.

The candidate reconstructor's own paid costs and economics functions are not
consulted. Unexplained consumed components fail, rather than silently becoming
sale credits or receiving ordinary-boots identity.
"""
    remaining = Counter(consumed)
    used = Counter()
    allowed = {int(i) for i in (dragon.item(FOOTWEAR) or {}).get("into", [])}

    def acquire(item, parent=None, ancestors=()):
        if item in ancestors:
            raise ValueError("cyclic_static_recipe")
        physical = item
        if item == ORDINARY_BOOTS and parent in allowed and not remaining[item] and remaining[FOOTWEAR]:
            physical = FOOTWEAR
        if parent is not None and remaining[physical]:
            remaining[physical] -= 1
            used[physical] += 1
            return 0
        data = dragon.item(item)
        if not data:
            raise ValueError("unknown_static_item")
        children = [int(i) for i in data.get("from", [])]
        if not children:
            return int(data["gold"]["total"])
        return int(data["gold"]["base"]) + sum(
            acquire(child, parent=item, ancestors=ancestors + (item,)) for child in children)

    price = acquire(item_id)
    if +remaining:
        raise ValueError("unexplained_consumed_components")
    return price, sorted(used.elements())


def source_components(batch, dragon, *, mid_complete):
    """Separate the reviewed automatic mid-quest upgrade marker from cost.

After source-observed quest1201, a tier-two boot purchase may also DESTROY that
same tier-two id because it immediately becomes its zero-base-cost successor.
It was never an already-owned recipe component. Require the exact pinned
single-component free successor; do not drop arbitrary self-destructions.
"""
    destroyed = Counter(e["itemId"] for e in batch if e["type"] == "ITEM_DESTROYED"
                        and e.get("itemId") not in (1201, 1202))
    if mid_complete:
        for purchase in (e["itemId"] for e in batch if e["type"] == "ITEM_PURCHASED"):
            data = dragon.item(purchase) or {}
            if not destroyed[purchase] or "Boots" not in data.get("tags", []):
                continue
            successors = [dragon.item(int(i)) or {} for i in data.get("into", [])]
            reviewed = [entry for entry in successors if entry.get("from") == [str(purchase)]
                        and entry.get("gold", {}).get("base") == 0
                        and entry.get("maps", {}).get("11")]
            if len(reviewed) == 1:
                destroyed[purchase] -= 1
    return sorted((+destroyed).elements())


def source_footwear_qa(match, timeline, grants, dragon):
    """Audit physical 2422 ownership and source-observed boot undo refunds.

This deliberately narrow ledger follows the free component only. It does not
claim to reconstruct every item. Ordinary inventory and all-player endpoint
checks must still run. The driver must separately reject split contiguous
receipt transactions, because source timestamp unions cannot establish the
reconstructor's transaction grouping or actual computed spend.
"""
    wearers = selected_wearers(match)
    batches = defaultdict(list)
    all_events = source_events(timeline)
    for event in all_events:
        if event.get("participantId") in wearers and event.get("type", "").startswith("ITEM_"):
            batches[event["timestamp"], event["participantId"]].append(event)
    grants_by_owner = defaultdict(list)
    for grant in grants:
        grants_by_owner[grant["participant_id"]].append(grant)
    issues, ledger = [], []
    source_checks = Counter()
    for owner in sorted(wearers):
        inventory = 0
        grant_queue = sorted(grants_by_owner[owner], key=lambda g: g["timestamp"])
        purchase_history, sale_history = defaultdict(list), []
        bot_complete = mid_complete = False

        def fail(ts, code, **detail):
            issues.append({"participant_id": owner, "timestamp": ts, "code": code, **detail})

        def materialize(grant):
            nonlocal inventory
            inventory += 1
            ledger.append({"participant_id": owner, "timestamp": grant["timestamp"],
                           "kind": "candidate_system_grant", "physical_2422_after": inventory})
            if inventory > 1:
                fail(grant["timestamp"], "duplicate_physical_2422")

        for (stamp, pid), batch in sorted(batches.items()):
            if pid != owner:
                continue
            while grant_queue and grant_queue[0]["timestamp"] < stamp:
                materialize(grant_queue.pop(0))
            if any(g["timestamp"] == stamp for g in grant_queue) and any(
                    e.get("itemId") == FOOTWEAR or e.get("afterId") == FOOTWEAR
                    or e.get("beforeId") == FOOTWEAR for e in batch):
                fail(stamp, "grant_tied_to_raw_footwear_evidence")
            destroyed = Counter(e["itemId"] for e in batch if e["type"] == "ITEM_DESTROYED")
            purchases = [e for e in batch if e["type"] == "ITEM_PURCHASED"]
            bot_complete |= bool(destroyed[1202])
            mid_complete |= bool(destroyed[1201])
            for event in batch:
                kind, item = event["type"], event.get("itemId", 0)
                if item == FOOTWEAR and kind in ("ITEM_DESTROYED", "ITEM_SOLD"):
                    source_checks["source_2422_removals"] += 1
                    if inventory != 1:
                        fail(stamp, "raw_2422_removal_without_owned_component", source_type=kind,
                             physical_count=inventory)
                    # A bot-lane role-slot move preserves ownership. Do not
                    # mistake it for component consumption or a second grant.
                    role_transfer = kind == "ITEM_DESTROYED" and bot_complete and not purchases
                    if not role_transfer:
                        inventory -= 1
                    if kind == "ITEM_SOLD":
                        sale_history.append({"timestamp": stamp, "price": 90})
                    elif not purchases and not role_transfer:
                        fail(stamp, "2422_destruction_without_explained_upgrade_or_role_transfer")
                    ledger.append({"participant_id": owner, "timestamp": stamp, "kind": kind,
                                   "physical_2422_after": inventory, "role_transfer": role_transfer})
                elif kind == "ITEM_PURCHASED":
                    if item == FOOTWEAR:
                        fail(stamp, "explicit_free_boot_receipt_requires_separate_handler")
                    raw = dragon.item(item) or {}
                    is_boot = "Boots" in raw.get("tags", [])
                    if is_boot:
                        if len(purchases) > 1 and destroyed[FOOTWEAR]:
                            fail(stamp, "multiple_simultaneous_boot_receipt_purchases")
                        consumed = source_components(batch, dragon, mid_complete=mid_complete)
                        try:
                            price, components = source_recipe_price(item, consumed, dragon)
                        except ValueError as error:
                            fail(stamp, str(error), item_id=item)
                            price, components = None, consumed
                        purchase_history[item].append({"timestamp": stamp, "price": price,
                                                       "components": components})
                        ledger.append({"participant_id": owner, "timestamp": stamp,
                                       "kind": "source_boot_purchase", "item_id": item,
                                       "expected_paid_gold": price, "consumed": components})
                elif kind == "ITEM_UNDO":
                    before, after = event.get("beforeId", 0), event.get("afterId", 0)
                    if before in purchase_history and purchase_history[before]:
                        transaction = purchase_history[before].pop()
                        source_checks["source_boot_purchase_undo_refunds"] += 1
                        if event.get("goldGain") != transaction["price"]:
                            fail(stamp, "source_boot_undo_refund_disagrees_with_recipe",
                                 observed=event.get("goldGain"), expected=transaction["price"])
                        if after and after not in transaction["components"]:
                            fail(stamp, "undo_after_id_not_in_original_components", item_id=after)
                        inventory += transaction["components"].count(FOOTWEAR)
                    elif after == FOOTWEAR and not before:
                        source_checks["source_2422_sale_undo_refunds"] += 1
                        if not sale_history:
                            fail(stamp, "2422_sale_undo_without_sale")
                        else:
                            sale_history.pop()
                        if event.get("goldGain") != -90:
                            fail(stamp, "2422_sale_undo_wrong_gold", observed=event.get("goldGain"))
                        inventory += 1
                    elif before == FOOTWEAR or after == FOOTWEAR:
                        fail(stamp, "unlinked_2422_undo")
                    if inventory > 1:
                        fail(stamp, "duplicate_physical_2422_after_undo")
            # A grant on a freeing source timestamp is after its raw batch.
            # Any same-timestamp footwear transaction was rejected above.
            while grant_queue and grant_queue[0]["timestamp"] == stamp:
                materialize(grant_queue.pop(0))
        for grant in grant_queue:
            materialize(grant)
    return {"version": ORACLE_VERSION, "passed": not issues,
            "checks": dict(source_checks), "issues": issues, "ledger": ledger,
            "limit": "Physical free-component/source-refund check only; candidate paid ledger and all-player endpoints are separate gates."}


def grant_boundary_reason(event, grants, *, guard_ms):
    """Reject narrow uncertainty windows without declaring a tick delay proven.

The guard is a sensitivity parameter supplied by the caller. Source evidence
does not establish a universal upper bound. Every ten-player decision anchor
is checked, including the causal death anchor; own visits crossing a grant are
also excluded so their labels cannot straddle an uncertain inventory change.
"""
    if type(guard_ms) is not int or guard_ms < 0:
        raise ValueError("guard_ms must be a nonnegative integer")
    anchor = event["ts"] - 15_000 if event.get("is_save") else event["ts"]
    for grant in grants:
        stamp = grant["timestamp"]
        if abs(anchor - stamp) <= guard_ms:
            return "all_board_anchor_near_inferred_grant"
        if (grant["participant_id"] == event["participant_id"]
                and event["ts"] - guard_ms <= stamp <= event["ts_end"] + guard_ms):
            return "own_visit_near_or_spans_inferred_grant"
    return None


def validate_timer_result(match, timeline, events, debug, dragon):
    """Combined gate for the isolated runner, with source-vs-candidate checks.

No endpoints, source-future perturbations or inventory-capacity judgments are
made here: the runner owns those existing full-board checks. This gate ensures
the candidate actually held the physical free component at source removals and
priced observed free-component transactions as the independent recipe says.
"""
    grants = debug.get("footwear_grants", [])
    rune = grant_rune_qa(match, timeline, grants)
    source = source_footwear_qa(match, timeline, grants, dragon)
    issues, transaction_checks = [], []
    wearers = selected_wearers(match)
    mid_quests = {}
    for event in source_events(timeline):
        if event.get("type") == "ITEM_DESTROYED" and event.get("itemId") == 1201:
            mid_quests.setdefault(event["participantId"], event["timestamp"])
    source_removals = Counter(
        (e.get("participantId"), e["timestamp"], e["type"])
        for e in source_events(timeline)
        if e.get("participantId") in wearers and e.get("itemId") == FOOTWEAR
        and e.get("type") in ("ITEM_DESTROYED", "ITEM_SOLD"))
    inspected_removals = Counter()
    for record in debug.get("footwear_batches", []):
        owner, stamp = record["participant_id"], record["ts"]
        if owner not in wearers:
            continue
        batch = record["batch"]
        removals = [e for e in batch if e.get("itemId") == FOOTWEAR
                    and e.get("type") in ("ITEM_DESTROYED", "ITEM_SOLD")]
        for event in removals:
            inspected_removals[owner, stamp, event["type"]] += 1
        if removals and record["before"].count(FOOTWEAR) != 1:
            issues.append({"participant_id": owner, "timestamp": stamp,
                           "code": "candidate_did_not_hold_physical_2422_at_source_removal",
                           "candidate_before": record["before"]})
        # Sale and the normal one-purchase combine are independently priced.
        # Splits/multiple purchases are explicit driver blockers, not silently
        # merged into a convenient synthetic transaction here.
        purchases = [e for e in batch if e["type"] == "ITEM_PURCHASED"]
        undo = any(e["type"] == "ITEM_UNDO" for e in batch)
        if not removals or undo:
            continue
        if len(purchases) > 1:
            issues.append({"participant_id": owner, "timestamp": stamp,
                           "code": "multiple_purchases_in_free_component_batch"})
            continue
        destroyed = source_components(batch, dragon,
                                      mid_complete=mid_quests.get(owner, float("inf")) <= stamp)
        sales = [e["itemId"] for e in batch if e["type"] == "ITEM_SOLD"]
        if not purchases and not sales:
            continue  # Standalone role transfer is audited in the source ledger.
        try:
            expected = source_recipe_price(purchases[0]["itemId"], destroyed, dragon)[0] if purchases else 0
        except ValueError as error:
            issues.append({"participant_id": owner, "timestamp": stamp, "code": str(error)})
            continue
        expected -= sum(int(dragon.item(i)["gold"]["sell"]) for i in sales)
        # Exact owner/timestamp comparison is valid only if the driver also
        # rejects split receipt batches. Gold ledger entries have own provenance.
        owner_gold = debug.get("gold_events", {}).get(owner,
                     debug.get("gold_events", {}).get(str(owner), []))
        actual = sum(amount for ts, amount, kind in owner_gold if ts == stamp and kind == "spend")
        checked = {"participant_id": owner, "timestamp": stamp,
                   "expected_net_spend": expected, "candidate_net_spend": actual,
                   "passed": expected == actual}
        transaction_checks.append(checked)
        if not checked["passed"]:
            issues.append({"participant_id": owner, "timestamp": stamp,
                           "code": "candidate_free_component_transaction_cost_disagrees",
                           "expected": expected, "actual": actual})
    if source_removals != inspected_removals:
        issues.append({"code": "candidate_removal_trace_incomplete",
                       "source_count": sum(source_removals.values()),
                       "inspected_count": sum(inspected_removals.values())})
    for index, event in enumerate(events):
        if any((i.get("item_id") if isinstance(i, dict) else i) == FOOTWEAR
               for i in event.get("label_bought", [])):
            issues.append({"code": "free_grant_became_purchase_label", "event_index": index})
    return {"version": ORACLE_VERSION, "passed": rune["passed"] and source["passed"] and not issues,
            "rune_counter_qa": rune, "source_component_qa": source,
            "candidate_transaction_checks": transaction_checks, "issues": issues,
            "delivery_tick_independently_certified": False,
            "training_admission": False}
