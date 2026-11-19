"""Experimental causal windows for the unobserved Magical Footwear grant.

This module never inserts an item, reconstructs an inventory, or admits a game.
It identifies when the missing one-time grant can still affect a decision's
ten-player board. Actual inventory, transaction, slot and source audits remain
mandatory. Only fixed rune selections and strictly earlier timeline events are
used; final inventories, final KDA and rune outcome counters are not read.

The nominal 12-minute deadline moves 45 seconds earlier for each champion
takedown. Crossing the deadline on a late takedown cannot backdate eligibility.
Eligibility is NOT evidence that delivery occurred: a full inventory can delay
it. A missing grant remains unresolved until an observed 2422 sale, or a 2422
destruction paired with a compatible boot purchase at the same timestamp.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping


FOOTWEAR_RUNE = 8304
MAGICAL_BOOTS = 2422
ORDINARY_BOOTS = 1001
BASE_ELIGIBILITY_MS = 720_000
TAKEDOWN_REDUCTION_MS = 45_000
WINDOW_VERSION = "magical-footwear-windows-prototype-v1"


def _timestamp(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer timestamp")
    return value


def _rune_selected(participant):
    # An absent legacy rune payload is unknown, not evidence of no footwear.
    styles = (participant.get("perks") or {}).get("styles")
    if not isinstance(styles, list) or not styles:
        raise ValueError(f"Missing fixed rune selections for participant {participant['participantId']}")
    selected = False
    seen_selection = False
    for style in styles:
        selections = style.get("selections")
        if not isinstance(selections, list):
            raise ValueError("Missing fixed rune selections")
        seen_selection |= bool(selections)
        selected |= any(selection.get("perk") == FOOTWEAR_RUNE for selection in selections)
    if not seen_selection:
        raise ValueError("Missing fixed rune selections")
    return selected


def _is_boot(item_id, dragon):
    classification = dragon.classify(item_id)
    return bool(classification.get("is_boots") or classification.get("is_basic_boots"))


def compatible_footwear_upgrade(item_id: int, dragon) -> bool:
    """Require a reviewed 2422 upgrade branch and a normal-boot ingredient.

The explicit 2422 ``into`` list avoids silently accepting an unrelated/map-
specific item merely because its recipe contains 1001. Recursive descendants
allow an already completed mid quest's automatic advanced-boot upgrade.
"""
    if item_id in (MAGICAL_BOOTS, ORDINARY_BOOTS) or not _is_boot(item_id, dragon):
        return False
    allowed = {int(i) for i in (dragon.item(MAGICAL_BOOTS) or {}).get("into", [])}
    if not allowed:
        return False
    ancestors, pending = set(), [item_id]
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(dragon.from_ids(current))
    return ORDINARY_BOOTS in ancestors and bool(allowed & ancestors)


@dataclass
class _Wearer:
    participant_id: int
    deadline: int = BASE_ELIGIBILITY_MS
    first_possible: int | None = None
    resolved_at: int | None = None
    resolved_by: str | None = None
    takedowns: int = 0
    diagnostics: list = field(default_factory=list)
    invalid: bool = False

    def advance(self, timestamp):
        if self.first_possible is None and self.deadline <= timestamp:
            self.first_possible = self.deadline

    def takedown(self, timestamp):
        self.advance(timestamp)
        # Later takedowns cannot change an already reached eligibility time.
        if self.first_possible is not None:
            return
        self.takedowns += 1
        self.deadline -= TAKEDOWN_REDUCTION_MS
        if self.deadline <= timestamp:
            self.first_possible = timestamp

    def note(self, code, timestamp, *, invalid=False):
        self.diagnostics.append({"code": code, "timestamp": timestamp})
        self.invalid |= invalid

    def state_at(self, timestamp):
        possible = self.first_possible if self.first_possible is not None else self.deadline
        # A transition tied with this query is not part of its strict prefix.
        if self.invalid:
            phase = "invalid"
        elif self.resolved_at is not None:
            phase = "resolved"
        elif possible < timestamp:
            phase = "grant_unobserved"
        else:
            phase = "pre_grant"
        return {
            "participant_id": self.participant_id,
            "selected": True,
            "phase": phase,
            "first_possible_grant_at": possible,
            "grant_time_observed": False,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "window_known": phase in ("pre_grant", "resolved"),
            "boot_purchases_blocked": phase != "resolved",
            "diagnostics": [dict(entry) for entry in self.diagnostics],
        }


def _apply_group(timestamp, events, wearers, teams, dragon):
    for wearer in wearers.values():
        wearer.advance(timestamp)

    # Equal-timestamp events are observed together only by LATER decisions.
    # Count at most one takedown per participant per champion death, even if a
    # malformed assister list repeats the killer or an assisting participant.
    for event in events:
        if event.get("type") != "CHAMPION_KILL":
            continue
        victim = event.get("victimId")
        if victim not in teams:
            continue
        credited = set(event.get("assistingParticipantIds") or [])
        credited.add(event.get("killerId"))
        for pid in credited:
            if pid in wearers and pid != victim and teams.get(pid) != teams[victim]:
                wearers[pid].takedown(timestamp)

    batches = defaultdict(list)
    for event in events:
        owner = event.get("participantId")
        if owner in wearers:
            batches[owner].append(event)

    for owner, batch in batches.items():
        wearer = wearers[owner]
        eligible = wearer.first_possible is not None
        purchases = [event.get("itemId") for event in batch
                     if event.get("type") == "ITEM_PURCHASED"]
        boot_receipt = any(item and _is_boot(item, dragon) for item in purchases)
        footwear_evidence = any(
            event.get("itemId") == MAGICAL_BOOTS
            and event.get("type") in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED")
            for event in batch
        )
        if not eligible and (boot_receipt or footwear_evidence):
            wearer.note("boot_evidence_before_earliest_eligibility", timestamp, invalid=True)

        for event in batch:
            if event.get("type") != "ITEM_UNDO":
                continue
            before, after = event.get("beforeId") or 0, event.get("afterId") or 0
            if not before and not after:
                wearer.note("undo_without_item_ids", timestamp, invalid=True)
            elif not eligible and any(item and _is_boot(item, dragon) for item in (before, after)):
                wearer.note("boot_undo_before_earliest_eligibility", timestamp, invalid=True)

        sold = any(event.get("type") == "ITEM_SOLD" and event.get("itemId") == MAGICAL_BOOTS
                   for event in batch)
        destroyed = any(event.get("type") == "ITEM_DESTROYED" and event.get("itemId") == MAGICAL_BOOTS
                        for event in batch)
        upgraded = any(item and compatible_footwear_upgrade(item, dragon) for item in purchases)
        if wearer.resolved_at is None and eligible and (sold or (destroyed and upgraded)):
            wearer.resolved_at = timestamp
            wearer.resolved_by = "observed_2422_sale" if sold else "observed_2422_upgrade"
        elif wearer.resolved_at is None and destroyed:
            wearer.note("2422_destruction_without_compatible_upgrade", timestamp)
        if wearer.resolved_at is None and MAGICAL_BOOTS in purchases:
            # Explicit grant receipts need their own transaction integration.
            # This prototype deliberately does not invent an inventory record.
            wearer.note("2422_receipt_requires_inventory_replay", timestamp)


def boots_purchase_blocked(footwear_state: Mapping, item_id: int, dragon) -> bool:
    """Additional footwear-only legality guard; not a purchase simulator.

Unknown delivery is conservatively blocked as it can still be pending due to
full slots. A resolved sale/upgrade permits normal boot rules again, including
undo-restored footwear; the separate inventory ledger owns those actual items.
"""
    if not _is_boot(item_id, dragon):
        return False
    return footwear_state.get("selected") is not False and footwear_state.get("phase") not in ("resolved", "not_selected")


def classify_decision_windows(match: Mapping, timeline: Mapping,
                              decisions: Iterable[Mapping], dragon) -> list[dict]:
    """Annotate candidate decisions without modifying either input.

Each decision supplies ``participant_id`` and ``anchor_ts`` (or ``ts``).
The caller must supply the causal anchor, including the existing offset for
save examples. Original fields and input order are preserved. The returned
``footwear_window_known`` concerns ONLY missing-footwear uncertainty, never
overall game or inventory eligibility. Every wearer participates in the board
gate, even when someone else is shopping.
"""
    participants = match["info"]["participants"]
    teams, selected = {}, {}
    for participant in participants:
        pid = participant["participantId"]
        if pid in teams:
            raise ValueError("Duplicate participant id")
        teams[pid] = participant["teamId"]
        selected[pid] = _rune_selected(participant)
    wearers = {pid: _Wearer(pid) for pid, active in selected.items() if active}

    queries = []
    for index, decision in enumerate(decisions):
        row = dict(decision)
        if row.get("participant_id") not in teams:
            raise ValueError("Decision participant is absent from match roster")
        anchor = row.get("anchor_ts", row.get("ts"))
        queries.append((_timestamp(anchor, "Decision anchor"), index, row))
    queries.sort(key=lambda entry: (entry[0], entry[1]))

    event_groups = defaultdict(list)
    if wearers and queries:
        maximum_anchor = queries[-1][0]
        for frame in timeline.get("info", {}).get("frames", []):
            for event in frame.get("events", []):
                timestamp = _timestamp(event.get("timestamp"), "Event timestamp")
                if timestamp < maximum_anchor:
                    event_groups[timestamp].append(event)
    groups = iter(sorted(event_groups.items()))
    current = next(groups, None)
    results = [None] * len(queries)
    for anchor, index, row in queries:
        while current is not None and current[0] < anchor:
            _apply_group(current[0], current[1], wearers, teams, dragon)
            current = next(groups, None)
        board = [wearers[pid].state_at(anchor) for pid in sorted(wearers)]
        states = {state["participant_id"]: state for state in board}
        owner_state = states.get(row["participant_id"], {
            "participant_id": row["participant_id"], "selected": False,
            "phase": "not_selected", "first_possible_grant_at": None,
            "grant_time_observed": False, "resolved_at": None, "resolved_by": None,
            "window_known": True, "boot_purchases_blocked": False, "diagnostics": [],
        })
        blockers = [state["participant_id"] for state in board if not state["window_known"]]
        results[index] = {
            **row,
            "footwear_window_version": WINDOW_VERSION,
            "footwear_state": owner_state,
            "footwear_board_states": board,
            "footwear_blocking_participant_ids": blockers,
            "footwear_window_known": not blockers,
            "footwear_diagnostics": [
                {"participant_id": state["participant_id"], **entry}
                for state in board for entry in state["diagnostics"]
            ],
        }
    return results
