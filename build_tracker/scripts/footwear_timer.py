"""Experimental causal timer/slot hypothesis for Magical Footwear.

The pinned rune describes 12 minutes minus 45 seconds per champion takedown.
Delivery at the first available slot is an explicit replay hypothesis: neither
an exact delivery tick nor possession is observable from that description.
This module emits candidate system grants, never training admission decisions.
Only fixed rune selections, earlier raw events and current inventory callbacks
are consulted. Postgame counters, final inventories and future events are not.

Integration order for each contiguous raw batch at ``ts``::

    scheduler.advance_before(ts)  # inventory still precedes the raw batch
    apply_raw_batch(batch)
    scheduler.observe_batch(ts, batch)

Do not invoke the last two steps for a speculative future event. Grants inside
a raw timestamp are boundaries the driver must exclude from decision inputs.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from magical_footwear import (
    BASE_ELIGIBILITY_MS, MAGICAL_BOOTS, TAKEDOWN_REDUCTION_MS, _rune_selected,
)

TIMER_VERSION = "magical-footwear-timer-slots-prototype-v1"
DELIVERY_HYPOTHESIS = "timer-and-first-available-slot-v1"
CONTROL_WARDS = frozenset({2055, 772043})
QUEST_MARKERS = frozenset(range(1200, 1205))


def _timestamp(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Timestamp must be a nonnegative integer")
    return value


def physical_slot_state(inventory: Iterable[int], dragon, *, role: str = "",
                        bot_quest_complete: bool | None = None) -> dict:
    """Count raw physical slots without dropping skipped model-context items.

    Potions use pinned stack sizes; control wards use a dedicated support slot.
    A completed bottom quest moves one boot to its role slot. Known trinkets and
    quest markers take no regular slots. Unknown items remain explicit blockers.
    ``inventory`` contains one integer entry per physical item copy, not payload
    dictionaries or model-normalized items.
    """
    owned = Counter()
    for item_id in inventory:
        if type(item_id) is not int or item_id < 0:
            raise ValueError("Physical inventory requires nonnegative integer item ids")
        if item_id:
            owned[item_id] += 1
    regular = boots = ward_copies = role_copies = 0
    unknown, invalid_stacks, omitted = [], [], []
    for item_id, copies in sorted(owned.items()):
        if item_id in QUEST_MARKERS:
            omitted.append(item_id)
            continue
        raw = dragon.item(item_id)
        if not raw:
            unknown.append(item_id)
            continue
        if "Trinket" in (raw.get("tags") or []):
            omitted.append(item_id)
            continue
        raw_stack = raw.get("stacks", 1)
        # Pinned 16.17 also uses explicit zero on ordinary nonstacked items
        # (3133, 3134, 3916). Zero/null/missing therefore mean one per slot.
        stack = 1 if raw_stack is None or type(raw_stack) is int and raw_stack == 0 else raw_stack
        if type(stack) is not int or stack < 1:
            invalid_stacks.append(item_id)
            continue
        classification = dragon.classify(item_id)
        is_boot = (item_id == MAGICAL_BOOTS or classification.get("is_boots")
                   or classification.get("is_basic_boots"))
        boots += copies if is_boot else 0
        if item_id in CONTROL_WARDS:
            ward_copies += copies
            if copies > stack:
                invalid_stacks.append(item_id)
            if role == "UTILITY":
                role_copies += copies
                # Capacity beyond the dedicated stack is contradictory even
                # though it does not create another ordinary item slot.
                continue
        regular += (copies + stack - 1) // stack
    if role == "BOTTOM" and bot_quest_complete is True and boots:
        regular -= 1
        role_copies += 1
    return {
        "regular_slots": regular,
        "boot_copies": boots,
        "ward_copies": ward_copies,
        "role_slot_item_copies": role_copies,
        "unknown_item_ids": unknown,
        "invalid_stack_item_ids": invalid_stacks,
        "omitted_nonregular_item_ids": omitted,
        "known": not unknown and not invalid_stacks,
    }


@dataclass
class _Wearer:
    participant_id: int
    nominal_deadline: int = BASE_ELIGIBILITY_MS
    takedowns: int = 0
    eligible_at: int | None = None
    granted_at: int | None = None
    first_full_slot_at: int | None = None
    pending_reason: str | None = None
    invalid: bool = False
    diagnostics: list[dict] = field(default_factory=list)


class TimerFootwearScheduler:
    """Event-driven candidate grants with callbacks into an isolated replay.

    ``grant(pid, ts, metadata)`` must synchronously insert one physical 2422 and
    update inventory history, without a buy event, visit or gold transaction.
    The callback must succeed before the scheduler records a completed grant.
    Role and quest callbacks must expose only current causal state.
    """

    def __init__(self, participants: Iterable[Mapping], dragon,
                 inventory_for: Callable[[int], Iterable[int]],
                 role_for: Callable[[int], str],
                 completion_for: Callable[[int], bool | None],
                 grant: Callable[[int, int, dict], None]):
        self.dragon = dragon
        self.inventory_for = inventory_for
        self.role_for = role_for
        self.completion_for = completion_for
        self.grant_callback = grant
        self.teams, self.wearers = {}, {}
        for participant in participants:
            pid = participant["participantId"]
            if type(pid) is not int or pid <= 0 or pid in self.teams:
                raise ValueError("Invalid or duplicate participant id")
            team = participant["teamId"]
            if type(team) is not int or team <= 0:
                raise ValueError("Invalid participant team")
            self.teams[pid] = team
            if _rune_selected(participant):
                self.wearers[pid] = _Wearer(pid)
        self.grants: list[dict] = []
        self.boundaries: list[dict] = []
        self.diagnostics: list[dict] = []
        self._seen_kills: set[tuple[int, int, int]] = set()
        self._latest_timestamp = -1
        self._before_timestamp: int | None = None
        self._source_batch_index = 0

    def _note(self, wearer, code, timestamp, *, invalid=False, **detail):
        entry = {"participant_id": wearer.participant_id,
                 "code": code, "timestamp": timestamp, **detail}
        wearer.invalid |= invalid
        wearer.diagnostics.append(entry)
        self.diagnostics.append(entry)

    def _attempt(self, wearer, timestamp, *, phase, source_timestamp):
        if wearer.granted_at is not None or wearer.eligible_at is None:
            return
        inventory = list(self.inventory_for(wearer.participant_id))
        kwargs = {"role": self.role_for(wearer.participant_id),
                  "bot_quest_complete": self.completion_for(wearer.participant_id)}
        before = physical_slot_state(inventory, self.dragon, **kwargs)
        after = physical_slot_state([*inventory, MAGICAL_BOOTS], self.dragon, **kwargs)
        if not before["known"] or not after["known"]:
            reason = "unknown_physical_slot_state"
        elif before["regular_slots"] > 6:
            reason = "existing_physical_slot_overflow"
        elif before["boot_copies"]:
            reason = "boot_owned_before_inferred_grant"
        elif after["regular_slots"] > 6:
            reason = "full_slots"
        else:
            reason = None
        if reason:
            if reason != wearer.pending_reason:
                self._note(wearer, reason, timestamp,
                           invalid=reason != "full_slots", slot_state=before)
            wearer.pending_reason = reason
            if reason == "full_slots" and wearer.first_full_slot_at is None:
                wearer.first_full_slot_at = timestamp
            return
        metadata = {
            "participant_id": wearer.participant_id,
            "timestamp": timestamp,
            "reason": "first_available_slot" if wearer.first_full_slot_at is not None else "timer_eligibility",
            "eligible_at": wearer.eligible_at,
            "nominal_deadline_at_grant": wearer.nominal_deadline,
            "takedowns_before_grant": wearer.takedowns,
            "full_slot_wait_ms": (timestamp - wearer.first_full_slot_at
                                  if wearer.first_full_slot_at is not None else 0),
            "slot_state_before": before,
            "slot_state_after": after,
            "phase": phase,
            "current_source_timestamp": source_timestamp,
            "source_batch_index": self._source_batch_index,
            "item_id": MAGICAL_BOOTS,
            "event_kind": "system_grant",
            "gold_delta": 0,
            "purchase_cost": 0,
            "purchase_label": False,
            "grant_time_observed": False,
            "delivery_tick_verified": False,
            "hypothesis": DELIVERY_HYPOTHESIS,
            "timer_version": TIMER_VERSION,
        }
        self.grant_callback(wearer.participant_id, timestamp, deepcopy(metadata))
        wearer.granted_at = timestamp
        wearer.pending_reason = None
        self.grants.append(metadata)
        self.boundaries.append({
            "participant_id": wearer.participant_id,
            "timestamp": timestamp, "kind": "inferred_grant",
            "phase": phase, "source_batch_index": self._source_batch_index,
        })

    def advance_before(self, timestamp: int) -> None:
        """Fire only deadlines strictly before the next raw event timestamp.

        A pending full-slot grant cannot progress here: inventories change only
        in a raw batch, whose post-batch callback checks the newly freed space.
        Calling this at a decision anchor gives its strictly earlier timer state.
        """
        timestamp = _timestamp(timestamp)
        if timestamp < self._latest_timestamp:
            raise ValueError("Cannot replay timestamps backwards")
        self._latest_timestamp = timestamp
        self._before_timestamp = timestamp
        due = sorted((w.nominal_deadline, w.participant_id, w)
                     for w in self.wearers.values()
                     if w.eligible_at is None and w.granted_at is None
                     and w.nominal_deadline < timestamp)
        for deadline, _pid, wearer in due:
            # A causal takedown crossing its revised deadline establishes
            # eligibility immediately in observe_batch, never retroactively.
            wearer.eligible_at = deadline
            self.boundaries.append({"participant_id": wearer.participant_id,
                                    "timestamp": deadline, "kind": "eligibility"})
            self._attempt(wearer, deadline, phase="before_raw_timestamp",
                          source_timestamp=timestamp)

    def observe_batch(self, timestamp: int, events: Iterable[Mapping]) -> None:
        """Observe one already-applied contiguous raw batch, then try grants."""
        timestamp = _timestamp(timestamp)
        if self._before_timestamp != timestamp or timestamp < self._latest_timestamp:
            raise ValueError("advance_before(ts) is required before applying each raw timestamp")
        batch = list(events)
        if any(event.get("timestamp") != timestamp for event in batch):
            raise ValueError("Every event in a raw batch must have the supplied timestamp")
        self._source_batch_index += 1
        for event in batch:
            if event.get("type") != "CHAMPION_KILL":
                continue
            victim, killer = event.get("victimId"), event.get("killerId")
            if victim not in self.teams:
                continue
            identity = (timestamp, victim, killer)
            if identity in self._seen_kills:
                continue
            self._seen_kills.add(identity)
            credited = set(event.get("assistingParticipantIds") or [])
            credited.add(killer)
            for pid in credited:
                if pid not in self.wearers or pid == victim or self.teams[pid] == self.teams[victim]:
                    continue
                wearer = self.wearers[pid]
                if wearer.granted_at is not None:
                    continue
                wearer.takedowns += 1
                wearer.nominal_deadline = BASE_ELIGIBILITY_MS - TAKEDOWN_REDUCTION_MS * wearer.takedowns
        for wearer in sorted(self.wearers.values(), key=lambda w: w.participant_id):
            if wearer.granted_at is not None:
                continue
            if wearer.eligible_at is None and wearer.nominal_deadline <= timestamp:
                wearer.eligible_at = timestamp
                self.boundaries.append({"participant_id": wearer.participant_id,
                                        "timestamp": timestamp, "kind": "eligibility"})
            self._attempt(wearer, timestamp, phase="after_raw_batch",
                          source_timestamp=timestamp)

    def state(self, participant_id: int) -> dict:
        if participant_id not in self.teams:
            raise ValueError("Unknown participant id")
        wearer = self.wearers.get(participant_id)
        if wearer is None:
            return {"participant_id": participant_id, "selected": False,
                    "phase": "not_selected", "invalid": False}
        return deepcopy({
            "participant_id": participant_id, "selected": True,
            "phase": "granted" if wearer.granted_at is not None else
                     "pending" if wearer.eligible_at is not None else "pre_eligibility",
            "nominal_deadline": wearer.nominal_deadline,
            "takedowns": wearer.takedowns,
            "eligible_at": wearer.eligible_at,
            "granted_at": wearer.granted_at,
            "first_full_slot_at": wearer.first_full_slot_at,
            "pending_reason": wearer.pending_reason,
            "invalid": wearer.invalid,
            "diagnostics": wearer.diagnostics,
            "delivery_hypothesis": DELIVERY_HYPOTHESIS,
        })

    def snapshot(self) -> list[dict]:
        return [self.state(pid) for pid in sorted(self.teams)]
