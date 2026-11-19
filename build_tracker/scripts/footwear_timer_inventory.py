"""Isolated inventory extensions for the timer experiment, not active replay.

Keep the base batch, quest, transformation and undo logic unchanged. A temporary
inventory list intercepts only its purchase append for the three reviewed
elixirs. Pinned consumed+consumeOnFull flags authorize immediate consumption
only when exact current physical occupancy is six slots and the new elixir
would need a seventh. This is independent of Magical Footwear grant timing.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy

from app.inventory import InventoryReplay
from footwear_economics import components_of
from footwear_timer import physical_slot_state

INVENTORY_VERSION = "footwear-timer-inventory-elixirs-prototype-v1"
REVIEWED_ELIXIRS = frozenset({2138, 2139, 2140})


class _SourceBatch(list):
    """Expose the exact current event while the unchanged base loops over it."""

    def __init__(self, events, replay):
        super().__init__(events)
        self.replay = replay

    def __iter__(self):
        try:
            for event in super().__iter__():
                self.replay._active_raw_event = event
                yield event
        finally:
            self.replay._active_raw_event = None


class _PurchaseInventory(list):
    def __init__(self, inventory, replay):
        super().__init__(inventory)
        self.replay = replay

    def append(self, item):
        if not self.replay._consume_new_elixir(self, item):
            super().append(item)


class FootwearTimerInventoryReplay(InventoryReplay):
    """Normal replay plus physical 2422 recipes and full-inventory elixirs.

    ``role_for`` is a current, fixed roster-role callback (or owner->role map).
    Missing/unknown roles disable the elixir shortcut. Completion comes from
    the base replay's causal bot_complete set. The original inventory list
    object is preserved for callers holding references to it.

    Auto consumption records carry no gold refund and no buy target; unchanged
    caller-side pricing continues to charge the observed raw purchase.
    """

    def __init__(self, inventories, dragon, *, role_for: Callable | Mapping | None = None):
        super().__init__(inventories, dragon)
        if dragon.version != "16.17.1":
            raise ValueError("Elixir extension is reviewed only for pinned 16.17.1")
        self.role_for = role_for
        self.auto_consumptions: list[dict] = []
        self._active_owner = None
        self._active_ts = None
        self._active_raw_event = None

    def _consumes(self, purchase, component):
        return super()._consumes(purchase, component) or (
            purchase != component and component in components_of(purchase, self.dragon))

    def _consume_new_elixir(self, inventory, item):
        event = self._active_raw_event
        if (item not in REVIEWED_ELIXIRS or self._active_owner is None
                or event is None or event.get("type") != "ITEM_PURCHASED"
                or event.get("itemId") != item or event.get("timestamp") != self._active_ts):
            return False
        raw = self.dragon.item(item) or {}
        if raw.get("consumed") is not True or raw.get("consumeOnFull") is not True:
            return False
        if callable(self.role_for):
            role = self.role_for(self._active_owner)
        elif isinstance(self.role_for, Mapping):
            role = self.role_for.get(self._active_owner)
        else:
            return False
        if role not in {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}:
            return False
        kwargs = {"role": role, "bot_quest_complete": self._active_owner in self.bot_complete}
        before = physical_slot_state(inventory, self.dragon, **kwargs)
        after = physical_slot_state([*inventory, item], self.dragon, **kwargs)
        if (not before["known"] or not after["known"]
                or before["regular_slots"] != 6 or after["regular_slots"] != 7):
            return False
        self.auto_consumptions.append({
            "participant_id": self._active_owner,
            "timestamp": self._active_ts,
            "item_id": item,
            "event_kind": "elixir_auto_consumption_on_purchase",
            "reason": "reviewed_consume_on_full",
            "purchase_label": False,
            "gold_refund": 0,
            "raw_purchase_remains_chargeable": True,
            "inventory_before": list(inventory),
            "slot_state_before": before,
            "slot_state_if_retained": after,
            "inventory_version": INVENTORY_VERSION,
        })
        return True

    def apply_batch(self, owner, ts, batch, undo_restores=None):
        if self._active_owner is not None:
            raise RuntimeError("Nested inventory batch replay is unsupported")
        original = self.inventories[owner]
        proxy = _PurchaseInventory(original, self)
        events = list(batch)
        self._active_owner, self._active_ts = owner, ts
        self.inventories[owner] = proxy
        try:
            return super().apply_batch(owner, ts, _SourceBatch(events, self), undo_restores)
        finally:
            # Use builtin slice assignment: a restored elixir must not pass
            # through the purchase interception, including after exceptions.
            original[:] = proxy
            self.inventories[owner] = original
            self._active_owner = self._active_ts = self._active_raw_event = None

    def auto_consumption_records(self):
        return deepcopy(self.auto_consumptions)
