"""Isolated 16.17 Magical Footwear economics; not wired into the active pipeline.

No grant timing is inferred here. A caller must separately establish whether a
grant is causally observable. Source evidence and pinned payload hashes live in
data/acceptance/footwear_repair_20260911/sources/mechanics-evidence.md.

Riot 16.17.1 item.json: physical item 2422 sells for 90; it replaces the
300-value 1001 component in its listed boot upgrades. Never normalize the raw
inventory to 1001: doing so loses sale and undo identity.
"""

from __future__ import annotations

from collections import Counter
import hashlib

from app.ddragon import DataDragon
from app.shop_econ import PINK, SAVE_ITEM, inventory_shape, payload_ids

MAGICAL_FOOTWEAR = 2422
ORDINARY_BOOTS = 1001
FOOTWEAR_ECONOMICS_VERSION = "footwear-economics-16.17-v1"


class FootwearDataDragon:
    """Read-only facade: keep 2422 as context, never as a purchase target.

    Construction performs no downloads or static-cache writes. The caller owns
    the underlying DataDragon. All unchanged accessors delegate to that object;
    modified classification dictionaries and the signature belong to this view.
    """

    def __init__(self, source: DataDragon):
        if source.version != "16.17.1":
            raise ValueError("Footwear prototype is reviewed only for 16.17.1")
        raw = source.item(MAGICAL_FOOTWEAR) or {}
        gold = raw.get("gold") or {}
        if (gold.get("total"), gold.get("sell"), gold.get("purchasable")) != (300, 90, False):
            raise ValueError("Unreviewed Magical Footwear economics")
        self._source = source
        self._classification: dict[int, dict] = {}
        self.signature = hashlib.sha256(
            f"{source.signature}:{FOOTWEAR_ECONOMICS_VERSION}".encode("utf-8")
        ).hexdigest()

    def __getattr__(self, name):
        return getattr(self._source, name)

    def classify(self, item_id: int) -> dict:
        if item_id not in self._classification:
            info = dict(self._source.classify(item_id))
            if item_id == MAGICAL_FOOTWEAR:
                info.update(skip=False, target_skip=True, is_basic_boots=True, is_boots=False)
            self._classification[item_id] = info
        return self._classification[item_id]


def context_inventory(items: list[int], dragon: DataDragon) -> list[int]:
    """Keep all modeled copies, including physical 2422; do not truncate."""
    return [int(i) for i in items if i and not dragon.classify(int(i))["skip"]]


def purchase_targets(items: list[int], dragon: DataDragon) -> list[int]:
    """Free footwear is an owned item, not an imitation-learning buy label."""
    return [int(i) for i in items if i and not dragon.classify(int(i))["target_skip"]]


def grant_economics() -> dict:
    """Metadata only: this function neither infers nor inserts a grant."""
    return {
        "item_id": MAGICAL_FOOTWEAR,
        "event_kind": "system_grant",
        "gold_delta": 0,
        "purchase_cost": 0,
        "purchase_label": False,
        "economics_version": FOOTWEAR_ECONOMICS_VERSION,
    }


def _physical_component(required: int, parent: int, owned: Counter, dragon: DataDragon) -> int | None:
    if owned.get(required, 0) > 0:
        return required
    # Limit equivalence to the boot recipes listed on physical 2422. Recursive
    # upgrades reach one of these parent recipes naturally, without rewriting
    # static recipes or mapping item ids in the inventory.
    upgrades = (dragon.item(MAGICAL_FOOTWEAR) or {}).get("into") or []
    if (required == ORDINARY_BOOTS and owned.get(MAGICAL_FOOTWEAR, 0) > 0
            and str(parent) in {str(i) for i in upgrades}):
        return MAGICAL_FOOTWEAR
    return None


def combine_cost(item_id: int, owned: Counter, dragon: DataDragon, consume: bool = False) -> int:
    """Shared recipe semantics with physical 2422 substitution at boot leaves.

    As in the production helper, this quotes recipe costs, not purchase
    permission. Use try_purchase for candidate legality. consume=True removes
    the actual physical component so a caller can preserve an undo ledger.
    """
    available = owned.copy()
    if any(count < 0 for count in available.values()):
        raise ValueError("Negative owned item count")

    def rec(iid: int) -> int:
        recipe = dragon.from_ids(iid)
        gold = dragon.gold_block(iid)
        if not recipe:
            return gold["total"] or gold["base"]
        cost = gold["base"]
        for required in recipe:
            physical = _physical_component(required, iid, available, dragon)
            if physical is None:
                cost += rec(required)
            else:
                available[physical] -= 1
        return cost

    cost = rec(item_id)
    if consume:
        owned.clear()
        owned.update(available)
    return cost


def components_of(item_id: int, dragon: DataDragon) -> set[int]:
    """Recipe descendants including physical equivalents accepted by cost()."""
    result: set[int] = set()
    pending = [item_id]
    visited: set[int] = set()
    while pending:
        parent = pending.pop()
        if parent in visited:
            continue
        visited.add(parent)
        for required in dragon.from_ids(parent):
            result.add(required)
            if _physical_component(required, parent, Counter({MAGICAL_FOOTWEAR: 1}), dragon) == MAGICAL_FOOTWEAR:
                result.add(MAGICAL_FOOTWEAR)
            pending.append(required)
    return result


def is_blocked(
    item_id: int, inventory: list[int], dragon: DataDragon, *,
    magical_footwear_selected: bool = False, footwear_granted: bool | None = None,
) -> bool:
    """Boot-family restrictions, with an explicit fail-closed pregrant state."""
    inv = [int(i) for i in inventory if i]
    candidate = dragon.classify(item_id)
    if item_id == MAGICAL_FOOTWEAR or candidate.get("target_skip"):
        return True
    if candidate.get("is_completed") and item_id in inv:
        return True
    if candidate.get("is_boots") or candidate.get("is_basic_boots"):
        if magical_footwear_selected and footwear_granted is not True:
            return True
        ingredients = components_of(item_id, dragon)
        for iid in inv:
            info = dragon.classify(iid)
            if (info.get("is_boots") or info.get("is_basic_boots")) and iid not in ingredients:
                return True
    return False


def try_purchase(
    item_id: int, inventory: list[int], budget: float, dragon: DataDragon, *,
    role: str = "", bot_quest_complete: bool | None = None,
    unmodeled_regular_slots: int = 0, role_slot_inventory_unknown: bool = False,
    protected: Counter | None = None, magical_footwear_selected: bool = False,
    footwear_granted: bool | None = None,
) -> tuple[int, Counter] | None:
    """Prototype purchase simulation using unchanged production slot rules."""
    if (item_id == SAVE_ITEM or not dragon.item(item_id)
            or is_blocked(item_id, inventory, dragon,
                          magical_footwear_selected=magical_footwear_selected,
                          footwear_granted=footwear_granted)):
        return None
    candidate = dragon.classify(item_id)
    if role_slot_inventory_unknown and (
        (role == "UTILITY" and item_id in PINK)
        or (role == "BOTTOM" and (candidate.get("is_boots") or candidate.get("is_basic_boots")))
    ):
        return None
    owned = Counter(int(i) for i in inventory if i)
    reserved = protected or Counter()
    if not reserved <= owned:
        raise ValueError("Previously displayed purchases are missing from simulated inventory")
    owned.subtract(reserved)
    cost = combine_cost(item_id, owned, dragon, consume=True)
    owned.update(reserved)
    owned[item_id] += 1
    if cost > budget:
        return None
    shape = inventory_shape(list(owned.elements()), dragon, role=role,
                            bot_quest_complete=bot_quest_complete)
    if shape["regular_slots"] + max(0, unmodeled_regular_slots) > 6 or shape["boot_copies"] > 1:
        return None
    for ward in PINK:
        if owned[ward] > int((dragon.item(ward) or {}).get("stacks") or 1):
            return None
    return cost, +owned


def buy_cost(item_id: int, inventory: list[int], dragon: DataDragon) -> int:
    return combine_cost(item_id, Counter(int(i) for i in inventory if i), dragon)


def has_direct_components(item_id: int, owned: Counter, dragon: DataDragon) -> bool:
    """Whether every direct recipe component is held, preserving multiplicity.

    This intentionally does not count partial descendants: inventory_state and
    decision_kind distinguish a ready combine from a partially owned recipe.
    Full recursive affordability is handled by combine_cost instead.
    """
    available = owned.copy()
    recipe = dragon.from_ids(item_id)
    for required in recipe:
        physical = _physical_component(required, item_id, available, dragon)
        if physical is None:
            return False
        available[physical] -= 1
    return bool(recipe)


def inventory_state(inventory: list[int], gold: int, dragon: DataDragon) -> dict:
    """Production feature semantics with 2422-equivalent direct components."""
    inv = [int(i) for i in inventory if i]
    owned = Counter(inv)
    affordable = [
        (base, iid) for iid, _recipe, base in dragon.combine_recipes()
        if base <= gold and has_direct_components(iid, owned, dragon)
    ]
    cheapest = min((cost for cost, _iid in affordable), default=0)
    return {
        "can_complete": 1 if affordable else 0,
        "n_completable": len(affordable),
        "cheapest_complete": cheapest,
        "gold_after_complete": gold - cheapest if cheapest else gold,
        "n_inventory": len(inv),
    }


def decision_kind(bought: list[int], inventory: list[int], dragon: DataDragon) -> str:
    """Categorize retained purchase targets; a free grant never becomes a buy.

    As in production, an empty target set has category 'save'. This helper does
    not emit a shop event or no-buy label: a grant-only event must be excluded
    by the caller rather than turned into a training decision.
    """
    ids = purchase_targets(bought, dragon)
    if not ids or all(iid in PINK for iid in ids):
        return "save"
    owned = Counter(int(i) for i in inventory if i)
    for iid in ids:
        if dragon.classify(iid).get("is_completed") or has_direct_components(iid, owned, dragon):
            return "complete"
    if any(dragon.from_ids(iid) for iid in ids):
        return "component"
    return "start"


def net_spent(before: list | None, bought: list | None, consumed: list | None, dragon: DataDragon) -> int:
    """Diagnostic accounting only; never derive a predecision wallet from it."""
    remaining = Counter(payload_ids(consumed))
    spent = 0
    for iid in sorted(payload_ids(bought), key=lambda i: dragon.classify(i).get("depth") or 0, reverse=True):
        if iid == MAGICAL_FOOTWEAR:
            raise ValueError("A system grant cannot be recorded as a purchase")
        spent += combine_cost(iid, remaining, dragon, consume=True)
    return spent - sum(dragon.gold_block(i)["sell"] * n for i, n in remaining.items())
