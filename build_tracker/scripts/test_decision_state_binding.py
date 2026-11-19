"""decision_state_binding must accept an EMPTY decision inventory.

2026-09-23: the 16.18 v4 audit failed on exactly one of 1,249,675 shop visits
(KR_8388382904, ts 1368998): Zed sold all six items for 5255 g and then bought
6676. The reconstruction wrote decision_inventory = [] — the correct state —
and the gate read `[] or inventory_before` as "field absent", substituted the
pre-sale inventory and failed correct data. These cases pin the rule: [] is a
state; only a missing field falls back; a freed slot must have been paid for.

    .venv/Scripts/python.exe scripts/test_decision_state_binding.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from audit_patch_corpus import decision_state_binding  # noqa: E402


def items(*ids):
    return [{"item_id": i, "name": str(i)} for i in ids]


BEFORE = items(1054, 2520, 3171, 3364, 6699, 1036)
SORTED_BEFORE = sorted([1054, 2520, 3171, 3364, 6699, 1036])

# 1. The real failure: everything sold before the first purchase.
ok, detail = decision_state_binding({
    "inventory_before": BEFORE, "decision_inventory": [],
    "opening_sales": [1036, 1054, 2520, 3171, 3364, 6699], "opening_sale_gold": 5255})
assert ok and detail is None, detail

# 2. Legacy payload without the field: falls back to the pre-sale snapshot.
ok, _ = decision_state_binding({"inventory_before": BEFORE, "opening_sale_gold": 0})
assert ok

# 3. Partial sale, consistent.
ok, _ = decision_state_binding({
    "inventory_before": BEFORE, "decision_inventory": items(2520, 3171, 3364, 6699, 1036),
    "opening_sales": [1054], "opening_sale_gold": 700})
assert ok

# 4. Declared sales not removed from the decision inventory: the real invariant, still fatal,
#    and the detail now carries both inventories so the next failure needs no DB dig.
ok, detail = decision_state_binding({
    "inventory_before": BEFORE, "decision_inventory": BEFORE,
    "opening_sales": [1054], "opening_sale_gold": 700})
assert not ok and detail == {
    "opening_sales": [1054], "opening_sale_gold": 700,
    "inventory_before": SORTED_BEFORE, "decision_inventory": SORTED_BEFORE}, detail

# 5. A freed slot must have been paid for.
ok, _ = decision_state_binding({
    "inventory_before": BEFORE, "decision_inventory": items(2520, 3171, 3364, 6699, 1036),
    "opening_sales": [1054], "opening_sale_gold": 0})
assert not ok

# 6. Proceeds without a freed modeled slot (v3 precedent, 012e562): legitimate.
ok, _ = decision_state_binding({
    "inventory_before": BEFORE, "decision_inventory": BEFORE,
    "opening_sales": [], "opening_sale_gold": 15})
assert ok

# 7. Selling something never held is impossible.
ok, _ = decision_state_binding({
    "inventory_before": BEFORE, "decision_inventory": BEFORE,
    "opening_sales": [3031], "opening_sale_gold": 1000})
assert not ok

# 8. An empty decision inventory without sales is consistent only with an empty pre-visit inventory.
ok, _ = decision_state_binding({
    "inventory_before": BEFORE, "decision_inventory": [], "opening_sales": [], "opening_sale_gold": 0})
assert not ok
ok, _ = decision_state_binding({
    "inventory_before": [], "decision_inventory": [], "opening_sales": [], "opening_sale_gold": 0})
assert ok

print("ok decision_state_binding: [] is a state, a missing field falls back, a freed slot must be paid for")
