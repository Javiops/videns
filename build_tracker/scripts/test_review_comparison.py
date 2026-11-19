"""Review display legality: synthetic rows and pinned static items only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ddragon import dragon_for_patch
from export_review_demo import _peer_comparison

dragon = dragon_for_patch("16.18")


def compare(held, item, budget):
    query = {"ts": 600000, "gold_est": budget, "level": 8, "inventory": held, "role": "MIDDLE"}
    rows = [{"ts": 600000, "gold_est": budget, "level": 8,
             "inventory": held, "label_counts": {str(item): 1}} for _ in range(40)]
    return _peer_comparison(query, rows, dragon)


assert compare([3009], 1001, 500)["status"] == "unavailable", "must not offer a second pair of boots"
full = [1056, 3009, 1082, 1028, 1052, 2031]
assert compare(full, 1033, 2000)["status"] == "unavailable", "the refillable potion occupies the sixth slot"
assert compare(full, 3803, 2000)["status"] == "descriptive", "a legal combine can reuse the Ruby Crystal slot"
print("review comparison legality passed")
