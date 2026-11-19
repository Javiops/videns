"""Opening-sale fields must never reach the tensors.

`opening_sale_gold` (and any sale identity) is already reflected where it
belongs — inside `gold_est` and the row's `inventory`/affordability block. If
the raw fields leaked into featurization they would add a second, redundant
path for the same information and a selection artifact: in the corpus,
P(buy | just sold) = 1 by construction (sale-only visits never export), so a
"just sold" indicator is a training-only shortcut the live regime cannot match
in distribution. stream_rows' whitelist drops unknown keys; this pins it.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import train_prefix as tp

ROW = {
    "match_id": "KR_T", "champion": "Ahri", "champion_id": 103, "role": "MIDDLE",
    "team_id": 100, "gold": 400, "gold_est": 900, "gold_est_version": "prequential-v3",
    "total_gold": 4000, "ally_obj": {}, "enemy_obj": {}, "level": 6, "ts": 600000,
    "kills": 0, "deaths": 0, "inventory": [1056], "label_id": 1058, "label_ids": [1058],
    "can_complete": 0, "n_completable": 0, "cheapest_complete": 0,
    "gold_after_complete": 400, "n_inventory": 1, "decision": "component",
    "save_kind": "build_spend", "others": [],
    # The fields under test:
    "opening_sale_gold": 350, "opening_sales": [1036],
    "decision_inventory": [{"item_id": 1056}],
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        path.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
        streamed = list(tp.stream_rows(path))
    assert len(streamed) == 1
    row = streamed[0]
    for key in ("opening_sale_gold", "opening_sales", "decision_inventory"):
        assert key not in row, f"{key} survived stream_rows and could be featurized"
    assert row.get("gold_est") == 900 and row.get("inventory") == [1056]
    print("ok isolation: opening-sale fields are dropped before featurization")


if __name__ == "__main__":
    main()
