import json

import pytest

from app import inventory
from app.config import game_dir

PROTO = "EUW1_7983109023"


def tl(events):
    return {"info": {"frames": [{"timestamp": 0, "events": events,
                                 "participantFrames": {}}]}}


def ev(kind, t, pid, **kw):
    return {"type": kind, "timestamp": t, "participantId": pid, **kw}


def test_purchase_sell_destroy_undo():
    events = [
        ev("ITEM_PURCHASED", 1000, 1, itemId=1001),
        ev("ITEM_PURCHASED", 2000, 1, itemId=1036),
        ev("ITEM_SOLD", 3000, 1, itemId=1001),
        ev("ITEM_PURCHASED", 4000, 1, itemId=2003),
        ev("ITEM_DESTROYED", 5000, 1, itemId=2003),      # poción usada
        ev("ITEM_PURCHASED", 6000, 1, itemId=3134),
        ev("ITEM_UNDO", 7000, 1, beforeId=3134, afterId=0),
    ]
    inv = inventory.inventories_at(tl(events), 10.0)
    assert inventory.items_list(inv[1]) == [1036]


def test_participant_zero_ignored():
    events = [ev("ITEM_PURCHASED", 0, 0, itemId=3865)]
    inv = inventory.inventories_at(tl(events), 10.0)
    assert all(not c for c in inv.values())


def test_support_chain_upgrade():
    events = [
        ev("ITEM_PURCHASED", 1000, 5, itemId=3865),
        ev("ITEM_DESTROYED", 60000, 5, itemId=3865),     # sube a 3866
    ]
    inv = inventory.inventories_at(tl(events), 100.0)
    assert inventory.items_list(inv[5]) == [3866]


def test_gold_value():
    from collections import Counter
    assert inventory.gold_value(Counter({1036: 2}), {1036: 350}) == 700


@pytest.mark.skipif(not (game_dir(PROTO) / "mv5_timeline.json").exists(),
                    reason="partida prototipo no presente")
def test_final_inventories_vs_match_real_game():
    g = game_dir(PROTO)
    timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    try:
        from app import ddragon
        v = ddragon.version_for_patch("16.18")
        boots = {iid for iid, it in ddragon.items(v)["data"].items()
                 if "Boots" in it.get("tags", [])}
        boots = {int(i) for i in boots}
    except Exception:
        pytest.skip("ddragon no disponible (sin red ni cache)")
    diff = inventory.final_diff_vs_match(timeline, match, boots_ids=boots)
    bad = {pid: d for pid, d in diff.items() if not d["ok"]}
    # sobre lo comparable (sin visión ni botas, support canonicalizado) la
    # reconstrucción debe clavar el final de los 10
    assert not bad, f"inventarios finales que no cuadran: {bad}"
