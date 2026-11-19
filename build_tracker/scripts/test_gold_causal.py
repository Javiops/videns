"""gold_est must be causal: nothing after the decision may change it.

This is stronger than the wallet identity, which only checks that reconstructed
spending reconciles two frames — it cannot detect that the estimate READ one of
those frames from the future. Version "leaky-v1" passed the identity check and
still leaked: it prorated a residual computed from the frame AFTER the visit.

Run: .venv\\Scripts\\python.exe scripts\\test_gold_causal.py
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ddragon import DataDragon
from app.reconstruct import GOLD_EST_VERSION, reconstruct_game

dragon = DataDragon()

MATCH = {
    "metadata": {"matchId": "KR_GOLD_CAUSAL"},
    "info": {
        "gameCreation": 1,
        "gameDuration": 1800,
        "gameVersion": "16.17.1.1",
        "queueId": 420,
        "participants": [
            {
                "participantId": 8, "puuid": "shopper", "championId": 103,
                "championName": "Ahri", "teamId": 200, "teamPosition": "MIDDLE",
                "riotIdGameName": "a", "riotIdTagline": "KR1", "win": True,
                "kills": 2, "deaths": 0, "assists": 0,
            },
            {
                "participantId": 1, "puuid": "other", "championId": 64,
                "championName": "LeeSin", "teamId": 100, "teamPosition": "JUNGLE",
                "riotIdGameName": "b", "riotIdTagline": "KR1", "win": False,
                "kills": 0, "deaths": 2, "assists": 0,
            },
        ],
    },
}

VISIT_TS = 150_000


def frame(ts: int, current: int, total: int, events: list) -> dict:
    return {
        "timestamp": ts,
        "events": events,
        "participantFrames": {
            "8": {
                "participantId": 8, "currentGold": current, "totalGold": total,
                "level": 8, "minionsKilled": 60, "jungleMinionsKilled": 0,
            },
            "1": {
                "participantId": 1, "currentGold": 100, "totalGold": 1200,
                "level": 8, "minionsKilled": 10, "jungleMinionsKilled": 40,
            },
        },
    }


def timeline() -> dict:
    return {
        "info": {
            "frames": [
                frame(60_000, 300, 1000, [
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1056, "timestamp": 20_000},
                ]),
                frame(120_000, 500, 1600, []),
                # the visit itself lives in this frame's event list
                frame(180_000, 120, 2100, [
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1026, "timestamp": VISIT_TS},
                ]),
                frame(240_000, 400, 2600, []),
            ]
        }
    }


def visit_estimate(tl: dict) -> tuple[int, dict]:
    _game, events = reconstruct_game(MATCH, tl, dragon)
    shop = next(
        e for e in events
        if e["type"] == "shop" and e["participant_id"] == 8 and e["ts"] == VISIT_TS
    )
    return shop["gold_est"], shop


base_est, base_event = visit_estimate(timeline())
assert base_event["gold_est_version"] == GOLD_EST_VERSION, base_event["gold_est_version"]
assert base_event["gold_gap"]["frame_ts"] == 120_000, base_event["gold_gap"]
assert base_event["gold_gap"]["prior_residual"] is not None, "prior-window forecast not applied"
print(f"baseline gold_est = {base_est} ({GOLD_EST_VERSION}), gap {base_event['gold_gap']}")

# 1. The frame AFTER the visit may not influence the estimate.
future_frame = timeline()
future_frame["info"]["frames"][3] = frame(240_000, 9_999, 99_999, [])
est, _ = visit_estimate(future_frame)
assert est == base_est, f"post-visit frame changed gold_est: {est} != {base_est}"

# 2. Neither may the frame the visit's own events were listed under, when its
#    stats move (that frame is stamped at 180s, i.e. after t).
own_frame = timeline()
own_frame["info"]["frames"][2]["participantFrames"]["8"]["currentGold"] = 4_242
own_frame["info"]["frames"][2]["participantFrames"]["8"]["totalGold"] = 42_424
est, _ = visit_estimate(own_frame)
assert est == base_est, f"the visit's enclosing (later) frame changed gold_est: {est} != {base_est}"

# 3. Events after t may not influence it: a fat bounty and a later purchase.
after_events = timeline()
after_events["info"]["frames"][2]["events"].extend([
    {"type": "CHAMPION_KILL", "killerId": 8, "victimId": 1, "timestamp": 170_000,
     "bounty": 300, "shutdownBounty": 600},
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1036, "timestamp": 175_000},
])
est, _ = visit_estimate(after_events)
assert est == base_est, f"post-visit events changed gold_est: {est} != {base_est}"

# 4. Control — the test must be able to fail. Pre-decision information DOES
#    move the estimate, otherwise assertions 1-3 would be vacuous.
prior = timeline()
prior["info"]["frames"][0]["participantFrames"]["8"]["totalGold"] = 1_450  # smaller prior income
est_prior, _ = visit_estimate(prior)
assert est_prior != base_est, "prior-window income does not affect the estimate (term is dead)"

pre_bounty = timeline()
pre_bounty["info"]["frames"][2]["events"].insert(0, {
    "type": "CHAMPION_KILL", "killerId": 8, "victimId": 1, "timestamp": 130_000,
    "bounty": 300, "shutdownBounty": 0,
})
est_bounty, _ = visit_estimate(pre_bounty)
assert est_bounty == base_est + 300, f"pre-visit bounty not credited: {est_bounty} vs {base_est}"

pre_spend = timeline()
pre_spend["info"]["frames"][2]["events"].insert(0, {
    "type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1036, "timestamp": 130_000,
})
est_spend, _ = visit_estimate(pre_spend)
assert est_spend < base_est, f"pre-visit spend not debited: {est_spend} vs {base_est}"

# 5. The visit's own purchase (same timestamp) is never debited.
assert base_event["gold_gap"]["spent"] == 0, base_event["gold_gap"]

print("ok gold_est is causal: future frames and events cannot move it")
print(f"   controls: prior income {est_prior}, +bounty {est_bounty}, -spend {est_spend}")


# ---------------------------------------------------------------------------
# Opening-sale pins (prequential-v3): a sale strictly before the visit's first
# purchase frees its slot and credits its proceeds; nothing else may.

SELL_1036 = int(dragon.gold_block(1036)["sell"])


def owns_1036(tl: dict) -> dict:
    tl["info"]["frames"][0]["events"].append(
        {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1036, "timestamp": 25_000})
    return tl


def visit_events(tl: dict, events: list) -> dict:
    tl["info"]["frames"][2]["events"] = events
    return tl


def payload_ids(payload: list) -> list:
    return sorted(int(i["item_id"]) for i in payload or [])


# Reference: same ownership history, purchase-only visit at the same t.
ref_est, ref_event = visit_estimate(visit_events(owns_1036(timeline()), [
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1026, "timestamp": VISIT_TS},
]))
assert ref_event["opening_sales"] == [] and ref_event["opening_sale_gold"] == 0
assert payload_ids(ref_event["decision_inventory"]) == payload_ids(ref_event["inventory_before"])

# 6. Opening sale: exact arithmetic — proceeds credited, slot freed,
#    the visit's own buys still never debited.
sale_est, sale_event = visit_estimate(visit_events(owns_1036(timeline()), [
    {"type": "ITEM_SOLD", "participantId": 8, "itemId": 1036, "timestamp": VISIT_TS},
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1026, "timestamp": VISIT_TS + 800},
]))
assert sale_est == ref_est + SELL_1036, (sale_est, ref_est, SELL_1036)
assert sale_event["opening_sales"] == [1036] and sale_event["opening_sale_gold"] == SELL_1036
expected = list(payload_ids(sale_event["inventory_before"]))
expected.remove(1036)
assert payload_ids(sale_event["decision_inventory"]) == expected, sale_event["decision_inventory"]
assert sale_event["gold_gap"]["spent"] == 0, sale_event["gold_gap"]

# 7. ANTI-LEAK: a combine's ITEM_DESTROYED precedes its purchase in the same
#    batch; destroys must never shape the decision state (they reveal the label).
tome = owns_1036(timeline())
tome["info"]["frames"][0]["events"].append(
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1052, "timestamp": 30_000})
combine_est, combine_event = visit_estimate(visit_events(tome, [
    {"type": "ITEM_DESTROYED", "participantId": 8, "itemId": 1052, "timestamp": VISIT_TS},
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 3108, "timestamp": VISIT_TS},
]))
assert combine_event["opening_sales"] == [] and combine_event["opening_sale_gold"] == 0
assert payload_ids(combine_event["decision_inventory"]) == payload_ids(combine_event["inventory_before"]), \
    "a combine destroy shaped decision_inventory: label leak"

# 8. No-future: mutating the frame after the visit moves none of the new fields.
future2 = visit_events(owns_1036(timeline()), [
    {"type": "ITEM_SOLD", "participantId": 8, "itemId": 1036, "timestamp": VISIT_TS},
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1026, "timestamp": VISIT_TS + 800},
])
future2["info"]["frames"][3] = frame(240_000, 9_999, 99_999, [])
est2, event2 = visit_estimate(future2)
assert est2 == sale_est and event2["opening_sale_gold"] == SELL_1036
assert payload_ids(event2["decision_inventory"]) == payload_ids(sale_event["decision_inventory"])

# 9. An undone opening sale cancels itself: no credit, no freed slot.
undo_est, undo_event = visit_estimate(visit_events(owns_1036(timeline()), [
    {"type": "ITEM_SOLD", "participantId": 8, "itemId": 1036, "timestamp": VISIT_TS},
    {"type": "ITEM_UNDO", "participantId": 8, "beforeId": 0, "afterId": 1036,
     "goldGain": -SELL_1036, "timestamp": VISIT_TS + 300},
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1026, "timestamp": VISIT_TS + 800},
]))
assert undo_est == ref_est, (undo_est, ref_est)
assert undo_event["opening_sales"] == [] and undo_event["opening_sale_gold"] == 0
assert payload_ids(undo_event["decision_inventory"]) == payload_ids(undo_event["inventory_before"])

print("ok opening sale: exact credit, freed slot, combine destroys and undos never leak")
