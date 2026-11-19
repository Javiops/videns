"""Tracker sobre snapshots sintéticos.

Cubre las trampas medidas sobre capturas reales (`price` es el coste de
combinación; los nombres de los eventos van sin tag; los nombres ocultos
llegan como riotId '#') y los casos del review: partidas seguidas con el mismo
roster, nombres repetidos, modos sin Barón, muertos que reviven antes del poll.
"""

from __future__ import annotations

import pytest

from app.tracker import (
    BARON_BUFF_S,
    BARON_FIRST_SPAWN_S,
    BARON_RESPAWN_S,
    DRAGON_FIRST_SPAWN_S,
    DRAGON_RESPAWN_S,
    ELDER_BUFF_S,
    ELDER_RESPAWN_S,
    LATE_START_S,
    OverlayTracker,
    item_gold,
)

# id: (price del API = base, total de Data Dragon). Filo infinito es el desajuste canónico.
ITEMS = {1036: (350, 350), 3031: (725, 3500), 2003: (50, 50), 3340: (0, 0)}


class Catalog:
    version = "test"

    @staticmethod
    def total(item_id: int):
        return ITEMS[item_id][1] if item_id in ITEMS else None

    @staticmethod
    def champion_image_id(raw, name=None):
        # como Data Dragon: el id de imagen es el sufijo del nombre crudo, salvo Wukong
        suffix = (raw or "").removeprefix("game_character_displayname_")
        return {"Wukong": "MonkeyKing"}.get(suffix, suffix) or None


def player(name, team, champ, items=(), dead=False, level=10, tag="TAG", respawn=12.0):
    return {
        "riotId": f"{name}#{tag}", "riotIdGameName": name, "riotIdTagLine": tag,
        "summonerName": f"{name}#{tag}", "team": team, "championName": champ,
        "rawChampionName": f"game_character_displayname_{champ}",
        "isDead": dead, "respawnTimer": respawn if dead else 0.0, "level": level,
        "items": [
            {"itemID": item_id, "price": ITEMS.get(item_id, (900, 0))[0], "count": count, "slot": slot}
            for slot, (item_id, count) in enumerate(items)
        ],
    }


def hidden(team, champ, dead=False):
    """Jugador con el nombre oculto, tal cual lo sirve el cliente."""
    p = player("", team, champ, dead=dead)
    p.update({"riotId": "#", "riotIdGameName": "", "riotIdTagLine": "", "summonerName": champ})
    return p


def snapshot(t, players, events=(), me="Alice", gold=0.0, mode="CLASSIC", map_number=11, first_id=0):
    data = {"gameMode": mode, "gameTime": float(t)}
    if map_number is not None:
        data["mapNumber"] = map_number
    return {
        "activePlayer": {"riotId": f"{me}#TAG", "riotIdGameName": me, "currentGold": gold},
        "allPlayers": players,
        "events": {"Events": [dict(e, EventID=first_id + i) for i, e in enumerate(events)]},
        "gameData": data,
    }


def kill(name, t, killer, **extra):
    return {"EventName": name, "EventTime": float(t), "KillerName": killer, "Assisters": [], "Stolen": "False", **extra}


def champions(holders):
    return None if holders is None else [h["champion"] for h in holders]


ROSTER = [
    player("Alice", "ORDER", "Ahri"), player("Bob", "ORDER", "Garen"), player("Cid", "ORDER", "LeeSin", dead=True),
    player("Dan", "CHAOS", "Jinx"), player("Eve", "CHAOS", "Thresh"),
]
BARON = [kill("BaronKill", 1400, "Alice", Assisters=["Bob"])]
NO_DRAGONS = {"100": 0, "200": 0, "unknown": 0}


@pytest.fixture
def catalog():
    return Catalog()


def test_item_gold_uses_catalog_totals_times_count(catalog):
    roster = [
        player("Alice", "ORDER", "Ahri", [(1036, 1), (3031, 1), (2003, 2), (3340, 1)]),
        player("Bob", "ORDER", "Garen", [(999999, 1)]),
        player("Dan", "CHAOS", "Jinx", [(3031, 1)]),
        player("Eve", "CHAOS", "Thresh", []),
    ]
    out = OverlayTracker().observe(snapshot(600, roster, gold=1018.8), catalog)
    assert out["in_game"] and out["my_team"] == 100 and out["game_time_s"] == 600 and out["game_over"] is False
    assert out["catalog_version"] == "test" and out["rift"] is True and out["map_number"] == 11
    gold = out["gold"]
    order, chaos = gold["teams"]
    by = {p["name"]: p for team in gold["teams"] for p in team["players"]}
    assert by["Alice"]["item_gold"] == 350 + 3500 + 2 * 50 + 0
    assert by["Alice"]["is_me"] and by["Alice"]["wallet"] == 1018 and by["Alice"]["approx"] is False
    assert by["Bob"]["item_gold"] == 900 and by["Bob"]["approx"] is True, "id desconocido: price del API, marcado"
    assert by["Dan"]["item_gold"] == 3500 and by["Dan"]["wallet"] is None and by["Dan"]["is_me"] is False
    assert order["item_gold"] == 3950 + 900 and chaos["item_gold"] == 3500
    assert order["approx"] is True and chaos["approx"] is False
    assert order["side"] == "ORDER" and chaos["side"] == "CHAOS"
    assert gold["lead"] == {"team_id": 100, "amount": 1350, "approx": True}, "la ventaja hereda la aproximación"
    assert gold["wallet"] == 1018
    assert [p["name"] for p in order["players"]] == ["Alice", "Bob"], "ordenado por valor"
    assert by["Alice"]["icon"] == "Ahri" and by["Dan"]["icon"] == "Jinx"


def test_rows_follow_lanes_when_the_api_gives_positions(catalog):
    def at(name, team, champ, lane, gold_item):
        p = player(name, team, champ, [(3031, 1)] if gold_item else [])
        p["position"] = lane
        return p
    roster = [
        at("Sup", "ORDER", "Thresh", "UTILITY", False), at("Mid", "ORDER", "Ahri", "MIDDLE", True),
        at("Top", "ORDER", "Garen", "TOP", False), at("Bot", "ORDER", "Jinx", "BOTTOM", True), at("Jg", "ORDER", "LeeSin", "JUNGLE", False),
        at("Rjg", "CHAOS", "Warwick", "JUNGLE", True), at("Rtop", "CHAOS", "Darius", "TOP", False),
    ]
    out = OverlayTracker().observe(snapshot(600, roster), catalog)
    blue, red = out["gold"]["teams"]
    assert [p["position"] for p in blue["players"]] == ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], "por línea, no por oro"
    assert [p["position"] for p in red["players"]] == ["TOP", "JUNGLE"]
    # sin posiciones (Practice Tool, personalizadas): por valor de objetos, y position vacía
    out = OverlayTracker().observe(snapshot(600, [player("A", "ORDER", "Ahri"), player("B", "ORDER", "Garen", [(3031, 1)])]), catalog)
    assert [p["name"] for p in out["gold"]["teams"][0]["players"]] == ["B", "A"]
    assert all(p["position"] == "" for p in out["gold"]["teams"][0]["players"])
    # una posición desconocida ("NONE" o basura) va al final, por oro
    weird = [at("X", "ORDER", "Ahri", "NONE", True), at("Y", "ORDER", "Garen", "MIDDLE", False)]
    out = OverlayTracker().observe(snapshot(600, weird), catalog)
    assert [p["name"] for p in out["gold"]["teams"][0]["players"]] == ["Y", "X"]


def test_item_gold_edge_cases(catalog):
    assert item_gold([{"itemID": "3031", "count": None, "price": 725}], catalog) == (3500, False)
    assert item_gold(None, catalog) == (0, False)
    assert item_gold([None, {"itemID": "x"}], catalog) == (0, False)
    assert item_gold([{"itemID": 3031, "count": 1, "price": 725}], None) == (725, True), "sin catálogo: price, aproximado"


def test_icons_follow_the_catalog(catalog):
    roster = [player("Alice", "ORDER", "Wukong"), player("Dan", "CHAOS", "Jinx")]
    out = OverlayTracker().observe(snapshot(1401, roster, [kill("BaronKill", 1400, "Alice")]), catalog)
    by = {p["name"]: p for team in out["gold"]["teams"] for p in team["players"]}
    assert by["Alice"]["icon"] == "MonkeyKing", "el id de imagen, no el nombre visible"
    holders = out["objectives"]["buffs"][0]["holders"]
    assert [(h["champion"], h["icon"]) for h in holders] == [("Wukong", "MonkeyKing")]
    out = OverlayTracker().observe(snapshot(1401, roster, [kill("BaronKill", 1400, "Alice")]), None)
    assert all(p["icon"] is None for team in out["gold"]["teams"] for p in team["players"]), "sin catálogo no hay iconos"
    assert out["objectives"]["buffs"][0]["holders"][0]["icon"] is None


def test_objectives_before_any_kill(catalog):
    obj = OverlayTracker().observe(snapshot(600, ROSTER), catalog)["objectives"]
    assert obj["baron"] == {"state": "spawning", "spawns_at": float(BARON_FIRST_SPAWN_S)}
    assert obj["dragon"]["kind"] == "dragon" and obj["dragon"]["state"] == "up" and obj["dragon"]["uncertain"] is False
    assert obj["dragon"]["spawns_at"] == float(DRAGON_FIRST_SPAWN_S) and obj["dragon"]["counts"] == NO_DRAGONS
    assert obj["buffs"] == []


def test_objectives_only_on_summoners_rift(catalog):
    aram = OverlayTracker().observe(snapshot(1300, ROSTER, mode="ARAM", map_number=12), catalog)
    assert aram["rift"] is False and aram["objectives"] is None and aram["gold"]["teams"][0]["players"]
    arena = OverlayTracker().observe(snapshot(1300, ROSTER, mode="CHERRY", map_number=30), catalog)
    assert arena["objectives"] is None
    practice = OverlayTracker().observe(snapshot(1300, ROSTER, mode="PRACTICETOOL", map_number=11), catalog)
    assert practice["objectives"] is not None
    no_map = OverlayTracker().observe(snapshot(1300, ROSTER, mode="CLASSIC", map_number=None), catalog)
    assert no_map["objectives"] is not None, "sin mapNumber decide el modo"
    no_map_aram = OverlayTracker().observe(snapshot(1300, ROSTER, mode="ARAM", map_number=None), catalog)
    assert no_map_aram["objectives"] is None


def test_baron_buff_holders_alive_at_kill_and_dropped_on_death(catalog):
    tracker = OverlayTracker()
    out = tracker.observe(snapshot(1401.5, ROSTER, BARON), catalog)
    buffs = out["objectives"]["buffs"]
    assert len(buffs) == 1 and buffs[0]["kind"] == "baron" and buffs[0]["team_id"] == 100 and buffs[0]["side"] == "ORDER"
    assert buffs[0]["killed_at"] == 1400 and buffs[0]["ends_at"] == 1400 + BARON_BUFF_S and buffs[0]["stolen"] is False
    assert champions(buffs[0]["holders"]) == ["Ahri", "Garen"], "Cid estaba muerto cuando murió Barón"
    assert out["objectives"]["baron"] == {"state": "spawning", "spawns_at": 1400 + BARON_RESPAWN_S}

    later = [
        player("Alice", "ORDER", "Ahri"), player("Bob", "ORDER", "Garen", dead=True), player("Cid", "ORDER", "LeeSin"),
        player("Dan", "CHAOS", "Jinx"), player("Eve", "CHAOS", "Thresh"),
    ]
    out = tracker.observe(snapshot(1450, later, BARON), catalog)
    assert champions(out["objectives"]["buffs"][0]["holders"]) == ["Ahri"], "Bob murió; Cid revivió tarde"
    out = tracker.observe(snapshot(1400 + BARON_BUFF_S + 0.5, later, BARON), catalog)
    assert out["objectives"]["buffs"] == [] and out["objectives"]["baron"]["state"] == "spawning"
    out = tracker.observe(snapshot(1400 + BARON_RESPAWN_S, later, BARON), catalog)
    assert out["objectives"]["baron"]["state"] == "up"


def test_dead_at_the_kill_but_alive_at_the_next_poll_is_not_a_holder(catalog):
    tracker = OverlayTracker()
    before = [player("Alice", "ORDER", "Ahri"), player("Cid", "ORDER", "LeeSin", dead=True, respawn=0.6), player("Dan", "CHAOS", "Jinx")]
    tracker.observe(snapshot(1399.9, before, []), catalog)
    after = [player("Alice", "ORDER", "Ahri"), player("Cid", "ORDER", "LeeSin"), player("Dan", "CHAOS", "Jinx")]
    out = tracker.observe(snapshot(1400.9, after, [kill("BaronKill", 1400.0, "Alice")]), catalog)
    assert champions(out["objectives"]["buffs"][0]["holders"]) == ["Ahri"], "Cid revivió a 1400.5, después de la kill"
    # si el temporizador ya había vencido antes de la kill, sí es portador
    tracker = OverlayTracker()
    before = [player("Alice", "ORDER", "Ahri"), player("Cid", "ORDER", "LeeSin", dead=True, respawn=0.05), player("Dan", "CHAOS", "Jinx")]
    tracker.observe(snapshot(1399.9, before, []), catalog)
    out = tracker.observe(snapshot(1400.9, after, [kill("BaronKill", 1400.0, "Alice")]), catalog)
    assert champions(out["objectives"]["buffs"][0]["holders"]) == ["Ahri", "LeeSin"]


def test_late_start_keeps_timers_but_not_holders(catalog):
    out = OverlayTracker().observe(snapshot(1400 + LATE_START_S + 1, ROSTER, BARON), catalog)
    late = out["objectives"]["buffs"][0]
    assert late["holders"] is None and late["ends_at"] == 1400 + BARON_BUFF_S and late["team_id"] == 100
    assert out["objectives"]["baron"]["spawns_at"] == 1400 + BARON_RESPAWN_S


def test_stolen_baron_belongs_to_the_stealer(catalog):
    out = OverlayTracker().observe(snapshot(1401, ROSTER, [kill("BaronKill", 1400, "Dan", Stolen="True")]), catalog)
    b = out["objectives"]["buffs"][0]
    assert b["team_id"] == 200 and b["side"] == "CHAOS" and b["stolen"] is True and champions(b["holders"]) == ["Jinx", "Thresh"]


def test_non_champion_killers(catalog):
    b = OverlayTracker().observe(
        snapshot(1401, ROSTER, [kill("BaronKill", 1400, "Minion_T100L1S64N0391", Assisters=["Eve"])]), catalog
    )["objectives"]["buffs"][0]
    assert b["team_id"] == 200 and champions(b["holders"]) == ["Jinx", "Thresh"]
    b = OverlayTracker().observe(
        snapshot(1401, ROSTER, [kill("BaronKill", 1400, "Turret_TOrder_L1_P3_1242677625_0")]), catalog
    )["objectives"]["buffs"][0]
    assert b["team_id"] is None and b["side"] is None and b["holders"] is None and b["ends_at"] == 1400 + BARON_BUFF_S


def test_tagged_names_are_only_a_fallback(catalog):
    b = OverlayTracker().observe(snapshot(1401, ROSTER, [kill("BaronKill", 1400, "Alice#TAG")]), catalog)["objectives"]["buffs"][0]
    assert b["team_id"] == 100


def test_same_game_name_on_both_teams_is_not_guessed(catalog):
    roster = [player("Faker", "ORDER", "Ahri", tag="KR1"), player("Bob", "ORDER", "Garen"),
              player("Faker", "CHAOS", "Jinx", tag="EUW"), player("Eve", "CHAOS", "Thresh")]
    b = OverlayTracker().observe(snapshot(1401, roster, [kill("BaronKill", 1400, "Faker")]), catalog)["objectives"]["buffs"][0]
    assert b["team_id"] is None and b["holders"] is None, "nombre ambiguo sin asistentes: equipo desconocido"
    b = OverlayTracker().observe(snapshot(1401, roster, [kill("BaronKill", 1400, "Faker", Assisters=["Eve"])]), catalog)["objectives"]["buffs"][0]
    assert b["team_id"] == 200 and champions(b["holders"]) == ["Jinx", "Thresh"], "un asistente resuelve"
    # el mismo nombre dentro de un equipo: portadores por identidad, no por nombre
    roster = [player("Faker", "ORDER", "Ahri", tag="KR1"), player("Faker", "ORDER", "Zed", tag="EUW"),
              player("Bob", "ORDER", "Garen"), player("Dan", "CHAOS", "Jinx")]
    tracker = OverlayTracker()
    b = tracker.observe(snapshot(1401, roster, [kill("BaronKill", 1400, "Bob")]), catalog)["objectives"]["buffs"][0]
    assert champions(b["holders"]) == ["Ahri", "Zed", "Garen"]
    roster[1] = player("Faker", "ORDER", "Zed", tag="EUW", dead=True)
    b = tracker.observe(snapshot(1450, roster, [kill("BaronKill", 1400, "Bob")]), catalog)["objectives"]["buffs"][0]
    assert champions(b["holders"]) == ["Ahri", "Garen"], "muere el Zed, no el Ahri"


def test_hidden_names_resolve_by_summoner_name_and_never_show_a_hash(catalog):
    roster = [hidden("ORDER", "Ahri"), player("Bob", "ORDER", "Garen"), hidden("CHAOS", "Jinx"), player("Eve", "CHAOS", "Thresh")]
    out = OverlayTracker().observe(snapshot(1401, roster, [kill("BaronKill", 1400, "Ahri", Assisters=["Bob"])]), catalog)
    names = {p["champion"]: p["name"] for team in out["gold"]["teams"] for p in team["players"]}
    assert names["Ahri"] == "Ahri" and names["Jinx"] == "Jinx", "nunca '#'"
    b = out["objectives"]["buffs"][0]
    assert b["team_id"] == 100 and champions(b["holders"]) == ["Ahri", "Garen"]
    assert [h["name"] for h in b["holders"]] == ["Ahri", "Bob"]


def test_dragons_soul_and_elder(catalog):
    ev = [
        kill("DragonKill", 400, "Alice", DragonType="Fire"),
        kill("DragonKill", 720, "Alice", DragonType="Earth"),
        kill("DragonKill", 1030, "Dan", DragonType="Water"),
    ]
    tracker = OverlayTracker()
    d = tracker.observe(snapshot(1031, ROSTER, ev), catalog)["objectives"]["dragon"]
    assert d["counts"] == {"100": 2, "200": 1, "unknown": 0} and d["kind"] == "dragon"
    assert d["spawns_at"] == 1030 + DRAGON_RESPAWN_S and d["state"] == "spawning"

    ev += [kill("DragonKill", 1340, "Alice", DragonType="Air"), kill("DragonKill", 1650, "Bob", DragonType="Chemtech")]
    out = tracker.observe(snapshot(1651, ROSTER, ev), catalog)
    d = out["objectives"]["dragon"]
    assert d["counts"]["100"] == 4 and d["kind"] == "elder" and d["spawns_at"] == 1650 + ELDER_RESPAWN_S
    assert out["objectives"]["buffs"] == [], "un alma no es un buff con tiempo"

    ev += [kill("DragonKill", 2010, "Dan", DragonType="Elder")]
    out = tracker.observe(snapshot(2011, ROSTER, ev), catalog)
    b, d = out["objectives"]["buffs"], out["objectives"]["dragon"]
    assert len(b) == 1 and b[0]["kind"] == "elder" and b[0]["team_id"] == 200 and b[0]["ends_at"] == 2010 + ELDER_BUFF_S
    assert champions(b[0]["holders"]) == ["Jinx", "Thresh"]
    assert d["kind"] == "elder" and d["spawns_at"] == 2010 + ELDER_RESPAWN_S

    out = tracker.observe(snapshot(2100, ROSTER, ev + [{"EventName": "GameEnd", "EventTime": 2100.0, "Result": "Win"}]), catalog)
    assert out["game_over"] is True


def test_unattributed_dragon_makes_the_next_spawn_uncertain(catalog):
    ev = [kill("DragonKill", 400, "Alice", DragonType="Fire"), kill("DragonKill", 720, "Alice", DragonType="Earth"),
          kill("DragonKill", 1030, "Alice", DragonType="Water"), kill("DragonKill", 1340, "Tibbers", DragonType="Air")]
    d = OverlayTracker().observe(snapshot(1341, ROSTER, ev), catalog)["objectives"]["dragon"]
    assert d["counts"] == {"100": 3, "200": 0, "unknown": 1}
    assert d["uncertain"] is True, "pudo ser el alma de ORDER: 5:00 o 6:00, no se sabe"
    ev2 = ev[:1] + [kill("DragonKill", 720, "Tibbers", DragonType="Earth")]
    d = OverlayTracker().observe(snapshot(721, ROSTER, ev2), catalog)["objectives"]["dragon"]
    assert d["counts"]["unknown"] == 1 and d["uncertain"] is False, "lejos del alma no cambia nada"


def test_new_game_resets(catalog):
    tracker = OverlayTracker()
    tracker.observe(snapshot(1401, ROSTER, BARON), catalog)
    out = tracker.observe(snapshot(5, ROSTER, []), catalog)   # el reloj retrocede
    assert out["game_over"] is False and out["objectives"]["buffs"] == []
    assert out["objectives"]["dragon"]["counts"] == NO_DRAGONS
    assert out["objectives"]["baron"]["spawns_at"] == float(BARON_FIRST_SPAWN_S)
    assert out["objectives"]["dragon"]["spawns_at"] == float(DRAGON_FIRST_SPAWN_S)

    tracker.observe(snapshot(1401, ROSTER, BARON), catalog)
    other = [player("Zed", "ORDER", "Zed"), player("Yi", "CHAOS", "MasterYi")]
    out = tracker.observe(snapshot(1500, other, []), catalog)   # otro roster, reloj hacia delante
    assert out["objectives"]["buffs"] == [] and out["my_team"] is None


def test_same_roster_next_game_is_detected_by_the_event_stream(catalog):
    """EventIDs reinician a 0: la partida siguiente con el mismo roster y reloj más avanzado."""
    tracker = OverlayTracker()
    game1 = [kill("DragonKill", 600, "Alice", DragonType="Fire"), kill("DragonKill", 900, "Alice", DragonType="Earth")]
    tracker.observe(snapshot(900, ROSTER, game1), catalog)
    game2 = [kill("DragonKill", 600, "Dan", DragonType="Water"), kill("BaronKill", 1300, "Dan")]
    out = tracker.observe(snapshot(1301, ROSTER, game2), catalog)
    d = out["objectives"]["dragon"]
    assert d["counts"] == {"100": 0, "200": 1, "unknown": 0}, "los dragones de la partida anterior no se arrastran"
    assert out["objectives"]["baron"] == {"state": "spawning", "spawns_at": 1300 + BARON_RESPAWN_S}
    assert champions(out["objectives"]["buffs"][0]["holders"]) == ["Jinx", "Thresh"]

    # GameEnd de la partida anterior no congela la nueva
    tracker = OverlayTracker()
    tracker.observe(snapshot(95, ROSTER, [{"EventName": "GameEnd", "EventTime": 90.0, "Result": "Lose"}]), catalog)
    out = tracker.observe(snapshot(401, ROSTER, [kill("DragonKill", 400, "Alice", DragonType="Fire")]), catalog)
    assert out["game_over"] is False and out["objectives"]["dragon"]["counts"]["100"] == 1

    # misma partida: la lista crece y nada se reinicia
    tracker = OverlayTracker()
    tracker.observe(snapshot(901, ROSTER, game1), catalog)
    out = tracker.observe(snapshot(1301, ROSTER, game1 + [kill("BaronKill", 1300, "Alice")]), catalog)
    assert out["objectives"]["dragon"]["counts"]["100"] == 2 and len(out["objectives"]["buffs"]) == 1


def test_malformed_snapshots_never_raise(catalog):
    out = OverlayTracker().observe({"fake": True}, catalog)
    assert out["in_game"] and out["my_team"] is None and out["gold"]["wallet"] is None
    assert out["gold"]["lead"] == {"team_id": None, "amount": 0, "approx": False} and out["objectives"] is None
    out = OverlayTracker().observe(
        {"allPlayers": [None, {"team": "ORDER", "items": [None, {"itemID": "x"}]}], "events": {"Events": [None]},
         "gameData": {"gameMode": "CLASSIC", "gameTime": "nope", "mapNumber": "?"}}, catalog
    )
    assert out["gold"]["teams"][0]["item_gold"] == 0 and out["objectives"] is not None
    out = OverlayTracker().observe(snapshot(600, ROSTER), None)
    assert out["catalog_version"] is None and out["gold"]["teams"][0]["approx"] is False, "sin objetos no hay nada que aproximar"
