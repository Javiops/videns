"""Las features no pueden contener información que el POV no podía tener.

Estas pruebas fallan con el código anterior: `feature_vector` mandaba la
región real de los cinco enemigos aunque estuvieran en niebla, y `pos_at`
podía coger una muestra POSTERIOR al instante del snapshot.
"""
from app import dataset, trajectory
from app.snapshot import _observed_for


def _row(pid, team, champ, region="JGL_PA", pos=(0.3, 0.3)):
    return {"pid": pid, "champ": champ, "role": "JG", "team": team,
            "pos": list(pos), "pos_age_s": 0.5, "region": region,
            "region_team": region, "level": 9, "cs": 100, "dead": False,
            "respawn_in": 0.0, "gold_total_prev_frame": 5000,
            "gold_current_prev_frame": 100, "xp_prev_frame": 0,
            "items": [], "item_gold": 3000}


def test_observed_hides_region_of_invisible_enemy():
    row = _row(7, 200, "Elise")
    enemy_state = {"Elise": {"champ": "Elise", "known": True, "visible": False,
                             "dead": False, "conf": 20.0, "last_seen_t": None}}
    obs = _observed_for(row, my_team=100, enemy_state=enemy_state, traj={}, t=500.0)
    assert obs["known"] is False
    assert obs["region_team"] is None, "un enemigo en niebla no aporta región"


def test_observed_uses_last_sighting_not_current_position():
    """Visto en JGL_PA en t=100; en t=500 está en otro sitio. El POV solo
    puede saber el avistamiento viejo, nunca la posición actual."""
    row = _row(7, 200, "Elise", region="BASE_ENEMIGA", pos=(0.9, 0.9))
    traj = {"Elise": [(98.0, 0.30, 0.30), (100.0, 0.31, 0.31),
                      (500.0, 0.90, 0.90)]}
    enemy_state = {"Elise": {"champ": "Elise", "known": True, "visible": False,
                             "dead": False, "conf": 10.0, "last_seen_t": 100.0}}
    obs = _observed_for(row, my_team=100, enemy_state=enemy_state,
                        traj=traj, t=500.0)
    assert obs["known"] is True
    assert obs["source"] == "ultimo_avistamiento"
    assert obs["pos"][0] < 0.5 and obs["pos"][1] < 0.5, "usó la posición actual"
    assert obs["age_s"] == 400.0


def test_own_team_is_always_known():
    row = _row(2, 100, "Shaco")
    obs = _observed_for(row, my_team=100, enemy_state={}, traj={}, t=500.0)
    assert obs["known"] is True and obs["region_team"] == "JGL_PA"


def test_feature_vector_does_not_leak_hidden_enemy_region():
    """La feature de región de un enemigo oculto tiene que ser desconocida."""
    players = []
    for pid in range(1, 11):
        team = 100 if pid <= 5 else 200
        r = _row(pid, team, f"C{pid}", region="JGL_ER")
        # capa observable: aliados conocidos, enemigos ocultos y no fiables
        r["observed"] = ({"known": True, "source": "own_team",
                          "region_team": "JGL_ER", "pos": [0.3, 0.3],
                          "age_s": 0.5, "visible": True} if team == 100 else
                         {"known": False, "source": "no_fiable",
                          "region_team": None, "pos": None, "age_s": None,
                          "visible": None})
        players.append(r)
    dec = {"snapshot": {
        "t": 600.0, "frame_t": 600.0, "snapshot_v": "s2",
        "me": {"pid": 2, "champ": "C2", "team": 100, "region_team": "JGL_ER",
               "jcs": 50, "cs_lane": 10},
        "players": players, "camps": {c: True for c in dataset.CAMP_ORDER},
        "objectives": {"dragons": {100: 1, 200: 0}, "grubs": {100: 0, 200: 0},
                       "herald": {100: 0, 200: 0}, "baron": {100: 0, 200: 0},
                       "towers_lost": {100: 0, 200: 1}, "soul_team": None,
                       "next_dragon_in": 120.0},
        "vision": {"available": True, "usable": False, "info_pct": None,
                   "enemies": [], "reason": "prueba"},
        "minimap": {}}}
    f = dataset.feature_vector(dec)
    for i in range(5):
        assert f[f"enemy{i}_known"] == 0
        assert f[f"enemy{i}_region"] == -1, "región de enemigo oculto filtrada"
    # los aliados sí se conocen
    assert all(f[f"ally{i}_known"] == 1 for i in range(5))
    # sin visibilidad utilizable no se publica info_pct
    assert f["vision_usable"] == 0 and f["info_pct"] == -1.0


def test_pos_at_causal_never_returns_future_samples():
    pts = [(10.0, 0.1, 0.1), (20.0, 0.9, 0.9)]
    pos, age = trajectory.pos_at_causal(pts, 11.0)
    assert pos == (0.1, 0.1) and age == 1.0
    # pos_at (no causal) coge la más cercana aunque sea del FUTURO: por eso no
    # vale para features de inferencia
    assert trajectory.pos_at(pts, 19.0) == (0.9, 0.9)
    # la causal, en el mismo instante, jamás devuelve la muestra de t=20
    pos2, age2 = trajectory.pos_at_causal(pts, 19.0, max_age=15.0)
    assert pos2 == (0.1, 0.1) and age2 == 9.0
    # y si la última muestra pasada es demasiado vieja -> desconocido, no
    # se rellena con la siguiente
    assert trajectory.pos_at_causal(pts, 19.0)[0] is None
    assert trajectory.pos_at_causal(pts, 9.0)[0] is None
    assert trajectory.pos_at_causal(pts, 100.0)[0] is None
