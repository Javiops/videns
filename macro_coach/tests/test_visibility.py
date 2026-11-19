from app import visibility
from app.config import CONF_HORIZON_S


def test_intervals_from_times():
    iv = visibility.intervals_from_times([10, 12, 13, 40, 41])
    assert iv == [(10, 13), (40, 41)]


def test_confidence_decay():
    iv = [(0.0, 0.0), (100.0, 110.0)]
    assert visibility.confidence(105.0, iv, dead=False) == 100.0     # visible
    assert visibility.confidence(110.0 + CONF_HORIZON_S / 2, iv, dead=False) == 50.0
    assert visibility.confidence(110.0 + CONF_HORIZON_S + 1, iv, dead=False) == 0.0
    assert visibility.confidence(300.0, iv, dead=True) == 100.0      # muerto = sabido


def test_spawn_seed_and_team_info():
    # hace falta declarar la sensibilidad: sin ella la visibilidad es
    # DESCONOCIDA a propósito (ver tests/test_visibility_quality.py)
    deaths = [(0.0, {"Vi": False}), (200.0, {"Vi": True}), (215.0, {"Vi": False})]
    tv = visibility.TeamVisibility({"Vi": [(100.0, 110.0)]}, deaths,
                                   sensitivity={"Vi": 0.9})
    # t=20: sembrado en t=0, decae
    e = tv.enemy("Vi", 20.0)
    assert e["conf"] == round(100 * (1 - 20 / CONF_HORIZON_S), 1)
    # t=105: visible
    assert tv.enemy("Vi", 105.0)["conf"] == 100.0
    # t=205: muerta
    assert tv.enemy("Vi", 205.0)["dead"] is True
    assert tv.enemy("Vi", 205.0)["conf"] == 100.0
    # con un solo enemigo conocido NO se publica media (VIS_MIN_USABLE_ENEMIES)
    assert tv.info(20.0)["info_pct"] is None
    # con tres o más, la media es la de los conocidos
    ivs = {c: [(100.0, 110.0)] for c in ("Vi", "Lux", "Jinx")}
    deaths3 = [(0.0, {c: False for c in ivs})]
    tv3 = visibility.TeamVisibility(ivs, deaths3,
                                    sensitivity={c: 0.9 for c in ivs})
    info3 = tv3.info(20.0)
    assert info3["n_known"] == 3
    assert info3["info_pct"] == tv3.enemy("Vi", 20.0)["conf"]
