"""Desconocido explícito en la visibilidad.

Falla con el código anterior: `TeamVisibility` no conocía la sensibilidad, así
que un campeón que el detector casi nunca ve salía como "no visible" (conf
baja) en vez de DESCONOCIDO, y una pasada sospechosa seguía dando números.
"""
from app import visibility
from app.config import VIS_MIN_SENSITIVITY

DEATHS = [(0.0, {"Elise": False, "Riven": False, "Yone": False,
                 "Ziggs": False, "Camille": False})]
IVS = {c: [(100.0, 110.0)] for c in ("Elise", "Riven", "Yone", "Ziggs", "Camille")}
GOOD = {c: 0.85 for c in IVS}


def test_low_sensitivity_champion_is_unknown_not_invisible():
    sens = dict(GOOD, Elise=0.15)          # el detector casi nunca la ve
    tv = visibility.TeamVisibility(IVS, DEATHS, sensitivity=sens)
    e = tv.enemy("Elise", 300.0)
    assert e["known"] is False
    assert e["conf"] is None and e["visible"] is None, \
        "una no-detección con sensibilidad 0.15 no prueba que esté oculta"
    # uno con buena sensibilidad sí da número
    assert tv.enemy("Riven", 300.0)["known"] is True
    assert tv.enemy("Riven", 300.0)["conf"] is not None


def test_info_pct_averages_only_known_enemies():
    sens = dict(GOOD, Elise=0.10, Riven=0.10)
    tv = visibility.TeamVisibility(IVS, DEATHS, sensitivity=sens)
    info = tv.info(300.0)
    assert info["n_known"] == 3 and info["n_enemies"] == 5
    assert info["info_pct"] is not None


def test_info_pct_is_none_when_too_few_known():
    sens = {c: 0.10 for c in IVS}
    sens["Riven"] = 0.9
    tv = visibility.TeamVisibility(IVS, DEATHS, sensitivity=sens)
    info = tv.info(300.0)
    assert info["info_pct"] is None, "con 1 enemigo conocido no se publica media"
    assert info["usable"] is False


def test_suspect_pass_makes_everything_unknown():
    tv = visibility.TeamVisibility(IVS, DEATHS, sensitivity=GOOD, usable=False,
                                   reason="suspect_no_fog")
    info = tv.info(300.0)
    assert info["info_pct"] is None and info["usable"] is False
    assert all(e["known"] is False for e in info["enemies"])


def test_dead_enemy_is_known_even_if_sensitivity_low():
    deaths = [(0.0, {"Elise": False}), (200.0, {"Elise": True})]
    tv = visibility.TeamVisibility({"Elise": [(100.0, 110.0)]}, deaths,
                                   sensitivity={"Elise": 0.05})
    e = tv.enemy("Elise", 250.0)
    assert e["dead"] is True and e["known"] is True and e["conf"] == 100.0


def test_missing_sensitivity_is_unknown():
    tv = visibility.TeamVisibility(IVS, DEATHS, sensitivity={})
    assert tv.enemy("Riven", 300.0)["known"] is False


def test_mode_usable_flags():
    assert visibility.mode_usable({"blue": {"suspect_no_fog": True}}, "blue")[0] is False
    assert visibility.mode_usable({"blue": {"per_champion": {}}}, "blue")[0] is True
    assert visibility.mode_usable({}, "blue")[0] is False


def test_threshold_is_the_documented_one():
    sens = {c: VIS_MIN_SENSITIVITY for c in IVS}
    tv = visibility.TeamVisibility(IVS, DEATHS, sensitivity=sens)
    assert tv.champ_usable("Riven") is True
    sens2 = {c: VIS_MIN_SENSITIVITY - 0.01 for c in IVS}
    tv2 = visibility.TeamVisibility(IVS, DEATHS, sensitivity=sens2)
    assert tv2.champ_usable("Riven") is False
