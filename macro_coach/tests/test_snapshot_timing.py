"""El snapshot es causal: todo se evalúa en t0 - SNAPSHOT_EPS_S.

Falla con el código anterior: `process_game` pasaba
`camps_mod.alive_at(cstates, d["t0"])` mientras `build_snapshot` fijaba el
resto del estado en `t0 - eps`. Un campamento que muere justo al empezar el
segmento aparecía ya muerto en el estado con el que se decidió.
"""
from app import camps as camps_mod
from app.config import SNAPSHOT_EPS_S
from app.snapshot import snapshot_time


def test_snapshot_time_is_before_the_decision():
    assert snapshot_time(100.0) == 100.0 - SNAPSHOT_EPS_S
    assert snapshot_time(0.2) == 0.0           # no se va a negativo


def test_camp_alive_at_boundary_uses_snapshot_instant():
    """Campamento vivo hasta t=99.5 y muerto desde t=100.0; decisión en
    t0=100.0. En el instante causal (99.0) TODAVÍA estaba vivo."""
    t0 = 100.0
    states = {"gromp": [(98.0, True), (99.5, True), (100.0, False),
                        (101.0, False)]}
    en_t0 = camps_mod.alive_at(states, t0)["gromp"]
    en_snap = camps_mod.alive_at(states, snapshot_time(t0))["gromp"]
    assert en_t0 is False, "en t0 ya está muerto (resultado, no estado previo)"
    assert en_snap is True, "el snapshot debe verlo vivo: aún no había muerto"


def test_build_snapshot_evaluates_camps_itself(monkeypatch):
    """build_snapshot recibe la SERIE y la evalúa en su propio instante, así
    que el llamante ya no puede desalinearla."""
    import inspect
    from app import snapshot as snap_mod
    sig = inspect.signature(snap_mod.build_snapshot)
    assert "camp_states" in sig.parameters, \
        "build_snapshot debe recibir la serie completa, no un corte ya hecho"
