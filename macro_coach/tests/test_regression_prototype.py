"""Regresión contra la partida auditada de fase 1 (EUW1_7983109023, Evelynn
jungla ORDER). La salida auditada tenía 75 segmentos; anclas conocidas:
farmeo rojo 47-113, farmeo azul 122-171, GANK MID ~180 sobre Lux.
Se salta si el corpus no contiene la partida."""
import json

import pytest

from app import camps as camps_mod
from app import lexicon, segments, trajectory
from app.config import game_dir

PROTO = "EUW1_7983109023"
g = game_dir(PROTO)

pytestmark = pytest.mark.skipif(not (g / "traj.jsonl").exists(),
                                reason="partida prototipo no presente")


@pytest.fixture(scope="module")
def pipeline():
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
    P = segments.participants(match)
    enemies = {v["champ"] for v in P.values() if v["team"] == 200}
    dets = trajectory.load_detections(g / "traj.jsonl")
    anchors = trajectory.mv5_anchors(timeline, P)
    traj = trajectory.fuse(dets, anchors, enemies)
    dots = json.loads((g / "camps.json").read_text(encoding="utf-8"))
    ckills = camps_mod.camp_kills(dots, traj)
    me = next(pid for pid, v in P.items() if v["champ"] == "Evelynn")
    segs = segments.build_segments(match, timeline, traj, ckills, me)
    return match, segs


def test_segment_count_close_to_audit(pipeline):
    # La densidad de segmentos depende de cuántas detecciones sobrevivan al
    # filtro de score. Con TRAJ_MIN_SCORE=0.82 (fase 2) salen ~75, como las 75
    # filas auditadas; con el 0.895 de fase 1 caían a ~53 porque se tiraban
    # detecciones buenas (ver docs/history.md). Rango amplio a propósito: el
    # contrato real son las anclas de los tests siguientes, no el número.
    _, segs = pipeline
    assert 60 <= len(segs) <= 90, f"{len(segs)} segmentos vs ~75 auditados"


def test_early_farm_sequence(pipeline):
    _, segs = pipeline
    early = [s for s in segs if s["t1"] > 40 and s["t0"] < 200]
    jungles = [s for s in early if s["kind"] == "jungle"]
    assert jungles, "sin segmentos de jungla tempranos"
    first = jungles[0]
    assert first["region_team"] == "JGL_PR" and first["farming"], \
        f"primer farmeo esperado en cuadrante rojo: {first}"


def test_first_gank_mid_on_lux(pipeline):
    # auditado: "180-187 GANK MID sobre Lux (MID)". El "sobre Lux" venía de los
    # rivales detectados cerca (enemies_near), no de un kill: aquí es `rivals`.
    _, segs = pipeline
    ganks = [s for s in segs if s["kind"] == "lane_gank"]
    assert ganks, "sin ganks detectados"
    first = ganks[0]
    assert 170 <= first["t0"] <= 195 and first["lane"] == "MID", str(first)
    involved = set(first["rivals"]) | {m["other"] for m in first["marks"]}
    assert "Lux" in involved, f"gank sin Lux: {first}"


def test_decisions_stream_sane(pipeline):
    _, segs = pipeline
    decs = lexicon.decisions_from_segments(segs)
    assert 20 <= len(decs) <= 60, f"{len(decs)} decisiones"
    toks = {d["token"] for d in decs}
    assert "FARMEAR_PROPIA_ROJA" in toks
    assert "GANK_MID" in toks
    assert "RESET" in toks
    # el prototipo registró exactamente una invasión (cuadrante azul)
    assert "INVADIR_AZUL" in toks
    for d in decs:
        assert d["token"] in lexicon.ALL_TOKENS
