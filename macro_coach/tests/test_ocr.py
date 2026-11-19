"""OCR del oro: que lea lo suyo, que generalice y que NO invente.

El control positivo (lee la muestra que le enseñamos) no prueba casi nada: es
su propio material de entrenamiento. Los dos que importan son el leave-one-out
(clasificar un glifo que no estaba en las plantillas) y el control negativo:
el marcador en vista de ITEMS no tiene oro y ninguna lectura debe salir de ahí.
Con 2b del contrato en mente — sin control negativo, una medición de visión no
es admisible.
"""
import json
import random

import cv2
import numpy as np
import pytest

from app import ocr
from app.config import CORPUS, DATA, is_excluded

MUESTRA = DATA / "ocr" / "sample_gold.json"


def _muestra():
    spec = json.loads(MUESTRA.read_text(encoding="utf-8"))
    img = cv2.imread(str(MUESTRA.parent / spec["captura"]))
    if img is None:
        pytest.skip("falta data/ocr/sample_gold.png")
    return spec, img


def _glifos_etiquetados(spec, img):
    out = []
    for c in spec["celdas"]:
        x0, y0, x1, y1 = c["box"]
        m = ocr._mascara(img[y0:y1, x0:x1])
        gs = ocr.glifos(m)
        assert len(gs) == len(c["texto"]), \
            f"segmentacion de {c['campeon']}: {len(gs)} glifos para {c['texto']}"
        for (a, b), ch in zip(gs, c["texto"]):
            out.append((m[:, a:b], ch))       # parche crudo: normaliza el lector
    return out


def _plantillas(spec, img):
    return ocr.aprende([(img[c["box"][1]:c["box"][3], c["box"][0]:c["box"][2]],
                         c["texto"]) for c in spec["celdas"]])


def test_lee_las_diez_celdas_de_la_muestra():
    spec, img = _muestra()
    r = ocr.lee_marcador(img, _plantillas(spec, img))
    esperado = [ocr.parsea_oro(c["texto"]) for c in spec["celdas"]]
    assert r["gold"] == esperado, r["why"]


def test_el_indice_es_el_participantid():
    """Fila de arriba a abajo = pid 1..5 (azul) y 6..10 (rojo).

    Verificado contra Match-V5 en el mismo instante: el orden del marcador es
    el de participantes, no por rol ni por oro. Si el HUD cambiara ese orden,
    el oro se asignaría al jugador equivocado sin fallar nada más.
    """
    spec, img = _muestra()
    r = ocr.lee_marcador(img, _plantillas(spec, img))
    for c in spec["celdas"]:
        assert r["gold"][c["pid"] - 1] == ocr.parsea_oro(c["texto"]), c["campeon"]


def test_leave_one_out_clasifica_glifos_no_vistos():
    """Generalizacion real: se aprende sin el glifo y luego se le pide leerlo."""
    spec, img = _muestra()
    glifos = _glifos_etiquetados(spec, img)
    mal = []
    for i, (g, ch) in enumerate(glifos):
        resto = {}
        for j, (g2, ch2) in enumerate(glifos):
            if i != j:
                resto.setdefault(ch2, []).append(g2)
        if ch not in resto:
            continue
        pl = {c: np.median(np.stack([ocr.normaliza(x) for x in v]), axis=0)
              for c, v in resto.items()}
        pred, _ = ocr._lee_glifo(g, pl)
        if pred != ch:
            mal.append((ch, pred))
    assert not mal, f"glifos mal clasificados sin verse antes: {mal}"


def test_vista_de_items_no_produce_oro():
    """CONTROL NEGATIVO. Un marcador en vista de ITEMS no tiene oro que leer:
    el OCR debe devolver None, nunca un número plausible (regla 7).

    Las partidas se eligen por lo que dice su meta, no por suponer: en cuanto
    una se recaptura con oro deja de servir como control negativo, y dar por
    hecho que "las del corpus son de items" haría que el control se volviera
    falso solo, sin avisar.
    """
    crops = []
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir() or is_excluded(d.name):
            continue
        meta = d / "pass_omni" / "meta.json"
        if meta.exists():
            gv = json.loads(meta.read_text(encoding="utf-8")) \
                .get("scoreboard", {}).get("gold_view") or {}
            if gv.get("gold_view"):
                continue          # esta SÍ tiene oro: no es control negativo
        crops += sorted((d / "pass_omni").glob("sb_*.png"))
    if len(crops) < 20:
        pytest.skip("no hay recortes de marcador en el corpus")
    spec, img = _muestra()
    pl = _plantillas(spec, img)
    random.seed(0)
    inventadas = 0
    for p in random.sample(crops, 60):
        inventadas += ocr.lee_marcador(cv2.imread(str(p)), pl)["leidos"]
    assert inventadas == 0, f"{inventadas} celdas de oro inventadas sobre items"


def _timeline(anclas):
    """Timeline sintético: {segundo: oro_total} igual para los diez."""
    return {"info": {"frames": [
        {"timestamp": s * 1000,
         "participantFrames": {str(p): {"totalGold": g, "currentGold": 0}
                               for p in range(1, 11)}}
        for s, g in sorted(anclas.items())]}}


def _filas(t, total):
    return [{"t": t, "gold": [(0, total)] * 10, "leidos": 10}]


def test_horquilla_admite_el_desfase_del_hud():
    """El HUD va por detrás del reloj (<1s, medido): un recorte tomado justo
    después de un anclaje enseña el estado de un instante anterior y lee algo
    POR DEBAJO del oro de ese anclaje. Es correcto, no un dígito mal leído."""
    tl = _timeline({60: 1000, 120: 1500, 180: 2000})
    inf = ocr.valida_contra_timeline(_filas(120.2, 1499), tl)
    assert inf["horquilla_fuera"] == 0, \
        "una lectura 1 de oro por debajo del anclaje recién pasado es el lag"


def test_horquilla_pilla_una_lectura_imposible():
    """Lo que sí tiene que saltar: un valor que ningún desfase explica."""
    tl = _timeline({60: 1000, 120: 1500, 180: 2000})
    assert ocr.valida_contra_timeline(_filas(150.0, 9999), tl)["horquilla_fuera"] == 10
    assert ocr.valida_contra_timeline(_filas(150.0, 100), tl)["horquilla_fuera"] == 10
    # y dentro de la horquilla no salta
    assert ocr.valida_contra_timeline(_filas(150.0, 1700), tl)["horquilla_fuera"] == 0


def test_parseo_rechaza_lo_imposible():
    assert ocr.parsea_oro("92(6457)") == (92, 6457)
    assert ocr.parsea_oro("6457)") is None        # sin apertura
    assert ocr.parsea_oro("92(645") is None       # sin cierre
    assert ocr.parsea_oro("5/4/2") is None        # KDA
    assert ocr.parsea_oro("105") is None          # CS
    assert ocr.parsea_oro("9999(12)") is None     # actual > total
    assert ocr.parsea_oro("") is None
