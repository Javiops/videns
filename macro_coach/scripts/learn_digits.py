"""Aprende las plantillas de digitos del marcador y las valida.

Arranque: `data/ocr/sample_gold.json` trae diez celdas etiquetadas a mano
(unica intervencion humana de todo el OCR). De ahi salen las plantillas.

Validacion: NO se cree lo que lee. Compara contra Match-V5, que da el oro de
los diez cada 60s. La captura del HUD y el frame de Match-V5 no son del mismo
instante exacto, asi que se admite un margen: la renta pasiva son ~2 oro/s.

Uso: python scripts/learn_digits.py [--check-corpus]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app import ocr
from app.config import CORPUS, DATA, is_excluded

MUESTRA = DATA / "ocr" / "sample_gold.json"
# Margen de oro admitido entre la lectura del HUD y el frame de Match-V5 del
# mismo segundo. La renta pasiva es ~2 oro/s; un esbirro son 20-60. Con 25 se
# detecta un digito mal leido (cambia el numero en cientos o miles) sin marcar
# como error el desfase natural de unas decimas de segundo.
TOLERANCIA = 25


def aprende_de_muestra() -> dict:
    spec = json.loads(MUESTRA.read_text(encoding="utf-8"))
    img = cv2.imread(str(MUESTRA.parent / spec["captura"]))
    if img is None:
        raise SystemExit(f"no encuentro {spec['captura']}")
    muestras = []
    for c in spec["celdas"]:
        x0, y0, x1, y1 = c["box"]
        muestras.append((img[y0:y1, x0:x1], c["texto"]))
    plantillas = ocr.aprende(muestras)
    usadas, total = plantillas["_muestras"]
    print(f"plantillas de {usadas}/{total} celdas etiquetadas")
    for c, n in zip(ocr.ALFABETO, plantillas["_n"]):
        marca = "" if n else "   <-- SIN EJEMPLOS"
        print(f"  '{c}': {n:2d} ejemplos{marca}")
    faltan = [c for c, n in zip(ocr.ALFABETO, plantillas["_n"]) if not n]
    if faltan:
        raise SystemExit(f"faltan ejemplos de {faltan}: no se puede leer todo")
    return plantillas, spec, img


def comprueba_muestra(plantillas, spec, img) -> bool:
    """Lee la propia muestra: es el minimo exigible, no una validacion."""
    r = ocr.lee_marcador(img, plantillas)
    esperado = {c["pid"]: ocr.parsea_oro(c["texto"]) for c in spec["celdas"]}
    mal = 0
    for pid in range(1, 11):
        leido, esp = r["gold"][pid - 1], esperado[pid]
        if leido != esp:
            mal += 1
            print(f"  pid {pid}: leido {leido} != etiquetado {esp}")
    print(f"relectura de la muestra: {10 - mal}/10 celdas exactas "
          f"(score minimo {r['score']})")
    return mal == 0


def valida_contra_mv5(plantillas) -> None:
    """Contraste real: recortes del corpus contra los anclajes de Match-V5."""
    casos, errores, ilegibles, partidas = 0, [], 0, 0
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir() or is_excluded(d.name):
            continue
        sb = sorted((d / "pass_omni").glob("sb_*.png"))
        tl_f = d / "mv5_timeline.json"
        if not sb or not tl_f.exists():
            continue
        partidas += 1
        tl = json.loads(tl_f.read_text(encoding="utf-8"))
        frames = {round(f["timestamp"] / 1000): f for f in tl["info"]["frames"]}
        for p in sb:
            t = int(p.stem.split("_")[1]) / 10.0
            # Solo los instantes con ancla: el frame mas cercano a menos de 1s
            cerca = [ts for ts in frames if abs(ts - t) <= 1.0]
            if not cerca:
                continue
            fr = frames[cerca[0]]
            r = ocr.lee_marcador(cv2.imread(str(p)), plantillas)
            for pid in range(1, 11):
                pf = fr["participantFrames"].get(str(pid))
                leido = r["gold"][pid - 1]
                if pf is None:
                    continue
                casos += 1
                if leido is None:
                    ilegibles += 1
                    continue
                errores.append(leido[1] - pf["totalGold"])
    if not casos:
        print("\nsin recortes de marcador con anclaje de Match-V5 en el corpus.\n"
              "Es lo esperado: las 40 partidas se capturaron con el marcador en\n"
              "vista de ITEMS. La validacion real llega con la proxima tanda.")
        return
    print(f"\nvalidacion contra Match-V5: {partidas} partidas, {casos} celdas")
    print(f"  ilegibles (None): {ilegibles} ({100*ilegibles/casos:.1f}%)")
    if not errores:
        print("  ninguna celda legible: no hay error que medir. Con el marcador")
        print("  en vista de ITEMS es lo correcto. Aqui solo se miran los")
        print("  instantes con ancla; el control negativo completo (todos los")
        print("  recortes del corpus) lo pasa scripts/read_gold.py.")
        return
    e = np.abs(errores)
    print(f"  |error| mediana {np.median(e):.0f}, p90 {np.percentile(e, 90):.0f}")
    print(f"  dentro de {TOLERANCIA} oro: {100*np.mean(e <= TOLERANCIA):.2f}%")


def main():
    plantillas, spec, img = aprende_de_muestra()
    ok = comprueba_muestra(plantillas, spec, img)
    if not ok:
        raise SystemExit("la relectura de la muestra falla: no se guardan "
                         "plantillas que no saben leer ni sus propios datos")
    ocr.guarda(plantillas)
    print(f"guardadas en {ocr.TEMPLATES_FILE}")
    if "--check-corpus" in sys.argv:
        valida_contra_mv5(plantillas)


if __name__ == "__main__":
    main()
