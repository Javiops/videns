"""Lee el oro de los diez desde los recortes del marcador -> `gold.jsonl`.

Fuente: los `sb_*.png` de la pasada TRUTH-SEEKER, tomados en los mismos
instantes que la `playerlist`. Sin esto el oro de las features sale del frame
de Match-V5 y llega hasta 60s rancio.

Se valida dos veces, y las dos solas:

1. **Anclajes de Match-V5** (cada 60s, los diez): oro verdadero contra oro
   leido. Es exacto pero disperso.
2. **Monotonia del oro total**: el acumulado no baja nunca. Un digito mal leido
   lo rompe casi siempre, y esto se puede comprobar en TODAS las muestras, no
   solo en las de ancla. Es el control denso.

Uso: python scripts/read_gold.py [MATCH_ID ...] [--force]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app import ocr
from app.config import CORPUS, is_excluded

TOLERANCIA = 25       # oro; ver scripts/learn_digits.py


def lee_partida(d: Path, plantillas: dict) -> list:
    filas = []
    for p in sorted((d / "pass_omni").glob("sb_*.png")):
        t = int(p.stem.split("_")[1]) / 10.0
        r = ocr.lee_marcador(cv2.imread(str(p)), plantillas)
        filas.append({"t": round(t, 2), "gold": r["gold"],
                      "leidos": r["leidos"], "score": r["score"]})
    return filas


def valida(d: Path, filas: list) -> dict:
    """Envuelve ocr.valida_contra_timeline aportando el contexto de la partida."""
    tl = json.loads((d / "mv5_timeline.json").read_text(encoding="utf-8"))
    inf = ocr.valida_contra_timeline(filas, tl, TOLERANCIA)
    # La velocidad de la pasada va en el informe: "legible" solo significa algo
    # junto a la velocidad a la que se capturo (32x es la de produccion).
    meta = json.loads((d / "pass_omni" / "meta.json").read_text(encoding="utf-8"))
    return {"speed": meta.get("speed"), **inf}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    plantillas = ocr.carga()
    mids = args or [d.name for d in sorted(CORPUS.iterdir())
                    if d.is_dir() and not is_excluded(d.name)]
    hechas, sin_recortes, sin_oro = 0, [], []
    for mid in mids:
        d = CORPUS / mid
        if not (d / "pass_omni").is_dir():
            continue
        if not list((d / "pass_omni").glob("sb_*.png")):
            sin_recortes.append(mid)
            continue
        out = d / "gold.jsonl"
        if out.exists() and not force:
            print(f"{mid}: ya existe gold.jsonl (--force para rehacer)")
            continue
        filas = lee_partida(d, plantillas)
        inf = valida(d, filas)
        if not inf["celdas_leidas"]:
            # Ni una celda legible: el marcador estaba en vista de items. No se
            # deja un gold.jsonl vacio, que luego parece "ya procesado" y hace
            # que una recaptura con oro se salte esta partida en silencio.
            sin_oro.append(mid)
            continue
        with open(out, "w", encoding="utf-8") as fh:
            for f in filas:
                fh.write(json.dumps(f) + "\n")
        (d / "gold_report.json").write_text(json.dumps(inf, indent=1),
                                            encoding="utf-8")
        hechas += 1
        print(f"{mid}: {inf['celdas_leidas']}/{inf['celdas_totales']} celdas "
              f"({100*inf['frac_legible']:.1f}%), "
              f"anclas {inf['anclas_mv5']}, "
              f"|error| mediana {inf.get('error_mediana', '-')}, "
              f"bajadas de total {inf['bajadas_de_total']}")

    if sin_oro:
        print(f"\n{len(sin_oro)} partidas con recortes pero SIN una sola celda "
              "legible: el marcador estaba en vista de items.")
        print("Es lo esperado en las 40 primeras. El oro empieza en la proxima")
        print("tanda; no se deja gold.jsonl vacio para que la recaptura no se")
        print("las salte.")
    if sin_recortes:
        print(f"{len(sin_recortes)} partidas sin recortes de marcador.")
    if not hechas and not sin_recortes and not sin_oro:
        print("nada que leer")


if __name__ == "__main__":
    main()
