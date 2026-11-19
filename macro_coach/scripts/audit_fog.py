"""Auditoría rápida: ¿tiene cada pasada de niebla realmente niebla de equipo?

Una pasada con niebla oculta zonas del mapa, así que sus frames son
sensiblemente MÁS OSCUROS que los de la pasada omni del mismo instante. Si el
brillo de `pass_blue` / `pass_red` es prácticamente el de `pass_omni`, esa
pasada se capturó con la niebla en "Todos" y no sirve para visibilidad — pasó
de verdad, por un clic que caía en la fila equivocada del desplegable.

Es mucho más barato que procesar la partida entera (segundos, no minutos), así
que sirve para decidir qué pasadas hay que recapturar.

Uso: python scripts/audit_fog.py [--json]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.config import CORPUS, is_excluded

SAMPLES = 24
MIN_RATIO = 0.93          # brillo_niebla / brillo_omni por debajo = hay niebla
MIN_BR_DIFF = 6.0         # diferencia media blue-vs-red para considerarlas distintas


def mean_brightness(pass_dir: Path, n: int = SAMPLES) -> float | None:
    files = sorted(glob.glob(str(pass_dir / "mm_*.png")))
    if not files:
        return None
    step = max(1, len(files) // n)
    vals = []
    for fp in files[::step][:n]:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            vals.append(float(img.mean()))
    return float(np.mean(vals)) if vals else None


def _frames_by_time(pass_dir: Path) -> dict:
    out = {}
    for fp in glob.glob(str(pass_dir / "mm_*.png")):
        t = int(os.path.basename(fp)[3:9]) / 10.0
        out[t] = fp
    return out


def blue_vs_red(g: Path, n: int = 16) -> float | None:
    """Diferencia media entre las dos pasadas de niebla en instantes casados.

    Con nieblas OPUESTAS las dos imagenes ocultan mitades distintas y difieren
    mucho. Si salen casi identicas, las dos pasadas llevan la MISMA niebla: el
    clic cayo en la misma fila las dos veces. Es el fallo que la comparacion
    contra omni no ve (las dos son oscuras, las dos 'parecen' tener niebla).
    """
    b = _frames_by_time(g / "pass_blue")
    r = _frames_by_time(g / "pass_red")
    if not b or not r:
        return None
    times = sorted(b)[::max(1, len(b) // n)][:n]
    diffs = []
    for t in times:
        tr = min(r, key=lambda x: abs(x - t))
        if abs(tr - t) > 3.0:
            continue
        ib = cv2.imread(b[t], cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(r[tr], cv2.IMREAD_GRAYSCALE)
        if ib is None or ir is None:
            continue
        if ib.shape != ir.shape:
            # 507 vs 508 px entre pasadas: recortar al minimo comun en vez de
            # descartar la comparacion (asi se quedaban partidas sin auditar)
            n = min(ib.shape[0], ir.shape[0], ib.shape[1], ir.shape[1])
            ib, ir = ib[:n, :n], ir[:n, :n]
        diffs.append(float(np.mean(np.abs(ib.astype(int) - ir.astype(int)))))
    return float(np.mean(diffs)) if diffs else None


def audit_game(g: Path) -> dict:
    out = {"match_id": g.name}
    base = mean_brightness(g / "pass_omni")
    out["omni"] = round(base, 1) if base else None
    for mode in ("blue", "red"):
        b = mean_brightness(g / f"pass_{mode}")
        if b is None or not base:
            out[mode] = None
            out[mode + "_ok"] = None
            continue
        ratio = b / base
        out[mode] = round(b, 1)
        out[mode + "_ratio"] = round(ratio, 3)
        out[mode + "_ok"] = bool(ratio < MIN_RATIO)
    d = blue_vs_red(g)
    out["blue_vs_red"] = round(d, 2) if d is not None else None
    # Con nieblas opuestas la diferencia es grande; por debajo de esto son la
    # misma niebla capturada dos veces.
    out["distinct_fogs"] = bool(d is not None and d >= MIN_BR_DIFF)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = []
    for g in sorted(CORPUS.iterdir()):
        if not g.is_dir() or not (g / "pass_omni").exists():
            continue
        if is_excluded(g.name):
            continue
        rows.append(audit_game(g))
    bad = []
    for r in rows:
        flags = []
        for mode in ("blue", "red"):
            ok = r.get(mode + "_ok")
            ratio = r.get(mode + "_ratio")
            if ok is None:
                flags.append(f"{mode}:falta")
            elif ok:
                flags.append(f"{mode}:niebla({ratio})")
            else:
                flags.append(f"{mode}:SIN NIEBLA({ratio})")
                bad.append((r["match_id"], mode))
        if not r.get("distinct_fogs") and r.get("blue_vs_red") is not None:
            flags.append(f"MISMA NIEBLA EN AMBAS(dif {r['blue_vs_red']})")
            bad.append((r["match_id"], "blue+red"))
        elif r.get("blue_vs_red") is not None:
            flags.append(f"distintas(dif {r['blue_vs_red']})")
        if not args.json:
            print(f"{r['match_id']}  omni {r['omni']}  " + "  ".join(flags))
    if args.json:
        print(json.dumps({"rows": rows, "bad": bad}, indent=1))
        return
    print(f"\n{len(rows)} partidas auditadas; {len(bad)} pasadas SIN niebla real")
    if bad:
        print("hay que recapturar:")
        for mid, mode in bad:
            print(f"  {mid}  pass_{mode}")


if __name__ == "__main__":
    main()
