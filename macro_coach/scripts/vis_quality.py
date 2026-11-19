"""Calidad de la visibilidad en el corpus: distribución de la sensibilidad
por campeón y cuánto se pierde con el umbral `VIS_MIN_SENSITIVITY`.

La sensibilidad se mide en la pasada del PROPIO equipo del campeón, donde por
definición siempre está dibujado: es la tasa de aciertos del detector para ESE
campeón. Por debajo del umbral, su no-detección no prueba que esté oculto, así
que su visibilidad se marca DESCONOCIDA en vez de "no visible".

Sirve para calibrar el umbral con datos en vez de a ojo.

Uso: python scripts/vis_quality.py [--threshold 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CORPUS, VIS_MIN_SENSITIVITY, is_excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=VIS_MIN_SENSITIVITY)
    args = ap.parse_args()

    vals, per_game, suspect = [], [], []
    for g in sorted(CORPUS.iterdir()):
        if not g.is_dir() or is_excluded(g.name):
            continue
        f = g / "visibility.json"
        if not f.exists():
            continue
        meta = json.loads(f.read_text(encoding="utf-8")).get("meta", {})
        game_vals = []
        for mode in ("blue", "red"):
            m = meta.get(mode, {})
            if m.get("suspect_no_fog"):
                suspect.append((g.name, mode))
            for champ, r in (m.get("per_champion") or {}).items():
                s = r.get("sensibilidad")
                if s is not None:
                    vals.append(s)
                    game_vals.append(s)
        if game_vals:
            usable = sum(1 for s in game_vals if s >= args.threshold)
            per_game.append((g.name, len(game_vals), usable,
                             sorted(game_vals)[len(game_vals) // 2]))

    if not vals:
        print("todavía no hay visibility.json en el corpus")
        return
    vals.sort()

    def q(x):
        return vals[min(int(len(vals) * x), len(vals) - 1)]

    print(f"sensibilidad por campeón-partida (n={len(vals)}):")
    print(f"  p05 {q(0.05):.2f}  p25 {q(0.25):.2f}  mediana {q(0.5):.2f}  "
          f"p75 {q(0.75):.2f}  p95 {q(0.95):.2f}")
    usable = sum(1 for s in vals if s >= args.threshold)
    print(f"  con umbral {args.threshold}: {usable}/{len(vals)} "
          f"({100*usable/len(vals):.0f}%) utilizables; el resto queda DESCONOCIDO")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        u = sum(1 for s in vals if s >= thr)
        print(f"    umbral {thr:.1f} -> {100*u/len(vals):5.1f}% utilizable")
    peor = sorted(per_game, key=lambda r: r[2] / max(r[1], 1))[:5]
    print("\npartidas con menos campeones utilizables:")
    for mid, n, u, med in peor:
        print(f"  {mid}  {u}/{n} utilizables, mediana {med:.2f}")
    if suspect:
        print(f"\npasadas marcadas suspect_no_fog: {suspect}")


if __name__ == "__main__":
    main()
