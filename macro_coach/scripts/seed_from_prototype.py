"""Deja la partida prototipo de fase 1 (EUW1_7983109023) en el layout
estándar del corpus: nombres canónicos y meta.json con el quirk de geometría
(el minimapa cambió de 508px a 414px en t=1148 durante aquella pasada).

Idempotente. Los datos ya deben estar copiados en data/corpus/EUW1_7983109023
(mv5_*.json, traj_v7.jsonl o traj.jsonl, camp_dots.json o camps.json,
pass_omni/). No hay pasadas con niebla para esta partida: vision no disponible.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import game_dir

MATCH_ID = "EUW1_7983109023"


def main():
    g = game_dir(MATCH_ID)
    if not g.exists():
        raise SystemExit(f"no existe {g}; copia primero los datos del prototipo")
    renames = [("traj_v7.jsonl", "traj.jsonl"), ("camp_dots.json", "camps.json")]
    for old, new in renames:
        if (g / old).exists() and not (g / new).exists():
            (g / old).rename(g / new)
            print(f"renombrado {old} -> {new}")
    pd = g / "pass_omni"
    meta_f = pd / "meta.json"
    if pd.exists() and not meta_f.exists():
        meta = {"mode": "omni",
                "fog_validation": {"mode": "omni", "ok": True,
                                   "why": "pasada de fase 1, sin niebla por defecto"},
                "geom_switches": [[0, [9, 5, 508]], [1148, [104, 101, 414]]],
                "geom": [9, 5, 508],
                "note": "captura de fase 1; la geometria cambio en t=1148"}
        meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print("escrito pass_omni/meta.json (con geom_switches)")
    gm = g / "meta.json"
    if not gm.exists():
        gm.write_text(json.dumps({"seeded_from": "fase 1", "patch": "16.18"},
                                 indent=1), encoding="utf-8")
    print(f"OK: {MATCH_ID} en layout estandar")


if __name__ == "__main__":
    main()
