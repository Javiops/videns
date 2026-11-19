"""Oro exacto en cada compra -> `buys.jsonl` por partida.

Cruza los instantes de compra de Match-V5 con las lecturas OCR del
TRUTH-SEEKER (ver `app/buys.py`). Requiere `gold.jsonl` (scripts/read_gold.py).

Uso: python scripts/resolve_buys.py [MATCH_ID ...] [--force]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import buys
from app.config import CORPUS, is_excluded


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    mids = args or [d.name for d in sorted(CORPUS.iterdir())
                    if d.is_dir() and not is_excluded(d.name)]
    hechas, sin_oro = 0, 0
    for mid in mids:
        d = CORPUS / mid
        if not (d / "gold.jsonl").exists():
            sin_oro += 1
            continue
        out = d / "buys.jsonl"
        if out.exists() and not force:
            print(f"{mid}: ya existe buys.jsonl (--force para rehacer)")
            continue
        filas, resumen = buys.resolver_partida(mid)
        with open(out, "w", encoding="utf-8") as fh:
            for f in filas:
                fh.write(json.dumps(f) + "\n")
        (d / "buys_report.json").write_text(json.dumps(resumen, indent=1),
                                            encoding="utf-8")
        hechas += 1
        print(f"{mid}: {resumen['compras']} eventos de tienda — "
              f"{resumen['exactas']} exactos, {resumen['acotadas']} acotados"
              + (f" (intervalo mediano {resumen['intervalo_mediano']} oro)"
                 if resumen["acotadas"] else "")
              + f", {resumen['sin_lectura']} sin lectura")
    if sin_oro:
        print(f"\n{sin_oro} partidas sin gold.jsonl (capturadas sin oro): "
              "el oro por compra llega con su recaptura.")


if __name__ == "__main__":
    main()
