"""Export versionado del oro por compra, para consumo EXTERNO (build_tracker).

Decision del owner (2026-09-20): la regla "no shared data" entre macro_coach y
build_tracker se levanta SOLO via artefacto exportado y versionado — nunca
codigo compartido ni acceso cruzado a los corpus. Este script produce ese
artefacto: un JSONL plano con el oro resuelto en cada evento de tienda de los
diez jugadores de cada partida del corpus, mas un meta con procedencia,
validacion agregada y sha256.

Contrato del formato (buys_export_v1):
  - una fila por evento de tienda (compra/venta/deshacer) de Match-V5;
  - `ts_ms` es el timestamp EXACTO del evento en Match-V5: es la clave de
    cruce natural con cualquier reconstruccion hecha sobre el mismo timeline;
  - `oro_antes`/`oro_despues` solo cuando el metodo es exacto
    (`verificado: true`); si no, `intervalo: [suelo, techo]` — la
    incertidumbre de la ventana de ~8 s del OCR. NUNCA un numero inventado
    (regla 7 de macro_coach);
  - `metodo` documenta como se resolvio (ventana_quieta / tramo_ida /
    tramo_vuelta / acotado / sin_lectura).

Uso: python scripts/export_buys.py [--out DIR]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CORPUS, DATA, is_excluded

VERSION = "buys_export_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "exports"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{VERSION}.jsonl"

    partidas = 0
    n = {"exactos": 0, "acotados": 0, "sin_lectura": 0}
    coste = {"n": 0, "exactos_5oro": 0, "err_max": 0.0}
    anclas = {"n": 0, "err_max": 0.0}
    with open(out, "w", encoding="utf-8") as fh:
        for d in sorted(CORPUS.iterdir()):
            f = d / "buys.jsonl"
            if not f.exists() or is_excluded(d.name):
                continue
            partidas += 1
            m = json.loads((d / "mv5_match.json").read_text(encoding="utf-8"))
            champ_de = {p["participantId"]: (p["championId"], p["championName"])
                        for p in m["info"]["participants"]}
            rep = json.loads((d / "buys_report.json").read_text(encoding="utf-8"))
            v = rep.get("validacion") or {}
            if v.get("coste"):
                coste["n"] += v["coste"]["n"]
                coste["exactos_5oro"] += v["coste"]["exactos_5oro"]
                coste["err_max"] = max(coste["err_max"], v["coste"]["err_max"])
            if v.get("anclas"):
                anclas["n"] += v["anclas"]["n"]
                anclas["err_max"] = max(anclas["err_max"], v["anclas"]["err_max"])
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                exacto = bool(r.get("verificado"))
                n["exactos" if exacto else
                  ("acotados" if r["metodo"] == "acotado" else "sin_lectura")] \
                    += 1 if exacto or r["metodo"] in ("acotado", "sin_lectura") else 0
                cid, cname = champ_de.get(r["pid"], (None, None))
                fh.write(json.dumps({
                    "match_id": d.name, "pid": r["pid"],
                    "champion_id": cid, "champion": cname,
                    "ts_ms": int(round(r["t"] * 1000)),
                    "tipo": r["tipo"], "item": r.get("item"),
                    "neto": r["neto"], "metodo": r["metodo"],
                    "verificado": exacto,
                    "oro_antes": r.get("oro_antes") if exacto else None,
                    "oro_despues": r.get("oro_despues") if exacto else None,
                    "intervalo": r.get("intervalo"),
                }) + "\n")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "version": VERSION,
        "generado_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fuente": ("macro_coach: OCR del marcador (TRUTH-SEEKER, ~8 s) + "
                   "anclas Match-V5 (60 s) + resolucion app/buys.py"),
        "partidas": partidas,
        "eventos": sum(n.values()),
        **n,
        "validacion": {
            "coste_universal_via_total": dict(coste),
            "anclas_mv5_sin_verlas": dict(anclas),
            "nota": ("coste = w1 + (tot2-tot1) - w2 en toda ventana de una "
                     "compra; anclas: resolucion solo-OCR contrastada contra "
                     "frames de Match-V5 que nunca vio"),
        },
        "sha256": sha,
        "contrato": ("ts_ms = timestamp exacto del evento en el timeline de "
                     "Match-V5 (clave de cruce). oro_antes/oro_despues solo si "
                     "verificado; si no, intervalo [suelo, techo]. Nunca un "
                     "numero inventado."),
    }
    meta_f = out_dir / f"{VERSION}.meta.json"
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{out}  ({meta['eventos']} eventos de {partidas} partidas)")
    print(f"  exactos {n['exactos']} / acotados {n['acotados']} / "
          f"sin_lectura {n['sin_lectura']}")
    print(f"  validacion coste: n={coste['n']}, <=5oro {coste['exactos_5oro']}, "
          f"err_max {coste['err_max']:.0f}; anclas: n={anclas['n']}, "
          f"err_max {anclas['err_max']:.1f}")
    print(f"  sha256 {sha}")
    print(f"  meta: {meta_f}")


if __name__ == "__main__":
    main()
