"""Exporta la revisión de decisiones (los POV seed: SkewMond/Agurin) a la web
estática videns.gg. Es el mismo contenido que sirve review_app.py, pero
precalculado: JSON por partida + minimapas recomprimidos a WebP.

GitHub Pages no tiene servidor, así que todo lo que review_app calcula por
petición (oro causal, compras de la última tienda, top-3 del modelo) se
resuelve aquí una vez. Los veredictos del owner los guarda el navegador en
localStorage y se descargan como feedback.jsonl con el mismo esquema que
POST /api/feedback.

Uso: .venv\\Scripts\\python.exe scripts\\export_web_review.py
     escribe en ../videns-site/review-data/
"""
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

from app.config import CORPUS
from app.lexicon import DISPLAY
import review_app  # oro_en / buys_de / model: exactamente la misma lógica

OUT = Path(__file__).resolve().parent.parent.parent / "videns-site" / "review-data"
IMG_SIZE = 360      # review_app las muestra a 300px; 360 deja margen de zoom
IMG_QUALITY = 75    # WebP: ~22 KB por crop frente a ~230 KB del PNG de 508px

PASSES = [("pass_omni", "omni"), ("pass_blue", "blue"), ("pass_red", "red")]


def convertir(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    im = Image.open(src).convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=IMG_QUALITY)
    dst.write_bytes(buf.getvalue())


def exportar_partida(d: Path, seed: dict) -> dict | None:
    mid, pid = d.name, seed["pid"]
    img_dir = OUT / "img" / mid
    img_dir.mkdir(parents=True, exist_ok=True)
    decisiones = []
    with open(d / "decisions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["pid"] != pid:
                continue
            snap = r["snapshot"]
            minimap = {}
            for pass_name, corto in PASSES:
                ref = snap["minimap"].get(pass_name)
                if not ref:
                    minimap[corto] = None
                    continue
                src = d / ref["file"]
                fname = f"{corto}_{Path(ref['file']).stem.split('_')[-1]}.webp"
                convertir(src, img_dir / fname)
                minimap[corto] = {"img": f"img/{mid}/{fname}", "dt": ref["dt"]}
            v = snap["vision"]
            vision = {k: v.get(k) for k in
                      ("available", "info_pct", "usable", "reason",
                       "n_known", "n_enemies")}
            vision["enemies"] = [{k: e.get(k) for k in
                                  ("champ", "known", "conf", "dead")}
                                 for e in (v.get("enemies") or [])]
            players = [dict({k: p.get(k) for k in
                             ("pid", "champ", "role", "team", "level",
                              "items", "item_gold", "dead", "region_team")},
                            observed={k: (p.get("observed") or {}).get(k)
                                      for k in ("known", "source",
                                                "region_team", "age_s")})
                       for p in snap["players"]]
            decisiones.append({
                "t0": round(r["t0"], 1), "t1": round(r["t1"], 1),
                "token": r["token"],
                "minimap": minimap,
                "vision": vision,
                "players": players,
                "objectives": snap["objectives"],
                "oro": review_app.oro_en(mid, r["t0"]),
                "compras": review_app.buys_de(mid, pid, r["t0"], r["t1"]),
            })
    if not decisiones:
        return None
    equipo = next((p["team"] for p in decisiones[0]["players"]
                   if p["pid"] == pid), None)
    (OUT / f"{mid}.json").write_text(
        json.dumps({"mid": mid, "pid": pid, "seed": seed["label"],
                    "champ": seed["champ"], "team": equipo,
                    "decisions": decisiones},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    return {"mid": mid, "pid": pid, "label": seed["label"],
            "champ": seed["champ"], "team": equipo,
            "n": len(decisiones), "dur": decisiones[-1]["t1"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    juegos = []
    t0 = time.time()
    for d in sorted(CORPUS.iterdir()):
        meta_f = d / "meta.json"
        if not meta_f.exists() or not (d / "decisions.jsonl").exists():
            continue
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        seed = meta.get("seed") or {}
        # solo los POV seed (challengers etiquetados); el prototipo sin seed
        # (la soloq del owner) queda fuera, igual que del corpus de training
        if not seed.get("label"):
            continue
        fila = exportar_partida(d, seed)
        if fila:
            fila["patch"] = meta.get("patch")
            juegos.append(fila)
            print(f"{fila['mid']}  {fila['label']:<9} {fila['champ']:<12} "
                  f"{fila['n']:>3} decisiones  [{time.time()-t0:5.0f}s]")
    (OUT / "games.json").write_text(
        json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M"),
                    "display": DISPLAY,
                    "games": juegos},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n{len(juegos)} partidas, "
          f"{sum(g['n'] for g in juegos)} decisiones, "
          f"{total/1e6:.0f} MB en {OUT}")


if __name__ == "__main__":
    main()
