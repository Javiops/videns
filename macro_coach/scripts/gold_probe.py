"""¿Se puede reconstruir el oro exacto sin mirar el HUD?

Match-V5 da, cada 60s y para los diez: `currentGold` (cartera) y `totalGold`
(ganado acumulado). Entre dos anclas:

    cartera(N+1) = cartera(N) + [totalGold(N+1) - totalGold(N)] - gastado

donde `gastado` sale de los eventos de compra/venta con los precios de ddragon.
Si esa identidad se cumple, la maquinaria es exacta en los instantes de ancla y
el único error en un instante intermedio viene de repartir el ingreso — que es
un problema mucho más pequeño, y medible.

Esto NO necesita que nadie pulse nada: sirve para tandas desatendidas.

Uso: python scripts/gold_probe.py [MATCH_ID ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import CORPUS, is_excluded
from app.inventory import VISION_ITEMS


RECIPE = {}


def build_recipes(version: str):
    """{item: set(componentes directos e indirectos)} desde ddragon."""
    from app import ddragon
    data = ddragon.items(version)["data"]
    direct = {int(k): [int(x) for x in v.get("from", [])] for k, v in data.items()}

    def expand(i, seen=None):
        seen = seen or set()
        out = set()
        for c in direct.get(i, []):
            if c in seen:
                continue
            out.add(c)
            out |= expand(c, seen | {c})
        return out
    return {i: expand(i) for i in direct}


def probe(mid: str, prices: dict) -> list:
    g = CORPUS / mid
    tl = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
    frames = tl["info"]["frames"]
    errs = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        t0, t1 = a["timestamp"], b["timestamp"]
        # Eventos del intervalo, agrupados por instante: al comprar un item que
        # combina componentes que ya tenias, el juego cobra SOLO el coste de
        # combinacion. Los componentes consumidos llegan como ITEM_DESTROYED
        # en el MISMO timestamp, asi que el descuento se lee de los eventos en
        # vez de modelar la receta.
        evs = [e for f in frames for e in f["events"]
               if t0 < e["timestamp"] <= t1
               and 0 < e.get("participantId", 0) <= 10]
        spend = {pid: 0 for pid in range(1, 11)}
        destroyed = {}
        for e in evs:
            if e["type"] == "ITEM_DESTROYED":
                destroyed.setdefault((e["participantId"], e["timestamp"]), []
                                     ).append(e["itemId"])
        for e in evs:
            pid = e["participantId"]
            iid = e.get("itemId", 0)
            if e["type"] == "ITEM_PURCHASED":
                coste = prices.get(iid, 0)
                if iid not in RECIPE.get(iid, {}):
                    for comp in destroyed.get((pid, e["timestamp"]), []):
                        if comp in RECIPE.get(iid, set()):
                            coste -= prices.get(comp, 0)
                spend[pid] += max(coste, 0)
            elif e["type"] == "ITEM_SOLD":
                spend[pid] -= int(prices.get(iid, 0) * 0.7)   # reembolso 70%
            elif e["type"] == "ITEM_UNDO":
                spend[pid] -= e.get("goldGain", 0)
        for pid in range(1, 11):
            pa = a["participantFrames"].get(str(pid))
            pb = b["participantFrames"].get(str(pid))
            if not pa or not pb:
                continue
            ingreso = pb["totalGold"] - pa["totalGold"]
            pred = pa["currentGold"] + ingreso - spend[pid]
            errs.append(pred - pb["currentGold"])
    return errs


def main():
    from app import ddragon
    mids = sys.argv[1:] or [d.name for d in sorted(CORPUS.iterdir())
                            if d.is_dir() and not is_excluded(d.name)
                            and (d / "mv5_timeline.json").exists()][:10]
    allerr = []
    for mid in mids:
        m = json.loads((CORPUS / mid / "mv5_match.json").read_text(encoding="utf-8"))
        patch = ".".join(m["info"]["gameVersion"].split(".")[:2])
        v = ddragon.version_for_patch(patch)
        prices = ddragon.item_prices(v)
        global RECIPE
        RECIPE = build_recipes(v)
        e = probe(mid, prices)
        allerr += e
        ae = np.abs(e)
        print(f"{mid}  n={len(e):5d}  |error| mediana {np.median(ae):6.1f}  "
              f"p90 {np.percentile(ae, 90):7.1f}  exactos {100*np.mean(ae < 1):5.1f}%")
    ae = np.abs(allerr)
    print(f"\nTOTAL n={len(allerr)}  |error| mediana {np.median(ae):.1f} oro, "
          f"p90 {np.percentile(ae, 90):.1f}, p99 {np.percentile(ae, 99):.1f}")
    print(f"intervalos reconstruidos EXACTOS (<1 oro): {100*np.mean(ae < 1):.1f}%")
    print(f"dentro de 50 oro: {100*np.mean(ae < 50):.1f}%")


if __name__ == "__main__":
    main()
