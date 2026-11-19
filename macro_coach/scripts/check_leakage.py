"""Comprueba sobre el corpus real que las features no llevan omnisciencia.

Cuenta, entre los enemigos que el POV NO podía ver, cuántos llegan al modelo
con región. Antes del arreglo eran 312 de 315 en EUW1_7986655074.

Además vigila el ORO de la capa de verdad: ningún jugador ve el totalGold de
los demás (solo el espectador), así que ninguna feature puede llamarse ni
derivarse de él. Este agujero pasó por debajo cuando el check solo miraba
posiciones (f2 enviaba gold_total_pf de los diez y gold_diff_pf).

Uso: python scripts/check_leakage.py [MATCH_ID ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dataset
from app.config import CORPUS, is_excluded


def truth_gold_features() -> list:
    """Nombres de feature que leen oro de la capa de verdad (debe ser []).

    La única cartera admisible es la PROPIA (my_wallet_pf) y sus derivados
    observables (item_gold*, gold_diff_obs). Cualquier gold_total, o el viejo
    gold_diff_pf (suma de totalGold de los diez), es omnisciencia.
    """
    return [n for n in dataset.feature_names()
            if "gold_total" in n or n == "gold_diff_pf"]


def check(mid: str) -> dict:
    f = CORPUS / mid / "decisions.jsonl"
    if not f.exists():
        return {}
    hidden = leaked = known_enemies = 0
    future = 0
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            s = d["snapshot"]
            t = s["t"]
            my_team = s["me"]["team"]
            feats = dataset.feature_vector(d)
            enemies = sorted((p for p in s["players"] if p["team"] != my_team),
                             key=lambda p: p["pid"])
            for i, p in enumerate(enemies):
                obs = p.get("observed") or {}
                # muestra de posición posterior al instante del snapshot
                if p.get("pos_age_s") is not None and p["pos_age_s"] < 0:
                    future += 1
                if obs.get("known"):
                    known_enemies += 1
                    continue
                hidden += 1
                if feats.get(f"enemy{i}_region", -1) != -1:
                    leaked += 1
    return {"match_id": mid, "ocultos": hidden, "con_region_filtrada": leaked,
            "conocidos": known_enemies, "pos_del_futuro": future}


def main():
    bad_gold = truth_gold_features()
    if bad_gold:
        print(f"  FUGA en el vector de features: oro de la capa de verdad "
              f"{bad_gold}  (debe ser [])")
    mids = sys.argv[1:]
    if not mids:
        mids = [d.name for d in sorted(CORPUS.iterdir())
                if d.is_dir() and (d / "decisions.jsonl").exists()
                and not is_excluded(d.name)]
    tot_h = tot_l = tot_k = tot_f = 0
    for mid in mids:
        r = check(mid)
        if not r:
            continue
        tot_h += r["ocultos"]
        tot_l += r["con_region_filtrada"]
        tot_k += r["conocidos"]
        tot_f += r["pos_del_futuro"]
        if r["con_region_filtrada"] or r["pos_del_futuro"]:
            print(f"  FUGA en {mid}: {r}")
    print(f"{len(mids)} partidas | enemigos conocidos {tot_k}, ocultos {tot_h}")
    print(f"regiones de ocultos filtradas al modelo: {tot_l}  (debe ser 0)")
    print(f"posiciones tomadas del futuro: {tot_f}  (debe ser 0)")
    print(f"features de oro de la capa de verdad: {len(bad_gold)}  (debe ser 0)")
    return 1 if (tot_l or tot_f or bad_gold) else 0


if __name__ == "__main__":
    sys.exit(main())
