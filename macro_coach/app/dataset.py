"""Dataset: vector de features f3 (solo lo observable) desde un snapshot, split congelado por
partida y export a JSONL. El split es temporal por gameCreation (70/15/15) y
NUNCA se regenera en silencio (--rebuild-split explícito, como build_tracker).
"""
import json
from pathlib import Path

from .config import (ML, CORPUS, SPLIT_FRACS, LEXICON_V, FEATURES_V, SNAPSHOT_V)
from .lexicon import ALL_TOKENS
from .config import LBN

REGION_IDX = {name: i for i, name in enumerate(sorted(LBN.values()))}
TOKEN_IDX = {t: i for i, t in enumerate(ALL_TOKENS)}

CAMP_ORDER = ["gromp", "azul", "lobos", "raptors", "rojo", "krugs",
              "gromp enemigo", "azul enemigo", "lobos enemigo",
              "raptors enemigo", "rojo enemigo", "krugs enemigo"]


def _tri(v):
    """True/False/None -> 1/0/-1 (los árboles toleran el -1 como 'no sé')."""
    if v is None:
        return -1
    return 1 if v else 0


def feature_vector(decision: dict) -> dict:
    """Features planas y deterministas (versión f3: SOLO lo observable). El orden lo fija
    feature_names(); mantenerlo estable dentro de una versión."""
    s = decision["snapshot"]
    me = s["me"]
    my_team = me["team"]
    players = s["players"]
    mine = [p for p in players if p["team"] == my_team]
    theirs = [p for p in players if p["team"] != my_team]
    my_row = next(p for p in players if p["pid"] == me["pid"])

    f = {"t": s["t"],
         "my_region": REGION_IDX.get(my_row["region_team"], -1),
         "my_level": my_row["level"] or 0,
         "my_jcs": me["jcs"], "my_cs_lane": me["cs_lane"],
         "my_item_gold": my_row["item_gold"],
         "my_dead": _tri(my_row["dead"])}

    # Posición/región SOLO desde la capa observable. Un enemigo en niebla no
    # aporta región: aporta su último avistamiento y su antigüedad, o nada.
    # (Antes se enviaba su región real: omnisciencia.)
    for label, group in (("ally", mine), ("enemy", theirs)):
        group = sorted(group, key=lambda p: p["pid"])
        for i, p in enumerate(group):
            pre = f"{label}{i}"
            obs = p.get("observed") or {}
            known = bool(obs.get("known"))
            f[f"{pre}_role"] = {"TOP": 0, "JG": 1, "MID": 2, "ADC": 3, "SUP": 4,
                                "?": -1}[p["role"]]
            f[f"{pre}_known"] = 1 if known else 0
            f[f"{pre}_region"] = (REGION_IDX.get(obs.get("region_team"), -1)
                                  if known else -1)
            f[f"{pre}_pos_age_s"] = (round(float(obs["age_s"]), 1)
                                     if known and obs.get("age_s") is not None
                                     else -1.0)
            f[f"{pre}_visible"] = _tri(obs.get("visible")) if known else -1
            # nivel/ítems del marcador: visibles para ambos equipos en partida
            # real (el Tab enseña las diez builds). El ORO TOTAL no lo ve
            # ningún jugador — solo el espectador —, así que no entra: la nota
            # del 19-09 sobre el marcador bajo niebla es del ESPECTADOR y no
            # extiende al oro (era omnisciencia; retirado en f3).
            f[f"{pre}_level"] = p["level"] or 0
            f[f"{pre}_item_gold"] = p["item_gold"]
            f[f"{pre}_dead"] = _tri(p["dead"])

    # Diferencia de oro OBSERVABLE (f3): lo que un jugador estima desde su POV
    # — valor de ítems propios menos rivales (builds visibles) más su propia
    # cartera (solo él la conoce; la mejor medida disponible es el frame
    # previo de Match-V5, hasta 60 s rancia, pero es SU dato, no omnisciencia).
    # El gold_diff desde totalGold de los diez era capa de verdad: retirado.
    f["item_gold_diff"] = (sum(p["item_gold"] for p in mine)
                           - sum(p["item_gold"] for p in theirs))
    f["my_wallet_pf"] = my_row["gold_current_prev_frame"] or 0
    f["gold_diff_obs"] = f["item_gold_diff"] + f["my_wallet_pf"]

    camps = s["camps"] or {}
    for c in CAMP_ORDER:
        f[f"camp_{c.replace(' ', '_')}"] = _tri(camps.get(c))

    o = s["objectives"]
    own, opp = (100, 200) if my_team == 100 else (200, 100)

    def _team(d, k):
        return d[k] if k in d else d[str(k)]   # claves int en memoria, str desde JSON

    for key in ("dragons", "grubs", "herald", "baron", "towers_lost"):
        f[f"{key}_own"] = _team(o[key], own)
        f[f"{key}_opp"] = _team(o[key], opp)
    f["next_dragon_in"] = o["next_dragon_in"]
    f["soul_own"] = _tri(None if o["soul_team"] is None else o["soul_team"] == own)

    v = s["vision"]
    usable = bool(v.get("usable")) and v.get("info_pct") is not None
    f["vision_usable"] = 1 if usable else 0
    f["info_pct"] = float(v["info_pct"]) if usable else -1.0
    confs = sorted((e["conf"] for e in (v.get("enemies") or [])
                    if e.get("known") and e.get("conf") is not None),
                   reverse=True) if usable else []
    f["n_enemies_known"] = len(confs)
    for i in range(5):
        f[f"enemy_conf_{i}"] = confs[i] if i < len(confs) else -1.0
    return f


def feature_names() -> list:
    """Orden canónico de las features: el de un vector de ejemplo sintético."""
    dummy = _dummy_decision()
    return list(feature_vector(dummy).keys())


def _dummy_decision():
    players = []
    for pid in range(1, 11):
        team = 100 if pid <= 5 else 200
        players.append({"pid": pid, "champ": f"C{pid}",
                        "role": ["TOP", "JG", "MID", "ADC", "SUP"][(pid - 1) % 5],
                        "team": team, "pos": [0.1, 0.1], "pos_age_s": 0.0,
                        "region": "BASE",
                        "region_team": "BASE", "level": 1, "cs": 0, "dead": False,
                        "respawn_in": 0.0, "gold_total_prev_frame": 500,
                        "gold_current_prev_frame": 500, "xp_prev_frame": 0,
                        "items": [], "item_gold": 0,
                        # capa observable (esquema s2): aliados conocidos,
                        # enemigos desconocidos por defecto
                        "observed": ({"known": True, "source": "own_team",
                                      "region_team": "BASE", "pos": [0.1, 0.1],
                                      "age_s": 0.0, "visible": True}
                                     if team == 100 else
                                     {"known": False, "source": "no_fiable",
                                      "region_team": None, "pos": None,
                                      "age_s": None, "visible": None})})
    return {"snapshot": {
        "t": 0.0, "frame_t": 0.0,
        "me": {"pid": 2, "champ": "C2", "team": 100, "region_team": "BASE",
               "jcs": 0, "cs_lane": 0},
        "players": players,
        "camps": {c: None for c in CAMP_ORDER},
        "objectives": {"dragons": {100: 0, 200: 0}, "grubs": {100: 0, 200: 0},
                       "herald": {100: 0, 200: 0}, "baron": {100: 0, 200: 0},
                       "towers_lost": {100: 0, 200: 0}, "soul_team": None,
                       "next_dragon_in": 300.0},
        "vision": {"available": False, "usable": False, "info_pct": None,
                   "n_known": 0, "n_enemies": 5, "enemies": None,
                   "reason": "dummy"},
        "snapshot_v": SNAPSHOT_V,
        "minimap": {}}}


def corpus_games() -> list:
    """[(match_id, game_creation_ms)] de partidas con decisions.jsonl,
    saltando las excluidas (ver EXCLUDED_MATCHES en app/config.py)."""
    from .config import is_excluded
    out = []
    for d in sorted(CORPUS.iterdir()) if CORPUS.exists() else []:
        if not (d / "decisions.jsonl").exists() or is_excluded(d.name):
            continue
        try:
            m = json.loads((d / "mv5_match.json").read_text(encoding="utf-8"))
            out.append((d.name, m["info"]["gameCreation"]))
        except Exception:
            continue
    out.sort(key=lambda x: x[1])
    return out


class SplitError(RuntimeError):
    """El split guardado no se puede usar tal cual."""


def validate_split(split: dict, corpus_ids: set | None = None) -> list:
    """Lista de motivos por los que un split NO es utilizable (vacía = OK).

    Un split congelado es un compromiso, pero sólo vale si sigue describiendo
    el corpus y el código actuales. Se devolvía tal cual del disco y llegó a
    incluir EUW1_7983109023 (partida propia, excluida): 100 de 220 filas de
    train salían de ella.
    """
    from .config import is_excluded
    reasons = []
    if corpus_ids is None:
        corpus_ids = {m for m, _ in corpus_games()}
    ids = [m for part in ("train", "val", "test") for m in split.get(part, [])]
    dup = {m for m in ids if ids.count(m) > 1}
    if dup:
        reasons.append(f"partidas repetidas entre particiones: {sorted(dup)}")
    excl = sorted(m for m in ids if is_excluded(m))
    if excl:
        reasons.append(f"contiene partidas EXCLUIDAS del corpus: {excl}")
    missing = sorted(m for m in ids if m not in corpus_ids)
    if missing:
        reasons.append(f"IDs que ya no están en el corpus: {missing[:5]}"
                       + (" ..." if len(missing) > 5 else ""))
    if split.get("lexicon_v") != LEXICON_V:
        reasons.append(f"lexicón {split.get('lexicon_v')} != {LEXICON_V}")
    if split.get("features_v") != FEATURES_V:
        reasons.append(f"features {split.get('features_v')} != {FEATURES_V}")
    return reasons


def frozen_split(rebuild: bool = False) -> dict:
    """{'train': [ids], 'val': [ids], 'test': [ids]} temporal por partida.

    Un split existente se VALIDA antes de devolverlo; si no describe el corpus
    y el código actuales se rechaza (SplitError) en vez de contaminar el
    entrenamiento en silencio. Reconstruirlo es siempre explícito.
    """
    path = ML / "split.json"
    if path.exists() and not rebuild:
        split = json.loads(path.read_text(encoding="utf-8"))
        problems = validate_split(split)
        if problems:
            raise SplitError(
                "data/ml/split.json no es utilizable:\n  - "
                + "\n  - ".join(problems)
                + "\nRegenéralo a propósito: build_dataset.py --rebuild-split")
        return split
    games = corpus_games()
    n = len(games)
    ids = [g[0] for g in games]
    if n < 7:
        # con tan pocas partidas un split no significa nada: todo a train,
        # marcado provisional; el split real llega con --rebuild-split
        split = {"train": ids, "val": [], "test": [], "provisional": True,
                 "lexicon_v": LEXICON_V, "features_v": FEATURES_V, "n_games": n}
    else:
        n_tr = int(n * SPLIT_FRACS[0])
        n_va = int(n * SPLIT_FRACS[1])
        split = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va],
                 "test": ids[n_tr + n_va:], "lexicon_v": LEXICON_V,
                 "features_v": FEATURES_V, "n_games": n}
    ML.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


def export(split: dict) -> dict:
    """Escribe decisions_{train,val,test}.jsonl con features + etiqueta.

    Vuelve a comprobar las exclusiones aquí: el export es la última puerta
    antes del entrenamiento y no debe fiarse de que el split esté limpio.
    """
    from .config import is_excluded, SNAPSHOT_V
    problems = validate_split(split)
    if problems:
        raise SplitError("split inválido al exportar:\n  - "
                         + "\n  - ".join(problems))
    counts = {}
    for part in ("train", "val", "test"):
        rows = []
        for mid in split[part]:
            if is_excluded(mid):
                raise SplitError(f"{mid} está excluida y no puede exportarse")
            f = CORPUS / mid / "decisions.jsonl"
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    if d.get("lexicon_v") != LEXICON_V:
                        raise ValueError(f"{mid}: lexicón {d.get('lexicon_v')} "
                                         f"!= {LEXICON_V}; reprocesar la partida")
                    sv = d.get("snapshot", {}).get("snapshot_v")
                    if sv != SNAPSHOT_V:
                        raise ValueError(
                            f"{mid}: snapshot {sv} != {SNAPSHOT_V}; la partida "
                            "viene de otra versión del código, reprocésala")
                    rows.append({"match_id": mid, "pid": d["pid"], "t": d["t0"],
                                 "token": d["token"],
                                 "features": feature_vector(d)})
        out = ML / f"decisions_{part}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        counts[part] = len(rows)
    return counts
