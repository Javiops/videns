"""Lexicón v1 — tokens canónicos de decisión. El contrato está en LEXICON.md;
este módulo es su única implementación. Cambiar reglas = subir LEXICON_V en
config y regenerar decisions.jsonl de todo el corpus.

Un segmento estructurado (app/segments.py) se mapea aquí a un token. El punto
de decisión es el inicio del segmento; los tránsitos heredan la etiqueta del
siguiente segmento real (salvo jungla enemiga: pisarla ya es INVADIR).
"""
from .config import HERALD_SPAWN_S, BARON_SPAWN_S

RESET = "RESET"

FARM = {"JGL_PA": "FARMEAR_PROPIA_AZUL", "JGL_PR": "FARMEAR_PROPIA_ROJA"}
INVADE = {"JGL_EA": "INVADIR_AZUL", "JGL_ER": "INVADIR_ROJA"}
GANK = {"TOP": "GANK_TOP", "MID": "GANK_MID", "BOT": "GANK_BOT"}
PELEA = {"TOP": "PELEAR_TOP", "MID": "PELEAR_MID", "BOT": "PELEAR_BOT"}
CUBRIR = {"TOP": "CUBRIR_TOP", "MID": "CUBRIR_MID", "BOT": "CUBRIR_BOT"}
CENTINELA = {"RIO_TOP": "CENTINELA_TOP", "RIO_BOT": "CENTINELA_BOT"}
ROTAR = {"RIO_TOP": "ROTAR_RIO_TOP", "RIO_BOT": "ROTAR_RIO_BOT"}
OBJETIVO = {"DRAGON": "OBJETIVO_DRAGON", "LARVAS": "OBJETIVO_LARVAS",
            "HERALDO": "OBJETIVO_HERALDO", "BARON": "OBJETIVO_BARON"}

ALL_TOKENS = ([RESET] + sorted(FARM.values()) + sorted(INVADE.values())
              + [GANK[l] for l in ("TOP", "MID", "BOT")]
              + [PELEA[l] for l in ("TOP", "MID", "BOT")]
              + [CUBRIR[l] for l in ("TOP", "MID", "BOT")]
              + [CENTINELA["RIO_TOP"], CENTINELA["RIO_BOT"]]
              + [ROTAR["RIO_TOP"], ROTAR["RIO_BOT"]]
              + [OBJETIVO[o] for o in ("DRAGON", "LARVAS", "HERALDO", "BARON")])

DISPLAY = {
    RESET: "resetea (vuelve a base)",
    "FARMEAR_PROPIA_AZUL": "farmea jungla propia · cuadrante azul",
    "FARMEAR_PROPIA_ROJA": "farmea jungla propia · cuadrante rojo",
    "INVADIR_AZUL": "invade jungla enemiga · cuadrante azul",
    "INVADIR_ROJA": "invade jungla enemiga · cuadrante rojo",
    "GANK_TOP": "gankea TOP", "GANK_MID": "gankea MID", "GANK_BOT": "gankea BOT",
    "PELEAR_TOP": "pelea en TOP (lado propio)",
    "PELEAR_MID": "pelea en MID (lado propio)",
    "PELEAR_BOT": "pelea en BOT (lado propio)",
    "CUBRIR_TOP": "cubre TOP", "CUBRIR_MID": "cubre MID", "CUBRIR_BOT": "cubre BOT",
    "CENTINELA_TOP": "centinela superior", "CENTINELA_BOT": "centinela inferior",
    "ROTAR_RIO_TOP": "rota por río superior", "ROTAR_RIO_BOT": "rota por río inferior",
    "OBJETIVO_DRAGON": "juega el dragón", "OBJETIVO_LARVAS": "juega las larvas",
    "OBJETIVO_HERALDO": "juega el heraldo", "OBJETIVO_BARON": "juega el Barón",
}


def baron_pit_objective(t: float, monster_type: str | None = None) -> str:
    """Desambiguación del foso de Barón: por evento si lo hay, por reloj si no."""
    if monster_type == "HORDE":
        return OBJETIVO["LARVAS"]
    if monster_type == "RIFTHERALD":
        return OBJETIVO["HERALDO"]
    if monster_type == "BARON_NASHOR":
        return OBJETIVO["BARON"]
    if t < HERALD_SPAWN_S:
        return OBJETIVO["LARVAS"]
    if t < BARON_SPAWN_S:
        return OBJETIVO["HERALDO"]
    return OBJETIVO["BARON"]


def token_for_segment(seg: dict) -> str | None:
    """Token de un segmento estructurado, o None si es tránsito/hueco.

    seg: {"kind": ..., "region_team": región en frame del equipo, "lane": ...,
          "farming": bool, "monster": tipo|None, "t0": s}
    kinds: base, reset, lane_gank, lane_fight, lane_cover, jungle, river,
           pit_dragon, pit_baron, dead
    """
    k = seg["kind"]
    if k == "reset":
        return RESET
    if k == "base" or k == "dead":
        return None                      # estar en base / muerto no es decisión
    if k == "lane_gank":
        return GANK[seg["lane"]]
    if k == "lane_fight":
        return PELEA[seg["lane"]]
    if k == "lane_cover":
        return CUBRIR[seg["lane"]]
    if k in ("jungle", "river") and seg.get("fight_lanes"):
        # regla auditada fase 1: pelea con un laner rival dentro de la jungla
        # se nombra por el carril de la víctima (GANK X -> INVADE ...)
        return GANK[seg["fight_lanes"][0]]
    if k == "jungle":
        reg = seg["region_team"]
        if reg in FARM:
            return FARM[reg] if seg.get("farming") else None   # cruza propia = tránsito
        if reg in INVADE:
            return INVADE[reg]           # pisar jungla enemiga ya es la decisión
        return None
    if k == "river":
        reg = seg["region_team"]
        if seg.get("farming"):
            return CENTINELA[reg]
        return ROTAR[reg]
    if k == "pit_dragon":
        return OBJETIVO["DRAGON"]
    if k == "pit_baron":
        return baron_pit_objective(seg["t0"], seg.get("monster"))
    raise ValueError(f"kind desconocido: {k}")


def decisions_from_segments(segments: list) -> list:
    """Puntos de decisión: (t0, token, cadena). Tránsitos heredan hacia
    delante; tokens consecutivos idénticos se colapsan."""
    toks = [token_for_segment(s) for s in segments]
    # herencia hacia delante para tránsitos (None que no sean base/muerte)
    for i in range(len(segments) - 1, -1, -1):
        if toks[i] is None and segments[i]["kind"] in ("jungle", "river"):
            toks[i] = toks[i + 1] if i + 1 < len(toks) else None
    out = []
    prev = None
    for seg, tok in zip(segments, toks):
        if tok is None:
            if seg["kind"] in ("base", "dead"):
                prev = None              # tras base/muerte, repetir token es decisión nueva
            continue
        if tok == prev and out and seg["t0"] - out[-1]["t1"] <= 20:
            out[-1]["t1"] = seg["t1"]
            out[-1]["chain"].append(seg["t0"])
            continue
        out.append({"t0": seg["t0"], "t1": seg["t1"], "token": tok,
                    "chain": [seg["t0"]], "kind": seg["kind"]})
        prev = tok
    return out
