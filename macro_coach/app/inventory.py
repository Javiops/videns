"""Builds de los 10 por reconstrucción de eventos de Match-V5.

Aproximaciones documentadas (contrastadas empíricamente vía item0..item6 del
match al final — ver test_inventory):
- participantId <= 0: se ignora (quirk real: la partida prototipo trae un
  ITEM_PURCHASED con participantId 0 en t=0).
- Ítems de visión (trinkets 3340/3363/3364 y ward de control 2055) FUERA:
  decisión del owner (skip the wards). Ni se reconstruyen ni se comparan;
  además su compra inicial no siempre emite evento.
- Línea de support (3865->3866->3867): sube por ITEM_DESTROYED del tier
  anterior. El tier final (3869..3877) se otorga SIN evento: al destruirse
  3867 lo conservamos como 3867 (valor equivalente, elección desconocida).
- Botas del parche 16: los tiers altos se otorgan/transforman sin compra
  (ids fantasma 1202/1207, dobles DESTROYED de 1001, tier 3 nunca en
  eventos). Se reconstruye lo comprado; la comparación final excluye la
  familia Boots entera (tag de ddragon) por incomparable.
- ITEM_UNDO: quita beforeId y devuelve afterId; los componentes de una
  combinación deshecha vuelven en silencio (no hay evento) — discrepancia
  conocida y medible, no corregible desde la API.
- ITEM_DESTROYED de un ítem no poseído (consumibles regalados, quirks): se
  ignora salvo cadena de support.
"""
from collections import Counter

SUPPORT_CHAIN = {3865: 3866, 3866: 3867}
SUPPORT_TERMINAL = 3867
SUPPORT_FINALS = {3869, 3870, 3871, 3876, 3877}
VISION_ITEMS = {3340, 3363, 3364, 2055}

ITEM_EVENTS = ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO")


def item_events(timeline: dict) -> list:
    evs = []
    for f in timeline["info"]["frames"]:
        for e in f["events"]:
            if e["type"] in ITEM_EVENTS:
                evs.append(e)
    evs.sort(key=lambda e: e["timestamp"])
    return evs


def inventories_at(timeline: dict, t_s: float) -> dict:
    """{pid: Counter(itemId)} con los eventos hasta t_s (segundos)."""
    inv: dict[int, Counter] = {pid: Counter() for pid in range(1, 11)}
    cutoff = t_s * 1000.0
    for e in item_events(timeline):
        if e["timestamp"] > cutoff:
            break
        pid = e.get("participantId", 0)
        if pid <= 0 or pid > 10:
            continue
        kind = e["type"]
        iid = e.get("itemId", 0)
        if iid in VISION_ITEMS:
            continue
        c = inv[pid]
        if kind == "ITEM_PURCHASED":
            c[iid] += 1
        elif kind == "ITEM_SOLD":
            if c[iid] > 0:
                c[iid] -= 1
        elif kind == "ITEM_DESTROYED":
            if iid == SUPPORT_TERMINAL and c[iid] > 0:
                pass                         # sube a un final desconocido: lo conservamos
            elif c[iid] > 0:
                c[iid] -= 1
                if iid in SUPPORT_CHAIN:
                    c[SUPPORT_CHAIN[iid]] += 1
            elif iid in SUPPORT_CHAIN:       # otorgado sin compra visible
                c[SUPPORT_CHAIN[iid]] += 1
        elif kind == "ITEM_UNDO":
            before, after = e.get("beforeId", 0), e.get("afterId", 0)
            if before > 0 and c[before] > 0:
                c[before] -= 1
            if after > 0 and after not in VISION_ITEMS:
                c[after] += 1
        c += Counter()                       # purga ceros
    return inv


def items_list(counter: Counter) -> list:
    out = []
    for iid in sorted(counter):
        out.extend([iid] * counter[iid])
    return out


def gold_value(counter: Counter, prices: dict) -> int:
    return int(sum(prices.get(iid, 0) * n for iid, n in counter.items()))


def final_diff_vs_match(timeline: dict, match: dict,
                        boots_ids: set | None = None) -> dict:
    """Discrepancia por jugador entre la reconstrucción al final y los
    item0..item6 del match, sobre lo COMPARABLE: fuera visión (decisión del
    owner), finales de support canonicalizados a 3867 y, si se pasa
    boots_ids (tag Boots de ddragon), fuera la familia de botas (se otorgan
    sin evento en el parche 16)."""
    end_t = timeline["info"]["frames"][-1]["timestamp"] / 1000.0
    inv = inventories_at(timeline, end_t + 60)
    boots = boots_ids or set()
    out = {}
    for p in match["info"]["participants"]:
        pid = p["participantId"]
        official = Counter()
        for i in range(7):
            iid = p[f"item{i}"]
            if iid == 0 or iid in VISION_ITEMS or iid in boots:
                continue
            if iid in SUPPORT_FINALS:
                iid = SUPPORT_TERMINAL
            official[iid] += 1
        mine = Counter({iid: n for iid, n in inv[pid].items()
                        if iid not in boots})
        missing = official - mine
        extra = mine - official
        out[pid] = {"missing": dict(missing), "extra": dict(extra),
                    "ok": not missing and not extra}
    return out
