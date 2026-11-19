"""Oro exacto en el instante de cada compra, para los diez jugadores.

El owner lo dijo claro (2026-09-19): no vale la cartera de hace 10 segundos —
hace falta EN el instante de comprar. Ninguna fuente lo da sola:

- Match-V5 da los INSTANTES exactos de compra (`ITEM_PURCHASED`, ms) pero la
  cartera solo cada 60 s.
- El TRUTH-SEEKER da la cartera cada ~8 s (OCR del marcador, `gold.jsonl`)
  pero no sabe de compras.

La resolucion encierra cada compra entre la lectura OCR anterior y la
posterior. En esa ventana el unico ingreso CONTINUO es el oro pasivo
(20,4 / 10 s desde 1:50, determinista), y la propia diferencia entre las dos
lecturas MIDE cuanto ingreso hubo de verdad:

    ingreso_medido = ocr(t2) - ocr(t1) + gastado_en_ventana

Si el ingreso medido coincide con el pasivo (±TOL), la ventana es "quieta"
—que es el caso normal: se compra en fuente— y la cartera en tp es exacta:

    oro_antes(compra_i) = ocr(t1) + pasivo(t1→tp_i) + Σ netos anteriores

Si NO coincide, el caso dominante (medido: los extras son +22, +78, +224... =
CS y asesinatos justo antes de ir a tienda) se resuelve HACIA ATRAS desde la
lectura posterior: despues de tu propia muerte, o quieto en base, tu unico
ingreso es el pasivo — y eso es CERTIFICABLE con lo que ya capturamos (muertes
y respawn de la playerlist cada 8 s, posicion de la trayectoria cada ~3 s, y
los eventos con oro del timeline como marcadores de suciedad):

    oro_antes(compra_i) = ocr(t2) - pasivo(tp_i→t2) - Σ netos desde la compra i

Cuando ni el tramo de ida ni el de vuelta se pueden certificar, la regla 7
manda: no se inventa un numero. La compra queda ACOTADA en [suelo, techo] y
marcada `verificado: false`.

La identidad de la ventana es ademas la VALIDACION: cada compra exacta lleva
su comprobacion encima, no una tolerancia global elegida a mano. Y si un tramo
certificado "limpio" contradice la identidad, gana la identidad: acotado.
"""
import bisect
import json

from .config import CORPUS
from .ocr import LAG_HUD_S     # desfase frame/HUD (<1s); usado en la validacion

# Oro pasivo de la Grieta: 20,4 cada 10 s a partir de 1:50.
PASSIVE_PER_S = 2.04
PASSIVE_START_S = 110.0
# |ingreso_medido - pasivo| admisible en una ventana quieta: el lag del HUD
# (<1 s por lado, ±2 oro/s) mas el redondeo a enteros de las dos lecturas.
TOL_QUIETA = 6.0
# La lectura t2 tiene que ser posterior al ultimo evento con margen: el HUD
# dibujado va por detras del reloj (<1 s medido; ver ocr.LAG_HUD_S).
MARGEN_LECTURA_S = 1.5


def pasivo(t0: float, t1: float) -> float:
    """Oro pasivo devengado en [t0, t1)."""
    a, b = max(t0, PASSIVE_START_S), max(t1, PASSIVE_START_S)
    return PASSIVE_PER_S * max(0.0, b - a)


# ------------------------------------------------------------------- eventos

def eventos_de_compra(timeline: dict, prices: dict, sells: dict) -> dict:
    """{pid: [evento]} con el efecto NETO en cartera de cada evento.

    - compra: -coste_pagado. Al combinar, el juego cobra el total del item
      menos el total de los componentes consumidos, que llegan como
      ITEM_DESTROYED en el MISMO timestamp — se leen de ahi, no se modela.
    - venta: +precio de venta (ddragon `gold.sell`, no un 70% aproximado).
    - deshacer: +goldGain del evento.
    """
    evs = [e for f in timeline["info"]["frames"] for e in f["events"]
           if e["type"] in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_UNDO",
                            "ITEM_DESTROYED")
           and 0 < e.get("participantId", 0) <= 10]
    destruidos = {}
    for e in evs:
        if e["type"] == "ITEM_DESTROYED":
            destruidos.setdefault(
                (e["participantId"], e["timestamp"]), []).append(e["itemId"])
    out = {pid: [] for pid in range(1, 11)}
    for e in sorted(evs, key=lambda x: x["timestamp"]):
        pid, iid = e["participantId"], e.get("itemId", 0)
        t = e["timestamp"] / 1000.0
        if e["type"] == "ITEM_PURCHASED":
            coste = prices.get(iid, 0)
            # Al combinar, el juego cobra el total del item menos lo que ya
            # tenias y consume. Se descuenta TODO lo destruido en el mismo
            # instante, no solo lo que figura en la receta ddragon: hay items
            # (8010 come 3147+3108) que consumen cosas que no son componentes
            # literales, y fiarse de la receta cobra el item entero de mas.
            # EXCEPTO el propio item comprado: un consumible (los tres
            # elixires: 2138 hierro, 2139 hechiceria, 2140 ira) se destruye
            # AL comprarse, en el mismo timestamp, y descontarselo
            # a si mismo lo dejaba a coste 0 cuando la cartera pierde 500
            # (medido por el control de coste certificado; ninguna receta
            # contiene al propio item, asi que la excepcion no rompe combinar).
            for comp in destruidos.get((pid, e["timestamp"]), []):
                if comp == iid:
                    continue
                coste -= prices.get(comp, 0)
            out[pid].append({"t": t, "tipo": "compra", "item": iid,
                             "neto": -max(coste, 0)})
        elif e["type"] == "ITEM_SOLD":
            out[pid].append({"t": t, "tipo": "venta", "item": iid,
                             "neto": +sells.get(iid, 0)})
        elif e["type"] == "ITEM_UNDO":
            out[pid].append({"t": t, "tipo": "deshacer", "item": iid,
                             "neto": +e.get("goldGain", 0)})
    return out


# ----------------------------------------------------------------- resolucion

def _lectura(lecturas: list, ts: list, pid: int, i0: int, paso: int):
    """Primera lectura desde i0 (hacia atras o adelante) con cartera de pid."""
    i = i0
    while 0 <= i < len(lecturas):
        g = lecturas[i][1][pid - 1]
        if g is not None:
            return i, lecturas[i][0], g[0]
        i += paso
    return None, None, None


def resolver_pid(lecturas: list, eventos: list, pid: int,
                 solo_pasivo=None) -> list:
    """Resuelve los eventos de tienda de UN jugador contra sus lecturas.

    lecturas: [(t, [10 x (actual,total) | None])] ordenadas.
    solo_pasivo(ta, tb) -> True si se puede CERTIFICAR que en [ta, tb] el
    jugador solo cobro oro pasivo (muerto, o quieto en base, sin eventos con
    oro). Con None, solo se resuelven las ventanas quietas.

    Devuelve una fila por evento: exacta (ventana quieta o tramo certificado),
    acotada (intervalo) o sin_lectura.
    """
    ts = [l[0] for l in lecturas]
    filas = []
    i = 0
    while i < len(eventos):
        # ventana: eventos consecutivos entre las mismas lecturas utiles
        e0 = eventos[i]
        j0 = bisect.bisect_right(ts, e0["t"]) - 1
        ia, t1, w1 = _lectura(lecturas, ts, pid, j0, -1)
        grupo = [e0]
        k = i + 1
        while True:
            fin = grupo[-1]["t"] + MARGEN_LECTURA_S
            jb = bisect.bisect_left(ts, fin)
            ib, t2, w2 = _lectura(lecturas, ts, pid, jb, +1)
            # si el siguiente evento cae antes de t2, pertenece a esta ventana
            if k < len(eventos) and t2 is not None and eventos[k]["t"] < t2:
                grupo.append(eventos[k]); k += 1
                continue
            break
        i = k

        if t1 is None or t2 is None:
            for e in grupo:
                filas.append(dict(e, pid=pid, metodo="sin_lectura",
                                  oro_antes=None, oro_despues=None,
                                  verificado=False))
            continue
        ventana = {"t1": round(t1, 2), "t2": round(t2, 2)}

        neto_total = sum(e["neto"] for e in grupo)
        ingreso_medido = (w2 - w1) - neto_total
        esperado = pasivo(t1, t2)
        extra = ingreso_medido - esperado
        quieta = abs(extra) <= TOL_QUIETA

        acumulado = 0.0
        neto_grupo = [e["neto"] for e in grupo]
        for gi, e in enumerate(grupo):
            adelante = w1 + pasivo(t1, e["t"]) + acumulado
            if quieta:
                fila = dict(e, pid=pid, **ventana, metodo="ventana_quieta",
                            oro_antes=int(round(adelante)),
                            oro_despues=int(round(adelante + e["neto"])),
                            verificado=True)
            else:
                # Ventana con ingreso extra. Dos tramos posibles de rescate:
                #   ida  [t1, tp]: limpio => adelante es exacto;
                #   vuelta [tp, t2]: limpio => se resuelve hacia atras desde w2
                # (el caso tipico: pelea -> muere/vuelve -> compra: el ingreso
                # sucio cae ANTES de la compra y la vuelta esta limpia).
                atras = (w2 - pasivo(e["t"], t2) - sum(neto_grupo[gi:]))
                ida_ok = solo_pasivo is not None and solo_pasivo(t1, e["t"])
                vuelta_ok = solo_pasivo is not None and solo_pasivo(e["t"], t2)
                if ida_ok and vuelta_ok:
                    # ambos limpios pero la identidad dice que hubo ingreso
                    # extra: el certificado se equivoca en algun lado => manda
                    # la identidad y la compra queda acotada
                    ida_ok = vuelta_ok = False
                if ida_ok or vuelta_ok:
                    antes = adelante if ida_ok else atras
                    fila = dict(e, pid=pid, **ventana,
                                metodo="tramo_ida" if ida_ok else "tramo_vuelta",
                                oro_antes=int(round(antes)),
                                oro_despues=int(round(antes + e["neto"])),
                                verificado=True)
                else:
                    lo, hi = sorted((adelante, adelante + extra))
                    fila = dict(e, pid=pid, **ventana, metodo="acotado",
                                oro_antes=None, oro_despues=None,
                                intervalo=[int(round(lo)), int(round(hi))],
                                extra_ingreso=int(round(extra)),
                                verificado=False)
            filas.append(fila)
            acumulado += e["neto"]
    return filas


def _marcadores_sucios(timeline: dict, equipo_de: dict) -> dict:
    """{pid: [t]} de eventos que pueden dar oro al pid ademas del pasivo.

    Asesinatos donde participa, y todo lo que reparte oro GLOBAL a su equipo
    (torres, placas, monstruos epicos). No se modela CUANTO da cada uno — solo
    marcan un tramo como no-certificable.
    """
    out = {pid: [] for pid in range(1, 11)}
    for f in timeline["info"]["frames"]:
        for e in f["events"]:
            t = e["timestamp"] / 1000.0
            tipo = e["type"]
            if tipo == "CHAMPION_KILL":
                for pid in [e.get("killerId", 0)] + list(
                        e.get("assistingParticipantIds") or []):
                    if 0 < pid <= 10:
                        out[pid].append(t)
            elif tipo in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED",
                          "ELITE_MONSTER_KILL"):
                # el equipo que lo tira cobra; killerTeamId o el equipo del
                # killer. 0 = ejecutado/desconocido: ensucia a los dos equipos.
                kt = e.get("killerTeamId")
                if kt is None:
                    kt = equipo_de.get(e.get("killerId", 0), 0)
                if tipo == "BUILDING_KILL":
                    # el edificio pertenece al equipo VICTIMA: cobra el otro
                    kt = 200 if e.get("teamId") == 100 else 100
                for pid in range(1, 11):
                    if kt == 0 or equipo_de.get(pid) == kt:
                        out[pid].append(t)
    return out


def _certificador_datos(timeline: dict, equipo_de: dict, champ_de: dict,
                        pl: list):
    """Construye solo_pasivo(pid) -> f(ta, tb) desde datos ya en memoria.

    Un tramo es "solo pasivo" para pid si se puede PROBAR que estuvo muerto en
    TODO [ta, tb]: muerto no se farmea ni se cobra objetivo (comprar si se
    puede). El certificado es exacto, no muestreado:

      - el timeline fecha cada muerte al ms (CHAMPION_KILL con pid victima);
      - cada muestra muerta de la playerlist trae `respawn` restante, o sea el
        instante exacto en que ESA muerte termina.

    Se exige: muerte td <= ta, ninguna muerte dentro de (ta, tb] (implicaria
    que estuvo vivo en el tramo), todas las muestras de [td, tb] muertas, y al
    menos una cuyo respawn cubra tb. Antes bastaba con que las muestras
    INTERNAS al tramo estuvieran muertas, y con ~8 s de cadencia el extremo
    quedaba descubierto: vivo en ta, una oleada de CS (que no es evento del
    timeline), muerte, y todas las muestras ya lo pillan muerto — la misma
    trampa medida con "quieto en base" (+234 de oro) colandose por el extremo.
    Sin certificado completo: False, nunca True por defecto.
    """
    sucios = _marcadores_sucios(timeline, equipo_de)
    muertes = {pid: [] for pid in range(1, 11)}
    for f in timeline["info"]["frames"]:
        for e in f["events"]:
            if e["type"] == "CHAMPION_KILL" and 0 < e.get("victimId", 0) <= 10:
                muertes[e["victimId"]].append(e["timestamp"] / 1000.0)
    ts_pl = [r[0] for r in pl]

    def para(pid: int):
        champ = champ_de[pid]
        marcas = sorted(sucios[pid])
        mts = sorted(muertes[pid])

        def solo_pasivo(ta: float, tb: float) -> bool:
            i = bisect.bisect_left(marcas, ta - 1.0)
            if i < len(marcas) and marcas[i] <= tb + 1.0:
                return False
            # la muerte que cubre el tramo empieza ANTES de ta...
            j = bisect.bisect_right(mts, ta) - 1
            if j < 0:
                return False           # nunca ha muerto antes de ta
            td = mts[j]
            # ...y no muere otra vez dentro: habria estado vivo en el tramo
            if j + 1 < len(mts) and mts[j + 1] <= tb:
                return False
            # entre td y tb solo puede haber muestras muertas (de ESA muerte:
            # no hay otra en el rango), y alguna debe traer un respawn que
            # llegue a tb — el extremo de cola, cerrado con el instante exacto
            i0 = bisect.bisect_left(ts_pl, td)
            i1 = bisect.bisect_right(ts_pl, tb + 0.1)
            cubierto = False
            for k in range(i0, i1):
                estado = pl[k][1].get(champ)
                if estado is None or not estado.get("dead"):
                    return False
                r = estado.get("respawn")
                if r is not None and ts_pl[k] + float(r) >= tb:
                    cubierto = True
            return cubierto

        return solo_pasivo

    return para


def _certificador(mid: str, match: dict, timeline: dict):
    """solo_pasivo(pid) para una partida del corpus (ver _certificador_datos)."""
    from .segments import participants
    from .snapshot import load_playerlist

    g = CORPUS / mid
    P = participants(match)
    equipo_de = {pid: v["team"] for pid, v in P.items()}
    champ_de = {pid: v["champ"] for pid, v in P.items()}
    pl = load_playerlist(g / "pass_omni")
    return _certificador_datos(timeline, equipo_de, champ_de, pl)


def _celda(lecturas: list, ts: list, pid: int, t: float):
    """(actual, total) de pid en la lectura de instante ~t, o None.

    t llega redondeado a 2 decimales desde la fila (ventana["t1"/"t2"]), asi
    que se busca con tolerancia en vez de exigir igualdad exacta de floats.
    """
    i = bisect.bisect_left(ts, t - 0.011)
    if i >= len(lecturas) or abs(lecturas[i][0] - t) > 0.03:
        return None
    return lecturas[i][1][pid - 1]


def _valida(filas: list, lecturas: list, timeline: dict) -> dict:
    """Dos controles independientes del resolutor (regla 2b).

    1. **Coste contra la pantalla, en TODAS las ventanas de una compra.** El
       TOTAL del marcador es oro GANADO: no lo mueven ni compras ni ventas,
       asi que `tot2 - tot1` MIDE el ingreso de la ventana — sin asumir "solo
       pasivo" y sin certificado. El coste pagado sale entero de columnas
       medidas:

           coste = w1 + (tot2 - tot1) - w2

       Historia de este control: la seleccion por metodo == ventana_quieta era
       tautologica (media su propio criterio de admision: |err| <= TOL por
       construccion); la seleccion por certificado de muerte ya era un control
       real y cazo el bug de los elixires, pero un certificado falso —ingreso
       real en un tramo "muerto": +190/+243 medidos con lecturas dirigidas—
       metia errores fantasma de ~240 oro que parecian del modelo de precios.
       El total no necesita creerse nada: con el, esos "errores" dieron 1301
       vs 1300, 2497 vs 2500, 1000 vs 1000.
    2. **Anclajes dentro de ventanas**: si un frame de Match-V5 (oro verdadero)
       cae dentro de una ventana con eventos, el oro resuelto se puede
       extrapolar a ese instante y comparar con la verdad. Valida el camino
       hacia atras (tramo_vuelta), que la identidad de la ventana no cubre —
       y tambien el modelo de precios, porque el neto entra en la extrapolacion.
    """
    ts = [l[0] for l in lecturas]

    # -- 1: coste medido (via total) vs calculado, en toda ventana de 1 compra
    errs_coste = []
    por_ventana = {}
    for f in filas:
        if "t1" in f:
            por_ventana.setdefault((f["pid"], f["t1"], f["t2"]), []).append(f)
    for (pid, t1, t2), grupo in por_ventana.items():
        # ventas y deshacer mueven la cartera sin mover el total: solo compras
        if len(grupo) != 1 or grupo[0].get("tipo") != "compra":
            continue
        c1 = _celda(lecturas, ts, pid, t1)
        c2 = _celda(lecturas, ts, pid, t2)
        if c1 is None or c2 is None:
            continue
        medido = (c1[0] + (c2[1] - c1[1])) - c2[0]
        errs_coste.append(medido - (-grupo[0]["neto"]))

    # -- 2: anclajes de Match-V5 dentro de ventanas resueltas
    errs_ancla = []
    frames = {round(fr["timestamp"] / 1000): fr
              for fr in timeline["info"]["frames"]}
    for (pid, t1, t2), grupo in por_ventana.items():
        metodo = grupo[0]["metodo"]
        if metodo not in ("ventana_quieta", "tramo_vuelta"):
            continue
        for ta in [x for x in frames if t1 < x < t2]:
            pf = frames[ta]["participantFrames"].get(str(pid))
            if pf is None:
                continue
            # El frame de Match-V5 en `ta` puede YA reflejar una compra que su
            # PROPIO stream de eventos fecha unas decimas despues (visto en
            # EUW1_7986562911: frame a 540.0 con cartera 8 = post-compra, pero
            # la compra que la vacio esta fechada a 540.2). Comparar contra ese
            # frame una prediccion PRE-compra da un error enorme que no es del
            # dato sino del desajuste frame/evento de la fuente. Se salta el
            # ancla si hay un evento a menos de LAG_HUD_S: no es evaluable.
            if any(abs(f["t"] - ta) <= LAG_HUD_S for f in grupo):
                continue
            # extrapolar desde la ultima fila exacta anterior al ancla; si no
            # la hay, desde w1 (solo en ventana quieta: ahi el tramo es limpio)
            prev = [f for f in grupo if f["t"] <= ta and f["verificado"]]
            if prev:
                base = prev[-1]
                pred = base["oro_despues"] + pasivo(base["t"], ta)
            elif metodo == "ventana_quieta":
                i1 = bisect.bisect_left(ts, t1)
                pred = lecturas[i1][1][pid - 1][0] + pasivo(t1, ta)
            else:
                continue          # tramo sucio antes del ancla: no evaluable
            errs_ancla.append(pred - pf["currentGold"])

    out = {}
    if errs_coste:
        e = sorted(abs(x) for x in errs_coste)
        out["coste"] = {"n": len(e), "err_mediana": e[len(e) // 2],
                        "err_max": e[-1],
                        "exactos_5oro": sum(1 for x in e if x <= 5)}
    if errs_ancla:
        e = sorted(abs(x) for x in errs_ancla)
        out["anclas"] = {"n": len(e), "err_mediana": e[len(e) // 2],
                         "err_max": e[-1]}
    return out


def lecturas_ocr(g) -> list:
    """Lecturas del marcador (OCR), cada ~8 s: [(t, [10 x (cur,tot)|None])]."""
    out = []
    with open(g / "gold.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append((r["t"], [tuple(x) if x else None for x in r["gold"]]))
    return sorted(out)


def lecturas_mv5(timeline: dict) -> list:
    """Cada frame de Match-V5 es oro EXACTO de los diez (cartera y acumulado).

    Son anclas de verdad, gratis y solo de TRUTH-SEEKER (omnisciencia: los diez
    a la vez). Puestas junto a las de OCR acortan las ventanas y las fijan con
    un valor exacto — justo lo que resuelve las ventanas feas donde el OCR solo
    dejaba un intervalo. NO entran en las features de un POV: ahi el oro es solo
    el propio (regla 6); esto es la capa de verdad.
    """
    out = []
    for fr in timeline["info"]["frames"]:
        t = fr["timestamp"] / 1000.0
        fila = []
        for pid in range(1, 11):
            pf = fr["participantFrames"].get(str(pid))
            fila.append((pf["currentGold"], pf["totalGold"]) if pf else None)
        out.append((t, fila))
    return out


def resolver_partida(mid: str) -> tuple:
    """(filas, resumen) para las compras de los DIEZ jugadores de una partida."""
    from . import ddragon
    g = CORPUS / mid
    timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    ocr = lecturas_ocr(g)
    mv5 = lecturas_mv5(timeline)
    lecturas = sorted(ocr + mv5)          # produccion: OCR + anclas exactas
    patch = ".".join(match["info"]["gameVersion"].split(".")[:2])
    v = ddragon.version_for_patch(patch)
    data = ddragon.items(v)["data"]
    prices = {int(k): x["gold"]["total"] for k, x in data.items()}
    sells = {int(k): x["gold"]["sell"] for k, x in data.items()}

    eventos = eventos_de_compra(timeline, prices, sells)
    certificador = _certificador(mid, match, timeline)
    filas = []
    for pid in range(1, 11):
        filas += resolver_pid(lecturas, eventos[pid], pid,
                              solo_pasivo=certificador(pid))
    filas.sort(key=lambda f: (f["t"], f["pid"]))

    EXACTOS = ("ventana_quieta", "tramo_ida", "tramo_vuelta")
    resumen = {
        "compras": len(filas),
        "exactas": sum(1 for f in filas if f["metodo"] in EXACTOS),
        "por_metodo": {m: sum(1 for f in filas if f["metodo"] == m)
                       for m in EXACTOS + ("acotado", "sin_lectura")},
        "acotadas": sum(1 for f in filas if f["metodo"] == "acotado"),
        "sin_lectura": sum(1 for f in filas if f["metodo"] == "sin_lectura"),
    }
    if resumen["acotadas"]:
        anchos = [f["intervalo"][1] - f["intervalo"][0]
                  for f in filas if f["metodo"] == "acotado"]
        resumen["intervalo_mediano"] = sorted(anchos)[len(anchos) // 2]

    # Validacion HONESTA: las anclas de Match-V5 ahora son ENTRADA de la
    # produccion, asi que no pueden ser tambien su prueba. Se resuelve aparte
    # solo con OCR y se contrasta esa resolucion contra las anclas — que la
    # version OCR nunca vio. El control de coste mide el ingreso con el TOTAL
    # del marcador: independiente de la clasificacion y de las anclas.
    filas_ocr = []
    for pid in range(1, 11):
        filas_ocr += resolver_pid(ocr, eventos[pid], pid,
                                  solo_pasivo=certificador(pid))
    resumen["validacion"] = _valida(filas_ocr, ocr, timeline)
    resumen["anclas_mv5_en_produccion"] = sum(
        1 for _, fila in mv5 for g in fila if g is not None)
    return filas, resumen
