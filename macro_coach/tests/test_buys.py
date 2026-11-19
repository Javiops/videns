"""Oro exacto en el instante de compra (app/buys.py).

Lo que se protege: que una ventana quieta se resuelva exacta; que el coste de
combinacion salga de los ITEM_DESTROYED (no de la receta ddragon, que cobra de
mas en items como 8010); que un ingreso no localizable NO se invente sino que
acote; y que un certificado de "muerto" rescate un tramo pero uno de posicion
no (el CS entre muestras se colaba).
"""
from app import buys


def _tl(frames):
    """timeline minimo: frames = [(t_s, {pid: (cur,tot)}, [eventos])]."""
    return {"info": {"frames": [
        {"timestamp": int(t * 1000),
         "participantFrames": {str(p): {"currentGold": c, "totalGold": g}
                               for p, (c, g) in pf.items()},
         # los eventos reales llevan su propio timestamp (ms); si el test no lo
         # da, se sella con el del frame
         "events": [dict(e, timestamp=e.get("timestamp", int(t * 1000)))
                    for e in evs]}
        for t, pf, evs in frames]}}


def _lect(rows):
    """rows = [(t, {pid: (cur,tot)})] -> formato de lecturas de resolver_pid."""
    out = []
    for t, d in rows:
        fila = [d.get(p) for p in range(1, 11)]
        out.append((t, fila))
    return sorted(out)


def test_ventana_quieta_es_exacta():
    # pid1 compra un item de 500 entre dos lecturas; el ingreso medido = pasivo
    lect = _lect([(300.0, {1: (900, 5000)}),
                  (320.0, {1: (441, 5041)})])   # 900+~41 pasivo-500 = 441
    ev = [{"t": 310.0, "tipo": "compra", "item": 1055, "neto": -500}]
    filas = buys.resolver_pid(lect, ev, 1)
    assert filas[0]["metodo"] == "ventana_quieta"
    assert filas[0]["verificado"] is True
    # oro_antes = 900 + pasivo(300->310) ~ 920
    assert abs(filas[0]["oro_antes"] - 920) <= 2


def test_ingreso_no_localizable_se_acota_no_se_inventa():
    # sube 300 de mas (un CS/asesinato sin evento): sin certificado -> acotado
    lect = _lect([(300.0, {1: (900, 5000)}),
                  (320.0, {1: (741, 5341)})])   # +300 extra
    ev = [{"t": 310.0, "tipo": "compra", "item": 1055, "neto": -500}]
    filas = buys.resolver_pid(lect, ev, 1)      # sin solo_pasivo
    assert filas[0]["metodo"] == "acotado"
    assert filas[0]["oro_antes"] is None
    lo, hi = filas[0]["intervalo"]
    assert hi - lo >= 250          # el ancho refleja el ingreso no localizado


def test_muerto_certifica_tramo_pero_posicion_no():
    # ingreso extra en la ventana; la compra ocurre tras la muerte del jugador.
    lect = _lect([(300.0, {1: (900, 5000)}),
                  (330.0, {1: (761, 5361)})])   # +300 extra en la ventana
    ev = [{"t": 320.0, "tipo": "compra", "item": 1055, "neto": -500}]
    # certificado: limpio (solo pasivo) DESPUES de t=315 (murio y compro muerto)
    filas = buys.resolver_pid(lect, ev, 1,
                              solo_pasivo=lambda a, b: a >= 315.0)
    assert filas[0]["metodo"] == "tramo_vuelta"
    assert filas[0]["verificado"] is True
    # sin certificado, la misma ventana se acota
    filas2 = buys.resolver_pid(lect, ev, 1, solo_pasivo=lambda a, b: False)
    assert filas2[0]["metodo"] == "acotado"


def test_identidad_veta_certificado_contradictorio():
    # ambos tramos "limpios" segun el certificado PERO la identidad ve ingreso
    # extra: gana la identidad, se acota (el certificado miente en algun lado).
    lect = _lect([(300.0, {1: (900, 5000)}),
                  (320.0, {1: (741, 5341)})])   # +300 extra
    ev = [{"t": 310.0, "tipo": "compra", "item": 1055, "neto": -500}]
    filas = buys.resolver_pid(lect, ev, 1, solo_pasivo=lambda a, b: True)
    assert filas[0]["metodo"] == "acotado"


def test_coste_de_combinacion_desde_item_destroyed():
    # 8010 (total 2900) consume 3147(1300)+3108(850) que ya se tenian: el juego
    # cobra 2900-1300-850 = 750, y eso hay que leerlo de los ITEM_DESTROYED.
    tl = _tl([(300.0, {}, [
        {"type": "ITEM_DESTROYED", "participantId": 1, "itemId": 3147},
        {"type": "ITEM_DESTROYED", "participantId": 1, "itemId": 3108},
        {"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 8010},
    ])])
    prices = {8010: 2900, 3147: 1300, 3108: 850}
    ev = buys.eventos_de_compra(tl, prices, sells={})
    assert ev[1][0]["neto"] == -750, "debe descontar lo destruido, no cobrar 2900"


def test_consumible_no_se_descuenta_a_si_mismo():
    """Un elixir se destruye al comprarse (ITEM_DESTROYED del MISMO item en el
    mismo ms): eso es el consumo de lo recien pagado, no un componente. El
    descuento de combinacion lo dejaba a coste 0; la cartera pierde 500
    (visto en EUW1_7985615447 pid8 t=1373.9 por el control certificado)."""
    tl = _tl([(1373.9, {}, [
        {"type": "ITEM_DESTROYED", "participantId": 8, "itemId": 2138},
        {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 2138},
    ])])
    ev = buys.eventos_de_compra(tl, prices={2138: 500}, sells={})
    assert ev[8][0]["neto"] == -500


def test_venta_usa_precio_ddragon():
    tl = _tl([(300.0, {}, [
        {"type": "ITEM_SOLD", "participantId": 2, "itemId": 1055}])])
    ev = buys.eventos_de_compra(tl, prices={1055: 450}, sells={1055: 180})
    assert ev[2][0]["neto"] == 180


def test_validacion_ignora_anclas_pegadas_a_una_compra():
    """El frame de Match-V5 puede reflejar una compra que su propio stream fecha
    unas decimas despues (visto: frame a 540.0 con cartera post-compra, compra
    fechada a 540.2). La validacion por anclas debe SALTARSE esos anclajes, o
    marcaria como error de ~1600 oro un dato que es correcto."""
    # cartera 1602 -> compra cara (1600) -> 2; el frame en 540 ya muestra los 2
    tl = _tl([
        (534.0, {2: (1245, 3195)}, []),
        (540.0, {2: (2, 3558)}, [                       # frame POST-compra
            {"type": "ITEM_PURCHASED", "participantId": 2, "itemId": 3031,
             "timestamp": 540200}]),                    # ...compra fechada +0.2s
        (547.0, {2: (16, 3572)}, []),
    ])
    ocr = _lect([(534.0, {2: (1245, 3195)}), (547.0, {2: (16, 3572)})])
    ev = buys.eventos_de_compra(tl, prices={3031: 1600}, sells={})
    filas = buys.resolver_pid(ocr + buys.lecturas_mv5(tl), ev[2], 2,
                              solo_pasivo=lambda a, b: False)
    inf = buys._valida(filas, ocr, tl)
    # sin el salto, saldria un error de ~1600; con el, el ancla no es evaluable
    assert inf.get("anclas", {}).get("err_max", 0) < 50, inf


def test_certificado_muerte_exige_cubrir_extremos():
    """Regresion del hueco del extremo: vivo en ta, farmea (el CS no es evento
    del timeline), muere, y todas las muestras INTERNAS del tramo ya lo pillan
    muerto. El certificado viejo daba True; el nuevo exige muerte fechada
    <= ta (CHAMPION_KILL, exacto al ms) y una muestra cuyo respawn cubra tb."""
    equipo = {p: (100 if p <= 5 else 200) for p in range(1, 11)}
    champ = {p: f"C{p}" for p in range(1, 11)}
    pl = [(308.0, {"C1": {"dead": True, "respawn": 20.0}}),
          (316.0, {"C1": {"dead": True, "respawn": 12.0}})]
    # muere a t=305, DENTRO del tramo [300, 320]: estuvo vivo en [300, 305)
    tl = _tl([(305.0, {}, [
        {"type": "CHAMPION_KILL", "victimId": 1, "killerId": 6}])])
    sp = buys._certificador_datos(tl, equipo, champ, pl)(1)
    assert sp(300.0, 320.0) is False, "muerte posterior a ta: hueco de cabeza"
    # muerto ANTES del tramo y respawn que cubre tb: certificado valido
    tl2 = _tl([(298.0, {}, [
        {"type": "CHAMPION_KILL", "victimId": 1, "killerId": 6}])])
    sp2 = buys._certificador_datos(tl2, equipo, champ, pl)(1)
    assert sp2(300.0, 320.0) is True
    # respawn corto: resucita antes de tb -> el extremo de cola queda vivo
    pl3 = [(308.0, {"C1": {"dead": True, "respawn": 5.0}})]
    sp3 = buys._certificador_datos(tl2, equipo, champ, pl3)(1)
    assert sp3(300.0, 320.0) is False


def test_control_de_coste_ve_un_precio_erroneo():
    """El control de coste mide el ingreso con el TOTAL del marcador (oro
    GANADO: las compras no lo mueven), asi que no asume "solo pasivo" ni
    necesita certificado, y cubre TODA ventana de una compra. La seleccion
    vieja por metodo == ventana_quieta era tautologica (|err| <= TOL por
    construccion); la intermedia por certificado de muerte metia errores
    fantasma cuando el certificado fallaba. Aqui la pantalla mide 500 de
    coste (900 + 41 de ingreso segun el total - 441) y el neto dice 450:
    los 50 de discrepancia SE VEN."""
    ocr = _lect([(300.0, {1: (900, 5000)}),
                 (320.0, {1: (441, 5041)})])   # salieron ~500 de la cartera
    fila = {"pid": 1, "t1": 300.0, "t2": 320.0, "neto": -450, "t": 310.0,
            "tipo": "compra", "metodo": "acotado", "verificado": False}
    tl = _tl([(300.0, {}, [])])
    inf = buys._valida([fila], ocr, tl)
    assert inf["coste"]["n"] == 1
    assert inf["coste"]["err_max"] >= 40
    # el ingreso viene del TOTAL, no de asumir pasivo: con un ingreso extra
    # de 300 (CS) reflejado en ambas columnas, el coste medido no se ensucia
    ocr2 = _lect([(300.0, {1: (900, 5000)}),
                  (320.0, {1: (741, 5341)})])  # +300 extra, coste real 500
    fila2 = dict(fila, neto=-500)
    inf2 = buys._valida([fila2], ocr2, tl)
    assert inf2["coste"]["err_max"] <= 6


def test_pasivo():
    assert buys.pasivo(0, 110) == 0.0          # nada antes de 1:50
    assert abs(buys.pasivo(110, 120) - 20.4) < 1e-6
