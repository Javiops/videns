"""Lectura del oro exacto del marcador (OCR).

Con la accion de oro activada, el marcador del espectador muestra
`actual(total)` para los diez. Medido contra Match-V5 en el MISMO instante
(EUW1_7986655074, t=900): coincide jugador a jugador salvo 1-2 de oro, que es
la renta pasiva de la fraccion de segundo entre una lectura y la otra. Es la
UNICA fuente de oro instantaneo que tenemos: los `participantFrames` de
Match-V5 son de cada 60s y la Live Client API no expone oro en replay.

La celda se localiza SOLA. En cada fila del marcador se segmentan los grupos de
texto y se acepta el unico que encaja con `N(N)`: el KDA lleva barras, el CS y
la puntuacion de vision no llevan parentesis, y la recompensa ("200 oro") lleva
letras. Asi no hay coordenadas de columna escritas a mano que se rompan cuando
el owner mueve el HUD.

Contrato (regla 7): un glifo por debajo de MIN_SCORE no se adivina. Se descarta
la lectura de esa celda y el oro de ese jugador queda en None, nunca en un
numero plausible.
"""
import cv2
import numpy as np

from .config import DATA

TEMPLATES_FILE = DATA / "ocr_digits.npz"

ALFABETO = "0123456789()"
UMBRAL = 130           # pixel de texto: el fondo del marcador es casi negro
GLIFO_H, GLIFO_W = 16, 12    # caja normalizada del glifo
MIN_SCORE = 0.80       # correlacion minima para dar por leido un glifo
FILAS = 5              # jugadores por equipo
# Separacion (px) a partir de la cual dos glifos son de celdas distintas.
# Medido en el marcador: dentro de un numero los glifos dejan hasta 5px (los
# '1' son estrechos y quedan sueltos) y entre columnas hay 55px o mas. El
# margen es comodo, asi que el valor no es un ajuste fino.
GAP_CELDA = 12
MAX_ORO_ACTUAL = 30000
MAX_ORO_TOTAL = 100000

# El HUD dibujado va DETRAS del reloj de la replay: el recorte tomado cuando la
# API dice `t` muestra el estado de un instante algo anterior. Medido en
# EUW1_7984516268 a 32x (195 recortes): de los 4 recortes tomados dentro del
# primer segundo tras un anclaje, 3 leen por debajo del oro de ese anclaje; de
# los 191 tomados a mas de 1s, ninguno. Asi que el desfase es < 1s de partida
# (a 32x, ~30ms de reloj real: uno o dos fotogramas de render).
#
# Para las features da igual —es informacion del pasado, nunca del futuro— pero
# la horquilla tiene que tenerlo en cuenta o marca como error una lectura
# correcta de hace media exhalacion.
LAG_HUD_S = 1.0


# --------------------------------------------------------------- segmentacion

def _mascara(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return (g > UMBRAL).astype(np.uint8)


def _bandas(proj: np.ndarray, minimo: int, largo_min: int) -> list:
    """Tramos consecutivos de `proj` por encima de `minimo`."""
    out, ini = [], None
    for i, v in enumerate(proj):
        if v > minimo and ini is None:
            ini = i
        elif v <= minimo and ini is not None:
            if i - ini >= largo_min:
                out.append((ini, i))
            ini = None
    if ini is not None and len(proj) - ini >= largo_min:
        out.append((ini, len(proj)))
    return out


def celdas(m: np.ndarray) -> list:
    """Cajas (x0, y0, x1, y1) de cada celda de texto del marcador.

    Se dilata en horizontal para pegar los glifos de un mismo numero y se
    toman componentes conexas: cada una es una celda. No se buscan filas por
    proyeccion porque la fila del marcador no tiene una altura constante (la
    recompensa "200 oro" vive por encima del KDA y desplaza la banda); las
    celdas, en cambio, salen limpias y con ellas vienen sus coordenadas.
    """
    d = cv2.dilate(m, np.ones((1, GAP_CELDA), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(d, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 8 or w < 8 or area < 24:
            continue
        out.append((x, y, x + w, y + h))
    return out


def glifos(m: np.ndarray) -> list:
    """Glifos (x0, x1) dentro de un grupo: aqui NO se unen."""
    return _bandas(m.sum(axis=0), minimo=0, largo_min=2)


def normaliza(parche: np.ndarray) -> np.ndarray:
    """Glifo a caja fija conservando la proporcion.

    La proporcion es informacion: el '1' mide 5px de ancho y los parentesis 3,
    frente a 8-9 del resto. Estirarlos a la misma caja los volveria confundibles.
    """
    ys = np.where(parche.any(axis=1))[0]
    if len(ys) == 0:
        return np.zeros((GLIFO_H, GLIFO_W), np.float32)
    p = parche[ys[0]:ys[-1] + 1]
    h, w = p.shape
    nw = max(1, min(GLIFO_W, int(round(w * GLIFO_H / h))))
    r = cv2.resize(p.astype(np.float32), (nw, GLIFO_H), interpolation=cv2.INTER_AREA)
    out = np.zeros((GLIFO_H, GLIFO_W), np.float32)
    x0 = (GLIFO_W - nw) // 2
    out[:, x0:x0 + nw] = r
    return out


# ------------------------------------------------------------------ plantillas

def aprende(muestras: list) -> dict:
    """muestras: [(imagen_celda, texto)] -> {caracter: plantilla}.

    Se segmenta cada celda y se emparejan sus glifos, en orden, con el texto.
    Si el numero de glifos no cuadra con el del texto la muestra se descarta:
    un emparejamiento desplazado envenenaria todas las plantillas.
    """
    acumulado = {c: [] for c in ALFABETO}
    usadas = 0
    for img, texto in muestras:
        m = _mascara(img)
        gs = glifos(m)
        if len(gs) != len(texto):
            continue
        for (x0, x1), ch in zip(gs, texto):
            if ch in acumulado:
                acumulado[ch].append(normaliza(m[:, x0:x1]))
        usadas += 1
    plantillas = {c: np.median(np.stack(v), axis=0)
                  for c, v in acumulado.items() if v}
    plantillas["_muestras"] = np.array([usadas, len(muestras)])
    plantillas["_n"] = np.array([len(acumulado[c]) for c in ALFABETO])
    return plantillas


def guarda(plantillas: dict, ruta=TEMPLATES_FILE):
    np.savez_compressed(ruta, **plantillas)


def carga(ruta=TEMPLATES_FILE) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(
            f"no hay plantillas de digitos en {ruta}; "
            "generalas con scripts/learn_digits.py")
    z = np.load(ruta)
    return {k: z[k] for k in z.files}


# -------------------------------------------------------------------- lectura

def _lee_glifo(parche: np.ndarray, plantillas: dict) -> tuple:
    g = normaliza(parche)
    gn = g - g.mean()
    dg = float(np.sqrt((gn ** 2).sum())) or 1e-6
    mejor, score = None, -1.0
    for c in ALFABETO:
        t = plantillas.get(c)
        if t is None:
            continue
        tn = t - t.mean()
        dt = float(np.sqrt((tn ** 2).sum())) or 1e-6
        s = float((gn * tn).sum() / (dg * dt))
        if s > score:
            mejor, score = c, s
    return mejor, score


def lee_grupo(m: np.ndarray, plantillas: dict) -> tuple:
    """(texto, peor_score) de un grupo de texto. texto=None si algun glifo cae
    por debajo del umbral."""
    gs = glifos(m)
    if not gs:
        return None, 0.0
    chars, peor = [], 1.0
    for x0, x1 in gs:
        c, s = _lee_glifo(m[:, x0:x1], plantillas)
        if s < MIN_SCORE:
            return None, s
        chars.append(c)
        peor = min(peor, s)
    return "".join(chars), peor


def parsea_oro(texto: str):
    """'92(6457)' -> (92, 6457). None si no encaja con el patron."""
    if not texto or texto[-1] != ")" or "(" not in texto:
        return None
    izq, _, der = texto[:-1].partition("(")
    if not izq.isdigit() or not der.isdigit():
        return None
    actual, total = int(izq), int(der)
    if actual > MAX_ORO_ACTUAL or total > MAX_ORO_TOTAL or actual > total:
        return None
    return actual, total


def valida_contra_timeline(filas: list, timeline: dict,
                           tolerancia: int = 25) -> dict:
    """Contrasta lo leido con Match-V5. Dos controles, los dos automaticos.

    1. **Anclajes**: Match-V5 da el oro de los diez cada 60s. Exacto pero
       disperso. El margen `tolerancia` absorbe el desfase de la fraccion de
       segundo entre la captura y el frame (la renta pasiva son ~2 oro/s); un
       digito mal leido mueve el numero en cientos o miles, asi que no se
       cuela.
    2. **Monotonia del oro total**: el acumulado no baja nunca. Esto se puede
       comprobar en TODAS las muestras, no solo en las de ancla — es el control
       denso, y pilla justo lo que el disperso se salta.

    `filas`: [{'t', 'gold': [10 x (actual,total)|None], 'leidos'}]
    """
    frames = {round(f["timestamp"] / 1000): f for f in timeline["info"]["frames"]}
    errores, anclas, ilegibles = [], 0, 0
    for fila in filas:
        cerca = [ts for ts in frames if abs(ts - fila["t"]) <= 1.0]
        if not cerca:
            continue
        pf = frames[cerca[0]]["participantFrames"]
        for pid in range(1, 11):
            if str(pid) not in pf:
                continue
            anclas += 1
            leido = fila["gold"][pid - 1]
            if leido is None:
                ilegibles += 1
                continue
            errores.append(leido[1] - pf[str(pid)]["totalGold"])

    # CONTROL DENSO: horquilla entre anclajes. El oro total no baja nunca, asi
    # que una lectura en t tiene que caer entre el total del anclaje anterior y
    # el del siguiente. Se puede exigir en TODAS las muestras, no solo en las
    # 40 que caen junto a un anclaje, y —a diferencia del error contra el
    # anclaje mas cercano— no depende de cuanto se desvie la captura en el
    # tiempo: un digito mal leido mueve el numero en decenas o cientos y se
    # sale de la horquilla; medio segundo de desfase, no.
    ancladas = sorted(frames)
    fuera, comprobadas = 0, 0
    for fila in filas:
        # El limite inferior admite el desfase del HUD (LAG_HUD_S): la lectura
        # corresponde a algun instante de [t - lag, t]. El superior no lo
        # necesita — un HUD que va por detras nunca enseña oro del futuro.
        prev = [ts for ts in ancladas if ts <= fila["t"] - LAG_HUD_S]
        sig = [ts for ts in ancladas if ts >= fila["t"]]
        for pid in range(1, 11):
            g = fila["gold"][pid - 1]
            if g is None:
                continue
            pf_prev = frames[prev[-1]]["participantFrames"] if prev else None
            pf_sig = frames[sig[0]]["participantFrames"] if sig else None
            lo = pf_prev[str(pid)]["totalGold"] if pf_prev else 0
            hi = pf_sig[str(pid)]["totalGold"] if pf_sig else float("inf")
            comprobadas += 1
            if not (lo <= g[1] <= hi):
                fuera += 1

    bajadas = 0
    for pid in range(10):
        alto = None
        for fila in filas:
            g = fila["gold"][pid]
            if g is None:
                continue
            if alto is not None and g[1] < alto:
                bajadas += 1
            alto = max(alto or 0, g[1])

    celdas = sum(f["leidos"] for f in filas)
    total = len(filas) * 10
    inf = {"muestras": len(filas), "celdas_leidas": celdas,
           "celdas_totales": total,
           "frac_legible": round(celdas / total, 4) if total else 0.0,
           "anclas_mv5": anclas, "anclas_ilegibles": ilegibles,
           "bajadas_de_total": bajadas,
           "horquilla_comprobadas": comprobadas,
           "horquilla_fuera": fuera}
    if errores:
        e = np.abs(errores)
        inf["error_mediana"] = float(np.median(e))
        inf["error_p90"] = float(np.percentile(e, 90))
        inf["error_max"] = float(np.max(e))
        inf["frac_en_tolerancia"] = float(np.mean(e <= tolerancia))
    return inf


def lee_marcador(crop: np.ndarray, plantillas: dict) -> dict:
    """Oro de los diez a partir del recorte del marcador.

    Devuelve {'gold': [10 x (actual,total) o None], 'filas', 'leidos', 'score'}.
    El indice es participantId-1: las filas van de arriba abajo y la mitad
    izquierda es el equipo azul (verificado contra Match-V5 en
    scripts/check_ocr.py). Sin las cinco filas no se sabe a que jugador
    corresponde cada una, asi que la lectura entera se descarta.
    """
    m = _mascara(crop)
    mitad = crop.shape[1] // 2
    lados = {"blue": [], "red": []}
    peor = 1.0
    for x0, y0, x1, y1 in celdas(m):
        texto, s = lee_grupo(m[y0:y1, x0:x1], plantillas)
        oro = parsea_oro(texto) if texto else None
        if oro is None:
            continue
        lados["blue" if (x0 + x1) // 2 < mitad else "red"].append(
            ((y0 + y1) / 2, oro))
        peor = min(peor, s)

    salida = [None] * 10
    for lado, base in (("blue", 0), ("red", 5)):
        cel = sorted(lados[lado])
        if len(cel) == FILAS:
            for i, (_, oro) in enumerate(cel):
                salida[base + i] = oro
        elif cel:
            # Faltan celdas: sin las cinco no se sabe que fila es cual, salvo
            # que el otro equipo este completo y preste sus alturas de fila.
            otro = sorted(lados["red" if lado == "blue" else "blue"])
            if len(otro) == FILAS:
                ys = [y for y, _ in otro]
                for y, oro in cel:
                    i = int(np.argmin([abs(y - r) for r in ys]))
                    salida[base + i] = oro
    leidos = sum(x is not None for x in salida)
    return {"gold": salida, "celdas": len(lados["blue"]) + len(lados["red"]),
            "leidos": leidos, "score": round(peor, 3),
            "why": "ok" if leidos == 10 else f"solo {leidos}/10 celdas legibles"}
