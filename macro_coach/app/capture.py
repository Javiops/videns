"""Captura de las tres pasadas por partida (omni + niebla azul + niebla roja).

Cambios sobre fase 1, todos forzados por lo medido en la primera captura real
(ver docs/history.md):

- **La geometría del minimapa se calibra**, no se asume: ahora sale centrado y
  con otro tamaño. Cada frame se guarda YA RECORTADO al minimapa
  (`format: minimap-crop-v2`), así el procesado no necesita geometría.
- **La niebla NO se controla por la Replay API.** `render.fogOfWar` es un
  booleano que no da perspectiva de equipo y `selectionName` es de solo
  lectura (devuelve el invocador seleccionado). El control real es el
  desplegable "Niebla de guerra" del HUD de replay: Azul / Rojo / Todos.
  Se maneja con clics en coordenadas relativas al borde inferior izquierdo.
- **La validación de niebla mira campeones, no píxeles, y a mitad de partida**:
  en t=0 los diez están en sus fuentes y cualquier umbral de píxeles falla.
  Con niebla de un equipo, los suyos deben seguir visibles y los rivales no
  pueden aumentar.
"""
import ctypes
import json
import queue
import ssl
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import (REPLAY_API, REPLAY_SPEED, CAPTURE_MON,
                     PLAYERLIST_EVERY_S, SAMPLE_EVERY_GAME_S,
                     CAPTURE_LOOP_OVERHEAD_S)
from . import calibrate, detect

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# Desplegable de niebla del HUD de replay, en coordenadas relativas al borde
# inferior de la pantalla (medido en 1920x1080; ver docs/history.md).
# Todo relativo a la etiqueta "Niebla de guerra" localizada en pantalla
# (data/fog_label.png), medido en el HUD de referencia a 1920x1080.
FOG_LABEL_FILE = Path(__file__).resolve().parent.parent / "data" / "fog_label.png"
FOG_LABEL_MIN_SCORE = 0.55
# Cambio minimo en la zona de la lista para dar por abierto el desplegable.
FOG_MENU_MIN_DIFF = 6.0
FOG_DROPDOWN_OFF = (210, 27)
FOG_OPTION_OFF = {"blue": (213, -44), "red": (213, -22), "omni": (213, 0)}
MOUSE_PARK = (1700, 300)
# Si se descubre que el clic cae en la fila equivocada, se invierte
# el mapeo azul/rojo para el resto de la sesion.
FOG_CLICK_SWAP = {"on": False}
# Se corta la pasada antes del final: al terminar la replay el cliente cierra
# el juego y las pasadas siguientes se quedan sin Replay API.
END_MARGIN_S = 20.0
# Instante de validacion de niebla: pronto, para no encadenar un seek largo
# hacia atras (con el, el juego se cerraba antes de empezar a capturar).
VALIDATE_AT_S = 150.0
# Frames identicos seguidos que se toleran antes de repetir la pasada.
DUP_STREAK_ABORT = 12
# Verificacion periodica de que el minimapa sigue donde se calibro.
GEOM_CHECK_EVERY = 25
GEOM_MIN_SCORE = 0.35
# Diferencia de tamano de minimapa entre pasadas que se tolera: la calibracion
# jitterea +-1-2px; un cambio real de escala del HUD son cientos.
GEOM_SIZE_TOL = 8

# HUD en el que se midieron esas coordenadas. Se impone al preparar cada
# partida: ocultar la linea de tiempo o los marcos mueve el panel de replay.
HUD_STATE = {"interfaceAll": True, "interfaceMinimap": True,
             "interfaceReplay": True, "interfaceFrames": True,
             "interfaceTimeline": True, "interfaceScoreboard": True,
             "fogOfWar": False}

# Las tres pasadas y para que sirve cada una. El espectador NO limita la ficha
# de un jugador a lo que su equipo ve (confirmado por el owner): el marcador
# reparte la build de los diez con cualquier niebla. Asi que las pasadas de
# niebla NO son la fuente de las builds — son la fuente de la VISIBILIDAD, que
# es lo unico que no se puede sacar de ningun otro sitio. "Lo que veia el azul"
# se construye despues: build exacta (playerlist) filtrada por esa visibilidad.
ROLES = {"omni": "TRUTH-SEEKER", "blue": "REAL-BLUE", "red": "REAL-RED"}

# El marcador con la accion de oro puesta muestra `actual(total)` de los diez.
# Es la unica fuente de oro instantaneo: Match-V5 va de 60 en 60s y la Live
# Client API no expone oro en replay. Se recorta en la pasada TRUTH-SEEKER, en
# los mismos instantes que la playerlist, y lo lee `app/ocr.py`.
# x1 en 0.70 (1344px a 1920) y no 0.72: con el minimapa del espectador a escala
# 5 ocupa 508px pegado a la esquina y empieza en x=1389, a 7px del recorte
# antiguo. La barra del marcador acaba sobre x=1335, asi que recortando en 1344
# no se pierde nada y quedan ~45px de margen en vez de 7.
SCOREBOARD_FRAC = (0.28, 0.70, 0.76, 1.00)     # x0, x1, y0, y1 relativos


# Conexion HTTPS persistente: urllib abria un TLS nuevo por llamada y el bucle
# de captura consulta el reloj de la replay en CADA frame. Ese coste marcaba el
# suelo del bucle (~114ms) y con el la densidad maxima alcanzable a cada
# velocidad. Con keep-alive el suelo baja y 32x deja de adelgazar la muestra.
_CONN = {"c": None}


def _conn():
    import http.client
    if _CONN["c"] is None:
        host = REPLAY_API.split("://", 1)[1]
        _CONN["c"] = http.client.HTTPSConnection(host, timeout=6, context=CTX)
    return _CONN["c"]


def api_get(path):
    for intento in (1, 2):
        try:
            c = _conn()
            c.request("GET", path)
            return json.loads(c.getresponse().read())
        except Exception:
            try:
                _CONN["c"].close()
            except Exception:
                pass
            _CONN["c"] = None
            if intento == 2:
                raise
    return None


def api_post(path, body):
    req = urllib.request.Request(REPLAY_API + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=CTX, timeout=6) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def focus_game():
    """RuntimeError (no SystemExit) si el juego no esta: que se cierre la
    ventana es RECUPERABLE — el bucle de reintentos reabre la replay. Con
    SystemExit se paraba la sesion entera por una partida."""
    u32 = ctypes.windll.user32
    hwnd = u32.FindWindowW("RiotWindowClass", None)
    if not hwnd:
        raise RuntimeError("no encuentro la ventana del juego (se cerro?)")
    for _ in range(6):
        u32.keybd_event(0x12, 0, 0, 0)      # Alt down: permite robar foco
        u32.ShowWindow(hwnd, 9)             # SW_RESTORE
        u32.SetForegroundWindow(hwnd)
        u32.keybd_event(0x12, 0, 2, 0)      # Alt up
        time.sleep(1.0)
        if u32.GetForegroundWindow() == hwnd:
            time.sleep(1.0)
            return
    raise RuntimeError("no consigo poner el juego en primer plano")


def _click(x, y, settle=0.85):
    u32 = ctypes.windll.user32
    u32.SetCursorPos(int(x), int(y))
    time.sleep(0.3)
    u32.mouse_event(0x0002, 0, 0, 0, 0)     # LEFTDOWN
    time.sleep(0.06)
    u32.mouse_event(0x0004, 0, 0, 0, 0)     # LEFTUP
    time.sleep(settle)


def locate_fog_panel(screen_bgr) -> tuple:
    """(x, y) de la etiqueta "Niebla de guerra" en pantalla, o None.

    NUNCA se pulsa a ciegas: el panel de replay se mueve con el HUD y un clic
    en coordenadas fijas puede caer en cualquier control — una vez cerro el
    juego a mitad de sesion. Se localiza la etiqueta (que no cambia de texto,
    al contrario que el valor del desplegable) y todo lo demas es relativo.
    """
    tpl = cv2.imread(str(FOG_LABEL_FILE))
    if tpl is None:
        return None
    best = None
    for scale in (0.9, 0.95, 1.0, 1.05, 1.1):
        h, w = tpl.shape[:2]
        t = cv2.resize(tpl, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
        if t.shape[0] >= screen_bgr.shape[0] or t.shape[1] >= screen_bgr.shape[1]:
            continue
        res = cv2.matchTemplate(screen_bgr, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[0]:
            best = (float(mx), loc, scale)
    if best is None or best[0] < FOG_LABEL_MIN_SCORE:
        return None
    return (best[1][0], best[1][1], round(best[0], 3), best[2])



def _find_option_rows(region_bgr) -> dict:
    """{'blue': y, 'red': y, 'omni': y} dentro de la lista desplegada.

    Las filas se localizan por el COLOR DEL TEXTO ("Azul" en azul, "Rojo" en
    rojo), no por desplazamientos fijos: la lista se coloca respecto al valor
    seleccionado, asi que los desplazamientos solo valian con "Todos" puesto.
    Fiarse de ellos hizo que las dos pasadas de niebla de una partida salieran
    con la MISMA niebla, cosa que la validacion del momento no detecta.
    """
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    strong = (sat > 90) & (val > 110)
    blue = ((hue >= 95) & (hue <= 125)) & strong
    red = ((hue >= 170) | (hue <= 6)) & strong
    blue_rows = blue.sum(axis=1)
    red_rows = red.sum(axis=1)
    if blue_rows.max() < 8 or red_rows.max() < 8:
        return {}
    by = int(np.argmax(blue_rows))
    ry = int(np.argmax(red_rows))
    if abs(by - ry) < 6:
        return {}
    # "Todos" es la tercera fila, a la misma distancia que separa Azul de Rojo
    step = ry - by
    ty = ry + step
    if not (0 <= ty < region_bgr.shape[0]):
        ty = min(region_bgr.shape[0] - 3, ry + abs(step))
    return {"blue": by, "red": ry, "omni": ty}

def set_fog(mode: str, monitor: dict = None, sct=None) -> dict:
    """Selecciona Azul / Rojo / Todos localizando antes el panel."""
    import mss
    mon = monitor or CAPTURE_MON
    focus_game()
    time.sleep(0.4)
    own = sct is None
    ctx = mss.MSS() if own else None
    try:
        grabber = ctx if own else sct
        found = locate_fog_panel(grab(grabber, mon))
    finally:
        if own:
            ctx.close()
    if found is None:
        raise RuntimeError(
            "no localizo el panel 'Niebla de guerra' en pantalla: no pulso a "
            "ciegas (un clic perdido llego a cerrar el juego). Comprueba que "
            "el HUD de replay este visible.")
    lx, ly, score, scale = found
    # Con seleccion por color el objetivo es literal; el intercambio solo
    # tiene sentido en el camino de respaldo por desplazamientos.
    target = mode
    dx = lx + int(FOG_DROPDOWN_OFF[0] * scale)
    dy = ly + int(FOG_DROPDOWN_OFF[1] * scale)
    # Zona donde se despliega la lista (justo encima del desplegable).
    rx0, ry0 = lx + int(150 * scale), ly - int(60 * scale)
    rx1, ry1 = lx + int(300 * scale), ly + int(10 * scale)

    def _strip(grabber):
        return grab(grabber, mon)[max(ry0, 0):ry1, max(rx0, 0):rx1].astype(int)

    import mss as _mss
    opened = False
    with _mss.MSS() as g2:
        for intento in range(3):
            before = _strip(g2)
            _click(dx, dy)
            after = _strip(g2)
            if float(np.mean(np.abs(after - before))) > FOG_MENU_MIN_DIFF:
                opened = True
                break
            print(f"  el desplegable de niebla no se abrio (intento {intento + 1})",
                  flush=True)
            time.sleep(0.6)
            focus_game()
    if not opened:
        ctypes.windll.user32.SetCursorPos(*MOUSE_PARK)
        raise RuntimeError(
            "el desplegable de niebla no llega a abrirse; NO pulso la opcion "
            "(si la lista no esta abierta, el clic cae en el mundo y puede "
            "cerrar la replay o dejar la niebla anterior sin cambiar)")
    # Con la lista abierta, localizar las filas por color y pulsar la correcta.
    with _mss.MSS() as g3:
        shot = grab(g3, mon)
    rows = _find_option_rows(shot[max(ry0, 0):ry1, max(rx0, 0):rx1])
    how = "color"
    if rows and target in rows:
        cx = max(rx0, 0) + (rx1 - max(rx0, 0)) // 2
        cy = max(ry0, 0) + rows[target]
    else:
        how = "offsets"
        fallback = target
        if FOG_CLICK_SWAP["on"] and target in ("blue", "red"):
            fallback = "red" if target == "blue" else "blue"
        ox, oy = FOG_OPTION_OFF[fallback]
        cx, cy = lx + int(ox * scale), ly + int(oy * scale)
        print("  aviso: no distingo las filas por color; uso desplazamientos",
              flush=True)
    _click(cx, cy)
    ctypes.windll.user32.SetCursorPos(*MOUSE_PARK)
    time.sleep(1.0)
    return {"label_at": [lx, ly], "score": score, "scale": scale,
            "clicked": target, "how": how, "row_y": cy}


class _AsyncWriter:
    """Escribe los PNG en un hilo aparte.

    Con la escritura dentro del bucle, el disco lo bloqueaba a rachas y el
    juego avanzaba sin que capturaramos: medido, 1 de cada 3 ensayos a 16x/32x
    salia con ~97% de frames duplicados. Sin escritura sincrona, 0%.
    """

    def __init__(self, maxsize: int = 256):
        self.q = queue.Queue(maxsize=maxsize)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.dropped = 0
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            path, img = item
            try:
                cv2.imwrite(path, img)
            except Exception:
                pass
            self.q.task_done()

    def write(self, path, img):
        try:
            self.q.put_nowait((path, img))
        except queue.Full:
            self.dropped += 1     # mejor perder un frame que frenar la captura

    def close(self):
        self.q.put(None)
        self.t.join(timeout=60)


def _loop_sleep() -> float:
    """Espera del bucle para mantener la densidad de muestreo constante sea
    cual sea la velocidad de reproduccion."""
    return max(0.0, SAMPLE_EVERY_GAME_S / REPLAY_SPEED - CAPTURE_LOOP_OVERHEAD_S)


def grab(sct, monitor: dict = None):
    return np.array(sct.grab(monitor or CAPTURE_MON))[:, :, :3]


def scoreboard_rect(monitor: dict = None) -> tuple:
    """(x0, y0, x1, y1) del marcador, anclado abajo al centro de la pantalla."""
    mon = monitor or CAPTURE_MON
    fx0, fx1, fy0, fy1 = SCOREBOARD_FRAC
    return (mon["left"] + int(mon["width"] * fx0),
            mon["top"] + int(mon["height"] * fy0),
            mon["left"] + int(mon["width"] * fx1),
            mon["top"] + int(mon["height"] * fy1))


def replay_api_up() -> bool:
    try:
        api_get("/replay/playback")
        return True
    except Exception:
        return False


def ensure_replay_open(match_id: str, timeout_s: float = 300.0) -> bool:
    """True si hubo que reabrir la replay. El juego SE CIERRA al llegar la
    replay a su fin (medido), asi que entre pasadas puede no haber nada vivo."""
    if replay_api_up():
        return False
    from . import lcu
    print("  la replay no esta abierta; reabriendo...", flush=True)
    lcu.watch_replay(match_id)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if replay_api_up():
            time.sleep(10)          # deja terminar la carga
            return True
        time.sleep(3)
    raise RuntimeError(f"la replay {match_id} no abre (Replay API muda)")


def seen_champions(mm, teams: dict, templates: dict) -> set:
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    size = mm.shape[0]
    out = set()
    for team in ("ORDER", "CHAOS"):
        names = [c for c, t in teams.items() if t == team]
        for d in detect.detect_frame(mm, hsv, team, names, templates, size):
            out.add(d["champ"])
    return out


def validate_fog(mode: str, baseline: dict, mm, teams: dict,
                 templates: dict) -> dict:
    """Con niebla de un equipo: los suyos siguen visibles, los rivales no
    aumentan. baseline = conteos con 'Todos' en el MISMO instante."""
    seen = seen_champions(mm, teams, templates)
    own_team = {"blue": "ORDER", "red": "CHAOS"}.get(mode)
    n = {t: sum(1 for c in seen if teams[c] == t) for t in ("ORDER", "CHAOS")}
    verdict = {"mode": mode, "seen": n, "baseline": baseline, "ok": True, "why": ""}
    if mode == "omni":
        verdict["ok"] = (n["ORDER"] + n["CHAOS"]) >= 6
        verdict["why"] = ("suficientes campeones visibles" if verdict["ok"]
                          else f"solo {n['ORDER'] + n['CHAOS']}/10 detectados sin niebla")
        return verdict
    other = "CHAOS" if own_team == "ORDER" else "ORDER"
    if n[own_team] < baseline[own_team] - 1:
        verdict["ok"] = False
        verdict["why"] = (f"la niebla {mode} oculta a los SUYOS "
                          f"({own_team} {baseline[own_team]} -> {n[own_team]}): "
                          "seguramente se selecciono el equipo contrario")
    elif n[other] > baseline[other] + 1:
        # +1 de tolerancia: el detector fluctua un campeon entre capturas y
        # eso NO es un fallo de niebla. Tratarlo como fallo disparaba una
        # "correccion" que cambiaba de equipo y grababa la misma niebla dos
        # veces. La comprobacion de verdad es scripts/audit_fog.py.
        verdict["ok"] = False
        verdict["why"] = (f"con niebla {mode} se ven MAS rivales que sin ella "
                          f"({other} {baseline[other]} -> {n[other]})")
    else:
        verdict["why"] = (f"propios {own_team} {n[own_team]}/{baseline[own_team]}, "
                          f"rivales {other} {n[other]}/{baseline[other]}")
    return verdict



def _middle_click():
    """Boton central = alternar oro/items en el marcador.

    El owner reasigno esa accion a mouse3 porque el cliente IGNORA el teclado
    inyectado (probados keybd_event, SendInput por VK y por scancode,
    PostMessage, SendMessage y AttachThreadInput: cero efecto) mientras que el
    RATON inyectado si llega. Sin ese remapeo no hay oro desatendido.
    """
    u32 = ctypes.windll.user32
    u32.SetCursorPos(*MOUSE_PARK)
    time.sleep(0.25)
    u32.mouse_event(0x0020, 0, 0, 0, 0)     # MIDDLEDOWN
    time.sleep(0.07)
    u32.mouse_event(0x0040, 0, 0, 0, 0)     # MIDDLEUP
    time.sleep(1.2)


def _scoreboard_colorfulness(sct, mon=None) -> float:
    """Fraccion de pixeles vivos del marcador.

    La vista de ITEMS esta llena de iconos de colores; la de ORO son cifras
    claras sobre fondo oscuro. Medido a los 15 min: 8.75% con items, 4.41%
    con oro.
    """
    x0, y0, x1, y1 = scoreboard_rect(mon)
    crop = grab(sct, mon)[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 1] > 110) & (hsv[..., 2] > 90)))


_OCR_PL = {"p": None, "faltan": False}


def _celdas_de_oro(sct, mon=None):
    """Cuantas de las diez celdas de oro se leen AHORA en el marcador.

    None si aun no hay plantillas de digitos (entonces no se puede verificar
    leyendo y se cae al criterio de color, dejando constancia en la meta).
    """
    if _OCR_PL["faltan"]:
        return None
    if _OCR_PL["p"] is None:
        try:
            from . import ocr
            _OCR_PL["p"] = (ocr, ocr.carga())
        except FileNotFoundError:
            _OCR_PL["faltan"] = True
            print("  AVISO: sin plantillas de digitos (scripts/learn_digits.py); "
                  "la vista de oro se verificara solo por color", flush=True)
            return None
    ocr, plantillas = _OCR_PL["p"]
    x0, y0, x1, y1 = scoreboard_rect(mon)
    return ocr.lee_marcador(grab(sct, mon)[y0:y1, x0:x1], plantillas)["leidos"]


def ensure_gold_view(sct, mon=None) -> dict:
    """Deja el marcador mostrando ORO y lo comprueba LEYENDOLO.

    La comprobacion anterior era indirecta —la vista de oro es menos colorida
    que la de items— y por eso las 40 primeras partidas se capturaron enteras
    con la vista equivocada sin que saltara nada. Leer las diez celdas no deja
    lugar a duda: si no se leen, no hay oro en pantalla.

    Mientras no haya plantillas se conserva el criterio de color, pero la meta
    de la pasada dice con cual se verifico.
    """
    n = _celdas_de_oro(sct, mon)
    if n is not None:
        pasos = [n]
        for _ in range(2):
            if n == 10:
                return {"gold_view": True, "verified_by": "ocr", "celdas": pasos,
                        "toggled": len(pasos) - 1}
            _middle_click()
            n = _celdas_de_oro(sct, mon)
            pasos.append(n)
        return {"gold_view": False, "verified_by": "ocr", "celdas": pasos,
                "toggled": len(pasos) - 1}

    a = _scoreboard_colorfulness(sct, mon)
    _middle_click()
    b = _scoreboard_colorfulness(sct, mon)
    if b < a:
        return {"gold_view": True, "verified_by": "color",
                "colorfulness": [round(a, 4), round(b, 4)], "toggled": 1}
    # ya estabamos en oro: se vuelve
    _middle_click()
    c = _scoreboard_colorfulness(sct, mon)
    ok = c <= a + 1e-6
    return {"gold_view": bool(ok), "verified_by": "color",
            "colorfulness": [round(a, 4), round(b, 4), round(c, 4)],
            "toggled": 2}

def _roster(pl: list) -> list:
    """Lo que NO cambia durante la partida: campeon, equipo, runas, hechizos.

    Las runas las pide el owner para el estado de decision y solo estan aqui:
    Match-V5 da las perks de la partida entera, la `playerlist` las da tal cual
    se juegan. Como son constantes, van una vez en la meta de la pasada en vez
    de repetirse en cada muestra.
    """
    out = []
    for p in pl:
        r = p.get("runes") or {}
        s = p.get("summonerSpells") or {}
        out.append({
            "champ": p.get("championName"),
            "team": p.get("team"),
            "summoner": p.get("summonerName"),
            "skin": p.get("skinID"),
            "runas": {k: {"id": (r.get(k) or {}).get("id"),
                          "nombre": (r.get(k) or {}).get("displayName")}
                      for k in ("keystone", "primaryRuneTree", "secondaryRuneTree")
                      if r.get(k)},
            "hechizos": [(s.get(k) or {}).get("displayName")
                         for k in ("summonerSpellOne", "summonerSpellTwo")],
        })
    return out


def prepare(match: dict, sct) -> dict:
    """Calibra y deja el HUD listo. Devuelve {'geom', 'templates', 'teams'}."""
    from .segments import participants
    P = participants(match)
    teams = {v["champ"]: ("ORDER" if v["team"] == 100 else "CHAOS")
             for v in P.values()}
    patch = ".".join(match["info"]["gameVersion"].split(".")[:2])
    # Estado de HUD EXACTO en el que se midieron las coordenadas del
    # desplegable de niebla: el panel esta anclado abajo a la izquierda y se
    # desplaza si se ocultan la linea de tiempo o los marcos de jugador. Se
    # fija siempre, por si quedo tocado de una sesion anterior.
    api_post("/replay/render", HUD_STATE)
    focus_game()
    time.sleep(1.0)
    # La calibracion correlaciona contra el mapa SIN niebla: con la niebla de
    # un equipo puesta, media mapa esta oscuro y la correlacion se hunde
    # (0.14 medido). Se pone "Todos" antes de calibrar. Importa porque la
    # niebla persiste entre pasadas y entre partidas.
    # Alejar la replay del final ANTES de calibrar: al terminar, el cliente
    # superpone el cartel de VICTORIA/DERROTA justo encima del minimapa y la
    # correlacion se hunde a 0.20 (visto en la partida 11). Tambien evita
    # calibrar sobre la pantalla de carga.
    try:
        length = api_get("/replay/playback")["length"]
        api_post("/replay/playback",
                 {"time": max(60.0, length * 0.3), "paused": True})
        time.sleep(5)
    except Exception as e:
        print(f"  aviso: no pude recolocar la replay antes de calibrar: {e}",
              flush=True)
    focus_game()
    # La Replay API responde antes de que el HUD termine de dibujarse: si se
    # calibra demasiado pronto se correlaciona contra una pantalla de carga
    # (0.27 medido). Se reintenta dejando tiempo al render, REAPLICANDO la
    # niebla "Todos" en cada intento (si el clic no entro, el mapa sigue a
    # oscuras y reintentar sobre la misma imagen no arregla nada).
    geom, last = None, None
    for intento in range(5):
        try:
            set_fog("omni")
            geom = calibrate.find_minimap(grab(sct))
            break
        except RuntimeError as e:
            last = e
            print(f"  calibracion aun no (intento {intento + 1}): {e}", flush=True)
            time.sleep(6)
            focus_game()
    if geom is None:
        raise RuntimeError(f"calibracion fallida tras 5 intentos: {last}")
    templates = detect.build_templates(list(teams), patch, geom["size"])
    # El marcador debe mostrar ORO: es la unica fuente de la cartera (la Live
    # Client API no expone oro en replay). El toggle es de sesion, asi que hay
    # que ponerlo cada vez que se abre la replay.
    gold = ensure_gold_view(sct)
    detalle = (f"celdas leidas {gold['celdas']}" if "celdas" in gold
               else f"colorido {gold.get('colorfulness')}")
    print(f"  marcador en vista de oro: {gold['gold_view']} "
          f"(por {gold.get('verified_by')}, {detalle})", flush=True)
    return {"geom": geom, "templates": templates, "teams": teams,
            "participants": P, "patch": patch, "gold_view": gold}


def capture_pass(out_dir: Path, mode: str, ctx: dict, timeline: dict,
                 t_start: float = 0.0, validate_at: float = None) -> dict:
    """Una pasada completa a 8x. ctx viene de prepare(). Guarda cada frame ya
    recortado al minimapa."""
    import mss
    out_dir.mkdir(parents=True, exist_ok=True)
    geom, teams, templates = ctx["geom"], ctx["teams"], ctx["templates"]
    # La pasada TRUTH-SEEKER existe para traer el ORO. Sin vista de oro sus
    # recortes del marcador son de items y no hay forma de saberlo despues sin
    # mirarlos: asi se capturaron las 40 primeras partidas. Antes se avisaba y
    # se capturaba igual; ahora se niega.
    gv = ctx.get("gold_view") or {}
    if mode == "omni" and not gv.get("gold_view"):
        raise RuntimeError(
            f"el marcador NO esta en vista de oro ({gv}); no capturo una pasada "
            "TRUTH-SEEKER sin oro. Es el toggle que el owner remapeo a mouse3: "
            "comprueba que el bind sigue puesto (evtToggleScoreboardGoldDisplay "
            "en Config/input.ini) y que el juego lo esta cogiendo.")
    # Las tres pasadas de una partida DEBEN compartir tamano de minimapa (los
    # iconos se aprenden en omni y se buscan en niebla). Si se recaptura una
    # pasada suelta y el HUD reseteo el minimapa, saldria de otro tamano y la
    # visibilidad se inflaria sin fallar. Se comprueba contra las pasadas
    # hermanas ya guardadas y se aborta la partida (no la noche) si no cuadra.
    for hermana in ("pass_blue", "pass_red", "pass_omni"):
        hdir = out_dir.parent / hermana
        if hdir == out_dir or not (hdir / "meta.json").exists():
            continue
        hsize = json.loads((hdir / "meta.json").read_text(encoding="utf-8")).get("size")
        # Tolerancia: la calibracion por correlacion multiescala varia +-1-2px
        # entre capturas del MISMO tamano (medido: el corpus tiene 508 y 507
        # mezclados). Eso es ruido inofensivo. Lo que hay que cazar es un cambio
        # de ESCALA del HUD (508 vs 255 = 2x), que son cientos de px.
        if hsize and abs(hsize - geom["size"]) > GEOM_SIZE_TOL:
            api_post("/replay/playback", {"speed": 1, "paused": True})
            raise RuntimeError(
                f"minimapa de {geom['size']}px pero {hermana} es de {hsize}px: "
                "el HUD cambio de tamano. Ajusta con scripts/set_minimap.py y "
                "recaptura las tres pasadas de esta partida.")
    length = api_get("/replay/playback")["length"]
    # Se valida PRONTO (t=150s): los diez ya estan repartidos por el mapa y el
    # salto hacia atras hasta t_start es corto. Validar a mitad de partida
    # obligaba a un seek largo hacia atras y ahi se cerraba el juego.
    t_val = validate_at if validate_at is not None else min(VALIDATE_AT_S, length * 0.5)

    with mss.MSS() as sct:
        # --- validación en un instante con los campeones repartidos ---
        api_post("/replay/playback", {"time": t_val, "paused": True})
        time.sleep(6)
        set_fog("omni")
        base_seen = seen_champions(calibrate.crop(grab(sct), geom), teams, templates)
        baseline = {t: sum(1 for c in base_seen if teams[c] == t)
                    for t in ("ORDER", "CHAOS")}
        set_fog(mode)
        verdict = validate_fog(mode, baseline,
                               calibrate.crop(grab(sct), geom), teams, templates)
        print(f"  validacion niebla [{mode}]: {verdict['why']}", flush=True)
        if not verdict["ok"]:
            set_fog("omni")
            # RuntimeError, no SystemExit: una replay mala (las hay: una
            # cerraba el cliente una y otra vez) debe costar UNA partida, no
            # la noche entera. El bucle de capture_queue reintenta y sigue.
            raise RuntimeError(
                f"ABORT niebla ({mode}): {verdict['why']}.\n"
                "El control es el desplegable 'Niebla de guerra' del HUD de la "
                "replay (abajo a la izquierda): Azul / Rojo / Todos. Revisa que "
                "el HUD este visible y las coordenadas de FOG_* en app/capture.py.")

        # --- pasada ---
        api_post("/replay/playback", {"time": t_start, "paused": True})
        time.sleep(5)
        focus_game()
        api_post("/replay/playback", {"speed": REPLAY_SPEED, "paused": False})
        rows, last_pl, n, bad_streak = [], -99.0, 0, 0
        roster = None
        geom_bad = 0
        prev_mm, n_dup, dup_streak = None, 0, 0
        writer = _AsyncWriter()
        t = t_start
        # El juego se CIERRA al terminar la replay, y con el se va la Replay
        # API para las pasadas siguientes: se para antes del final y se pausa.
        stop_at = max(length - END_MARGIN_S, 60.0)
        while True:
            pb = api_get("/replay/playback")
            t = pb["time"]
            if t >= stop_at:
                api_post("/replay/playback", {"speed": 1, "paused": True})
                break
            if pb["paused"] and t > 5:
                break
            # UNA sola captura de pantalla por vuelta: de ella salen el
            # minimapa y, cuando toca, el marcador. A 32x el presupuesto de la
            # vuelta son ~75ms de reloj real y cada grab cuesta ~10-15ms;
            # capturar dos veces adelgazaria la muestra justo por encima de los
            # 3.5s de separacion donde la sensibilidad de visibilidad se hunde.
            pantalla = grab(sct)
            mm = calibrate.crop(pantalla, geom)
            if mm.shape[0] != geom["size"] or mm.shape[1] != geom["size"]:
                bad_streak += 1
                if bad_streak >= 15:
                    api_post("/replay/playback", {"speed": 1, "paused": True})
                    raise RuntimeError("recorte invalido repetido; se repite la pasada")
                focus_game()
                continue
            hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
            if detect.team_pixels(hsv, "ORDER") + detect.team_pixels(hsv, "CHAOS") < 150:
                bad_streak += 1
                print(f"AVISO: frame sin iconos en t={t:.0f} (foco perdido?)", flush=True)
                if bad_streak >= 15:
                    api_post("/replay/playback", {"speed": 1, "paused": True})
                    raise RuntimeError("demasiados frames invalidos seguidos "
                                       "(foco perdido?); se repite la pasada")
                focus_game()
                continue
            bad_streak = 0
            # Frame IDENTICO al anterior = el juego avanzo pero la pantalla no
            # se redibujo. Guardarlo seria inventar una muestra donde no hay
            # informacion nueva: se cuenta y se descarta.
            if prev_mm is not None and mm.shape == prev_mm.shape                     and np.array_equal(mm, prev_mm):
                n_dup += 1
                dup_streak += 1
                if dup_streak >= DUP_STREAK_ABORT:
                    api_post("/replay/playback", {"speed": 1, "paused": True})
                    raise RuntimeError(
                        f"{dup_streak} frames seguidos identicos: la pantalla "
                        "no se refresca (velocidad demasiado alta o maquina "
                        "cargada); se repite la pasada")
                time.sleep(_loop_sleep())
                continue
            dup_streak = 0
            prev_mm = mm
            # El minimapa puede moverse a mitad de pasada (el owner ajustando
            # el HUD): a partir de ahí se recortaría de otro sitio sin fallar.
            # Chequeo barato cada GEOM_CHECK_EVERY frames.
            # Solo en la pasada omni: con niebla, medio mapa esta oscuro y la
            # correlacion contra el fondo limpio cae por debajo del umbral
            # aunque el recorte sea correcto (falso positivo que abortaba
            # todas las pasadas de niebla).
            if mode == "omni" and n % GEOM_CHECK_EVERY == 0 and n > 0:
                score = calibrate.still_aligned(mm)
                if score < GEOM_MIN_SCORE:
                    geom_bad += 1
                    print(f"AVISO: el minimapa ya no cuadra (corr {score:.2f}) "
                          f"en t={t:.0f}", flush=True)
                    if geom_bad >= 2:
                        api_post("/replay/playback", {"speed": 1, "paused": True})
                        raise RuntimeError(
                            "la geometria del minimapa cambio a mitad de pasada; "
                            "se repite la pasada entera con la nueva calibracion")
                else:
                    geom_bad = 0
            writer.write(str(out_dir / f"mm_{int(t * 10):06d}.png"), mm)
            row = {"t": round(t, 2)}
            if t - last_pl >= PLAYERLIST_EVERY_S:
                last_pl = t
                # El recorte del marcador se toma EN EL MISMO instante que el
                # estado de los diez: el oro sale de la imagen y los items de
                # la API, y casarlos con segundos de diferencia inventaria
                # estados que no existieron. El nombre del recorte va en la
                # fila para que el OCR no tenga que adivinar a que muestra
                # corresponde.
                if mode == "omni":
                    sx0, sy0, sx1, sy1 = scoreboard_rect()
                    nombre = f"sb_{int(t * 10):06d}.png"
                    writer.write(str(out_dir / nombre),
                                 pantalla[sy0:sy1, sx0:sx1])
                    row["sb"] = nombre
                try:
                    pl = api_get("/liveclientdata/playerlist")
                    row["players"] = [
                        {"champ": p["championName"], "cs": p["scores"]["creepScore"],
                         "level": p["level"], "dead": p["isDead"],
                         "respawn": round(p["respawnTimer"], 1),
                         "k": p["scores"]["kills"], "d": p["scores"]["deaths"],
                         "a": p["scores"]["assists"],
                         "ward": p["scores"].get("wardScore"),
                         "items": sorted(i["itemID"] for i in p.get("items", []))}
                        for p in pl]
                    if roster is None:
                        roster = _roster(pl)
                except Exception as e:
                    print(f"AVISO: playerlist fallo en t={t:.0f}: {e}", flush=True)
            rows.append(row)
            n += 1
            if n % 100 == 0:
                print(f"  [{mode}] t={t:.0f}s muestras={n}", flush=True)
            time.sleep(_loop_sleep())

    writer.close()
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    api_post("/replay/playback", {"speed": 1, "paused": True})
    meta = {"mode": mode, "format": "minimap-crop-v2", "fog_validation": verdict,
            "frames": n, "t_end": round(t, 1), "geom": geom,
            "size": geom["size"], "speed": REPLAY_SPEED,
            "dup_frames_skipped": n_dup,
            "dup_frac": round(n_dup / max(n + n_dup, 1), 3),
            "writer_dropped": writer.dropped}
    meta["role"] = ROLES.get(mode, mode)
    if roster:
        meta["roster"] = roster
    if mode == "omni":
        meta["scoreboard"] = {"rect": scoreboard_rect(),
                              "every_s": PLAYERLIST_EVERY_S,
                              "gold_view": ctx.get("gold_view"),
                              "note": "oro actual(total) de los 10, en el mismo "
                                      "instante que la playerlist; lo lee app/ocr.py"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"FIN [{mode}]: {n} muestras hasta t={t:.0f}s", flush=True)
    return meta
