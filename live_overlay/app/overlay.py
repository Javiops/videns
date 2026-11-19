"""Ventanas del overlay: dos widgets translúcidos, sin marco, siempre encima.

Una ventana para objetivos (Barón, dragón, buffs) y otra para el oro de
objetos, ancladas arriba a la derecha del cliente de League. Cada una es una
página del mismo servidor local (`/?panel=objectives|gold&overlay=1`).

Translucidez: pywebview no sabe hacer una ventana con alfa por píxel sobre el
escritorio (su modo `transparent` deja el fondo del formulario pintado, y
WebView2 ignora el color clave de las ventanas por capas), así que se usa una
ventana por capas de Win32 con alfa uniforme `ALPHA_TRANSLUCENT`: la ventana
es la propia tarjeta, sin márgenes, con el alto ajustado al contenido y las
esquinas redondeadas por DWM. Mientras se mantiene pulsada la tecla del
marcador (Tab, `SCOREBOARD_VK`) el alfa pasa a 255: opacas del todo. La tecla
solo se lee (GetAsyncKeyState); nunca se inyecta entrada.

Cada ventana ajusta su alto al contenido (la página llama a `api.fit`), no
roba el foco al juego al crearse y se puede arrastrar. La X de cualquiera
cierra las dos. Arranca el servidor local si no está ya en marcha y lo cierra
al salir si lo arrancó él. Una sola instancia por sesión (mutex) y todo
error va a data/overlay.log, porque pythonw no tiene consola.

League tiene que estar en modo Sin bordes (o Ventana): sobre pantalla
completa exclusiva no se puede dibujar; le pasa a cualquier overlay.

Uso: .venv\\Scripts\\pythonw.exe -m app.overlay   (o start_overlay.bat)
"""

from __future__ import annotations

import ctypes
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback

import httpx

from app.config import DATA_DIR, HOST, PORT, ROOT, SCOREBOARD_VK

BASE = f"http://{HOST}:{PORT}"
MARGIN, GAP = 16, 8
PANELS = {
    # ancho fijo; el alto inicial es un mínimo, la página lo ajusta al contenido
    "objectives": {"width": 250, "height": 40, "max_height": 260},
    "gold": {"width": 388, "height": 40, "max_height": 320},   # BLUE | diferencias | RED, cinco filas
}
BACKGROUND = "#0b0e13"                # la ventana ES la tarjeta: mismo color que el panel
ALPHA_TRANSLUCENT = 150               # ~59 %: se ve el juego a través de las tarjetas
ALPHA_OPAQUE = 255
DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND = 33, 2   # esquinas redondeadas (Windows 11)
LOG_PATH = DATA_DIR / "overlay.log"
LAYOUT_PATH = DATA_DIR / "layout.json"    # dónde dejó el jugador cada ventana
MUTEX_NAME = "Local\\videns_live_overlay"
ERROR_ALREADY_EXISTS = 183
KEY_POLL_S = 0.03
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
LWA_COLORKEY, LWA_ALPHA = 0x1, 0x2

# pythonw no tiene consola: sys.stdout/stderr son None y cualquier print o
# formatter de logging que los toque revienta sin dejar rastro.
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

log = logging.getLogger("overlay")
_mutex_handle = None


def setup_logging() -> None:
    if log.handlers:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def already_running() -> bool:
    """Una sola instancia por sesión: el .bat pulsado dos veces no abre cuatro ventanas."""
    global _mutex_handle
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        return False
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        return True
    _mutex_handle = handle   # vive lo que el proceso
    return False


def message_box(text: str, title: str = "Live Overlay") -> None:
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)


def load_layout(screen_w: int, screen_h: int) -> dict[str, dict]:
    """Posiciones guardadas por ventana; se ignora lo que caiga fuera de la pantalla actual."""
    try:
        raw = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    layout: dict[str, dict] = {}
    for panel, spec in PANELS.items():
        pos = raw.get(panel) if isinstance(raw, dict) else None
        try:
            x, y = int(pos["x"]), int(pos["y"])
        except (TypeError, KeyError, ValueError):
            continue
        if -spec["width"] // 2 <= x <= screen_w - spec["width"] // 2 and 0 <= y <= screen_h - 20:
            layout[panel] = {"x": x, "y": y}
    return layout


def save_layout(positions: dict[str, dict]) -> None:
    if not positions:
        return
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        LAYOUT_PATH.write_text(json.dumps(positions, indent=1), encoding="utf-8")
    except OSError as exc:
        log.warning("no se pudo guardar layout.json: %s", exc)


def server_up() -> bool:
    try:
        return httpx.get(f"{BASE}/api/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_server() -> subprocess.Popen | None:
    if server_up():
        return None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stderr = open(DATA_DIR / "server.err", "ab")   # el motivo de un arranque fallido queda en disco
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "uvicorn", "app.main:app",
         "--host", HOST, "--port", str(PORT), "--log-level", "warning"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    for _ in range(60):
        if proc.poll() is not None:
            raise SystemExit(f"El servidor del overlay murió al arrancar (código {proc.returncode}); ver data/server.err.")
        if server_up():
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit(f"No se pudo arrancar el servidor del overlay en el puerto {PORT}.")


# --- ventana por capas (Win32) -------------------------------------------------

def _user32():
    user32 = ctypes.windll.user32
    user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
    user32.SetLayeredWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_uint]
    user32.SetLayeredWindowAttributes.restype = ctypes.c_bool
    return user32


def _top_level_windows_of_this_process() -> dict[str, int]:
    """título -> HWND de las ventanas visibles de este proceso (respaldo cuando pywebview no expone el nativo)."""
    found: dict[str, int] = {}
    if sys.platform != "win32":
        return found
    user32 = ctypes.windll.user32
    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    me = os.getpid()

    def callback(hwnd, _lparam):
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
        if pid.value == me and user32.IsWindowVisible(ctypes.c_void_p(hwnd)):
            buffer = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(ctypes.c_void_p(hwnd), buffer, 256)
            if buffer.value:
                found[buffer.value] = hwnd
        return True

    user32.EnumWindows(proc(callback), 0)
    return found


def window_hwnd(window) -> int | None:
    """HWND del formulario de pywebview; None si todavía no existe."""
    native = getattr(window, "native", None)
    handle = getattr(native, "Handle", None)
    if handle is not None:
        try:
            return int(handle.ToInt64())
        except Exception:
            try:
                return int(handle)
            except Exception:
                pass
    return _top_level_windows_of_this_process().get(getattr(window, "title", "") or "")


def set_translucency(hwnd: int, alpha: int) -> bool:
    """Alfa uniforme para toda la ventana (WebView2 no respeta un color clave, sí el alfa) y esquinas redondas."""
    if sys.platform != "win32" or not hwnd:
        return False
    user32 = _user32()
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if not style & WS_EX_LAYERED:
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)
    try:
        preference = ctypes.c_int(DWMWCP_ROUND)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd), DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(preference), ctypes.sizeof(preference)
        )
    except Exception:
        pass   # Windows 10: esquinas cuadradas
    return bool(user32.SetLayeredWindowAttributes(hwnd, 0, alpha, LWA_ALPHA))


class Api:
    """Lo que la página puede pedir a la ventana (window.pywebview.api)."""

    def __init__(self) -> None:
        self.windows: dict[str, object] = {}
        self.alpha = ALPHA_TRANSLUCENT

    def layered(self, panel: str) -> bool:
        """La página pregunta al estar lista: ¿es ya una ventana por capas? Solo entonces pinta el color clave."""
        window = self.windows.get(panel)
        hwnd = window_hwnd(window) if window is not None else None
        ok = bool(hwnd) and set_translucency(hwnd, self.alpha)
        log.info("layered(%s) -> %s (hwnd %s)", panel, ok, hwnd)
        return ok

    def close(self) -> None:
        for window in list(self.windows.values()):
            try:
                window.destroy()
            except Exception:
                pass

    def fit(self, panel: str, height) -> None:
        """Ajusta el alto de la ventana al contenido de la página (ancho fijo)."""
        window = self.windows.get(panel)
        spec = PANELS.get(panel)
        if window is None or spec is None:
            return
        try:
            wanted = int(float(height))
        except (TypeError, ValueError):
            return
        wanted = max(spec["height"], min(wanted, spec["max_height"]))
        try:
            window.resize(spec["width"], wanted)
            log.info("fit %s -> %s px", panel, wanted)
        except Exception as exc:   # pywebview relanza errores nativos variados
            log.warning("fit(%s, %s) falló: %s", panel, wanted, exc)


def apply_all(api: "Api", alpha: int) -> None:
    api.alpha = alpha
    for panel, window in list(api.windows.items()):
        hwnd = window_hwnd(window)
        ok = bool(hwnd) and set_translucency(hwnd, alpha)
        if not ok:
            log.warning("translucidez no aplicada a %s (hwnd %s)", panel, hwnd)
        try:
            window.evaluate_js(f"window.setOpaque && window.setOpaque({'true' if alpha >= ALPHA_OPAQUE else 'false'})")
        except Exception:
            pass


def watch_scoreboard_key(api: "Api", stop: threading.Event) -> None:
    """Translúcido en reposo; opaco mientras Tab esté pulsada. Solo lectura del estado de la tecla."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    for delay in (1.0, 2.0, 3.0):        # las ventanas aparecen y navegan en los primeros segundos
        time.sleep(delay)
        apply_all(api, ALPHA_TRANSLUCENT)
    down = False
    while not stop.is_set():
        pressed = bool(user32.GetAsyncKeyState(SCOREBOARD_VK) & 0x8000)
        if pressed != down:
            down = pressed
            apply_all(api, ALPHA_OPAQUE if down else ALPHA_TRANSLUCENT)
        time.sleep(KEY_POLL_S)


def main() -> None:
    setup_logging()
    if already_running():
        log.info("ya hay un overlay abierto; no se abre otro")
        return
    log.info("arrancando (python %s, pid %s)", sys.version.split()[0], os.getpid())
    owns_server = ensure_server()
    log.info("servidor %s (%s)", BASE, "propio" if owns_server else "ya estaba en marcha")
    try:
        import webview
    except ImportError as exc:
        log.error("pywebview no se puede importar: %s", exc)
        if owns_server is not None:
            owns_server.terminate()
        message_box("pywebview no está instalado o no carga; reinstala las dependencias:\n"
                    ".venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        return

    api = Api()
    screen_w = ctypes.windll.user32.GetSystemMetrics(0) if sys.platform == "win32" else 1920
    screen_h = ctypes.windll.user32.GetSystemMetrics(1) if sys.platform == "win32" else 1080
    layout = load_layout(screen_w, screen_h)
    positions: dict[str, dict] = dict(layout)
    y = MARGIN
    for panel, spec in PANELS.items():
        saved = layout.get(panel) or {}
        window = webview.create_window(
            f"Live Overlay · {panel}",
            f"{BASE}/?panel={panel}&overlay=1",
            width=spec["width"],
            height=spec["height"],
            x=int(saved.get("x", screen_w - spec["width"] - MARGIN)),
            y=int(saved.get("y", y)),
            min_size=(spec["width"], spec["height"]),   # pywebview impone 200x100 si no se le dice
            frameless=True,
            easy_drag=True,
            on_top=True,
            focus=False,
            background_color=BACKGROUND,
            js_api=api,
        )
        window.events.shown += (lambda w=window: set_translucency(window_hwnd(w) or 0, ALPHA_TRANSLUCENT))
        try:
            window.events.moved += (lambda x, y, p=panel: positions.__setitem__(p, {"x": int(x), "y": int(y)}))
        except Exception:
            pass   # sin evento moved no se recuerdan posiciones; nada más
        api.windows[panel] = window
        y += spec["max_height"] + GAP
    stop = threading.Event()

    def after_start() -> None:
        watch_scoreboard_key(api, stop)

    try:
        webview.start(after_start)
        # Cerrar las ventanas no para el servidor: la página /faces (marcas de splash) sigue
        # funcionando sin el overlay abierto, y el próximo arranque lo reutiliza. stop_overlay.bat lo para.
        log.info("ventanas cerradas; el servidor %s sigue en marcha (stop_overlay.bat lo para)", BASE)
    except BaseException:
        if owns_server is not None:
            owns_server.terminate()
        raise
    finally:
        stop.set()
        save_layout(positions)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        setup_logging()
        log.error("el lanzador murió:\n%s", traceback.format_exc())
        raise
