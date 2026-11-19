"""Graba una partida entera para la landing: la selección de campeones y la partida, con capturas de pantalla.

Mientras corre, en data/captures/<inicio>/:
  lcu/      la sesión de selección del cliente (/lol-champ-select/v1/session) cada segundo, si cambia,
            y la del gameflow (/lol-gameflow/v1/session) en cada cambio de fase
  live/     el Live Client API (allgamedata) cada 2 s durante la partida, como scripts/record.py
  shots/    capturas de toda la pantalla en JPEG: cada 2 s en la selección y cada 5 s en partida
  marks.jsonl  cada pulsación de F9: momento, fase y tiempo de juego; F9 dispara además 5 capturas seguidas

Solo lee: el API local del cliente (credenciales del lockfile de League) y el Live Client API, los dos
sancionados por Riot; de teclado solo el estado de F9 (GetAsyncKeyState), sin inyectar nada. Para solo:
Ctrl-C, o 90 s después de que la partida termine (fase EndOfGame / sin partida).

Uso: .venv\\Scripts\\python.exe scripts\\capture.py [--lockfile "C:\\Riot Games\\League of Legends\\lockfile"]
     .venv\\Scripts\\python.exe scripts\\capture.py --selftest     (una captura y el estado de los APIs, y sale)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import mss
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.live_client import fetch_snapshot

LOCKFILE = Path(r"C:\Riot Games\League of Legends\lockfile")
CAPTURE_DIR = DATA_DIR / "captures"
VK_F9 = 0x78
EVERY = {"lcu": 1.0, "shot_select": 2.0, "live": 2.0, "shot_game": 5.0, "phase": 1.0}
BURST, BURST_GAP_S = 5, 0.15
END_GRACE_S = 90.0
JPEG_QUALITY = 92

httpx_log = None


def now_ms() -> int:
    return int(time.time() * 1000)


class Lcu:
    """El API local del cliente de League; puerto y contraseña salen del lockfile, que cambia en cada arranque."""

    def __init__(self, lockfile: Path) -> None:
        self.lockfile = lockfile
        self.key: tuple[str, str] | None = None

    def _auth(self) -> tuple[str, str] | None:
        try:
            name, pid, port, password, proto = self.lockfile.read_text(encoding="utf-8").strip().split(":")
        except (OSError, ValueError):
            return None
        return f"{proto}://127.0.0.1:{port}", password

    def get(self, path: str):
        auth = self._auth()
        if auth is None:
            return None
        base, password = auth
        try:
            r = httpx.get(base + path, auth=("riot", password), verify=False, timeout=2.0)
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None


def screenshot(sct: mss.mss, path: Path) -> None:
    """Toda la pantalla (todos los monitores) en JPEG; recortar al cliente o al juego se hace después."""
    raw = sct.grab(sct.monitors[0])
    Image.frombytes("RGB", raw.size, raw.rgb).save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)


def f9_pressed() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_F9) & 0x8000)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def selftest(lcu: Lcu) -> None:
    out = CAPTURE_DIR / "selftest"
    out.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        t0 = time.perf_counter()
        screenshot(sct, out / "shot.jpg")
        ms = (time.perf_counter() - t0) * 1000
        size = sct.monitors[0]
    print(f"captura: {size['width']}x{size['height']} en {ms:.0f} ms -> {out / 'shot.jpg'} ({(out / 'shot.jpg').stat().st_size // 1024} KB)")
    print(f"lockfile: {'sí' if lcu.lockfile.is_file() else 'no (el cliente no está abierto)'} | fase: {lcu.get('/lol-gameflow/v1/gameflow-phase')!r}"
          f" | partida en curso: {'sí' if fetch_snapshot() else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lockfile", type=Path, default=LOCKFILE)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    import warnings
    warnings.filterwarnings("ignore")                       # certificados autofirmados del cliente
    lcu = Lcu(args.lockfile)
    if args.selftest:
        selftest(lcu)
        return

    run = CAPTURE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    for sub in ("lcu", "live", "shots"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    marks = (run / "marks.jsonl").open("a", encoding="utf-8")
    print(f"Grabando en {run}\nF9 marca un momento (5 capturas seguidas). Ctrl-C para parar.\n")

    last = {k: 0.0 for k in EVERY}
    phase, last_session, last_game_time = None, None, None
    counts = {"lcu": 0, "live": 0, "shots": 0, "marks": 0}
    saw_game, ended_at, f9_was = False, None, False
    with mss.mss() as sct:
        def shot(tag: str) -> None:
            screenshot(sct, run / "shots" / f"shot_{now_ms()}_{tag}.jpg")
            counts["shots"] += 1
        try:
            while True:
                t = time.monotonic()
                if t - last["phase"] >= EVERY["phase"]:
                    last["phase"] = t
                    new_phase = lcu.get("/lol-gameflow/v1/gameflow-phase")
                    if new_phase != phase:
                        phase = new_phase
                        flow = lcu.get("/lol-gameflow/v1/session")
                        if flow is not None:
                            write_json(run / "lcu" / f"gameflow_{now_ms()}_{phase}.json", flow)
                        print(f"\n[{datetime.now():%H:%M:%S}] fase: {phase}")
                in_select = phase == "ChampSelect"
                if in_select and t - last["lcu"] >= EVERY["lcu"]:
                    last["lcu"] = t
                    session = lcu.get("/lol-champ-select/v1/session")
                    if session is not None and session != last_session:
                        last_session = session
                        write_json(run / "lcu" / f"champselect_{now_ms()}.json", session)
                        counts["lcu"] += 1
                if in_select and t - last["shot_select"] >= EVERY["shot_select"]:
                    last["shot_select"] = t
                    shot("select")
                in_game = False
                if t - last["live"] >= EVERY["live"]:
                    last["live"] = t
                    snap = fetch_snapshot()
                    if snap:
                        in_game = saw_game = True
                        ended_at = None
                        last_game_time = (snap.get("gameData") or {}).get("gameTime")
                        write_json(run / "live" / f"live_{now_ms()}_{int(last_game_time or 0):05d}.json", snap)
                        counts["live"] += 1
                if (in_game or phase == "InProgress") and saw_game and t - last["shot_game"] >= EVERY["shot_game"]:
                    last["shot_game"] = t
                    shot("game")
                pressed = f9_pressed()
                if pressed and not f9_was:
                    marks.write(json.dumps({"ms": now_ms(), "phase": phase, "game_time": last_game_time}) + "\n")
                    marks.flush()
                    counts["marks"] += 1
                    if in_select and last_session is not None:
                        write_json(run / "lcu" / f"champselect_{now_ms()}_mark.json", last_session)
                    for _ in range(BURST):
                        shot("mark")
                        time.sleep(BURST_GAP_S)
                    print(f"\n[{datetime.now():%H:%M:%S}] F9: momento marcado")
                f9_was = pressed
                # fin: la partida terminó y ya no hay ni partida ni selección
                if saw_game and phase not in ("InProgress", "ChampSelect", "GameStart", "Reconnect"):
                    ended_at = ended_at or t
                    if t - ended_at >= END_GRACE_S:
                        print("\nPartida terminada: fin de la grabación.")
                        break
                print(f"\r  fase {phase or '-':<14} selección {counts['lcu']:>3}  partida {counts['live']:>4}  capturas {counts['shots']:>4}  marcas {counts['marks']}   ", end="", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nParado.")
    marks.close()
    print(f"Guardado en {run}: {counts}")


if __name__ == "__main__":
    main()
