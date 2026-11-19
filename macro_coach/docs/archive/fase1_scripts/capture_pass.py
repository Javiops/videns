"""Pasada de captura a 8x: pone el juego en primer plano, reproduce la replay
completa a 8x y muestrea minimapa (pantalla) + reloj y playerlist (API).

Salida: <out_dir>/mm_<t en decisegundos>.png  y  <out_dir>/samples.jsonl
"""
import ctypes
import json
import ssl
import sys
import time
import urllib.request

import cv2
import numpy as np
import mss

OUT = sys.argv[1]
T_START = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
API = "https://127.0.0.1:2999"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(path):
    with urllib.request.urlopen(API + path, context=CTX, timeout=5) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=CTX, timeout=5) as r:
        return json.loads(r.read())


def focus_game():
    u32 = ctypes.windll.user32
    hwnd = u32.FindWindowW("RiotWindowClass", None)
    if not hwnd:
        raise SystemExit("ERROR: no encuentro la ventana del juego")
    for _ in range(6):
        u32.keybd_event(0x12, 0, 0, 0)      # Alt down: permite robar foco
        u32.ShowWindow(hwnd, 9)             # SW_RESTORE
        u32.SetForegroundWindow(hwnd)
        u32.keybd_event(0x12, 0, 2, 0)      # Alt up
        time.sleep(1.0)
        if u32.GetForegroundWindow() == hwnd:
            time.sleep(1.0)
            return
    raise SystemExit("ERROR: no consigo poner el juego en primer plano")


def looks_like_map(mm_bgr):
    hsv = cv2.cvtColor(mm_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    team = (((hue >= 92) & (hue <= 112)) | (hue >= 170) | (hue <= 4)) \
        & (sat > 120) & (val > 120)
    return team.sum() > 150


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    focus_game()
    length = get("/replay/playback")["length"]
    post("/replay/playback", {"time": T_START, "paused": True})
    time.sleep(6)                    # deja cargar el seek
    post("/replay/playback", {"speed": 8, "paused": False})
    rows, last_pl, n, bad_streak = [], -99.0, 0, 0
    with mss.mss() as sct:
        mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        while True:
            pb = get("/replay/playback")
            t = pb["time"]
            if t >= length - 1 or (pb["paused"] and t > 5):
                break
            img = np.array(sct.grab(mon))[:, :, :3]
            crop = img[540:1060, 1380:1905]
            if not looks_like_map(crop):
                bad_streak += 1
                print(f"AVISO: frame invalido en t={t:.0f} (foco perdido?)", flush=True)
                if bad_streak >= 15:
                    post("/replay/playback", {"speed": 1, "paused": True})
                    raise SystemExit("ABORT: demasiados frames invalidos seguidos")
                focus_game()
                continue
            bad_streak = 0
            cv2.imwrite(f"{OUT}/mm_{int(t * 10):06d}.png", crop)
            row = {"t": round(t, 2)}
            if t - last_pl >= 8:
                pl = get("/liveclientdata/playerlist")
                row["players"] = [
                    {"champ": p["championName"], "cs": p["scores"]["creepScore"],
                     "level": p["level"], "dead": p["isDead"],
                     "respawn": round(p["respawnTimer"], 1)} for p in pl]
                last_pl = t
            rows.append(row)
            n += 1
            if n % 50 == 0:
                print(f"t={t:.0f}s muestras={n}", flush=True)
            time.sleep(0.30)
    with open(f"{OUT}/samples.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    post("/replay/playback", {"speed": 1, "paused": True})
    print(f"FIN: {n} muestras hasta t={t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
