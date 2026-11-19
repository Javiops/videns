"""Construye la mascara de regiones del minimapa y valida que ningun
campamento caiga en un carril.

Parametros (coordenadas normalizadas, (0,0) = base azul):
  - carriles: polilineas con su codo; ancho W, que se ensancha hasta W_MAX
    hacia el centro en el cruce del rio (transicion suave, sin escalones).
  - bases: cuarto de circulo completo, radio R_BASE (medido en los pixeles).
  - fosos: circulos propios; rio por el color azul real del mapa.
"""
import cv2
import numpy as np

S = "C:/Users/Javier/AppData/Local/Temp/claude/C--Users-Javier-build-tracker/19666e3e-e394-403c-9692-3d8eafc40471/scratchpad"

TOP = [(0.085, 0.105), (0.045, 0.80), (0.125, 0.915), (0.85, 0.955)]
BOT = [(0.105, 0.085), (0.80, 0.045), (0.915, 0.125), (0.955, 0.85)]
MID = [(0.125, 0.125), (0.875, 0.875)]
W, W_MAX, FADE = 0.075, 0.095, 0.55
W_MID = 0.058
R_BASE = 0.37
PIT_BARON, PIT_DRAGON, R_PIT = (0.335, 0.695), (0.672, 0.302), 0.062
RIVER_BAND = 0.14
R_CAMP = 0.042

CB = {"gromp": (0.142, 0.568), "azul": (0.256, 0.531), "lobos": (0.254, 0.433),
      "raptors": (0.467, 0.342), "rojo": (0.525, 0.272), "krugs": (0.563, 0.181)}
LB = {"BASE": 1, "BASE_ENEMIGA": 2, "TOP": 3, "MID": 4, "BOT": 5, "RIO_TOP": 6,
      "RIO_BOT": 7, "JGL_PA": 8, "JGL_PR": 9, "JGL_EA": 10, "JGL_ER": 11,
      "PIT_BARON": 12, "PIT_DRAGON": 13}
NAME = {v: k for k, v in LB.items()}


def build():
    bg = cv2.imread(f"{S}/map_background.png")
    N = bg.shape[0]
    hsv = cv2.cvtColor(cv2.medianBlur(bg, 5), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    ys, xs = np.mgrid[0:N, 0:N]
    nx, ny = xs / N, 1 - ys / N
    sq2 = np.sqrt(2)

    def poly(pts):
        d = np.full((N, N), 9.0)
        inner = np.zeros((N, N), bool)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            vx, vy = x1 - x0, y1 - y0
            L2 = vx * vx + vy * vy
            t = np.clip(((nx - x0) * vx + (ny - y0) * vy) / L2, 0, 1)
            px, py = x0 + t * vx, y0 + t * vy
            dd = np.hypot(nx - px, ny - py)
            m = dd < d
            toward = ((nx - px) * (0.5 - px) + (ny - py) * (0.5 - py)) > 0
            d = np.where(m, dd, d)
            inner = np.where(m, toward, inner)
        return d, inner

    d_top, in_top = poly(TOP)
    d_bot, in_bot = poly(BOT)
    d_mid, _ = poly(MID)
    ramp = np.clip(1 - (np.abs(nx + ny - 1) / sq2) / FADE, 0, 1)
    w_in = W + (W_MAX - W) * ramp
    lane_top = (d_top < W) | (in_top & (d_top < w_in))
    lane_bot = (d_bot < W) | (in_bot & (d_bot < w_in))
    river = ((hue >= 88) & (hue <= 112) & (sat >= 50) & (val >= 40)
             & (np.abs(nx + ny - 1) / sq2 < RIVER_BAND))

    lab = np.zeros((N, N), np.uint8)
    own = nx + ny < 1
    lab[own & (ny > nx)] = LB["JGL_PA"]
    lab[own & (ny <= nx)] = LB["JGL_PR"]
    lab[~own & (ny < nx)] = LB["JGL_EA"]
    lab[~own & (ny >= nx)] = LB["JGL_ER"]
    lab[river & (ny > nx)] = LB["RIO_TOP"]
    lab[river & (ny <= nx)] = LB["RIO_BOT"]
    lab[lane_top] = LB["TOP"]
    lab[lane_bot] = LB["BOT"]
    lab[d_mid < W_MID] = LB["MID"]
    # un campamento es jungla por definicion: se recorta su disco y vuelve
    # a su cuadrante aunque el carril pase al lado (gromp y krugs lo hacen)
    for cx, cy in list(CB.values()) + [(1 - x, 1 - y) for x, y in CB.values()]:
        disc = np.hypot(nx - cx, ny - cy) < R_CAMP
        if cx + cy < 1:
            lab[disc] = LB["JGL_PA"] if cy > cx else LB["JGL_PR"]
        else:
            lab[disc] = LB["JGL_EA"] if cy < cx else LB["JGL_ER"]
    lab[np.hypot(nx - PIT_BARON[0], ny - PIT_BARON[1]) < R_PIT] = LB["PIT_BARON"]
    lab[np.hypot(nx - PIT_DRAGON[0], ny - PIT_DRAGON[1]) < R_PIT] = LB["PIT_DRAGON"]
    lab[np.hypot(nx, ny) < R_BASE] = LB["BASE"]
    lab[np.hypot(1 - nx, 1 - ny) < R_BASE] = LB["BASE_ENEMIGA"]
    return cv2.medianBlur(lab, 5), bg, N


def region_at(lab, N, p):
    x = min(max(int(p[0] * N), 0), N - 1)
    y = min(max(int((1 - p[1]) * N), 0), N - 1)
    return NAME[int(lab[y, x])]


def validate(lab, N):
    """Ningun campamento puede caer en un carril ni en una base."""
    bad = []
    for name, p in CB.items():
        for tag, q in ((f"{name}", p), (f"{name} enemigo", (1 - p[0], 1 - p[1]))):
            r = region_at(lab, N, q)
            if not r.startswith("JGL"):
                bad.append((tag, r))
    return bad


def render(lab, bg, N, path):
    COL = {1: (150, 110, 70), 2: (110, 90, 150), 3: (80, 190, 110), 4: (70, 210, 210),
           5: (190, 130, 220), 6: (220, 160, 70), 7: (180, 100, 40), 8: (60, 130, 60),
           9: (50, 90, 140), 10: (120, 70, 150), 11: (90, 80, 190), 12: (230, 80, 230),
           13: (60, 220, 255)}
    SHOW = {1: "BASE", 2: "BASE ENEM", 3: "TOP", 4: "MID", 5: "BOT", 6: "RÍO TOP",
            7: "RÍO BOT", 8: "JGL PROPIA AZUL", 9: "JGL PROPIA ROJO",
            10: "JGL ENEM AZUL", 11: "JGL ENEM ROJO", 12: "FOSO BARÓN", 13: "FOSO DRAGÓN"}
    Z = 2
    big = cv2.resize(bg, (N * Z, N * Z), interpolation=cv2.INTER_NEAREST)
    labz = cv2.resize(lab, (N * Z, N * Z), interpolation=cv2.INTER_NEAREST)
    ov = big.copy()
    for k, c in COL.items():
        ov[labz == k] = c
    vis = cv2.addWeighted(big, 0.55, ov, 0.45, 0)
    vis[cv2.Canny(labz * 18, 30, 90) > 0] = (255, 255, 255)
    for c, p in CB.items():
        for q in (p, (1 - p[0], 1 - p[1])):
            cx, cy = int(q[0] * N * Z), int((1 - q[1]) * N * Z)
            cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 7, (0, 0, 0), 1)
    for k, name in SHOW.items():
        m = (labz == k).astype(np.uint8)
        if m.sum() < 400:
            continue
        n, _, stats, cents = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            if stats[i][4] < 3000:
                continue
            cx, cy = int(cents[i][0]), int(cents[i][1])
            for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(vis, name, (cx - len(name) * 4, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, th)
    cv2.imwrite(path, vis)


if __name__ == "__main__":
    lab, bg, N = build()
    bad = validate(lab, N)
    np.save(f"{S}/region_labels.npy", lab)
    render(lab, bg, N, f"{S}/regiones_v2.png")
    if bad:
        print("CAMPAMENTOS MAL UBICADOS:", bad)
    else:
        print("validacion ok: los 12 campamentos caen en jungla")
    tot = lab.size
    for k in sorted(NAME):
        print(f"  {NAME[k]:14s} {100 * (lab == k).mean():5.1f}%")
