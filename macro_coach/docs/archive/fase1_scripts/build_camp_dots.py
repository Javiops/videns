"""Serie de presencia del rombo naranja por campamento, desde pass1."""
import cv2
import numpy as np
import glob
import os
import json

S = "C:/Users/Javier/AppData/Local/Temp/claude/C--Users-Javier-build-tracker/19666e3e-e394-403c-9692-3d8eafc40471/scratchpad"
GEOM = lambda t: (9, 5, 508) if t < 1148 else (104, 101, 414)
CB = {"gromp": (0.142, 0.568), "azul": (0.256, 0.531), "lobos": (0.254, 0.433),
      "raptors": (0.467, 0.342), "rojo": (0.525, 0.272), "krugs": (0.563, 0.181)}
CAMPS = dict(CB)
CAMPS.update({k + " enemigo": (1 - x, 1 - y) for k, (x, y) in CB.items()})

series = {c: [] for c in CAMPS}
for fp in sorted(glob.glob(f"{S}/pass1/mm_*.png")):
    t = int(os.path.basename(fp)[3:9]) / 10.0
    img = cv2.imread(fp)
    if img is None or img.shape[0] < 513:
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    MX0, MY0, MSIZE = GEOM(t)
    for c, (nx, ny) in CAMPS.items():
        x, y = int(MX0 + nx * MSIZE), int(MY0 + (1 - ny) * MSIZE)
        w = hsv[max(0, y - 10):y + 11, max(0, x - 10):x + 11]
        n = ((w[..., 0] >= 8) & (w[..., 0] <= 26)
             & (w[..., 1] > 140) & (w[..., 2] > 160)).sum()
        series[c].append((t, int(n)))
json.dump(series, open(f"{S}/camp_dots.json", "w"))
print("camp_dots.json reconstruido")
