"""Fija el tamaño del minimapa del espectador, que si no se resetea solo.

El owner mueve el minimapa desde el juego y al siguiente replay vuelve a estar
pequeño y en la esquina. La razon: hay DOS almacenes de ajustes y mandan el que
no toca el slider.

    clave                   game.cfg   PersistedSettings.json   se ve
    MinimapScaleSpectator     5.0              1.0              1.0
    MinimapNativeOffsetX    -0.3448          -0.0203          -0.0203

`game.cfg` es lo que escribe el juego; `PersistedSettings.json` es del cliente
(por cuenta) y lo pisa al arrancar. Asi que hay que escribir el segundo, y con
el juego CERRADO — si no, lo reescribe al salir.

Tamaño: importa porque los iconos de campeon se aprenden en la pasada omni y se
buscan en las de niebla, y porque con mas pixeles el detector acierta mas.

Posicion: hay que esquivar DOS cosas. En la esquina, a 508px el minimapa tapa
los marcos del equipo rojo (x>=1835, y=120-640) y con ellos el temporizador de
muerte del support — que es justo para lo que sirven esos marcos. Centrado y
demasiado bajo, se mete en la franja del marcador (y>=820) de la que sale el
oro. El hueco bueno es x=727, y=300 (offsets -0.3448 / -0.2269): centrado en
horizontal, y su borde inferior queda en y=808, por encima del marcador.

Los offsets son fraccion de pantalla y se miden desde la posicion de esquina:
con (0,0) el minimapa de 508px sale en (1389,545).

Uso:
  python scripts/set_minimap.py                 # ver los valores actuales
  python scripts/set_minimap.py --escala 5.0    # fijar y verificar
  python scripts/set_minimap.py --restaurar <fichero_backup>
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERSISTED = Path(r"C:\Riot Games\League of Legends\Config\PersistedSettings.json")
GAME_CFG = Path(r"C:\Riot Games\League of Legends\Config\game.cfg")
CLAVES = ("MinimapScaleSpectator", "MinimapScale",
          "MinimapNativeOffsetX", "MinimapNativeOffsetY")


def ajustes(d):
    """Devuelve la lista de dicts {'name','value'} de PersistedSettings."""
    out = []
    def rec(o):
        if isinstance(o, dict):
            if "name" in o and "value" in o:
                out.append(o)
            for v in o.values():
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)
    rec(d)
    return out


def muestra():
    d = json.loads(PERSISTED.read_text(encoding="utf-8"))
    print(f"{PERSISTED.name}:")
    for s in ajustes(d):
        if s["name"] in CLAVES:
            print(f"  {s['name']:26s} = {s['value']}")
    print(f"{GAME_CFG.name}:")
    for linea in GAME_CFG.read_text(encoding="utf-8").splitlines():
        if any(linea.startswith(k + "=") for k in CLAVES):
            print(f"  {linea}")


def juego_abierto() -> bool:
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq League of Legends.exe"],
                       capture_output=True, text=True)
    return "League of Legends.exe" in r.stdout


def fija(escala: float, offx=None, offy=None):
    if juego_abierto():
        raise SystemExit("cierra el juego antes: al salir reescribe los ajustes "
                         "y se pierde el cambio")
    copia = PERSISTED.with_suffix(".json.macro_coach_bak")
    if not copia.exists():
        shutil.copy2(PERSISTED, copia)
        print(f"copia de seguridad: {copia}")
    d = json.loads(PERSISTED.read_text(encoding="utf-8"))
    tocados = 0
    for s in ajustes(d):
        if s["name"] == "MinimapScaleSpectator":
            print(f"  MinimapScaleSpectator {s['value']} -> {escala:.4f}")
            s["value"] = f"{escala:.4f}"
            tocados += 1
        # Offsets = fraccion de pantalla, medidos desde la esquina. Ver la
        # cabecera para el hueco bueno y por que.
        elif s["name"] == "MinimapNativeOffsetX" and offx is not None:
            print(f"  MinimapNativeOffsetX {s['value']} -> {offx:.4f}")
            s["value"] = f"{offx:.4f}"
            tocados += 1
        elif s["name"] == "MinimapNativeOffsetY" and offy is not None:
            print(f"  MinimapNativeOffsetY {s['value']} -> {offy:.4f}")
            s["value"] = f"{offy:.4f}"
            tocados += 1
    if not tocados:
        raise SystemExit("no encuentro MinimapScaleSpectator en PersistedSettings")
    PERSISTED.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"escrito. Abre una replay y comprueba el tamaño con "
          f"scripts/preflight_gold.py o app.calibrate.find_minimap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escala", type=float)
    ap.add_argument("--offset-x", type=float, dest="offx",
                    help="desplazamiento X en fraccion de ancho de pantalla "
                         "(0 = esquina; -0.3448 = centrado)")
    ap.add_argument("--offset-y", type=float, dest="offy",
                    help="desplazamiento Y en fraccion de alto de pantalla")
    ap.add_argument("--restaurar")
    args = ap.parse_args()
    if args.restaurar:
        if juego_abierto():
            raise SystemExit("cierra el juego antes de restaurar")
        shutil.copy2(args.restaurar, PERSISTED)
        print(f"restaurado desde {args.restaurar}")
        return
    if args.escala is None:
        muestra()
        return
    fija(args.escala, args.offx, args.offy)
    muestra()


if __name__ == "__main__":
    main()
