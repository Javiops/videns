"""Tanda nocturna: recaptura TODAS las partidas del parche con ORO y procesa.

Fases, en SERIE (nunca se procesa mientras se captura — eso mata al renderer):

  1. marcar: renombra pass_omni -> pass_omni_sin_oro en las partidas sin oro,
     para que la cola vuelva a capturar solo la pasada TRUTH-SEEKER (las de
     niebla valen: la visibilidad no cambia y el minimapa ya es de 508px).
  2. capturar: capture_queue recaptura la omni de cada una (2 vueltas), con la
     vista de oro verificada por OCR y el guardia de geometria puesto.
  3. leer oro:  read_gold  -> gold.jsonl por partida.
  4. compras:   resolve_buys -> buys.jsonl (oro exacto en cada compra).
  5. procesar:  process_all --jobs 8 (reprocesa con la omni nueva).
  6. fugas:     check_leakage (debe dar 0 y 0).

Cada fase deja su marca en el log. Si la captura falla en una partida, la cola
la reintenta y sigue; el procesado se salta lo que no tenga pasadas.

Lanzar DESACOPLADO (no como hijo del shell del agente) y con la maquina libre:
  Start-Process .venv\\Scripts\\python.exe -ArgumentList "-u","scripts\\batch_regold.py" ...
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def corre(args, fase):
    log(f"=== {fase}: {' '.join(args)}")
    r = subprocess.run([PY, "-u"] + args, cwd=str(ROOT))
    log(f"=== {fase}: fin (codigo {r.returncode})")
    return r.returncode


def main():
    t0 = time.time()
    log("TANDA arranca")

    # 1. marcar las que faltan (idempotente: las ya marcadas se saltan)
    corre(["scripts/mark_omni_for_regold.py", "--hazlo"], "marcar")

    # 2. capturar (recaptura solo omni; blue/red ya estan)
    corre(["scripts/capture_queue.py"], "capturar")

    # 2.5 RESCATE: si una partida agoto sus reintentos, su pass_omni quedo
    # renombrada y sin reemplazo -> se le devuelve la original (sin oro pero
    # procesable), para no perder una partida que antes estaba bien.
    from app.config import CORPUS, is_excluded
    rescatadas = 0
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir() or is_excluded(d.name):
            continue
        omni, viejo = d / "pass_omni", d / "pass_omni_sin_oro"
        if not (omni / "samples.jsonl").exists() and (viejo / "samples.jsonl").exists():
            if omni.exists():
                import shutil
                shutil.rmtree(omni)
            viejo.rename(omni)
            rescatadas += 1
            log(f"rescate: {d.name} sin oro nuevo -> restaurada la omni original")
    log(f"=== rescate: {rescatadas} partidas restauradas a su omni original")

    # 3-4. oro y compras (solo dependen del omni nuevo + timeline + playerlist)
    corre(["scripts/read_gold.py"], "leer-oro")
    corre(["scripts/resolve_buys.py"], "compras")

    # 5. BORRAR derivados antes de reprocesar. El omni nuevo tiene OTROS nombres
    # de frame; si no se reconstruye, las decisiones apuntarian a frames que ya
    # no existen (el review se quedaria sin minimapa). Y process_all reutiliza
    # lo cacheado salvo que no exista: por eso se borra, no se confia en --redo
    # a secas (ya pasó una vez: "--redo terminó en 6s reusando cache vieja").
    derivados = ("traj.jsonl", "camps.json", "visibility.json",
                 "segments.jsonl", "decisions.jsonl")
    borradas = 0
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir() or is_excluded(d.name):
            continue
        if not (d / "pass_omni" / "samples.jsonl").exists():
            continue
        for f in derivados:
            (d / f).unlink(missing_ok=True)
        borradas += 1
    log(f"=== derivados borrados en {borradas} partidas (rebuild limpio)")

    # 6. reprocesar todo desde el omni nuevo, y las puertas
    corre(["scripts/process_all.py", "--redo", "--jobs", "8"], "procesar")
    corre(["scripts/check_leakage.py"], "fugas")

    # 7. limpiar las copias sin_oro de las que SI recapturaron con exito
    #    (deja intactas las rescatadas: esas siguen siendo su unica omni).
    corre(["scripts/mark_omni_for_regold.py", "--limpiar", "--hazlo"], "limpiar")

    mins = (time.time() - t0) / 60
    log(f"TANDA completa en {mins:.0f} min")


if __name__ == "__main__":
    main()
