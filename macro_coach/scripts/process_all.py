"""Procesa todas las partidas del corpus con pass_omni y sin decisions.jsonl
(o todas con --redo). Uso típico: la mañana después de capture_queue.py.

La captura ya no está usando la pantalla, así que aquí sí conviene paralelizar:
cada partida es un proceso independiente. Se limitan los hilos de OpenCV por
proceso para no sobresuscribir la CPU.

Uso: python scripts/process_all.py [--redo] [--jobs N]
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import CORPUS, is_excluded

SCRIPT = Path(__file__).resolve().parent / "process_game.py"


def run_one(mid: str, threads: int) -> tuple:
    env = dict(os.environ)
    env["OPENCV_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    log = CORPUS.parent / "logs" / f"proc_{mid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, str(SCRIPT), mid],
                             stdout=fh, stderr=subprocess.STDOUT, env=env)
    return mid, rc, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="partidas en paralelo (la captura ya no usa pantalla)")
    ap.add_argument("--skip-audit", action="store_true",
                    help="NO usar salvo depuración: salta la puerta de niebla")
    args = ap.parse_args()

    # --- PUERTA: auditoría de niebla antes de procesar el lote ---
    # Una pasada con la niebla equivocada produce visibilidad que parece
    # correcta y no lo es; la validación del momento de captura no la ve.
    if not args.skip_audit:
        import audit_fog
        bad = []
        for g in sorted(CORPUS.iterdir()):
            if not g.is_dir() or not (g / "pass_omni").exists() or is_excluded(g.name):
                continue
            r = audit_fog.audit_game(g)
            problems = [m for m in ("blue", "red") if r.get(m + "_ok") is False]
            if r.get("blue_vs_red") is not None and not r.get("distinct_fogs"):
                problems.append("misma niebla en ambas")
            if problems:
                bad.append((g.name, problems))
        if bad:
            print("PUERTA DE NIEBLA: no proceso el lote, hay pasadas malas:")
            for mid, problems in bad:
                print(f"  {mid}: {problems}")
            print("Recaptura esas pasadas (borra el pass_* y relanza "
                  "capture_queue.py) o pasa --skip-audit a sabiendas.")
            return 2
        print("puerta de niebla: OK en todas las partidas", flush=True)

    pend = []
    for d in sorted(CORPUS.iterdir()) if CORPUS.exists() else []:
        if not (d / "pass_omni" / "samples.jsonl").exists():
            continue
        if is_excluded(d.name):
            print(f"  (excluida del corpus: {d.name})")
            continue
        if (d / "decisions.jsonl").exists() and not args.redo:
            continue
        pend.append(d.name)

    jobs = max(1, args.jobs)
    threads = max(1, (os.cpu_count() or 4) // (jobs * 2))
    print(f"{len(pend)} partidas por procesar, {jobs} en paralelo "
          f"({threads} hilos de OpenCV cada una)", flush=True)

    done, fails = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for mid, rc, secs, log in pool.map(lambda m: run_one(m, threads), pend):
            done += 1
            estado = "OK " if rc == 0 else "FALLO"
            if rc != 0:
                fails.append(mid)
            eta = (time.time() - t0) / done * (len(pend) - done) / 60
            print(f"[{done}/{len(pend)}] {estado} {mid} ({secs/60:.1f} min) "
                  f"— quedan ~{eta:.0f} min   log: {log.name}", flush=True)

    print(f"\nOK {len(pend) - len(fails)}, fallos {len(fails)}: {fails}")
    if pend and not fails:
        print("Siguiente: scripts/audit_fog.py (si no se ha pasado) y luego "
              "build_dataset.py --rebuild-split")


if __name__ == "__main__":
    sys.exit(main() or 0)
