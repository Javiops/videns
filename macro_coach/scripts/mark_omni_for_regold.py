"""Marca las pasadas TRUTH-SEEKER existentes para volver a capturarlas con oro.

Las 40 primeras partidas se capturaron con el marcador en vista de items: no
hay un solo pixel de oro que leer. Las pasadas de niebla SI valen (la
visibilidad no cambia), asi que solo hay que rehacer la omni.

No borra nada: RENOMBRA `pass_omni` a `pass_omni_sin_oro`. Con eso
`capture_queue.py` ve la omni como "falta" y la recaptura, y si algo sale mal
se deshace con --revertir. Cuando la tanda este verificada, --limpiar borra las
copias viejas.

Uso:
  python scripts/mark_omni_for_regold.py            # ver que haria
  python scripts/mark_omni_for_regold.py --hazlo
  python scripts/mark_omni_for_regold.py --revertir
  python scripts/mark_omni_for_regold.py --limpiar  # tras verificar la tanda
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CORPUS, is_excluded

VIEJO = "pass_omni_sin_oro"


def partidas():
    """Todas, o solo los MATCH_ID que se pasen por linea de ordenes."""
    pedidas = [a for a in sys.argv[1:] if not a.startswith("--")]
    return [d for d in sorted(CORPUS.iterdir())
            if d.is_dir() and not is_excluded(d.name)
            and (not pedidas or d.name in pedidas)]


def main():
    hazlo = "--hazlo" in sys.argv
    revertir = "--revertir" in sys.argv
    limpiar = "--limpiar" in sys.argv

    n = 0
    for d in partidas():
        omni, viejo = d / "pass_omni", d / VIEJO
        if limpiar:
            if viejo.exists() and (omni / "samples.jsonl").exists():
                print(f"  borrar {viejo.name} de {d.name}")
                if hazlo:
                    shutil.rmtree(viejo)
                n += 1
            continue
        if revertir:
            if viejo.exists():
                print(f"  {d.name}: {VIEJO} -> pass_omni")
                if hazlo:
                    if omni.exists():
                        shutil.rmtree(omni)
                    viejo.rename(omni)
                n += 1
            continue
        if viejo.exists():
            print(f"  {d.name}: ya marcada")
            continue
        if not (omni / "samples.jsonl").exists():
            continue
        con_oro = list(omni.glob("sb_*.png"))
        print(f"  {d.name}: pass_omni -> {VIEJO} "
              f"({len(con_oro)} recortes de marcador, sin oro)")
        if hazlo:
            omni.rename(viejo)
        n += 1

    accion = ("borradas" if limpiar else "revertidas" if revertir else "marcadas")
    if hazlo:
        print(f"\n{n} partidas {accion}.")
        if not limpiar and not revertir:
            print("Ahora: capture_queue.py recapturara SOLO la omni de cada una.")
            print("Si algo sale mal: --revertir --hazlo")
    else:
        print(f"\n{n} partidas se {accion}. Nada tocado: repite con --hazlo")


if __name__ == "__main__":
    main()
