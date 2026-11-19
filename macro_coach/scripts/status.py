"""Estado actual del proyecto: clave, corpus por etapas, dataset, modelo,
feedback y git. Ejecutar antes de empezar cualquier sesión (contrato CLAUDE.md)."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (ROOT, CORPUS, ML, FEEDBACK, QUEUE_FILE,
                        RIOT_API_KEY, LEXICON_V, FEATURES_V, SNAPSHOT_V,
                        VIS_MIN_SENSITIVITY)


def stage_row(g: Path) -> str:
    def has(x):
        return "x" if (g / x).exists() else "."
    passes = "".join(has(f"pass_{m}") for m in ("omni", "blue", "red"))
    meta = {}
    if (g / "meta.json").exists():
        meta = json.loads((g / "meta.json").read_text(encoding="utf-8"))
    n = meta.get("n_decisions", "-")
    lex = meta.get("lexicon_v", "-")
    return (f"  {g.name:22s} mv5:{has('mv5_match.json')} pases:{passes} "
            f"traj:{has('traj.jsonl')} camps:{has('camps.json')} "
            f"vis:{has('visibility.json')} dec:{has('decisions.jsonl')} "
            f"n={n} lex={lex}")


def main():
    print(f"macro_coach fase 2 — lexicon {LEXICON_V}, features {FEATURES_V}, "
          f"snapshot {SNAPSHOT_V}, umbral sensibilidad {VIS_MIN_SENSITIVITY}")
    print(f"RIOT_API_KEY: {'presente' if RIOT_API_KEY else 'FALTA (.env)'}")

    games = sorted(CORPUS.iterdir()) if CORPUS.exists() else []
    games = [g for g in games if g.is_dir()]
    print(f"\ncorpus: {len(games)} partidas")
    for g in games[:30]:
        print(stage_row(g))
    if len(games) > 30:
        print(f"  ... y {len(games) - 30} mas")

    if QUEUE_FILE.exists():
        n = sum(1 for _ in open(QUEUE_FILE, encoding="utf-8"))
        print(f"\ncola de partidas: {n} en {QUEUE_FILE.name}")

    split_f = ML / "split.json"
    if split_f.exists():
        sp = json.loads(split_f.read_text(encoding="utf-8"))
        print(f"\nsplit congelado: {len(sp['train'])}/{len(sp['val'])}/"
              f"{len(sp['test'])} partidas (lex {sp.get('lexicon_v')}, "
              f"features {sp.get('features_v')})")
        from app import dataset
        problems = dataset.validate_split(sp)
        if problems:
            print("  SPLIT NO UTILIZABLE:")
            for x in problems:
                print(f"    - {x}")
        else:
            print("  split válido para el corpus y código actuales")
        for part in ("train", "val", "test"):
            f = ML / f"decisions_{part}.jsonl"
            if f.exists():
                n = sum(1 for _ in open(f, encoding="utf-8"))
                print(f"  decisions_{part}.jsonl: {n} filas")
    model_f = ML / "model.json"
    if model_f.exists():
        mm = json.loads(model_f.read_text(encoding="utf-8"))
        print(f"modelo: {mm.get('kind')} entrenado {mm.get('trained_at')} "
              f"(games={mm.get('n_games')}, smoke={mm.get('smoke')})")
    else:
        print("\nmodelo: ninguno")

    fb = FEEDBACK / "feedback.jsonl"
    if fb.exists():
        n = sum(1 for _ in open(fb, encoding="utf-8"))
        print(f"feedback del owner: {n} veredictos")

    try:
        out = subprocess.run(["git", "status", "--short"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10).stdout
        n = len([l for l in out.splitlines() if l.strip()])
        print(f"\ngit: {n} cambios sin commitear")
    except Exception:
        pass


if __name__ == "__main__":
    main()
