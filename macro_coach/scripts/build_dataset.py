"""Congela el split por partida y exporta decisions_{train,val,test}.jsonl.

Uso: python scripts/build_dataset.py [--rebuild-split]
El split existente nunca se toca sin --rebuild-split (disciplina build_tracker:
las partidas nuevas quedan fuera hasta que se reconstruye explícitamente).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dataset
from app.config import ensure_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-split", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    games = dataset.corpus_games()
    print(f"{len(games)} partidas procesadas en el corpus (excluidas aparte)")
    try:
        split = dataset.frozen_split(rebuild=args.rebuild_split)
    except dataset.SplitError as e:
        print(f"SPLIT RECHAZADO\n{e}")
        return 2
    print(f"split: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test "
          f"(lexicon {split['lexicon_v']}, features {split.get('features_v')})")
    if split.get("provisional"):
        print("AVISO: split PROVISIONAL (pocas partidas): todo a train, "
              "sin val ni test. No se puede reportar métrica con esto.")
    new = [g for g, _ in games
           if g not in set(split["train"] + split["val"] + split["test"])]
    if new:
        print(f"AVISO: {len(new)} partidas fuera del split congelado "
              f"(usa --rebuild-split cuando toque): {new[:5]}")
    try:
        counts = dataset.export(split)
    except (dataset.SplitError, ValueError) as e:
        print(f"EXPORT RECHAZADO: {e}")
        return 2
    print(f"exportado: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
