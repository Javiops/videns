"""Modelo básico v1: HistGradientBoosting sobre features f1, contra dos
baselines obligatorios (clase mayoritaria y regla-de-región). Evalúa SOLO en
val; test queda para una única confirmación final, manual y explícita.

Se niega a reportar con menos de MIN_GAMES_REPORT partidas. --smoke entrena
sobre datos sintéticos solo para validar la fontanería y lo deja dicho en el
artefacto (nunca escribe informe en data/reports).

Uso: python scripts/train_baseline.py [--smoke]
"""
import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import dataset
from app.config import (ML, REPORTS, MIN_GAMES_REPORT, LEXICON_V,
                        FEATURES_V, SNAPSHOT_V, ensure_dirs)
from app.lexicon import ALL_TOKENS


def load_part(part: str) -> tuple:
    rows = []
    f = ML / f"decisions_{part}.jsonl"
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh]
    return rows


def to_xy(rows: list, names: list) -> tuple:
    X = np.array([[r["features"].get(n, -1) for n in names] for r in rows],
                 dtype=np.float64)
    y = np.array([r["token"] for r in rows])
    return X, y


def synthetic_rows(n: int = 400) -> list:
    """Datos sintéticos para --smoke: features del dummy + ruido, tokens al azar."""
    import random
    rng = random.Random(7)
    base = dataset.feature_vector(dataset._dummy_decision())
    rows = []
    for i in range(n):
        f = {k: v + rng.random() for k, v in base.items()}
        rows.append({"match_id": f"SMOKE_{i % 10}", "pid": 2, "t": i,
                     "token": rng.choice(ALL_TOKENS), "features": f})
    return rows


def topk_acc(proba: np.ndarray, classes: np.ndarray, y: np.ndarray, k: int) -> float:
    idx = np.argsort(-proba, axis=1)[:, :k]
    hits = sum(1 for i, row in enumerate(idx) if y[i] in classes[row])
    return hits / max(len(y), 1)


def region_rule_baseline(train_rows: list, val_rows: list) -> float:
    """Baseline: el token más frecuente para (mi región). Lo mínimo que un
    modelo tiene que superar para valer algo."""
    table = defaultdict(Counter)
    for r in train_rows:
        table[r["features"]["my_region"]][r["token"]] += 1
    fallback = Counter(r["token"] for r in train_rows).most_common(1)[0][0]
    hits = 0
    for r in val_rows:
        c = table.get(r["features"]["my_region"])
        pred = c.most_common(1)[0][0] if c else fallback
        hits += pred == r["token"]
    return hits / max(len(val_rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    if args.smoke:
        rows = synthetic_rows()
        train_rows, val_rows = rows[:300], rows[300:]
        n_games = 0
        print("SMOKE: datos sinteticos, solo fontaneria; nada de esto es una metrica")
    else:
        train_rows, val_rows = load_part("train"), load_part("val")
        games = {r["match_id"] for r in train_rows} | {r["match_id"] for r in val_rows}
        n_games = len(games)
        if n_games < MIN_GAMES_REPORT:
            raise SystemExit(
                f"NEGADO: {n_games} partidas < MIN_GAMES_REPORT={MIN_GAMES_REPORT}. "
                "Con tan pocas partidas cualquier numero es ruido con buena cara "
                "(docs/history.md). Captura mas partidas o usa --smoke para fontaneria.")

    names = dataset.feature_names()
    Xtr, ytr = to_xy(train_rows, names)
    Xva, yva = to_xy(val_rows, names)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xva)
    pred = clf.classes_[np.argmax(proba, axis=1)]

    maj = Counter(ytr).most_common(1)[0][0]
    metrics = {
        "top1": float((pred == yva).mean()),
        "top3": topk_acc(proba, clf.classes_, yva, 3),
        "macro_f1": float(f1_score(yva, pred, average="macro", zero_division=0)),
        "baseline_majority": float((yva == maj).mean()),
        "baseline_region_rule": region_rule_baseline(train_rows, val_rows),
        "n_train": len(train_rows), "n_val": len(val_rows), "n_games": n_games,
        "n_games_train": len({r["match_id"] for r in train_rows}),
        "n_games_val": len({r["match_id"] for r in val_rows}),
        "split": "val", "lexicon_v": LEXICON_V, "features_v": FEATURES_V,
        "snapshot_v": SNAPSHOT_V,
        "vision_usable_frac": round(
            sum(1 for r in val_rows if r["features"].get("vision_usable") == 1)
            / max(len(val_rows), 1), 3),
    }
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(ML / "model.pkl", "wb") as fh:
        pickle.dump({"clf": clf, "feature_names": names,
                     "classes": list(clf.classes_)}, fh)
    meta = {"kind": "HistGradientBoosting", "trained_at": ts, "smoke": args.smoke,
            "n_games": n_games, "lexicon_v": LEXICON_V, "features_v": FEATURES_V}
    if not args.smoke:
        report = REPORTS / f"{ts}_baseline_val.json"
        report.write_text(json.dumps({"metrics": metrics,
                                      "per_class_support": dict(Counter(yva))},
                                     indent=1), encoding="utf-8")
        meta["report"] = str(report)
        print(f"informe: {report}")
    else:
        print("SMOKE OK: artefacto marcado smoke=true, sin informe")
    (ML / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
