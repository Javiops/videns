"""Legality of the cohort truths at the estimated wallet, and every arm's exactness conditioned on it.

Owner's question (2026-09-27): "con respecto a las legales, cuanto es el
porcentaje?" — the same accounting the reviewer did on the v3 probe (budget-
blocked vs structurally blocked main purchases; exact rate on the feasible
ones). Computed on the v4 cohort for R1, the old mixture and every stage-2
arm from their persisted rows. Read-only; no model, no training.
"""
import json
import time
from collections import Counter
from pathlib import Path

from stage2_newgen import OUT_DIR, STAGE1, setup, now_utc, v1, ROOT
from basket_wardfree import stripped_truth

OUT = ROOT / "data" / "reports" / "cohort_legality_v4_20260927"
ROWS = {"new_bf": STAGE1 / "rows_new_bf_full.jsonl", "old_bf": STAGE1 / "rows_old_bf_full.jsonl",
        "old_mixture": STAGE1 / "rows_mixture_full.jsonl", "wardfree_new": OUT_DIR / "rows_wardfree_new_full.jsonl",
        "control_ft_new": OUT_DIR / "rows_control_ft_new_full.jsonl", "econ_ft_new": OUT_DIR / "rows_econ_ft_new_full.jsonl",
        "M2": OUT_DIR / "rows_M2_mix_full.jsonl"}


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    S = setup(False, need_gpu=False, load_fit=False, load_cohort=True, cohort_tensors=False, load_model=False)
    ps = S.purchase_sequence
    legal_est, legal_inf, has_main = [], [], []
    t0 = time.monotonic()
    for i, row in enumerate(S.val_rows):
        t = stripped_truth(row)
        if not t:
            has_main.append(False)
            legal_est.append(None)
            legal_inf.append(None)
            continue
        action = tuple(sorted(t.elements()))
        has_main.append(True)
        legal_est.append(ps(action, row, float(row["gold_est"]), S.dragon) is not None)
        legal_inf.append(ps(action, row, 1e9, S.dragon) is not None)
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{S.n_val} ({(i + 1) / (time.monotonic() - t0):.0f} filas/s)", flush=True)
    n_main = sum(has_main)
    n_legal = sum(1 for m, e in zip(has_main, legal_est) if m and e)
    n_budget = sum(1 for m, e, f in zip(has_main, legal_est, legal_inf) if m and not e and f)
    n_struct = sum(1 for m, f in zip(has_main, legal_inf) if m and not f)
    report = {"schema": 1, "created_utc": now_utc(), "cohort_rows": S.n_val, "cohort_games": len(S.cohort_games),
              "wallet": "gold_est (prequential-v3)", "truth": "ward-stripped label_counts; empty -> no main purchase",
              "main_purchases": n_main, "legal_at_estimated_wallet": n_legal, "budget_blocked": n_budget,
              "structurally_blocked": n_struct, "share_budget_blocked": round(n_budget / n_main, 5),
              "share_structurally_blocked": round(n_struct / n_main, 5), "share_legal": round(n_legal / n_main, 5),
              "arms": {}}
    ident = [(r["match_id"], r["ts"], r.get("champion")) for r in S.val_rows]
    for name, path in ROWS.items():
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        assert [(r["match_id"], r["ts"], r.get("champion")) for r in recs] == ident, name
        hits_main = sum(r["first_action_exact"] for r, m in zip(recs, has_main) if m)
        hits_legal = sum(r["first_action_exact"] for r, m, e in zip(recs, has_main, legal_est) if m and e)
        hits_blocked = sum(r["first_action_exact"] for r, m, e in zip(recs, has_main, legal_est) if m and not e)
        hits_all = sum(r["first_action_exact"] for r in recs)
        any3_legal = [r["any_of_three_exact"] for r, m, e in zip(recs, has_main, legal_est) if m and e and r.get("any_of_three_exact") is not None]
        report["arms"][name] = {"first_exact_all": round(hits_all / S.n_val, 5),
                                "first_exact_main_purchases": round(hits_main / n_main, 5),
                                "first_exact_on_legal_mains": round(hits_legal / n_legal, 5),
                                "hits_on_blocked_mains": hits_blocked,
                                "any_of_three_on_legal_mains": round(sum(any3_legal) / max(len(any3_legal), 1), 5),
                                "n_legal": n_legal, "n_main": n_main}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    (OUT / "legality.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=1), flush=True)
    for k, v in report["arms"].items():
        print(f"{k:15s} all {v['first_exact_all']:.4f} | mains {v['first_exact_main_purchases']:.4f} | LEGAL mains {v['first_exact_on_legal_mains']:.4f} | top3 legal {v['any_of_three_on_legal_mains']:.4f} | hits on blocked {v['hits_on_blocked_mains']}", flush=True)
    print(f"report: {OUT / 'legality.json'}", flush=True)


if __name__ == "__main__":
    main()
