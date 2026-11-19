"""Frozen 50/50 of the NEW Bf (v4) with the OLD frozen mixture package.

Protocol: docs/experiment_newbf_mixture_v1.md (reviewer's step 2 after the
v4 stage-1 result, 2026-09-25). Inference only: no training, no weight
tuning, one pre-declared configuration, scored on the v4 forward cohort.

The old package's 5,692 ward-free groups are a SUBSET of the new 9,333
(reviewer check). Alignment is by canonical ward-stripped basket key, never
by index: every old group must map to exactly one new group, new-only
groups receive probability 0 from the old side, group 0 = "no main
purchase" on both sides. Both parents must reproduce their stage-1 persisted
rows with 0 mismatches before the mixture counts.

Run via footwear_runtime with PREFIX_ML_DIR = the v4 generation and
--review = the v4 snapshot's static review; [--smoke] caps rows.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import basket_key
from basket_wardfree import wardfree_record
from basket_bf_newgen import group_map, basket_tensors, decode, sha256_file

OUT_DIR = ROOT / "data" / "experiments" / "16.18-newbf-mixture-v1"
STAGE1 = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v4-bf-stage1"
WEIGHT_NEW = 0.5          # declared in the pre-registration, never tuned
WEIGHT_OLD = 0.5


def canon(pred):
    """JSON round-trips int basket keys to strings; compare canonically."""
    return "SAVE" if pred == "SAVE" else {str(k): int(v) for k, v in pred.items()}


def load_rows(path, n):
    with open(path, encoding="utf-8") as fh:
        out = []
        for line in fh:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def mismatches(recs, ref):
    assert len(ref) == len(recs)
    bad = {"pred_first": 0, "first_action_exact": 0, "any_of_three_exact": 0}
    for a, b in zip(recs, ref):
        assert (a["match_id"], a["ts"], a.get("champion")) == (b["match_id"], b["ts"], b.get("champion"))
        bad["pred_first"] += canon(a["pred_first"]) != canon(b["pred_first"])
        for k in ("first_action_exact", "any_of_three_exact"):
            bad[k] += a[k] != b[k]
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="mixture-50-50-v1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    schema = json.loads((generation / "split_manifest.json").read_text(encoding="utf-8"))["input_schema"]
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    config = torch.load(ref_artifact, map_location="cpu", weights_only=False)["config"]
    config = {**config, "max_items": schema["max_modeled_inventory_items"],
              "max_copies": schema["max_copies_of_one_item"],
              "max_basket_items": schema["max_basket_items"]}
    tp.apply_config(config)
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    tp.D_MODEL, tp.N_LAYERS, tp.FF_DIM, tp.N_HEADS = (int(config[k]) for k in ("d_model", "layers", "ff_dim", "heads"))
    assert not tp.USE_HISTORY and tp.D_MODEL == 64

    assignment = json.loads((generation / "split_assignment.json").read_text(encoding="utf-8"))
    train_set = {m for m, f in assignment.items() if f == "train"}
    val_set = {m for m, f in assignment.items() if f == "val"}
    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path, val_path = generation / "visits_train.jsonl", generation / "visits_val.jsonl"

    # ---- the new Bf's action space, rebuilt exactly as stage 1 built it ----
    champ_ids, item_ids, label_counts, catalogue, _, _ = v1.first_pass(train_path, train_set, None, None)
    champ_index, item_index = tp._index_map(champ_ids), tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(), key=lambda kv: (-kv[1], kv[0]))]
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gids, group_main = group_map(baskets)
    gid = gids.to(device)
    n_groups = len(group_main)
    cat_idx, cat_cnt, n_items = basket_tensors(baskets, n_cat, device)
    report1 = json.loads((STAGE1 / "bf_stage1_report_full.json").read_text(encoding="utf-8"))
    assert len(baskets) == report1["catalogue"] and n_groups == report1["groups"], "catalogue differs from stage 1"
    ckpt = STAGE1 / "bf_newgen_ep16_full.pt"
    ckpt_sha = sha256_file(ckpt)
    assert ckpt_sha == report1["checkpoint"]["sha256"], "checkpoint digest differs from the stage-1 report"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = v1.BasketNet(len(champ_index), len(item_index), len(label_index), len(item_index) + 1, n_items).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    # ---- cohort rows + tensors (new featurization, precomputed once) ----
    cap_val = 1500 if args.smoke else None
    val_rows = list(v1.stream_split(val_path, set(), False, champ_index, dragon, cap_val))
    n_val = len(val_rows)
    val_ds = tp.ShopDataset.from_stream(iter(val_rows), n_val, champ_index, item_index, label_index, dragon)
    if device.type == "cuda":
        val_ds.to_device(device)
    cohort_games = sorted({r["match_id"] for r in val_rows})
    assert set(cohort_games) <= val_set

    # ---- the old package and the old -> new alignment by canonical key ----
    import app.mixture_policy as mp
    policy = mp.load_package(args.package, dragon=dragon, device=str(device))
    assert policy.n_options == 3, "the common decode walks to 3 options"
    old_groups = [Counter(dict(g)) for g in policy.group_main]
    new_pos = {None: 0}
    for g in range(1, n_groups):
        k = basket_key(sorted(group_main[g].items()))
        assert k not in new_pos, "duplicate canonical key in the new space"
        new_pos[k] = g
    perm = torch.zeros(policy.n_act, dtype=torch.long)
    absent = []
    for g in range(1, policy.n_act):
        k = basket_key(sorted(old_groups[g].items()))
        if k in new_pos:
            perm[g] = new_pos[k]
        else:
            absent.append(g)
    assert not absent, f"{len(absent)} old groups are absent from the new space"
    assert len(set(perm.tolist())) == policy.n_act, "alignment is not injective"
    same_index = int((perm == torch.arange(policy.n_act)).sum())
    perm_d = perm.to(device)
    tp.apply_config(config)                    # the package re-points tp globals; ours back
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    print(f"setup listo ({time.monotonic() - started:.0f}s): cohorte {n_val} decisiones / {len(cohort_games)} partidas; "
          f"grupos nuevos {n_groups}, antiguos {policy.n_act} (todos alineados; {same_index} posiciones iguales)", flush=True)

    rec = {"new_bf": [], "old_mixture": [], "M": []}
    agree = Counter()
    t0 = time.monotonic()
    B = 256
    with torch.no_grad():
        vecs = model.baskets.vectors(cat_idx, cat_cnt, torch.arange(n_cat, device=device))
        for s in range(0, n_val, B):
            sl = slice(s, min(s + B, n_val))
            rows = val_rows[sl]
            batch = {k: v[sl] for k, v in {"champs": val_ds.champs, "sides": val_ds.sides, "items": val_ds.items,
                                            "query": val_ds.query, "inv": val_ds.inv, "hist": val_ds.hist}.items()}
            scores = model.scores(model.state(batch, device), vecs)
            b = scores.shape[0]
            gidx = gid.unsqueeze(0).expand(b, -1)
            gmax = torch.full((b, n_groups), float("-inf"), device=device)
            gmax.scatter_reduce_(1, gidx, scores, reduce="amax", include_self=True)
            gsum = torch.zeros(b, n_groups, device=device)
            gsum.scatter_add_(1, gidx, (scores - gmax.gather(1, gidx)).exp())
            logit_new = gmax + gsum.log()
            P_new = torch.softmax(logit_new, dim=1)
            # the package featurizes the rows itself, with its own config
            P_old_pkg = policy.probabilities(rows).to(device)
            tp.apply_config(config)
            tp.INVENTORY_ENCODING = "canonical-multiset-v1"
            P_old = torch.zeros_like(P_new)
            P_old.scatter_(1, perm_d.unsqueeze(0).expand(b, -1), P_old_pkg)
            P_M = WEIGHT_NEW * P_new + WEIGHT_OLD * P_old
            k_new = min(v1.SHORTLIST, n_groups)
            top_new = torch.topk(logit_new.cpu(), k_new, dim=1).indices          # stage-1 ordering
            top_old = torch.topk(P_old_pkg.cpu(), min(policy.shortlist, policy.n_act), dim=1).indices  # as options()
            top_M = torch.topk(P_M.cpu(), k_new, dim=1).indices
            for j, row in enumerate(rows):
                o_new = decode(top_new[j].tolist(), group_main, row, dragon, purchase_sequence)
                o_old = decode(top_old[j].tolist(), old_groups, row, dragon, purchase_sequence)
                o_m = decode(top_M[j].tolist(), group_main, row, dragon, purchase_sequence)
                rec["new_bf"].append(wardfree_record(row, o_new))
                rec["old_mixture"].append(wardfree_record(row, o_old))
                rec["M"].append(wardfree_record(row, o_m))
                firsts = {k: json.dumps(rec[k][-1]["pred_first"], sort_keys=True) for k in rec}
                agree["new_eq_old"] += firsts["new_bf"] == firsts["old_mixture"]
                agree["M_eq_new"] += firsts["M"] == firsts["new_bf"]
                agree["M_eq_old"] += firsts["M"] == firsts["old_mixture"]
                agree["M_novel"] += firsts["M"] != firsts["new_bf"] and firsts["M"] != firsts["old_mixture"]
            if (s // B) % 40 == 0:
                rate = (s + len(rows)) / max(time.monotonic() - t0, 1e-9)
                print(f"  {s + len(rows)}/{n_val} ({rate:.0f} filas/s, ETA {(n_val - s - len(rows)) / max(rate, 1e-9) / 60:.0f} min)", flush=True)

    ident = [(r["match_id"], r["ts"], r.get("champion")) for r in rec["new_bf"]]
    for k, recs in rec.items():
        assert [(r["match_id"], r["ts"], r.get("champion")) for r in recs] == ident, f"identidad desalineada: {k}"
        p2.dump(OUT_DIR / f"rows_{k}_{stamp}.jsonl", recs)

    # ---- integrity gate: both parents reproduce their stage-1 rows ----
    ref_new = load_rows(STAGE1 / "rows_new_bf_full.jsonl", n_val)
    ref_old = load_rows(STAGE1 / "rows_mixture_full.jsonl", n_val)
    ref_old_bf = load_rows(STAGE1 / "rows_old_bf_full.jsonl", n_val)
    repro = {"new_bf_vs_stage1_rows": mismatches(rec["new_bf"], ref_new),
             "old_mixture_vs_stage1_rows": mismatches(rec["old_mixture"], ref_old)}
    gate_ok = all(v == 0 for d in repro.values() for v in d.values())
    print("reproduccion:", json.dumps(repro), "| gate", "OK" if gate_ok else "FAILED", flush=True)
    assert [(r["match_id"], r["ts"], r.get("champion")) for r in ref_old_bf] == ident

    # ---- metrics ----
    subsets = {"single": lambda r: r["true_size"] == 1, "multi": lambda r: r["true_size"] > 1,
               "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
               "ward_only": lambda r: r.get("save_kind") == "ward_only"}
    arms = {**rec, "old_bf_persisted": ref_old_bf}
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/experiment_newbf_mixture_v1.md", "mode": stamp,
              "scope": "v4 forward development cohort (generation val; exposed by stage 1); no training; single declared configuration",
              "generation": generation.name, "cohort_rows": n_val, "cohort_games": len(cohort_games),
              "weights": {"new_bf": WEIGHT_NEW, "old_mixture": WEIGHT_OLD}, "weights_tuned": False,
              "alignment": {"new_groups": n_groups, "old_groups": policy.n_act, "old_subset_of_new": True,
                            "injective": True, "positions_unchanged": same_index, "rule": "canonical ward-stripped basket key"},
              "parents": {"new_bf": {"path": str(ckpt.relative_to(ROOT)), "sha256": ckpt_sha},
                          "old_mixture": {"package": args.package, "manifest_sha256": policy.digest}},
              "reproduction_mismatches": repro, "integrity_gate_passed": gate_ok,
              "first_recommendation_agreement": dict(agree),
              "aggregates": {k: p2.aggregate(v) for k, v in arms.items()},
              "paired": {}, "slices": {}}
    for x, y in (("M", "new_bf"), ("M", "old_mixture"), ("M", "old_bf_persisted"), ("new_bf", "old_mixture")):
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{y}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(arms[x], arms[y], metric)
        for label, pred in subsets.items():
            report["slices"][f"{x}_vs_{y}_{label}"] = p2.paired([r for r in arms[x] if pred(r)],
                                                                  [r for r in arms[y] if pred(r)], "first_action_exact")
    for x, y in (("M", "new_bf"), ("M", "old_mixture")):
        per = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
        for t, recs in (("x", arms[x]), ("y", arms[y])):
            for r in recs:
                per[t][r["match_id"]][0] += r["first_action_exact"]
                per[t][r["match_id"]][1] += 1
        report["paired"][f"{x}_vs_{y}"]["ci95_2000draws"] = list(v1.paired_bootstrap(per["x"], per["y"], seed=16, n=2000))
    moves = {}
    for y in ("new_bf", "old_mixture"):
        w = l = 0
        for m_, o in zip(rec["M"], rec[y]):
            w += m_["first_action_exact"] > o["first_action_exact"]
            l += m_["first_action_exact"] < o["first_action_exact"]
        moves[f"M_vs_{y}"] = {"corrected": w, "broken": l, "net": w - l}
    both_wrong_now_right = sum(1 for m_, a, b_ in zip(rec["M"], rec["new_bf"], rec["old_mixture"])
                               if m_["first_action_exact"] and not a["first_action_exact"] and not b_["first_action_exact"])
    both_right_now_wrong = sum(1 for m_, a, b_ in zip(rec["M"], rec["new_bf"], rec["old_mixture"])
                               if not m_["first_action_exact"] and a["first_action_exact"] and b_["first_action_exact"])
    union = sum(1 for a, b_ in zip(rec["new_bf"], rec["old_mixture"]) if a["first_action_exact"] or b_["first_action_exact"]) / max(n_val, 1)
    report["movement"] = {**moves, "both_parents_wrong_mixture_right": both_wrong_now_right,
                          "both_parents_right_mixture_wrong": both_right_now_wrong, "oracle_union_parents": union}
    prim = report["paired"]["M_vs_new_bf"]
    sec = report["paired"]["M_vs_old_mixture"]
    adopt = bool(gate_ok and prim["diff"] > 0 and prim["ci95"][0] > 0 and sec["diff"] > 0 and sec["ci95"][0] > 0)
    report["decision"] = {
        "rule": "M replaces new_bf as the practical reference only if M-new_bf > 0 with the 95% CI excluding 0 (and M > old_mixture likewise); integrity gate must pass",
        "M_vs_new_bf_points": round(prim["diff"] * 100, 3), "ci95_points": [round(x * 100, 3) for x in prim["ci95"]],
        "M_vs_old_mixture_points": round(sec["diff"] * 100, 3),
        "adopt_M_as_reference": adopt}
    report["product_summary"] = {
        "cohort": f"{n_val} decisions / {len(cohort_games)} games",
        "M_first_exact": report["aggregates"]["M"]["first_action_exact"]["value"],
        "new_bf_first_exact": report["aggregates"]["new_bf"]["first_action_exact"]["value"],
        "old_mixture_first_exact": report["aggregates"]["old_mixture"]["first_action_exact"]["value"],
        "M_vs_new_bf_points": round(prim["diff"] * 100, 3), "M_vs_new_bf_ci95": [round(x * 100, 3) for x in prim["ci95"]],
        "M_vs_new_bf_net_decisions": moves["M_vs_new_bf"]["net"],
        "M_vs_old_mixture_points": round(sec["diff"] * 100, 3), "M_vs_old_mixture_net_decisions": moves["M_vs_old_mixture"]["net"],
        "adopt_M_as_reference": adopt}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = OUT_DIR / f"mixture_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"product_summary": report["product_summary"], "movement": report["movement"],
                      "agreement": report["first_recommendation_agreement"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
