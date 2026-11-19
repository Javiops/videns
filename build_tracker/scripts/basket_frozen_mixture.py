"""Frozen 50/50 probability mixture of econ_ft_ep8 (grouped) + wardfree_ep16.

Protocol: docs/experiment_frozen_mixture_v1.md. No training, no weight
tuning, no VAL/test read. The two frozen models cover the SAME 5.692
main-purchase actions but index them differently (only 31 positions
coincide), so the mixture aligns them by canonical multiset key and
asserts the permutation is a bijection before averaging NORMALIZED
probabilities. Decode and legality are the common ones.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps rows.
"""
import argparse
import hashlib
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
from basket_vocabulary_probe import basket_key, membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR as FT_DIR
from basket_wardfree import (strip_counter, stripped_truth, wardfree_record,
                             OUT_DIR as WF_DIR)

OUT_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
WEIGHT_A = 0.5          # declared in the pre-registration, never tuned
WEIGHT_B = 0.5


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(order, group_main, row, dragon, purchase_sequence):
    """Common decode: walk `order` (best first) to 3 distinct legal mains."""
    budget = float(row["gold_est"])
    options = []
    calls = 0
    for act in order:
        if act == 0:
            options.append(("save", None))
        else:
            main = group_main[act]
            calls += 1
            if purchase_sequence(tuple(sorted(main.elements())), row, budget,
                                 dragon) is not None:
                options.append(("buy", Counter(main)))
        if len(options) >= 3:
            break
    if not options:
        options = [("save", None)]
    return options, calls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = "smoke" if args.smoke else "full"

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    schema = json.loads((generation / "split_manifest.json").read_text(
        encoding="utf-8"))["input_schema"]
    tp.MAX_ITEMS = schema["max_modeled_inventory_items"]
    tp.MAX_COPIES = schema["max_copies_of_one_item"]
    tp.MAX_BASKET_ITEMS = schema["max_basket_items"]
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    plan = json.loads((generation / "split_plan.json").read_text(encoding="utf-8"))
    fit_games, probe_games, _ = split_fit_probe(plan)
    assert membership_digest(fit_games) == v1.FROZEN_FIT_SHA
    assert membership_digest(probe_games) == v1.FROZEN_PROBE_SHA
    fit_set = set(fit_games)

    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"
    champ_ids, item_ids, label_counts, catalogue, _, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets_A = [k for k, _ in sorted(catalogue.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets_A).encode()).hexdigest()
    n_cat_A = len(baskets_A) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- A's action space: ward-stripped groups over the full catalogue ----
    group_of_key = {None: 0}
    group_main = [Counter()]                    # 0 = no main purchase
    group_ids = torch.zeros(n_cat_A, dtype=torch.long)
    for cid, bkey in enumerate(baskets_A, start=1):
        stripped = strip_counter(Counter(
            {int(part.split(":")[0]): int(part.split(":")[1])
             for part in bkey.split(",")}))
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = len(group_main)
            group_main.append(stripped)
        group_ids[cid] = group_of_key[skey]
    n_act = len(group_main)

    # ---- B's action space: ward-free catalogue rebuilt from the FIT stream ----
    wf_cat = Counter()
    for row in v1.stream_split(train_path, fit_set, True, champ_index,
                               dragon, None):
        t = stripped_truth(row)
        if t:
            wf_cat[basket_key(sorted(t.items()))] += 1
    baskets_B = [k for k, _ in sorted(wf_cat.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
    n_cat_B = len(baskets_B) + 1
    assert n_cat_B == n_act, f"espacios distintos: {n_cat_B} vs {n_act}"

    # ---- explicit permutation B -> A by canonical key (never by index) ----
    keys_A = [None] + [basket_key(sorted(group_main[g].items()))
                       for g in range(1, n_act)]
    keys_B = [None] + baskets_B
    assert set(keys_A) == set(keys_B), "los conjuntos de acciones difieren"
    pos_A = {k: i for i, k in enumerate(keys_A)}
    assert len(pos_A) == n_act, "claves canonicas duplicadas en A"
    perm_B_to_A = torch.tensor([pos_A[k] for k in keys_B], dtype=torch.long)
    assert len(set(perm_B_to_A.tolist())) == n_act, "la permutacion no es biyectiva"
    same_index = int((perm_B_to_A == torch.arange(n_act)).sum())
    print(f"acciones {n_act}; permutacion biyectiva; "
          f"posiciones que coinciden: {same_index}", flush=True)

    # ---- probe tensors, econ cache, domain ----
    cache_report = json.loads(
        (CACHE_DIR / "cache_report_full.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    probe = torch.load(CACHE_DIR / "frozen_probe_full.pt", weights_only=True)
    econ_probe = torch.load(CACHE_DIR / "econ_probe_full.pt", weights_only=True)
    n_probe = probe["n_rows"]
    probe_domain = probe["top_idx"].long()

    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas_fit = econ_fit[:, :, 1] > 0
    stats_mean, stats_std = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas_fit].float()
    for d in range(2, N_ECON):
        stats_mean[d] = flat[:, d].mean()
        stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = stats_mean.to(device), stats_std.to(device)
    del econ_fit, flat, feas_fit

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index,
                                         dragon)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, None))
    digest = hashlib.sha256()
    for row in probe_rows:
        digest.update(f"{row['match_id']}|{row['ts']};".encode())
    assert digest.hexdigest() == probe["rows_sha256"], "probe desalineado"
    if args.smoke:
        n_probe = min(2000, n_probe)
        probe_rows = probe_rows[:n_probe]
        for name in ("champs", "sides", "items", "query", "inv", "hist"):
            setattr(probe_ds, name, getattr(probe_ds, name)[:n_probe])
    if device.type == "cuda":
        probe_ds.to_device(device)

    def basket_tensors(baskets, n_cat):
        cat_items = sorted({i for b in baskets for part in b.split(",")
                            for i in [int(part.split(":")[0])]})
        idx_map = {it: k + 1 for k, it in enumerate(cat_items)}
        cat_idx = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
        cat_cnt = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
        for row_i, b in enumerate(baskets, start=1):
            for slot, part in enumerate(b.split(",")[:v1.MAX_BASKET]):
                i, n_ = part.split(":")
                cat_idx[row_i, slot] = idx_map[int(i)]
                cat_cnt[row_i, slot] = min(int(n_), v1.MAX_BASKET)
        return cat_idx.to(device), cat_cnt.to(device), len(cat_items)

    cat_idx_A, cat_cnt_A, n_items_A = basket_tensors(baskets_A, n_cat_A)
    cat_idx_B, cat_cnt_B, n_items_B = basket_tensors(baskets_B, n_cat_B)

    # ---- frozen models ----
    ck_A = FT_DIR / "econ_ft_ep8.pt"
    ck_B = WF_DIR / "wardfree_ep16.pt"
    blob_A = torch.load(ck_A, map_location="cpu", weights_only=False)
    model_A = TunedNet(v1.BasketNet(len(champ_index), len(item_index),
                                    len(label_index), len(item_index) + 1,
                                    n_items_A).to(device),
                       CorrectionNet().to(device))
    model_A.load_state_dict(blob_A["state_dict"])
    model_A.eval()
    blob_B = torch.load(ck_B, map_location="cpu", weights_only=False)
    model_B = v1.BasketNet(len(champ_index), len(item_index), len(label_index),
                           len(item_index) + 1, n_items_B).to(device)
    model_B.load_state_dict(blob_B["state_dict"])
    model_B.eval()
    gid = group_ids.to(device)
    perm = perm_B_to_A.to(device)
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    rec = {"A": [], "B": [], "M": []}
    calls = 0
    agree = Counter()
    with torch.no_grad():
        vecs_B = model_B.baskets.vectors(
            cat_idx_B, cat_cnt_B, torch.arange(n_cat_B, device=device))
        for s in range(0, n_probe, 512):
            sl = slice(s, min(s + 512, n_probe))
            batch = {k: v[sl] for k, v in {
                "champs": probe_ds.champs, "sides": probe_ds.sides,
                "items": probe_ds.items, "query": probe_ds.query,
                "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
            b = batch["champs"].shape[0]
            # --- A: full catalogue -> ward-marginalized group log-probs ---
            dom = probe_domain[sl].to(device)
            econ = ((econ_probe[sl].float().to(device) - sm) / ss)
            scores_A, _ = model_A.corrected_scores(
                batch, device, cat_idx_A, cat_cnt_A, n_cat_A, dom, econ)
            gidx = gid.unsqueeze(0).expand(b, -1)
            gmax = torch.full((b, n_act), float("-inf"), device=device)
            gmax.scatter_reduce_(1, gidx, scores_A, reduce="amax",
                                 include_self=True)
            gsum = torch.zeros(b, n_act, device=device)
            gsum.scatter_add_(1, gidx, (scores_A - gmax.gather(1, gidx)).exp())
            logit_A = gmax + gsum.log()
            P_A = torch.softmax(logit_A, dim=1)
            # --- B: ward-free catalogue, permuted into A's action order ---
            scores_B = model_B.scores(model_B.state(batch, device), vecs_B)
            P_B = torch.zeros_like(P_A)
            P_B.scatter_(1, perm.unsqueeze(0).expand(b, -1),
                         torch.softmax(scores_B, dim=1))
            P_M = WEIGHT_A * P_A + WEIGHT_B * P_B
            tops = {k: torch.topk(v, min(v1.SHORTLIST, n_act), dim=1)
                    .indices.cpu() for k, v in
                    (("A", P_A), ("B", P_B), ("M", P_M))}
            for j in range(b):
                row = probe_rows[s + j]
                firsts = {}
                for arm in ("A", "B", "M"):
                    opts, c = decode(tops[arm][j].tolist(), group_main, row,
                                     dragon, purchase_sequence)
                    calls += c
                    rec[arm].append(wardfree_record(row, opts))
                    firsts[arm] = json.dumps(rec[arm][-1]["pred_first"],
                                             sort_keys=True)
                agree["A_eq_B"] += firsts["A"] == firsts["B"]
                agree["M_eq_A"] += firsts["M"] == firsts["A"]
                agree["M_eq_B"] += firsts["M"] == firsts["B"]
                agree["M_novel"] += (firsts["M"] != firsts["A"]
                                     and firsts["M"] != firsts["B"])
    for arm in ("A", "B", "M"):
        p2.dump(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", rec[arm])

    # ---- integrity: A and B must reproduce their persisted rows ----
    def canon(pred):
        """JSON round-trips int basket keys to strings; compare canonically."""
        return ("SAVE" if pred == "SAVE"
                else {str(k): int(v) for k, v in pred.items()})

    def mismatches(recs, path):
        if not Path(path).exists():
            return None
        ref = [json.loads(l) for l in open(path, encoding="utf-8")][:len(recs)]
        assert len(ref) == len(recs)
        bad = {"first_action_exact": 0, "pred_first": 0,
               "any_of_three_exact": 0}
        for a, b_ in zip(recs, ref):
            assert (a["match_id"], a["ts"]) == (b_["match_id"], b_["ts"])
            bad["pred_first"] += canon(a["pred_first"]) != canon(b_["pred_first"])
            for k in ("first_action_exact", "any_of_three_exact"):
                bad[k] += a[k] != b_[k]
        return bad

    repro = {
        "A_vs_rows_grouped_econft_full": mismatches(
            rec["A"], WF_DIR / "rows_grouped_econft_full.jsonl"),
        "B_vs_rows_wardfree_full": mismatches(
            rec["B"], WF_DIR / "rows_wardfree_full.jsonl"),
    }
    print("reproduccion:", json.dumps(repro), flush=True)

    # ---- metrics ----
    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/experiment_frozen_mixture_v1.md",
        "scope": "internal chronological TRAIN probe (already exposed); "
                 "no VAL, no test, no training",
        "mode": stamp,
        "weights": {"A_grouped_econft": WEIGHT_A, "B_wardfree": WEIGHT_B},
        "weights_tuned": False,
        "n_actions": n_act, "permutation_positions_unchanged": same_index,
        "permutation_is_bijection": True,
        "lazy_legality_calls": calls,
        "checkpoints": {"A": {"path": str(ck_A.relative_to(ROOT)),
                              "sha256": sha256_file(ck_A)},
                        "B": {"path": str(ck_B.relative_to(ROOT)),
                              "sha256": sha256_file(ck_B)}},
        "reproduction_mismatches": repro,
        "first_recommendation_agreement": dict(agree),
        "aggregates": {k: p2.aggregate(v) for k, v in rec.items()},
        "paired": {}, "slices": {},
    }
    for x, y in (("M", "A"), ("M", "B"), ("A", "B")):
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{y}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(rec[x], rec[y], metric)
        for label, pred in subsets.items():
            report["slices"][f"{x}_vs_{y}_{label}"] = p2.paired(
                [r for r in rec[x] if pred(r)],
                [r for r in rec[y] if pred(r)], "first_action_exact")

    # primary also with the reviewer's 2000-draw convention
    for x, y in (("M", "A"), ("M", "B")):
        per = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
        for tag, recs in (("x", rec[x]), ("y", rec[y])):
            for r in recs:
                per[tag][r["match_id"]][0] += r["first_action_exact"]
                per[tag][r["match_id"]][1] += 1
        ci = v1.paired_bootstrap(per["x"], per["y"], seed=16, n=2000)
        report["paired"][f"{x}_vs_{y}"]["ci95_2000draws"] = list(ci)

    # where the mixture moved: resolved vs broken, against each parent
    moves = {}
    for y in ("A", "B"):
        w = l = 0
        for m_, o in zip(rec["M"], rec[y]):
            w += m_["first_action_exact"] > o["first_action_exact"]
            l += m_["first_action_exact"] < o["first_action_exact"]
        moves[f"M_vs_{y}"] = {"corrected": w, "broken": l, "net": w - l}
    both_wrong_now_right = sum(
        1 for m_, a, b_ in zip(rec["M"], rec["A"], rec["B"])
        if m_["first_action_exact"] and not a["first_action_exact"]
        and not b_["first_action_exact"])
    both_right_now_wrong = sum(
        1 for m_, a, b_ in zip(rec["M"], rec["A"], rec["B"])
        if not m_["first_action_exact"] and a["first_action_exact"]
        and b_["first_action_exact"])
    report["movement"] = {
        **moves,
        "both_parents_wrong_mixture_right": both_wrong_now_right,
        "both_parents_right_mixture_wrong": both_right_now_wrong,
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"mixture_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": report["aggregates"],
                      "paired": report["paired"],
                      "movement": report["movement"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
