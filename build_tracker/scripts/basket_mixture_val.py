"""Frozen 50/50 mixture on VAL, paired against both parents, + latency.

Protocol: docs/experiment_mixture_val_v1.md. Configuration frozen and
bound by digest in
data/experiments/16.18-frozen-mixture-v1/frozen_mixture_v1.json. Three
arms (A grouped econ, B ward-free, M mixture) scored on the same VAL rows
in one pass, identical decode, paired by decision. Everything computed
fresh -- no cache covers VAL. Test is NOT touched.

Integrity gates: the alignment permutation must be a bijection leaving 31
positions unchanged, and arm A must reproduce the previous VAL run's rows.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps rows.
"""
import argparse
import hashlib
import json
import os
import statistics
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
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR, econ_features, to_action
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR as FT_DIR
from basket_wardfree import strip_counter, stripped_truth, wardfree_record

OUT_DIR = ROOT / "data" / "experiments" / "16.18-mixture-val-v1"
MIX_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
WF_DIR = ROOT / "data" / "experiments" / "16.18-wardfree-v1"
V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
VAL_REF = (ROOT / "data" / "experiments" / "16.18-val-window-check-v1"
           / "rows_val_baseline_full.jsonl")
LATENCY_SAMPLE = 200


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game_ci(records, metric):
    per_game, zeros = {}, {}
    for r in records:
        if r.get(metric) is None:
            continue
        e = per_game.setdefault(r["match_id"], [0.0, 0])
        e[0] += r[metric]
        e[1] += 1
        zeros[r["match_id"]] = [0.0, 0]
    return list(v1.paired_bootstrap(per_game, zeros))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((MIX_DIR / "frozen_mixture_v1.json").read_text(
        encoding="utf-8"))
    wA = frozen["weights"]["A"]
    wB = frozen["weights"]["B"]
    assert not frozen["weights_tuned"]
    for arm in ("arm_A_grouped_econ", "arm_B_wardfree"):
        for k, spec in frozen["components"][arm].items():
            if k == "role":
                continue
            assert sha256_file(ROOT / spec["path"]) == spec["sha256"], \
                f"digest cambiado: {spec['path']}"
    print("manifiesto verificado: todos los componentes casan por digest",
          flush=True)

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
    val_path = generation / "visits_val.jsonl"
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

    # ---- A's grouped action space ----
    group_of_key = {None: 0}
    group_main = [Counter()]
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

    # ---- B's ward-free catalogue, rebuilt from the same FIT stream ----
    wf_cat = Counter()
    for row in v1.stream_split(train_path, fit_set, True, champ_index,
                               dragon, None):
        t = stripped_truth(row)
        if t:
            wf_cat[basket_key(sorted(t.items()))] += 1
    baskets_B = [k for k, _ in sorted(wf_cat.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
    n_cat_B = len(baskets_B) + 1
    assert n_cat_B == n_act

    keys_A = [None] + [basket_key(sorted(group_main[g].items()))
                       for g in range(1, n_act)]
    keys_B = [None] + baskets_B
    assert set(keys_A) == set(keys_B)
    pos_A = {k: i for i, k in enumerate(keys_A)}
    assert len(pos_A) == n_act
    perm_B_to_A = torch.tensor([pos_A[k] for k in keys_B], dtype=torch.long)
    assert len(set(perm_B_to_A.tolist())) == n_act, "permutacion no biyectiva"
    same_index = int((perm_B_to_A == torch.arange(n_act)).sum())
    assert same_index == frozen["alignment"]["positions_unchanged"], same_index
    print(f"alineacion verificada: {n_act} acciones, biyectiva, "
          f"{same_index} posiciones intactas", flush=True)

    # ---- FIT normalization stats (the frozen ones) ----
    cache_report = json.loads(
        (CACHE_DIR / "cache_report_full.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas_fit = econ_fit[:, :, 1] > 0
    stats_mean, stats_std = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas_fit].float()
    for d in range(2, N_ECON):
        stats_mean[d] = flat[:, d].mean()
        stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = stats_mean.to(device), stats_std.to(device)
    del econ_fit, flat, feas_fit

    def basket_tensors(baskets, n_cat):
        items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
        idx_map = {it: k + 1 for k, it in enumerate(items)}
        ci = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
        cc = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
        for row_i, b in enumerate(baskets, start=1):
            for slot, part in enumerate(b.split(",")[:v1.MAX_BASKET]):
                i, n_ = part.split(":")
                ci[row_i, slot] = idx_map[int(i)]
                cc[row_i, slot] = min(int(n_), v1.MAX_BASKET)
        return ci.to(device), cc.to(device), len(items)

    cat_idx_A, cat_cnt_A, n_items_A = basket_tensors(baskets_A, n_cat_A)
    cat_idx_B, cat_cnt_B, n_items_B = basket_tensors(baskets_B, n_cat_B)

    def arch(n_items):
        return (len(champ_index), len(item_index), len(label_index),
                len(item_index) + 1, n_items)

    parent = v1.BasketNet(*arch(n_items_A)).to(device)
    parent.load_state_dict(torch.load(V2_DIR / "Bf_ep16.pt", map_location="cpu",
                                      weights_only=False)["state_dict"])
    parent.eval()
    tuned = TunedNet(v1.BasketNet(*arch(n_items_A)).to(device),
                     CorrectionNet().to(device))
    tuned.load_state_dict(torch.load(FT_DIR / "econ_ft_ep8.pt",
                                     map_location="cpu",
                                     weights_only=False)["state_dict"])
    tuned.eval()
    wardfree = v1.BasketNet(*arch(n_items_B)).to(device)
    wardfree.load_state_dict(torch.load(WF_DIR / "wardfree_ep16.pt",
                                        map_location="cpu",
                                        weights_only=False)["state_dict"])
    wardfree.eval()

    cap = 2000 if args.smoke else None
    val_rows = list(v1.stream_split(val_path, set(), False, champ_index,
                                    dragon, cap))
    n_val = len(val_rows)
    val_ds = tp.ShopDataset.from_stream(iter(val_rows), n_val, champ_index,
                                        item_index, label_index, dragon)
    if device.type == "cuda":
        val_ds.to_device(device)
    gid = group_ids.to(device)
    perm = perm_B_to_A.to(device)
    all_cat_A = torch.arange(n_cat_A, device=device)
    all_cat_B = torch.arange(n_cat_B, device=device)
    games = len({r["match_id"] for r in val_rows})
    print(f"val: {n_val} decisiones / {games} partidas "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    def decode(order, row):
        budget = float(row["gold_est"])
        options = []
        for act in order:
            if act == 0:
                options.append(("save", None))
            else:
                main = group_main[act]
                if purchase_sequence(tuple(sorted(main.elements())), row,
                                     budget, dragon) is not None:
                    options.append(("buy", Counter(main)))
            if len(options) >= 3:
                break
        return options or [("save", None)]

    rec = {"A": [], "B": [], "M": []}
    counters = Counter()
    agree = Counter()
    with torch.no_grad():
        pvecs = parent.baskets.vectors(cat_idx_A, cat_cnt_A, all_cat_A)
        wvecs = wardfree.baskets.vectors(cat_idx_B, cat_cnt_B, all_cat_B)
        for s in range(0, n_val, 512):
            sl = slice(s, min(s + 512, n_val))
            batch = {k: v[sl] for k, v in {
                "champs": val_ds.champs, "sides": val_ds.sides,
                "items": val_ds.items, "query": val_ds.query,
                "inv": val_ds.inv, "hist": val_ds.hist}.items()}
            b = batch["champs"].shape[0]
            pscores = parent.scores(parent.state(batch, device), pvecs)
            dom = torch.topk(pscores, v1.SHORTLIST, dim=1).indices
            dom_cpu = dom.cpu()
            econ = torch.zeros((b, v1.SHORTLIST, N_ECON))
            for j in range(b):
                row = val_rows[s + j]
                budget = float(row["gold_est"])
                for k in range(v1.SHORTLIST):
                    cand = int(dom_cpu[j, k])
                    if cand == 0:
                        econ[j, k, 0] = 1.0
                    else:
                        econ[j, k] = torch.tensor(econ_features(
                            to_action(baskets_A[cand - 1]), row, budget,
                            dragon, counters))
            econ_n = (econ.to(device) - sm) / ss
            scores_A, _ = tuned.corrected_scores(batch, device, cat_idx_A,
                                                 cat_cnt_A, n_cat_A, dom,
                                                 econ_n)
            gidx = gid.unsqueeze(0).expand(b, -1)
            gmax = torch.full((b, n_act), float("-inf"), device=device)
            gmax.scatter_reduce_(1, gidx, scores_A, reduce="amax",
                                 include_self=True)
            gsum = torch.zeros(b, n_act, device=device)
            gsum.scatter_add_(1, gidx, (scores_A - gmax.gather(1, gidx)).exp())
            P_A = torch.softmax(gmax + gsum.log(), dim=1)
            scores_B = wardfree.scores(wardfree.state(batch, device), wvecs)
            P_B = torch.zeros_like(P_A)
            P_B.scatter_(1, perm.unsqueeze(0).expand(b, -1),
                         torch.softmax(scores_B, dim=1))
            P_M = wA * P_A + wB * P_B
            tops = {k: torch.topk(v, min(v1.SHORTLIST, n_act), dim=1)
                    .indices.cpu() for k, v in
                    (("A", P_A), ("B", P_B), ("M", P_M))}
            for j in range(b):
                row = val_rows[s + j]
                firsts = {}
                for armn in ("A", "B", "M"):
                    r = wardfree_record(row, decode(tops[armn][j].tolist(), row))
                    rec[armn].append(r)
                    firsts[armn] = json.dumps(r["pred_first"], sort_keys=True)
                agree["A_eq_B"] += firsts["A"] == firsts["B"]
                agree["M_eq_A"] += firsts["M"] == firsts["A"]
                agree["M_eq_B"] += firsts["M"] == firsts["B"]
                agree["M_novel"] += (firsts["M"] != firsts["A"]
                                     and firsts["M"] != firsts["B"])
            if (s // 512) % 20 == 0:
                rate = (s + b) / max(time.monotonic() - started, 1)
                print(f"  {s + b}/{n_val} ({rate:.0f} filas/s, ETA "
                      f"{(n_val - s - b) / max(rate, 1) / 60:.0f} min)",
                      flush=True)
    for armn in ("A", "B", "M"):
        p2.dump(OUT_DIR / f"rows_val_{armn}_{stamp}.jsonl", rec[armn])

    # ---- integrity gate: arm A must reproduce the earlier VAL rows ----
    def canon(pred):
        return ("SAVE" if pred == "SAVE"
                else {str(k): int(v) for k, v in pred.items()})

    repro = None
    if VAL_REF.exists():
        ref = [json.loads(l) for l in open(VAL_REF, encoding="utf-8")]
        ref = ref[:len(rec["A"])]
        bad = {"first_action_exact": 0, "pred_first": 0,
               "any_of_three_exact": 0}
        aligned = len(ref) == len(rec["A"])
        if aligned:
            for a, b_ in zip(rec["A"], ref):
                aligned &= (a["match_id"], a["ts"]) == (b_["match_id"], b_["ts"])
                bad["pred_first"] += canon(a["pred_first"]) != canon(b_["pred_first"])
                for k in ("first_action_exact", "any_of_three_exact"):
                    bad[k] += a[k] != b_[k]
        repro = {"rows_compared": len(ref), "row_identity_aligned": aligned,
                 "mismatches": bad, "reference": str(VAL_REF.relative_to(ROOT))}
        print("reproduccion arm A vs VAL previa:", json.dumps(repro), flush=True)

    # ---------------------------- latency ----------------------------
    lat = {}
    sample = min(LATENCY_SAMPLE, n_val)
    stages = defaultdict(list)
    sync = (torch.cuda.synchronize if device.type == "cuda" else (lambda: None))

    def clock():
        sync()
        return time.perf_counter()

    with torch.no_grad():
        for j in range(min(10, sample)):          # warm-up, not recorded
            one = {k: v[j:j + 1] for k, v in {
                "champs": val_ds.champs, "sides": val_ds.sides,
                "items": val_ds.items, "query": val_ds.query,
                "inv": val_ds.inv, "hist": val_ds.hist}.items()}
            parent.scores(parent.state(one, device), pvecs)
            wardfree.scores(wardfree.state(one, device), wvecs)
        for j in range(sample):
            row = val_rows[j]
            budget = float(row["gold_est"])
            one = {k: v[j:j + 1] for k, v in {
                "champs": val_ds.champs, "sides": val_ds.sides,
                "items": val_ds.items, "query": val_ds.query,
                "inv": val_ds.inv, "hist": val_ds.hist}.items()}
            t0 = clock()
            ps = parent.scores(parent.state(one, device), pvecs)
            d1 = torch.topk(ps, v1.SHORTLIST, dim=1).indices
            t1 = clock()
            d_cpu = d1.cpu()
            e = torch.zeros((1, v1.SHORTLIST, N_ECON))
            for k in range(v1.SHORTLIST):
                cand = int(d_cpu[0, k])
                if cand == 0:
                    e[0, k, 0] = 1.0
                else:
                    e[0, k] = torch.tensor(econ_features(
                        to_action(baskets_A[cand - 1]), row, budget, dragon,
                        Counter()))
            t2 = clock()
            sA, _ = tuned.corrected_scores(one, device, cat_idx_A, cat_cnt_A,
                                           n_cat_A, d1, (e.to(device) - sm) / ss)
            t3 = clock()
            sB = wardfree.scores(wardfree.state(one, device), wvecs)
            t4 = clock()
            gi1 = gid.unsqueeze(0)
            gm = torch.full((1, n_act), float("-inf"), device=device)
            gm.scatter_reduce_(1, gi1, sA, reduce="amax", include_self=True)
            gs = torch.zeros(1, n_act, device=device)
            gs.scatter_add_(1, gi1, (sA - gm.gather(1, gi1)).exp())
            pa = torch.softmax(gm + gs.log(), dim=1)
            pb = torch.zeros_like(pa)
            pb.scatter_(1, perm.unsqueeze(0), torch.softmax(sB, dim=1))
            order = torch.topk(wA * pa + wB * pb,
                               min(v1.SHORTLIST, n_act), dim=1).indices.cpu()
            t5 = clock()
            decode(order[0].tolist(), row)
            t6 = clock()
            for name, dt in (("1_parent_domain", t1 - t0),
                             ("2_econ_features", t2 - t1),
                             ("3_tuned_scores", t3 - t2),
                             ("4_wardfree_scores", t4 - t3),
                             ("5_group_mix_rank", t5 - t4),
                             ("6_decode_legality", t6 - t5)):
                stages[name].append(dt * 1000.0)
            stages["total_mixture"].append((t6 - t0) * 1000.0)
            stages["total_single_model_A"].append(
                ((t3 - t0) + (t6 - t4)) * 1000.0)
    for name, xs in stages.items():
        xs.sort()
        lat[name] = {"median_ms": round(statistics.median(xs), 3),
                     "p90_ms": round(xs[int(0.9 * (len(xs) - 1))], 3),
                     "mean_ms": round(statistics.fmean(xs), 3)}
    lat["_meta"] = {"n_decisions": sample, "batch_size": 1,
                    "device": (torch.cuda.get_device_name(0)
                               if device.type == "cuda" else "cpu"),
                    "warmup_decisions": min(10, sample),
                    "note": "total_single_model_A excludes stage 4 "
                            "(the ward-free forward); the mixture's added "
                            "cost is the measured difference"}
    print("latencia:", json.dumps(
        {k: v["median_ms"] for k, v in lat.items() if k != "_meta"}), flush=True)

    # ---------------------------- metrics ----------------------------
    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/experiment_mixture_val_v1.md",
        "frozen_config": "data/experiments/16.18-frozen-mixture-v1/"
                         "frozen_mixture_v1.json",
        "manifest_digests_verified": True,
        "split": "val (frozen temporal, later window) - PREVIOUSLY EXPOSED "
                 "once for arm A; not a blind confirmation",
        "mode": stamp, "n_decisions": n_val, "n_games": games,
        "weights": {"A": wA, "B": wB}, "weights_tuned": False,
        "n_actions": n_act, "permutation_positions_unchanged": same_index,
        "arm_A_reproduction_vs_previous_val": repro,
        "first_recommendation_agreement": dict(agree),
        "feature_counters": dict(counters),
        "aggregates": {}, "ci": {}, "paired": {}, "slices": {},
        "latency_ms": lat,
    }
    for armn in ("A", "B", "M"):
        report["aggregates"][armn] = p2.aggregate(rec[armn])
        report["ci"][armn] = {
            m: game_ci(rec[armn], m)
            for m in ("first_action_exact", "any_of_three_exact")}
    for x, y in (("M", "A"), ("M", "B"), ("A", "B")):
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{y}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(rec[x], rec[y], metric)
        for label, pred in subsets.items():
            report["slices"][f"{x}_vs_{y}_{label}"] = p2.paired(
                [r for r in rec[x] if pred(r)],
                [r for r in rec[y] if pred(r)], "first_action_exact")
    for x, y in (("M", "A"), ("M", "B")):
        per = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
        for tag, recs in (("x", rec[x]), ("y", rec[y])):
            for r in recs:
                per[tag][r["match_id"]][0] += r["first_action_exact"]
                per[tag][r["match_id"]][1] += 1
        report["paired"][f"{x}_vs_{y}"]["ci95_2000draws"] = list(
            v1.paired_bootstrap(per["x"], per["y"], seed=16, n=2000))
    moves = {}
    for y in ("A", "B"):
        w = l = 0
        for m_, o in zip(rec["M"], rec[y]):
            w += m_["first_action_exact"] > o["first_action_exact"]
            l += m_["first_action_exact"] < o["first_action_exact"]
        moves[f"M_vs_{y}"] = {"corrected": w, "broken": l, "net": w - l}
    moves["both_parents_wrong_mixture_right"] = sum(
        1 for m_, a, b_ in zip(rec["M"], rec["A"], rec["B"])
        if m_["first_action_exact"] and not a["first_action_exact"]
        and not b_["first_action_exact"])
    moves["both_parents_right_mixture_wrong"] = sum(
        1 for m_, a, b_ in zip(rec["M"], rec["A"], rec["B"])
        if not m_["first_action_exact"] and a["first_action_exact"]
        and b_["first_action_exact"])
    union = sum(1 for a, b_ in zip(rec["A"], rec["B"])
                if a["first_action_exact"] or b_["first_action_exact"])
    moves["oracle_union_rate"] = union / max(n_val, 1)
    report["movement"] = moves
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"mixture_val_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": report["aggregates"],
                      "paired": report["paired"],
                      "movement": report["movement"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
