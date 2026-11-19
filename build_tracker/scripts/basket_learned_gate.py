"""Learned per-decision gate over the two frozen models, vs the fixed 50/50.

Protocol: docs/experiment_learned_gate_v1.md. The parents (econ_ft_ep8
grouped, wardfree_ep16) are frozen and bound by the frozen mixture's
digests. A tiny gate maps 8 decision-time features to w in (0,1);
P = w*P_A + (1-w)*P_B. Control: the same gate with inputs zeroed (learns a
constant). Reference: the fixed 50/50 mixture's persisted rows.

Trained OUT-OF-FOLD ON THE PROBE (5 folds by match): the parents were
trained on FIT, so their FIT probabilities are overconfident and a gate
trained there would learn which parent overfits more. The probe is also
the selection set; the result is an out-of-fold estimate on it.

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
import torch.nn as nn

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import basket_key, membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR as FT_DIR
from basket_wardfree import (strip_counter, stripped_truth, wardfree_record,
                             OUT_DIR as WF_DIR)

OUT_DIR = ROOT / "data" / "experiments" / "16.18-learned-gate-v1"
MIX_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
FOLDS = 5
STEPS = 300
LR = 1e-2
N_FEAT = 8


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((MIX_DIR / "frozen_mixture_v1.json").read_text(encoding="utf-8"))
    for arm in ("arm_A_grouped_econ", "arm_B_wardfree"):
        for k, spec in frozen["components"][arm].items():
            if k != "role":
                got = hashlib.sha256((ROOT / spec["path"]).read_bytes()).hexdigest()
                assert got == spec["sha256"], f"digest cambiado: {spec['path']}"

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
    baskets_A = [k for k, _ in sorted(catalogue.items(), key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets_A).encode()).hexdigest()
    n_cat_A = len(baskets_A) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group_of_key = {None: 0}
    group_main = [Counter()]
    group_ids = torch.zeros(n_cat_A, dtype=torch.long)
    for cid, bkey in enumerate(baskets_A, start=1):
        stripped = strip_counter(Counter(
            {int(p.split(":")[0]): int(p.split(":")[1]) for p in bkey.split(",")}))
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = len(group_main)
            group_main.append(stripped)
        group_ids[cid] = group_of_key[skey]
    n_act = len(group_main)
    wf_cat = Counter()
    for row in v1.stream_split(train_path, fit_set, True, champ_index, dragon, None):
        t = stripped_truth(row)
        if t:
            wf_cat[basket_key(sorted(t.items()))] += 1
    baskets_B = [k for k, _ in sorted(wf_cat.items(), key=lambda kv: (-kv[1], kv[0]))]
    assert len(baskets_B) + 1 == n_act
    keys_A = [None] + [basket_key(sorted(group_main[g].items())) for g in range(1, n_act)]
    pos_A = {k: i for i, k in enumerate(keys_A)}
    perm = torch.tensor([pos_A[k] for k in [None] + baskets_B], dtype=torch.long)
    assert len(set(perm.tolist())) == n_act
    assert int((perm == torch.arange(n_act)).sum()) == frozen["alignment"]["positions_unchanged"]

    cache_report = json.loads((CACHE_DIR / "cache_report_full.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    probe = torch.load(CACHE_DIR / "frozen_probe_full.pt", weights_only=True)
    econ_probe = torch.load(CACHE_DIR / "econ_probe_full.pt", weights_only=True)
    n_probe = probe["n_rows"]
    probe_domain = probe["top_idx"].long()
    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas = econ_fit[:, :, 1] > 0
    sm_, ss_ = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    for d in range(2, N_ECON):
        sm_[d] = flat[:, d].mean()
        ss_[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = sm_.to(device), ss_.to(device)
    del econ_fit, flat, feas

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index, dragon)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index, dragon, None))
    digest = hashlib.sha256()
    for row in probe_rows:
        digest.update(f"{row['match_id']}|{row['ts']};".encode())
    assert digest.hexdigest() == probe["rows_sha256"]
    if args.smoke:
        n_probe = 3000
        probe_rows = probe_rows[:n_probe]
        for name in ("champs", "sides", "items", "query", "inv", "hist"):
            setattr(probe_ds, name, getattr(probe_ds, name)[:n_probe])
    if device.type == "cuda":
        probe_ds.to_device(device)

    def basket_tensors(baskets, n):
        items = sorted({i for b in baskets for p in b.split(",") for i in [int(p.split(":")[0])]})
        idx = {it: k + 1 for k, it in enumerate(items)}
        ci = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
        cc = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
        for row_i, b in enumerate(baskets, start=1):
            for slot, p in enumerate(b.split(",")[:v1.MAX_BASKET]):
                i, c = p.split(":")
                ci[row_i, slot] = idx[int(i)]
                cc[row_i, slot] = min(int(c), v1.MAX_BASKET)
        return ci.to(device), cc.to(device), len(items)

    cat_idx_A, cat_cnt_A, n_items_A = basket_tensors(baskets_A, n_cat_A)
    cat_idx_B, cat_cnt_B, n_items_B = basket_tensors(baskets_B, n_act)
    arch = lambda n: (len(champ_index), len(item_index), len(label_index),
                      len(item_index) + 1, n)
    model_A = TunedNet(v1.BasketNet(*arch(n_items_A)).to(device), CorrectionNet().to(device))
    model_A.load_state_dict(torch.load(FT_DIR / "econ_ft_ep8.pt", map_location="cpu",
                                       weights_only=False)["state_dict"])
    model_A.eval()
    model_B = v1.BasketNet(*arch(n_items_B)).to(device)
    model_B.load_state_dict(torch.load(WF_DIR / "wardfree_ep16.pt", map_location="cpu",
                                       weights_only=False)["state_dict"])
    model_B.eval()
    gid, permd = group_ids.to(device), perm.to(device)

    # truth group per probe row (-1 = outside the action space)
    y = torch.full((n_probe,), -1, dtype=torch.long)
    for i, row in enumerate(probe_rows):
        t = stripped_truth(row)
        key = basket_key(sorted(t.items())) if t else None
        y[i] = pos_A.get(key, -1)
    print(f"setup listo ({time.monotonic() - started:.0f}s); verdad en el espacio: "
          f"{int((y >= 0).sum())}/{n_probe}", flush=True)

    def distributions(sl):
        batch = {k: v[sl] for k, v in {
            "champs": probe_ds.champs, "sides": probe_ds.sides,
            "items": probe_ds.items, "query": probe_ds.query,
            "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
        b = batch["champs"].shape[0]
        dom = probe_domain[sl].to(device)
        econ = (econ_probe[sl].float().to(device) - sm) / ss
        sA, _ = model_A.corrected_scores(batch, device, cat_idx_A, cat_cnt_A, n_cat_A, dom, econ)
        gidx = gid.unsqueeze(0).expand(b, -1)
        gmax = torch.full((b, n_act), float("-inf"), device=device)
        gmax.scatter_reduce_(1, gidx, sA, reduce="amax", include_self=True)
        gsum = torch.zeros(b, n_act, device=device)
        gsum.scatter_add_(1, gidx, (sA - gmax.gather(1, gidx)).exp())
        P_A = torch.softmax(gmax + gsum.log(), dim=1)
        sB = model_B.scores(model_B.state(batch, device),
                            model_B.baskets.vectors(cat_idx_B, cat_cnt_B,
                                                    torch.arange(n_act, device=device)))
        P_B = torch.zeros_like(P_A)
        P_B.scatter_(1, permd.unsqueeze(0).expand(b, -1), torch.softmax(sB, dim=1))
        return P_A, P_B

    # ---- pass 1: decision-time features + P(y) under each parent ----
    feats = torch.zeros((n_probe, N_FEAT))
    pAy = torch.full((n_probe,), float("nan"))
    pBy = torch.full((n_probe,), float("nan"))
    with torch.no_grad():
        for s in range(0, n_probe, 512):
            sl = slice(s, min(s + 512, n_probe))
            P_A, P_B = distributions(sl)
            ent = lambda P: -(P * (P + 1e-12).log()).sum(1)
            f = torch.stack([
                P_A.max(1).values, ent(P_A), P_B.max(1).values, ent(P_B),
                (P_A.argmax(1) == P_B.argmax(1)).float(),
                torch.tensor([float(r["gold_est"]) / 1000.0 for r in probe_rows[sl]], device=device),
                torch.tensor([float(r.get("level") or 0) / 18.0 for r in probe_rows[sl]], device=device),
                torch.tensor([float(r.get("n_inventory") or 0) / 8.0 for r in probe_rows[sl]], device=device),
            ], dim=1)
            feats[sl] = f.cpu()
            yy = y[sl].to(device)
            ok = yy >= 0
            idx = torch.arange(yy.shape[0], device=device)[ok]
            pAy[sl][ok.cpu()] = P_A[idx, yy[ok]].cpu()
            pBy[sl][ok.cpu()] = P_B[idx, yy[ok]].cpu()

    # ---- folds by match; train gate + control out-of-fold ----
    games = sorted({r["match_id"] for r in probe_rows})
    order = torch.randperm(len(games), generator=torch.Generator().manual_seed(16)).tolist()
    fold_of_game = {games[g]: k % FOLDS for k, g in enumerate(order)}
    fold = torch.tensor([fold_of_game[r["match_id"]] for r in probe_rows])
    w_gate = torch.zeros(n_probe)
    w_ctrl = torch.zeros(n_probe)
    fold_stats = []
    valid = (y >= 0)
    for k in range(FOLDS):
        tr = valid & (fold != k)
        te = fold == k
        xf, pa, pb = feats[tr], pAy[tr], pBy[tr]
        for name, zero in (("gate", False), ("control", True)):
            torch.manual_seed(16 + k)
            net = Gate()
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            x_tr = torch.zeros_like(xf) if zero else xf
            for _ in range(STEPS):
                opt.zero_grad()
                w = net(x_tr)
                loss = -(w * pa + (1 - w) * pb + 1e-12).log().mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                x_te = torch.zeros((int(te.sum()), N_FEAT)) if zero else feats[te]
                w_te = net(x_te)
            (w_ctrl if zero else w_gate)[te] = w_te
            fold_stats.append({"fold": k, "arm": name, "train_rows": int(tr.sum()),
                               "test_rows": int(te.sum()),
                               "final_train_nll": round(float(loss), 5),
                               "w_test_mean": round(float(w_te.mean()), 4),
                               "w_test_std": round(float(w_te.std()), 4)})
        print(f"  fold {k}: gate w={fold_stats[-2]['w_test_mean']:.3f}±{fold_stats[-2]['w_test_std']:.3f} "
              f"| control w={fold_stats[-1]['w_test_mean']:.3f}", flush=True)

    # ---- pass 2: decode with per-row weights ----
    def decode(order_, row):
        budget = float(row["gold_est"])
        options = []
        for act in order_:
            if act == 0:
                options.append(("save", None))
            else:
                main = group_main[act]
                if purchase_sequence(tuple(sorted(main.elements())), row, budget, dragon) is not None:
                    options.append(("buy", Counter(main)))
            if len(options) >= 3:
                break
        return options or [("save", None)]

    rec = {"gate": [], "control": []}
    with torch.no_grad():
        for s in range(0, n_probe, 512):
            sl = slice(s, min(s + 512, n_probe))
            P_A, P_B = distributions(sl)
            for name, wv in (("gate", w_gate), ("control", w_ctrl)):
                w = wv[sl].to(device).unsqueeze(1)
                P = w * P_A + (1 - w) * P_B
                top = torch.topk(P, min(v1.SHORTLIST, n_act), dim=1).indices.cpu()
                for j in range(top.shape[0]):
                    rec[name].append(wardfree_record(probe_rows[s + j], decode(top[j].tolist(), probe_rows[s + j])))
    for name, recs in rec.items():
        p2.dump(OUT_DIR / f"rows_{name}_{stamp}.jsonl", recs)

    ref = [json.loads(l) for l in open(MIX_DIR / "rows_M_full.jsonl", encoding="utf-8")][:n_probe]
    ident = [(r["match_id"], r["ts"], r["champion"]) for r in rec["gate"]]
    assert [(r["match_id"], r["ts"], r["champion"]) for r in ref] == ident
    assert [(r["match_id"], r["ts"], r["champion"]) for r in rec["control"]] == ident
    rec["fixed"] = ref

    subsets = {"single": lambda r: r["true_size"] == 1, "multi": lambda r: r["true_size"] > 1,
               "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
               "ward_only": lambda r: r.get("save_kind") == "ward_only"}
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/experiment_learned_gate_v1.md",
              "scope": "internal probe, OUT-OF-FOLD by match; the probe is the selection set",
              "mode": stamp, "n": n_probe, "folds": FOLDS, "steps": STEPS, "lr": LR,
              "features": ["max_PA", "entropy_PA", "max_PB", "entropy_PB", "top1_agree",
                           "gold_est/1000", "level/18", "n_inventory/8"],
              "truth_in_action_space": int(valid.sum()),
              "fold_stats": fold_stats,
              "w_gate": {"mean": round(float(w_gate.mean()), 4), "std": round(float(w_gate.std()), 4),
                         "p10": round(float(w_gate.quantile(0.1)), 4), "p90": round(float(w_gate.quantile(0.9)), 4)},
              "w_control_mean": round(float(w_ctrl.mean()), 4),
              "aggregates": {k: p2.aggregate(v) for k, v in rec.items()},
              "paired": {}, "slices": {}}
    for x, yv in (("gate", "fixed"), ("gate", "control"), ("control", "fixed")):
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{yv}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(rec[x], rec[yv], metric)
        for label, pred in subsets.items():
            report["slices"][f"{x}_vs_{yv}_{label}"] = p2.paired(
                [r for r in rec[x] if pred(r)], [r for r in rec[yv] if pred(r)], "first_action_exact")
        per = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
        for t, recs in (("x", rec[x]), ("y", rec[yv])):
            for r in recs:
                per[t][r["match_id"]][0] += r["first_action_exact"]
                per[t][r["match_id"]][1] += 1
        report["paired"][f"{x}_vs_{yv}"]["ci95_2000draws"] = list(
            v1.paired_bootstrap(per["x"], per["y"], seed=16, n=2000))
    mv = {}
    for x, yv in (("gate", "fixed"), ("gate", "control")):
        c = b = 0
        for a, o in zip(rec[x], rec[yv]):
            c += a["first_action_exact"] > o["first_action_exact"]
            b += a["first_action_exact"] < o["first_action_exact"]
        mv[f"{x}_vs_{yv}"] = {"corrected": c, "broken": b, "net": c - b}
    report["movement"] = mv
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = OUT_DIR / f"learned_gate_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": {k: v["first_action_exact"]["value"] for k, v in report["aggregates"].items()},
                      "paired": {k: v for k, v in report["paired"].items() if not k.endswith("_any3")},
                      "w_gate": report["w_gate"], "w_control_mean": report["w_control_mean"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
