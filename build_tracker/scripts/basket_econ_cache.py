"""Economic-feature cache for the residual reranker (basket-residual-econ-v1).

Protocol: docs/experiment_basket_residual_econ_v1.md. Stores, per decision,
everything the correction trainer needs so the frozen Bf_ep16 backbone is
never re-run during training: frozen states, top-128 shortlist ids + scores,
the frozen tail logsumexp (exact full-CE normalizer), target position/score,
and 10 economic features per shortlist candidate computed with the SAME
activated engine the decoder uses (purchase_sequence / try_purchase),
preserving duplicates and purchase ordering at the observed gold_est budget.

Reviewer's gate: CPU feature throughput is measured on a real sample FIRST
and the projected build time is printed; if it exceeds the declared bound
the build stops and reports instead of running. --bench-only exits after
the benchmark.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] caps rows.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

import train_basket_ab as v1
import train_prefix as tp
from basket_vocabulary_probe import membership_digest, split_fit_probe

V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
OUT_DIR = ROOT / "data" / "experiments" / "16.18-basket-residual-econ-v1"
N_ECON = 10
TAIL_TOP = 64
BENCH_ROWS = 200
BUILD_BOUND_H = 8.0


def to_action(bkey):
    return tuple(sorted(i for part in bkey.split(",")
                        for i in [int(part.split(":")[0])] * int(part.split(":")[1])))


def econ_features(action, row, budget, dragon, counters):
    """10-dim description of executing `action` from `row` at `budget`.

    Mirrors purchase_sequence's semantics exactly, then REPLAYS the returned
    order step by step with try_purchase to track the consumed multiset and
    slot deltas. Purely descriptive: no preference encoded.
    """
    from app.conditional_policy import footwear_arguments, purchase_sequence
    from app.shop_econ import combine_cost, try_purchase
    out = [0.0] * N_ECON
    seq = purchase_sequence(action, row, budget, dragon)
    if seq is None:
        counters["infeasible"] += 1
        return out
    out[1] = 1.0
    total_cost = float(sum(s["cost"] for s in seq))
    out[2] = total_cost / 1000.0
    out[3] = (budget - total_cost) / 1000.0
    out[4] = min(max(total_cost / max(budget, 1.0), 0.0), 1.0)
    out[5] = float(len(seq))
    footwear = footwear_arguments(
        row, dragon,
        required_version=getattr(dragon, "footwear_state_version", None))
    target = Counter(action)
    pending = target.copy()
    inv_before = Counter(int(i) for i in (row.get("inventory") or []) if i)
    inv = inv_before.copy()
    wallet = float(budget)
    consumed = Counter()
    try:
        for step in seq:
            item_id = step["item_id"]
            protected = target - pending
            got = try_purchase(
                item_id, list(inv.elements()), wallet, dragon,
                role=row.get("role") or "",
                bot_quest_complete=row.get("bot_quest_complete"),
                unmodeled_regular_slots=row.get("unmodeled_regular_slots") or 0,
                role_slot_inventory_unknown=bool(
                    row.get("role_slot_inventory_unknown")),
                protected=protected, **footwear)
            if got is None:
                raise ValueError("replay diverged from purchase_sequence")
            cost, after = got
            consumed += (inv + Counter([item_id])) - after
            inv = after
            wallet -= cost
            pending[item_id] -= 1
            if not pending[item_id]:
                del pending[item_id]
    except Exception:
        counters["replay_failed"] += 1
        return out                       # cost features stand; replay dims 0
    preowned = consumed & inv_before
    out[6] = float(sum(preowned.values()))
    out[7] = sum(combine_cost(i, Counter(), dragon) * n
                 for i, n in preowned.items()) / 1000.0
    out[8] = float(sum(inv.values()) - sum(inv_before.values()))
    out[9] = float(sum(1 for s in seq if dragon.from_ids(s["item_id"])))
    return out


def rows_digest(rows_iter):
    h = hashlib.sha256()
    n = 0
    for row in rows_iter:
        h.update(f"{row['match_id']}|{row['ts']};".encode())
        n += 1
    return h.hexdigest(), n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(16)

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

    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"
    cap_fit = 3000 if args.smoke else None
    cap_probe = 1000 if args.smoke else None
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets).encode()).hexdigest()
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_ds = tp.ShopDataset.from_cache(v1_dir / "cache_fit_full_v2.pt",
                                       champ_index, item_index, label_index,
                                       dragon)
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index,
                                         dragon)
    fit_targets_all = torch.load(v1_dir / "cache_targets_full_v2.pt",
                                 weights_only=True)
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)

    cat_items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
    cat_item_index = {it: k + 1 for k, it in enumerate(cat_items)}
    cat_idx = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    cat_cnt = torch.zeros((n_cat, v1.MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, part in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, n_ = part.split(":")
            cat_idx[row_i, slot] = cat_item_index[int(i)]
            cat_cnt[row_i, slot] = min(int(n_), v1.MAX_BASKET)
    cat_idx, cat_cnt = cat_idx.to(device), cat_cnt.to(device)

    model = v1.BasketNet(len(champ_index), len(item_index), len(label_index),
                         len(item_index) + 1, len(cat_items)).to(device)
    blob = torch.load(V2_DIR / "Bf_ep16.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    # ---------------- frozen forward pass (GPU, minutes) ----------------
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "parent": "Bf_ep16", "n_cat": n_cat,
              "baskets_sha256": baskets_sha, "shortlist": v1.SHORTLIST,
              "tail_top": TAIL_TOP, "n_econ": N_ECON, "counters": {},
              "bench": {}}
    with torch.no_grad():
        basket_vecs = model.baskets.vectors(cat_idx, cat_cnt,
                                            torch.arange(n_cat, device=device))
    torch.save(basket_vecs.half().cpu(), OUT_DIR / f"basket_vecs_{stamp}.pt")

    def frozen_pass(ds, n_rows, targets, tail_top):
        states, top_i, top_s, tails, tpos, tscore = [], [], [], [], [], []
        tail_i, tail_s = [], []
        with torch.no_grad():
            for s in range(0, n_rows, tp.BATCH):
                sl = slice(s, min(s + tp.BATCH, n_rows))
                batch = {k: v[sl] for k, v in {
                    "champs": ds.champs, "sides": ds.sides, "items": ds.items,
                    "query": ds.query, "inv": ds.inv, "hist": ds.hist}.items()}
                state = model.state(batch, device)
                scores = model.scores(state, basket_vecs)
                tv, ti = torch.topk(scores, v1.SHORTLIST, dim=1)
                lse_all = torch.logsumexp(scores, dim=1)
                lse_top = torch.logsumexp(tv, dim=1)
                x = torch.clamp(lse_top - lse_all, max=-1e-7)
                tails.append((lse_all + torch.log1p(-torch.exp(x))).cpu())
                states.append(state.half().cpu())
                top_i.append(ti.int().cpu())
                top_s.append(tv.cpu())
                if targets is not None:
                    tg = targets[sl].to(device)
                    tscore.append(scores.gather(1, tg.unsqueeze(1)).squeeze(1).cpu())
                    eq = ti == tg.unsqueeze(1)
                    pos = torch.where(eq.any(1), eq.float().argmax(1),
                                      torch.full_like(tg, -1))
                    tpos.append(pos.short().cpu())
                if tail_top:
                    masked = scores.scatter(1, ti, float("-inf"))
                    wv, wi = torch.topk(masked, tail_top, dim=1)
                    tail_i.append(wi.int().cpu())
                    tail_s.append(wv.cpu())
        out = {"states": torch.cat(states), "top_idx": torch.cat(top_i),
               "top_score": torch.cat(top_s), "tail_lse": torch.cat(tails)}
        if targets is not None:
            out["target_score"] = torch.cat(tscore)
            out["target_short_pos"] = torch.cat(tpos)
        if tail_top:
            out["tail_top_idx"] = torch.cat(tail_i)
            out["tail_top_score"] = torch.cat(tail_s)
        return out

    n_fit = min(cap_fit or n_fit_all, n_fit_all)
    fit_targets = fit_targets_all[:n_fit]
    if args.smoke:                       # slice cached tensors for smoke
        for ds, n in ((fit_ds, n_fit), (probe_ds, cap_probe)):
            for name in ("champs", "sides", "items", "query", "inv", "hist"):
                setattr(ds, name, getattr(ds, name)[:n])
    n_probe = probe_ds.champs.shape[0]
    fit_pack = frozen_pass(fit_ds, n_fit, fit_targets, 0)
    probe_pack = frozen_pass(probe_ds, n_probe, None, TAIL_TOP)
    print(f"forward congelado listo: fit {n_fit} probe {n_probe} "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    # ---------------- econ features (CPU, benchmark-gated) ----------------
    def stream(want_fit, cap):
        return v1.stream_split(train_path, fit_set, want_fit, champ_index,
                               dragon, cap)

    counters = Counter()

    def features_for(row, cand_ids):
        budget = float(row["gold_est"])
        feats = torch.zeros((len(cand_ids), N_ECON))
        for k, cand in enumerate(cand_ids):
            if cand == 0:
                feats[k, 0] = 1.0
                continue
            feats[k].copy_(torch.tensor(econ_features(
                to_action(baskets[cand - 1]), row, budget, dragon, counters)))
        return feats

    bench_n = min(BENCH_ROWS, n_fit)
    t0 = time.monotonic()
    for i, row in enumerate(stream(True, bench_n)):
        features_for(row, fit_pack["top_idx"][i].tolist())
    per_row = (time.monotonic() - t0) / bench_n
    projected_h = per_row * (n_fit + n_probe) / 3600
    report["bench"] = {"rows": bench_n, "ms_per_decision": round(per_row * 1e3, 2),
                       "ms_per_candidate": round(per_row * 1e3 / v1.SHORTLIST, 4),
                       "projected_build_hours": round(projected_h, 2),
                       "bound_hours": BUILD_BOUND_H}
    print(f"BENCH: {per_row * 1e3:.1f} ms/decision "
          f"({per_row * 1e3 / v1.SHORTLIST:.3f} ms/candidato), "
          f"proyeccion {projected_h:.2f} h para {n_fit + n_probe} decisiones",
          flush=True)
    if args.bench_only or projected_h > BUILD_BOUND_H:
        if projected_h > BUILD_BOUND_H:
            report["aborted"] = "projected build exceeds declared bound"
            print("ABORTADO: proyeccion supera el limite declarado", flush=True)
        (OUT_DIR / f"cache_report_{stamp}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return

    for name, want_fit, pack, n_rows, cap in (
            ("fit", True, fit_pack, n_fit, n_fit if args.smoke else None),
            ("probe", False, probe_pack, n_probe,
             n_probe if args.smoke else None)):
        path = OUT_DIR / f"econ_{name}_{stamp}.pt"
        if path.exists():
            print(f"econ {name}: ya existe, salto", flush=True)
            continue
        feats = torch.zeros((n_rows, v1.SHORTLIST, N_ECON), dtype=torch.float16)
        digest = hashlib.sha256()
        t0 = time.monotonic()
        for i, row in enumerate(stream(want_fit, cap)):
            digest.update(f"{row['match_id']}|{row['ts']};".encode())
            feats[i] = features_for(row, pack["top_idx"][i].tolist()).half()
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.monotonic() - t0)
                print(f"  econ {name} {i + 1}/{n_rows} "
                      f"({rate:.1f} filas/s, ETA {(n_rows - i - 1) / rate / 60:.0f} min)",
                      flush=True)
        assert i + 1 == n_rows, f"stream/{name} desalineado: {i + 1} vs {n_rows}"
        torch.save(feats, path)
        pack["rows_sha256"] = digest.hexdigest()
        pack["n_rows"] = n_rows
        torch.save(pack, OUT_DIR / f"frozen_{name}_{stamp}.pt")
        print(f"econ {name} listo ({(time.monotonic() - t0) / 60:.1f} min)",
              flush=True)

    report["counters"] = dict(counters)
    report["fit_rows"] = n_fit
    report["probe_rows"] = n_probe
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    (OUT_DIR / f"cache_report_{stamp}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CACHE COMPLETA: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
