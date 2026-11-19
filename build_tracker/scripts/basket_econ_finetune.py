"""Twin economic fine-tune on a fixed feature domain (basket-econ-finetune-v1).

Protocol: docs/experiment_basket_econ_finetune_v1.md — the reviewer's six
conditions. Both arms fine-tune the FULL Bf_ep16 (trunk + basket encoder +
scorer) plus a shared zero-init correction head:

    score(c) = current_base_score(c) + [c in D(x)] * correction(c)

D(x) = the frozen parent's cached top-128 (never refreshed, never a hard
action mask). Control receives the econ block zeroed; candidate receives
real features. Full-catalogue CE recomputed with gradients (microbatch 32,
effective 256); no frozen tail/scores reused; out-of-domain targets stay in
the loss through their updated base score.

Requires the caches of basket_econ_cache.py. Run via footwear_runtime with
PREFIX_ML_DIR set; [--smoke] uses the smoke cache and 1 epoch.
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
import torch.nn as nn

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR, to_action
from basket_residual_econ import CorrectionNet

V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
OUT_DIR = ROOT / "data" / "experiments" / "16.18-basket-econ-finetune-v1"
LR = 1e-4
EPOCHS = 8
N_EXAMPLES = 30
LATENCY_SAMPLE = 200


class TunedNet(nn.Module):
    def __init__(self, base, correction):
        super().__init__()
        self.base = base
        self.correction = correction

    def corrected_scores(self, batch, device, cat_idx, cat_cnt, n_cat,
                         domain, econ):
        state = self.base.state(batch, device)
        vecs = self.base.baskets.vectors(cat_idx, cat_cnt,
                                         torch.arange(n_cat, device=device))
        scores = self.base.scores(state, vecs)
        corr = self.correction(state, vecs[domain], econ)
        return scores.scatter_add(1, domain, corr), corr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    cache_stamp = "smoke" if args.smoke else "full"
    stamp = cache_stamp
    epochs = 1 if args.smoke else EPOCHS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    from app.shop_econ import components_of
    dragon = dragon_for_patch(manifest_gen["patch"])
    fam_cache = {}

    def family_of(t):
        if t not in fam_cache:
            fam_cache[t] = set(components_of(t, dragon)) | {t}
        return fam_cache[t]

    train_path = generation / "visits_train.jsonl"
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets).encode()).hexdigest()
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_report = json.loads(
        (CACHE_DIR / f"cache_report_{cache_stamp}.json").read_text(
            encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    fit = torch.load(CACHE_DIR / f"frozen_fit_{cache_stamp}.pt",
                     weights_only=True)
    probe = torch.load(CACHE_DIR / f"frozen_probe_{cache_stamp}.pt",
                       weights_only=True)
    econ_fit = torch.load(CACHE_DIR / f"econ_fit_{cache_stamp}.pt",
                          weights_only=True)
    econ_probe = torch.load(CACHE_DIR / f"econ_probe_{cache_stamp}.pt",
                            weights_only=True)
    n_fit, n_probe = fit["n_rows"], probe["n_rows"]
    fit_domain = fit["top_idx"].long()          # D(x): frozen, both arms
    probe_domain = probe["top_idx"].long()

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_ds = tp.ShopDataset.from_cache(v1_dir / "cache_fit_full_v2.pt",
                                       champ_index, item_index, label_index,
                                       dragon)
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index,
                                         dragon)
    fit_targets = torch.load(v1_dir / "cache_targets_full_v2.pt",
                             weights_only=True)
    if args.smoke:
        for ds, n in ((fit_ds, n_fit), (probe_ds, n_probe)):
            for name in ("champs", "sides", "items", "query", "inv", "hist"):
                setattr(ds, name, getattr(ds, name)[:n])
        fit_targets = fit_targets[:n_fit]
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)
    fit_targets = fit_targets.to(device)

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
    arch_args = (len(champ_index), len(item_index), len(label_index),
                 len(item_index) + 1, len(cat_items))

    # Same standardization contract as the residual experiment.
    feas = econ_fit[:, :, 1] > 0
    stats_mean = torch.zeros(N_ECON)
    stats_std = torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    if flat.numel():
        for d in range(2, N_ECON):
            stats_mean[d] = flat[:, d].mean()
            stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = stats_mean.to(device), stats_std.to(device)

    def econ_block(raw, arm):
        if arm == "control":
            return torch.zeros(*raw.shape[:-1], N_ECON, device=device)
        return (raw.float().to(device) - sm) / ss

    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, n_probe if args.smoke else None))
    digest = hashlib.sha256()
    for row in probe_rows:
        digest.update(f"{row['match_id']}|{row['ts']};".encode())
    assert digest.hexdigest() == probe["rows_sha256"], "probe desalineado"

    g = torch.Generator().manual_seed(16)
    subset_idx = torch.randperm(n_fit_all, generator=g)[:10000]
    subset_idx = subset_idx[subset_idx < n_fit].to(device)
    parent_blob = torch.load(V2_DIR / "Bf_ep16.pt", map_location="cpu",
                             weights_only=False)
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    def fixed_subset_ce(model, arm):
        """Combined-model full-CE on the fixed FIT subset."""
        nll_s = nn.CrossEntropyLoss(reduction="sum")
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for s0 in range(0, len(subset_idx), 256):
                idxs = subset_idx[s0:s0 + 256]
                batch = {k: v[idxs.to(v.device)] for k, v in {
                    "champs": fit_ds.champs, "sides": fit_ds.sides,
                    "items": fit_ds.items, "query": fit_ds.query,
                    "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                dom = fit_domain[idxs.cpu()].to(device)
                scores, _ = model.corrected_scores(
                    batch, device, cat_idx, cat_cnt, n_cat, dom,
                    econ_block(econ_fit[idxs.cpu()], arm))
                total += float(nll_s(scores, fit_targets[idxs]))
                n += len(idxs)
        model.train()
        return total / max(n, 1)

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "parent": "Bf_ep16", "lr": LR, "epochs": epochs,
              "micro": p2.MICRO, "effective": p2.EFFECTIVE,
              "step0_check": {}, "curves": {}, "aggregates": {}, "paired": {},
              "net_vs": {}, "domain_diagnostics": {}, "latency": {
                  "feature_bench": cache_report["bench"]}, "examples": []}
    rows_by_arm = {}
    nll = nn.CrossEntropyLoss()

    for arm in ("control", "econ"):
        torch.manual_seed(16)
        base = v1.BasketNet(*arch_args).to(device)
        base.load_state_dict(parent_blob["state_dict"])
        correction = CorrectionNet().to(device)
        model = TunedNet(base, correction)
        # Step-0 verification (condition 2): zero correction, parent CE.
        with torch.no_grad():
            probe_batch = {k: v[:64] for k, v in {
                "champs": fit_ds.champs, "sides": fit_ds.sides,
                "items": fit_ds.items, "query": fit_ds.query,
                "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
            _, corr0 = model.corrected_scores(
                probe_batch, device, cat_idx, cat_cnt, n_cat,
                fit_domain[:64].to(device),
                econ_block(econ_fit[:64], arm))
            max_corr0 = float(corr0.abs().max())
        ce0 = fixed_subset_ce(model, arm)
        report["step0_check"][arm] = {"max_abs_correction": max_corr0,
                                      "fixed_subset_full_ce": ce0}
        print(f"step0 {arm}: |corr|max {max_corr0:.2e} fixedCE {ce0:.4f}",
              flush=True)
        assert max_corr0 == 0.0, "correction not zero at init"

        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        order_gen = torch.Generator().manual_seed(16)   # same batches per arm
        curve = []
        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.monotonic()
            order = torch.randperm(n_fit, generator=order_gen)
            total, n = 0.0, 0
            for s in range(0, n_fit, p2.EFFECTIVE):
                idxs_all = order[s:s + p2.EFFECTIVE]
                opt.zero_grad()
                for m0 in range(0, len(idxs_all), p2.MICRO):
                    idxs = idxs_all[m0:m0 + p2.MICRO]
                    batch = {k: v[idxs.to(v.device)] for k, v in {
                        "champs": fit_ds.champs, "sides": fit_ds.sides,
                        "items": fit_ds.items, "query": fit_ds.query,
                        "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                    scores, _ = model.corrected_scores(
                        batch, device, cat_idx, cat_cnt, n_cat,
                        fit_domain[idxs].to(device),
                        econ_block(econ_fit[idxs], arm))
                    loss = nll(scores, fit_targets[idxs.to(device)])
                    (loss * len(idxs) / len(idxs_all)).backward()
                    total += float(loss.detach()) * len(idxs)
                opt.step()
                n += len(idxs_all)
            sched.step()
            fixed = fixed_subset_ce(model, arm)
            curve.append({"epoch": epoch, "train_loss": total / max(n, 1),
                          "fixed_subset_full_ce": fixed,
                          "seconds": round(time.monotonic() - t_ep, 1)})
            print(f"  {arm} epoch {epoch}/{epochs} loss {total / max(n, 1):.3f}"
                  f" fixedCE {fixed:.3f} ({curve[-1]['seconds']}s)", flush=True)
            if epoch in (4, epochs):
                p2.save_checkpoint(OUT_DIR / f"{arm}_ft_ep{epoch}.pt", model,
                                   opt, sched, epoch)
        report["curves"][arm] = curve

        # ---------------- eval (condition 4: D is never a mask) ----------
        model.eval()
        records = []
        in_d_first = in_d_top3 = 0
        mass_out_sum = 0.0
        lazy_calls = 0
        t_eval = time.monotonic()
        with torch.no_grad():
            for s in range(0, n_probe, 512):
                sl = slice(s, min(s + 512, n_probe))
                batch = {k: v[sl] for k, v in {
                    "champs": probe_ds.champs, "sides": probe_ds.sides,
                    "items": probe_ds.items, "query": probe_ds.query,
                    "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
                dom = probe_domain[sl].to(device)
                scores, _ = model.corrected_scores(
                    batch, device, cat_idx, cat_cnt, n_cat, dom,
                    econ_block(econ_probe[sl], arm))
                probs = torch.softmax(scores, dim=1)
                mass_in = probs.gather(1, dom).sum(dim=1)
                mass_out_sum += float((1.0 - mass_in).sum())
                top = torch.topk(scores, min(v1.SHORTLIST, n_cat),
                                 dim=1).indices.cpu()
                for j in range(top.shape[0]):
                    i = s + j
                    row = probe_rows[i]
                    budget = float(row["gold_est"])
                    d_set = set(probe_domain[i].tolist())
                    feas_map = {int(probe_domain[i, k]):
                                bool(econ_probe[i, k, 1] > 0)
                                for k in range(probe_domain.shape[1])}
                    options, opt_cands = [], []
                    for cand in top[j].tolist():
                        if cand == 0:
                            options.append(("save", None))
                            opt_cands.append(cand)
                        else:
                            action = to_action(baskets[cand - 1])
                            if cand in feas_map:
                                legal = feas_map[cand]
                            else:
                                lazy_calls += 1
                                legal = purchase_sequence(
                                    action, row, budget, dragon) is not None
                            if legal:
                                options.append(("buy", Counter(action)))
                                opt_cands.append(cand)
                        if len(options) >= 3:
                            break
                    fallback = not options
                    if fallback:
                        options = [("save", None)]
                        opt_cands = [0]
                    in_d_first += int(opt_cands[0] in d_set)
                    in_d_top3 += int(all(c in d_set for c in opt_cands))
                    rec = p2.base_record(row, options, family_of, dragon)
                    rec["fallback_save"] = fallback
                    records.append(rec)
        eval_s = time.monotonic() - t_eval
        report["domain_diagnostics"][arm] = {
            "first_option_in_domain": in_d_first / n_probe,
            "top3_all_in_domain": in_d_top3 / n_probe,
            "mean_prob_mass_outside_domain": mass_out_sum / n_probe}
        report["latency"][arm] = {"eval_seconds": round(eval_s, 1),
                                  "lazy_legality_calls": lazy_calls}
        p2.dump(OUT_DIR / f"rows_{arm}_ft_{stamp}.jsonl", records)
        rows_by_arm[arm] = records
        report["aggregates"][arm] = p2.aggregate(records)
        print(f"eval {arm}: exact "
              f"{report['aggregates'][arm]['first_action_exact']['value']:.4f}"
              f" out-of-D mass {mass_out_sum / n_probe:.4f} ({eval_s:.0f}s)",
              flush=True)
        # Condition 5: the online contract is frozen-parent forward (for
        # D(x)) + economic features + tuned forward. Time the two forwards
        # on single decisions; the feature cost is the cache bench figure.
        # Offline estimate, NOT an online p95.
        if arm == "econ" and not args.smoke:
            parent = v1.BasketNet(*arch_args).to(device)
            parent.load_state_dict(parent_blob["state_dict"])
            parent.eval()
            comps = {}
            with torch.no_grad():
                for label, net_fn in (
                        ("parent_forward", lambda b: parent.scores(
                            parent.state(b, device), parent.baskets.vectors(
                                cat_idx, cat_cnt,
                                torch.arange(n_cat, device=device)))),
                        ("tuned_forward", lambda b: model.corrected_scores(
                            b, device, cat_idx, cat_cnt, n_cat,
                            probe_domain[:1].to(device),
                            econ_block(econ_probe[:1], arm))[0])):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.monotonic()
                    for i in range(LATENCY_SAMPLE):
                        b1 = {k: v[i:i + 1] for k, v in {
                            "champs": probe_ds.champs, "sides": probe_ds.sides,
                            "items": probe_ds.items, "query": probe_ds.query,
                            "inv": probe_ds.inv,
                            "hist": probe_ds.hist}.items()}
                        net_fn(b1)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    comps[label] = round(
                        (time.monotonic() - t0) / LATENCY_SAMPLE * 1e3, 2)
            comps["econ_features"] = cache_report["bench"]["ms_per_decision"]
            comps["total_estimate"] = round(sum(comps.values()), 2)
            comps["note"] = ("offline single-decision estimate over "
                             f"{LATENCY_SAMPLE} probe rows; not an online p95")
            report["latency"]["online_contract_ms"] = comps
            del parent
        del model, base, correction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------ comparisons ------------------------
    def key_of(r):
        return (r["match_id"], r["ts"], r["champion"])

    subsets = {
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
    }
    report["paired"]["econ_vs_control"] = p2.paired(
        rows_by_arm["econ"], rows_by_arm["control"], "first_action_exact")
    for label, pred in subsets.items():
        report["paired"][f"econ_vs_control_{label}"] = p2.paired(
            [r for r in rows_by_arm["econ"] if pred(r)],
            [r for r in rows_by_arm["control"] if pred(r)],
            "first_action_exact")
    if not args.smoke:
        parent_rows = [json.loads(l) for l in open(
            V2_DIR / "rows_Bf_full.jsonl", encoding="utf-8")]
        parent_by_key = {key_of(r): r for r in parent_rows}
        for arm in ("econ", "control"):
            report["paired"][f"{arm}_vs_parent"] = p2.paired(
                rows_by_arm[arm], parent_rows, "first_action_exact")
        for name_a, name_b, rows_a, rows_b in (
                ("econ", "parent", rows_by_arm["econ"], parent_rows),
                ("control", "parent", rows_by_arm["control"], parent_rows),
                ("econ", "control", rows_by_arm["econ"],
                 rows_by_arm["control"])):
            by_key_b = {key_of(r): r for r in rows_b}
            cor = bro = 0
            for r in rows_a:
                p = by_key_b.get(key_of(r))
                if p is None or r.get("first_action_exact") is None:
                    continue
                a, b = p["first_action_exact"], r["first_action_exact"]
                cor += a < b
                bro += a > b
            report["net_vs"][f"{name_a}_vs_{name_b}"] = {
                "corrected": cor, "broken": bro, "net": cor - bro}

        def norm_first(pf):
            return pf if isinstance(pf, str) else {str(k): v
                                                   for k, v in pf.items()}

        econ_by_key = {key_of(r): r for r in rows_by_arm["econ"]}
        ctrl_by_key = {key_of(r): r for r in rows_by_arm["control"]}
        for i, row in enumerate(probe_rows):
            if len(report["examples"]) >= N_EXAMPLES:
                break
            k = (row["match_id"], row["ts"], row.get("champion"))
            r, c = econ_by_key.get(k), ctrl_by_key.get(k)
            if (not r or not c
                    or norm_first(r["pred_first"]) == norm_first(c["pred_first"])):
                continue
            feats = econ_probe[i].float()
            cands = []
            for kk in range(8):
                cand = int(probe_domain[i, kk])
                cands.append({
                    "basket": "SAVE" if cand == 0 else baskets[cand - 1],
                    "feasible": bool(feats[kk, 1] > 0) or cand == 0,
                    "cost": round(float(feats[kk, 2]) * 1000),
                    "gold_left": round(float(feats[kk, 3]) * 1000),
                    "preowned_consumed": int(feats[kk, 6]),
                    "slot_delta": int(feats[kk, 8])})
            report["examples"].append({
                "match_id": row["match_id"], "ts": row["ts"],
                "champion": row.get("champion"), "gold_est": row["gold_est"],
                "control_first": c["pred_first"], "econ_first": r["pred_first"],
                "truth_hit_control": c["first_action_exact"],
                "truth_hit_econ": r["first_action_exact"],
                "domain_head": cands})

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"finetune_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["paired"], indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
