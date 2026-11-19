"""CE decomposition + gradient-interference diagnostic for the ward term.

Protocol: docs/experiment_ward_loss_diagnostic_v1.md. NO training, no
weight update, no VAL/test read. Measures, on a fixed FIT subset:

  -log p(b*) = [logZ - logZ_g*]  +  [logZ_g* - s_b*]
                main purchase        ward | main purchase

(exact chain rule over the ward-stripped grouping of the catalogue), and
the gradients of BOTH terms w.r.t. the SHARED parameters: their norms,
their cosine, and how much the update direction would move if the ward
term were downweighted to 0.25.

Two measurement points, both frozen:
  Bf_ep16     - plain full-catalogue CE (the phase-2 recipe)
  econ_ft_ep8 - corrected scores (base + econ correction on the fixed
                top-128 domain), the current dev baseline

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
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import (basket_entries, basket_key,
                                     membership_digest, split_fit_probe)
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR as FT_DIR
from basket_wardfree import strip_counter

OUT_DIR = ROOT / "data" / "experiments" / "16.18-ward-loss-diagnostic-v1"
BF_CKPT = ROOT / "data" / "experiments" / "16.18-whole-basket-v2" / "Bf_ep16.pt"
SUBSET = 10000          # same fixed-subset construction as p2/wardfree
WARD_WEIGHT = 0.25      # the weight the proposed contrast would use
MICRO = 32


def flat_grad(loss, params, retain):
    g = torch.autograd.grad(loss, params, retain_graph=retain,
                            allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1)
                      for p, gi in zip(params, g)])


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

    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"
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

    # ---- ward-stripped grouping of the catalogue (same map as the decoder) --
    group_of_key = {None: 0}
    n_groups = 1
    group_ids = torch.zeros(n_cat, dtype=torch.long)
    has_ward = torch.zeros(n_cat, dtype=torch.bool)
    for cid, bkey in enumerate(baskets, start=1):
        full = Counter({int(part.split(":")[0]): int(part.split(":")[1])
                        for part in bkey.split(",")})
        stripped = strip_counter(full)
        has_ward[cid] = sum(stripped.values()) != sum(full.values())
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = n_groups
            n_groups += 1
        group_ids[cid] = group_of_key[skey]
    group_size = torch.bincount(group_ids, minlength=n_groups)

    # ---- FIT targets (exact catalogue index, wards included) ----
    fit_targets = []
    for row in v1.stream_split(train_path, fit_set, True, champ_index,
                               dragon, None):
        entries = basket_entries(row.get("label_counts"))
        if not entries or any(i == v1.SAVE_SENTINEL for i, _ in entries):
            fit_targets.append(0)
        else:
            fit_targets.append(basket_pos[basket_key(entries)])
    n_fit = len(fit_targets)
    assert n_fit == n_fit_all
    fit_targets = torch.tensor(fit_targets, dtype=torch.long)

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_ds = tp.ShopDataset.from_cache(v1_dir / "cache_fit_full_v2.pt",
                                       champ_index, item_index, label_index,
                                       dragon)
    # identical construction to p2.full_ce_on_subset / basket_wardfree
    g = torch.Generator().manual_seed(16)
    subset_idx = torch.randperm(n_fit, generator=g)[:SUBSET]
    if args.smoke:
        subset_idx = subset_idx[:512]
    if device.type == "cuda":
        fit_ds.to_device(device)
    fit_targets = fit_targets.to(device)
    gid = group_ids.to(device)
    n_sub = len(subset_idx)

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
    arch = (len(champ_index), len(item_index), len(label_index),
            len(item_index) + 1, len(cat_items))

    # ---- econ cache (only for the econ_ft measurement point) ----
    cache_report = json.loads(
        (CACHE_DIR / "cache_report_full.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    frozen_fit = torch.load(CACHE_DIR / "frozen_fit_full.pt", weights_only=True)
    assert frozen_fit["n_rows"] == n_fit
    fit_domain = frozen_fit["top_idx"][subset_idx].long()
    del frozen_fit
    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas = econ_fit[:, :, 1] > 0
    stats_mean, stats_std = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    for d in range(2, N_ECON):
        stats_mean[d] = flat[:, d].mean()
        stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    econ_sub = econ_fit[subset_idx].clone()
    del econ_fit, flat, feas
    sm, ss = stats_mean.to(device), stats_std.to(device)
    print(f"setup listo ({time.monotonic() - started:.0f}s); "
          f"catalogo {n_cat}, grupos {n_groups}, subset {n_sub}", flush=True)

    # ---- the two frozen measurement points ----
    def load_point(name):
        path = BF_CKPT if name == "Bf_ep16" else FT_DIR / "econ_ft_ep8.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if name == "Bf_ep16":
            model = v1.BasketNet(*arch).to(device)
        else:
            model = TunedNet(v1.BasketNet(*arch).to(device),
                             CorrectionNet().to(device))
        model.load_state_dict(blob["state_dict"])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return model, (None if name == "Bf_ep16" else "econ"), sha

    results = {}
    for point in ("Bf_ep16", "econ_ft_ep8"):
        model, kind, ckpt_sha = load_point(point)
        model.eval()                      # no dropout: a reproducible reading
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        g_main = torch.zeros(n_params, device=device)
        g_ward = torch.zeros(n_params, device=device)
        sums = Counter()
        ward_rows = {"ward_target": [0, 0.0, 0.0], "no_ward_target": [0, 0.0, 0.0]}
        singleton_rows = 0
        t0 = time.monotonic()
        for s in range(0, n_sub, MICRO):
            idxs = subset_idx[s:s + MICRO]
            w = len(idxs) / n_sub
            batch = {k: v[idxs.to(v.device)] for k, v in {
                "champs": fit_ds.champs, "sides": fit_ds.sides,
                "items": fit_ds.items, "query": fit_ds.query,
                "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
            if kind == "econ":
                dom = fit_domain[s:s + MICRO].to(device)
                econ = ((econ_sub[s:s + MICRO].float().to(device) - sm) / ss)
                scores, _ = model.corrected_scores(batch, device, cat_idx,
                                                   cat_cnt, n_cat, dom, econ)
            else:
                state = model.state(batch, device)
                vecs = model.baskets.vectors(
                    cat_idx, cat_cnt, torch.arange(n_cat, device=device))
                scores = model.scores(state, vecs)
            t = fit_targets[idxs.to(device)]
            logZ = torch.logsumexp(scores, dim=1)
            same = gid.unsqueeze(0) == gid[t].unsqueeze(1)
            logZ_g = torch.logsumexp(
                scores.masked_fill(~same, float("-inf")), dim=1)
            s_t = scores.gather(1, t.unsqueeze(1)).squeeze(1)
            l_main = (logZ - logZ_g)
            l_ward = (logZ_g - s_t)
            L_main, L_ward = l_main.mean(), l_ward.mean()
            g_main += flat_grad(L_main * w, params, True)
            g_ward += flat_grad(L_ward * w, params, False)
            with torch.no_grad():
                sums["main"] += float(l_main.sum())
                sums["ward"] += float(l_ward.sum())
                sums["total"] += float((logZ - s_t).sum())
                tw = has_ward.to(device)[t]
                for tag, m in (("ward_target", tw), ("no_ward_target", ~tw)):
                    ward_rows[tag][0] += int(m.sum())
                    ward_rows[tag][1] += float(l_main[m].sum())
                    ward_rows[tag][2] += float(l_ward[m].sum())
                singleton_rows += int((group_size.to(device)[gid[t]] == 1).sum())
        g_total = g_main + g_ward
        g_weighted = g_main + WARD_WEIGHT * g_ward
        nm, nw = float(g_main.norm()), float(g_ward.norm())
        nt, ng = float(g_total.norm()), float(g_weighted.norm())
        cos = float(torch.dot(g_main, g_ward) / max(nm * nw, 1e-12))
        results[point] = {
            "ce_nats": {"total": sums["total"] / n_sub,
                        "main_purchase": sums["main"] / n_sub,
                        "ward_given_main": sums["ward"] / n_sub},
            "ward_share_of_ce": sums["ward"] / max(sums["total"], 1e-12),
            "decomposition_residual": abs(
                sums["total"] - sums["main"] - sums["ward"]) / n_sub,
            "rows": {
                "n": n_sub,
                "target_contains_ward": ward_rows["ward_target"][0],
                "target_group_is_singleton_zero_ward_loss": singleton_rows,
                "ce_main_ward_targets": (ward_rows["ward_target"][1]
                                         / max(ward_rows["ward_target"][0], 1)),
                "ce_ward_ward_targets": (ward_rows["ward_target"][2]
                                         / max(ward_rows["ward_target"][0], 1)),
                "ce_ward_nonward_targets": (
                    ward_rows["no_ward_target"][2]
                    / max(ward_rows["no_ward_target"][0], 1)),
            },
            "gradients_on_shared_parameters": {
                "n_parameters": n_params,
                "norm_main": nm, "norm_ward": nw, "norm_total": nt,
                "norm_ratio_ward_over_main": nw / max(nm, 1e-12),
                "cosine_main_ward": cos,
                "cosine_total_vs_weighted": float(
                    torch.dot(g_total, g_weighted) / max(nt * ng, 1e-12)),
                "relative_step_change_at_w0.25": (
                    float((g_weighted - g_total).norm()) / max(nt, 1e-12)),
                "ward_weight_probed": WARD_WEIGHT,
                "mode": "eval (dropout off), no optimizer step",
            },
            "checkpoint_sha256": ckpt_sha,
            "seconds": round(time.monotonic() - t0, 1),
        }
        print(f"{point}: CE {results[point]['ce_nats']['total']:.5f} = "
              f"main {results[point]['ce_nats']['main_purchase']:.5f} + "
              f"ward {results[point]['ce_nats']['ward_given_main']:.5f}; "
              f"cos(main,ward) {cos:+.4f}; "
              f"|g_ward|/|g_main| {nw / max(nm, 1e-12):.4f}", flush=True)
        del model, params, g_main, g_ward, g_total, g_weighted
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/experiment_ward_loss_diagnostic_v1.md",
        "scope": "fixed FIT subset; no training, no weight update, "
                 "no VAL, no test",
        "mode": stamp,
        "subset": {"n": n_sub, "construction":
                   "torch.randperm(n_fit, generator=manual_seed(16))[:10000] "
                   "- the same fixed subset as the phase-2 training curves"},
        "catalogue": {"n_cat": n_cat, "n_groups": n_groups,
                      "baskets_sha256": baskets_sha},
        "points": results,
        "limits": [
            "A gradient reading at one frozen point does not predict the "
            "trajectory of a full fine-tune, and is not a substitute for "
            "running the contrast.",
            "Measured with dropout off; training gradients carry dropout "
            "noise this reading does not.",
            "The CE decomposition is exact by the chain rule; the gradient "
            "cosine is a local property of these parameters and this subset.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    out = OUT_DIR / f"ward_loss_diagnostic_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
