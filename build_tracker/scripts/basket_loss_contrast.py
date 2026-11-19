"""Ward-weight loss contrast: twin arms w=1.00 vs w=0.25 from Bf_ep16.

Protocol: docs/experiment_loss_contrast_v1.md. Both arms clone the econ
fine-tune recipe exactly (real econ features in BOTH, zero-init correction
head, frozen top-128 domain, full-catalogue CE with gradients, micro 32 ->
effective 256, Adam 1e-4 cosine T_max=8, 8 epochs, seed 16, shared batch
order). The ONLY difference is the weight on the ward term of the exact
chain-rule decomposition

    CE = (logZ - logZ_g*) + w * (logZ_g* - s_b*)

Each arm is scored alone (grouped decoder) and mixed 50/50 with the frozen
wardfree_ep16 under the frozen mixture rule. Two primaries: mix(w025) vs
mix(w1) for attribution, and mix(w025) vs the current frozen mixture for
practical progress. Complementarity against wardfree is measured for every
arm. Probe only -- no VAL, no test.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] uses the smoke
caches and 1 epoch.
"""
import argparse
import hashlib
import json
import os
import shutil
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

V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
MIX_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
OUT_DIR = ROOT / "data" / "experiments" / "16.18-loss-contrast-v1"
LR = 1e-4
EPOCHS = 8
ARMS = {"w1": 1.00, "w025": 0.25}
PILOT_ABORT_SECONDS = 700
MIN_FREE_DISK_GB = 40


def check_lambda_one_equals_ce(gid, n_act, device):
    """Gate 1: w=1 must equal nn.CrossEntropyLoss in value AND in gradient
    w.r.t. the logits. Uses a LOCAL generator so the global RNG is never
    touched; the global state is asserted unchanged around the check."""
    before = torch.get_rng_state()
    before_cuda = (torch.cuda.get_rng_state_all()
                   if device.type == "cuda" else None)
    gen = torch.Generator(device="cpu").manual_seed(20260922)
    n_cat = gid.numel()
    scores = (torch.randn(48, n_cat, generator=gen) * 3.0).to(device)
    targets = torch.randint(0, n_cat, (48,), generator=gen).to(device)
    a = scores.clone().requires_grad_(True)
    b = scores.clone().requires_grad_(True)
    l_dec, _, _ = decomposed_loss(a, targets, gid, n_act, 1.0)
    l_ce = nn.CrossEntropyLoss()(b, targets)
    (g_dec,) = torch.autograd.grad(l_dec, a)
    (g_ce,) = torch.autograd.grad(l_ce, b)
    v_dec, v_ce = float(l_dec.detach()), float(l_ce.detach())
    out = {
        "loss_decomposed_w1": v_dec, "loss_cross_entropy": v_ce,
        "abs_value_diff": abs(v_dec - v_ce),
        "max_abs_grad_diff": float((g_dec - g_ce).abs().max()),
        "grad_scale": float(g_ce.abs().max()),
        "n_rows": 48, "n_cat": n_cat,
        "rng_untouched": bool(torch.equal(torch.get_rng_state(), before)),
    }
    if before_cuda is not None:
        out["cuda_rng_untouched"] = all(
            torch.equal(x, y) for x, y in
            zip(torch.cuda.get_rng_state_all(), before_cuda))
    assert out["rng_untouched"], "la comprobacion movio el RNG global"
    assert out.get("cuda_rng_untouched", True), "la comprobacion movio el RNG CUDA"
    assert out["abs_value_diff"] < 1e-4, out
    assert out["max_abs_grad_diff"] < 1e-6, out
    return out


def decomposed_loss(scores, targets, gid, n_groups, w):
    """CE split by the ward-stripped grouping; w scales the ward term."""
    logZ = torch.logsumexp(scores, dim=1)
    same = gid.unsqueeze(0) == gid[targets].unsqueeze(1)
    logZ_g = torch.logsumexp(scores.masked_fill(~same, float("-inf")), dim=1)
    s_t = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    main = (logZ - logZ_g).mean()
    ward = (logZ_g - s_t).mean()
    return main + w * ward, float(main.detach()), float(ward.detach())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    cache_stamp = stamp
    epochs = 1 if args.smoke else EPOCHS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(OUT_DIR).free / 1e9
    if not args.smoke and free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(f"disco libre {free_gb:.1f} GB < {MIN_FREE_DISK_GB}")

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
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets_A = [k for k, _ in sorted(catalogue.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets_A).encode()).hexdigest()
    n_cat = len(baskets_A) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- grouping (identical rule to the decoder and the diagnostic) ----
    group_of_key = {None: 0}
    group_main = [Counter()]
    group_ids = torch.zeros(n_cat, dtype=torch.long)
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

    # ---- ward-free catalogue + permutation (the frozen mixture rule) ----
    frozen_mix = json.loads((MIX_DIR / "frozen_mixture_v1.json").read_text(
        encoding="utf-8"))
    wA, wB = frozen_mix["weights"]["A"], frozen_mix["weights"]["B"]
    assert not frozen_mix["weights_tuned"]
    wf_cat = Counter()
    for row in v1.stream_split(train_path, fit_set, True, champ_index,
                               dragon, None):
        t = stripped_truth(row)
        if t:
            wf_cat[basket_key(sorted(t.items()))] += 1
    baskets_B = [k for k, _ in sorted(wf_cat.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
    assert len(baskets_B) + 1 == n_act
    keys_A = [None] + [basket_key(sorted(group_main[g].items()))
                       for g in range(1, n_act)]
    pos_A = {k: i for i, k in enumerate(keys_A)}
    assert len(pos_A) == n_act
    perm_B_to_A = torch.tensor([pos_A[k] for k in [None] + baskets_B],
                               dtype=torch.long)
    assert len(set(perm_B_to_A.tolist())) == n_act, "permutacion no biyectiva"
    same_index = int((perm_B_to_A == torch.arange(n_act)).sum())
    assert same_index == frozen_mix["alignment"]["positions_unchanged"]
    print(f"alineacion: {n_act} acciones, biyectiva, {same_index} intactas",
          flush=True)

    # ---- caches (exactly the fine-tune's) ----
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
    fit_domain = fit["top_idx"].long()
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

    def basket_tensors(baskets, n):
        items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
        idx_map = {it: k + 1 for k, it in enumerate(items)}
        ci = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
        cc = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
        for row_i, b in enumerate(baskets, start=1):
            for slot, part in enumerate(b.split(",")[:v1.MAX_BASKET]):
                i, n_ = part.split(":")
                ci[row_i, slot] = idx_map[int(i)]
                cc[row_i, slot] = min(int(n_), v1.MAX_BASKET)
        return ci.to(device), cc.to(device), len(items)

    cat_idx, cat_cnt, n_items_A = basket_tensors(baskets_A, n_cat)
    cat_idx_B, cat_cnt_B, n_items_B = basket_tensors(baskets_B, n_act)

    def arch(n_items):
        return (len(champ_index), len(item_index), len(label_index),
                len(item_index) + 1, n_items)

    feas = econ_fit[:, :, 1] > 0
    stats_mean, stats_std = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    if flat.numel():
        for d in range(2, N_ECON):
            stats_mean[d] = flat[:, d].mean()
            stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = stats_mean.to(device), stats_std.to(device)
    del flat, feas

    def econ_block(raw):
        """Real economic features in BOTH arms (not the control contrast)."""
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
    gid = group_ids.to(device)
    perm = perm_B_to_A.to(device)
    all_cat = torch.arange(n_cat, device=device)
    all_cat_B = torch.arange(n_act, device=device)

    # frozen ward-free partner for every mixture
    wardfree = v1.BasketNet(*arch(n_items_B)).to(device)
    wardfree.load_state_dict(torch.load(WF_DIR / "wardfree_ep16.pt",
                                        map_location="cpu",
                                        weights_only=False)["state_dict"])
    wardfree.eval()
    lambda_gate = check_lambda_one_equals_ce(gid, n_act, device)
    print(f"puerta lambda=1: |dL| {lambda_gate['abs_value_diff']:.3e}, "
          f"|dgrad|max {lambda_gate['max_abs_grad_diff']:.3e} "
          f"(escala {lambda_gate['grad_scale']:.3e}), RNG intacto "
          f"{lambda_gate['rng_untouched']}", flush=True)
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    def fixed_subset_ce(model):
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
                scores, _ = model.corrected_scores(
                    batch, device, cat_idx, cat_cnt, n_cat,
                    fit_domain[idxs.cpu()].to(device),
                    econ_block(econ_fit[idxs.cpu()]))
                total += float(nll_s(scores, fit_targets[idxs]))
                n += len(idxs)
        model.train()
        return total / max(n, 1)

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

    def score_model(model):
        """Returns (rows_alone, rows_mixed) under the common decode."""
        model.eval()
        alone, mixed = [], []
        with torch.no_grad():
            wvecs = wardfree.baskets.vectors(cat_idx_B, cat_cnt_B, all_cat_B)
            for s in range(0, n_probe, 512):
                sl = slice(s, min(s + 512, n_probe))
                batch = {k: v[sl] for k, v in {
                    "champs": probe_ds.champs, "sides": probe_ds.sides,
                    "items": probe_ds.items, "query": probe_ds.query,
                    "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
                b = batch["champs"].shape[0]
                scores, _ = model.corrected_scores(
                    batch, device, cat_idx, cat_cnt, n_cat,
                    probe_domain[sl].to(device), econ_block(econ_probe[sl]))
                gidx = gid.unsqueeze(0).expand(b, -1)
                gmax = torch.full((b, n_act), float("-inf"), device=device)
                gmax.scatter_reduce_(1, gidx, scores, reduce="amax",
                                     include_self=True)
                gsum = torch.zeros(b, n_act, device=device)
                gsum.scatter_add_(1, gidx, (scores - gmax.gather(1, gidx)).exp())
                P_A = torch.softmax(gmax + gsum.log(), dim=1)
                sB = wardfree.scores(wardfree.state(batch, device), wvecs)
                P_B = torch.zeros_like(P_A)
                P_B.scatter_(1, perm.unsqueeze(0).expand(b, -1),
                             torch.softmax(sB, dim=1))
                top_a = torch.topk(P_A, min(v1.SHORTLIST, n_act),
                                   dim=1).indices.cpu()
                top_m = torch.topk(wA * P_A + wB * P_B,
                                   min(v1.SHORTLIST, n_act), dim=1).indices.cpu()
                for j in range(b):
                    row = probe_rows[s + j]
                    alone.append(wardfree_record(row, decode(top_a[j].tolist(), row)))
                    mixed.append(wardfree_record(row, decode(top_m[j].tolist(), row)))
        return alone, mixed

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/experiment_loss_contrast_v1.md",
              "scope": "internal chronological TRAIN probe; no VAL, no test",
              "mode": stamp, "parent": "Bf_ep16", "lr": LR, "epochs": epochs,
              "micro": p2.MICRO, "effective": p2.EFFECTIVE,
              "arms": ARMS, "econ_features_in_both_arms": True,
              "mixture_partner": "wardfree_ep16",
              "mixture_weights": {"A": wA, "B": wB}, "weights_tuned": False,
              "n_actions": n_act, "permutation_positions_unchanged": same_index,
              "lambda_one_equals_ce_gate": lambda_gate,
              "step0_check": {}, "curves": {}, "aggregates": {},
              "paired": {}, "slices": {}, "complementarity": {}}
    rows = {}

    for arm, w in ARMS.items():
        torch.manual_seed(16)
        base = v1.BasketNet(*arch(n_items_A)).to(device)
        base.load_state_dict(parent_blob["state_dict"])
        model = TunedNet(base, CorrectionNet().to(device))
        with torch.no_grad():
            pb = {k: v[:64] for k, v in {
                "champs": fit_ds.champs, "sides": fit_ds.sides,
                "items": fit_ds.items, "query": fit_ds.query,
                "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
            _, corr0 = model.corrected_scores(
                pb, device, cat_idx, cat_cnt, n_cat,
                fit_domain[:64].to(device), econ_block(econ_fit[:64]))
            max_corr0 = float(corr0.abs().max())
        assert max_corr0 == 0.0, "correction not zero at init"
        ce0 = fixed_subset_ce(model)
        report["step0_check"][arm] = {"max_abs_correction": max_corr0,
                                      "fixed_subset_full_ce": ce0}
        print(f"step0 {arm}: |corr|max {max_corr0:.2e} fixedCE {ce0:.5f}",
              flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        order_gen = torch.Generator().manual_seed(16)
        curve = []
        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.monotonic()
            order = torch.randperm(n_fit, generator=order_gen)
            tot = tot_m = tot_w = 0.0
            n = 0
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
                        fit_domain[idxs].to(device), econ_block(econ_fit[idxs]))
                    loss, lm, lw = decomposed_loss(
                        scores, fit_targets[idxs.to(device)], gid, n_act, w)
                    (loss * len(idxs) / len(idxs_all)).backward()
                    tot += float(loss.detach()) * len(idxs)
                    tot_m += lm * len(idxs)
                    tot_w += lw * len(idxs)
                opt.step()
                n += len(idxs_all)
            sched.step()
            fixed = fixed_subset_ce(model)
            secs = round(time.monotonic() - t_ep, 1)
            free_now = shutil.disk_usage(OUT_DIR).free / 1e9
            point = {"arm": arm, "ward_weight": w, "epoch": epoch,
                     "weighted_loss": tot / max(n, 1),
                     "main_term": tot_m / max(n, 1),
                     "ward_term": tot_w / max(n, 1),
                     "fixed_subset_full_ce": fixed, "seconds": secs,
                     "free_disk_gb": round(free_now, 1),
                     "utc": datetime.now(timezone.utc).isoformat()}
            curve.append(point)
            # persisted as the epoch ends: a run that dies mid-way must still
            # leave its curves behind (we have lost them twice to buffering)
            with open(OUT_DIR / f"progress_{stamp}.jsonl", "a",
                      encoding="utf-8", newline=chr(10)) as fh:
                fh.write(json.dumps(point) + chr(10))
                fh.flush()
                os.fsync(fh.fileno())
            print(f"  {arm} ep{epoch}/{epochs} L {tot / max(n, 1):.4f} "
                  f"(main {tot_m / max(n, 1):.4f} ward {tot_w / max(n, 1):.4f}) "
                  f"fixedCE {fixed:.4f} ({secs}s)", flush=True)
            if epoch == 1 and not args.smoke and secs > PILOT_ABORT_SECONDS:
                raise RuntimeError(
                    f"puerta piloto: epoca 1 {secs}s > {PILOT_ABORT_SECONDS}s")
            if not args.smoke and free_now < MIN_FREE_DISK_GB:
                raise RuntimeError(
                    f"disco libre {free_now:.1f} GB < {MIN_FREE_DISK_GB} "
                    f"tras {arm} epoca {epoch}")
            if epoch in (4, epochs):
                p2.save_checkpoint(OUT_DIR / f"{arm}_ep{epoch}.pt", model,
                                   opt, sched, epoch)
        report["curves"][arm] = curve
        alone, mixed = score_model(model)
        rows[arm] = alone
        rows[f"mix_{arm}"] = mixed
        p2.dump(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", alone)
        p2.dump(OUT_DIR / f"rows_mix_{arm}_{stamp}.jsonl", mixed)
        del model, base, opt, sched
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- frozen references: econ_ft_ep8 alone and the current mixture ----
    ref = TunedNet(v1.BasketNet(*arch(n_items_A)).to(device),
                   CorrectionNet().to(device))
    ref.load_state_dict(torch.load(FT_DIR / "econ_ft_ep8.pt",
                                   map_location="cpu",
                                   weights_only=False)["state_dict"])
    rows["econ_ft_ep8"], rows["mix_frozen"] = score_model(ref)
    del ref
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # integrity: the frozen references must reproduce their persisted rows
    def canon(p_):
        return ("SAVE" if p_ == "SAVE"
                else {str(k): int(v) for k, v in p_.items()})

    def mismatches(recs, path):
        if not Path(path).exists():
            return None
        ref_rows = [json.loads(l) for l in open(path, encoding="utf-8")][:len(recs)]
        if len(ref_rows) != len(recs):
            return {"rows_compared": len(ref_rows), "aligned": False}
        bad = {"first_action_exact": 0, "pred_first": 0,
               "any_of_three_exact": 0}
        for a, b_ in zip(recs, ref_rows):
            # identity keys on champion too: the ten perspectives of a game
            # share timestamps and (match_id, ts) has already collided here
            assert ((a["match_id"], a["ts"], a.get("champion"))
                    == (b_["match_id"], b_["ts"], b_.get("champion"))),                 "identidad de fila desalineada"
            bad["pred_first"] += canon(a["pred_first"]) != canon(b_["pred_first"])
            for k in ("first_action_exact", "any_of_three_exact"):
                bad[k] += a[k] != b_[k]
        return bad

    report["reproduction_mismatches"] = {
        "econ_ft_ep8_alone_vs_grouped_rows": mismatches(
            rows["econ_ft_ep8"], WF_DIR / "rows_grouped_econft_full.jsonl"),
        "mix_frozen_vs_mixture_rows": mismatches(
            rows["mix_frozen"], MIX_DIR / "rows_M_full.jsonl"),
        "note": ("in --smoke the normalization stats come from the SMOKE econ "
                 "cache, not full FIT, so a handful of tie-break flips against "
                 "the persisted rows is expected there and ONLY there; in full "
                 "mode the stats are byte-identical to the decoder's and these "
                 "counts must be 0"),
    }
    print("reproduccion:", json.dumps(report["reproduction_mismatches"]),
          flush=True)
    if not args.smoke:
        for name in ("econ_ft_ep8_alone_vs_grouped_rows",
                     "mix_frozen_vs_mixture_rows"):
            bad = report["reproduction_mismatches"][name]
            if bad is None:
                raise RuntimeError(
                    f"referencia congelada ausente para {name}: sin ella las "
                    "comparaciones no son atribuibles")
            if any(v for v in bad.values()):
                raise RuntimeError(
                    f"puerta de reproduccion fallida en {name}: {bad}")
    report["reproduction_gate"] = ("hard in full mode: any mismatch or a "
                                   "missing frozen reference aborts the run")

    # ---- complementarity of each economic arm against wardfree ----
    wf_rows = [json.loads(l) for l in open(
        WF_DIR / "rows_wardfree_full.jsonl", encoding="utf-8")][:n_probe]
    for tag in ("econ_ft_ep8", "w1", "w025"):
        if tag not in rows:
            continue
        diff_first = diff_corr = only_x = only_y = union = 0
        for a, b_ in zip(rows[tag], wf_rows):
            assert ((a["match_id"], a["ts"], a.get("champion"))
                    == (b_["match_id"], b_["ts"], b_.get("champion")))
            diff_first += canon(a["pred_first"]) != canon(b_["pred_first"])
            x, y = a["first_action_exact"], b_["first_action_exact"]
            diff_corr += x != y
            only_x += x > y
            only_y += y > x
            union += bool(x or y)
        report["complementarity"][f"{tag}_vs_wardfree"] = {
            "n": len(wf_rows),
            "different_first_recommendation": diff_first,
            "different_correctness": diff_corr,
            "arm_only_correct": only_x, "wardfree_only_correct": only_y,
            "oracle_union_rate": union / max(len(wf_rows), 1)}

    # ---------------------------- metrics ----------------------------
    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    # every paired comparison below zips by position: prove the row lists are
    # the same decisions in the same order, keyed on champion too
    ident = [(r["match_id"], r["ts"], r.get("champion")) for r in rows["w1"]]
    assert len(set(ident)) == len(ident), "claves de identidad duplicadas"
    for tag, recs in rows.items():
        assert [(r["match_id"], r["ts"], r.get("champion"))
                for r in recs] == ident, f"identidad desalineada en {tag}"
    report["row_identity"] = {"key": "(match_id, ts, champion)",
                              "n": len(ident), "aligned_across_arms": True}
    for tag, recs in rows.items():
        report["aggregates"][tag] = p2.aggregate(recs)
    comparisons = [
        ("mix_w025", "mix_w1"),          # PRIMARY 1: attribution
        ("mix_w025", "mix_frozen"),      # PRIMARY 2: practical progress
        ("w1", "econ_ft_ep8"),           # noise floor of this contrast
        ("w025", "w1"),                  # the loss change on the arm alone
        ("w025", "econ_ft_ep8"),
        ("mix_w1", "mix_frozen"),
    ]
    for x, y in comparisons:
        if x not in rows or y not in rows:
            continue
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{y}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(rows[x], rows[y], metric)
        for label, pred in subsets.items():
            report["slices"][f"{x}_vs_{y}_{label}"] = p2.paired(
                [r for r in rows[x] if pred(r)],
                [r for r in rows[y] if pred(r)], "first_action_exact")
    for x, y in comparisons[:3]:
        if x not in rows or y not in rows:
            continue
        per = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
        for t, recs in (("x", rows[x]), ("y", rows[y])):
            for r in recs:
                per[t][r["match_id"]][0] += r["first_action_exact"]
                per[t][r["match_id"]][1] += 1
        report["paired"][f"{x}_vs_{y}"]["ci95_2000draws"] = list(
            v1.paired_bootstrap(per["x"], per["y"], seed=16, n=2000))
    report["reading_rules"] = {
        "w1_vs_econ_ft_ep8": (
            "REPRODUCTION DIAGNOSTIC, not a noise floor and not an estimate "
            "of training variability: same recipe, seeds and batch order, "
            "only the loss EXPRESSION differs, so the two should agree. A "
            "large discrepancy is a reason to investigate the implementation; "
            "nothing is declared uninterpretable on the basis of its size."),
        "primaries": (
            "Both primaries compare MIXTURES. PRIMARY 1 positive with "
            "PRIMARY 2 unresolved means: improvement over the twin control "
            "mixture, no demonstrated improvement over the frozen mixture. "
            "It does NOT establish overlap with the ward-free model - the "
            "same pattern arises if mix_w1 simply regressed, which "
            "mix_w1_vs_mix_frozen reports. Arm-alone results (w025 vs w1, "
            "each vs econ_ft_ep8) are separate questions and are never read "
            "as product results."),
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"loss_contrast_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": {k: v["first_action_exact"]
                                     for k, v in report["aggregates"].items()},
                      "paired": report["paired"],
                      "complementarity": report["complementarity"]},
                     indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
