"""Purchase-history contrast: residual twin arms over a frozen backbone.

Protocol: docs/experiment_history_contrast_v1.md.

Frozen: econ_ft_ep8 (the economic arm) and wardfree_ep16 (its mixture
partner), both bound by sha256 from frozen_mixture_v1.json. A small
zero-init scoring correction is trained on top; step 0 reproduces the
parent exactly. Twin arms differ in ONE thing:

    hist    : the correction sees the player's prior purchases
    control : it sees that same block ZEROED

CS, assists and inter-visit timings are deliberately excluded so the
signal under test is purchase history alone.

Both arms are scored alone AND mixed 50/50 with the frozen ward-free model
under the frozen mixture rule. Primaries compare MIXTURES. A declared
secondary re-scores the candidate with history truncated to simulate the
live cold start, because `app/` cannot supply history today.

Requires scripts/build_history_cache.py to have run: the ordinary caches
carry an all-zero `hist` and are refused here by fingerprint.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] uses the smoke
caches and 2 epochs.
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

OUT_DIR = ROOT / "data" / "experiments" / "16.18-history-contrast-v1"
MIX_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
LR = 1e-4
EPOCHS = 6
ARMS = ("control", "hist")
MIN_FREE_DISK_GB = 40
HIDE_CUT = 0.5   # share of each game's earliest decisions whose history is hidden
# NOT a late-attach simulation. After the cut this returns the FULL history,
# including purchases made before the supposed attach, so it measures
# SENSITIVITY TO HIDING HISTORY and nothing else. A real simulation would fix
# a join instant and accumulate only purchases observed after it, never
# recovering earlier ones. Kept under its true name and scope.


class HistoryCorrection(nn.Module):
    """Residual score correction on [state | basket_vec | history-as-a-SET].

    The history block is a SET encoding of the last HIST_LEN purchased
    units. It is NOT a sequence encoder and must not be described as one:
    the position embedding is ADDED before pooling and the pooling is a
    mean, so mean(e_i + p_i) over occupied slots = mean(e) + mean(p over
    occupied). Swapping two items between occupied positions gives an
    identical vector. The positions can therefore inform HOW MANY slots
    are filled, never WHICH item came first.

    So this pilot tests whether the SET of recent purchases carries extra
    signal. A result here says nothing about the value of ordered
    sequence, and no claim about click order or canonical order applies --
    order is not read at all. Zero-init last layer: step 0 == parent.
    """

    def __init__(self, n_item, hist_dim=48):
        super().__init__()
        self.emb = nn.Embedding(n_item + 1, hist_dim, padding_idx=0)
        self.pos = nn.Embedding(tp.HIST_LEN, hist_dim)
        self.pool = nn.Sequential(nn.Linear(hist_dim, hist_dim), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(v1.STATE_DIM * 2 + hist_dim + 1, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def encode(self, hist):
        """hist [B, HIST_LEN] -> ([B, hist_dim] SET code, [B,1] has-history).

        Permutation-invariant over occupied slots, by construction.
        """
        pos = torch.arange(tp.HIST_LEN, device=hist.device).unsqueeze(0)
        mask = (hist != 0).float().unsqueeze(-1)
        tok = (self.emb(hist) + self.pos(pos)) * mask
        pooled = tok.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.pool(pooled), (mask.sum(dim=1) > 0).float()

    def forward(self, state, basket_vecs, hist):
        h, has = self.encode(hist)
        k = basket_vecs.shape[1]
        s = state.unsqueeze(1).expand(-1, k, -1)
        hh = h.unsqueeze(1).expand(-1, k, -1)
        aa = has.unsqueeze(1).expand(-1, k, -1)
        return self.net(torch.cat([s, basket_vecs, hh, aa], dim=-1)).squeeze(-1)


class HistTunedNet(nn.Module):
    """Frozen parent + history correction added on the fixed domain."""

    def __init__(self, parent, correction):
        super().__init__()
        self.parent = parent
        self.correction = correction
        for p in self.parent.parameters():
            p.requires_grad_(False)

    def scores(self, batch, device, cat_idx, cat_cnt, n_cat, domain, econ,
               hist):
        # The frozen backbone MUST keep seeing an all-zero history.
        #
        # Measured, 3,000 FIT rows: feeding it a populated `hist` moves its
        # own CE from 1.628 to 3.865. It was trained with those 12 tokens
        # always padded, so its `hist_pos` embedding was never updated;
        # activating them drops untrained vectors into the attention input.
        # Giving the parent real history does not "add information", it
        # corrupts the model we froze. Only the correction sees history.
        blind = dict(batch)
        blind["hist"] = torch.zeros_like(batch["hist"])
        with torch.no_grad():
            base, _ = self.parent.corrected_scores(
                blind, device, cat_idx, cat_cnt, n_cat, domain, econ)
            state = self.parent.base.state(blind, device)
            vecs = self.parent.base.baskets.vectors(
                cat_idx, cat_cnt, torch.arange(n_cat, device=device))
        corr = self.correction(state, vecs[domain], hist)
        return base.scatter_add(1, domain, corr), corr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    cache_stamp = stamp
    epochs = 2 if args.smoke else EPOCHS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.smoke and shutil.disk_usage(OUT_DIR).free / 1e9 < MIN_FREE_DISK_GB:
        raise RuntimeError("disco libre insuficiente")

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    schema = json.loads((generation / "split_manifest.json").read_text(
        encoding="utf-8"))["input_schema"]
    tp.MAX_ITEMS = schema["max_modeled_inventory_items"]
    tp.MAX_COPIES = schema["max_copies_of_one_item"]
    tp.MAX_BASKET_ITEMS = schema["max_basket_items"]
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    plan = json.loads((generation / "split_plan.json").read_text(
        encoding="utf-8"))
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
    n_cat = len(baskets_A) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- grouping + ward-free permutation (the frozen mixture rule) ----
    frozen_mix = json.loads((MIX_DIR / "frozen_mixture_v1.json").read_text(
        encoding="utf-8"))
    wA, wB = frozen_mix["weights"]["A"], frozen_mix["weights"]["B"]
    assert not frozen_mix["weights_tuned"]
    for arm_key in ("arm_A_grouped_econ", "arm_B_wardfree"):
        for k, spec in frozen_mix["components"][arm_key].items():
            if k == "role":
                continue
            got = hashlib.sha256((ROOT / spec["path"]).read_bytes()).hexdigest()
            assert got == spec["sha256"], f"digest cambiado: {spec['path']}"

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
    perm = torch.tensor([pos_A[k] for k in [None] + baskets_B],
                        dtype=torch.long)
    assert len(set(perm.tolist())) == n_act, "permutacion no biyectiva"
    same_index = int((perm == torch.arange(n_act)).sum())
    assert same_index == frozen_mix["alignment"]["positions_unchanged"]
    print(f"alineacion: {n_act} acciones, biyectiva, {same_index} intactas",
          flush=True)

    # ---- econ cache (frozen domain + features), unchanged ----
    cache_report = json.loads(
        (CACHE_DIR / f"cache_report_{cache_stamp}.json").read_text(
            encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    fit_c = torch.load(CACHE_DIR / f"frozen_fit_{cache_stamp}.pt",
                       weights_only=True)
    probe_c = torch.load(CACHE_DIR / f"frozen_probe_{cache_stamp}.pt",
                         weights_only=True)
    econ_fit = torch.load(CACHE_DIR / f"econ_fit_{cache_stamp}.pt",
                          weights_only=True)
    econ_probe = torch.load(CACHE_DIR / f"econ_probe_{cache_stamp}.pt",
                            weights_only=True)
    n_fit, n_probe = fit_c["n_rows"], probe_c["n_rows"]
    fit_domain, probe_domain = fit_c["top_idx"].long(), probe_c["top_idx"].long()

    # ---- HISTORY caches: the FULL featurization is asserted, not just the
    # history flag. Checking only `history` would have let through a cache
    # built with default extras/gold flags -- a different QUERY regime that
    # the frozen backbone never saw. That happened in this script's first
    # smoke and is why the expectation below is the whole fingerprint.
    base_config = torch.load(
        ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt",
        map_location="cpu", weights_only=False)["config"]
    expect = {k: base_config[k] for k in
              ("extras", "gold_est", "gold_x", "runes", "pov_gold",
               "query_dim", "max_items", "max_copies", "inventory_encoding")}
    expect["history"] = True
    expect["hist_len"] = tp.HIST_LEN
    suffix = "_smoke" if args.smoke else ""
    fit_ds = tp.ShopDataset.from_cache(
        OUT_DIR / f"cache_fit_hist_v1{suffix}.pt", champ_index, item_index,
        label_index, dragon, expect=expect)
    probe_ds = tp.ShopDataset.from_cache(
        OUT_DIR / f"cache_probe_hist_v1{suffix}.pt", champ_index, item_index,
        label_index, dragon, expect=expect)
    print(f"featurizacion de la cache verificada al completo: "
          f"{json.dumps(expect)}", flush=True)
    for name, ds in (("fit", fit_ds), ("probe", probe_ds)):
        nz = int((ds.hist != 0).sum())
        if nz == 0:
            raise RuntimeError(f"{name}: hist a ceros pese al fingerprint")
        print(f"  cache {name}: hist no-cero {nz}, "
              f"{float((ds.hist != 0).any(dim=1).float().mean()):.3f} de filas",
              flush=True)
    assert fit_ds.champs.shape[0] == n_fit, "cache fit desalineada con econ"
    assert probe_ds.champs.shape[0] == n_probe, "cache probe desalineada"

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    fit_targets = torch.load(v1_dir / "cache_targets_full_v2.pt",
                             weights_only=True)[:n_fit]
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

    # Normalization stats come from the FULL FIT econ cache in every mode.
    # Deriving them from the smoke cache changes the parent's scores and
    # leaves the reproduction gate with a built-in "expected" mismatch --
    # a gate with a known false positive is not a gate.
    stats_src = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas = stats_src[:, :, 1] > 0
    sm_, ss_ = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = stats_src[feas].float()
    for d in range(2, N_ECON):
        sm_[d] = flat[:, d].mean()
        ss_[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = sm_.to(device), ss_.to(device)
    del flat, feas, stats_src

    def econ_block(raw):
        return (raw.float().to(device) - sm) / ss

    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, n_probe if args.smoke else None))
    digest = hashlib.sha256()
    for row in probe_rows:
        digest.update(f"{row['match_id']}|{row['ts']};".encode())
    assert digest.hexdigest() == probe_c["rows_sha256"], "probe desalineado"

    parent = TunedNet(v1.BasketNet(*arch(n_items_A)).to(device),
                      CorrectionNet().to(device))
    parent.load_state_dict(torch.load(FT_DIR / "econ_ft_ep8.pt",
                                      map_location="cpu",
                                      weights_only=False)["state_dict"])
    parent.eval()
    wardfree = v1.BasketNet(*arch(n_items_B)).to(device)
    wardfree.load_state_dict(torch.load(WF_DIR / "wardfree_ep16.pt",
                                        map_location="cpu",
                                        weights_only=False)["state_dict"])
    wardfree.eval()
    gid = group_ids.to(device)
    permd = perm.to(device)
    all_cat_B = torch.arange(n_act, device=device)

    # cold-start mask: hide history for the earliest HIDE_CUT of each
    # game's probe decisions -- what a mid-game attach would actually know
    by_game = defaultdict(list)
    for i, row in enumerate(probe_rows):
        by_game[row["match_id"]].append(i)
    hidden_mask = torch.zeros(n_probe, dtype=torch.bool)
    for idxs in by_game.values():
        cut = int(len(idxs) * HIDE_CUT)
        for i in idxs[:cut]:
            hidden_mask[i] = True
    print(f"prueba de ocultacion (NO es enganche tardio): oculto en "
          f"{int(hidden_mask.sum())}/{n_probe} decisiones", flush=True)

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

    def hist_block(raw_hist, arm, hide=None):
        h = raw_hist.long().to(device)
        if arm == "control":
            return torch.zeros_like(h)
        if hide is not None:
            h = h.clone()
            h[hide.to(device)] = 0
        return h

    def score(model, arm, hide=None):
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
                # EVERY frozen model here (the econ backbone AND the ward-free
                # partner) was trained with the history tokens always padded.
                # Real history in their input corrupts them -- measured: the
                # backbone's own CE goes 1.628 -> 3.865. History reaches the
                # correction head ONLY.
                batch["hist"] = torch.zeros_like(batch["hist"])
                hb = hist_block(probe_ds.hist[sl], arm,
                                None if hide is None else hide[sl])
                sc, _ = model.scores(batch, device, cat_idx, cat_cnt, n_cat,
                                     probe_domain[sl].to(device),
                                     econ_block(econ_probe[sl]), hb)
                gidx = gid.unsqueeze(0).expand(b, -1)
                gmax = torch.full((b, n_act), float("-inf"), device=device)
                gmax.scatter_reduce_(1, gidx, sc, reduce="amax",
                                     include_self=True)
                gsum = torch.zeros(b, n_act, device=device)
                gsum.scatter_add_(1, gidx, (sc - gmax.gather(1, gidx)).exp())
                P_A = torch.softmax(gmax + gsum.log(), dim=1)
                sB = wardfree.scores(wardfree.state(batch, device), wvecs)
                P_B = torch.zeros_like(P_A)
                P_B.scatter_(1, permd.unsqueeze(0).expand(b, -1),
                             torch.softmax(sB, dim=1))
                ta = torch.topk(P_A, min(v1.SHORTLIST, n_act), dim=1).indices.cpu()
                tm = torch.topk(wA * P_A + wB * P_B,
                                min(v1.SHORTLIST, n_act), dim=1).indices.cpu()
                for j in range(b):
                    row = probe_rows[s + j]
                    alone.append(wardfree_record(row, decode(ta[j].tolist(), row)))
                    mixed.append(wardfree_record(row, decode(tm[j].tolist(), row)))
        return alone, mixed

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "docs/experiment_history_contrast_v1.md",
              "scope": "internal chronological TRAIN probe; no VAL, no test",
              "mode": stamp, "epochs": epochs, "lr": LR,
              "frozen_parent": "econ_ft_ep8", "mixture_partner": "wardfree_ep16",
              "mixture_weights": {"A": wA, "B": wB}, "weights_tuned": False,
              "n_actions": n_act, "permutation_positions_unchanged": same_index,
              "history_cache": {
                  "fit_hist_nonzero": int((fit_ds.hist != 0).sum()),
                  "probe_hist_nonzero": int((probe_ds.hist != 0).sum()),
                  "fit_meta": fit_ds.cache_meta, "probe_meta": probe_ds.cache_meta},
              "history_hiding_probe": {"cut": HIDE_CUT,
                                       "hidden_decisions": int(hidden_mask.sum()),
                                       "is_late_attach_simulation": False},
              "step0_check": {}, "curves": {}, "aggregates": {},
              "paired": {}, "slices": {}, "complementarity": {}}
    rows = {}
    nll = nn.CrossEntropyLoss()

    # ---- GATE, BEFORE TRAINING ANYTHING ----------------------------
    # Score the frozen references and check reproduction first: two arms
    # cost real GPU time, and a failed gate invalidates every comparison
    # they would produce. Normalization comes from the FULL FIT cache in
    # every mode, so this gate has no built-in expected mismatch.
    class _Passthrough(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def scores(self, batch, device, cat_idx, cat_cnt, n_cat, domain,
                   econ, hist):
            # blind to history for the same reason as HistTunedNet: this is
            # the FROZEN reference and must reproduce the persisted rows
            blind = dict(batch)
            blind["hist"] = torch.zeros_like(batch["hist"])
            return self.parent.corrected_scores(blind, device, cat_idx,
                                                cat_cnt, n_cat, domain, econ)

    rows["econ_ft_ep8"], rows["mix_frozen"] = score(_Passthrough(parent),
                                                    "control")

    def canon(x):
        return ("SAVE" if x == "SAVE" else {str(k): int(v) for k, v in x.items()})

    def mismatches(recs, path):
        if not Path(path).exists():
            return None
        ref = [json.loads(l) for l in open(path, encoding="utf-8")][:len(recs)]
        if len(ref) != len(recs):
            return {"aligned": False, "rows_compared": len(ref)}
        bad = {"first_action_exact": 0, "pred_first": 0, "any_of_three_exact": 0}
        for a, b_ in zip(recs, ref):
            assert ((a["match_id"], a["ts"], a.get("champion"))
                    == (b_["match_id"], b_["ts"], b_.get("champion")))
            bad["pred_first"] += canon(a["pred_first"]) != canon(b_["pred_first"])
            for k in ("first_action_exact", "any_of_three_exact"):
                bad[k] += a[k] != b_[k]
        return bad

    report["reproduction_mismatches"] = {
        "econ_ft_ep8_alone": mismatches(rows["econ_ft_ep8"],
                                        WF_DIR / "rows_grouped_econft_full.jsonl"),
        "mix_frozen": mismatches(rows["mix_frozen"], MIX_DIR / "rows_M_full.jsonl"),
    }
    print("reproduccion:", json.dumps(report["reproduction_mismatches"]),
          flush=True)
    for name, bad in report["reproduction_mismatches"].items():
            if bad is None:
                raise RuntimeError(f"referencia congelada ausente: {name}")
            if any(v for v in bad.values() if isinstance(v, int)):
                raise RuntimeError(f"puerta de reproduccion fallida: {name} {bad}")


    for arm in ARMS:
        torch.manual_seed(16)
        model = HistTunedNet(parent, HistoryCorrection(len(item_index)).to(device))
        with torch.no_grad():
            pb = {k: v[:64] for k, v in {
                "champs": fit_ds.champs, "sides": fit_ds.sides,
                "items": fit_ds.items, "query": fit_ds.query,
                "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
            _, c0 = model.scores(pb, device, cat_idx, cat_cnt, n_cat,
                                 fit_domain[:64].to(device),
                                 econ_block(econ_fit[:64]),
                                 hist_block(fit_ds.hist[:64], arm))
            max_c0 = float(c0.abs().max())
        assert max_c0 == 0.0, "la correccion no es cero en el paso 0"
        report["step0_check"][arm] = {"max_abs_correction": max_c0}
        print(f"step0 {arm}: |corr|max {max_c0:.2e}", flush=True)

        params = [p for p in model.correction.parameters()]
        opt = torch.optim.Adam(params, lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        order_gen = torch.Generator().manual_seed(16)
        curve = []
        for epoch in range(1, epochs + 1):
            model.correction.train()
            t_ep = time.monotonic()
            order = torch.randperm(n_fit, generator=order_gen)
            tot, n = 0.0, 0
            for s in range(0, n_fit, p2.EFFECTIVE):
                idxs_all = order[s:s + p2.EFFECTIVE]
                opt.zero_grad()
                for m0 in range(0, len(idxs_all), p2.MICRO):
                    idxs = idxs_all[m0:m0 + p2.MICRO]
                    batch = {k: v[idxs.to(v.device)] for k, v in {
                        "champs": fit_ds.champs, "sides": fit_ds.sides,
                        "items": fit_ds.items, "query": fit_ds.query,
                        "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
                    batch["hist"] = torch.zeros_like(batch["hist"])
                    sc, _ = model.scores(
                        batch, device, cat_idx, cat_cnt, n_cat,
                        fit_domain[idxs].to(device), econ_block(econ_fit[idxs]),
                        hist_block(fit_ds.hist[idxs.to(fit_ds.hist.device)], arm))
                    loss = nll(sc, fit_targets[idxs.to(device)])
                    (loss * len(idxs) / len(idxs_all)).backward()
                    tot += float(loss.detach()) * len(idxs)
                opt.step()
                n += len(idxs_all)
            sched.step()
            secs = round(time.monotonic() - t_ep, 1)
            point = {"arm": arm, "epoch": epoch, "train_loss": tot / max(n, 1),
                     "seconds": secs,
                     "free_disk_gb": round(shutil.disk_usage(OUT_DIR).free / 1e9, 1),
                     "utc": datetime.now(timezone.utc).isoformat()}
            curve.append(point)
            with open(OUT_DIR / f"progress_{stamp}.jsonl", "a",
                      encoding="utf-8", newline=chr(10)) as fh:
                fh.write(json.dumps(point) + chr(10))
                fh.flush()
                os.fsync(fh.fileno())
            print(f"  {arm} ep{epoch}/{epochs} loss {tot / max(n, 1):.4f} "
                  f"({secs}s)", flush=True)
        report["curves"][arm] = curve
        torch.save({"state_dict": model.correction.state_dict(), "arm": arm},
                   OUT_DIR / f"correction_{arm}_{stamp}.pt")
        alone, mixed = score(model, arm)
        rows[arm] = alone
        rows[f"mix_{arm}"] = mixed
        p2.dump(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", alone)
        p2.dump(OUT_DIR / f"rows_mix_{arm}_{stamp}.jsonl", mixed)
        if arm == "hist":                      # declared hiding-sensitivity probe
            cold_alone, cold_mixed = score(model, arm, hide=hidden_mask)
            rows["hist_hidden"] = cold_alone
            rows["mix_hist_hidden"] = cold_mixed
            p2.dump(OUT_DIR / f"rows_mix_hist_hidden_{stamp}.jsonl", cold_mixed)
        del model, opt, sched
        if device.type == "cuda":
            torch.cuda.empty_cache()


    ident = [(r["match_id"], r["ts"], r.get("champion")) for r in rows["control"]]
    for tag, recs in rows.items():
        assert [(r["match_id"], r["ts"], r.get("champion"))
                for r in recs] == ident, f"identidad desalineada en {tag}"
    report["row_identity"] = {"key": "(match_id, ts, champion)", "n": len(ident)}

    wf_rows = [json.loads(l) for l in open(
        WF_DIR / "rows_wardfree_full.jsonl", encoding="utf-8")][:n_probe]
    for tag in ("econ_ft_ep8", "control", "hist"):
        d_first = d_corr = only_x = only_y = union = 0
        for a, b_ in zip(rows[tag], wf_rows):
            d_first += canon(a["pred_first"]) != canon(b_["pred_first"])
            x, y = a["first_action_exact"], b_["first_action_exact"]
            d_corr += x != y
            only_x += x > y
            only_y += y > x
            union += bool(x or y)
        report["complementarity"][f"{tag}_vs_wardfree"] = {
            "n": len(wf_rows), "different_first_recommendation": d_first,
            "different_correctness": d_corr, "arm_only_correct": only_x,
            "wardfree_only_correct": only_y,
            "oracle_union_rate": union / max(len(wf_rows), 1)}

    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    for tag, recs in rows.items():
        report["aggregates"][tag] = p2.aggregate(recs)
    comparisons = [
        ("mix_hist", "mix_control"),            # PRIMARY 1
        ("mix_hist", "mix_frozen"),             # PRIMARY 2
        ("mix_control", "mix_frozen"),          # did the control regress?
        ("hist", "control"),
        ("hist", "econ_ft_ep8"),
        ("mix_hist_hidden", "mix_hist"),     # declared cold-start cost
        ("mix_hist_hidden", "mix_frozen"),
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
    for x, y in comparisons[:2]:
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
        "primaries": ("Both primaries compare MIXTURES. PRIMARY 1 positive "
                      "with PRIMARY 2 unresolved means: better than the twin "
                      "control mixture, no demonstrated improvement over the "
                      "frozen mixture. It does NOT establish overlap with the "
                      "ward-free model -- a regressed control gives the same "
                      "pattern, which mix_control_vs_mix_frozen reports. "
                      "Arm-alone results are a separate question."),
        "history_hiding_probe": (
            f"mix_hist_hidden hides history for the earliest {HIDE_CUT:.0%} "
            "of each game's decisions. It is NOT a late-attach simulation: "
            "after the cut it returns the FULL history, including purchases "
            "made before the supposed join, so it measures SENSITIVITY TO "
            "HIDING HISTORY and nothing else. A real simulation would fix a "
            "join instant and accumulate only purchases observed after it."),
        "live_scope": (
            "app/ cannot supply history today (prefix_items exists nowhere "
            "in app/). The full-history result is therefore a DIAGNOSTIC "
            "UNDER IDEAL INFORMATION -- not a mathematical upper bound on "
            "live accuracy, which is a different quantity this experiment "
            "does not measure."),
        "zeroing_is_not_the_parent": (
            "Zeroing the history block does NOT return the parent: the "
            "correction keeps its learned biases and still receives the "
            "state and the basket vector. mix_control is the twin control "
            "for that reason; mix_frozen is the only true parent reference."),
        "negative_scope": ("A negative rules out THIS correction on THIS "
                           "representation over THIS frozen backbone -- not "
                           "purchase history, not a from-scratch model with "
                           "history, not inter-visit timings, not CS/assists."),
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = OUT_DIR / f"history_contrast_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregates": {k: v["first_action_exact"]
                                     for k, v in report["aggregates"].items()},
                      "paired": report["paired"]}, indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
