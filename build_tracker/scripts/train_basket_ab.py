"""A/B contrast for experiment 16.18-whole-basket-v1 (see docs/experiment_whole_basket_v1.md).

Arm A — incumbent formulation: PrefixModel trained by the canonical two-stage
recipe (trunk 16 epochs, frozen-trunk graft 6) on the 2,000 fit games only,
saved as a standard artifact and decoded through the REAL app.predictor path
(predict_options, allow_save, budget_slack=0), exactly as eval_policy scores
production.

Arm B — whole-basket scorer: the SAME PrefixModel trunk (clean init) + a
permutation-invariant Deep Sets basket encoder over the fit-side catalogue,
scored with bilinear + deep interaction terms and trained on full-catalogue
softmax NLL of the complete basket (SAVE is an explicit catalogue action).

Shared contract: POV inputs, corrected wallet contract (CACHE_VERSION v15),
identical tensors for the shared parts, fit/probe memberships verified against
the frozen diagnostic digests. The probe (500 later games) is internal
development only; no validation or test payload is read. Measured wallets are
joined as a DIAGNOSTIC only.

Run through the activated 16.18 runtime:
  .venv/Scripts/python.exe scripts/footwear_runtime.py --entry scripts/train_basket_ab.py -- [--smoke]
with PREFIX_ML_DIR pointing at the v3 generation.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

# The experiment's fixed feature recipe — set BEFORE train_prefix import.
os.environ.setdefault("PREFIX_GOLDEST", "1")
os.environ.setdefault("PREFIX_GOLDX", "1")
os.environ.setdefault("PREFIX_RUNES", "1")
os.environ.setdefault("PREFIX_POV_GOLD", "1")
os.environ.setdefault("PREFIX_POSW_SCHED", "8:24")
os.environ.setdefault("PREFIX_COSINE", "1")

import torch
import torch.nn as nn

import train_prefix as tp
from basket_vocabulary_probe import (basket_entries, basket_key,
                                     membership_digest, split_fit_probe)

FROZEN_FIT_SHA = "d534b9f7a2b7c0e54e02279212091d0965407803388e8059a1d69d2515003dba"
FROZEN_PROBE_SHA = "f53db64a07d55689a15303320d3e7af4b1ab441376d03b4f101221862f468765"
SAVE_SENTINEL = 999999
MAX_BASKET = 6
SHORTLIST = 128
STATE_DIM = 128
RUN_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
BOOTSTRAP = 400


# --------------------------------------------------------------------- data

def stream_split(path, fit_set, want_fit, champ_index, dragon, cap=None):
    rows = tp.fix_champ_ids(tp.with_targets(tp.with_history(
        tp.stream_rows(path)), dragon), champ_index)
    n = 0
    for row in rows:
        if (row["match_id"] in fit_set) != want_fit:
            continue
        yield row
        n += 1
        if cap and n >= cap:
            return


def first_pass(path, fit_set, cap_fit=None, cap_probe=None):
    """One stream over TRAIN: vocab from FIT only + row counts + catalogue."""
    champ_ids, item_ids = set(), set()
    label_counts: Counter = Counter()
    catalogue: Counter = Counter()
    n_fit = n_probe = 0
    for row in tp.stream_rows(path):
        in_fit = row["match_id"] in fit_set
        if in_fit:
            if cap_fit and n_fit >= cap_fit:
                continue
            n_fit += 1
            champ_ids.add(int(row.get("champion_id") or 0))
            for p in row.get("others") or []:
                cid = p.get("champion_id") or (
                    abs(hash(p.get("champion") or "")) % 10000 + 1)
                champ_ids.add(int(cid))
                item_ids.update(int(i) for i in (p.get("items") or []) if i)
            item_ids.update(int(i) for i in (row.get("inventory") or []) if i)
            for item_id in tp.purchase_counts(row):
                if item_id:
                    label_counts[int(item_id)] += 1
            entries = basket_entries(row.get("label_counts"))
            if entries and not any(i == SAVE_SENTINEL for i, _ in entries):
                catalogue[basket_key(entries)] += 1
        else:
            if cap_probe and n_probe >= cap_probe:
                continue
            n_probe += 1
    champ_ids.discard(0)
    return champ_ids, item_ids, label_counts, catalogue, n_fit, n_probe


# ---------------------------------------------------------------- arm B net

class BasketEncoder(nn.Module):
    """Deep Sets over (item, count) embeddings; index 0 is the SAVE action."""

    def __init__(self, n_items: int, dim: int = STATE_DIM):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, 64, padding_idx=0)
        self.count = nn.Embedding(MAX_BASKET + 1, 16, padding_idx=0)
        self.phi = nn.Sequential(nn.Linear(80, 128), nn.GELU(),
                                 nn.Linear(128, 128))
        self.rho = nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, dim))
        self.save_vec = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, idx, cnt):
        x = torch.cat([self.item(idx), self.count(cnt)], dim=-1)
        x = self.phi(x) * (idx > 0).unsqueeze(-1)
        return self.rho(x.sum(dim=1))

    def vectors(self, cat_idx, cat_cnt, members):
        """Basket vectors for catalogue members; SAVE (id 0) -> save_vec."""
        out = self.forward(cat_idx[members], cat_cnt[members])
        save_mask = members == 0
        if bool(save_mask.any()):
            out = out.clone()
            out[save_mask] = self.save_vec
        return out


class BasketNet(nn.Module):
    """PrefixModel trunk (identical featurization) + whole-basket scorer."""

    def __init__(self, n_champ, n_item, n_label, inv_dim, n_basket_items):
        super().__init__()
        self.trunk = tp.PrefixModel(n_champ, n_item, n_label, inv_dim)
        self.state_proj = nn.Sequential(nn.Linear(2 * tp.D_MODEL, 256),
                                        nn.GELU(), nn.Linear(256, STATE_DIM))
        self.baskets = BasketEncoder(n_basket_items)
        self.deep = nn.Sequential(nn.Linear(STATE_DIM * 3, 256), nn.GELU(),
                                  nn.Linear(256, 64), nn.GELU(),
                                  nn.Linear(64, 1))

    def state(self, batch, device):
        m = self.trunk
        # ShopDataset stores index tensors compactly (int16); the canonical
        # batches() casts to long — mirror that here for manual slices.
        champs = batch["champs"].to(device).long()
        sides = batch["sides"].to(device).long()
        items = batch["items"].to(device).long()
        hist = batch["hist"].to(device).long()
        item_e = m.item_emb(items).masked_fill((items == 0).unsqueeze(-1), 0)
        board = m.champ_emb(champs) + m.side_emb(sides) + m.item_mix(item_e.flatten(2))
        q = m.query_proj(torch.cat([batch["query"].to(device).float(),
                                    batch["inv"].to(device).float()],
                                   dim=1)).unsqueeze(1)
        positions = torch.arange(tp.HIST_LEN, device=device).unsqueeze(0)
        hist_tok = (m.item_emb(hist) + m.side_emb(torch.full_like(hist, tp.SIDE_HISTORY))
                    + m.hist_pos(positions))
        seq = torch.cat([q, board, hist_tok], dim=1)
        never = torch.zeros(sides.size(0), 1, dtype=torch.bool, device=device)
        key_pad = torch.cat([never, sides == 0, hist == 0], dim=1)
        encoded = m.encoder(seq, src_key_padding_mask=key_pad)
        board_enc = encoded[:, 1:tp.BOARD + 1]
        opp_vec = (board_enc * (sides == tp.SIDE_LANE_OPP).unsqueeze(-1)).sum(dim=1)
        return self.state_proj(torch.cat([encoded[:, 0], opp_vec], dim=1))

    def scores(self, state, basket_vecs, chunk: int = 1024):
        """Combined score (bilinear + deep interaction), CHUNKED over the
        candidate axis so no [B, N, 3*D] intermediate is ever materialized.

        The first full run spilled ~7 GB into shared system memory exactly
        because of that intermediate (measured: 6.9 GB shared next to 6.9 GB
        dedicated on the 8 GB card, a ~10x slowdown, ~75 min/epoch)."""
        outs = []
        for s0 in range(0, basket_vecs.shape[0], chunk):
            vecs = basket_vecs[s0:s0 + chunk]                  # [C, D]
            bilinear = state @ vecs.T                          # [B, C]
            s = state.unsqueeze(1).expand(-1, vecs.shape[0], -1)
            b = vecs.unsqueeze(0).expand(state.shape[0], -1, -1)
            deep = self.deep(torch.cat([s, b, s * b], dim=-1)).squeeze(-1)
            outs.append(bilinear + deep)
        return torch.cat(outs, dim=1)


# ------------------------------------------------------------------- arm A

def train_arm_a(fit_ds, label_ids, pos_weight, device, epochs_trunk,
                epochs_graft):
    model = tp.PrefixModel(len(fit_ds.champ_index), len(fit_ds.item_index),
                           len(label_ids), len(fit_ds.item_index) + 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    count_fn, save_fn, target_fn = (nn.CrossEntropyLoss(),
                                    nn.BCEWithLogitsLoss(),
                                    nn.CrossEntropyLoss())

    def run_stage(name, epochs, save_w, target_w, freeze):
        if freeze:
            for pname, p in model.named_parameters():
                p.requires_grad = pname.startswith(("save_head.", "target_head."))
        opt = torch.optim.Adam((p for p in model.parameters()
                                if p.requires_grad), lr=0.001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for epoch in range(1, epochs + 1):
            model.train()
            total, n = 0.0, 0
            for batch in fit_ds.batches(tp.BATCH, shuffle=True):
                opt.zero_grad()
                out = model(batch["champs"].to(device), batch["sides"].to(device),
                            batch["items"].to(device), batch["query"].to(device),
                            batch["inv"].to(device), batch["hist"].to(device))
                logits = out["items"].masked_fill(~batch["legal"].to(device), -1e4)
                loss = loss_fn(logits, batch["targets"].to(device))
                counts = batch["counts"].to(device)
                bought = counts > 0
                if bool(bought.any()):
                    loss = loss + tp.COUNT_LOSS_W * count_fn(
                        out["counts"][bought], counts[bought] - 1)
                if save_w:
                    loss = loss + save_w * save_fn(
                        out["save"].squeeze(1), batch["is_save"].to(device))
                if target_w:
                    tgt = batch["target"].to(device)
                    known = tgt >= 0
                    if bool(known.any()):
                        loss = loss + target_w * target_fn(out["target"][known],
                                                           tgt[known])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch["targets"])
                n += len(batch["targets"])
            sched.step()
            print(f"  A/{name} epoch {epoch}/{epochs} loss {total / max(n, 1):.3f}",
                  flush=True)
    run_stage("trunk", epochs_trunk, 0.0, 0.0, freeze=False)
    run_stage("graft", epochs_graft, 0.5, 0.25, freeze=True)
    return model


def save_arm_a(model, fit_ds, label_ids, dragon, runtime_decl, path):
    config = {
        "d_model": tp.D_MODEL, "layers": tp.N_LAYERS, "ff_dim": tp.FF_DIM,
        "heads": tp.N_HEADS, "max_others": tp.MAX_OTHERS,
        "patch": runtime_decl["patch"], "ddragon_version": dragon.version,
        "static_data_sha256": dragon.signature,
        "footwear_state_version": runtime_decl["footwear_state_version"],
        "footwear_runtime": runtime_decl,
        "max_items": tp.MAX_ITEMS, "max_copies": tp.MAX_COPIES,
        "max_basket_items": tp.MAX_BASKET_ITEMS,
        "inventory_encoding": tp.INVENTORY_ENCODING,
        "target_encoding": tp.TARGET_ENCODING,
        "query_dim": tp.QUERY_DIM, "extras": True, "cosine": True,
        "history": False, "gold_est": True, "gold_x": True, "runes": True,
        "pov_gold": True, "gold_posterior": None, "gold_measured": None,
        "counts": True, "save_head": True, "target_head": True,
        "basket_threshold": tp.BASKET_THRESHOLD,
        "gold_input": "prequential-v3",
    }
    payload = {
        "state_dict": model.state_dict(),
        "champ_index": fit_ds.champ_index, "item_index": fit_ds.item_index,
        "label_ids": label_ids, "config": config,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "status": "probe_only",
            "purpose": "whole-basket internal A/B (fit games only)",
            "protocol": "docs/experiment_whole_basket_v1.md",
            "footwear_runtime": runtime_decl,
            "footwear_state_version": runtime_decl["footwear_state_version"],
            "gold_input": "prequential-v3",
            "scored_on": "internal chronological probe within TRAIN",
            "fit_games_sha256": FROZEN_FIT_SHA,
            "probe_games_sha256": FROZEN_PROBE_SHA,
        },
    }
    with open(path, "wb") as fh:
        torch.save(payload, fh)


# ------------------------------------------------------------------ metrics

def truth_of(row):
    entries = basket_entries(row.get("label_counts"))
    if not entries or any(i == SAVE_SENTINEL for i, _ in entries):
        return None                       # no-buy visit
    return Counter(dict(entries))


def paired_bootstrap(per_game_a, per_game_b, seed=16, n=BOOTSTRAP):
    games = sorted(per_game_a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        pick = [games[rng.randrange(len(games))] for _ in games]
        num_a = sum(per_game_a[g][0] for g in pick)
        num_b = sum(per_game_b[g][0] for g in pick)
        den = sum(per_game_a[g][1] for g in pick)
        if den:
            diffs.append((num_a - num_b) / den)
    diffs.sort()
    return (diffs[int(0.025 * (len(diffs) - 1))],
            diffs[int(0.975 * (len(diffs) - 1))])


def aggregate(results):
    """results: list of dicts with match_id + metric floats/None."""
    out = {}
    for metric in ("first_action_exact", "any_of_three_exact"):
        vals = [r[metric] for r in results if r.get(metric) is not None]
        out[metric] = {"value": sum(vals) / max(len(vals), 1), "n": len(vals)}
    for name, key in (("no_buy_death_recall", "no_buy_death"),
                      ("ward_only_first_action", "ward_only")):
        vals = [r["first_action_exact"] for r in results
                if r.get("save_kind") == key and r.get("first_action_exact") is not None]
        out[name] = {"value": sum(vals) / max(len(vals), 1), "n": len(vals)}
    return out


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny caps + 2/1 epochs: plumbing check only")
    ap.add_argument("--reuse-arm-a", action="store_true",
                    help="load the existing arm_a artifact instead of "
                         "retraining it (same code, same tensors)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    started = time.monotonic()
    torch.manual_seed(16)
    random.seed(16)

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    out_dir = Path(args.out) if args.out else RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = "smoke" if args.smoke else "full"

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
    if (membership_digest(fit_games) != FROZEN_FIT_SHA
            or membership_digest(probe_games) != FROZEN_PROBE_SHA):
        raise SystemExit("fit/probe memberships do not match the frozen "
                         "diagnostics; refusing to run")
    fit_set = set(fit_games)
    print(f"split verificado: fit {len(fit_games)} / probe {len(probe_games)}",
          flush=True)

    import footwear_runtime as fr
    runtime_decl = fr.declaration_for_manifest(manifest_gen)
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])

    train_path = generation / "visits_train.jsonl"
    cap_fit = 20000 if args.smoke else None
    cap_probe = 4000 if args.smoke else None
    champ_ids, item_ids, label_counts, catalogue, n_fit, n_probe = first_pass(
        train_path, fit_set, cap_fit, cap_probe)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_ids = sorted(label_counts)
    label_index = {item_id: i for i, item_id in enumerate(label_ids)}
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    basket_pos = {b: i + 1 for i, b in enumerate(baskets)}    # 0 = SAVE
    print(f"fit rows {n_fit}  probe rows {n_probe}  labels {len(label_ids)}  "
          f"catalogo {len(baskets)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_fit = out_dir / f"cache_fit_{stamp}_v2.pt"
    cache_probe = out_dir / f"cache_probe_{stamp}_v2.pt"
    cache_targets = out_dir / f"cache_targets_{stamp}_v2.pt"
    if cache_fit.exists() and cache_probe.exists() and cache_targets.exists():
        fit_ds = tp.ShopDataset.from_cache(cache_fit, champ_index,
                                           item_index, label_index, dragon)
        probe_ds = tp.ShopDataset.from_cache(cache_probe, champ_index,
                                             item_index, label_index, dragon)
        fit_targets = torch.load(cache_targets, weights_only=True)
        print("tensors desde cache", flush=True)
    else:
        fit_ds = tp.ShopDataset.from_stream(
            stream_split(train_path, fit_set, True, champ_index, dragon, cap_fit),
            n_fit, champ_index, item_index, label_index, dragon)
        probe_ds = tp.ShopDataset.from_stream(
            stream_split(train_path, fit_set, False, champ_index, dragon, cap_probe),
            n_probe, champ_index, item_index, label_index, dragon)
        fit_targets = torch.zeros(n_fit, dtype=torch.long)
        for i, row in enumerate(stream_split(train_path, fit_set, True,
                                             champ_index, dragon, cap_fit)):
            t = truth_of(row)
            fit_targets[i] = 0 if t is None else basket_pos[
                basket_key(sorted(t.items()))]
        fit_ds.save_cache(cache_fit)
        probe_ds.save_cache(cache_probe)
        torch.save(fit_targets, cache_targets)
    probe_rows = list(stream_split(train_path, fit_set, False, champ_index,
                                   dragon, cap_probe))
    assert len(probe_rows) == n_probe
    if device.type == "cuda":
        fit_ds.to_device(device)
        probe_ds.to_device(device)
    fit_targets = fit_targets.to(device)
    print(f"tensors listos ({time.monotonic() - started:.0f}s)", flush=True)
    cat_items = sorted({i for b in baskets for part in b.split(",")
                        for i in [int(part.split(":")[0])]})
    cat_item_index = {it: k + 1 for k, it in enumerate(cat_items)}
    n_cat = len(baskets) + 1
    cat_idx = torch.zeros((n_cat, MAX_BASKET), dtype=torch.long)
    cat_cnt = torch.zeros((n_cat, MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, part in enumerate(b.split(",")[:MAX_BASKET]):
            i, n = part.split(":")
            cat_idx[row_i, slot] = cat_item_index[int(i)]
            cat_cnt[row_i, slot] = min(int(n), MAX_BASKET)
    cat_idx, cat_cnt = cat_idx.to(device), cat_cnt.to(device)

    # posw schedule from FIT counts (canonical 8:24)
    counts_t = torch.zeros(len(label_ids))
    for item_id, c in label_counts.items():
        counts_t[label_index[item_id]] = c
    ratio = (n_fit - counts_t) / counts_t.clamp(min=1.0)
    lo, hi = 8.0, 24.0
    logc = counts_t.clamp(min=1.0).log()
    spread = (logc.max() - logc.min()).clamp(min=1e-6)
    pos_weight = torch.minimum(ratio, lo + (hi - lo) * (logc.max() - logc) / spread)

    ep_trunk, ep_graft, ep_b = (2, 1, 2) if args.smoke else (16, 6, 16)

    # ---------------- arm A ----------------
    artifact_a = out_dir / f"arm_a_{stamp}.pt"
    if args.reuse_arm_a and artifact_a.exists():
        print(f"=== arm A: reutilizando {artifact_a.name} (mismo codigo y "
              "tensores) ===", flush=True)
    else:
        print("=== arm A: incumbent formulation (canonical two-stage) ===",
              flush=True)
        model_a = train_arm_a(fit_ds, label_ids, pos_weight, device,
                              ep_trunk, ep_graft)
        if artifact_a.exists():
            artifact_a.unlink()
        save_arm_a(model_a, fit_ds, label_ids, dragon, runtime_decl, artifact_a)
        del model_a
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------- arm B ----------------
    print("=== arm B: whole-basket scorer (sampled-softmax NLL) ===", flush=True)
    model_b = BasketNet(len(champ_index), len(item_index), len(label_ids),
                        len(item_index) + 1, len(cat_items)).to(device)
    opt = torch.optim.Adam(model_b.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_b)
    nll = nn.CrossEntropyLoss()
    # Sampled softmax (the protocol's stated objective): candidates = the
    # batch's true baskets + SAVE + NEG uniform negatives. The first run's
    # full-catalogue NLL was correct but spilled 7 GB to shared memory
    # (~75 min/epoch, epoch 9/16 after 11 h); this trains the same scorer
    # inside dedicated VRAM.
    NEG = 2048
    for epoch in range(1, ep_b + 1):
        model_b.train()
        t_ep = time.monotonic()
        total, n = 0.0, 0
        order = torch.randperm(n_fit, device=device)
        for s in range(0, n_fit, tp.BATCH):
            idxs = order[s:s + tp.BATCH]
            batch = {k: v[idxs.to(v.device)] for k, v in {
                "champs": fit_ds.champs, "sides": fit_ds.sides,
                "items": fit_ds.items, "query": fit_ds.query.float(),
                "inv": fit_ds.inv, "hist": fit_ds.hist}.items()}
            opt.zero_grad()
            state = model_b.state(batch, device)
            targets = fit_targets[idxs]
            negs = torch.randint(1, len(baskets) + 1, (NEG,), device=device)
            cand = torch.cat([targets, torch.zeros(1, dtype=torch.long,
                                                   device=device), negs])
            uniq, inv_pos = torch.unique(cand, return_inverse=True)
            vecs = model_b.baskets.vectors(cat_idx, cat_cnt, uniq)
            scores = model_b.scores(state, vecs)
            loss = nll(scores, inv_pos[:len(targets)])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idxs)
            n += len(idxs)
        sched.step()
        print(f"  B epoch {epoch}/{ep_b} loss {total / max(n, 1):.3f} "
              f"({time.monotonic() - t_ep:.0f}s)", flush=True)
    torch.save({"state_dict": model_b.state_dict(),
                "baskets": baskets, "cat_items": cat_items},
               out_dir / f"arm_b_{stamp}.pt")

    # ---------------- evaluation on the probe ----------------
    from app.conditional_policy import purchase_sequence
    from app.predictor import Predictor
    from eval_policy import serving_row

    def to_action(bkey):
        return tuple(sorted(i for part in bkey.split(",")
                            for i in [int(part.split(":")[0])] * int(part.split(":")[1])))

    print("=== eval: arm B (score catalogue, legality-filter, rank) ===", flush=True)
    model_b.eval()
    results_b = []
    decomp = Counter()
    with torch.no_grad():
        basket_vecs = model_b.baskets.vectors(
            cat_idx, cat_cnt, torch.arange(n_cat, device=device))
        for s in range(0, n_probe, tp.BATCH):
            sl = slice(s, min(s + tp.BATCH, n_probe))
            batch = {k: v[sl] for k, v in {
                "champs": probe_ds.champs, "sides": probe_ds.sides,
                "items": probe_ds.items, "query": probe_ds.query,
                "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
            state = model_b.state(batch, device)
            scores = model_b.scores(state, basket_vecs)          # [b, n_cat]
            top = torch.topk(scores, min(SHORTLIST, scores.shape[1]),
                             dim=1).indices.cpu()
            for j in range(top.shape[0]):
                row = probe_rows[s + j]
                truth = truth_of(row)
                budget = float(row["gold_est"])
                legal_found = []
                for cand in top[j].tolist():
                    if cand == 0:
                        legal_found.append(("save", None))
                    else:
                        action = to_action(baskets[cand - 1])
                        if purchase_sequence(action, row, budget, dragon) is not None:
                            legal_found.append(("buy", Counter(action)))
                    if len(legal_found) >= 3:
                        break
                first = legal_found[0] if legal_found else ("save", None)
                res = {"match_id": row["match_id"],
                       "save_kind": row.get("save_kind")}
                if truth is None:
                    res["first_action_exact"] = float(first[0] == "save")
                    res["any_of_three_exact"] = None
                else:
                    res["first_action_exact"] = float(
                        first[0] == "buy" and first[1] == truth)
                    res["any_of_three_exact"] = float(any(
                        k == "buy" and c == truth for k, c in legal_found))
                    tkey = basket_key(sorted(truth.items()))
                    tpos = basket_pos.get(tkey)
                    if tpos is None:
                        decomp["catalogue_miss"] += 1
                    elif tpos not in set(top[j].tolist()):
                        decomp["shortlist_miss"] += 1
                    elif purchase_sequence(to_action(tkey), row, budget,
                                           dragon) is None:
                        decomp["legality_block"] += 1
                    elif not res["first_action_exact"]:
                        decomp["ranking_miss"] += 1
                    else:
                        decomp["hit"] += 1
                    dup = any(n > 1 for n in truth.values())
                    decomp[f"size{sum(truth.values())}_n"] += 1
                    decomp[f"size{sum(truth.values())}_hit"] += int(
                        res["first_action_exact"])
                    if dup:
                        decomp["dup_n"] += 1
                        decomp["dup_hit"] += int(res["first_action_exact"])
                res["pred_b"] = ("SAVE" if first[0] == "save"
                                 else dict(first[1]))
                results_b.append(res)
    print(f"arm B eval listo ({time.monotonic() - started:.0f}s)", flush=True)

    print("=== eval: arm A (real predictor path) ===", flush=True)
    predictor = Predictor(artifact_a)
    results_a = []
    for row in probe_rows:
        truth = truth_of(row)
        budget = float(row["gold_est"])
        options = predictor.predict_options(serving_row(row, budget),
                                            budget_slack=0, allow_save=True)
        first = options[0] if options else {"kind": "buy", "items": []}
        pred = Counter(int(i["item_id"]) for i in first.get("items") or [])
        predicted_save = first.get("kind") == "save"
        res = {"match_id": row["match_id"], "save_kind": row.get("save_kind")}
        if truth is None:
            res["first_action_exact"] = float(predicted_save)
            res["any_of_three_exact"] = None
        else:
            res["first_action_exact"] = float(not predicted_save and pred == truth)
            res["any_of_three_exact"] = float(any(
                o.get("kind") != "save"
                and Counter(int(i["item_id"]) for i in o.get("items") or []) == truth
                for o in options))
        res["pred_a"] = "SAVE" if predicted_save else dict(pred)
        results_a.append(res)
    print(f"arm A eval listo ({time.monotonic() - started:.0f}s)", flush=True)

    # paired diff + bootstrap by game
    per_game = {"A": defaultdict(lambda: [0.0, 0]),
                "B": defaultdict(lambda: [0.0, 0])}
    for tag, results in (("A", results_a), ("B", results_b)):
        for r in results:
            if r["first_action_exact"] is not None:
                per_game[tag][r["match_id"]][0] += r["first_action_exact"]
                per_game[tag][r["match_id"]][1] += 1
    ci = paired_bootstrap(per_game["B"], per_game["A"])

    agg_a, agg_b = aggregate(results_a), aggregate(results_b)
    diff = (agg_b["first_action_exact"]["value"]
            - agg_a["first_action_exact"]["value"])

    # inspectable disagreements
    examples = []
    for ra, rb, row in zip(results_a, results_b, probe_rows):
        if ra["first_action_exact"] != rb["first_action_exact"]:
            truth = truth_of(row)
            examples.append({
                "match_id": row["match_id"], "ts": row["ts"],
                "champion": row.get("champion"),
                "minute": int(row["ts"] / 60000),
                "gold_est": row["gold_est"],
                "inventory": row.get("inventory"),
                "truth": "SAVE" if truth is None else dict(truth),
                "arm_a_first": ra["pred_a"], "arm_b_first": rb["pred_b"],
                "winner": "B" if rb["first_action_exact"] else "A"})
    rng = random.Random(16)
    b_wins = [e for e in examples if e["winner"] == "B"]
    a_wins = [e for e in examples if e["winner"] == "A"]
    sample_examples = (rng.sample(b_wins, min(25, len(b_wins)))
                       + rng.sample(a_wins, min(25, len(a_wins))))

    # measured-wallet diagnostic (join expected ~0 on this window; verify)
    measured_path = ROOT / "data/imports/gold_measured_v1.json"
    measured_diag = {"file_present": measured_path.exists(), "joined": 0,
                     "joined_exact": 0, "joined_interval": 0,
                     "budget_blocked_at_est_legal_at_measured": 0}
    if measured_path.exists():
        rows_m = json.loads(measured_path.read_text(encoding="utf-8"))["rows"]
        for row in probe_rows:
            key = f"{row['match_id']}|{int(row['ts'])}|{int(row.get('champion_id') or 0)}"
            m = rows_m.get(key)
            if not m:
                continue
            measured_diag["joined"] += 1
            measured_diag["joined_exact" if m["exact"] else "joined_interval"] += 1
            truth = truth_of(row)
            if truth is None:
                continue
            action = to_action(basket_key(sorted(truth.items())))
            est_legal = purchase_sequence(action, row, float(row["gold_est"]),
                                          dragon) is not None
            meas_legal = purchase_sequence(action, row, float(m["hi"]),
                                           dragon) is not None
            if not est_legal and meas_legal:
                measured_diag["budget_blocked_at_est_legal_at_measured"] += 1

    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/experiment_whole_basket_v1.md",
        "mode": stamp, "device": str(device),
        "fit_rows": n_fit, "probe_rows": n_probe,
        "catalogue_baskets": len(baskets),
        "epochs": {"A_trunk": ep_trunk, "A_graft": ep_graft, "B": ep_b},
        "arm_b_objective": f"sampled-softmax NLL, {NEG} uniform negatives + "
                           "batch truths + SAVE; full-catalogue chunked "
                           "scoring at eval",
        "arm_a_reused": bool(args.reuse_arm_a and artifact_a.exists()),
        "arm_a": agg_a, "arm_b": agg_b,
        "paired_diff_b_minus_a": {
            "first_action_exact": {"diff": diff, "ci95": list(ci),
                                   "bootstrap": BOOTSTRAP,
                                   "games": len(per_game["A"])}},
        "arm_b_decomposition": dict(decomp),
        "measured_wallet_diagnostic": measured_diag,
        "artifacts": {"arm_a": artifact_a.name, "arm_b": f"arm_b_{stamp}.pt"},
        "hurdle": "+5.0 points first_action_exact, CI excluding zero",
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    (out_dir / f"ab_report_{stamp}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"ab_examples_{stamp}.json").write_text(
        json.dumps(sample_examples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("arm_a", "arm_b", "paired_diff_b_minus_a",
                       "arm_b_decomposition", "measured_wallet_diagnostic")},
                     indent=1), flush=True)
    print(f"\nreport: {out_dir / f'ab_report_{stamp}.json'}", flush=True)


if __name__ == "__main__":
    main()
