"""Matched residual-reranker contrast: economic features vs zeroed control.

Protocol: docs/experiment_basket_residual_econ_v1.md. Binding condition from
the review loop: control AND candidate both start from the same frozen
Bf_ep16 and receive IDENTICAL additional tuning; ONLY the economic features
differ. The backbone is never updated; the correction is defined only on the
parent's per-decision top-128 shortlist and is identically zero elsewhere,
so full-catalogue CE is exact via the cached frozen tail logsumexp:

    Z = logsumexp([s_i + c_i for shortlist] + [tail_lse])

Targets outside the shortlist keep their frozen score — never dropped,
never inserted into candidates. Both arms: same net (final layer zero-init,
so step 0 reproduces Bf_ep16 exactly), same seed, batches, epochs, LR.

Requires scripts/basket_econ_cache.py output. Run via footwear_runtime with
PREFIX_ML_DIR set; [--smoke] uses the smoke cache and 2 epochs.
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
from basket_econ_cache import N_ECON, OUT_DIR, to_action

V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
LR = 1e-3
EPOCHS = 8
BATCH = 4096
N_EXAMPLES = 30


class CorrectionNet(nn.Module):
    """Residual score correction on [state | basket_vec | econ]."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(v1.STATE_DIM * 2 + N_ECON, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, basket_vecs, econ):
        # state [B, D]; basket_vecs [B, K, D]; econ [B, K, E] -> [B, K]
        s = state.unsqueeze(1).expand(-1, basket_vecs.shape[1], -1)
        return self.net(torch.cat([s, basket_vecs, econ], dim=-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=16,
                    help="correction init + batch order only; the cache and "
                         "frozen parent are seed-independent")
    args = ap.parse_args()
    started = time.monotonic()
    cache_stamp = "smoke" if args.smoke else "full"
    stamp = cache_stamp
    if args.seed != 16:
        stamp = f"{stamp}_seed{args.seed}"
    epochs = 2 if args.smoke else EPOCHS

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
    champ_ids, item_ids, label_counts, catalogue, _, _ = v1.first_pass(
        train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets).encode()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_report = json.loads(
        (OUT_DIR / f"cache_report_{cache_stamp}.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    basket_vecs = torch.load(OUT_DIR / f"basket_vecs_{cache_stamp}.pt",
                             weights_only=True).float().to(device)
    fit = torch.load(OUT_DIR / f"frozen_fit_{cache_stamp}.pt", weights_only=True)
    probe = torch.load(OUT_DIR / f"frozen_probe_{cache_stamp}.pt", weights_only=True)
    econ_fit = torch.load(OUT_DIR / f"econ_fit_{cache_stamp}.pt", weights_only=True)
    econ_probe = torch.load(OUT_DIR / f"econ_probe_{cache_stamp}.pt",
                            weights_only=True)
    n_fit, n_probe = fit["n_rows"], probe["n_rows"]

    # Standardization from FIT feasible-buy candidates (dims 2..9); the
    # is_save/feasible flags stay raw. Control arm: literal zero block.
    feas = econ_fit[:, :, 1] > 0
    stats_mean = torch.zeros(N_ECON)
    stats_std = torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    if flat.numel():
        for d in range(2, N_ECON):
            stats_mean[d] = flat[:, d].mean()
            stats_std[d] = flat[:, d].std().clamp(min=1e-4)

    def econ_block(raw, arm):
        if arm == "control":
            return torch.zeros(*raw.shape[:-1], N_ECON, device=device)
        return ((raw.float().to(device) - stats_mean.to(device))
                / stats_std.to(device))

    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, n_probe if args.smoke else None))
    digest = hashlib.sha256()
    for row in probe_rows:
        digest.update(f"{row['match_id']}|{row['ts']};".encode())
    assert digest.hexdigest() == probe["rows_sha256"], "probe desalineado"
    assert len(probe_rows) == n_probe

    fit_states = fit["states"].float()
    fit_top_idx = fit["top_idx"].long()
    fit_top_score = fit["top_score"]
    fit_tail = fit["tail_lse"]
    fit_tpos = fit["target_short_pos"].long()
    fit_tscore = fit["target_score"]
    print(f"caches listos: fit {n_fit} probe {n_probe} "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": stamp, "parent": "Bf_ep16", "lr": LR, "epochs": epochs,
              "batch": BATCH, "curves": {}, "aggregates": {}, "paired": {},
              "net_vs_parent": {}, "latency": {"cache_bench": cache_report["bench"]},
              "examples": []}
    rows_by_arm = {}

    for arm in ("control", "econ"):
        torch.manual_seed(args.seed)
        net = CorrectionNet().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        curve = []
        order_gen = torch.Generator().manual_seed(args.seed)  # same per arm
        for epoch in range(1, epochs + 1):
            net.train()
            t_ep = time.monotonic()
            order = torch.randperm(n_fit, generator=order_gen)
            total, n = 0.0, 0
            for s in range(0, n_fit, BATCH):
                idxs = order[s:s + BATCH]
                opt.zero_grad()
                state = fit_states[idxs].to(device)
                short = fit_top_idx[idxs].to(device)
                corr = net(state, basket_vecs[short],
                           econ_block(econ_fit[idxs], arm))
                logits = fit_top_score[idxs].to(device) + corr
                z = torch.logsumexp(torch.cat(
                    [logits, fit_tail[idxs].to(device).unsqueeze(1)], dim=1), 1)
                tpos = fit_tpos[idxs].to(device)
                tscore = fit_tscore[idxs].to(device)
                inside = tpos >= 0
                tlogit = torch.where(
                    inside, logits.gather(1, tpos.clamp(min=0).unsqueeze(1))
                    .squeeze(1), tscore)
                loss = (z - tlogit).mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idxs)
                n += len(idxs)
            sched.step()
            curve.append({"epoch": epoch, "train_loss": total / max(n, 1),
                          "seconds": round(time.monotonic() - t_ep, 1)})
            print(f"  {arm} epoch {epoch}/{epochs} "
                  f"loss {total / max(n, 1):.4f} ({curve[-1]['seconds']}s)",
                  flush=True)
        report["curves"][arm] = curve
        torch.save({"state_dict": net.state_dict(),
                    "stats_mean": stats_mean, "stats_std": stats_std,
                    "arm": arm, "parent": "Bf_ep16",
                    "baskets_sha256": baskets_sha},
                   OUT_DIR / f"correction_{arm}_{stamp}.pt")

        # -------- eval: merged decode (corrected top-128 + frozen tail-64)
        net.eval()
        records = []
        lazy_calls = 0
        t_eval = time.monotonic()
        with torch.no_grad():
            for s in range(0, n_probe, 1024):
                sl = slice(s, min(s + 1024, n_probe))
                state = probe["states"][sl].float().to(device)
                short = probe["top_idx"][sl].long().to(device)
                corr = net(state, basket_vecs[short],
                           econ_block(econ_probe[sl], arm))
                corrected = (probe["top_score"][sl].to(device) + corr).cpu()
                for j in range(corrected.shape[0]):
                    i = s + j
                    row = probe_rows[i]
                    budget = float(row["gold_est"])
                    feas_i = econ_probe[i, :, 1] > 0
                    save_i = econ_probe[i, :, 0] > 0
                    merged = ([(float(corrected[j, k]),
                                int(probe["top_idx"][i, k]),
                                bool(feas_i[k] or save_i[k]), True)
                               for k in range(corrected.shape[1])]
                              + [(float(probe["tail_top_score"][i, k]),
                                  int(probe["tail_top_idx"][i, k]), False,
                                  False)
                                 for k in range(probe["tail_top_idx"].shape[1])])
                    merged.sort(key=lambda t: -t[0])
                    options = []
                    for score, cand, known_legal, in_short in merged:
                        if cand == 0:
                            options.append(("save", None))
                        else:
                            action = to_action(baskets[cand - 1])
                            if in_short:
                                legal = known_legal
                            else:
                                lazy_calls += 1
                                legal = purchase_sequence(
                                    action, row, budget, dragon) is not None
                            if legal:
                                options.append(("buy", Counter(action)))
                        if len(options) >= 3:
                            break
                    fallback = not options
                    if fallback:
                        options = [("save", None)]
                    rec = p2.base_record(row, options, family_of, dragon)
                    rec["ts"] = row["ts"]
                    rec["fallback_save"] = fallback
                    records.append(rec)
        eval_s = time.monotonic() - t_eval
        report["latency"][arm] = {
            "eval_seconds": round(eval_s, 1),
            "ms_per_decision_decode": round(eval_s / n_probe * 1e3, 3),
            "lazy_tail_legality_calls": lazy_calls}
        p2.dump(OUT_DIR / f"rows_{arm}_{stamp}.jsonl", records)
        rows_by_arm[arm] = records
        report["aggregates"][arm] = p2.aggregate(records)
        print(f"eval {arm}: exact "
              f"{report['aggregates'][arm]['first_action_exact']['value']:.4f} "
              f"({eval_s:.0f}s)", flush=True)

    # ------------------------ comparisons ------------------------
    def key_of(r):
        # Ten perspectives per game: (match_id, ts) collides across players
        # (124 duplicate keys on this probe, caught by the external audit).
        # Champion is unique within a ranked game, so this triple identifies
        # the decision.
        return (r["match_id"], r["ts"], r["champion"])

    parent_rows = None
    if not args.smoke:
        parent_rows = [json.loads(l) for l in open(
            V2_DIR / "rows_Bf_full.jsonl", encoding="utf-8")]
    report["paired"]["econ_vs_control"] = p2.paired(
        rows_by_arm["econ"], rows_by_arm["control"], "first_action_exact")
    # Cost/benefit slices the summary must carry (audit round): SAVE and
    # ward-only losses ship next to the multi-item gain, not instead of it.
    subsets = {
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
    }
    for label, pred in subsets.items():
        report["paired"][f"econ_vs_control_{label}"] = p2.paired(
            [r for r in rows_by_arm["econ"] if pred(r)],
            [r for r in rows_by_arm["control"] if pred(r)],
            "first_action_exact")
    if parent_rows:
        for arm in ("econ", "control"):
            report["paired"][f"{arm}_vs_parent"] = p2.paired(
                rows_by_arm[arm], parent_rows, "first_action_exact")
        parent_by_key = {key_of(r): r for r in parent_rows}
        for arm in ("econ", "control"):
            corrected_n = broken_n = 0
            by_size = {}
            for r in rows_by_arm[arm]:
                p = parent_by_key.get(key_of(r))
                if p is None or r.get("first_action_exact") is None:
                    continue
                a, b = p["first_action_exact"], r["first_action_exact"]
                if a < b:
                    corrected_n += 1
                elif a > b:
                    broken_n += 1
                if a != b:
                    bucket = by_size.setdefault(
                        min(r["true_size"], 5), {"corrected": 0, "broken": 0})
                    bucket["corrected" if a < b else "broken"] += 1
            report["net_vs_parent"][arm] = {
                "corrected": corrected_n, "broken": broken_n,
                "net": corrected_n - broken_n,
                "by_true_size": {str(k): by_size[k] for k in sorted(by_size)}}
        # single vs multi and duplicates, per arm + parent
        for name, recs in (("parent", parent_rows),
                           ("control", rows_by_arm["control"]),
                           ("econ", rows_by_arm["econ"])):
            purch = [r for r in recs if r["true_size"] > 0]
            single = [r["first_action_exact"] for r in purch
                      if r["true_size"] == 1]
            multi = [r["first_action_exact"] for r in purch
                     if r["true_size"] > 1]
            report["aggregates"].setdefault("splits", {})[name] = {
                "single_exact": sum(single) / max(len(single), 1),
                "multi_exact": sum(multi) / max(len(multi), 1),
                "n_single": len(single), "n_multi": len(multi)}
        # inspectable examples where econ changes the first action.
        # Parent rows come from JSON (string keys); normalize before comparing.
        def norm_first(pf):
            return pf if isinstance(pf, str) else {str(k): v
                                                   for k, v in pf.items()}

        econ_by_key = {key_of(r): r for r in rows_by_arm["econ"]}
        for i, row in enumerate(probe_rows):
            if len(report["examples"]) >= N_EXAMPLES:
                break
            r = econ_by_key.get((row["match_id"], row["ts"]))
            p = parent_by_key.get((row["match_id"], row["ts"]))
            if (not r or not p
                    or norm_first(r["pred_first"]) == norm_first(p["pred_first"])):
                continue
            feats = econ_probe[i].float()
            cands = []
            for k in range(min(8, feats.shape[0])):
                cand = int(probe["top_idx"][i, k])
                cands.append({
                    "basket": "SAVE" if cand == 0 else baskets[cand - 1],
                    "feasible": bool(feats[k, 1] > 0) or cand == 0,
                    "cost": round(float(feats[k, 2]) * 1000),
                    "gold_left": round(float(feats[k, 3]) * 1000),
                    "preowned_consumed": int(feats[k, 6]),
                    "slot_delta": int(feats[k, 8])})
            report["examples"].append({
                "match_id": row["match_id"], "ts": row["ts"],
                "champion": row.get("champion"), "gold_est": row["gold_est"],
                "parent_first": p["pred_first"], "econ_first": r["pred_first"],
                "truth_hit_parent": p["first_action_exact"],
                "truth_hit_econ": r["first_action_exact"],
                "top_candidates": cands})

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / f"residual_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["paired"], indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
