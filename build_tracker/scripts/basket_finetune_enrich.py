"""Enriched per-decision rows (all 3 options + truth) for the fine-tune arms.

Read-only companion to basket_econ_finetune.py: loads the pre-declared +8
checkpoints and re-runs ONLY the evaluator decode, dumping option-level rows
in the prework format so ward-insensitive and other decode-side diagnostics
can be recomputed offline. No training, no new metrics claimed here.

Run via footwear_runtime with PREFIX_ML_DIR set.
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

import train_basket_ab as v1
import train_prefix as tp
from basket_vocabulary_probe import membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR, to_action
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR


def main():
    started = time.monotonic()
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
    baskets = [k for k, _ in sorted(catalogue.items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    baskets_sha = hashlib.sha256("|".join(baskets).encode()).hexdigest()
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_report = json.loads(
        (CACHE_DIR / "cache_report_full.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    probe = torch.load(CACHE_DIR / "frozen_probe_full.pt", weights_only=True)
    econ_probe = torch.load(CACHE_DIR / "econ_probe_full.pt",
                            weights_only=True)
    n_probe = probe["n_rows"]
    probe_domain = probe["top_idx"].long()

    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas_fit = econ_fit[:, :, 1] > 0
    stats_mean = torch.zeros(N_ECON)
    stats_std = torch.ones(N_ECON)
    flat = econ_fit[feas_fit].float()
    for d in range(2, N_ECON):
        stats_mean[d] = flat[:, d].mean()
        stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = stats_mean.to(device), stats_std.to(device)
    del econ_fit, flat, feas_fit

    def econ_block(raw, arm):
        if arm == "control":
            return torch.zeros(*raw.shape[:-1], N_ECON, device=device)
        return (raw.float().to(device) - sm) / ss

    v1_dir = ROOT / "data" / "experiments" / "16.18-whole-basket-v1"
    probe_ds = tp.ShopDataset.from_cache(v1_dir / "cache_probe_full_v2.pt",
                                         champ_index, item_index, label_index,
                                         dragon)
    if device.type == "cuda":
        probe_ds.to_device(device)
    probe_rows = list(v1.stream_split(train_path, fit_set, False, champ_index,
                                      dragon, None))
    digest = hashlib.sha256()
    for row in probe_rows:
        digest.update(f"{row['match_id']}|{row['ts']};".encode())
    assert digest.hexdigest() == probe["rows_sha256"], "probe desalineado"

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
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    for arm in ("control", "econ"):
        blob = torch.load(OUT_DIR / f"{arm}_ft_ep8.pt", map_location="cpu",
                          weights_only=False)
        base = v1.BasketNet(*arch_args).to(device)
        correction = CorrectionNet().to(device)
        model = TunedNet(base, correction)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        out_path = OUT_DIR / f"rows_{arm}_ft_full_enriched.jsonl"
        n_done = 0
        with torch.no_grad(), open(out_path, "w", encoding="utf-8",
                                   newline="\n") as fh:
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
                top = torch.topk(scores, min(v1.SHORTLIST, n_cat),
                                 dim=1).indices.cpu()
                for j in range(top.shape[0]):
                    i = s + j
                    row = probe_rows[i]
                    budget = float(row["gold_est"])
                    feas_map = {int(probe_domain[i, k]):
                                bool(econ_probe[i, k, 1] > 0)
                                for k in range(probe_domain.shape[1])}
                    options = []
                    for cand in top[j].tolist():
                        if cand == 0:
                            options.append({"kind": "save"})
                        else:
                            action = to_action(baskets[cand - 1])
                            legal = (feas_map[cand] if cand in feas_map
                                     else purchase_sequence(
                                         action, row, budget, dragon)
                                     is not None)
                            if legal:
                                options.append({"kind": "buy", "item_counts": {
                                    str(k): int(v_) for k, v_ in
                                    sorted(Counter(action).items())}})
                        if len(options) >= 3:
                            break
                    if not options:
                        options = [{"kind": "save"}]
                    truth = v1.truth_of(row)
                    fh.write(json.dumps({
                        "match_id": row["match_id"], "ts": row["ts"],
                        "champion": row.get("champion"),
                        "save_kind": row.get("save_kind"),
                        "gold_est": row["gold_est"],
                        "truth_item_counts": ({str(k): int(v_) for k, v_ in
                                               sorted(truth.items())}
                                              if truth else {}),
                        "options": options},
                        ensure_ascii=False, separators=(",", ":")) + "\n")
                    n_done += 1
        print(f"{arm}: {n_done} filas -> {out_path.name} "
              f"({time.monotonic() - started:.0f}s)", flush=True)
        del model, base, correction
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
