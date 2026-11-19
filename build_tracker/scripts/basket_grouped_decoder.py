"""Grouped (ward-marginalized) decoder for frozen econ_ft_ep8.

Reviewer's assignment: before investing in the combined training, evaluate
the FROZEN economic fine-tune with a decoder that (1) scores its full
catalogue, (2) groups baskets that are identical after removing wards,
(3) marginalizes P(main purchase | x) = logsumexp of each group's logits,
(4) checks legality of the MAIN (stripped) basket at the original budget,
(5) picks three DISTINCT main purchases. Scored under the same ward-free
metric, paired against the trained ward-free model on the same rows.
No retraining, no new econ cache.

Group semantics: catalogue id 0 (SAVE) and every basket that strips to
empty share group 0 = "no main purchase" (the reviewer's naming).

Run via footwear_runtime with PREFIX_ML_DIR set.
"""
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
from basket_vocabulary_probe import basket_key, membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR as FT_DIR
from basket_wardfree import WARDS, strip_counter, wardfree_record

OUT_DIR = ROOT / "data" / "experiments" / "16.18-wardfree-v1"


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

    # ---- group map: catalogue id -> ward-stripped main-purchase group ----
    group_of_key = {None: 0}
    group_main = [None]                       # group 0 = no main purchase
    group_ids = torch.zeros(n_cat, dtype=torch.long)
    for cid, bkey in enumerate(baskets, start=1):
        stripped = strip_counter(Counter(
            {int(part.split(":")[0]): int(part.split(":")[1])
             for part in bkey.split(",")}))
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = len(group_main)
            group_main.append(stripped)
        group_ids[cid] = group_of_key[skey]
    n_groups = len(group_main)
    print(f"grupos: {n_groups} (catalogo {n_cat}); "
          f"colapsan a vacio: {int((group_ids[1:] == 0).sum())}", flush=True)

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

    blob = torch.load(FT_DIR / "econ_ft_ep8.pt", map_location="cpu",
                      weights_only=False)
    base = v1.BasketNet(*arch_args).to(device)
    correction = CorrectionNet().to(device)
    model = TunedNet(base, correction)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    gid = group_ids.to(device)
    print(f"setup listo ({time.monotonic() - started:.0f}s)", flush=True)

    records = []
    lazy_calls = 0
    with torch.no_grad():
        for s in range(0, n_probe, 512):
            sl = slice(s, min(s + 512, n_probe))
            batch = {k: v[sl] for k, v in {
                "champs": probe_ds.champs, "sides": probe_ds.sides,
                "items": probe_ds.items, "query": probe_ds.query,
                "inv": probe_ds.inv, "hist": probe_ds.hist}.items()}
            dom = probe_domain[sl].to(device)
            econ = ((econ_probe[sl].float().to(device) - sm) / ss)
            scores, _ = model.corrected_scores(
                batch, device, cat_idx, cat_cnt, n_cat, dom, econ)
            b = scores.shape[0]
            gidx = gid.unsqueeze(0).expand(b, -1)
            gmax = torch.full((b, n_groups), float("-inf"), device=device)
            gmax.scatter_reduce_(1, gidx, scores, reduce="amax",
                                 include_self=True)
            gsum = torch.zeros(b, n_groups, device=device)
            gsum.scatter_add_(1, gidx,
                              (scores - gmax.gather(1, gidx)).exp())
            grouped = (gmax + gsum.log()).cpu()          # logsumexp per group
            top = torch.topk(grouped, min(v1.SHORTLIST, n_groups),
                             dim=1).indices
            for j in range(b):
                row = probe_rows[s + j]
                budget = float(row["gold_est"])
                options = []
                for grp in top[j].tolist():
                    if grp == 0:
                        options.append(("save", None))
                    else:
                        main = group_main[grp]
                        action = tuple(sorted(main.elements()))
                        lazy_calls += 1
                        if purchase_sequence(action, row, budget,
                                             dragon) is not None:
                            options.append(("buy", Counter(main)))
                    if len(options) >= 3:
                        break
                if not options:
                    options = [("save", None)]
                records.append(wardfree_record(row, options))
    p2.dump(OUT_DIR / "rows_grouped_econft_full.jsonl", records)

    wardfree_rows = [json.loads(l) for l in open(
        OUT_DIR / "rows_wardfree_full.jsonl", encoding="utf-8")]
    report = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "decoder": "ward-marginalized logsumexp over econ_ft_ep8",
              "n_groups": n_groups, "lazy_legality_calls": lazy_calls,
              "aggregates": {"grouped_econft": p2.aggregate(records),
                             "wardfree": p2.aggregate(wardfree_rows)},
              "paired": {}}
    report["paired"]["grouped_vs_wardfree"] = p2.paired(
        records, wardfree_rows, "first_action_exact")
    report["paired"]["grouped_vs_wardfree_any3"] = p2.paired(
        records, wardfree_rows, "any_of_three_exact")
    subsets = {
        "single": lambda r: r["true_size"] == 1,
        "multi": lambda r: r["true_size"] > 1,
        "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
        "ward_only": lambda r: r.get("save_kind") == "ward_only",
    }
    for label, pred in subsets.items():
        report["paired"][f"grouped_vs_wardfree_{label}"] = p2.paired(
            [r for r in records if pred(r)],
            [r for r in wardfree_rows if pred(r)],
            "first_action_exact")
    # Gain from grouping alone: vs the ungrouped econ_ft under the same
    # metric (its ward-free rescore, recomputed here from enriched rows).
    from basket_wardfree import stripped_truth
    ungrouped = []
    with open(ROOT / "data" / "experiments" / "16.18-basket-econ-finetune-v1"
              / "rows_econ_ft_full_enriched.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            row = probe_rows[i]
            assert (row["match_id"], row["ts"]) == (r["match_id"], r["ts"])
            options = [("save", None) if o["kind"] == "save"
                       else ("buy", Counter({int(k): int(v_) for k, v_ in
                                             o["item_counts"].items()}))
                       for o in r["options"]]
            ungrouped.append(wardfree_record(row, options or [("save", None)]))
    report["aggregates"]["econft_ungrouped"] = p2.aggregate(ungrouped)
    report["paired"]["grouped_vs_ungrouped"] = p2.paired(
        records, ungrouped, "first_action_exact")
    report["paired"]["grouped_vs_ungrouped_any3"] = p2.paired(
        records, ungrouped, "any_of_three_exact")

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out = OUT_DIR / "grouped_decoder_report_full.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["paired"], indent=1), flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
