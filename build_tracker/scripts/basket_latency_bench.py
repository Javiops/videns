"""A-vs-M latency benchmark: two COMPLETE circuits, alternating order.

The VAL run's `total_single_model_A` was arithmetic, not a measurement: it
subtracted the ward-free forward from the mixture's total while still
including the mixture's grouping, blending and its decode. It therefore
does not answer "what does running A alone cost?".

This script runs each circuit end to end, per decision, on a FIXED FIT
sample (never VAL, never the probe used for selection, never test):

  circuit A : parent -> D(x) -> econ features -> tuned scores ->
              grouped logsumexp -> rank -> decode + legality
  circuit M : the same, PLUS the ward-free forward, the canonical-key
              permutation and the 50/50 blend before ranking

Order is alternated per decision (A,M then M,A) so neither circuit is
systematically favoured by cache or clock drift, and both orders are
reported separately as well as pooled. Timing only: predictions are
discarded and no metric is computed here.

Run via footwear_runtime with PREFIX_ML_DIR set; [--smoke] shortens it.
"""
import argparse
import hashlib
import json
import os
import statistics
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
from basket_vocabulary_probe import basket_key, membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR, econ_features, to_action
from basket_residual_econ import CorrectionNet
from basket_econ_finetune import TunedNet, OUT_DIR as FT_DIR
from basket_wardfree import strip_counter, stripped_truth, OUT_DIR as WF_DIR

OUT_DIR = ROOT / "data" / "experiments" / "16.18-mixture-val-v1"
MIX_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
V2_DIR = ROOT / "data" / "experiments" / "16.18-whole-basket-v2"
SAMPLE = 400
WARMUP = 20


def stats_ms(xs):
    xs = sorted(xs)
    return {"n": len(xs),
            "median_ms": round(statistics.median(xs), 3),
            "mean_ms": round(statistics.fmean(xs), 3),
            "p90_ms": round(xs[int(0.9 * (len(xs) - 1))], 3),
            "p99_ms": round(xs[int(0.99 * (len(xs) - 1))], 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    stamp = "smoke" if args.smoke else "full"
    sample = 40 if args.smoke else SAMPLE
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

    frozen = json.loads((MIX_DIR / "frozen_mixture_v1.json").read_text(
        encoding="utf-8"))
    wA, wB = frozen["weights"]["A"], frozen["weights"]["B"]

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
    keys_A = [None] + [basket_key(sorted(group_main[g].items()))
                       for g in range(1, n_act)]
    pos_A = {k: i for i, k in enumerate(keys_A)}
    perm_B_to_A = torch.tensor([pos_A[k] for k in [None] + baskets_B],
                               dtype=torch.long)
    assert len(set(perm_B_to_A.tolist())) == n_act
    assert int((perm_B_to_A == torch.arange(n_act)).sum()) == \
        frozen["alignment"]["positions_unchanged"]

    cache_report = json.loads(
        (CACHE_DIR / "cache_report_full.json").read_text(encoding="utf-8"))
    assert cache_report["baskets_sha256"] == baskets_sha
    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas = econ_fit[:, :, 1] > 0
    stats_mean, stats_std = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    for d in range(2, N_ECON):
        stats_mean[d] = flat[:, d].mean()
        stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    sm, ss = stats_mean.to(device), stats_std.to(device)
    del econ_fit, flat, feas

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

    parent = v1.BasketNet(*arch(n_items_A)).to(device)
    parent.load_state_dict(torch.load(V2_DIR / "Bf_ep16.pt", map_location="cpu",
                                      weights_only=False)["state_dict"])
    parent.eval()
    tuned = TunedNet(v1.BasketNet(*arch(n_items_A)).to(device),
                     CorrectionNet().to(device))
    tuned.load_state_dict(torch.load(FT_DIR / "econ_ft_ep8.pt",
                                     map_location="cpu",
                                     weights_only=False)["state_dict"])
    tuned.eval()
    wardfree = v1.BasketNet(*arch(n_items_B)).to(device)
    wardfree.load_state_dict(torch.load(WF_DIR / "wardfree_ep16.pt",
                                        map_location="cpu",
                                        weights_only=False)["state_dict"])
    wardfree.eval()

    # FIXED FIT sample: never VAL, never the selection probe, never test
    g = torch.Generator().manual_seed(16)
    fit_rows = list(v1.stream_split(train_path, fit_set, True, champ_index,
                                    dragon, None))
    pick = torch.randperm(len(fit_rows), generator=g)[:sample + WARMUP].tolist()
    rows = [fit_rows[i] for i in pick]
    ds = tp.ShopDataset.from_stream(iter(rows), len(rows), champ_index,
                                    item_index, label_index, dragon)
    if device.type == "cuda":
        ds.to_device(device)
    gid = group_ids.to(device)
    perm = perm_B_to_A.to(device)
    all_cat = torch.arange(n_cat, device=device)
    all_cat_B = torch.arange(n_act, device=device)
    print(f"setup listo ({time.monotonic() - started:.0f}s); muestra FIT "
          f"{sample} (+{WARMUP} calentamiento)", flush=True)

    sync = (torch.cuda.synchronize if device.type == "cuda" else (lambda: None))

    def clock():
        sync()
        return time.perf_counter()

    raw = []   # per-decision rows, kept so the order effect can be chased

    def one(j, mixture, pvecs, wvecs):
        """One COMPLETE circuit for decision j. Returns elapsed ms."""
        row = rows[j]
        budget = float(row["gold_est"])
        batch = {k: v[j:j + 1] for k, v in {
            "champs": ds.champs, "sides": ds.sides, "items": ds.items,
            "query": ds.query, "inv": ds.inv, "hist": ds.hist}.items()}
        t0 = clock()
        ps = parent.scores(parent.state(batch, device), pvecs)
        dom = torch.topk(ps, v1.SHORTLIST, dim=1).indices
        dom_cpu = dom.cpu()
        e = torch.zeros((1, v1.SHORTLIST, N_ECON))
        for k in range(v1.SHORTLIST):
            cand = int(dom_cpu[0, k])
            if cand == 0:
                e[0, k, 0] = 1.0
            else:
                e[0, k] = torch.tensor(econ_features(
                    to_action(baskets_A[cand - 1]), row, budget, dragon,
                    Counter()))
        sA, _ = tuned.corrected_scores(batch, device, cat_idx, cat_cnt, n_cat,
                                       dom, (e.to(device) - sm) / ss)
        gi = gid.unsqueeze(0)
        gm = torch.full((1, n_act), float("-inf"), device=device)
        gm.scatter_reduce_(1, gi, sA, reduce="amax", include_self=True)
        gs = torch.zeros(1, n_act, device=device)
        gs.scatter_add_(1, gi, (sA - gm.gather(1, gi)).exp())
        P = torch.softmax(gm + gs.log(), dim=1)
        if mixture:
            sB = wardfree.scores(wardfree.state(batch, device), wvecs)
            PB = torch.zeros_like(P)
            PB.scatter_(1, perm.unsqueeze(0), torch.softmax(sB, dim=1))
            P = wA * P + wB * PB
        order = torch.topk(P, min(v1.SHORTLIST, n_act), dim=1).indices.cpu()
        options = []
        for act in order[0].tolist():
            if act == 0:
                options.append("save")
            else:
                main = group_main[act]
                if purchase_sequence(tuple(sorted(main.elements())), row,
                                     budget, dragon) is not None:
                    options.append("buy")
            if len(options) >= 3:
                break
        t1 = clock()
        raw.append({"row": j, "circuit": "M" if mixture else "A",
                    "ms": (t1 - t0) * 1000.0,
                    "t_start_perf": t0, "t_end_perf": t1,
                    "utc": datetime.now(timezone.utc).isoformat()})
        return (t1 - t0) * 1000.0

    timings = {"A": {"A_first": [], "M_first": []},
               "M": {"A_first": [], "M_first": []}}
    with torch.no_grad():
        pvecs = parent.baskets.vectors(cat_idx, cat_cnt, all_cat)
        wvecs = wardfree.baskets.vectors(cat_idx_B, cat_cnt_B, all_cat_B)
        for j in range(WARMUP):
            one(j, False, pvecs, wvecs)
            one(j, True, pvecs, wvecs)
        for i in range(sample):
            j = WARMUP + i
            if i % 2 == 0:                       # alternate which runs first
                ta = one(j, False, pvecs, wvecs)
                tm = one(j, True, pvecs, wvecs)
                timings["A"]["A_first"].append(ta)
                timings["M"]["A_first"].append(tm)
            else:
                tm = one(j, True, pvecs, wvecs)
                ta = one(j, False, pvecs, wvecs)
                timings["M"]["M_first"].append(tm)
                timings["A"]["M_first"].append(ta)

    pooled = {k: v["A_first"] + v["M_first"] for k, v in timings.items()}
    paired_delta = []
    for order_tag in ("A_first", "M_first"):
        for ta, tm in zip(timings["A"][order_tag], timings["M"][order_tag]):
            paired_delta.append(tm - ta)
    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "what": "complete A and M circuits, per decision, alternating order",
        "why": ("the VAL run's total_single_model_A was arithmetic (mixture "
                "total minus the ward-free forward) and still contained the "
                "mixture's grouping, blend and decode; it was never a "
                "measurement of circuit A"),
        "split": "FIXED FIT sample (not VAL, not the selection probe, not test)",
        "mode": stamp, "batch_size": 1, "n_decisions": sample,
        "warmup_decisions": WARMUP,
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda"
                   else "cpu"),
        "sample_construction":
            "torch.randperm(n_fit, generator=manual_seed(16))[:sample+warmup]",
        "circuit_A": stats_ms(pooled["A"]),
        "circuit_M": stats_ms(pooled["M"]),
        "by_order": {
            "A_measured_when_A_ran_first": stats_ms(timings["A"]["A_first"]),
            "A_measured_when_M_ran_first": stats_ms(timings["A"]["M_first"]),
            "M_measured_when_M_ran_first": stats_ms(timings["M"]["M_first"]),
            "M_measured_when_A_ran_first": stats_ms(timings["M"]["A_first"]),
        },
        "paired_delta_M_minus_A_ms": stats_ms(paired_delta),
        "relative_overhead_median":
            round(statistics.median(pooled["M"])
                  / max(statistics.median(pooled["A"]), 1e-9) - 1.0, 4),
    }
    # aggregates hide the tails' structure: persist every measurement with
    # its position in the pair and a timestamp, so the order effect observed
    # on 2026-09-22 (the circuit measured SECOND carries a ~900 ms p99) can
    # be investigated without re-running anything.
    order_of = {}
    for i, r in enumerate(raw):
        r["is_warmup"] = i < WARMUP * 2
        r["position_in_pair"] = 1 if (i % 2 == 0) else 2
    rawpath = OUT_DIR / f"latency_bench_raw_{stamp}.jsonl"
    with open(rawpath, "w", encoding="utf-8", newline=chr(10)) as fh:
        for r in raw:
            fh.write(json.dumps(r) + chr(10))
        fh.flush()
        os.fsync(fh.fileno())
    # VERIFY BEFORE CLOSING. The previous run of this benchmark reported
    # aggregates while its per-decision records were never written, and the
    # tail could not be chased afterwards. So the file is read back here,
    # counted, and the aggregates are RECOMPUTED from it and compared with
    # the ones about to be published. A mismatch aborts rather than shipping
    # a report whose backing records do not support it.
    reread = [json.loads(l) for l in
              rawpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    expect_rows = len(raw)
    if len(reread) != expect_rows:
        raise RuntimeError(
            f"registros incompletos: {len(reread)} filas en disco, "
            f"{expect_rows} medidas")
    pooled_back = {"A": [], "M": []}
    for r in reread:
        if not r["is_warmup"]:
            pooled_back[r["circuit"]].append(r["ms"])
    recomputed = {k: stats_ms(v) for k, v in pooled_back.items()}
    for circuit, key in (("A", "circuit_A"), ("M", "circuit_M")):
        for field in ("n", "median_ms", "p90_ms", "p99_ms", "mean_ms"):
            if recomputed[circuit][field] != report[key][field]:
                raise RuntimeError(
                    f"los registros no reproducen {key}.{field}: "
                    f"{recomputed[circuit][field]} vs {report[key][field]}")
    report["raw_measurements"] = {
        "file": str(rawpath.relative_to(ROOT)),
        "sha256": hashlib.sha256(rawpath.read_bytes()).hexdigest(),
        "rows": len(reread),
        "warmup_rows": sum(1 for r in reread if r["is_warmup"]),
        "measured_rows": sum(1 for r in reread if not r["is_warmup"]),
        "aggregates_reproduced_from_records": True,
        "fields_checked": ["n", "median_ms", "p90_ms", "p99_ms", "mean_ms"],
    }
    print(f"registros verificados: {len(reread)} filas, agregados "
          f"reproducidos desde disco", flush=True)
    report["tail_note"] = (
        "Observed: high tails when a circuit runs SECOND in its pair "
        "(A-second p99 928 ms vs A-first 122; M-second 895 vs M-first 136). "
        "Alternating the order does NOT prove the effect is symmetric or "
        "that the paired median delta is unaffected. Cause and effect on the "
        "comparison are undetermined; per-decision rows are persisted "
        "alongside so this can be chased later.")
    out = OUT_DIR / f"latency_bench_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("circuit_A", "circuit_M", "paired_delta_M_minus_A_ms",
                       "relative_overhead_median", "by_order")}, indent=1),
          flush=True)
    print(f"report: {out}", flush=True)


if __name__ == "__main__":
    main()
