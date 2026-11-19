"""Shared setup and scoring for the stage-2 experiments on the v4 generation.

Protocol: docs/experiment_stage2_econ_wardfree_v1.md (owner GO 2026-09-25).
Everything derives from the v4 generation and the stage-1 artefacts; the new
Bf (frozen reference R1, `frozen_reference_v4.json`) is re-instantiated from
its checkpoint and must reproduce its stage-1 rows before anything counts.
Recipes are the phase-2 / twin fine-tune ones, unchanged; only the inputs
are generation-agnostic. Every script in this family runs through
footwear_runtime with PREFIX_ML_DIR = the v4 generation and --review = the
v4 snapshot's static review; [--smoke] caps rows and epochs.
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

import train_basket_ab as v1
import train_basket_phase2 as p2
import train_prefix as tp
from basket_vocabulary_probe import basket_key
from basket_wardfree import wardfree_record
from basket_bf_newgen import group_map, basket_tensors, decode, sha256_file
from basket_newbf_mixture import load_rows, mismatches

OUT_DIR = ROOT / "data" / "experiments" / "16.18-stage2-v4"
STAGE1 = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v4-bf-stage1"
SMOKE_FIT, SMOKE_COHORT = 3000, 1500
WARMUP, TIMED = 2, 12                       # untimed warm-up, then timed effective batches
BOUNDS_H = {"econ_cache": 6.0, "twin_finetune": 12.0, "wardfree": 8.0}   # declared by the owner
EXIT_BOUND = 3                              # "pilot exceeded its declared bound": stop this segment only
EXIT_GATE = 2                               # an integrity gate failed
SUBSETS = {"single": lambda r: r["true_size"] == 1, "multi": lambda r: r["true_size"] > 1,
           "no_buy_death": lambda r: r.get("save_kind") == "no_buy_death",
           "ward_only": lambda r: r.get("save_kind") == "ward_only"}


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def rows_digest(rows_iter):
    d = hashlib.sha256()
    n = 0
    for row in rows_iter:
        d.update(f"{row['match_id']}|{row['ts']};".encode())
        n += 1
    return d.hexdigest(), n


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def slice_batch(ds, idxs):
    return {k: v[idxs.to(v.device)] for k, v in {"champs": ds.champs, "sides": ds.sides, "items": ds.items,
                                                   "query": ds.query, "inv": ds.inv, "hist": ds.hist}.items()}


def range_batch(ds, sl):
    return {k: v[sl] for k, v in {"champs": ds.champs, "sides": ds.sides, "items": ds.items,
                                    "query": ds.query, "inv": ds.inv, "hist": ds.hist}.items()}


def group_logits(scores, gid_dev, n_groups):
    """Ward-marginalised group log-scores: logsumexp of basket scores per group."""
    b = scores.shape[0]
    gidx = gid_dev.unsqueeze(0).expand(b, -1)
    gmax = torch.full((b, n_groups), float("-inf"), device=scores.device)
    gmax.scatter_reduce_(1, gidx, scores, reduce="amax", include_self=True)
    gsum = torch.zeros(b, n_groups, device=scores.device)
    gsum.scatter_add_(1, gidx, (scores - gmax.gather(1, gidx)).exp())
    return gmax + gsum.log()


def setup(smoke, need_gpu=True, load_fit=True, load_cohort=True, cohort_tensors=True, load_model=True):
    started = time.monotonic()
    stamp = "smoke" if smoke else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    schema = json.loads((generation / "split_manifest.json").read_text(encoding="utf-8"))["input_schema"]
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    config = torch.load(ref_artifact, map_location="cpu", weights_only=False)["config"]
    config = {**config, "max_items": schema["max_modeled_inventory_items"],
              "max_copies": schema["max_copies_of_one_item"],
              "max_basket_items": schema["max_basket_items"]}
    tp.apply_config(config)
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    tp.D_MODEL, tp.N_LAYERS, tp.FF_DIM, tp.N_HEADS = (int(config[k]) for k in ("d_model", "layers", "ff_dim", "heads"))
    assert not tp.USE_HISTORY and tp.D_MODEL == 64

    assignment = json.loads((generation / "split_assignment.json").read_text(encoding="utf-8"))
    train_set = {m for m, f in assignment.items() if f == "train"}
    val_set = {m for m, f in assignment.items() if f == "val"}
    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path, val_path = generation / "visits_train.jsonl", generation / "visits_val.jsonl"
    champ_ids, item_ids, label_counts, catalogue, n_fit_all, _ = v1.first_pass(train_path, train_set, None, None)
    champ_index, item_index = tp._index_map(champ_ids), tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets = [k for k, _ in sorted(catalogue.items(), key=lambda kv: (-kv[1], kv[0]))]
    n_cat = len(baskets) + 1
    device = torch.device("cuda" if need_gpu and torch.cuda.is_available() else "cpu")
    gids, group_main = group_map(baskets)
    n_groups = len(group_main)
    cat_idx, cat_cnt, n_items = basket_tensors(baskets, n_cat, device)
    report1 = json.loads((STAGE1 / "bf_stage1_report_full.json").read_text(encoding="utf-8"))
    assert len(baskets) == report1["catalogue"] and n_groups == report1["groups"], "catalogue differs from stage 1"
    assert n_fit_all == report1["fit_rows"], "FIT row count differs from stage 1"
    ckpt = STAGE1 / "bf_newgen_ep16_full.pt"
    ckpt_sha = sha256_file(ckpt)
    assert ckpt_sha == report1["checkpoint"]["sha256"], "R1 checkpoint digest differs from the stage-1 report"
    arch = (len(champ_index), len(item_index), len(label_index), len(item_index) + 1, n_items)
    S = SimpleNamespace(stamp=stamp, smoke=smoke, generation=generation, manifest_gen=manifest_gen, config=config,
                        train_set=train_set, val_set=val_set, train_path=train_path, val_path=val_path,
                        champ_index=champ_index, item_index=item_index, label_index=label_index,
                        baskets=baskets, n_cat=n_cat, gids=gids, gid=gids.to(device), group_main=group_main,
                        n_groups=n_groups, cat_idx=cat_idx, cat_cnt=cat_cnt, n_items=n_items, device=device,
                        dragon=dragon, purchase_sequence=purchase_sequence, report1=report1, ckpt=ckpt,
                        ckpt_sha=ckpt_sha, arch=arch, n_fit_all=n_fit_all)
    if load_fit:
        expect = {"history": False, "extras": True, "gold_est": True, "gold_x": True, "runes": True, "pov_gold": True}
        fit_ds = tp.ShopDataset.from_cache(STAGE1 / "cache_fit_full.pt", champ_index, item_index, label_index,
                                           dragon, expect=expect)
        fit_targets = torch.load(STAGE1 / "targets_fit_full.pt", weights_only=True)
        n_fit = fit_ds.champs.shape[0]
        assert n_fit == n_fit_all and fit_targets.shape[0] == n_fit
        if smoke:
            n_fit = min(SMOKE_FIT, n_fit)
            for name in ("champs", "sides", "items", "query", "inv", "hist"):
                setattr(fit_ds, name, getattr(fit_ds, name)[:n_fit])
            fit_targets = fit_targets[:n_fit]
        if device.type == "cuda":
            fit_ds.to_device(device)
        S.fit_ds, S.fit_targets, S.n_fit = fit_ds, fit_targets.to(device), n_fit
    if load_cohort:
        cap = SMOKE_COHORT if smoke else None
        val_rows = list(v1.stream_split(val_path, set(), False, champ_index, dragon, cap))
        n_val = len(val_rows)
        cohort_games = sorted({r["match_id"] for r in val_rows})
        assert set(cohort_games) <= val_set
        S.val_rows, S.n_val, S.cohort_games = val_rows, n_val, cohort_games
        if cohort_tensors:
            val_ds = tp.ShopDataset.from_stream(iter(val_rows), n_val, champ_index, item_index, label_index, dragon)
            if device.type == "cuda":
                val_ds.to_device(device)
            S.val_ds = val_ds
    if load_model:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = v1.BasketNet(*arch).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        S.blob, S.model = blob, model
    S.setup_seconds = round(time.monotonic() - started, 1)
    return S


def load_econ_fit(stamp):
    """The FIT feature tensor: one file, or the shards the builder left when RAM was short."""
    single = OUT_DIR / f"econ_fit_{stamp}.pt"
    if single.exists():
        return torch.load(single, weights_only=True)
    manifest = json.loads((OUT_DIR / f"econ_fit_{stamp}_shards.json").read_text(encoding="utf-8"))
    # Preallocate and fill shard by shard: torch.cat would hold the shards AND
    # the result at once (2 x 1.9 GB on the full FIT), which is what pushed the
    # fine-tune's job over its commit ceiling on 2026-09-26.
    full = None
    for sh in manifest["shards"]:
        part = torch.load(OUT_DIR / sh["path"], weights_only=True)
        if full is None:
            full = torch.empty((manifest["n_rows"], *part.shape[1:]), dtype=part.dtype)
        assert part.shape[0] == sh["rows"]
        full[sh["start"]:sh["start"] + sh["rows"]] = part
        del part
    assert full is not None and full.shape[0] == manifest["n_rows"]
    return full


def reference_rows(n):
    """R1's stage-1 cohort rows (prefix for smoke; the cohort stream order is deterministic)."""
    return load_rows(STAGE1 / "rows_new_bf_full.jsonl", n)


def decode_batch(order, rows, group_main, S):
    return [wardfree_record(row, decode(order[j].tolist(), group_main, row, S.dragon, S.purchase_sequence))
            for j, row in enumerate(rows)]


def pilot(step, n_fit, device, seed=99):
    """Seconds per effective batch, steady state (WARMUP untimed, TIMED timed)."""
    g = torch.Generator().manual_seed(seed)
    batches = [torch.randperm(n_fit, generator=g)[:p2.EFFECTIVE] for _ in range(WARMUP + TIMED)]
    t0 = None
    for i, idxs_all in enumerate(batches):
        if i == WARMUP:
            sync(device)
            t0 = time.monotonic()
        step(idxs_all)
    sync(device)
    return (time.monotonic() - t0) / TIMED


def bound_check(report, key, seconds_per_batch, n_fit, epochs_total, path, smoke):
    """Project and enforce the declared bound; exit EXIT_BOUND with a published reason."""
    hours = seconds_per_batch * (n_fit / p2.EFFECTIVE) * epochs_total / 3600
    report["pilot"] = {"sec_per_effective_batch": round(seconds_per_batch, 3), "epochs_total": epochs_total,
                       "projected_hours": round(hours, 2), "bound_hours": BOUNDS_H[key],
                       "warmup_batches_untimed": WARMUP, "timed_batches": TIMED}
    print(f"pilot {key}: {seconds_per_batch:.3f} s/lote -> {hours:.2f} h proyectadas (tope {BOUNDS_H[key]})", flush=True)
    if hours > BOUNDS_H[key] and not smoke:
        report["aborted"] = f"pilot projects {hours:.2f} h > declared bound {BOUNDS_H[key]} h"
        report["finished_utc"] = now_utc()
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("ABORTADO: " + report["aborted"], flush=True)
        sys.exit(EXIT_BOUND)
    return hours


def compare(arms, pairs, report, draw_pairs=()):
    """Paired-by-game contrasts (first-exact + top-3), slices, movement; 2,000-draw CIs for draw_pairs."""
    ident = [(r["match_id"], r["ts"], r.get("champion")) for r in next(iter(arms.values()))]
    for k, recs in arms.items():
        assert [(r["match_id"], r["ts"], r.get("champion")) for r in recs] == ident, f"identidad desalineada: {k}"
    report.setdefault("aggregates", {})
    report.setdefault("paired", {})
    report.setdefault("slices", {})
    report.setdefault("movement", {})
    for k, recs in arms.items():
        report["aggregates"][k] = p2.aggregate(recs)
    for x, y in pairs:
        for metric in ("first_action_exact", "any_of_three_exact"):
            tag = f"{x}_vs_{y}" + ("_any3" if metric.startswith("any") else "")
            report["paired"][tag] = p2.paired(arms[x], arms[y], metric)
        for label, pred in SUBSETS.items():
            report["slices"][f"{x}_vs_{y}_{label}"] = p2.paired([r for r in arms[x] if pred(r)],
                                                                  [r for r in arms[y] if pred(r)], "first_action_exact")
        c = bk = 0
        for a, o in zip(arms[x], arms[y]):
            c += a["first_action_exact"] > o["first_action_exact"]
            bk += a["first_action_exact"] < o["first_action_exact"]
        report["movement"][f"{x}_vs_{y}"] = {"corrected": c, "broken": bk, "net": c - bk}
    for x, y in draw_pairs:
        per = {t: defaultdict(lambda: [0.0, 0]) for t in ("x", "y")}
        for t, recs in (("x", arms[x]), ("y", arms[y])):
            for r in recs:
                per[t][r["match_id"]][0] += r["first_action_exact"]
                per[t][r["match_id"]][1] += 1
        report["paired"][f"{x}_vs_{y}"]["ci95_2000draws"] = list(v1.paired_bootstrap(per["x"], per["y"], seed=16, n=2000))
    return report


def beats(report, x, y):
    """Declared reading: positive with the 95% CI excluding zero."""
    p = report["paired"][f"{x}_vs_{y}"]
    return bool(p["diff"] > 0 and p["ci95"][0] > 0)


def write_report(path, report):
    report["finished_utc"] = now_utc()
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {path}", flush=True)


def progress_line(path, entry):
    with open(path, "a", encoding="utf-8", newline=chr(10)) as fh:
        fh.write(json.dumps(entry) + chr(10))
        fh.flush()
        os.fsync(fh.fileno())
