"""Throughput pilot for the whole-basket experiment (16.18-whole-basket-v1).

Go/no-go measurement, NOT a model: builds the fit-side basket catalogue,
encodes it with the Deep Sets basket encoder the experiment specifies, and
times the three costs that decide feasibility on the real device:

  1. full-catalogue bilinear scoring (states x ~8k baskets) + top-K,
  2. deep interaction scoring on the top-128 shortlist,
  3. the economy executor (purchase_sequence) per candidate — the likely
     decision-time bottleneck, measured under the activated 16.18 runtime.

Refuses to run if the recomputed fit/probe memberships do not match the
frozen digests from the strategy diagnostics (protocol: same split or
nothing). Reads ONLY visits_train.jsonl and the frozen sample. Writes a
JSON report; trains nothing; no validation/test reads.

Usage: python scripts/basket_pilot.py [--generation <dir>] [--out <dir>]
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from basket_vocabulary_probe import (basket_entries, basket_key,
                                     membership_digest, split_fit_probe)

FROZEN_FIT_SHA = "d534b9f7a2b7c0e54e02279212091d0965407803388e8059a1d69d2515003dba"
FROZEN_PROBE_SHA = "f53db64a07d55689a15303320d3e7af4b1ab441376d03b4f101221862f468765"
SAVE_SENTINEL = 999999
MAX_BASKET = 6
SHORTLIST = 128
STATE_DIM = 128


def build_catalogue(generation: Path, fit_set: set) -> tuple:
    """(baskets ordered by count desc then key asc, counts) from fit games."""
    vocab = Counter()
    n_fit_purchases = 0
    with open(generation / "visits_train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["match_id"] not in fit_set:
                continue
            entries = basket_entries(r.get("label_counts"))
            if not entries or any(i == SAVE_SENTINEL for i, _n in entries):
                continue
            vocab[basket_key(entries)] += 1
            n_fit_purchases += 1
    ordered = sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered], dict(ordered), n_fit_purchases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default=str(
        ROOT / "data/dataset_generations/16.18-reconstructible-development-v3"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    started = time.monotonic()
    generation = Path(args.generation)
    out_dir = Path(args.out) if args.out else (
        ROOT / "data" / "reports" /
        f"basket_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    plan = json.loads((generation / "split_plan.json").read_text(encoding="utf-8"))
    fit_games, probe_games, boundary = split_fit_probe(plan)
    fit_sha = membership_digest(fit_games)
    probe_sha = membership_digest(probe_games)
    if fit_sha != FROZEN_FIT_SHA or probe_sha != FROZEN_PROBE_SHA:
        raise SystemExit(
            "fit/probe membership does not match the frozen diagnostics — "
            f"fit {fit_sha[:12]} vs {FROZEN_FIT_SHA[:12]}, "
            f"probe {probe_sha[:12]} vs {FROZEN_PROBE_SHA[:12]}. "
            "The experiment runs on that exact split or not at all.")
    print(f"split verificado: fit {len(fit_games)} / probe {len(probe_games)} "
          f"(digests coinciden con la evidencia congelada)", flush=True)

    t0 = time.monotonic()
    baskets, counts, n_fit_purchases = build_catalogue(generation, set(fit_games))
    t_catalogue = time.monotonic() - t0
    items = sorted({i for b in baskets for part in b.split(",")
                    for i in [int(part.split(":")[0])]})
    print(f"catalogo fit: {len(baskets)} cestas, {len(items)} items, "
          f"{n_fit_purchases} compras fit ({t_catalogue:.1f}s)", flush=True)

    import torch
    import torch.nn as nn
    torch.manual_seed(16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (torch.cuda.get_device_name(0)
                   if device.type == "cuda" else "cpu")

    item_index = {it: k + 1 for k, it in enumerate(items)}   # 0 = pad
    idx = torch.zeros((len(baskets), MAX_BASKET), dtype=torch.long)
    cnt = torch.zeros((len(baskets), MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets):
        for slot, part in enumerate(b.split(",")[:MAX_BASKET]):
            i, n = part.split(":")
            idx[row_i, slot] = item_index[int(i)]
            cnt[row_i, slot] = min(int(n), MAX_BASKET)
    idx, cnt = idx.to(device), cnt.to(device)

    class BasketEncoder(nn.Module):
        """Deep Sets: phi over item+count embeddings, sum-pool, rho MLP."""
        def __init__(self, n_items, dim=STATE_DIM):
            super().__init__()
            self.item = nn.Embedding(n_items + 1, 64, padding_idx=0)
            self.count = nn.Embedding(MAX_BASKET + 1, 16, padding_idx=0)
            self.phi = nn.Sequential(nn.Linear(80, 128), nn.GELU(),
                                     nn.Linear(128, 128))
            self.rho = nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                     nn.Linear(128, dim))
        def forward(self, idx, cnt):
            x = torch.cat([self.item(idx), self.count(cnt)], dim=-1)
            x = self.phi(x) * (idx > 0).unsqueeze(-1)
            return self.rho(x.sum(dim=1))

    class DeepScorer(nn.Module):
        """Nonlinear interaction over (state, basket, hadamard) triples."""
        def __init__(self, dim=STATE_DIM):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(dim * 3, 256), nn.GELU(),
                                     nn.Linear(256, 64), nn.GELU(),
                                     nn.Linear(64, 1))
        def forward(self, state, basket):
            return self.mlp(torch.cat([state, basket, state * basket],
                                      dim=-1)).squeeze(-1)

    encoder = BasketEncoder(len(items)).to(device).eval()
    scorer = DeepScorer().to(device).eval()

    def timed(fn, repeats=5):
        if device.type == "cuda":
            torch.cuda.synchronize()
        fn()                                   # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.monotonic()
        for _ in range(repeats):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.monotonic() - t) / repeats

    timings = {}
    with torch.no_grad():
        t_enc = timed(lambda: encoder(idx, cnt))
        catalogue_vecs = encoder(idx, cnt)
        timings["catalogue_encode_ms"] = round(t_enc * 1000, 2)
        for batch in (256, 1024):
            states = torch.randn(batch, STATE_DIM, device=device)
            def stage1():
                scores = states @ catalogue_vecs.T
                torch.topk(scores, SHORTLIST, dim=1)
            t1 = timed(stage1)
            top = torch.topk(states @ catalogue_vecs.T, SHORTLIST, dim=1).indices
            shortlist_vecs = catalogue_vecs[top]           # [B, K, D]
            expanded = states.unsqueeze(1).expand(-1, SHORTLIST, -1)
            def stage2():
                scorer(expanded.reshape(-1, STATE_DIM),
                       shortlist_vecs.reshape(-1, STATE_DIM))
            t2 = timed(stage2)
            timings[f"batch{batch}"] = {
                "stage1_full_catalogue_topk_ms": round(t1 * 1000, 2),
                "stage2_deep_128_ms": round(t2 * 1000, 2),
                "visits_per_second": round(batch / (t1 + t2)),
            }
    vps = timings["batch1024"]["visits_per_second"]
    epoch_s = n_fit_purchases / vps

    # -- coste del ejecutor de economia por candidato, runtime activado
    manifest = json.loads((generation / "generation_manifest.json").read_text(encoding="utf-8"))
    import footwear_runtime as fr
    fr.validate_generation_runtime(generation)
    fr.activate_for_manifest(
        manifest, review_path=ROOT / "data/corpora/16.18-reconstructible-v3"
        / "static_review/static_review.json")
    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest["patch"])
    sample_path = (ROOT / "data/reports/strategy_analysis_20260920"
                   / "basket_probe_legality_sample_train.jsonl")
    rows = []
    with sample_path.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line)["row"])
            if len(rows) >= 100:
                break
    rng = random.Random(16)
    def to_action(b):
        return tuple(sorted(i for part in b.split(",")
                            for i in [int(part.split(":")[0])] * int(part.split(":")[1])))
    pairs = [(row, to_action(rng.choice(baskets[:1000])))
             for row in rows for _ in range(10)]
    t0 = time.monotonic()
    legal_n = sum(1 for row, action in pairs
                  if purchase_sequence(action, row, float(row["gold_est"]),
                                       dragon) is not None)
    t_exec = (time.monotonic() - t0) / len(pairs)
    exec_ms = t_exec * 1000

    report = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/experiment_whole_basket_v1.md",
        "device": device_name,
        "cuda_available": device.type == "cuda",
        "split_verified": {"fit_games_sha256": fit_sha,
                           "probe_games_sha256": probe_sha},
        "catalogue": {"baskets": len(baskets), "items": len(items),
                      "fit_purchases": n_fit_purchases,
                      "build_seconds": round(t_catalogue, 1)},
        "shortlist_k": SHORTLIST,
        "timings": timings,
        "scoring_epoch_estimate_seconds": round(epoch_s, 1),
        "economy_executor": {
            "pairs_timed": len(pairs),
            "ms_per_candidate": round(exec_ms, 3),
            "legal_fraction_random_pairs": round(legal_n / len(pairs), 3),
            "ms_to_filter_top128_per_visit": round(exec_ms * SHORTLIST, 1),
        },
        "note": ("random state vectors: measures the catalogue-scoring path, "
                 "not model quality; executor timed on real probe rows with "
                 "random catalogue candidates under the activated runtime"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot.json").write_text(json.dumps(report, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("device", "catalogue", "timings",
                       "scoring_epoch_estimate_seconds", "economy_executor")},
                     indent=1))
    print(f"\nreport: {out_dir / 'pilot.json'}")


if __name__ == "__main__":
    main()
