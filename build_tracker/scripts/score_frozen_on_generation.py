"""Score a FROZEN reference (R1 = new Bf, or M2 = package mixture-v4-m2-v1) on the validation cohort of
ANOTHER generation (patch 16.19 benchmark, docs/benchmark_1619_v1.md section 4), with the model's own world.

  * the model's vocabulary, catalogue, group space and Data Dragon (16.18.1) are the frozen ones: nothing is
    rebuilt from the target generation; the target rows are featurised and checked for legality with the
    model's pinned dragon (a frozen model is a frozen world);
  * rows carrying an item unknown to the model's vocabulary (e.g. 16.19's two new items) are scored anyway
    (unknown ids featurise as padding), COUNTED and listed; decode failures are counted, never dropped;
  * gate first: the model must reproduce its persisted v4-cohort rows 0/0/0 on the first --gate-rows rows of
    the v4 cohort in THIS process (binds the decoder, legality and static data to the frozen ones).

R1 mode runs through footwear_runtime with PREFIX_ML_DIR = the v4 generation (stage-1 vocabulary):
    footwear_runtime.py --entry scripts/score_frozen_on_generation.py --review <v4 review> -- --model R1 --generation <dir> --out <dir>
M2 mode runs DIRECTLY (the package activates its own runtime first):
    score_frozen_on_generation.py --model M2 --generation <dir> --out <dir>
Then pair_frozen_on_generation.py pairs the two row files.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

V4_GENERATION = ROOT / "data/dataset_generations/16.18-reconstructible-development-v4"
V4_REVIEW = ROOT / "data/acceptance/16.18_reconstructible_20260923/inputs/static_review.json"
STAGE1 = ROOT / "data/experiments/16.18-reconstructible-development-v4-bf-stage1"
STAGE2 = ROOT / "data/experiments/16.18-stage2-v4"
FROZEN_R1 = STAGE1 / "frozen_reference_v4.json"
FROZEN_R2 = STAGE2 / "frozen_reference_stage2.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_item_ids(row):
    ids = set()
    for item in row.get("inventory") or []:
        ids.add(int(item["item_id"] if isinstance(item, dict) else item))
    for k in (row.get("label_counts") or {}):
        ids.add(int(k))
    return ids


def unknown_report(rows, item_index):
    """Rows whose inventory or label carries an item the frozen vocabulary does not know."""
    rows_with_unknown, unknown_ids = 0, Counter()
    for row in rows:
        unk = {i for i in row_item_ids(row) if i > 0 and i != 999999 and i not in item_index}
        if unk:
            rows_with_unknown += 1
            unknown_ids.update(unk)
    return {"rows_with_unknown_items": rows_with_unknown, "unknown_item_ids": dict(sorted(unknown_ids.items()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("R1", "M2"), required=True)
    ap.add_argument("--generation", type=Path, required=True, help="target generation (its visits_val.jsonl is the cohort)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gate-rows", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=0, help="cap target rows (smoke)")
    args = ap.parse_args()
    started = time.monotonic()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    target = args.generation.resolve()
    if args.model == "M2":                       # the package runtime must be active before ANY trainer/policy import
        import app.mixture_policy as mp
        pkg_dir = ROOT / "data" / "packages" / "mixture-v4-m2-v1"
        active = mp.activate_runtime(pkg_dir, V4_REVIEW)
    from app.dataset_generations import verify_generation
    target_manifest = verify_generation(target)
    gen_name = target.name
    from basket_newbf_mixture import load_rows, mismatches
    from basket_wardfree import wardfree_record
    import train_basket_ab as v1
    import train_basket_phase2 as p2

    if args.model == "M2":
        policy = mp.load_package("mixture-v4-m2-v1")
        r2_ref = json.loads(FROZEN_R2.read_text(encoding="utf-8"))
        ref_path, ref_sha = STAGE2 / "rows_M2_mix_full.jsonl", r2_ref["rows"]["M2_mix"]
        champ_index, item_index, dragon = policy.champ_index, policy.item_index, policy.dragon
        identity = {"package": "mixture-v4-m2-v1", "package_identity_sha256": policy.digest, "runtime": active["runtime_version"]}

        def score(rows):
            recs = []
            for s in range(0, len(rows), 256):
                chunk = rows[s:s + 256]
                for row, opts in zip(chunk, policy.options(chunk)):
                    conv = [("save", None) if o["kind"] == "save" else ("buy", Counter({int(k): int(v) for k, v in o["items"].items()})) for o in opts]
                    recs.append(wardfree_record(row, conv))
            return recs
    else:
        from stage2_newgen import setup as stage2_setup, group_logits, decode_batch
        import torch
        S = stage2_setup(False, need_gpu=True, load_fit=False, load_cohort=False, load_model=True)
        r1_ref = json.loads(FROZEN_R1.read_text(encoding="utf-8"))
        ref_path, ref_sha = STAGE1 / "rows_new_bf_full.jsonl", r1_ref["rows"]["rows_new_bf_full.jsonl"]
        if S.ckpt_sha != r1_ref["checkpoint"]["sha256"]:
            raise SystemExit("R1 checkpoint digest differs from the frozen reference")
        champ_index, item_index, dragon = S.champ_index, S.item_index, S.dragon
        tp = sys.modules["train_prefix"]
        identity = {"checkpoint": str(S.ckpt), "checkpoint_sha256": S.ckpt_sha, "catalogue": len(S.baskets), "groups": S.n_groups}

        def score(rows):
            ds = tp.ShopDataset.from_stream(iter(rows), len(rows), champ_index, item_index, S.label_index, dragon)
            if S.device.type == "cuda":
                ds.to_device(S.device)
            recs = []
            with torch.no_grad():
                vecs = S.model.baskets.vectors(S.cat_idx, S.cat_cnt, torch.arange(S.n_cat, device=S.device))
                for s in range(0, len(rows), 512):
                    sl = slice(s, min(s + 512, len(rows)))
                    batch = {k: v[sl] for k, v in {"champs": ds.champs, "sides": ds.sides, "items": ds.items, "query": ds.query, "inv": ds.inv, "hist": ds.hist}.items()}
                    sc = S.model.scores(S.model.state(batch, S.device), vecs)
                    order = torch.topk(group_logits(sc, S.gid, S.n_groups).cpu(), min(v1.SHORTLIST, S.n_groups), dim=1).indices
                    recs.extend(decode_batch(order, rows[sl], S.group_main, S))
            return recs

    if sha256_file(ref_path) != ref_sha:
        raise SystemExit(f"{ref_path} digest differs from the frozen reference; no scoring")
    print(f"setup listo ({time.monotonic() - started:.0f}s): {args.model} {identity}", flush=True)

    # ---- gate: reproduce the persisted v4-cohort rows on a prefix, in this process ----
    t0 = time.monotonic()
    gate_rows = list(v1.stream_split(V4_GENERATION / "visits_val.jsonl", set(), False, champ_index, dragon, args.gate_rows))
    gate_recs = score(gate_rows)
    repro = mismatches(gate_recs, load_rows(ref_path, len(gate_rows)))
    gate_ok = all(v == 0 for v in repro.values())
    print(f"gate {args.model} on {len(gate_rows)} v4 rows: {repro} ({time.monotonic() - t0:.0f}s)", flush=True)
    if not gate_ok:
        (out / f"gate_failed_{args.model}_{gen_name}.json").write_text(json.dumps({"mismatches": repro}), encoding="utf-8")
        sys.exit(2)

    # ---- target cohort with the model's own world ----
    t0 = time.monotonic()
    rows = list(v1.stream_split(target / "visits_val.jsonl", set(), False, champ_index, dragon, args.limit or None))
    n = len(rows)
    games = sorted({r["match_id"] for r in rows})
    unknown = unknown_report(rows, item_index)
    print(f"cohorte {gen_name}: {n} decisiones / {len(games)} partidas; filas con objetos desconocidos {unknown['rows_with_unknown_items']} {unknown['unknown_item_ids']}", flush=True)
    recs = score(rows)
    assert len(recs) == n
    rows_path = out / f"rows_{args.model}_{gen_name}.jsonl"
    p2.dump(rows_path, recs)
    agg = p2.aggregate(recs)
    report = {"schema": 1, "protocol": "docs/benchmark_1619_v1.md section 4", "model": args.model, "identity": identity,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "target_generation": {"path": str(target), "name": gen_name, "patch": target_manifest.get("patch"),
                                    "manifest_sha256": sha256_file(target / "generation_manifest.json"),
                                    "rows_sha256": sha256_file(target / "visits_val.jsonl")},
              "scoring_world": {"ddragon_version": dragon.version, "static_data_sha256": dragon.signature,
                                "note": "frozen model, frozen vocabulary/catalogue/group space and pinned Data Dragon; target rows reconstructed under their own patch runtime"},
              "gate": {"v4_rows": len(gate_rows), "reference": str(ref_path), "reference_sha256": ref_sha, "mismatches": repro, "passed": True},
              "cohort": {"rows": n, "games": len(games), **unknown},
              "aggregates": agg, "rows_file": str(rows_path), "rows_file_sha256": sha256_file(rows_path),
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    (out / f"report_{args.model}_{gen_name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "rows": n, "games": len(games), "first_action_exact": agg["first_action_exact"]["value"],
                      "any_of_three_exact": agg.get("any_of_three_exact", {}).get("value"), "unknown_rows": unknown["rows_with_unknown_items"],
                      "elapsed_s": report["elapsed_seconds"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
