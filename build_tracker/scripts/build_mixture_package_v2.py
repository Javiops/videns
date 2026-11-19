"""Materialise M2 (stage 2 on the v4 generation) as a versioned mixture package.

Owner + reviewer, 2026-09-27: "conservar y empaquetar M2 reutilizando lo
construido". Same package contract as `build_mixture_package.py` /
`mixture-50-50-v1` so `app/mixture_policy.MixturePolicy` loads it unchanged:

  P = 0.5 * softmax(group-logsumexp of econ_tuned's full-catalogue scores over
      the frozen top-128 domain of domain_parent, with econ correction)
    + 0.5 * softmax(wardfree scores)

with domain_parent = R1 (the new Bf), econ_tuned = econ_ft_new_ep8, wardfree =
wardfree_new_ep16. Differences from v1, all structural: the ward-free arm was
trained directly on the group space (9,333 ward-stripped groups of the v4
catalogue), so `baskets_B` is the group list in group order and the
permutation B -> A is the identity (asserted, recorded); the economic
standardisation comes from the fine-tune's persisted `econ_stats_full.pt`
(the same FIT-feasible contract). Every input is verified against
`frozen_reference_stage2.json` before anything is packaged. Serving never
reads TRAIN: the single offline read happens here.

Run via footwear_runtime with PREFIX_ML_DIR = the v4 generation and
--review = the v4 snapshot's static review.
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

import train_basket_ab as v1
import train_prefix as tp
from basket_vocabulary_probe import basket_key, membership_digest
from basket_wardfree import WARDS
from basket_bf_newgen import group_map
from basket_econ_cache import N_ECON

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "data" / "packages"
STAGE1 = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v4-bf-stage1"
STAGE2 = ROOT / "data" / "experiments" / "16.18-stage2-v4"
PACKAGE_NAME = "mixture-v4-m2-v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(manifest):
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), blob


def basket_tensors(baskets, n):
    items = sorted({i for b in baskets for p in b.split(",") for i in [int(p.split(":")[0])]})
    idx = {it: k + 1 for k, it in enumerate(items)}
    ci = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
    cc = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
    for row_i, b in enumerate(baskets, start=1):
        for slot, p in enumerate(b.split(",")[:v1.MAX_BASKET]):
            i, c = p.split(":")
            ci[row_i, slot] = idx[int(i)]
            cc[row_i, slot] = min(int(c), v1.MAX_BASKET)
    return ci, cc, items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=PACKAGE_NAME)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    out_dir = PKG_ROOT / args.name
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} ya existe; usa --force para reescribirlo")
    out_dir.mkdir(parents=True, exist_ok=True)

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    split_manifest = json.loads((generation / "split_manifest.json").read_text(encoding="utf-8"))
    schema = split_manifest["input_schema"]
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    base_blob = torch.load(ref_artifact, map_location="cpu", weights_only=False)
    # the stage-2 scripts featurised with the reference config PLUS the v4
    # schema limits; the package must apply exactly the same knobs
    config = {**dict(base_blob["config"]), "max_items": schema["max_modeled_inventory_items"],
              "max_copies": schema["max_copies_of_one_item"], "max_basket_items": schema["max_basket_items"]}
    tp.apply_config(config)
    tp.INVENTORY_ENCODING = "canonical-multiset-v1"
    assert not tp.USE_HISTORY

    assignment = json.loads((generation / "split_assignment.json").read_text(encoding="utf-8"))
    train_ids = sorted(m for m, f in assignment.items() if f == "train")
    cohort_ids = sorted(m for m, f in assignment.items() if f == "val")
    train_set = set(train_ids)
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"

    # ---------- the only read of TRAIN, offline ----------
    print("leyendo TRAIN (1 pasada, offline)...", flush=True)
    champ_ids, item_ids, label_counts, catalogue, n_fit, _ = v1.first_pass(train_path, train_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets_A = [k for k, _ in sorted(catalogue.items(), key=lambda kv: (-kv[1], kv[0]))]
    n_cat = len(baskets_A) + 1
    gids, groups = group_map(baskets_A)                     # exactly stage 1's / stage 2's group space
    n_act = len(groups)
    group_main = [sorted(g.items()) for g in groups]        # (item, count) pairs; [] for group 0
    baskets_B = [basket_key(group_main[g]) for g in range(1, n_act)]
    perm_B_to_A = torch.arange(n_act, dtype=torch.long)     # the ward-free arm scores the groups themselves
    keys_A = [None] + [basket_key(g) for g in group_main[1:]]
    assert len(set(keys_A)) == n_act, "claves de grupo duplicadas"
    assert [None] + baskets_B == keys_A
    positions_unchanged = n_act
    stage1 = json.loads((STAGE1 / "bf_stage1_report_full.json").read_text(encoding="utf-8"))
    assert len(baskets_A) == stage1["catalogue"] and n_act == stage1["groups"], "el catalogo no es el de la etapa 1"
    print(f"TRAIN leido ({time.monotonic() - started:.0f}s): catalogo {len(baskets_A)}, grupos {n_act}", flush=True)

    cat_idx_A, cat_cnt_A, items_A = basket_tensors(baskets_A, n_cat)
    cat_idx_B, cat_cnt_B, items_B = basket_tensors(baskets_B, n_act)

    # ---------- verify the INPUTS against the frozen reference ----------
    ref = json.loads((STAGE2 / "frozen_reference_stage2.json").read_text(encoding="utf-8"))
    comp = ref["components"]
    weights_src = {"domain_parent": ROOT / comp["domain_parent_R1"]["path"],
                   "econ_tuned": ROOT / comp["econ_ft_new_ep8"]["path"],
                   "wardfree": ROOT / comp["wardfree_new_ep16"]["path"]}
    expected = {"domain_parent": comp["domain_parent_R1"]["sha256"], "econ_tuned": comp["econ_ft_new_ep8"]["sha256"],
                "wardfree": comp["wardfree_new_ep16"]["sha256"]}
    verified = {}
    for tag, src in weights_src.items():
        got = sha256_file(src)
        if got != expected[tag]:
            raise SystemExit(f"INPUT NO VERIFICADO: {tag} {src}\n  esperado {expected[tag]}\n  hallado  {got}")
        verified[tag] = got
    stats_path = ROOT / comp["econ_stats"]["path"]
    if sha256_file(stats_path) != comp["econ_stats"]["sha256"]:
        raise SystemExit("INPUT NO VERIFICADO: econ_stats")
    stats = torch.load(stats_path, weights_only=True)
    assert ref["components"]["weights"] == {"econ_ft_new": 0.5, "wardfree_new": 0.5, "tuned": False}
    print(f"entradas verificadas contra frozen_reference_stage2.json: {len(verified) + 1}/4", flush=True)

    # ---------- payload ----------
    (out_dir / "weights").mkdir(exist_ok=True)
    weights_meta = {}
    for tag, src in weights_src.items():
        blob = torch.load(src, map_location="cpu", weights_only=False)
        dst = out_dir / "weights" / f"{tag}.pt"
        torch.save({"state_dict": blob["state_dict"]}, dst)   # parameters only; no optimizer/RNG state
        weights_meta[tag] = {"file": f"weights/{tag}.pt", "sha256": sha256_file(dst),
                             "source": str(src.relative_to(ROOT)).replace("\\", "/"), "source_sha256": verified[tag],
                             "source_had_provenance": "provenance" in blob, "source_had_config": "config" in blob}
        del blob
    torch.save({"group_ids": gids, "perm_B_to_A": perm_B_to_A, "cat_idx_A": cat_idx_A, "cat_cnt_A": cat_cnt_A,
                "cat_idx_B": cat_idx_B, "cat_cnt_B": cat_cnt_B,
                "econ_stats_mean": stats["mean"], "econ_stats_std": stats["std"]}, out_dir / "tensors.pt")
    (out_dir / "vocab.json").write_text(json.dumps({
        "champ_index": {str(k): v for k, v in champ_index.items()}, "item_index": {str(k): v for k, v in item_index.items()},
        "label_index": {str(k): v for k, v in label_index.items()}, "basket_items_A": items_A, "basket_items_B": items_B},
        sort_keys=True), encoding="utf-8")
    (out_dir / "catalogues.json").write_text(json.dumps({"baskets_A": baskets_A, "baskets_B": baskets_B,
                                                          "group_main": group_main}, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema": 1, "package": args.name, "kind": "mixture-policy", "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "rule": ("P = wA*softmax(group-logsumexp of econ_tuned's full-catalogue scores over the frozen top-128 "
                     "domain of domain_parent, with econ correction) + wB*softmax(wardfree scores over the ward-stripped "
                     "groups themselves; permutation B->A is the identity)"),
            "weights": {"A": 0.5, "B": 0.5}, "weights_tuned": False, "shortlist": v1.SHORTLIST, "options": 3,
            "decode": ("top-SHORTLIST by P, walk to 3 DISTINCT legal main purchases via purchase_sequence(action, row, "
                       "budget, dragon); group 0 = 'no main purchase' (SAVE), always legal"),
            "wards_stripped": sorted(WARDS)},
        "weights": weights_meta,
        "structures": {"file": "tensors.pt", "sha256": sha256_file(out_dir / "tensors.pt"), "n_catalogue_A": n_cat,
                       "n_actions": n_act, "permutation_is_bijection": True,
                       "permutation_positions_unchanged": positions_unchanged, "permutation_is_identity": True},
        "vocab": {"file": "vocab.json", "sha256": sha256_file(out_dir / "vocab.json")},
        "catalogues": {"file": "catalogues.json", "sha256": sha256_file(out_dir / "catalogues.json")},
        "featurization": {k: config[k] for k in sorted(("extras", "history", "gold_est", "gold_x", "runes", "pov_gold",
                                                        "query_dim", "max_items", "max_copies", "max_basket_items",
                                                        "inventory_encoding", "target_encoding", "d_model", "layers",
                                                        "ff_dim", "heads")) if k in config},
        "featurization_source": {"file": str(ref_artifact.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(ref_artifact),
                                 "schema_overrides_from": "generation split_manifest.input_schema (max_items, max_copies, max_basket_items)"},
        "economic_engine": {"n_econ": N_ECON, "features": "scripts/basket_econ_cache.py:econ_features",
                            "features_sha256": sha256_file(ROOT / "scripts/basket_econ_cache.py"),
                            "normalization": "FIT feasible-buy mean/std, dims 2..n_econ (fine-tune's persisted econ_stats)",
                            "normalization_source": str(stats_path.relative_to(ROOT)).replace("\\", "/"),
                            "normalization_sha256": comp["econ_stats"]["sha256"],
                            "conditional_policy_sha256": sha256_file(ROOT / "app/conditional_policy.py"),
                            "shop_econ_sha256": sha256_file(ROOT / "app/shop_econ.py")},
        "footwear_runtime": base_blob["provenance"]["footwear_runtime"],
        "static_data_sha256": dragon.signature,
        "generation": {"name": generation.name, "patch": manifest_gen["patch"],
                       "split_manifest_sha256": sha256_file(generation / "split_manifest.json"),
                       "generation_manifest_sha256": sha256_file(generation / "generation_manifest.json"),
                       "export_fingerprint": split_manifest["export_fingerprint"],
                       "train_games": len(train_ids), "train_games_sha256": membership_digest(train_ids),
                       "cohort_games": len(cohort_ids), "cohort_games_sha256": membership_digest(cohort_ids),
                       "fit_rows": n_fit, "cohort_rows": stage1["cohort_rows"]},
        "provenance": {"gold_input": config.get("gold_input", "prequential-v3"),
                       "training_split": f"v4 TRAIN ({len(train_ids)} games, {n_fit} visits, played to 2026-09-19)",
                       "selection_split": (f"v4 forward development cohort ({len(cohort_ids)} games, {stage1['cohort_rows']} decisions, "
                                           "2026-09-20/21); EXPOSED: eight configurations scored on it; not the sealed test"),
                       "manifest_sha256": sha256_file(generation / "generation_manifest.json"),
                       "export_fingerprint": split_manifest["export_fingerprint"], "eval_version": split_manifest.get("eval_version"),
                       "status": "experimental_reference",
                       "reference": "data/experiments/16.18-stage2-v4/frozen_reference_stage2.json",
                       "note": ("Practical reference R2 by the rule declared in docs/experiment_stage2_econ_wardfree_v1.md "
                                "(M2 vs R1 +0.794 [+0.691, +0.889] on the exposed cohort). Packaging does not make it "
                                "eligible for serving: promotion needs a displayed-policy report on a frozen validation, "
                                "zero unaffordable baskets, corpus independence, a measured online cost, and the owner's gate.")},
        "serving": {"reads_train": False, "adapter": "app/mixture_policy.py",
                    "session_rule": "score only a shop session the player started explicitly; never on a poll"},
    }
    digest, blob = canonical_digest(manifest)
    (out_dir / "manifest.json").write_bytes(blob)
    (out_dir / "package_id.json").write_text(json.dumps({
        "package": args.name, "manifest_sha256": digest, "manifest_file": "manifest.json",
        "identity_rule": ("sha256 of the canonical manifest bytes (sort_keys, compact separators); it binds every "
                          "component by its own sha256, so this single digest identifies the whole evaluated policy"),
        "created_utc": manifest["created_utc"]}, indent=2) + "\n", encoding="utf-8")
    print(f"paquete: {out_dir}")
    print(f"identidad (manifest sha256): {digest}")
    print(f"componentes: {len(weights_meta)} pesos + tensores + vocab + catalogos; catalogo {n_cat - 1}; acciones {n_act}; "
          f"permutacion identidad; listo en {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
