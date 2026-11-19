"""Materialise the frozen mixture as a self-contained, versioned package.

Protocol: docs/mixture_promotion_prework_v1.md (owner's decision, 2026-09-22:
extend the contract with a versioned package; distillation is a later
experiment, not a prerequisite).

The point of this script is a single property:

    SERVING MUST NEVER READ TRAIN.

Today every piece the decoder needs -- the vocabularies, the 7,937-basket
catalogue, the ward-stripped group map, the ward-free catalogue, the
canonical-key permutation, the economic normalization statistics -- is
rebuilt at startup by `first_pass` over `visits_train.jsonl`. A serving path
that does that is reading the training corpus to answer a live request. So
this reads TRAIN exactly ONCE, here, offline, and writes everything out.

The package is identified by the sha256 of its CANONICAL manifest (sorted
keys, no whitespace variation), which itself binds every component by its
own sha256. That single digest is the candidate "A" of a policy report:
having several models does not prevent identifying one evaluated policy.

What this script does NOT do, deliberately: it does not fabricate
provenance. The three checkpoints carry no `provenance` and no `config`
blocks, and packaging them does not make them eligible for serving. The
manifest records what is actually known and marks the rest
`not_established`, so the gap is visible instead of papered over.

Run via footwear_runtime with PREFIX_ML_DIR set.
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

import train_basket_ab as v1
import train_prefix as tp
from basket_vocabulary_probe import basket_key, membership_digest, split_fit_probe
from basket_econ_cache import N_ECON, OUT_DIR as CACHE_DIR
from basket_wardfree import WARDS, strip_counter, stripped_truth

PKG_ROOT = ROOT / "data" / "packages"
MIX_DIR = ROOT / "data" / "experiments" / "16.18-frozen-mixture-v1"
PACKAGE_NAME = "mixture-50-50-v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(manifest):
    """Identity of the package: sha256 of the canonical manifest bytes."""
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), blob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=PACKAGE_NAME)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing package directory")
    args = ap.parse_args()
    started = time.monotonic()
    out_dir = PKG_ROOT / args.name
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} ya existe; usa --force para reescribirlo")
    out_dir.mkdir(parents=True, exist_ok=True)

    generation = Path(os.environ["PREFIX_ML_DIR"]).resolve()
    from app.dataset_generations import verify_generation
    manifest_gen = verify_generation(generation)
    split_manifest = json.loads(
        (generation / "split_manifest.json").read_text(encoding="utf-8"))
    schema = split_manifest["input_schema"]

    # The featurization the frozen backbone was trained with, taken from the
    # artifact that actually records it -- never hand-copied.
    ref_artifact = ROOT / "data/experiments/16.18-whole-basket-v1/arm_a_full.pt"
    base_blob = torch.load(ref_artifact, map_location="cpu",
                           weights_only=False)
    config = dict(base_blob["config"])
    tp.apply_config(config)
    assert not tp.USE_HISTORY, "el paquete servido no usa historial"

    plan = json.loads((generation / "split_plan.json").read_text(
        encoding="utf-8"))
    fit_games, probe_games, _ = split_fit_probe(plan)
    assert membership_digest(fit_games) == v1.FROZEN_FIT_SHA
    assert membership_digest(probe_games) == v1.FROZEN_PROBE_SHA
    fit_set = set(fit_games)

    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest_gen["patch"])
    train_path = generation / "visits_train.jsonl"

    # ---------- the only reads of TRAIN, both OFFLINE ----------
    # Two passes, not one: `first_pass` builds the vocabularies and the
    # full-basket catalogue, and a second stream builds the ward-free
    # catalogue. Merging them would mean duplicating first_pass's logic here,
    # which is a worse trade than a second offline read. The property that
    # matters is unaffected: SERVING never reads TRAIN — these passes happen
    # here, once, and everything they produce is written into the package.
    print("leyendo TRAIN (2 pasadas, offline)...", flush=True)
    champ_ids, item_ids, label_counts, catalogue, n_fit, n_probe = \
        v1.first_pass(train_path, fit_set, None, None)
    champ_index = tp._index_map(champ_ids)
    item_index = tp._index_map(item_ids)
    label_index = {i: k for k, i in enumerate(sorted(label_counts))}
    baskets_A = [k for k, _ in sorted(catalogue.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
    wf_cat = Counter()
    for row in v1.stream_split(train_path, fit_set, True, champ_index,
                               dragon, None):
        t = stripped_truth(row)
        if t:
            wf_cat[basket_key(sorted(t.items()))] += 1
    baskets_B = [k for k, _ in sorted(wf_cat.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
    print(f"TRAIN leido ({time.monotonic() - started:.0f}s): catalogo "
          f"{len(baskets_A)}, ward-free {len(baskets_B)}", flush=True)

    # ---------- derived structures, materialised ----------
    n_cat = len(baskets_A) + 1
    group_of_key = {None: 0}
    group_main = [[]]                      # list of (item, count) pairs
    group_ids = torch.zeros(n_cat, dtype=torch.long)
    for cid, bkey in enumerate(baskets_A, start=1):
        stripped = strip_counter(Counter(
            {int(p.split(":")[0]): int(p.split(":")[1])
             for p in bkey.split(",")}))
        skey = basket_key(sorted(stripped.items())) if stripped else None
        if skey not in group_of_key:
            group_of_key[skey] = len(group_main)
            group_main.append(sorted(stripped.items()))
        group_ids[cid] = group_of_key[skey]
    n_act = len(group_main)
    assert n_act == len(baskets_B) + 1, (n_act, len(baskets_B))

    keys_A = [None] + [basket_key(g) if g else None
                       for g in group_main[1:]]
    pos_A = {k: i for i, k in enumerate(keys_A)}
    assert len(pos_A) == n_act, "claves de grupo duplicadas"
    perm_B_to_A = torch.tensor([pos_A[k] for k in [None] + baskets_B],
                               dtype=torch.long)
    assert len(set(perm_B_to_A.tolist())) == n_act, "permutacion no biyectiva"
    positions_unchanged = int((perm_B_to_A == torch.arange(n_act)).sum())

    def basket_tensors(baskets, n):
        items = sorted({i for b in baskets for p in b.split(",")
                        for i in [int(p.split(":")[0])]})
        idx = {it: k + 1 for k, it in enumerate(items)}
        ci = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
        cc = torch.zeros((n, v1.MAX_BASKET), dtype=torch.long)
        for row_i, b in enumerate(baskets, start=1):
            for slot, p in enumerate(b.split(",")[:v1.MAX_BASKET]):
                i, c = p.split(":")
                ci[row_i, slot] = idx[int(i)]
                cc[row_i, slot] = min(int(c), v1.MAX_BASKET)
        return ci, cc, items

    cat_idx_A, cat_cnt_A, items_A = basket_tensors(baskets_A, n_cat)
    cat_idx_B, cat_cnt_B, items_B = basket_tensors(baskets_B, n_act)

    # economic normalization: from the FULL FIT econ cache, materialised so
    # serving never touches that 610 MB artifact either
    econ_fit = torch.load(CACHE_DIR / "econ_fit_full.pt", weights_only=True)
    feas = econ_fit[:, :, 1] > 0
    stats_mean, stats_std = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    for d in range(2, N_ECON):
        stats_mean[d] = flat[:, d].mean()
        stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    del econ_fit, flat, feas

    frozen_mix = json.loads((MIX_DIR / "frozen_mixture_v1.json").read_text(
        encoding="utf-8"))
    wA, wB = frozen_mix["weights"]["A"], frozen_mix["weights"]["B"]
    assert not frozen_mix["weights_tuned"]
    assert positions_unchanged == frozen_mix["alignment"]["positions_unchanged"]

    # ---------- verify the INPUTS before packaging anything ----------
    # Recording the sha256 of whatever happens to sit at a path is not a
    # check: a substituted file would simply be packaged with its new digest
    # and the package would look internally consistent. So every input is
    # first matched against the frozen mixture manifest. Recording the digest
    # of the EXPORTED content afterwards serves a different purpose (it binds
    # the package's own bytes), and both are needed.
    expected = {}
    for arm_key in ("arm_A_grouped_econ", "arm_B_wardfree"):
        for field, spec in frozen_mix["components"][arm_key].items():
            if field == "role":
                continue
            expected[spec["path"]] = spec["sha256"]
    verified = {}
    for rel, want in sorted(expected.items()):
        got = sha256_file(ROOT / rel)
        if got != want:
            raise SystemExit(
                f"INPUT NO VERIFICADO: {rel}\n  esperado {want}\n  hallado  {got}\n"
                "  el paquete no se construye sobre entradas que no casan con "
                "frozen_mixture_v1.json")
        verified[rel] = got
    print(f"entradas verificadas contra frozen_mixture_v1.json: "
          f"{len(verified)}/{len(expected)}", flush=True)

    # ---------- write the payload ----------
    weights_src = {
        "domain_parent": ROOT / "data/experiments/16.18-whole-basket-v2/Bf_ep16.pt",
        "econ_tuned": ROOT / "data/experiments/16.18-basket-econ-finetune-v1/econ_ft_ep8.pt",
        "wardfree": ROOT / "data/experiments/16.18-wardfree-v1/wardfree_ep16.pt",
    }
    for tag, src in weights_src.items():
        rel = str(src.relative_to(ROOT)).replace("\\", "/")
        if rel not in {k.replace("\\", "/") for k in verified}:
            raise SystemExit(
                f"{rel} no figura entre las entradas verificadas de "
                "frozen_mixture_v1.json; no se empaqueta")
    (out_dir / "weights").mkdir(exist_ok=True)
    weights_meta = {}
    for tag, src in weights_src.items():
        blob = torch.load(src, map_location="cpu", weights_only=False)
        dst = out_dir / "weights" / f"{tag}.pt"
        # only the parameters travel: optimizer/scheduler/RNG are training
        # state and have no business in a serving package
        torch.save({"state_dict": blob["state_dict"]}, dst)
        weights_meta[tag] = {
            "file": f"weights/{tag}.pt",
            "sha256": sha256_file(dst),
            "source": str(src.relative_to(ROOT)),
            "source_sha256": sha256_file(src),
            "source_had_provenance": "provenance" in blob,
            "source_had_config": "config" in blob,
        }
        del blob

    torch.save({
        "group_ids": group_ids,
        "perm_B_to_A": perm_B_to_A,
        "cat_idx_A": cat_idx_A, "cat_cnt_A": cat_cnt_A,
        "cat_idx_B": cat_idx_B, "cat_cnt_B": cat_cnt_B,
        "econ_stats_mean": stats_mean, "econ_stats_std": stats_std,
    }, out_dir / "tensors.pt")

    (out_dir / "vocab.json").write_text(json.dumps({
        "champ_index": {str(k): v for k, v in champ_index.items()},
        "item_index": {str(k): v for k, v in item_index.items()},
        "label_index": {str(k): v for k, v in label_index.items()},
        "basket_items_A": items_A, "basket_items_B": items_B,
    }, sort_keys=True), encoding="utf-8")
    (out_dir / "catalogues.json").write_text(json.dumps({
        "baskets_A": baskets_A, "baskets_B": baskets_B,
        "group_main": group_main,
    }, sort_keys=True), encoding="utf-8")

    runtime_decl = base_blob["provenance"]["footwear_runtime"]
    manifest = {
        "schema": 1,
        "package": args.name,
        "kind": "mixture-policy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "rule": ("P = wA*softmax(group-logsumexp of econ_tuned's "
                     "full-catalogue scores over the frozen top-128 domain of "
                     "domain_parent, with econ correction) + wB*softmax("
                     "wardfree scores), the latter permuted into group order "
                     "by canonical multiset key"),
            "weights": {"A": wA, "B": wB},
            "weights_tuned": False,
            "shortlist": v1.SHORTLIST,
            "options": 3,
            "decode": ("top-SHORTLIST by P, walk to 3 DISTINCT legal main "
                       "purchases via purchase_sequence(action, row, budget, "
                       "dragon); group 0 = 'no main purchase' (SAVE), always "
                       "legal"),
            "wards_stripped": sorted(WARDS),
        },
        "weights": weights_meta,
        "structures": {
            "file": "tensors.pt", "sha256": sha256_file(out_dir / "tensors.pt"),
            "n_catalogue_A": n_cat, "n_actions": n_act,
            "permutation_is_bijection": True,
            "permutation_positions_unchanged": positions_unchanged,
        },
        "vocab": {"file": "vocab.json",
                  "sha256": sha256_file(out_dir / "vocab.json")},
        "catalogues": {"file": "catalogues.json",
                       "sha256": sha256_file(out_dir / "catalogues.json")},
        "featurization": {k: config[k] for k in sorted(
            ("extras", "history", "gold_est", "gold_x", "runes", "pov_gold",
             "query_dim", "max_items", "max_copies", "max_basket_items",
             "inventory_encoding", "target_encoding", "d_model", "layers",
             "ff_dim", "heads")) if k in config},
        "featurization_source": {
            "file": str(ref_artifact.relative_to(ROOT)),
            "sha256": sha256_file(ref_artifact)},
        "economic_engine": {
            "n_econ": N_ECON,
            "features": "scripts/basket_econ_cache.py:econ_features",
            "features_sha256": sha256_file(ROOT / "scripts/basket_econ_cache.py"),
            "normalization": "FIT feasible-buy mean/std, dims 2..n_econ",
            "normalization_source": "data/experiments/16.18-basket-residual-econ-v1/econ_fit_full.pt",
            "conditional_policy_sha256": sha256_file(ROOT / "app/conditional_policy.py"),
            "shop_econ_sha256": sha256_file(ROOT / "app/shop_econ.py"),
        },
        "footwear_runtime": runtime_decl,
        "static_data_sha256": dragon.signature,
        "generation": {
            "name": generation.name,
            "patch": manifest_gen["patch"],
            "split_manifest_sha256": sha256_file(
                generation / "split_manifest.json"),
            "fit_games_sha256": v1.FROZEN_FIT_SHA,
            "probe_games_sha256": v1.FROZEN_PROBE_SHA,
            "fit_rows": n_fit, "probe_rows": n_probe,
        },
        # Deliberately NOT fabricated. These are the six REQUIRED_PROVENANCE
        # keys; what is unknown says so, so the gap stays visible.
        "provenance": {
            "gold_input": config.get("gold_input", "prequential-v3"),
            "training_split": "TRAIN/fit (2,000 frozen games)",
            "selection_split": "internal chronological probe within TRAIN "
                               "(500 games) -- NOT a validation selection",
            "manifest_sha256": "not_established",
            "export_fingerprint": "not_established",
            "eval_version": "not_established",
            "status": "experimental_reference",
            "note": ("The three checkpoints carry no provenance and no config "
                     "block of their own. Packaging them does not make them "
                     "eligible for serving: a formal candidate needs real "
                     "data, training subset, recipe and selection recorded, "
                     "and its selection was probe-internal, not validation. "
                     "This field is left honest on purpose."),
        },
        "serving": {
            "reads_train": False,
            "adapter": "app/mixture_policy.py",
            "session_rule": ("score only a shop session the player started "
                             "explicitly; never on a poll"),
        },
    }
    digest, blob = canonical_digest(manifest)
    (out_dir / "manifest.json").write_bytes(blob)
    (out_dir / "package_id.json").write_text(json.dumps({
        "package": args.name,
        "manifest_sha256": digest,
        "manifest_file": "manifest.json",
        "identity_rule": ("sha256 of the canonical manifest bytes "
                          "(sort_keys, compact separators); it binds every "
                          "component by its own sha256, so this single digest "
                          "identifies the whole evaluated policy"),
        "created_utc": manifest["created_utc"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"paquete: {out_dir}")
    print(f"identidad (manifest sha256): {digest}")
    print(f"componentes: {len(weights_meta)} pesos + tensores + vocab + "
          f"catalogos; acciones {n_act}; permutacion intactas "
          f"{positions_unchanged}")
    print(f"listo en {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
