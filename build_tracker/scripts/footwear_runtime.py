"""Run the recovered 16.17 experiment with explicit process-local semantics.

Usage: python scripts/footwear_runtime.py --entry scripts/train_prefix.py
       python scripts/footwear_runtime.py --entry scripts/eval_policy.py -- ...

PREFIX_ML_DIR must identify the completed recovered generation for training or
evaluation. Builders call activate() before importing their pipeline modules.
Production imports, ingestion and other Python processes remain unchanged.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

RUNTIME_VERSION = "footwear-development-runtime-v1"
RECONSTRUCTION_VERSION = "decision-start-footwear-v4"
FOOTWEAR_STATE_VERSION = "footwear-timer-slots-v1"
FOOTWEAR_FIELDS = ("footwear_state_version", "magical_footwear_selected", "footwear_granted")
TIMING_GUARD_MS = 1000
_ACTIVE = False


def declaration() -> dict:
    return {
        "schema": 1,
        "runtime_version": RUNTIME_VERSION,
        "patch": "16.17",
        "reconstruction_version": RECONSTRUCTION_VERSION,
        "footwear_state_version": FOOTWEAR_STATE_VERSION,
        "economics_version": "footwear-economics-16.17-v1",
        "inventory_version": "footwear-timer-inventory-elixirs-prototype-v1",
        "timing_guard_ms": TIMING_GUARD_MS,
        "delivery_time_independently_observed": False,
    }


def validate_state(row: dict) -> dict:
    if (row.get("footwear_state_version") != FOOTWEAR_STATE_VERSION
            or type(row.get("magical_footwear_selected")) is not bool
            or type(row.get("footwear_granted")) is not bool
            or row["footwear_granted"] and not row["magical_footwear_selected"]):
        raise ValueError("Recovered footwear rows require explicit, consistent predecision state")
    return {key: row[key] for key in FOOTWEAR_FIELDS}


def _install_baseline(baseline) -> None:
    import app.ddragon as ddragon
    import footwear_economics as economics

    if getattr(baseline, "FOOTWEAR_RUNTIME_VERSION", None) == RUNTIME_VERSION:
        return
    old_example, old_others = baseline._example, baseline._others

    def others(event):
        rows = old_others(event)
        players = [p for p in event.get("board") or [] if not p.get("is_self")]
        if len(rows) != len(players):
            raise ValueError("Footwear board projection lost a participant")
        for row, player in zip(rows, players, strict=True):
            row.update(validate_state(player))
        return rows

    def example(event, dragon):
        if event.get("reconstruction_version") != RECONSTRUCTION_VERSION:
            raise ValueError("Recovered exporter refuses a different reconstruction")
        state = validate_state(event)
        for player in event.get("board") or []:
            validate_state(player)
        result = old_example(event, dragon)
        if result is not None:
            result.update(state)
        return result

    baseline._others = others
    baseline._example = example
    baseline.inventory_state = economics.inventory_state
    baseline.decision_kind = economics.decision_kind
    baseline.default_dragon = ddragon.default_dragon
    baseline.dragon_for_patch = ddragon.dragon_for_patch
    baseline.RECONSTRUCTION_VERSION = RECONSTRUCTION_VERSION
    baseline.FOOTWEAR_RUNTIME_VERSION = RUNTIME_VERSION


def activate() -> dict:
    """Activate exactly once, before trainer/predictor imports, in this process."""
    global _ACTIVE
    if _ACTIVE:
        return declaration()
    forbidden = {"train_prefix", "scripts.train_prefix", "app.predictor", "app.conditional_policy",
                 "eval_policy", "scripts.eval_policy", "eval_conditional_policy", "scripts.eval_conditional_policy",
                 "eval_copy_policy", "scripts.eval_copy_policy"}
    if forbidden & sys.modules.keys():
        raise RuntimeError("Activate footwear runtime before importing the trainer, evaluator or predictor")
    import app.ddragon as ddragon
    import app.reconstruct as reconstruction
    import app.shop_econ as shop_econ
    import footwear_economics as economics

    source_dragon_for_patch = ddragon.dragon_for_patch
    source_data_dragon = ddragon.DataDragon

    def wrap_dragon(source):
        dragon = source if isinstance(source, economics.FootwearDataDragon) else economics.FootwearDataDragon(source)
        dragon.footwear_state_version = FOOTWEAR_STATE_VERSION
        return dragon

    def data_dragon(version=None):
        if version not in (None, "16.17.1"):
            raise ValueError("Recovered footwear runtime requires pinned Data Dragon 16.17.1")
        return wrap_dragon(source_data_dragon("16.17.1"))

    @lru_cache(maxsize=1)
    def dragon_for_patch(patch):
        if patch != "16.17":
            raise ValueError("Recovered footwear runtime is restricted to patch 16.17")
        return wrap_dragon(source_dragon_for_patch(patch))

    ddragon.DataDragon = data_dragon
    ddragon.dragon_for_patch = dragon_for_patch
    ddragon.default_dragon = lambda: dragon_for_patch("16.17")
    for name in ("combine_cost", "components_of", "is_blocked", "try_purchase",
                 "buy_cost", "inventory_state", "decision_kind", "net_spent"):
        setattr(shop_econ, name, getattr(economics, name))
    from app import footwear_reconstruction
    reconstruction.RECONSTRUCTION_VERSION = RECONSTRUCTION_VERSION
    reconstruction.reconstruct_game = footwear_reconstruction.reconstruct_game
    reconstruction.FOOTWEAR_STATE_VERSION = FOOTWEAR_STATE_VERSION
    reconstruction.FOOTWEAR_RUNTIME_VERSION = RUNTIME_VERSION
    # Reconstruct-visits/perspective resolve reconstruct_game in their module
    # globals. Bind their helpers too so any qualified corpus adapter agrees.
    reconstruction.dragon_for_patch = dragon_for_patch
    reconstruction.combine_cost = economics.combine_cost
    import baseline
    _install_baseline(baseline)
    _ACTIVE = True
    os.environ["PREFIX_FOOTWEAR_RUNTIME"] = RUNTIME_VERSION
    return declaration()


def declaration_for_manifest(manifest: dict) -> dict:
    """Resolve reviewed semantics by explicit patch, never by process defaults."""
    patch = manifest.get("patch")
    if patch == "16.17":
        expected = declaration()
    elif patch == "16.18":
        if manifest.get("purpose") not in (
            "patch_development_1618", "isolated_reconstruction_qualification",
        ):
            raise ValueError("16.18 footwear semantics require the explicitly scoped benchmark purpose")
        from benchmark_1618_runtime import declaration as benchmark_declaration
        expected = benchmark_declaration()
    elif patch == "16.19":                      # patch day 16.19 (2026-09-29): same contract, its own audited runtime
        if manifest.get("purpose") not in (
            "patch_development_1619", "isolated_reconstruction_qualification",
        ):
            raise ValueError("16.19 footwear semantics require the explicitly scoped benchmark purpose")
        from benchmark_1619_runtime import declaration as benchmark_declaration
        expected = benchmark_declaration()
    else:
        raise ValueError(f"No reviewed footwear runtime for patch {patch!r}")
    if manifest.get("footwear_runtime") != expected:
        raise ValueError("Manifest does not bind the exact reviewed footwear runtime")
    return expected


def activate_for_manifest(manifest: dict, corpus_path=None, review_path=None) -> dict:
    """Activate a bound corpus/generation before importing its consumers."""
    expected = declaration_for_manifest(manifest)
    if manifest["patch"] == "16.17":
        active = activate()
    else:
        if manifest["patch"] == "16.19":
            from benchmark_1619_runtime import activate as activate_benchmark
        else:
            from benchmark_1618_runtime import activate as activate_benchmark
        review_path = review_path or os.environ.get("PREFIX_RUNTIME_REVIEW")
        if review_path is None and corpus_path is not None:
            review_path = Path(corpus_path) / "static_review" / "static_review.json"
        if review_path is None:
            raise ValueError("16.18 activation requires the audited corpus or an explicit static review")
        active = activate_benchmark(Path(review_path))
    if active != expected:
        raise ValueError("Active footwear runtime differs from the bound manifest")
    return active


def validate_generation_runtime(directory: Path) -> dict:
    directory = Path(directory).resolve()
    manifests = []
    for filename in ("generation_manifest.json", "split_manifest.json"):
        data = json.loads((directory / filename).read_text(encoding="utf-8"))
        declaration_for_manifest(data)
        manifests.append(data)
    if any(manifests[0].get(key) != manifests[1].get(key)
           for key in ("patch", "purpose", "footwear_runtime")):
        raise ValueError("Generation and split manifest runtime scopes disagree")
    return declaration_for_manifest(manifests[0])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, type=Path)
    parser.add_argument("--review", type=Path, help="explicit audited 16.18 static review")
    args, remainder = parser.parse_known_args(argv)
    entry = args.entry.resolve()
    allowed = {ROOT / "scripts" / name for name in (
        "train_prefix.py", "eval_policy.py", "eval_conditional_policy.py", "eval_copy_policy.py",
        "mine_basket_misses_v2.py", "eval_basket_composition.py", "mine_lead_misses_v1.py",
        "eval_data_scale_v1.py", "audit_lead_feasibility_v1.py", "oracle_budget_ceiling_v1.py",
        "train_basket_ab.py", "train_basket_phase2.py",
        "basket_phase3_prework.py", "basket_size_diagnostics.py", "basket_extension.py",
        "basket_econ_cache.py", "basket_residual_econ.py", "basket_econ_finetune.py",
        "basket_finetune_enrich.py", "basket_wardfree.py", "basket_grouped_decoder.py",
        "basket_val_window_check.py", "basket_capacity_ab.py",
        "basket_frozen_mixture.py", "basket_ward_loss_diagnostic.py",
        "basket_mixture_val.py", "basket_loss_contrast.py",
        "basket_latency_bench.py", "econ_features_parallel.py",
        "build_history_cache.py", "basket_history_contrast.py",
        "build_mixture_package.py", "mixture_displayed_policy.py",
        "basket_learning_curve.py", "basket_learned_gate.py",
        "basket_bf_newgen.py", "basket_newbf_mixture.py", "stage2_budget_pilot.py",
        "basket_econ_cache_newgen.py", "basket_wardfree_newgen.py", "basket_econ_finetune_newgen.py",
        "basket_stage2_mixture.py", "cohort_legality_v4.py", "build_mixture_package_v2.py",
        "prepare_transfer_caches.py", "basket_transfer_ab.py", "basket_transfer_mixture.py", "score_frozen_on_generation.py", "pair_frozen_on_generation.py",
    )}
    if entry not in allowed:
        raise ValueError("Only the isolated trainer and policy evaluator are valid runtime entries")
    directory = os.environ.get("PREFIX_ML_DIR")
    if not directory:
        raise ValueError("Recovered runtime requires explicit PREFIX_ML_DIR")
    validate_generation_runtime(Path(directory))
    manifest = json.loads((Path(directory) / "generation_manifest.json").read_text(encoding="utf-8"))
    qualification_path = Path(directory) / "qualification.json"
    qualification = (json.loads(qualification_path.read_text(encoding="utf-8"))
                     if qualification_path.exists() else {})
    activate_for_manifest(manifest, corpus_path=qualification.get("corpus"), review_path=args.review)
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    sys.argv = [str(entry), *remainder]
    # Canonical name matters: Predictor imports train_prefix and must share the
    # exact same configured module, model classes, stream reader and dataset.
    module = importlib.import_module(entry.stem)
    module.main()


if __name__ == "__main__":
    # Later provenance/row helpers may import this API by its canonical name.
    # They must observe the same activation guard as this CLI invocation.
    sys.modules.setdefault("footwear_runtime", sys.modules[__name__])
    main()
