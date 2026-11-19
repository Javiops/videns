"""Opt in to the audited 16.19 semantics for a scoped development benchmark.

The reconstruction declaration and implementation remain frozen. This adapter
only supplies the constructor and module bindings needed by export, training
and the real displayed-policy evaluator in the same isolated Python process.
It does not enable production serving or clear the pipeline hold.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
_ACTIVE = False


def declaration() -> dict:
    from reconstructible_1619_runtime import declaration as audited_declaration
    return audited_declaration()


def activate(review_path: Path) -> dict:
    """Activate before any module captures trainer/predictor economics symbols."""
    global _ACTIVE
    if _ACTIVE:
        return declaration()
    forbidden = {
        "baseline", "scripts.baseline", "train_prefix", "scripts.train_prefix",
        "app.predictor", "app.conditional_policy", "eval_policy", "scripts.eval_policy",
        "eval_conditional_policy", "scripts.eval_conditional_policy",
        "eval_copy_policy", "scripts.eval_copy_policy",
    }
    if forbidden & sys.modules.keys():
        raise RuntimeError("Activate 16.19 benchmark runtime before export, trainer or policy imports")
    import reconstructible_1619_runtime as audited
    runtime = audited.activate(Path(review_path))
    import app.ddragon as ddragon
    import app.reconstruct as reconstruction
    import baseline
    import footwear_economics as economics

    dragon = ddragon.dragon_for_patch("16.19")

    def data_dragon(version=None):
        if version not in (None, "16.19.1"):
            raise ValueError("16.19 benchmark requires pinned Data Dragon 16.19.1")
        return dragon

    # Predictor constructs DataDragon directly, whereas exporter/trainer use
    # factory functions. All paths must share the same reviewed economics.
    ddragon.DataDragon = data_dragon
    baseline.default_dragon = ddragon.default_dragon
    baseline.dragon_for_patch = ddragon.dragon_for_patch
    baseline.inventory_state = economics.inventory_state
    baseline.decision_kind = economics.decision_kind
    baseline.RECONSTRUCTION_VERSION = runtime["reconstruction_version"]
    baseline.FOOTWEAR_RUNTIME_VERSION = runtime["runtime_version"]
    reconstruction.FOOTWEAR_STATE_VERSION = runtime["footwear_state_version"]
    reconstruction.FOOTWEAR_RUNTIME_VERSION = runtime["runtime_version"]
    _ACTIVE = True
    os.environ["PREFIX_FOOTWEAR_RUNTIME"] = runtime["runtime_version"]
    return runtime
