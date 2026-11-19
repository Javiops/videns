"""Create and validate the run plan for experiment 16.18-pov-gold-v1.

Builds data/experiments/16.18-pov-gold-v1/run_plan.json against the CURRENT
code tree (the v3 plan's code digests predate today's train_prefix changes and
the repository move), then proves it validates: verify_generation +
validate_development_run_plan + a trunk-stage validate_generation_training in
a subprocess with the exact launch environment, and an env-merge preflight for
the graft stage (its warm-start file cannot exist yet; every check before that
one must pass).

Refuses to overwrite an existing plan or run directory content: unique names,
never overwrite (protocol rule).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.dataset_artifacts import sha256
from app.dataset_generations import (validate_development_run_plan,
                                     verify_generation)

GEN = ROOT / "data" / "dataset_generations" / "16.18-reconstructible-development-v3"
RUN = ROOT / "data" / "experiments" / "16.18-pov-gold-v1"
V3_PLAN = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v3" / "run_plan.json"
PROTOCOL = ROOT / "docs" / "experiment_pov_gold_v1.md"

TRUNK = {"PREFIX_EPOCHS": "16", "PREFIX_FREEZE": "0", "PREFIX_INIT_FROM": "",
         "PREFIX_OUT": "trunk.pt", "PREFIX_SAVEW": "0", "PREFIX_TARGETW": "0"}
GRAFT = {"PREFIX_EPOCHS": "6", "PREFIX_FREEZE": "1",
         "PREFIX_INIT_FROM": "trunk.pt", "PREFIX_OUT": "candidate.pt",
         "PREFIX_SAVEW": "0.5", "PREFIX_TARGETW": "0.25"}


def fixed_environment() -> dict:
    return {
        "PREFIX_COSINE": "1", "PREFIX_DMODEL": "128", "PREFIX_EXTRAS": "1",
        "PREFIX_FF": "256", "PREFIX_GOLDEST": "1", "PREFIX_GOLDX": "1",
        "PREFIX_HEADS": "8", "PREFIX_HISTORY": "0", "PREFIX_LAYERS": "4",
        "PREFIX_ML_DIR": str(GEN), "PREFIX_POSW": "8",
        "PREFIX_POSW_CUSTOM": "0", "PREFIX_POSW_SCHED": "8:24",
        "PREFIX_RUNES": "1", "PREFIX_RUN_DIR": str(RUN),
        "PREFIX_SPLICE": "0", "PREFIX_TBLEND": "0",
        # The one knob this experiment moves (docs/experiment_pov_gold_v1.md).
        "PREFIX_POV_GOLD": "1",
    }


def main():
    if (RUN / "run_plan.json").exists():
        raise SystemExit(f"{RUN / 'run_plan.json'} already exists: unique "
                         "names, never overwrite")
    generation = verify_generation(GEN)
    v3 = json.loads(V3_PLAN.read_text(encoding="utf-8"))
    code = {}
    for name in v3["code_sha256"]:
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"code file from the v3 plan is missing: {name}")
        code[name] = sha256(path)

    plan = {
        "schema": 1,
        "benchmark_id": "16.18-pov-gold-v1",
        "purpose": generation["purpose"],
        "patch": "16.18",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "code_root": str(ROOT),
        "code_sha256": code,
        "dataset": str(GEN),
        "run_directory": str(RUN),
        "export_fingerprint": generation["export_fingerprint"],
        "generation_manifest_sha256": sha256(GEN / "generation_manifest.json"),
        "question": ("Does replacing the truth-layer board gold-diff features "
                     "(totalGold of all ten) with the POV-observable estimate "
                     "(visible item value + own wallet, PREFIX_POV_GOLD=1) "
                     "preserve the fixed-recipe candidate's advantage over the "
                     "conditional lookup on the v3 generation?"),
        "primary_metrics": ["first_action_exact", "action_family_match"],
        "populations": [
            "All scored validation decisions for first_action_exact",
            "Validation purchases with a known next final for action_family_match",
        ],
        "evaluation": {"split": "val", "games": 0, "compare_conditional": True},
        "evaluation_seed": 16,
        "evaluation_bootstrap": 400,
        "stopping_rule": ("One fixed trunk+graft training and exactly one "
                          "paired conditional-baseline evaluation; stop on any "
                          "failed gate; no sweeps, no promotion, no test "
                          "reads, no cross-export claims."),
        "resource_budget": {"max_train_processes": 1, "max_ram_gb": 10,
                            "minimum_free_disk_gb": 40,
                            "max_wall_seconds": 57600},
        "final_test_evaluation": False,
        "promotion_allowed": False,
        "production_hold_cleared": False,
        "owner_authorization": ("owner ruled POV-only inputs and instructed "
                                "execution on 2026-09-20 ('recuerda los POVs'; "
                                "'la key está viva, dale')"),
        "protocol_document": "docs/experiment_pov_gold_v1.md",
        "protocol_document_sha256": sha256(PROTOCOL),
        "fixed_environment": fixed_environment(),
        "training_stages": {"trunk": dict(TRUNK), "graft": dict(GRAFT)},
        "protected_before": {
            str(ROOT / "data" / "ml" / "prefix_model.pt"):
                (sha256(ROOT / "data" / "ml" / "prefix_model.pt")
                 if (ROOT / "data" / "ml" / "prefix_model.pt").is_file() else None),
            str(GEN / "generation_manifest.json"):
                sha256(GEN / "generation_manifest.json"),
            str(ROOT / "data" / "ml" / "deployment_manifest.json"):
                (sha256(ROOT / "data" / "ml" / "deployment_manifest.json")
                 if (ROOT / "data" / "ml" / "deployment_manifest.json").is_file() else None),
        },
    }
    RUN.mkdir(parents=True, exist_ok=True)
    plan_path = RUN / "run_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True),
                         encoding="utf-8")
    _, digest = validate_development_run_plan(GEN, RUN, generation, plan_path)
    print(f"plan valido: {plan_path}")
    print(f"  sha256 {digest}")

    # preflight de etapas en subproceso con el ENTORNO DE LANZAMIENTO exacto:
    # descubrir un desajuste aqui cuesta segundos; en la noche, una tanda.
    for stage, extra, esperado in (
            ("trunk", TRUNK, "ok"),
            # el warm-start de graft aun no existe; llegar a ESE error prueba
            # que todo lo anterior (merge de entorno, receta) valida
            ("graft", GRAFT, "warm start")):
        env = {**fixed_environment(), **extra,
               "PREFIX_STAGE": stage, "PREFIX_RUN_PLAN": str(plan_path)}
        code_py = (
            "import sys;"
            f"sys.path.insert(0, r'{ROOT}');"
            f"sys.path.insert(0, r'{ROOT / 'scripts'}');"
            "from app.dataset_generations import verify_generation;"
            "import train_prefix as tp;"
            "g = verify_generation(tp.ML_DIR);"
            "d = tp.validate_generation_training(g);"
            "print('stage ok:', d['stage'])"
        )
        import os
        r = subprocess.run(
            [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-c", code_py],
            capture_output=True, text=True, cwd=str(ROOT),
            env={**os.environ, **env})
        out = (r.stdout + r.stderr).strip()
        if esperado == "ok" and "stage ok: trunk" not in out:
            raise SystemExit(f"preflight {stage} FALLO:\n{out}")
        if esperado == "warm start" and "warm start" not in out:
            raise SystemExit(f"preflight {stage}: esperaba llegar al check de "
                             f"warm-start y no llego:\n{out}")
        print(f"preflight {stage}: OK")


if __name__ == "__main__":
    main()
