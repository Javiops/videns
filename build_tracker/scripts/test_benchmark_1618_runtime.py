"""Offline runtime binding tests; no raw games, exports or test payload reads."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from footwear_runtime import declaration, declaration_for_manifest, validate_generation_runtime
from reconstructible_1618_runtime import declaration as audited_declaration


class BenchmarkRuntimeTests(unittest.TestCase):
    def child(self, script):
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_declaration_is_exact_audited_identity(self):
        original = audited_declaration()
        manifest = {"patch": "16.18", "purpose": "patch_development_1618", "footwear_runtime": original}
        self.assertEqual(declaration_for_manifest(manifest), original)
        self.assertEqual(declaration_for_manifest({"patch": "16.17", "footwear_runtime": declaration()}), declaration())
        for changed in (
            {**manifest, "purpose": "patch_development"},
            {**manifest, "patch": "16.19"},
            {**manifest, "footwear_runtime": declaration()},
        ):
            with self.assertRaises(ValueError):
                declaration_for_manifest(changed)
        changed = deepcopy(manifest)
        changed["footwear_runtime"]["variant_sha256"]["footwear_economics"] = "0" * 64
        with self.assertRaises(ValueError):
            declaration_for_manifest(changed)

    def test_generation_and_split_must_share_scope(self):
        manifest = {"patch": "16.18", "purpose": "patch_development_1618", "footwear_runtime": audited_declaration()}
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("generation_manifest.json", "split_manifest.json"):
                (directory / name).write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_generation_runtime(directory), audited_declaration())
            (directory / "split_manifest.json").write_text(json.dumps({**manifest,
                "purpose": "isolated_reconstruction_qualification"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scopes disagree"):
                validate_generation_runtime(directory)

    def test_activation_pins_every_consumer_without_touching_audited_sources(self):
        self.child(r'''
import json, sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path.cwd() / "scripts"))
from app.dataset_artifacts import sha256
from reconstructible_1618_runtime import declaration as audited
from footwear_runtime import activate_for_manifest
expected = audited()
source_hashes = {name: sha256(Path(name)) for name in expected["base_code_sha256"]}
manifest = {"patch": "16.18", "purpose": "patch_development_1618", "footwear_runtime": expected}
with patch("httpx.get", side_effect=AssertionError("No network in offline runtime checks")):
    runtime = activate_for_manifest(manifest,
        review_path=Path("data/acceptance/16.18_reconstructible_20260912/inputs/static_review.json"))
    from app.ddragon import DataDragon, default_dragon, dragon_for_patch
    import app.reconstruct as reconstruct
    import app.shop_econ as shop
    import baseline, train_prefix, eval_policy, eval_conditional_policy
    import app.predictor as predictor
    import app.conditional_policy as conditional
    import footwear_economics as adapted
    dragon = DataDragon("16.18.1")
    assert dragon is default_dragon() is dragon_for_patch("16.18")
    assert predictor.DataDragon("16.18.1") is dragon
    assert train_prefix.dragon_for_patch("16.18") is dragon
    assert baseline.dragon_for_patch("16.18") is dragon
    assert train_prefix.RECONSTRUCTION_VERSION == baseline.RECONSTRUCTION_VERSION == runtime["reconstruction_version"]
    assert reconstruct.RECONSTRUCTION_VERSION == runtime["reconstruction_version"]
    assert shop.try_purchase is eval_policy.try_purchase is adapted.try_purchase
    assert train_prefix.combine_cost is adapted.combine_cost
    assert train_prefix is eval_policy.tp is predictor.tp
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "split_manifest.json"
        path.write_text(json.dumps({"eval_version": 1, "export_fingerprint": "synthetic1618",
            "gold_est_versions": {"prequential-v3": 1},
            "reconstruction_versions": {runtime["reconstruction_version"]: 1},
            "patch": "16.18", "ddragon_version": "16.18.1", "static_data_sha256": dragon.signature}))
        with patch.object(train_prefix, "SPLIT_MANIFEST_PATH", path):
            provenance = train_prefix.training_provenance()
        assert provenance["patch"] == "16.18" and provenance["ddragon_version"] == "16.18.1"
        assert provenance["reconstruction_version"] == runtime["reconstruction_version"]
        assert provenance["static_data_sha256"] == dragon.signature
    assert adapted.combine_cost(3020, {2422: 1}, dragon) == 800
    assert adapted.is_blocked(1001, [], dragon, magical_footwear_selected=True, footwear_granted=False)
    for call in (lambda: DataDragon("16.17.1"), lambda: dragon_for_patch("16.17")):
        try: call()
        except ValueError: pass
        else: raise AssertionError("Cross-patch static data accepted")
    assert activate_for_manifest(manifest, review_path="unused-after-valid-activation") == expected
assert audited() == expected
assert {name: sha256(Path(name)) for name in source_hashes} == source_hashes
''')

    def test_early_consumer_import_fails_closed(self):
        self.child(r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
import baseline
from benchmark_1618_runtime import activate
try:
    activate(Path("data/acceptance/16.18_reconstructible_20260912/inputs/static_review.json"))
except RuntimeError as error:
    assert "before export" in str(error)
else:
    raise AssertionError("A captured unadapted import was allowed")
''')


if __name__ == "__main__":
    unittest.main()
