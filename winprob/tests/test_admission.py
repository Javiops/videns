"""Synthetic admission safety regressions; no production payloads or DB."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from winprob.admission import canonical_bytes, digest_bytes, extraction_candidates, prepare_manifest, verify_manifest_sources, write_manifest


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data"
        self.root.mkdir()
        self.allow = self.put("dataset_generations/g/split_assignment.json", {
            "EUW1_1": "train", "EUW1_2": "train", "EUW1_3": "val", "EUW1_4": "post_cutoff"})

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def manifest(self, **kwargs):
        return prepare_manifest(self.root, [self.allow], "synthetic-v1", **kwargs)

    def external_reservation(self, ids):
        source = Path(self.tmp.name) / "external-split.json"
        source.write_text(json.dumps({"test": ids}), encoding="utf-8")
        artifact = Path(self.tmp.name) / "reserved_ids.json"
        artifact.write_bytes(canonical_bytes({"schema": "winprob-external-reservations-v1",
            "source_project": "synthetic-external", "source_metadata_path": str(source),
            "source_metadata_sha256": digest_bytes(source.read_bytes()), "reserved_test_ids": ids}))
        artifact.with_suffix(".sha256").write_text(digest_bytes(artifact.read_bytes()) + "  " + artifact.name + "\n", encoding="ascii")
        return artifact, source

    def test_external_test_overrides_build_train_before_metadata_lookup(self):
        artifact, _ = self.external_reservation(["EUW1_1"])
        metadata = {"EUW1_2": {"game_creation_ms": 1000, "patch": "16.18", "queue_id": 420},
                    "EUW1_3": {"game_creation_ms": 2000, "patch": "16.18", "queue_id": 420}}
        with patch("winprob.admission.read_game_metadata", return_value=metadata) as lookup:
            result = self.manifest(external_reservations=[artifact], db_path=Path("unused"))
            self.assertNotIn("EUW1_1", lookup.call_args.args[1])
        self.assertNotIn("EUW1_1", {entry["match_id"] for entry in result["games"]})
        self.assertEqual(result["external_reservations_excluded_games"], 1)
        verify_manifest_sources(result)

    def test_external_artifact_tampering_invalidates_manifest(self):
        artifact, _ = self.external_reservation(["EUW1_1"])
        result = self.manifest(external_reservations=[artifact])
        artifact.write_bytes(artifact.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "sidecar mismatch"):
            verify_manifest_sources(result)
        artifact.with_suffix(".sha256").write_text(digest_bytes(artifact.read_bytes()) + "  " + artifact.name, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            verify_manifest_sources(result)

    def test_external_source_metadata_change_invalidates_manifest(self):
        artifact, source = self.external_reservation(["EUW1_1"])
        result = self.manifest(external_reservations=[artifact])
        source.write_bytes(source.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "source metadata changed"):
            verify_manifest_sources(result)

    def test_prior_roles_preserved_when_external_exclusions_change_fractions(self):
        metadata = {"EUW1_1": {"game_creation_ms": 1000, "patch": "16.18", "queue_id": 420},
                    "EUW1_2": {"game_creation_ms": 2000, "patch": "16.18", "queue_id": 420},
                    "EUW1_3": {"game_creation_ms": 3000, "patch": "16.18", "queue_id": 420}}
        with patch("winprob.admission.read_game_metadata", return_value=metadata):
            prior = self.manifest(db_path=Path("unused"))
        next(entry for entry in prior["games"] if entry["match_id"] == "EUW1_2")["role"] = "confirmatory_reserved"
        prior_path, _ = write_manifest(prior, Path(self.tmp.name) / "prior")
        artifact, _ = self.external_reservation(["EUW1_1"])
        with patch("winprob.admission.read_game_metadata", return_value=metadata):
            result = self.manifest(db_path=Path("unused"), external_reservations=[artifact], prior_manifest=prior_path)
        self.assertEqual(next(entry for entry in result["games"] if entry["match_id"] == "EUW1_2")["role"], "confirmatory_reserved")
        self.assertEqual(result["prior_manifest"]["prior_confirmatory_games"], 1)
        verify_manifest_sources(result)
        next(entry for entry in result["games"] if entry["match_id"] == "EUW1_2")["role"] = "development"
        with self.assertRaisesRegex(ValueError, "prior frozen role changed"):
            verify_manifest_sources(result)

    def test_hidden_test_reservation_overrides_train_allowlist(self):
        self.put("acceptance/old/coverage/preserved_split_assignment.json", {"EUW1_1": "test"})
        result = self.manifest()
        self.assertEqual([entry["match_id"] for entry in result["games"]], ["EUW1_2", "EUW1_3"])
        self.assertEqual(result["excluded_games"]["reserved_test"], 1)
        self.assertFalse(result["raw_payload_reads"])

    def test_original_plan_reserves_test_and_historical_val_never_fits(self):
        self.put("archive/original_split_plan.json", {"assignment": {"EUW1_1": "test", "EUW1_2": "val"}})
        result = self.manifest()
        self.assertTrue(all(entry["role"] == "development_exposed" for entry in result["games"]))

    def test_unallowlisted_and_post_cutoff_never_admitted(self):
        self.put("archive/split_assignment.json", {"EUW1_9": "train"})
        result = self.manifest()
        self.assertEqual({entry["match_id"] for entry in result["games"]}, {"EUW1_1", "EUW1_2", "EUW1_3"})

    def test_malformed_or_unknown_fold_fails_closed(self):
        path = self.put("archive/split_plan.json", {"assignment": {"EUW1_9": "validation_new"}})
        with self.assertRaisesRegex(ValueError, "Unknown split fold"):
            self.manifest()
        path.write_text("{", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.manifest()

    def test_source_changes_and_new_reservations_invalidate_manifest(self):
        result = self.manifest()
        verify_manifest_sources(result)
        self.put("new/split_assignment.json", {"EUW1_2": "test"})
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            verify_manifest_sources(result)

    def test_raw_archive_subtree_is_not_scanned(self):
        self.put("raw_match_v5/timelines/split_plan.json", {"not_a_game": "unknown"})
        self.assertEqual(len(self.manifest()["split_metadata_sources"]), 1)

    def test_output_is_versioned_and_refuses_overwrite(self):
        result = self.manifest()
        path, digest = write_manifest(result, Path(self.tmp.name) / "artifacts")
        self.assertTrue(path.exists())
        self.assertEqual(len(digest), 64)
        with self.assertRaises(FileExistsError):
            write_manifest(result, Path(self.tmp.name) / "artifacts")

    def test_temporal_metadata_query_never_reads_outcomes(self):
        games = {f"EUW1_{index}": "train" for index in range(1, 21)}
        games["EUW1_21"] = "val"
        self.allow.write_text(json.dumps(games), encoding="utf-8")
        self.put("sealed/split_assignment.json", {"EUW1_1": "test", "EUW1_3": "val"})
        db = Path(self.tmp.name) / "metadata.db"
        with closing(sqlite3.connect(db)) as connection:
            # No outcome or payload columns exist: code must stay metadata-only.
            connection.execute("CREATE TABLE games (match_id TEXT PRIMARY KEY, game_creation INTEGER, patch TEXT, queue_id INTEGER)")
            connection.executemany("INSERT INTO games VALUES (?, ?, ?, ?)",
                                   [(mid, 1000 + (int(mid.split('_')[1]) // 2) * 1000, "16.18", 420) for mid in games])
            connection.commit()
        result = self.manifest(db_path=db)
        self.assertEqual(result["state"], "temporal_manifest_ready")
        self.assertNotIn("EUW1_1", {entry["match_id"] for entry in result["games"]})
        for entry in result["games"]:
            if entry["match_id"] in {"EUW1_3", "EUW1_21"}:
                self.assertEqual(entry["role"], "development_exposed")
        eligible = [entry for entry in result["games"] if not entry["historical_validation"]]
        role_by_timestamp = {}
        for entry in eligible:
            stamp = entry["game_creation_ms"]
            if stamp in role_by_timestamp:
                self.assertEqual(entry["role"], role_by_timestamp[stamp])
            role_by_timestamp[stamp] = entry["role"]
        self.assertGreater(result["split_policy"]["confirmatory_games"], 0)
        windows = result["temporal_windows"]
        self.assertLess(windows["train"]["max_creation_ms"], windows["calibration"]["min_creation_ms"])
        self.assertLess(windows["calibration"]["max_creation_ms"], windows["development"]["min_creation_ms"])
        self.assertLess(windows["development"]["max_creation_ms"], windows["confirmatory_reserved"]["min_creation_ms"])
        sample = extraction_candidates(result, max_games=8)
        self.assertEqual(len(sample), 8)
        self.assertFalse(any(entry["role"] in {"confirmatory_reserved", "development_exposed"} for entry in sample))
        self.assertEqual(sample, extraction_candidates(result, max_games=8))


if __name__ == "__main__":
    unittest.main()
