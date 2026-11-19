"""Synthetic context-only extraction/admission tests; no real run inputs."""

import copy
from argparse import Namespace
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_revision_contexts as context
from winprob.features import extract_game_states
from test_features import fixture


class ProtectedFields(dict):
    forbidden = {"win", "winner", "winningTeam", "teams", "gameDuration", "gameEndTimestamp",
                 "kills", "goldEarned", "item0", "teamPosition", "individualPosition", "puuid"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError("Forbidden outcome/final field accessed: " + key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError("Forbidden outcome/final field accessed: " + key)
        return super().get(key, default)


def protect(value):
    if isinstance(value, dict):
        return ProtectedFields({key: protect(item) for key, item in value.items()})
    if isinstance(value, list):
        return [protect(item) for item in value]
    return value


def small_population():
    roles = ("train", "calibration", "development", "development", "development", "confirmatory_reserved")
    entries = [{"match_id": f"EUW1_{index + 1}", "role": role, "historical_validation": False,
                "patch": "16.18" if index < 3 else "16.19", "game_creation_ms": 1000 * (index + 1)}
               for index, role in enumerate(roles)]
    old = [{"match_id": entry["match_id"], "source_role": entry["role"], "role": context.ROLE_MAP[entry["role"]],
            "game_index": index, "rows": 3} for index, entry in enumerate(entries[:4])]
    return {"games": entries}, {"games": old}


class OutcomeFreeExtractionTests(unittest.TestCase):
    def test_exact_parity_with_base_extractor_without_accessing_outcomes(self):
        match, timeline = fixture()
        expected = [{key: value for key, value in row.items() if key != "y"} for row in extract_game_states(match, timeline)]
        actual = context.extract_context_states(protect(match), protect(timeline))
        self.assertEqual(actual, expected)
        self.assertTrue(all("y" not in row for row in actual))
        # Negative control proves the guard catches the original metadata path.
        with self.assertRaises(AssertionError):
            extract_game_states(protect(match), protect(timeline))

    def test_labels_and_final_fields_can_be_absent(self):
        match, timeline = fixture()
        expected = context.extract_context_states(match, timeline)
        del match["info"]["teams"]
        for player in match["info"]["participants"]:
            for key in ("kills", "teamPosition"):
                player.pop(key, None)
        timeline["info"]["frames"][-1]["events"][0].pop("winningTeam", None)
        self.assertEqual(expected, context.extract_context_states(match, timeline))

    def test_future_mutation_cannot_change_prior_context(self):
        match, timeline = fixture()
        before = context.extract_context_states(match, timeline)[0]
        timeline["info"]["frames"][4]["participantFrames"]["1"]["totalGold"] = 999999
        timeline["info"]["frames"][3]["events"].append({"type": "CHAMPION_KILL", "timestamp": 180001,
                                                       "killerId": 1, "victimId": 6})
        self.assertEqual(before, context.extract_context_states(match, timeline)[0])

    def test_incomplete_frames_and_terminal_container_match_base(self):
        for case in ("incomplete", "terminal"):
            match, timeline = fixture()
            if case == "incomplete":
                del timeline["info"]["frames"][3]["participantFrames"]["10"]
            else:
                timeline["info"]["frames"][4]["events"] = [{"type": "GAME_END", "timestamp": 290000}]
            expected = extract_game_states(match, timeline)
            actual = context.extract_context_states(protect(match), protect(timeline))
            self.assertEqual([row["timestamp_ms"] for row in actual], [row["timestamp_ms"] for row in expected])
            self.assertEqual([row["x"] for row in actual], [row["x"] for row in expected])

    def test_duplicate_frame_identity_and_missing_terminal_rejected(self):
        for case in ("duplicate", "identity", "terminal"):
            match, timeline = fixture()
            if case == "duplicate":
                timeline["info"]["frames"].append(copy.deepcopy(timeline["info"]["frames"][3]))
            elif case == "identity":
                timeline["metadata"]["matchId"] = "OTHER"
            else:
                timeline["info"]["frames"][-1]["events"] = []
            with self.assertRaises(ValueError):
                context.extract_context_states(match, timeline)

    def test_npz_loader_never_requests_y(self):
        requests = []
        class Archive:
            files = list(context.ARRAY_KEYS) + ["y"]
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def __getitem__(self, key):
                requests.append(key)
                if key == "y":
                    raise AssertionError("Outcome array accessed")
                return np.array([1])
        with mock.patch.object(context.np, "load", return_value=Archive()):
            arrays = context.load_context_arrays("synthetic.npz")
        self.assertEqual(requests, list(context.ARRAY_KEYS))
        self.assertEqual(set(arrays), set(context.ARRAY_KEYS))
        self.assertNotIn("y", arrays)


class ContextAdmissionTests(unittest.TestCase):
    def select(self, manifest, source, forbidden=()):
        return context.select_population(manifest, source, forbidden,
            expected_roles={"fit": 1, "calibration": 1, "development": 2}, expected_reuse=3, expected_new_raw=1)

    def test_original_fit_calibration_and_all_new_patch_development_only(self):
        manifest, source = small_population()
        selected = self.select(manifest, source, {"EUW1_6"})
        self.assertEqual([row["match_id"] for row in selected], ["EUW1_1", "EUW1_2", "EUW1_4", "EUW1_5"])
        self.assertNotIn("EUW1_3", {row["match_id"] for row in selected})  # old-patch dev removed
        self.assertNotIn("EUW1_6", {row["match_id"] for row in selected})  # own confirmation still sealed

    def test_any_selected_external_or_test_id_is_rejected(self):
        manifest, source = small_population()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            self.select(manifest, source, {"EUW1_5"})

    def test_role_drift_and_count_drift_are_rejected(self):
        manifest, source = small_population()
        source["games"][0]["source_role"] = "calibration"
        with self.assertRaisesRegex(ValueError, "role"):
            self.select(manifest, source)
        manifest, source = small_population()
        manifest["games"][4]["patch"] = "16.18"
        with self.assertRaisesRegex(ValueError, "counts"):
            self.select(manifest, source)

    def test_forbidden_raw_id_fails_before_path_or_bytes_access(self):
        entry = {"match_id": "EUW1_5", "role": "development", "patch": "16.19"}
        ledger = []
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Raw bytes opened")):
            with self.assertRaisesRegex(ValueError, "admission"):
                context.read_new_payload(Path("unused"), "match", entry, Path("unused"), {"EUW1_5"}, {"EUW1_5"}, ledger.append)
        self.assertEqual(ledger, [])

    def test_new_raw_access_is_metadata_bound_and_digest_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            path = archive / "match" / "EUW1_5.json.gz"
            path.parent.mkdir()
            payload = {"metadata": {"matchId": "EUW1_5"}, "info": {"teams": [{"win": True}]}}
            raw = gzip.compress(json.dumps({"schema": 1, "endpoint": "match", "match_id": "EUW1_5", "payload": payload}).encode())
            path.write_bytes(raw)
            entry = {"match_id": "EUW1_5", "role": "development", "patch": "16.19"}
            ledger = []
            loaded, binding = context.read_new_payload(path, "match", entry, archive, {"EUW1_5"}, set(), ledger.append)
            self.assertEqual(loaded["metadata"]["matchId"], "EUW1_5")
            self.assertEqual(binding["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual([event["event"] for event in ledger], ["read_started", "read_complete"])
            self.assertEqual(binding["bytes"], len(raw))

    def test_small_pipeline_publishes_no_y_and_never_reopens_reused_raw(self):
        """Synthetic I/O smoke; real reservation auditing is tested separately."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source" / "raw_match_v5"
            base_dir = root / "base"
            base_dir.mkdir()
            manifest, old_source = small_population()
            manifest["source_data_root"] = str(archive.parent)
            manifest_path = root / "admission_manifest.json"

            def write_signed(path, value):
                path.write_text(json.dumps(value), encoding="utf-8")
                path.with_suffix(".sha256").write_text(context.digest(path) + "  " + path.name + "\n", encoding="ascii")

            write_signed(manifest_path, manifest)
            blocks = {key: [] for key in context.ARRAY_KEYS if key != "feature_names"}
            bindings = []
            for gid, entry in enumerate(old_source["games"]):
                metadata = manifest["games"][gid]
                match, timeline = fixture()
                match["metadata"]["matchId"] = timeline["metadata"]["matchId"] = entry["match_id"]
                match["info"]["gameVersion"] = metadata["patch"] + ".1"
                rows = context.extract_context_states(match, timeline)
                count = len(rows)
                entry.update(patch=metadata["patch"], game_creation_ms=metadata["game_creation_ms"], rows=count)
                values = np.asarray([[r["x"][name] for name in context.features.ALL_FEATURES] for r in rows], dtype=np.float32)
                part = {"x": values, "game": np.full(count, gid, dtype=np.int32),
                        "role": np.full(count, entry["role"], dtype="U11"),
                        "timestamp_ms": np.asarray([r["timestamp_ms"] for r in rows]),
                        "start_ms": np.asarray([r["game_start_ms"] for r in rows]),
                        "split_time_ms": np.full(count, entry["game_creation_ms"]),
                        "patch": np.full(count, entry["patch"], dtype="U16")}
                for key, value in part.items():
                    blocks[key].append(value)
                # These paths intentionally do not exist: reuse must not reopen them.
                bindings.append({"match_id": entry["match_id"], **{endpoint: {
                    "path": str(archive / endpoint / (entry["match_id"] + ".json.gz")), "sha256": "a" * 64, "bytes": 1}
                    for endpoint in ("match", "timeline")}})
            arrays = {key: np.concatenate(value) for key, value in blocks.items()}
            arrays["feature_names"] = np.asarray(context.features.ALL_FEATURES)
            base_path = base_dir / "states.npz"
            # Accessing this unused object-y with allow_pickle=False would fail.
            np.savez_compressed(base_path, **arrays, y=np.asarray([object()], dtype=object))
            old_source.update(compressed_sources=bindings, admission_manifest=str(manifest_path),
                              admission_manifest_sha256=context.digest(manifest_path), test_payload_reads=False,
                              confirmatory_payload_reads=False, historical_validation_payload_reads=False)
            source_path = base_dir / "source_manifest.json"
            source_path.write_text(json.dumps(old_source), encoding="utf-8")
            meta_path = base_dir / "cache_metadata.json"
            write_signed(meta_path, {"schema": "winprob-cache-v1", "state": "complete", "npz_path": str(base_path),
                "source_manifest_path": str(source_path), "npz_sha256": context.digest(base_path),
                "source_manifest_sha256": context.digest(source_path),
                "code_sha256": {name: context.digest(context.ROOT / "winprob" / name) for name in ("cache.py", "features.py", "admission.py")}})
            new = manifest["games"][4]
            match, timeline = fixture()
            match["metadata"]["matchId"] = timeline["metadata"]["matchId"] = new["match_id"]
            match["info"].update(queueId=420, gameCreation=new["game_creation_ms"])
            for endpoint, payload in (("match", match), ("timeline", timeline)):
                path = archive / endpoint / (new["match_id"] + ".json.gz")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(json.dumps({"schema": 1, "endpoint": endpoint,
                    "match_id": new["match_id"], "payload": payload}).encode()))
            args = Namespace(admission=manifest_path, base_cache=base_dir, output=root / "output", deadline_seconds=1200)
            with ExitStack() as stack:
                for name, value in (("EXPECTED_ROLES", {"fit": 1, "calibration": 1, "development": 2}),
                                    ("EXPECTED_REUSE", 3), ("EXPECTED_NEW_RAW", 1), ("EXPECTED_OVERLAPPING_DEVELOPMENT", 1)):
                    stack.enter_context(mock.patch.object(context, name, value))
                stack.enter_context(mock.patch.object(context, "reserved_ids", return_value=({"EUW1_6"}, {"confirmatory_reserved_ids": 1})))
                stack.enter_context(mock.patch.object(context, "verify_admitted_sources", return_value=(manifest_path, context.digest(manifest_path))))
                stack.enter_context(mock.patch.object(context, "set_below_normal_priority"))
                stack.enter_context(mock.patch.object(context, "check_resources"))
                result = context.run(args)
            self.assertEqual(result["schema"], "winprob-revision-context-v1")
            self.assertEqual(result["state"], "complete")
            self.assertEqual((result["cached_games"], result["reused_games"], result["new_raw_games"]), (4, 3, 1))
            self.assertFalse(result["contains_outcome_array"])
            with np.load(args.output / "states.npz", allow_pickle=False) as saved:
                self.assertNotIn("y", saved.files)
                self.assertEqual(set(saved.files), set(context.ARRAY_KEYS))
                np.testing.assert_array_equal(saved["x"][saved["game"] == 0], arrays["x"][arrays["game"] == 0])
            ledger = [json.loads(line) for line in (args.output / "access_ledger.jsonl").read_text().splitlines()]
            self.assertEqual({row["match_id"] for row in ledger}, {new["match_id"]})
            self.assertEqual(sum(row["event"] == "read_complete" for row in ledger), 2)


if __name__ == "__main__":
    unittest.main()
