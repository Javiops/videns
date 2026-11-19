"""Cache isolation tests using synthetic envelopes and split metadata only."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from winprob import admission, cache, features


def payloads(mid, created):
    match = {"metadata": {"matchId": mid}, "info": {
        "queueId": 420, "gameCreation": created, "gameStartTimestamp": created + 9000,
        "gameVersion": "16.18.1", "participants": [
            {"participantId": pid, "teamId": 100 if pid <= 5 else 200} for pid in range(1, 11)],
        "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}]}}
    frames = []
    for minute in range(6):
        participants = {str(pid): {"participantId": pid, "totalGold": 500 + minute * 100 + pid,
            "xp": minute * 100, "level": minute + 1, "minionsKilled": minute * 5,
            "jungleMinionsKilled": 0, "damageStats": {"totalDamageDoneToChampions": minute * 20,
                                                        "totalDamageTaken": minute * 20}}
                        for pid in range(1, 11)}
        frames.append({"timestamp": minute * 60000, "participantFrames": participants,
                       "events": [{"type": "GAME_END", "timestamp": minute * 60000}] if minute == 5 else []})
    timeline = {"metadata": {"matchId": mid}, "info": {"frameInterval": 60000, "frames": frames}}
    return match, timeline


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = self.root / "data"
        split = data / "dataset_generations" / "g" / "split_assignment.json"
        split.parent.mkdir(parents=True)
        split.write_text(json.dumps({"EUW1_1": "train", "EUW1_2": "train", "EUW1_3": "val",
                                     "EUW1_4": "train", "EUW1_5": "train"}), encoding="utf-8")
        manifest = admission.prepare_manifest(data, [split], "synthetic-cache-v1")
        roles = {"EUW1_1": "train", "EUW1_2": "calibration", "EUW1_3": "development_exposed",
                 "EUW1_4": "confirmatory_reserved", "EUW1_5": "development"}
        manifest["state"] = "temporal_manifest_ready"
        for index, entry in enumerate(manifest["games"]):
            entry.update({"role": roles[entry["match_id"]], "game_creation_ms": 1000000 + index * 100000,
                          "patch": "16.18", "queue_id": 420})
        self.manifest, _ = admission.write_manifest(manifest, self.root / "artifacts")
        self.entries = manifest["games"]
        self.data = data
        for entry in self.entries:
            if entry["role"] in cache.ROLE_MAP:
                match, timeline = payloads(entry["match_id"], entry["game_creation_ms"])
                self.archive("match", entry["match_id"], match)
                self.archive("timeline", entry["match_id"], timeline)
        self.output = self.root / "cache"

    def archive(self, endpoint, mid, payload, **overrides):
        path = self.data / "raw_match_v5" / endpoint / (mid + ".json.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"schema": 1, "endpoint": endpoint, "match_id": mid, "payload": payload, **overrides}
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(envelope, handle)
        return path

    def test_export_types_mapping_digests_and_no_reserved_reads(self):
        opened = []
        original = cache._read_payload
        def observed(path, endpoint, mid):
            opened.append(mid)
            return original(path, endpoint, mid)
        callbacks = []
        with patch.object(cache, "_read_payload", side_effect=observed):
            result = cache.build_cache(self.manifest, self.output, resource_check=lambda: callbacks.append(True))
        self.assertEqual(set(opened), {"EUW1_1", "EUW1_2", "EUW1_5"})
        self.assertEqual(result["cached_games"], 3)
        self.assertEqual(result["rows"], 6)
        self.assertEqual(len(callbacks), 2)
        with np.load(result["npz_path"], allow_pickle=False) as arrays:
            self.assertEqual(arrays["x"].shape, (6, len(features.ALL_FEATURES)))
            self.assertEqual(arrays["x"].dtype, np.float32)
            self.assertEqual(arrays["y"].dtype, np.int8)
            self.assertEqual(arrays["game"].dtype, np.int32)
            self.assertEqual(arrays["timestamp_ms"].dtype, np.int64)
            self.assertEqual(arrays["role"].dtype.kind, "U")
            self.assertEqual(set(arrays["role"]), {"fit", "calibration", "development"})
            self.assertTrue(np.all(arrays["start_ms"] == arrays["split_time_ms"] + 9000))
        self.assertEqual(result["npz_sha256"], cache._sha_file(Path(result["npz_path"])))
        self.assertEqual(json.loads((self.output / "status.json").read_text())["state"], "complete")

    def test_manifest_sidecar_mismatch_rejected_before_directory_or_reads(self):
        self.manifest.write_bytes(self.manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "sidecar mismatch"):
            cache.build_cache(self.manifest, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_envelope_is_fatal(self):
        entry = self.entries[0]
        match, _ = payloads(entry["match_id"], entry["game_creation_ms"])
        self.archive("match", entry["match_id"], match, schema=2)
        with self.assertRaisesRegex(ValueError, "Invalid archive envelope"):
            cache.build_cache(self.manifest, self.output)
        self.assertEqual(json.loads((self.output / "status.json").read_text())["state"], "failed")
        self.assertFalse((self.output / "cache_metadata.json").exists())

    def test_creation_and_queue_mismatches_are_fatal(self):
        entry = self.entries[0]
        match, _ = payloads(entry["match_id"], entry["game_creation_ms"] + 1)
        self.archive("match", entry["match_id"], match)
        with self.assertRaisesRegex(ValueError, "gameCreation mismatch"):
            cache.build_cache(self.manifest, self.output)
        match["info"]["gameCreation"] -= 1
        match["info"]["queueId"] = 450
        self.archive("match", entry["match_id"], match)
        with self.assertRaisesRegex(ValueError, "ranked queue"):
            cache.build_cache(self.manifest, self.root / "cache2")

    def test_missing_archive_and_zero_rows_record_exclusions(self):
        (self.data / "raw_match_v5" / "timeline" / "EUW1_2.json.gz").unlink()
        original = features.extract_game_states
        def no_first(match, timeline):
            return [] if match["metadata"]["matchId"] == "EUW1_1" else original(match, timeline)
        with patch.object(features, "extract_game_states", side_effect=no_first):
            result = cache.build_cache(self.manifest, self.output)
        self.assertEqual(result["exclusion_counts"], {"missing_archive": 1, "zero_valid_states": 1})
        self.assertEqual(result["cached_games"], 1)
        with np.load(result["npz_path"]) as arrays:
            self.assertEqual(set(arrays["game"]), {0})

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            cache.build_cache(self.manifest, self.output)

    def test_resource_failure_before_any_payload_read(self):
        def unavailable():
            raise RuntimeError("resource gate")
        with patch.object(cache, "_read_payload") as reader:
            with self.assertRaisesRegex(RuntimeError, "resource gate"):
                cache.build_cache(self.manifest, self.output, resource_check=unavailable)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
