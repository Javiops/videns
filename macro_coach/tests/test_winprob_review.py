"""Artefactos sintéticos y rutas aisladas; no lee ningún corpus real."""

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_app
from app.config import FEATURES_V, LEXICON_V
from app.winprob_review import (CurveStore, ReviewArtifactError, ReviewSplitError,
                               largest_nonoverlapping_falls, preceding_point, read_review_allowlist)

TRAIN, VAL, SEALED = "EUW1_100", "EUW1_200", "EUW1_300"
MODEL_DIGEST = "a" * 64


def signed_file(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "  " + path.name + "\n", encoding="ascii")
    return digest


def row(mid):
    points = [{"timestamp_ms": minute * 60000, "p_blue": p}
              for minute, p in ((3, .8), (4, .7), (6, .6), (7, .5), (9, .4), (12, .45))]
    return {"schema": "winprob-macro-curve-v1", "match_id": mid, "status": "available",
            "model_digest": MODEL_DIGEST, "exposure_role": "unseen", "unavailable_reason": None,
            "points": points, "observed_windows": [
                {"start_ms": 180000, "end_ms": 360000, "delta_blue_pp": -20.0},
                {"start_ms": 240000, "end_ms": 420000, "delta_blue_pp": -20.0},
                {"start_ms": 360000, "end_ms": 540000, "delta_blue_pp": -20.0},
                {"start_ms": 540000, "end_ms": 720000, "delta_blue_pp": 5.0},
            ]}


@pytest.fixture
def setup(tmp_path):
    split_path = tmp_path / "ml" / "split.json"
    split_path.parent.mkdir()
    split_path.write_text(json.dumps({"train": [TRAIN], "val": [VAL], "test": [SEALED],
                                     "n_games": 3, "features_v": FEATURES_V, "lexicon_v": LEXICON_V}), encoding="utf-8")
    directory = tmp_path / "imports" / "winprob_curves_v1"
    rows = [row(TRAIN), row(VAL)]
    meta = {"schema": "winprob-macro-curves-v1", "requested_game_ids": [TRAIN, VAL],
            "macro_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "model_digest": MODEL_DIGEST, "generated_utc": "2026-09-30T00:00:00Z",
            "model_validation": {"state": "validated_original_snapshot_development",
                                 "description": "Validación sintética del formato, no del modelo.",
                                 "report_path": str(tmp_path / "DO_NOT_OPEN_sibling_report.json"),
                                 "report_sha256": "b" * 64}}
    def write(rows_value=None, meta_value=None):
        values = rows if rows_value is None else rows_value
        metadata = meta if meta_value is None else meta_value
        raw = ("\n".join(json.dumps(value) for value in values) + "\n").encode("utf-8")
        metadata["curves_sha256"] = signed_file(directory / "curves.jsonl", raw)
        signed_file(directory / "meta.json", json.dumps(metadata).encode("utf-8"))
    write()
    return {"root": tmp_path, "split": split_path, "directory": directory,
            "rows": rows, "meta": meta, "write": write, "store": CurveStore(directory, split_path, excluded=set())}


def test_only_local_train_and_val_are_allowed(setup):
    allowed, _ = read_review_allowlist(setup["split"], excluded={VAL})
    assert allowed == {TRAIN}
    assert SEALED not in allowed


def test_duplicate_or_stale_split_fails_without_corpus_scan(setup):
    split = json.loads(setup["split"].read_text())
    split["test"] = [TRAIN]
    setup["split"].write_text(json.dumps(split))
    with pytest.raises(ReviewSplitError):
        read_review_allowlist(setup["split"])


def test_validated_curve_and_preceding_point_never_use_future(setup):
    curve = setup["store"].game(TRAIN)
    marker = preceding_point(curve["points"], 244000, 200)
    assert marker["timestamp_ms"] == 240000
    assert marker["age_ms"] == 4000
    assert marker["p_blue"] == .7
    assert marker["p_team"] == pytest.approx(.3)
    assert preceding_point(curve["points"], 179999, 100) is None
    assert preceding_point(curve["points"], float("nan"), 100) is None
    assert preceding_point(curve["points"], 1000000, 100)["stale"] is True


def test_descents_are_team_relative_nonoverlapping_and_not_action_labels(setup):
    windows = setup["rows"][0]["observed_windows"]
    blue = largest_nonoverlapping_falls(windows, 100)
    assert [(w["start_ms"], w["end_ms"]) for w in blue] == [(180000, 360000), (360000, 540000)]
    red = largest_nonoverlapping_falls(windows, 200)
    assert len(red) == 1 and red[0]["start_ms"] == 540000 and red[0]["delta_team_pp"] == -5
    assert not any("mistake" in key or "action" in key for window in blue for key in window)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_probabilities_rejected(setup, value):
    setup["rows"][0]["points"][0]["p_blue"] = value
    setup["write"]()
    with pytest.raises(ReviewArtifactError):
        setup["store"].load()


def test_unsorted_points_and_unmatched_windows_rejected(setup):
    points = setup["rows"][0]["points"]
    points[0], points[1] = points[1], points[0]
    setup["write"]()
    with pytest.raises(ReviewArtifactError):
        setup["store"].load()
    points[0], points[1] = points[1], points[0]
    setup["rows"][0]["observed_windows"][0]["delta_blue_pp"] = -25
    setup["write"]()
    with pytest.raises(ReviewArtifactError):
        setup["store"].load()


def test_digest_tampering_and_duplicate_rows_fail_closed(setup):
    with (setup["directory"] / "curves.jsonl").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ReviewArtifactError, match="SHA256"):
        setup["store"].load()
    setup["write"](setup["rows"] + [setup["rows"][0]])
    with pytest.raises(ReviewArtifactError):
        setup["store"].load()


def test_sealed_request_is_rejected_before_curve_bytes_open(setup, monkeypatch):
    setup["meta"]["requested_game_ids"].append(SEALED)
    setup["write"]()
    original = setup["store"]._verified_bytes
    def guarded(name):
        assert name != "curves.jsonl", "Sealed request must be rejected from metadata first"
        return original(name)
    monkeypatch.setattr(setup["store"], "_verified_bytes", guarded)
    with pytest.raises(ReviewArtifactError, match="selladas"):
        setup["store"].load()


def test_missing_and_unavailable_are_explicit(setup):
    setup["rows"][0].update(status="unavailable", points=[], observed_windows=[], unavailable_reason="No hay fuente admitida.")
    setup["write"]()
    assert setup["store"].game(TRAIN)["reason"] == "No hay fuente admitida."
    (setup["directory"] / "meta.sha256").unlink()
    with pytest.raises(ReviewArtifactError):
        setup["store"].load()


def test_validation_references_are_never_opened(setup, monkeypatch):
    original = io.open
    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            assert "DO_NOT_OPEN" not in str(file)
        return original(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", guarded)
    assert setup["store"].load()["meta"]["model_validation"]["report_sha256"] == "b" * 64


def decision():
    return {"pid": 1, "champ": "Synthetic", "t0": 245., "t1": 260., "token": "FARM",
            "snapshot": {"t": 244., "me": {"team": 200}, "minimap": {},
                         "vision": {"available": False}, "players": [], "objectives": {}, "camps": {}}}


@pytest.fixture
def client(setup, monkeypatch):
    corpus = setup["root"] / "corpus"
    for mid in (TRAIN, VAL, SEALED):
        directory = corpus / mid
        directory.mkdir(parents=True)
        (directory / "decisions.jsonl").write_text(json.dumps(decision()) + "\n", encoding="utf-8")
    monkeypatch.setattr(review_app, "CORPUS", corpus)
    monkeypatch.setattr(review_app, "ML", setup["split"].parent)
    monkeypatch.setattr(review_app, "WINPROB_IMPORT", setup["directory"])
    monkeypatch.setattr(review_app, "model_top3", lambda row: None)
    monkeypatch.setattr(review_app, "oro_en", lambda mid, timestamp: None)
    monkeypatch.setattr(review_app, "buys_de", lambda *args: [])
    # Guard both pathlib and builtin-open code paths against synthetic sealed files.
    original_io = io.open
    original_builtin = open
    def check(file):
        if isinstance(file, (str, Path)):
            assert not Path(file).resolve().is_relative_to((corpus / SEALED).resolve()), "Sealed corpus payload opened"
    def guarded_io(file, *args, **kwargs):
        check(file)
        return original_io(file, *args, **kwargs)
    def guarded_builtin(file, *args, **kwargs):
        check(file)
        return original_builtin(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", guarded_io)
    monkeypatch.setattr("builtins.open", guarded_builtin)
    return TestClient(review_app.app)


def test_all_review_routes_deny_sealed_games_before_opening_payloads(client):
    home = client.get("/")
    assert home.status_code == 200 and TRAIN in home.text and VAL in home.text and SEALED not in home.text
    for path in (f"/game/{SEALED}/1", f"/api/decisions/{SEALED}/1", f"/img/{SEALED}/pass_omni/mm_1.png",
                 f"/api/winprob/{SEALED}"):
        assert client.get(path).status_code == 404
    assert client.post("/api/feedback", json={"match_id": SEALED}).status_code == 404


def test_allowed_decisions_receive_past_marker_and_model_state_link(client):
    response = client.get(f"/api/decisions/{TRAIN}/1")
    assert response.status_code == 200
    item = response.json()[0]
    assert item["snapshot_t"] == 244 and item["pov_team"] == 200
    assert item["winprob_marker"]["timestamp_ms"] == 240000
    assert item["winprob_marker"]["p_team"] == pytest.approx(.3)
    assert client.get(f"/api/winprob/{TRAIN}").json()["status"] == "available"
    page = client.get(f"/game/{TRAIN}/1").text
    assert "Descenso estimado" in page and "/winprob/model" in page
    assert client.get("/winprob/model").status_code == 200


def test_missing_curve_preserves_existing_review_and_is_explicit(client, setup):
    (setup["directory"] / "meta.sha256").unlink()
    response = client.get(f"/api/decisions/{TRAIN}/1")
    assert response.status_code == 200 and response.json()[0]["winprob_marker"] is None
    assert client.get(f"/api/winprob/{TRAIN}").json()["status"] == "unavailable"


def test_invalid_split_stops_index_before_any_corpus_scan(client, setup):
    setup["split"].write_text("{}", encoding="utf-8")
    assert client.get("/").status_code == 503
    assert client.get(f"/api/decisions/{TRAIN}/1").status_code == 503
