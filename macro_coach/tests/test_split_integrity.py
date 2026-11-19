"""El split congelado no puede contener partidas excluidas ni versiones
viejas. Falla con el código anterior: `frozen_split` devolvía split.json tal
cual y el export metió 100 filas de EUW1_7983109023 (partida propia) en train.
"""
import json

import pytest

from app import dataset
from app.config import EXCLUDED_MATCHES, FEATURES_V, LEXICON_V

EXCLUDED = sorted(EXCLUDED_MATCHES)[0]


def test_excluded_game_is_rejected_by_validation():
    split = {"train": [EXCLUDED, "EUW1_OK"], "val": [], "test": [],
             "lexicon_v": LEXICON_V, "features_v": FEATURES_V}
    reasons = dataset.validate_split(split, corpus_ids={EXCLUDED, "EUW1_OK"})
    assert any("EXCLUID" in r.upper() for r in reasons), reasons


def test_stale_feature_version_is_rejected():
    split = {"train": ["EUW1_OK"], "val": [], "test": [],
             "lexicon_v": LEXICON_V, "features_v": "f_viejo"}
    reasons = dataset.validate_split(split, corpus_ids={"EUW1_OK"})
    assert any("features" in r for r in reasons), reasons


def test_ids_outside_corpus_are_rejected():
    split = {"train": ["EUW1_FANTASMA"], "val": [], "test": [],
             "lexicon_v": LEXICON_V, "features_v": FEATURES_V}
    reasons = dataset.validate_split(split, corpus_ids={"EUW1_OK"})
    assert any("corpus" in r for r in reasons), reasons


def test_duplicate_game_across_partitions_is_rejected():
    split = {"train": ["EUW1_OK"], "val": ["EUW1_OK"], "test": [],
             "lexicon_v": LEXICON_V, "features_v": FEATURES_V}
    reasons = dataset.validate_split(split, corpus_ids={"EUW1_OK"})
    assert any("repetid" in r for r in reasons), reasons


def test_clean_split_passes():
    split = {"train": ["EUW1_A"], "val": ["EUW1_B"], "test": ["EUW1_C"],
             "lexicon_v": LEXICON_V, "features_v": FEATURES_V}
    assert dataset.validate_split(
        split, corpus_ids={"EUW1_A", "EUW1_B", "EUW1_C"}) == []


def test_export_refuses_excluded_game():
    split = {"train": [EXCLUDED], "val": [], "test": [],
             "lexicon_v": LEXICON_V, "features_v": FEATURES_V}
    with pytest.raises(dataset.SplitError):
        dataset.export(split)


def test_corpus_games_never_lists_excluded():
    assert all(mid != EXCLUDED for mid, _ in dataset.corpus_games())


def test_frozen_split_on_disk_is_clean_or_rejected(tmp_path, monkeypatch):
    """Un split.json contaminado en disco debe dar SplitError, no colarse."""
    ml = tmp_path / "ml"
    ml.mkdir()
    monkeypatch.setattr(dataset, "ML", ml)
    (ml / "split.json").write_text(json.dumps(
        {"train": [EXCLUDED], "val": [], "test": [],
         "lexicon_v": LEXICON_V, "features_v": FEATURES_V}), encoding="utf-8")
    with pytest.raises(dataset.SplitError):
        dataset.frozen_split()
