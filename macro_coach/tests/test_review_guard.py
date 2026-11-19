"""La UI de revisión no puede proponer con un modelo de otra versión.

review_app rellenaba las feature_names del artefacto con `.get(n, -1)` en
silencio: tras el paso f2 -> f3 el modelo viejo siguió proponiendo con sus 11
features de oro a -1, top-3 de aspecto normal, y cada veredicto del owner lo
guardaba en el feedback. La guardia rechaza el artefacto en vez de adaptarlo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from app import dataset
from app.config import FEATURES_V, LEXICON_V
import review_app


def test_artefacto_de_otra_version_de_features_no_propone():
    meta = {"features_v": "f2", "lexicon_v": LEXICON_V}
    razon = review_app.incompatibilidad(meta, dataset.feature_names())
    assert razon is not None and "f2" in razon


def test_artefacto_con_features_retiradas_no_propone():
    # versión correcta en el meta pero nombres que ya no existen: también fuera
    meta = {"features_v": FEATURES_V, "lexicon_v": LEXICON_V}
    razon = review_app.incompatibilidad(
        meta, dataset.feature_names() + ["gold_diff_pf"])
    assert razon is not None and "gold_diff_pf" in razon


def test_artefacto_de_la_version_actual_si_propone():
    meta = {"features_v": FEATURES_V, "lexicon_v": LEXICON_V}
    assert review_app.incompatibilidad(meta, dataset.feature_names()) is None
