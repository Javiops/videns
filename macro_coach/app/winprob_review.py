"""Curvas OMNI importadas para revisión; nunca carga modelos ni datos hermanos.

Sólo lee el split local y la copia versionada de curves.jsonl/meta.json con sus
SHA256. Las referencias de procedencia son texto: nunca se siguen sus rutas.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from pathlib import Path

from .config import DATA, ML, EXCLUDED_MATCHES, FEATURES_V, LEXICON_V


IMPORT_DIRECTORY = DATA / "imports" / "winprob_curves_v1"
MATCH_ID = re.compile(r"^[A-Za-z0-9]+_[0-9]+$")
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
EXPOSURES = {"fit", "calibration", "development", "unseen"}


class ReviewSplitError(ValueError):
    pass


class ReviewArtifactError(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_review_allowlist(split_path: Path = ML / "split.json", excluded=None):
    """Valida únicamente metadatos; no llama a dataset.corpus_games()."""
    excluded = EXCLUDED_MATCHES if excluded is None else excluded
    try:
        raw = Path(split_path).read_bytes()
        split = json.loads(raw.decode("utf-8-sig"))
        if (not isinstance(split, dict) or split.get("provisional")
                or split.get("features_v") != FEATURES_V or split.get("lexicon_v") != LEXICON_V):
            raise ReviewSplitError("Split de revisión ausente, provisional o de otra versión.")
        folds = {}
        for name in ("train", "val", "test"):
            ids = split.get(name)
            if not isinstance(ids, list) or any(not isinstance(mid, str) or not MATCH_ID.fullmatch(mid) for mid in ids):
                raise ReviewSplitError("Identidades de partida inválidas en el split de revisión.")
            folds[name] = ids
        all_ids = [mid for ids in folds.values() for mid in ids]
        if len(all_ids) != len(set(all_ids)) or type(split.get("n_games")) is not int or split["n_games"] != len(all_ids):
            raise ReviewSplitError("Split de revisión con duplicados o recuentos incompatibles.")
        allowed = (set(folds["train"]) | set(folds["val"])) - set(folds["test"]) - set(excluded)
        return allowed, _sha(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewSplitError("No se puede verificar el split local de revisión.") from error


def preceding_point(points, cutoff_ms, team_id):
    """Último punto <= corte; nunca interpola ni toma el vecino futuro."""
    if team_id not in (100, 200) or not _number(cutoff_ms) or cutoff_ms < 0:
        return None
    index = bisect.bisect_right([point["timestamp_ms"] for point in points], cutoff_ms) - 1
    if index < 0:
        return None
    point = points[index]
    age = float(cutoff_ms - point["timestamp_ms"])
    return {"timestamp_ms": point["timestamp_ms"], "cutoff_ms": float(cutoff_ms),
            "p_blue": point["p_blue"], "p_team": point["p_blue"] if team_id == 100 else 1 - point["p_blue"],
            "team_id": team_id, "age_ms": age, "stale": age > 90000}


def largest_nonoverlapping_falls(windows, team_id, limit=3):
    """Descensos estimados retrospectivos; no etiqueta errores ni efectos."""
    if team_id not in (100, 200):
        return []
    candidates = []
    for window in windows:
        delta = window["delta_blue_pp"] if team_id == 100 else -window["delta_blue_pp"]
        if delta < 0:
            candidates.append({**window, "delta_team_pp": delta,
                               "duration_ms": window["end_ms"] - window["start_ms"]})
    chosen = []
    for candidate in sorted(candidates, key=lambda item: (item["delta_team_pp"], item["start_ms"])):
        if not any(candidate["start_ms"] < kept["end_ms"] and candidate["end_ms"] > kept["start_ms"] for kept in chosen):
            chosen.append(candidate)
        if len(chosen) == limit:
            break
    return chosen


class CurveStore:
    def __init__(self, directory=IMPORT_DIRECTORY, split_path=ML / "split.json", excluded=None):
        self.directory = Path(directory)
        self.split_path = Path(split_path)
        self.excluded = excluded

    def _verified_bytes(self, name):
        path = self.directory / name
        if self.directory.is_symlink() or path.is_symlink() or path.with_suffix(".sha256").is_symlink():
            raise ReviewArtifactError("El artefacto debe ser una copia local, no un enlace.")
        try:
            raw = path.read_bytes()
            tokens = path.with_suffix(".sha256").read_text(encoding="ascii").split()
        except (OSError, UnicodeError) as error:
            raise ReviewArtifactError("Curva no disponible: falta la copia local o su SHA256.") from error
        actual = _sha(raw)
        if len(tokens) != 2 or tokens[0].lower() != actual or tokens[1] != name:
            raise ReviewArtifactError("Curva no disponible: la firma SHA256 no coincide.")
        return raw, actual

    def load(self):
        allowed, split_hash = read_review_allowlist(self.split_path, self.excluded)
        raw_meta, meta_hash = self._verified_bytes("meta.json")
        try:
            meta = json.loads(raw_meta.decode("utf-8-sig"))
            if meta.get("schema") != "winprob-macro-curves-v1" or meta.get("macro_split_sha256") != split_hash:
                raise ReviewArtifactError("La curva pertenece a otro esquema o split local.")
            requested = meta.get("requested_game_ids")
            if (not isinstance(requested, list) or any(not isinstance(mid, str) or not MATCH_ID.fullmatch(mid) for mid in requested)
                    or len(requested) != len(set(requested)) or not set(requested) <= allowed):
                raise ReviewArtifactError("El artefacto solicita partidas selladas o ajenas al split permitido.")
            digest = meta.get("model_digest")
            validation = meta.get("model_validation")
            if (not isinstance(digest, str) or not SHA256.fullmatch(digest) or not isinstance(validation, dict)
                    or not all(isinstance(validation.get(key), str) and validation[key] for key in ("state", "description", "report_path"))
                    or not isinstance(validation.get("report_sha256"), str) or not SHA256.fullmatch(validation["report_sha256"])):
                raise ReviewArtifactError("La procedencia o validación del modelo está incompleta.")
            # Validate IDs before opening any per-game curve, even in the copied artifact.
            raw_curves, curves_hash = self._verified_bytes("curves.jsonl")
            if curves_hash != meta.get("curves_sha256"):
                raise ReviewArtifactError("El índice del artefacto no corresponde a estas curvas.")
            rows = {}
            for line in raw_curves.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                mid = row.get("match_id")
                if row.get("schema") != "winprob-macro-curve-v1" or mid not in requested or mid in rows:
                    raise ReviewArtifactError("Curva duplicada, ajena al índice o de otro esquema.")
                if row.get("model_digest") != digest or row.get("exposure_role") not in EXPOSURES:
                    raise ReviewArtifactError("Modelo o exposición de la partida incompatibles.")
                points, windows = row.get("points"), row.get("observed_windows")
                if not isinstance(points, list) or not isinstance(windows, list):
                    raise ReviewArtifactError("Puntos o ventanas de la curva inválidos.")
                if row.get("status") == "unavailable":
                    if points or windows or not isinstance(row.get("unavailable_reason"), str) or not row["unavailable_reason"]:
                        raise ReviewArtifactError("Una curva no disponible debe declarar el motivo y no incluir puntos.")
                elif row.get("status") == "available":
                    if not points:
                        raise ReviewArtifactError("Curva disponible sin puntos.")
                    by_time = {}
                    last = -1
                    for point in points:
                        timestamp, probability = point.get("timestamp_ms"), point.get("p_blue")
                        if (type(timestamp) is not int or timestamp <= last or not 180000 <= timestamp <= 2400000
                                or not _number(probability) or not 0 <= probability <= 1):
                            raise ReviewArtifactError("La curva contiene tiempos o probabilidades inválidos.")
                        by_time[timestamp] = float(probability)
                        last = timestamp
                    seen_windows = set()
                    for window in windows:
                        start, end, delta = window.get("start_ms"), window.get("end_ms"), window.get("delta_blue_pp")
                        if (type(start) is not int or type(end) is not int or start not in by_time or end not in by_time
                                or start >= end or end // 60000 - start // 60000 != 3
                                or not _number(delta) or (start, end) in seen_windows
                                or not math.isclose(delta, 100 * (by_time[end] - by_time[start]), abs_tol=1e-6)):
                            raise ReviewArtifactError("La ventana no corresponde a dos puntos observados separados por tres minutos nominales.")
                        seen_windows.add((start, end))
                else:
                    raise ReviewArtifactError("Estado de curva desconocido.")
                rows[mid] = row
            if set(rows) != set(requested):
                raise ReviewArtifactError("El artefacto omite alguna partida solicitada.")
            return {"meta": meta, "meta_sha256": meta_hash, "curves_sha256": curves_hash, "rows": rows}
        except (json.JSONDecodeError, UnicodeError, TypeError, AttributeError) as error:
            raise ReviewArtifactError("No se puede interpretar el artefacto local de curvas.") from error

    def game(self, match_id):
        bundle = self.load()
        row = bundle["rows"].get(match_id)
        if row is None:
            return {"status": "unavailable", "reason": "Partida no incluida en el artefacto de curvas."}
        if row["status"] == "unavailable":
            return {"status": "unavailable", "reason": row["unavailable_reason"], "exposure_role": row["exposure_role"]}
        return {"status": "available", "match_id": match_id, "points": row["points"],
                "observed_windows": row["observed_windows"], "model_digest": row["model_digest"],
                "exposure_role": row["exposure_role"], "artifact_sha256": bundle["curves_sha256"],
                "model_validation": bundle["meta"]["model_validation"],
                "falls_by_team": {str(team): largest_nonoverlapping_falls(row["observed_windows"], team) for team in (100, 200)}}
