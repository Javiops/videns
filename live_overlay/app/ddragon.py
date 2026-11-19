"""Catálogo de Data Dragon: coste total por objeto e id de imagen por campeón.

Reimplementado aquí a propósito (regla del proyecto: nada compartido con
../build_tracker ni ../macro_coach). Se descargan item.json y champion.json de
la versión más reciente a data/cache/; sin red se usa la versión cacheada más
nueva; sin red y sin caché no hay catálogo (None) y el tracker marca los
valores como aproximados y los iconos como desconocidos.

Por qué hace falta el coste de Data Dragon: el campo `price` del Live Client
API es el coste de combinación (base), no el total. Medido el 2026-09-29 sobre
981 objetos de 24 sesiones capturadas: 707 coinciden solo con `gold.base`, 274
con base == total (componentes) y ninguno solo con `gold.total`. Filo infinito
reporta 725 frente a un total de 3500.

Por qué hace falta el id de imagen: el API da `championName` (nombre visible,
"Wukong", "Nunu & Willump") y `rawChampionName`
("game_character_displayname_MonkeyKing"). El fichero de icono de Data Dragon
usa el id interno ("MonkeyKing.png"), que coincide con el sufijo del nombre
crudo salvo por mayúsculas ("FiddleSticks" frente a "Fiddlesticks").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from app.config import CACHE_DIR

VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DATA_URLS = {
    "item": "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json",
    "champion": "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json",
}
_CACHE_NAME = re.compile(r"^item-(?P<version>[0-9][0-9.]*)\.json$")
_RAW_PREFIX = "game_character_displayname_"


class Catalog:
    def __init__(self, version: str, items: dict, champions: dict | None = None):
        self.version = version
        self._totals: dict[int, int] = {}
        self._names: dict[int, str] = {}
        for key, data in (items or {}).items():
            try:
                item_id = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            gold = data.get("gold") or {}
            self._totals[item_id] = int(gold.get("total") or gold.get("base") or 0)
            self._names[item_id] = str(data.get("name") or "")
        self._champion_ids: dict[str, str] = {}      # id interno en minúsculas -> id de imagen
        self._champion_by_name: dict[str, str] = {}  # nombre visible en minúsculas -> id de imagen
        self._champion_names: dict[str, str] = {}    # id de imagen -> nombre visible
        self._champion_keys: dict[str, int] = {}     # id de imagen -> clave numérica de Riot (9 = Fiddlesticks)
        for champion_id, data in (champions or {}).items():
            if not isinstance(data, dict):
                continue
            image = str((data.get("image") or {}).get("full") or f"{champion_id}.png")
            image_id = image[:-4] if image.lower().endswith(".png") else image
            self._champion_ids[str(champion_id).lower()] = image_id
            display = str(data.get("name") or champion_id).strip()
            self._champion_names[image_id] = display
            try:
                self._champion_keys[image_id] = int(data.get("key"))
            except (TypeError, ValueError):
                pass
            name = display.lower()
            if name:
                self._champion_by_name[name] = image_id

    @classmethod
    def from_payloads(cls, items_payload: dict, champions_payload: dict | None = None) -> "Catalog":
        version = str(items_payload.get("version") or (champions_payload or {}).get("version") or "?")
        return cls(version, items_payload.get("data") or {}, (champions_payload or {}).get("data") or {})

    def total(self, item_id: int) -> int | None:
        """Coste total del objeto, o None si el id no está en esta versión."""
        return self._totals.get(item_id)

    def name(self, item_id: int) -> str:
        return self._names.get(item_id, "")

    def champion_image_id(self, raw_name: str | None, display_name: str | None = None) -> str | None:
        """Id de imagen de Data Dragon ("MonkeyKing") a partir del nombre crudo o visible; None si no se conoce."""
        raw = str(raw_name or "")
        if raw.startswith(_RAW_PREFIX):
            raw = raw[len(_RAW_PREFIX):]
        if raw:
            hit = self._champion_ids.get(raw.lower())
            if hit:
                return hit
        name = str(display_name or "").strip().lower()
        if name:
            hit = self._champion_by_name.get(name)
            if hit:
                return hit
            compact = "".join(ch for ch in name if ch.isalnum())
            hit = self._champion_ids.get(compact)
            if hit:
                return hit
        return None

    def champion_key(self, image_id: str) -> int | None:
        """Clave numérica de Riot del campeón (la que usan los assets del cliente), o None."""
        return self._champion_keys.get(image_id)

    def champion_list(self) -> list[dict]:
        """Todos los campeones del catálogo como {id de imagen, nombre visible}, por nombre."""
        return sorted(
            ({"id": image_id, "name": name} for image_id, name in self._champion_names.items()),
            key=lambda c: c["name"].lower(),
        )

    def __len__(self) -> int:
        return len(self._totals)

    @property
    def champions(self) -> int:
        return len(self._champion_ids)


def _version_key(version: str) -> tuple:
    return tuple(int(part) if part.isdigit() else -1 for part in version.split("."))


def latest_version(timeout: float = 5.0) -> str | None:
    try:
        response = httpx.get(VERSIONS_URL, timeout=timeout)
        response.raise_for_status()
        versions = response.json()
        return str(versions[0])
    except (httpx.HTTPError, ValueError, IndexError, TypeError):
        return None


def cached_versions(cache_dir: Path = CACHE_DIR) -> list[str]:
    """Versiones con item.json en caché, de más nueva a más vieja."""
    found = []
    if cache_dir.is_dir():
        for path in cache_dir.iterdir():
            match = _CACHE_NAME.match(path.name)
            if match:
                found.append(match.group("version"))
    return sorted(found, key=_version_key, reverse=True)


def _ensure_cached(kind: str, version: str, cache_dir: Path, timeout: float) -> Path | None:
    path = cache_dir / f"{kind}-{version}.json"
    if path.exists():
        return path
    try:
        response = httpx.get(DATA_URLS[kind].format(version=version), timeout=timeout * 4)
        response.raise_for_status()
        payload = json.loads(response.content)   # nunca cachear una descarga truncada o una página de login
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict) or not payload["data"]:
            return None
        # escritura atómica: un proceso matado a medias no deja medio fichero
        temp = path.with_suffix(".tmp")
        temp.write_bytes(response.content)
        temp.replace(path)
        return path
    except (httpx.HTTPError, ValueError, OSError):
        return None


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # una caché corrupta se descarta para que la próxima carga la vuelva a bajar
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return payload if isinstance(payload, dict) else None


def load_catalog(cache_dir: Path = CACHE_DIR, timeout: float = 5.0) -> Catalog | None:
    """Catálogo de la versión más reciente disponible; None sin red y sin caché utilizable. Nunca lanza."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    version = latest_version(timeout)
    candidates: list[str] = []
    if version and _ensure_cached("item", version, cache_dir, timeout) is not None:
        candidates.append(version)
    candidates += [v for v in cached_versions(cache_dir) if v not in candidates]
    for candidate in candidates:
        items = _read_json(cache_dir / f"item-{candidate}.json")
        if items is None:
            continue   # caché corrupta: ya borrada; probar la siguiente versión más nueva
        catalog = Catalog.from_payloads(items, _read_json(_ensure_cached("champion", candidate, cache_dir, timeout)))
        if len(catalog):
            return catalog
        try:
            (cache_dir / f"item-{candidate}.json").unlink()   # un item.json sin objetos no sirve de nada
        except OSError:
            pass
    return None
