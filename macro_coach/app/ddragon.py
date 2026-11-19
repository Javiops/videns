"""Data Dragon: precios de ítems y retratos de campeones. Sin clave de API.
Todo cacheado bajo data/cache/ddragon/<version>/."""
import json
from pathlib import Path

import httpx

from .config import CACHE

BASE = "https://ddragon.leagueoflegends.com"


def _dir(version: str) -> Path:
    d = CACHE / "ddragon" / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def versions() -> list:
    p = CACHE / "ddragon" / "versions.json"
    if p.exists():
        return json.loads(p.read_text())
    data = httpx.get(f"{BASE}/api/versions.json", timeout=20).json()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return data


def version_for_patch(patch: str) -> str:
    """'16.18' -> la versión ddragon que empieza por '16.18.'"""
    for v in versions():
        if v.startswith(patch + "."):
            return v
    raise ValueError(f"sin versión ddragon para el parche {patch}")


def items(version: str) -> dict:
    p = _dir(version) / "item.json"
    if not p.exists():
        data = httpx.get(f"{BASE}/cdn/{version}/data/en_US/item.json", timeout=30).json()
        p.write_text(json.dumps(data), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def item_prices(version: str) -> dict:
    """{itemId(int): oro total}."""
    return {int(k): v["gold"]["total"] for k, v in items(version)["data"].items()}


def item_names(version: str) -> dict:
    return {int(k): v["name"] for k, v in items(version)["data"].items()}


def portrait(version: str, champ: str) -> Path:
    """Cuadrado del campeón (para plantillas del detector). champ = nombre
    interno de Match-V5 (p. ej. 'Chogath', 'MonkeyKing')."""
    p = _dir(version) / "portraits" / f"{champ}.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(f"{BASE}/cdn/{version}/img/champion/{champ}.png", timeout=30)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p
