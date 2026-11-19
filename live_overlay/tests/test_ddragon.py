"""Catálogo de Data Dragon: totales por id, iconos de campeón, caché por versión, sin red."""

from __future__ import annotations

import json

import app.ddragon as ddragon
from app.ddragon import Catalog, cached_versions, load_catalog

ITEMS = {
    "version": "16.19.1",
    "data": {
        "1036": {"name": "Long Sword", "gold": {"base": 350, "total": 350}},
        "3031": {"name": "Infinity Edge", "gold": {"base": 725, "total": 3500}},
        "3340": {"name": "Stealth Ward", "gold": {"base": 0, "total": 0}},
        "bad": {"gold": {"total": 1}},
        "7000": "not a dict",
    },
}
CHAMPIONS = {
    "version": "16.19.1",
    "data": {
        "Aatrox": {"id": "Aatrox", "key": "266", "name": "Aatrox", "image": {"full": "Aatrox.png"}},
        "MonkeyKing": {"id": "MonkeyKing", "key": "62", "name": "Wukong", "image": {"full": "MonkeyKing.png"}},
        "Fiddlesticks": {"id": "Fiddlesticks", "key": "9", "name": "Fiddlesticks", "image": {"full": "Fiddlesticks.png"}},
        "Nunu": {"id": "Nunu", "name": "Nunu & Willump", "image": {"full": "Nunu.png"}},
        "Renata": {"id": "Renata", "name": "Renata Glasc"},
        "Broken": "not a dict",
    },
}


def test_catalog_totals_and_unknown_ids():
    catalog = Catalog.from_payloads(ITEMS, CHAMPIONS)
    assert catalog.version == "16.19.1" and len(catalog) == 3 and catalog.champions == 5
    assert catalog.total(3031) == 3500, "el total, nunca el coste base"
    assert catalog.total(1036) == 350 and catalog.total(3340) == 0
    assert catalog.total(999999) is None
    assert catalog.name(3031) == "Infinity Edge" and catalog.name(1) == ""


def test_champion_keys():
    catalog = Catalog.from_payloads(ITEMS, CHAMPIONS)
    assert catalog.champion_key("Fiddlesticks") == 9 and catalog.champion_key("MonkeyKing") == 62
    assert catalog.champion_key("Renata") is None, "sin key en el payload"
    assert catalog.champion_key("Nope") is None


def test_champion_list_is_sorted_by_display_name():
    catalog = Catalog.from_payloads(ITEMS, CHAMPIONS)
    names = [c["name"] for c in catalog.champion_list()]
    assert names == ["Aatrox", "Fiddlesticks", "Nunu & Willump", "Renata Glasc", "Wukong"], "por nombre visible, no por id"
    assert {c["id"] for c in catalog.champion_list()} == {"Aatrox", "Fiddlesticks", "Nunu", "Renata", "MonkeyKing"}
    assert Catalog.from_payloads(ITEMS).champion_list() == []


def test_champion_image_ids():
    catalog = Catalog.from_payloads(ITEMS, CHAMPIONS)
    assert catalog.champion_image_id("game_character_displayname_Aatrox", "Aatrox") == "Aatrox"
    assert catalog.champion_image_id("game_character_displayname_MonkeyKing", "Wukong") == "MonkeyKing"
    assert catalog.champion_image_id("game_character_displayname_FiddleSticks", "Fiddlesticks") == "Fiddlesticks", "mayúsculas"
    assert catalog.champion_image_id(None, "Nunu & Willump") == "Nunu", "por nombre visible"
    assert catalog.champion_image_id("", "renata glasc") == "Renata"
    assert catalog.champion_image_id("game_character_displayname_Nope", "Nope") is None
    assert catalog.champion_image_id(None, None) is None
    assert Catalog.from_payloads(ITEMS).champion_image_id("game_character_displayname_Aatrox", "Aatrox") is None, "sin champion.json"


def test_cached_versions_sorted_newest_first(tmp_path):
    for version in ("16.9.1", "16.19.1", "16.18.1"):
        (tmp_path / f"item-{version}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "champion-16.19.1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    assert cached_versions(tmp_path) == ["16.19.1", "16.18.1", "16.9.1"]
    assert cached_versions(tmp_path / "missing") == []


def test_load_catalog_offline_uses_newest_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: None)
    monkeypatch.setattr(ddragon.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(ddragon.httpx.ConnectError("offline")))
    assert load_catalog(tmp_path) is None, "sin red y sin caché no hay catálogo"
    (tmp_path / "item-16.18.1.json").write_text(json.dumps(dict(ITEMS, version="16.18.1")), encoding="utf-8")
    (tmp_path / "item-16.19.1.json").write_text(json.dumps(ITEMS), encoding="utf-8")
    (tmp_path / "champion-16.19.1.json").write_text(json.dumps(CHAMPIONS), encoding="utf-8")
    catalog = load_catalog(tmp_path)
    assert catalog is not None and catalog.version == "16.19.1" and catalog.total(3031) == 3500
    assert catalog.champion_image_id("game_character_displayname_MonkeyKing") == "MonkeyKing"


def _fake_get(calls):
    class Response:
        def __init__(self, payload):
            self.status_code = 200
            self.content = json.dumps(payload).encode()

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None, **kwargs):
        calls.append(url)
        return Response(CHAMPIONS if "champion.json" in url else ITEMS)

    return fake_get


def test_load_catalog_downloads_once_then_reads_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: "16.19.1")
    monkeypatch.setattr(ddragon.httpx, "get", _fake_get(calls))
    first = load_catalog(tmp_path)
    assert first is not None and first.total(3031) == 3500 and first.champions == 5
    assert len(calls) == 2, "item.json y champion.json, una vez cada uno"
    assert (tmp_path / "item-16.19.1.json").exists() and (tmp_path / "champion-16.19.1.json").exists()
    second = load_catalog(tmp_path)
    assert second is not None and len(calls) == 2, "la segunda carga sale de la caché"


def test_load_catalog_falls_back_when_download_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: "16.20.1")

    def failing_get(url, timeout=None, **kwargs):
        raise ddragon.httpx.ConnectError("offline")

    monkeypatch.setattr(ddragon.httpx, "get", failing_get)
    (tmp_path / "item-16.19.1.json").write_text(json.dumps(ITEMS), encoding="utf-8")
    catalog = load_catalog(tmp_path)
    assert catalog is not None and catalog.version == "16.19.1"
    assert catalog.champions == 0 and catalog.champion_image_id("game_character_displayname_Aatrox") is None


def test_corrupt_cache_is_discarded_and_never_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: None)
    (tmp_path / "item-16.19.1.json").write_text('{"version": "16.19.1", "data": {"1036": {', encoding="utf-8")
    assert load_catalog(tmp_path) is None
    assert not (tmp_path / "item-16.19.1.json").exists(), "la caché truncada se borra para volver a bajarla"

    class Truncated:
        status_code = 200
        content = b'{"version": "16.19.1", "data": {'

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: "16.19.1")
    monkeypatch.setattr(ddragon.httpx, "get", lambda url, timeout=None, **k: Truncated())
    assert load_catalog(tmp_path) is None
    assert not (tmp_path / "item-16.19.1.json").exists(), "una descarga truncada nunca llega a la caché"
    assert not list(tmp_path.glob("*.tmp")), "ni deja temporales"


def test_newest_corrupt_cache_falls_back_to_the_older_version(tmp_path, monkeypatch):
    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: None)
    (tmp_path / "item-16.18.1.json").write_text(json.dumps(dict(ITEMS, version="16.18.1")), encoding="utf-8")
    (tmp_path / "item-16.19.1.json").write_text("{corrupt", encoding="utf-8")
    catalog = load_catalog(tmp_path)
    assert catalog is not None and catalog.version == "16.18.1"
    assert not (tmp_path / "item-16.19.1.json").exists(), "la corrupta se descarta"


def test_login_page_with_200_is_never_cached_and_older_cache_still_serves(tmp_path, monkeypatch):
    class Portal:
        status_code = 200
        content = b"<html>login</html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: "16.20.1")
    monkeypatch.setattr(ddragon.httpx, "get", lambda url, timeout=None, **k: Portal())
    (tmp_path / "item-16.19.1.json").write_text(json.dumps(ITEMS), encoding="utf-8")
    catalog = load_catalog(tmp_path)
    assert catalog is not None and catalog.version == "16.19.1"
    assert not (tmp_path / "item-16.20.1.json").exists()


def test_empty_item_data_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ddragon, "latest_version", lambda timeout=5.0: None)
    (tmp_path / "item-16.19.1.json").write_text(json.dumps({"version": "16.19.1", "data": {}}), encoding="utf-8")
    assert load_catalog(tmp_path) is None
    assert not (tmp_path / "item-16.19.1.json").exists(), "un item.json sin objetos se descarta"


def test_unwritable_cache_dir_returns_none(tmp_path, monkeypatch):
    blocker = tmp_path / "cache"
    blocker.write_text("not a directory", encoding="utf-8")
    assert load_catalog(blocker) is None
