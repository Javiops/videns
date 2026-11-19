"""Servidor: sin partida -> in_game False; con partida -> payload del tracker; iconos cacheados; catálogo en segundo plano."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

import app.main as main
from app.tracker import OverlayTracker

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


class Catalog:
    version = "test"
    champions = 2

    @staticmethod
    def total(item_id: int):
        return {3031: 3500}.get(item_id)

    @staticmethod
    def champion_image_id(raw, name=None):
        return {"game_character_displayname_Ahri": "Ahri", "game_character_displayname_Garen": "Garen"}.get(raw or "")

    @staticmethod
    def champion_list():
        return [{"id": "Ahri", "name": "Ahri"}, {"id": "Garen", "name": "Garen"}, {"id": "Yunara", "name": "Yunara"}]

    @staticmethod
    def champion_key(image_id):
        return {"Ahri": 103, "Old": 9}.get(image_id)


def player(name, team, champ, dead=False):
    return {"riotId": f"{name}#TAG", "riotIdGameName": name, "summonerName": f"{name}#TAG",
            "team": team, "championName": champ, "rawChampionName": f"game_character_displayname_{champ}",
            "isDead": dead, "respawnTimer": 0.0, "level": 12,
            "items": [{"itemID": 3031, "price": 725, "count": 1, "slot": 0}]}


SNAP = {
    "activePlayer": {"riotId": "Alice#TAG", "riotIdGameName": "Alice", "currentGold": 300.4},
    "allPlayers": [player("Alice", "ORDER", "Ahri"), player("Bob", "ORDER", "Garen", dead=True), player("Dan", "CHAOS", "Jinx")],
    "events": {"Events": [{"EventID": 0, "EventName": "BaronKill", "EventTime": 1400.0, "KillerName": "Alice", "Assisters": [], "Stolen": "False"}]},
    "gameData": {"gameMode": "CLASSIC", "gameTime": 1401.0, "mapNumber": 11},
}


def _ready(monkeypatch, catalog):
    monkeypatch.setattr(main, "_catalog", catalog)
    monkeypatch.setattr(main, "_catalog_loading", False)
    monkeypatch.setattr(main, "_catalog_failed_at", None)


def test_endpoints(monkeypatch):
    monkeypatch.setattr(main, "_tracker", OverlayTracker())
    _ready(monkeypatch, Catalog())
    client = TestClient(main.app)

    assert client.get("/api/health").json() == {"ok": True}
    page = client.get("/")
    assert page.status_code == 200 and "/api/overlay" in page.text and "Live Overlay" in page.text

    monkeypatch.setattr(main, "fetch_snapshot", lambda timeout=2.0: None)
    assert client.get("/api/overlay").json() == {"in_game": False}

    monkeypatch.setattr(main, "fetch_snapshot", lambda timeout=2.0: SNAP)
    out = client.get("/api/overlay").json()
    assert out["in_game"] and out["my_team"] == 100 and out["gold"]["wallet"] == 300
    assert out["gold"]["teams"][0]["item_gold"] == 7000 and out["catalog_version"] == "test"
    icons = {p["champion"]: p["icon"] for t in out["gold"]["teams"] for p in t["players"]}
    assert icons == {"Ahri": "Ahri", "Garen": "Garen", "Jinx": None}
    holders = out["objectives"]["buffs"][0]["holders"]
    assert [(h["champion"], h["icon"]) for h in holders] == [("Ahri", "Ahri")], "Bob estaba muerto"


def test_icons_are_fetched_once_and_validated(tmp_path, monkeypatch):
    _ready(monkeypatch, Catalog())
    monkeypatch.setattr(main, "IMG_CACHE_DIR", tmp_path)
    calls = []

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    CDRAGON = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default"

    def fake_get(url, timeout=None, **kwargs):
        calls.append(url)
        if "champion/Ahri.png" in url or "icons/baron.png" in url:
            return Response(PNG)
        if url == f"{CDRAGON}/assets/characters/ahri/skins/base/images/ahri_splash_uncentered_0.jpg":
            return Response(JPEG)                       # el arte actual del cliente
        if url == f"{CDRAGON}/v1/champions/9.json":     # el patrón falló: la ruta exacta viene del JSON
            return Response(b'{"skins": [{"uncenteredSplashPath": "/lol-game-data/assets/ASSETS/Characters/Old/Skins/Base/Images/old_splash_uncentered_0.jpg"}]}')
        if url == f"{CDRAGON}/assets/characters/old/skins/base/images/old_splash_uncentered_0.jpg":
            return Response(JPEG)
        if "cdn/img/champion/splash/Fallback_0.jpg" in url:
            return Response(JPEG)                       # Data Dragon solo cuando CommunityDragon no tiene nada
        if "splash/Png_0.jpg" in url:
            return Response(PNG)
        if "champion/Html.png" in url:
            return Response(b"<html>not found</html>")
        raise main.httpx.ConnectError("offline")

    monkeypatch.setattr(main.httpx, "get", fake_get)
    client = TestClient(main.app)

    r = client.get("/img/champion/Ahri.png")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/png") and r.content == PNG
    assert calls == ["https://ddragon.leagueoflegends.com/cdn/test/img/champion/Ahri.png"]
    assert (tmp_path / "champion" / "test" / "Ahri.png").read_bytes() == PNG
    assert not list((tmp_path / "champion" / "test").glob("*.tmp")), "escritura atómica sin restos"
    assert client.get("/img/champion/Ahri.png").status_code == 200 and len(calls) == 1, "la segunda vez sale de la caché"

    assert client.get("/img/champion/Html.png").status_code == 404, "una respuesta que no es PNG no se sirve ni se cachea"
    assert not (tmp_path / "champion" / "test" / "Html.png").exists()
    assert client.get("/img/champion/Offline.png").status_code == 404
    assert client.get("/img/champion/..%2F..%2Fetc.png").status_code == 404, "sin recorrido de rutas"
    assert client.get("/img/champion/a b.png").status_code == 404

    r = client.get("/img/objective/baron.png")
    assert r.status_code == 200 and (tmp_path / "objective" / "baron.png").exists()
    assert calls[-1] == "https://raw.communitydragon.org/latest/game/assets/ux/minimap/icons/baron.png"
    assert client.get("/img/objective/atakhan.png").status_code == 404, "solo la lista blanca"

    r = client.get("/img/splash/Ahri.jpg")
    assert r.status_code == 200 and r.content == JPEG and (tmp_path / "splashes" / "Ahri.jpg").exists()
    assert r.headers["content-type"].startswith("image/jpeg")
    assert calls[-1] == f"{CDRAGON}/assets/characters/ahri/skins/base/images/ahri_splash_uncentered_0.jpg", "primero el arte actual del cliente"
    n = len(calls)
    assert client.get("/img/splash/Ahri.jpg").status_code == 200 and len(calls) == n, "de la caché"

    r = client.get("/img/splash/Old.jpg")
    assert r.status_code == 200 and r.content == JPEG, "patrón fallido -> ruta del JSON del campeón"
    assert (tmp_path / "cdragon" / "9.json").exists() and calls[-1].endswith("/old/skins/base/images/old_splash_uncentered_0.jpg")

    r = client.get("/img/splash/Fallback.jpg")
    assert r.status_code == 200 and calls[-1] == "https://ddragon.leagueoflegends.com/cdn/img/champion/splash/Fallback_0.jpg", "Data Dragon solo como respaldo"

    overrides = tmp_path / "overrides"
    overrides.mkdir()
    own = JPEG + b"own"
    (overrides / "Ahri.jpg").write_bytes(own)
    monkeypatch.setattr(main, "SPLASH_OVERRIDES_DIR", overrides)
    n = len(calls)
    r = client.get("/img/splash/Ahri.jpg")
    assert r.status_code == 200 and r.content == own and len(calls) == n, "el arte propio manda, sin red y sin caché"
    (overrides / "Empty.jpg").write_bytes(b"")
    assert client.get("/img/splash/Empty.jpg").status_code == 404, "un fichero vacío no cuenta como arte propio"
    assert client.get("/img/splash/Png.jpg").status_code == 404, "un PNG donde se espera JPEG no se sirve"
    assert client.get("/img/splash/..%2Fx.jpg").status_code == 404

    # disco fallando: 404, nunca 500
    monkeypatch.setattr(main, "IMG_CACHE_DIR", tmp_path / "file-not-dir")
    (tmp_path / "file-not-dir").write_text("x", encoding="utf-8")
    assert client.get("/img/objective/dragon.png").status_code == 404

    _ready(monkeypatch, None)
    monkeypatch.setattr(main, "_catalog_failed_at", main.time.monotonic())
    assert client.get("/img/champion/Ahri.png").status_code == 503, "sin catálogo no se conoce la versión"


def test_head_marks_are_validated_and_persisted(tmp_path, monkeypatch):
    path = tmp_path / "config" / "faces.json"
    monkeypatch.setattr(main, "FACES_PATH", path)
    client = TestClient(main.app)
    assert client.get("/api/faces").json() == {}, "sin fichero no hay cabezas"

    r = client.put("/api/faces/Yunara", json={"x": 0.63, "y": 0.21})
    assert r.status_code == 200 and r.json() == {"Yunara": {"x": 0.63, "y": 0.21, "w": None}}, "marca antigua de cabeza: sin anchura"
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    assert client.get("/api/faces").json()["Yunara"]["x"] == 0.63

    r = client.put("/api/faces/Yunara", json={"x": 0.63, "y": 0.21, "w": 0.4})
    assert r.status_code == 200 and r.json()["Yunara"] == {"x": 0.63, "y": 0.21, "w": 0.4}, "rectángulo: centro y anchura"
    assert client.put("/api/faces/Yunara", json={"x": 0.6, "y": 0.2, "w": 0}).status_code == 422, "anchura nula"
    assert client.put("/api/faces/Yunara", json={"x": 0.6, "y": 0.2, "w": 1.2}).status_code == 422, "más ancho que el splash"
    assert client.put("/api/faces/Yunara", json={"x": 1.5, "y": 0.2}).status_code == 422, "fuera del splash"
    assert client.put("/api/faces/Yunara", json={"x": 0.4}).status_code == 422, "campos obligatorios"
    assert client.put("/api/faces/bad%20id", json={"x": 0.4, "y": 0.3}).status_code == 404
    assert client.get("/api/faces").json()["Yunara"] == {"x": 0.63, "y": 0.21, "w": 0.4}, "un guardado inválido no toca lo guardado"

    path.write_text('{"Yunara": {"x": 0.63, "y": 0.21}, "Ahri": {"x": 0.5, "y": 0.3, "w": 0.45}, "Broken": "x", "Bad": {"x": 9, "y": 0}}', encoding="utf-8")
    assert client.get("/api/faces").json() == {"Yunara": {"x": 0.63, "y": 0.21, "w": None}, "Ahri": {"x": 0.5, "y": 0.3, "w": 0.45}}, "entradas rotas se ignoran; las antiguas siguen valiendo"

    _ready(monkeypatch, Catalog())
    overrides = tmp_path / "overrides"
    overrides.mkdir()
    (overrides / "Garen.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 8)
    monkeypatch.setattr(main, "SPLASH_OVERRIDES_DIR", overrides)
    listing = client.get("/api/champions").json()
    assert listing["version"] == "test"
    assert [(c["id"], c["head"]) for c in listing["champions"]] == [("Ahri", True), ("Garen", False), ("Yunara", True)], "la lista marca los ya hechos"
    versions = {c["id"]: c["art_v"] for c in listing["champions"]}
    assert versions["Ahri"] == 0 and versions["Garen"] > 0, "la versión del arte es la fecha del splash propio"
    monkeypatch.setattr(main, "_tracker", OverlayTracker())
    monkeypatch.setattr(main, "fetch_snapshot", lambda timeout=2.0: SNAP)
    served = {p["champion"]: p.get("art_v") for t in client.get("/api/overlay").json()["gold"]["teams"] for p in t["players"]}
    assert served["Garen"] > 0 and served["Ahri"] == 0 and served["Jinx"] is None, "el panel recibe la versión por jugador con icono"
    _ready(monkeypatch, None)
    monkeypatch.setattr(main, "_catalog_failed_at", main.time.monotonic())
    assert client.get("/api/champions").status_code == 503, "sin catálogo no hay lista"
    _ready(monkeypatch, Catalog())

    remaining = {"Ahri": {"x": 0.5, "y": 0.3, "w": 0.45}}
    assert client.delete("/api/faces/Yunara").json() == remaining
    assert client.delete("/api/faces/Yunara").json() == remaining, "borrar lo que no existe no falla"
    page = client.get("/faces")
    assert page.status_code == 200 and "/api/faces" in page.text and "rows.js" in page.text
    assert client.get("/rows.css").headers["content-type"].startswith("text/css")
    assert client.get("/rows.js").headers["content-type"].startswith("application/javascript")
    assert "artStyle" in client.get("/rows.js").text and "rows.js" in client.get("/").text


def test_catalog_loads_in_the_background_and_retries_with_backoff(monkeypatch):
    attempts = []
    started = threading.Event()

    def fake_load():
        attempts.append(1)
        started.set()
        return None

    _ready(monkeypatch, None)
    monkeypatch.setattr(main, "load_catalog", fake_load)
    assert main.get_catalog(wait=True) is None and len(attempts) == 1
    assert main.get_catalog() is None and len(attempts) == 1, "sin red no se martillea Data Dragon en cada poll"
    monkeypatch.setattr(main, "_catalog_failed_at", main.time.monotonic() - main.CATALOG_RETRY_S - 1)
    assert main.get_catalog(wait=True) is None and len(attempts) == 2

    monkeypatch.setattr(main, "load_catalog", lambda: Catalog())
    monkeypatch.setattr(main, "_catalog_failed_at", None)
    assert main.get_catalog(wait=True).version == "test"
    monkeypatch.setattr(main, "load_catalog", fake_load)
    assert main.get_catalog().version == "test", "cargado una vez, no se recarga"

    # una excepción dentro de la carga tampoco tumba nada
    _ready(monkeypatch, None)
    monkeypatch.setattr(main, "load_catalog", lambda: (_ for _ in ()).throw(OSError("disco lleno")))
    assert main.get_catalog(wait=True) is None and main._catalog_failed_at is not None

    # mientras carga, los polls no esperan
    gate = threading.Event()

    def slow_load():
        gate.wait(5)
        return Catalog()

    _ready(monkeypatch, None)
    monkeypatch.setattr(main, "load_catalog", slow_load)
    assert main.get_catalog() is None
    assert main.get_catalog() is None, "la carga en curso no se duplica ni bloquea"
    gate.set()
    for _ in range(200):
        if main._catalog is not None:
            break
        main.time.sleep(0.01)
    assert main.get_catalog().version == "test"


def test_overlay_poll_never_waits_for_the_catalog(monkeypatch):
    gate = threading.Event()
    _ready(monkeypatch, None)
    monkeypatch.setattr(main, "_tracker", OverlayTracker())
    monkeypatch.setattr(main, "load_catalog", lambda: gate.wait(5) or Catalog())
    monkeypatch.setattr(main, "fetch_snapshot", lambda timeout=2.0: SNAP)
    client = TestClient(main.app)
    out = client.get("/api/overlay").json()
    assert out["in_game"] and out["catalog_version"] is None and out["gold"]["teams"][0]["approx"] is True
    gate.set()
