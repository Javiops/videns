"""Servidor local del overlay: sirve el panel, `GET /api/overlay` y los iconos.

Sin modelo, sin clave de Riot, sin base de datos: solo el Live Client API del
cliente de League, el catálogo de Data Dragon y los iconos de minimapa de
CommunityDragon, todos cacheados en data/cache/ tras la primera descarga.

El catálogo se carga en un hilo aparte (al arrancar y, si falla, cada
CATALOG_RETRY_S): ningún poll del panel espera a la red; hasta que esté, el
oro sale del coste base marcado con ≈ y los iconos de campeón esperan.

Dev: .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8020
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import (
    CDRAGON_GAME_DATA, CDRAGON_MINIMAP_ICONS, DDRAGON_CDN, FACES_PATH, IMG_CACHE_DIR, SPLASH_OVERRIDES_DIR, WEB_DIR,
)
from app.ddragon import load_catalog
from app.live_client import fetch_snapshot
from app.tracker import OverlayTracker

CATALOG_RETRY_S = 60.0   # sin red, no reintentar Data Dragon en cada poll
IMAGE_TIMEOUT_S = 10.0
_IMAGE_ID = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
# Iconos de minimapa que el panel puede pedir (nombres tal cual en CommunityDragon).
OBJECTIVE_ICONS = frozenset({
    "baron", "dragon", "dragon_elder", "dragon_infernal", "dragon_cloud", "dragon_mountain",
    "dragon_ocean", "dragon_chemtech", "dragon_hextech", "riftherald",
})

log = logging.getLogger("overlay.server")

_tracker = OverlayTracker()
_catalog = None
_catalog_loading = False
_catalog_failed_at: float | None = None
_catalog_lock = threading.Lock()
_image_lock = threading.Lock()


def _load_catalog_job() -> None:
    global _catalog, _catalog_loading, _catalog_failed_at
    try:
        loaded = load_catalog()
    except Exception as exc:   # un fallo del catálogo nunca tumba el servidor
        log.warning("catálogo: %s", exc)
        loaded = None
    with _catalog_lock:
        _catalog = loaded
        _catalog_failed_at = None if loaded is not None else time.monotonic()
        _catalog_loading = False


def get_catalog(wait: bool = False):
    """El catálogo si ya está; si no, lanza (o relanza tras la espera) su carga en segundo plano y devuelve None."""
    global _catalog_loading
    with _catalog_lock:
        if _catalog is not None:
            return _catalog
        if _catalog_loading:
            return None
        if _catalog_failed_at is not None and time.monotonic() - _catalog_failed_at < CATALOG_RETRY_S:
            return None
        _catalog_loading = True
    thread = threading.Thread(target=_load_catalog_job, name="catalog", daemon=True)
    thread.start()
    if wait:
        thread.join()
        return _catalog
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_catalog()   # calentar la caché sin bloquear el primer poll
    yield


app = FastAPI(title="Live Overlay", version="0.2.0", lifespan=lifespan)


def cached_image(path: Path, url: str, magic: tuple[bytes, ...] = (_PNG_MAGIC,)) -> Path | None:
    """Descarga una imagen a la caché una sola vez; None si no se pudo (sin red, 404, formato inesperado, disco)."""
    with _image_lock:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
            response = httpx.get(url, timeout=IMAGE_TIMEOUT_S, follow_redirects=True)
            response.raise_for_status()
            if not response.content.startswith(magic):
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_bytes(response.content)
            temp.replace(path)
            return path
        except (httpx.HTTPError, OSError):
            return None


def _png(path: Path | None) -> FileResponse:
    if path is None:
        raise HTTPException(status_code=404, detail="icon unavailable")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def page() -> FileResponse:
    return FileResponse(WEB_DIR / "overlay.html")


@app.get("/faces")
def faces_page() -> FileResponse:
    """Marcar a mano dónde está la cabeza de cada campeón en su splash."""
    return FileResponse(WEB_DIR / "faces.html")


@app.get("/draft")
def draft_page() -> FileResponse:
    """Vista previa de los widgets de selección de campeones: un draft de ranked paso a paso con los datos
    del drafting tool (draft/scripts/tool_server.py en :8030, otro proyecto: solo por HTTP)."""
    return FileResponse(WEB_DIR / "draft.html")


@app.get("/game")
def game_page() -> FileResponse:
    """Vista previa del panel de partida con el diseño VIDENS. (una ventana, por secciones) sobre una pantalla
    de juego simulada, con un momento real de web/preview/game-fixture.json."""
    return FileResponse(WEB_DIR / "game.html")


_PREVIEW_FILE = re.compile(r"^[a-z0-9-]{1,40}\.(json|webp|png)$")


@app.get("/preview/{name}")
def preview_file(name: str) -> FileResponse:
    """Datos e imágenes de las vistas previas (web/preview/), solo nombres sencillos."""
    path = WEB_DIR / "preview" / name
    if not _PREVIEW_FILE.match(name) or not path.is_file():
        raise HTTPException(status_code=404, detail="no such preview file")
    return FileResponse(path)


@app.get("/videns-mark.svg")
def videns_mark() -> FileResponse:
    """La marca V. de Videns (copia de ../brand/v-mark.svg) para la cabecera de los widgets."""
    return FileResponse(WEB_DIR / "videns-mark.svg", media_type="image/svg+xml")


@app.get("/inter-var-latin.woff2")
def inter_font() -> FileResponse:
    """Inter (SIL OFL), la fuente de la marca VIDENS. (copia de videns-site/fonts) para el rótulo del panel."""
    return FileResponse(WEB_DIR / "inter-var-latin.woff2", media_type="font/woff2", headers={"Cache-Control": "max-age=86400"})


@app.get("/rows.css")
def rows_css() -> FileResponse:
    """Estilo de la fila de oro, compartido por el panel y /faces."""
    return FileResponse(WEB_DIR / "rows.css", media_type="text/css")


@app.get("/rows.js")
def rows_js() -> FileResponse:
    """Geometría de la franja del splash, compartida por el panel y /faces."""
    return FileResponse(WEB_DIR / "rows.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


# --- cabeza por campeón (para la franja del splash) ------------------------------

class Head(BaseModel):
    """Marca del campeón en su splash, como fracciones del ancho y del alto: centro (x, y) y anchura w del
    rectángulo que muestra la fila (su alto lo fija la proporción de la fila). Sin w es una marca antigua
    de cabeza, que el panel convierte a la franja desde la cabeza hasta el borde derecho."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float | None = Field(default=None, gt=0.0, le=1.0)


def load_faces() -> dict[str, dict]:
    try:
        raw = json.loads(FACES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    faces: dict[str, dict] = {}
    for image_id, value in (raw.items() if isinstance(raw, dict) else []):
        if not _IMAGE_ID.match(str(image_id)) or not isinstance(value, dict):
            continue
        try:
            faces[str(image_id)] = Head(**value).model_dump()
        except Exception:
            continue   # una entrada rota no invalida las demás
    return faces


def save_faces(faces: dict[str, dict]) -> None:
    FACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = FACES_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(sorted(faces.items())), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(FACES_PATH)


@app.get("/api/faces")
def api_faces() -> dict:
    return load_faces()


def art_version(image_id: str) -> int:
    """Versión del splash de un campeón para la URL: la fecha del fichero propio si lo hay, 0 si no.
    Cambia al reemplazar el fichero, así los navegadores no siguen enseñando el arte anterior."""
    try:
        return int((SPLASH_OVERRIDES_DIR / f"{image_id}.jpg").stat().st_mtime)
    except OSError:
        return 0


@app.get("/api/champions")
def api_champions() -> dict:
    """Todos los campeones del catálogo, por nombre, con si ya tienen marca y la versión de su arte."""
    catalog = get_catalog()
    if catalog is None:
        raise HTTPException(status_code=503, detail="no catalog yet")
    faces = load_faces()
    return {
        "version": catalog.version,
        "champions": [dict(c, head=c["id"] in faces, art_v=art_version(c["id"])) for c in catalog.champion_list()],
    }


@app.put("/api/faces/{image_id}")
def api_face_put(image_id: str, head: Head) -> dict:
    if not _IMAGE_ID.match(image_id):
        raise HTTPException(status_code=404, detail="bad champion id")
    faces = load_faces()
    faces[image_id] = head.model_dump()
    try:
        save_faces(faces)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"no se pudo guardar faces.json: {exc}")
    return faces


@app.delete("/api/faces/{image_id}")
def api_face_delete(image_id: str) -> dict:
    faces = load_faces()
    if faces.pop(image_id, None) is not None:
        try:
            save_faces(faces)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"no se pudo guardar faces.json: {exc}")
    return faces


@app.get("/api/overlay")
def api_overlay() -> dict:
    snap = fetch_snapshot()
    if not snap:
        return {"in_game": False}
    payload = _tracker.observe(snap, get_catalog())
    for team in payload["gold"]["teams"]:
        for player in team["players"]:
            if player.get("icon"):
                player["art_v"] = art_version(player["icon"])
    return payload


@app.get("/img/champion/{image_id}.png")
def champion_icon(image_id: str) -> FileResponse:
    if not _IMAGE_ID.match(image_id):
        raise HTTPException(status_code=404, detail="bad icon id")
    catalog = get_catalog()
    if catalog is None:
        raise HTTPException(status_code=503, detail="no catalog yet")
    path = IMG_CACHE_DIR / "champion" / catalog.version / f"{image_id}.png"
    return _png(cached_image(path, f"{DDRAGON_CDN}/{catalog.version}/img/champion/{image_id}.png"))


@app.get("/img/objective/{name}.png")
def objective_icon(name: str) -> FileResponse:
    if name not in OBJECTIVE_ICONS:
        raise HTTPException(status_code=404, detail="unknown objective icon")
    path = IMG_CACHE_DIR / "objective" / f"{name}.png"
    return _png(cached_image(path, f"{CDRAGON_MINIMAP_ICONS}/{name}.png"))


def _cdragon_splash_urls(image_id: str) -> list[str]:
    """URLs candidatas del splash actual del cliente, de más directa a más costosa.

    La carpeta de assets es el id interno en minúsculas (fiddlesticks, monkeyking); si ese patrón no
    existe, la ruta exacta sale del JSON del campeón (`skins[0].uncenteredSplashPath`), que se cachea.
    """
    folder = image_id.lower()
    urls = [f"{CDRAGON_GAME_DATA}/assets/characters/{folder}/skins/base/images/{folder}_splash_uncentered_0.jpg"]
    catalog = get_catalog()
    key = catalog.champion_key(image_id) if catalog is not None else None
    if key is not None:
        meta = IMG_CACHE_DIR / "cdragon" / f"{key}.json"
        try:
            if not meta.is_file():
                response = httpx.get(f"{CDRAGON_GAME_DATA}/v1/champions/{key}.json", timeout=IMAGE_TIMEOUT_S)
                response.raise_for_status()
                meta.parent.mkdir(parents=True, exist_ok=True)
                meta.write_bytes(response.content)
            skins = json.loads(meta.read_text(encoding="utf-8")).get("skins") or []
            rel = str((skins[0] if skins else {}).get("uncenteredSplashPath") or "")
            prefix = "/lol-game-data/assets/"
            if rel.startswith(prefix):
                urls.append(f"{CDRAGON_GAME_DATA}/{rel[len(prefix):].lower()}")
        except (httpx.HTTPError, OSError, ValueError, IndexError, AttributeError):
            pass
    return urls


@app.get("/img/splash/{image_id}.jpg")
def champion_splash(image_id: str) -> FileResponse:
    """Splash completo (1215x717) del campeón: el arte actual del cliente vía CommunityDragon; Data Dragon
    (`splash/<id>_0.jpg`, sin versión y a veces anterior a un rework) solo como respaldo."""
    if not _IMAGE_ID.match(image_id):
        raise HTTPException(status_code=404, detail="bad art id")
    override = SPLASH_OVERRIDES_DIR / f"{image_id}.jpg"   # arte puesto a mano por el owner: manda siempre
    if override.is_file() and override.stat().st_size > 0:
        return FileResponse(override, media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})
    path = IMG_CACHE_DIR / "splashes" / f"{image_id}.jpg"
    found = None
    if path.is_file() and path.stat().st_size > 0:
        found = path
    else:
        for url in _cdragon_splash_urls(image_id) + [f"{DDRAGON_CDN}/img/champion/splash/{image_id}_0.jpg"]:
            found = cached_image(path, url, magic=(_JPEG_MAGIC,))
            if found is not None:
                break
    if found is None:
        raise HTTPException(status_code=404, detail="splash unavailable")
    return FileResponse(found, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
