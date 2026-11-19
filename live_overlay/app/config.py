"""Rutas y constantes de despliegue. Nada de aquí apunta fuera de este proyecto."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # item-<version>.json de Data Dragon
SNAPSHOT_DIR = DATA_DIR / "snapshots"   # capturas propias de scripts/record.py
WEB_DIR = ROOT / "web"
FACES_PATH = ROOT / "config" / "faces.json"   # rectángulo de cada campeón en su splash (versionado, no es caché)
SPLASH_OVERRIDES_DIR = ROOT / "config" / "splashes"   # <id>.jpg propios del owner: mandan sobre cualquier fuente externa

IMG_CACHE_DIR = CACHE_DIR / "img"       # iconos de campeón y de objetivo, descargados una vez
DDRAGON_CDN = "https://ddragon.leagueoflegends.com/cdn"
# Iconos de minimapa de Barón y dragones: Data Dragon no los publica; CommunityDragon
# es el espejo comunitario de los assets del juego (solo lectura, solo iconos).
CDRAGON_MINIMAP_ICONS = "https://raw.communitydragon.org/latest/game/assets/ux/minimap/icons"
# Splashes: el `splash/<id>_0.jpg` sin versión de Data Dragon se queda viejo tras un rework (Fiddlesticks
# sirve aún el arte de antes de 2020); los assets del cliente en CommunityDragon van al día.
CDRAGON_GAME_DATA = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default"

HOST = "127.0.0.1"
PORT = 8020                             # build_tracker usa 8000 y macro_coach 8010

# Tecla del marcador (Tab). Mientras esté pulsada las ventanas se vuelven opacas.
# Solo se LEE su estado (GetAsyncKeyState); nunca se inyecta entrada.
SCOREBOARD_VK = 0x09

# El cliente de League sirve esto en local durante cualquier partida (incluida
# la Herramienta de práctica) con certificado autofirmado. Es el API
# sancionado por Riot; aquí no se lee memoria del juego.
LIVE_CLIENT_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
