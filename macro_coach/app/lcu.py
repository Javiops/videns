"""LCU (cliente de League): auth por lockfile y endpoints de replays.
El cliente debe estar abierto y logueado. Las replays solo existen para la
región del cliente y el parche instalado."""
import base64
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

DEFAULT_LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


class LcuError(RuntimeError):
    pass


def _lockfile() -> tuple:
    path = os.environ.get("LCU_LOCKFILE", DEFAULT_LOCKFILE)
    if not Path(path).exists():
        raise LcuError(f"no existe {path} — ¿está abierto el cliente de League?")
    name, pid, port, password, protocol = Path(path).read_text().split(":")
    return int(port), password


def _request(method: str, path: str, body=None, allow_404: bool = False):
    port, password = _lockfile()
    auth = base64.b64encode(f"riot:{password}".encode()).decode()
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        method=method)
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=15) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        if e.code == 404 and allow_404:
            return None
        raise LcuError(f"LCU {e.code} en {method} {path}: {e.read()[:200]}") from e


def game_id(match_id: str) -> int:
    """EUW1_7983109023 -> 7983109023."""
    return int(match_id.split("_", 1)[1])


def replay_metadata(match_id: str):
    """None si el plugin aun no tiene metadata local de esta partida."""
    return _request("GET", f"/lol-replays/v1/metadata/{game_id(match_id)}",
                    allow_404=True)


def create_replay_metadata(match_id: str, game_version: str, game_type: str,
                           queue_id: int) -> None:
    """El plugin exige crear la metadata local antes de descargar una partida
    que no está en el historial visible (verificado contra /help del LCU)."""
    gid = game_id(match_id)
    _request("POST",
             f"/lol-replays/v1/metadata/{gid}/create/gameVersion/{game_version}"
             f"/gameType/{game_type}/queueId/{queue_id}", body={})


def download_replay(match_id: str, match: dict | None = None,
                    timeout_s: float = 240.0) -> dict:
    """Crea metadata si falta, lanza la descarga y espera estado 'watch'."""
    gid = game_id(match_id)
    meta = replay_metadata(match_id)
    if meta is None:
        if match is None:
            raise LcuError(f"{match_id}: sin metadata local y sin mv5_match "
                           "para crearla (pasa el match dict)")
        info = match["info"]
        create_replay_metadata(match_id, info["gameVersion"],
                               info.get("gameType", "MATCHED_GAME"),
                               info.get("queueId", 420))
        meta = replay_metadata(match_id) or {}
    if meta.get("state") == "watch":
        return meta
    _request("POST", f"/lol-replays/v1/rofls/{gid}/download",
             body={"componentType": "replay-button_match-history"})
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        meta = replay_metadata(match_id) or {}
        state = meta.get("state")
        if state == "watch":
            return meta
        if state in ("incompatible", "missingOnServer", "unsupported", "error"):
            raise LcuError(f"replay {match_id} no disponible: {state} "
                           "(¿parche distinto al del cliente?)")
        time.sleep(3)
    raise LcuError(f"timeout descargando la replay {match_id} "
                   f"(ultimo estado: {meta.get('state')})")


def watch_replay(match_id: str) -> None:
    """Abre la replay en el cliente de juego (tarda ~1 min en cargar)."""
    _request("POST", f"/lol-replays/v1/rofls/{game_id(match_id)}/watch",
             body={"componentType": "replay-button_match-history"})
