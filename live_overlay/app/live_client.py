"""Lectura del Live Client Data API que el cliente de League sirve en local.

https://developer.riotgames.com/docs/lol#game-client-api_live-client-data-api

Una sola llamada (`allgamedata`) trae activePlayer, allPlayers, events y
gameData. Sin partida el puerto rechaza la conexión y devolvemos None; el
certificado es autofirmado, de ahí verify=False.
"""

from __future__ import annotations

import httpx

from app.config import LIVE_CLIENT_URL


def fetch_snapshot(timeout: float = 2.0) -> dict | None:
    """El snapshot completo, o None si no hay partida o el API falla."""
    try:
        response = httpx.get(LIVE_CLIENT_URL, verify=False, timeout=timeout)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None
