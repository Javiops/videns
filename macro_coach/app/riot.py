"""Cliente Riot con límite de tasa y caché gzip. Misma disciplina que
build_tracker (reimplementado, no compartido): 1.3s entre peticiones,
95 req/2min, 404 -> None, 401/403 abortan con mensaje claro, 429 respeta
Retry-After. Respuestas archivadas en data/cache/riot/ y releídas de ahí."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import httpx

from .config import RIOT_API_KEY, CACHE, PLATFORM, REGIONAL

MIN_GAP_S = 1.3
WINDOW_S = 120.0
WINDOW_MAX = 95


class RiotError(RuntimeError):
    pass


class Riot:
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        self.key = api_key if api_key is not None else RIOT_API_KEY
        self.cache = Path(cache_dir) if cache_dir else CACHE / "riot"
        self.cache.mkdir(parents=True, exist_ok=True)
        self._times: list[float] = []
        self._client = httpx.Client(timeout=20)

    # --- caché ---
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()[:24]
        return self.cache / f"{h}.json.gz"

    def _cache_get(self, url: str):
        p = self._cache_path(url)
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _cache_put(self, url: str, data) -> None:
        with gzip.open(self._cache_path(url), "wt", encoding="utf-8") as fh:
            json.dump(data, fh)

    # --- tasa ---
    def _throttle(self):
        now = time.time()
        self._times = [t for t in self._times if now - t < WINDOW_S]
        if self._times and now - self._times[-1] < MIN_GAP_S:
            time.sleep(MIN_GAP_S - (now - self._times[-1]))
        if len(self._times) >= WINDOW_MAX:
            wait = WINDOW_S - (time.time() - self._times[0]) + 0.5
            if wait > 0:
                time.sleep(wait)
        self._times.append(time.time())

    def get(self, url: str, cacheable: bool = True):
        if cacheable:
            hit = self._cache_get(url)
            if hit is not None:
                return hit
        if not self.key:
            raise RiotError("RIOT_API_KEY vacía: copia .env.example a .env y pon la clave")
        for attempt in range(4):
            self._throttle()
            r = self._client.get(url, headers={"X-Riot-Token": self.key})
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise RiotError(f"{r.status_code} en {url} — clave caducada o sin permiso; "
                                "las claves de desarrollo caducan cada 24h")
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", "10")))
                continue
            if r.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if cacheable:
                self._cache_put(url, data)
            return data
        raise RiotError(f"agotados los reintentos en {url}")

    # --- endpoints ---
    def challenger_league(self, queue: str = "RANKED_SOLO_5x5"):
        return self.get(f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/"
                        f"challengerleagues/by-queue/{queue}", cacheable=False)

    def account_by_puuid(self, puuid: str):
        """{'puuid','gameName','tagLine'} o None. Cacheado: los renombres son
        raros dentro de una ventana de parche."""
        return self.get(f"https://{REGIONAL}.api.riotgames.com/riot/account/v1/"
                        f"accounts/by-puuid/{puuid}")

    def account_by_riot_id(self, game_name: str, tag_line: str):
        from urllib.parse import quote
        return self.get(f"https://{REGIONAL}.api.riotgames.com/riot/account/v1/"
                        f"accounts/by-riot-id/{quote(game_name)}/{quote(tag_line)}")

    def match_ids(self, puuid: str, count: int = 20, queue: int = 420,
                  start: int = 0):
        return self.get(f"https://{REGIONAL}.api.riotgames.com/lol/match/v5/matches/"
                        f"by-puuid/{puuid}/ids?queue={queue}&count={count}&start={start}",
                        cacheable=False)

    def match(self, match_id: str):
        return self.get(f"https://{REGIONAL}.api.riotgames.com/lol/match/v5/matches/{match_id}")

    def timeline(self, match_id: str):
        return self.get(f"https://{REGIONAL}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline")
