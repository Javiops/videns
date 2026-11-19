"""Los dos trackers del overlay: oro de objetos por equipo y timers de Barón / Elder.

Entrada: el snapshot de `allgamedata` (app.live_client). Salida: un dict listo
para el panel. Solo guarda estado donde el API olvida: qué eventos ya se
procesaron, quién estaba vivo cuando murió Barón o Elder, y quién estaba
muerto en el poll anterior.

* **Oro de objetos**: valor de inventario por jugador. El wallet propio es
  exacto (`activePlayer.currentGold`); el resto es solo objetos, lo mismo que
  enseña el marcador. El valor sale de Data Dragon (`gold.total` x `count`),
  nunca del campo `price` del API, que es el coste de combinación (ver
  app/ddragon.py). Un id que no esté en el catálogo usa `price` y se marca
  como aproximado.

* **Objetivos** (solo en la Grieta del Invocador, mapa 11): buffs de Barón y
  Elder con portadores y cuenta atrás, más respawn de Barón y de dragón, a
  partir de los eventos `BaronKill` y `DragonKill`. El API no expone buffs,
  así que los portadores se derivan: miembros del equipo que mató, vivos en
  el primer poll en que aparece el evento (y no muertos en el poll anterior
  con temporizador que cubra el instante de la kill), y se quitan en cuanto
  `isDead` cambia. Una kill vista por primera vez más de LATE_START_S después
  de ocurrir (overlay abierto a mitad de partida) tiene portadores
  desconocidos; el timer de equipo y los respawn siguen siendo exactos porque
  la lista de eventos acumula toda la partida.

Nombres en los eventos: `KillerName` y `Assisters` son `riotIdGameName` cuando
el jugador lo tiene; los jugadores con nombre oculto llegan como
riotIdGameName "", riotId "#" y summonerName igual al campeón, y los eventos
usan ese summonerName (23 de 2.065 kills del corpus medido). El riotId con
tag nunca coincide. Torres y súbditos también aparecen como killer. Dos
jugadores pueden compartir riotIdGameName con tags distintos: un nombre
ambiguo no se adjudica; se resuelve por los asistentes o queda en "?".

Partidas seguidas: los EventID reinician a 0 en cada partida. Se detecta una
partida nueva por roster distinto, por reloj que retrocede, o porque un
EventID ya visto vuelve con otro evento u hora (conflicto de flujo).

Las duraciones son constantes del juego, no medidas de este proyecto: el API
no observa buffs. Los spawn y respawn se contrastaron con 15 partidas CLASSIC
capturadas (primera kill y mínimo intervalo entre kills). Revisar al cambiar
de temporada.
"""

from __future__ import annotations

import threading
from collections import defaultdict

TEAM_IDS = {"ORDER": 100, "CHAOS": 200}
SIDES = {100: "ORDER", 200: "CHAOS"}
LANES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")   # orden de las filas de oro cuando el API da posiciones
LANE_ORDER = {lane: i for i, lane in enumerate(LANES)}
SUMMONERS_RIFT = 11
RIFT_MODES = {"CLASSIC", "PRACTICETOOL"}   # respaldo cuando el snapshot no trae mapNumber

# --- constantes del juego (segundos) -----------------------------------------
BARON_FIRST_SPAWN_S = 20 * 60   # corpus: primera BaronKill a 22:26
BARON_RESPAWN_S = 6 * 60        # corpus: mínimo entre BaronKills 404 s (compatible, no ajustado)
BARON_BUFF_S = 180              # Mano de Barón (wiki); no observable por el API
DRAGON_FIRST_SPAWN_S = 5 * 60   # corpus: primera DragonKill a 5:41
DRAGON_RESPAWN_S = 5 * 60       # corpus: mínimo entre DragonKills 305 s (ajustado)
ELDER_RESPAWN_S = 6 * 60        # wiki; el corpus tiene 2 Elder, ningún par consecutivo
ELDER_BUFF_S = 150              # Aspecto del Dragón (wiki); no observable por el API
SOUL_DRAGONS = 4                # dragones elementales de un equipo antes de que todo spawn sea Elder
LATE_START_S = 5.0              # una kill más vieja que esto al verla por primera vez: portadores desconocidos
NEW_GAME_REWIND_S = 2.0         # el reloj retrocede más que esto: partida nueva

CONSTANTS = {
    "baron_first_spawn_s": BARON_FIRST_SPAWN_S,
    "baron_respawn_s": BARON_RESPAWN_S,
    "baron_buff_s": BARON_BUFF_S,
    "dragon_first_spawn_s": DRAGON_FIRST_SPAWN_S,
    "dragon_respawn_s": DRAGON_RESPAWN_S,
    "elder_respawn_s": ELDER_RESPAWN_S,
    "elder_buff_s": ELDER_BUFF_S,
    "soul_dragons": SOUL_DRAGONS,
}


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _game_time(snap: dict) -> float:
    return _num((snap.get("gameData") or {}).get("gameTime"), 0.0)


def _map_number(snap: dict) -> int | None:
    value = (snap.get("gameData") or {}).get("mapNumber")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_rift(snap: dict) -> bool:
    number = _map_number(snap)
    if number is not None:
        return number == SUMMONERS_RIFT
    return ((snap.get("gameData") or {}).get("gameMode") or "") in RIFT_MODES


def _team_id(player: dict | None) -> int | None:
    return TEAM_IDS.get((player or {}).get("team") or "")


def _position(player: dict) -> str:
    """Línea del jugador tal como la da el API (vacía en Practice Tool, personalizadas y modos sin líneas)."""
    lane = str(player.get("position") or "").upper()
    return lane if lane in LANE_ORDER else ""


def _player_key(player: dict) -> str:
    """Identidad estable dentro de una partida (los nombres ocultos llegan como riotId '#')."""
    ident = player.get("riotId") or player.get("summonerName") or ""
    return f"{player.get('team') or ''}|{player.get('championName') or ''}|{ident}"


def _display_name(player: dict) -> str:
    for field in ("riotIdGameName", "summonerName", "riotId"):
        value = str(player.get(field) or "").strip()
        if value and value != "#":
            return value
    return str(player.get("championName") or "")


def _name_index(players: list[dict]) -> dict[str, list[dict]]:
    """Cada nombre que puede aparecer en un evento -> jugadores que lo llevan (varios si hay colisión)."""
    index: dict[str, list[dict]] = defaultdict(list)
    for field in ("riotIdGameName", "summonerName", "riotId"):
        for player in players:
            name = player.get(field)
            if name and name != "#" and not any(p is player for p in index[name]):
                index[name].append(player)
    return index


def _unique(by_name: dict, name) -> dict | None:
    candidates = by_name.get(name or "") or []
    return candidates[0] if len(candidates) == 1 else None


def _team_of_name(by_name: dict, name) -> int | None:
    """Equipo de un nombre de evento; None si no está o si lo comparten jugadores de equipos distintos."""
    teams = {_team_id(p) for p in (by_name.get(name or "") or [])}
    teams.discard(None)
    return teams.pop() if len(teams) == 1 else None


def _icon(catalog, player: dict | None) -> str | None:
    """Id de imagen de Data Dragon del campeón, o None sin catálogo o sin coincidencia."""
    if catalog is None or not player:
        return None
    try:
        return catalog.champion_image_id(player.get("rawChampionName"), player.get("championName"))
    except AttributeError:
        return None


def item_gold(items: list[dict] | None, catalog) -> tuple[int, bool]:
    """(valor del inventario, ¿aproximado?). Catálogo None: todo por `price`, aproximado."""
    total = 0
    approx = False
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("itemID") or 0)
        except (TypeError, ValueError):
            item_id = 0
        if not item_id:
            continue
        try:
            count = max(int(item.get("count") or 1), 1)
        except (TypeError, ValueError):
            count = 1
        value = catalog.total(item_id) if catalog is not None else None
        if value is None:
            # fuera del catálogo (desfase de parche o sin catálogo): el coste base es un suelo, marcado
            approx = True
            try:
                value = int(item.get("price") or 0)
            except (TypeError, ValueError):
                value = 0
        total += value * count
    return total, approx


class OverlayTracker:
    """Estado mínimo: eventos ya vistos, portadores de buff y muertes del poll anterior."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._roster: tuple | None = None
        self._last_game_time: float | None = None
        self._seen_events: dict = {}          # EventID -> (EventName, EventTime)
        self._buffs: list[dict] = []
        self._last_baron_kill: float | None = None
        self._last_dragon_kill: float | None = None
        self._last_dragon_was_elder = False
        self._dragons = {100: 0, 200: 0}
        self._dragons_unknown = 0             # kills de dragón cuyo equipo no se pudo resolver
        self._game_over = False
        self._prev_dead: dict[str, tuple[float, float]] = {}   # key -> (game_time del poll, respawnTimer)

    # -- público ------------------------------------------------------------------

    def observe(self, snap: dict, catalog) -> dict:
        with self._lock:
            return self._observe(snap, catalog)

    # -- interno ------------------------------------------------------------------

    def _observe(self, snap: dict, catalog) -> dict:
        players = [p for p in (snap.get("allPlayers") or []) if isinstance(p, dict)]
        game_time = _game_time(snap)
        events = [e for e in ((snap.get("events") or {}).get("Events") or []) if isinstance(e, dict)]
        roster = tuple(sorted(_player_key(p) for p in players))
        rewound = self._last_game_time is not None and game_time < self._last_game_time - NEW_GAME_REWIND_S
        if (roster and self._roster and roster != self._roster) or rewound or self._stream_conflict(events):
            self._reset()
        if roster:
            self._roster = roster
        self._last_game_time = game_time

        by_name = _name_index(players)
        by_key = {_player_key(p): p for p in players}
        self._ingest_events(events, players, by_name, game_time)
        self._update_holders(by_key, game_time)
        self._prev_dead = {
            _player_key(p): (game_time, _num(p.get("respawnTimer"), 0.0))
            for p in players if p.get("isDead")
        }

        active = snap.get("activePlayer") or {}
        me = _unique(by_name, active.get("riotId")) or _unique(by_name, active.get("riotIdGameName"))
        rift = _is_rift(snap)

        return {
            "in_game": True,
            "game_time_s": game_time,
            "game_mode": (snap.get("gameData") or {}).get("gameMode"),
            "map_number": _map_number(snap),
            "rift": rift,
            "game_over": self._game_over,
            "my_team": _team_id(me),
            "catalog_version": getattr(catalog, "version", None),
            "gold": self._gold_payload(players, me, active, catalog),
            "objectives": self._objectives_payload(game_time, by_key, catalog) if rift else None,
            "constants": CONSTANTS,
        }

    def _stream_conflict(self, events: list[dict]) -> bool:
        """Un EventID ya visto con otro evento u hora, o menos eventos que antes: es otra partida."""
        if not self._seen_events:
            return False
        if len(events) < len(self._seen_events):
            return True
        for event in events:
            seen = self._seen_events.get(event.get("EventID"))
            if seen is not None and seen != (event.get("EventName"), round(_num(event.get("EventTime")), 3)):
                return True
        return False

    def _ingest_events(self, events: list[dict], players: list[dict], by_name: dict, game_time: float) -> None:
        for event in sorted(events, key=lambda e: _num(e.get("EventID"), 0)):
            event_id = event.get("EventID")
            if event_id in self._seen_events:
                continue
            name = event.get("EventName")
            when = _num(event.get("EventTime"), 0.0)
            self._seen_events[event_id] = (name, round(when, 3))
            if name == "GameEnd":
                self._game_over = True
            elif name == "BaronKill":
                self._last_baron_kill = when
                self._register_buff("baron", event, when, BARON_BUFF_S, players, by_name, game_time)
            elif name == "DragonKill":
                self._last_dragon_kill = when
                is_elder = str(event.get("DragonType") or "").lower() == "elder"
                self._last_dragon_was_elder = is_elder
                if is_elder:
                    self._register_buff("elder", event, when, ELDER_BUFF_S, players, by_name, game_time)
                else:
                    team = self._killer_team(event, by_name)
                    if team in self._dragons:
                        self._dragons[team] += 1
                    else:
                        self._dragons_unknown += 1

    def _killer_team(self, event: dict, by_name: dict) -> int | None:
        team = _team_of_name(by_name, event.get("KillerName"))
        if team is None:
            # torres, súbditos, mascotas o un nombre repetido en los dos equipos: un asistente resuelve
            for assister in event.get("Assisters") or []:
                team = _team_of_name(by_name, assister)
                if team is not None:
                    break
        return team

    def _dead_at(self, player: dict, when: float) -> bool:
        """Muerto en el instante de la kill según el poll anterior (respawnTimer cubre `when`)."""
        prev = self._prev_dead.get(_player_key(player))
        return prev is not None and prev[0] + prev[1] > when

    def _register_buff(
        self, kind: str, event: dict, when: float, duration: float,
        players: list[dict], by_name: dict, game_time: float,
    ) -> None:
        team = self._killer_team(event, by_name)
        holders: list[str] | None
        if team is None or game_time - when > LATE_START_S:
            holders = None
        else:
            holders = [
                _player_key(p) for p in players
                if _team_id(p) == team and not p.get("isDead") and not self._dead_at(p, when)
            ]
        self._buffs.append({
            "kind": kind,
            "team": team,
            "killed_at": when,
            "ends_at": when + duration,
            "holders": holders,
            "stolen": str(event.get("Stolen")).lower() == "true",
        })

    def _update_holders(self, by_key: dict, game_time: float) -> None:
        kept = []
        for buff in self._buffs:
            if buff["ends_at"] <= game_time:
                continue
            if buff["holders"] is not None:
                buff["holders"] = [
                    key for key in buff["holders"]
                    if not (by_key.get(key) or {}).get("isDead")
                ]
            kept.append(buff)
        self._buffs = kept

    def _gold_payload(self, players: list[dict], me: dict | None, active: dict, catalog) -> dict:
        wallet = int(_num(active.get("currentGold"), 0.0))
        teams: dict[int, list[dict]] = {100: [], 200: []}
        for player in players:
            team = _team_id(player)
            if team not in teams:
                continue
            value, approx = item_gold(player.get("items"), catalog)
            is_me = player is me
            teams[team].append({
                "name": _display_name(player),
                "champion": player.get("championName") or "",
                "icon": _icon(catalog, player),
                "position": _position(player),
                "level": player.get("level"),
                "dead": bool(player.get("isDead")),
                "item_gold": value,
                "approx": approx,
                "is_me": is_me,
                "wallet": wallet if is_me else None,
            })
        totals = {team: sum(p["item_gold"] for p in members) for team, members in teams.items()}
        approx_any = any(p["approx"] for members in teams.values() for p in members)
        lead_team = None
        if totals[100] != totals[200]:
            lead_team = 100 if totals[100] > totals[200] else 200
        return {
            "teams": [
                {
                    "team_id": team,
                    "side": SIDES[team],
                    "item_gold": totals[team],
                    "approx": any(p["approx"] for p in teams[team]),
                    # por línea (top, jungla, mid, bot, support) para que cada fila enfrente a los rivales
                    # de la misma línea; sin posiciones, por valor de objetos
                    "players": sorted(teams[team], key=lambda p: (LANE_ORDER.get(p["position"], 9), -p["item_gold"])),
                }
                for team in (100, 200)
            ],
            "lead": {"team_id": lead_team, "amount": abs(totals[100] - totals[200]), "approx": approx_any},
            "wallet": wallet if me is not None else None,
        }

    def _objectives_payload(self, game_time: float, by_key: dict, catalog=None) -> dict:
        def holder_entries(keys: list[str] | None) -> list[dict] | None:
            if keys is None:
                return None
            entries = []
            for key in keys:
                player = by_key.get(key) or {}
                champion = player.get("championName") or key.split("|")[1]
                entries.append({
                    "name": _display_name(player) if player else champion,
                    "champion": champion,
                    "icon": _icon(catalog, player),
                })
            return entries

        if self._last_baron_kill is None:
            baron_spawn = float(BARON_FIRST_SPAWN_S)
        else:
            baron_spawn = self._last_baron_kill + BARON_RESPAWN_S

        next_is_elder = self._last_dragon_was_elder or max(self._dragons.values()) >= SOUL_DRAGONS
        # con kills sin equipo resuelto, un alma pudo completarse sin que lo sepamos
        uncertain = (not next_is_elder and self._dragons_unknown > 0
                     and max(self._dragons.values()) + self._dragons_unknown >= SOUL_DRAGONS)
        if self._last_dragon_kill is None:
            dragon_spawn = float(DRAGON_FIRST_SPAWN_S)
        else:
            dragon_spawn = self._last_dragon_kill + (ELDER_RESPAWN_S if next_is_elder else DRAGON_RESPAWN_S)

        return {
            "baron": {
                "state": "up" if game_time >= baron_spawn else "spawning",
                "spawns_at": baron_spawn,
            },
            "dragon": {
                "kind": "elder" if next_is_elder else "dragon",
                "state": "up" if game_time >= dragon_spawn else "spawning",
                "spawns_at": dragon_spawn,
                "uncertain": uncertain,
                "counts": {"100": self._dragons[100], "200": self._dragons[200], "unknown": self._dragons_unknown},
            },
            "buffs": [
                {
                    "kind": buff["kind"],
                    "team_id": buff["team"],
                    "side": SIDES.get(buff["team"]),
                    "killed_at": buff["killed_at"],
                    "ends_at": buff["ends_at"],
                    "holders": holder_entries(buff["holders"]),
                    "stolen": buff["stolen"],
                }
                for buff in self._buffs
            ],
        }
