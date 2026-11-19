"""Riot ID -> compiled public profile, the read-only op.gg-shaped lookup.

This module only *describes* what a player did: recent matches, rank, and the
aggregates you can compute straight from Match-V5. It deliberately never calls
the shop model. Ranking scores are not calibrated confidence and a profile page
is not a shop decision, so advice has no business here.

Every Riot call goes through one shared, throttled client behind a lock: a web
page must not race an ingest run for the same rate-limit budget.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache

import httpx

from app.config import CACHE_DIR
from app.db import db
from app.ddragon import DDRAGON_VERSIONS, latest_version, load_json
from app.riot import RiotClient, RiotError

DDRAGON_CDN = "https://ddragon.leagueoflegends.com/cdn"
CDRAGON_GAME_DATA = (
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default"
)

# platform host, account-v1 routing, match-v5 routing, display label.
# Account routing is replicated globally (americas/asia/europe only); match
# routing later added `sea`, and the two disagree for every SEA shard. Keeping
# both columns is the whole point of this table.
REGIONS: dict[str, dict[str, str]] = {
    "euw": {"platform": "euw1", "account": "europe", "match": "europe", "label": "EUW"},
    "eune": {"platform": "eun1", "account": "europe", "match": "europe", "label": "EUNE"},
    "na": {"platform": "na1", "account": "americas", "match": "americas", "label": "NA"},
    "kr": {"platform": "kr", "account": "asia", "match": "asia", "label": "KR"},
    "br": {"platform": "br1", "account": "americas", "match": "americas", "label": "BR"},
    "lan": {"platform": "la1", "account": "americas", "match": "americas", "label": "LAN"},
    "las": {"platform": "la2", "account": "americas", "match": "americas", "label": "LAS"},
    "jp": {"platform": "jp1", "account": "asia", "match": "asia", "label": "JP"},
    "tr": {"platform": "tr1", "account": "europe", "match": "europe", "label": "TR"},
    "ru": {"platform": "ru", "account": "europe", "match": "europe", "label": "RU"},
    "me": {"platform": "me1", "account": "europe", "match": "europe", "label": "ME"},
    "oce": {"platform": "oc1", "account": "americas", "match": "sea", "label": "OCE"},
    "sg": {"platform": "sg2", "account": "asia", "match": "sea", "label": "SG"},
    "ph": {"platform": "ph2", "account": "asia", "match": "sea", "label": "PH"},
    "th": {"platform": "th2", "account": "asia", "match": "sea", "label": "TH"},
    "tw": {"platform": "tw2", "account": "asia", "match": "sea", "label": "TW"},
    "vn": {"platform": "vn2", "account": "asia", "match": "sea", "label": "VN"},
}

QUEUE_NAMES = {
    100: "ARAM",
    400: "Normal Draft",
    420: "Ranked Solo",
    430: "Normal Blind",
    440: "Ranked Flex",
    450: "ARAM",
    490: "Quickplay",
    700: "Clash",
    720: "ARAM Clash",
    830: "Co-op vs IA",
    840: "Co-op vs IA",
    850: "Co-op vs IA",
    900: "ARURF",
    1020: "One for All",
    1300: "Nexus Blitz",
    1700: "Arena",
    1710: "Arena",
    1750: "Arena",
    1810: "Swarm",
    1820: "Swarm",
    1830: "Swarm",
    1840: "Swarm",
    1900: "URF",
    2300: "Brawl",
    2400: "ARAM Mayhem",
}

# Queue ids churn every rotation; `gameMode` on the DTO does not. When an id is
# unmapped the mode is a better label than a shrug.
MODE_NAMES = {
    "CLASSIC": "Grieta",
    "ARAM": "ARAM",
    "CHERRY": "Arena",
    "URF": "URF",
    "NEXUSBLITZ": "Nexus Blitz",
    "ONEFORALL": "One for All",
    "ULTBOOK": "Ultimate Spellbook",
    "STRAWBERRY": "Swarm",
    "TUTORIAL": "Tutorial",
}

# Arena splits the lobby into small subteams but still reports teamId 100/200,
# so that field says nothing about who you played with — `playerSubteamId` does,
# and the subteam size varies by rotation, so it is counted, never assumed.
# Placement is the result there; a win/loss flag over half the lobby is not.
ARENA_MODE = "CHERRY"

ROLE_LABELS = {
    "TOP": "Top",
    "JUNGLE": "Jungla",
    "MIDDLE": "Medio",
    "BOTTOM": "ADC",
    "UTILITY": "Support",
}

# Ranked solo first: it is the queue the ladder and the corpus are built on.
QUEUE_TYPE_LABELS = {
    "RANKED_SOLO_5x5": "Solo/Duo",
    "RANKED_FLEX_SR": "Flex 5v5",
    "RANKED_FLEX_TT": "Flex 3v3",
}

PROFILE_TTL_S = 180.0
# An explicit Update may bypass the profile cache, but not more often than this:
# a held-down button must not spend the shared Riot budget.
PROFILE_REFRESH_S = 30.0
CORPUS_TTL_S = 600.0
VERSION_TTL_S = 3600.0
AUGMENTS_RETRY_S = 300.0
ACCOUNT_TTL_S = 3600.0
# Riot's id list is newest-first and pages by offset; past ~1000 it is not
# worth the calls for a "Load more" button.
MATCH_PAGE_MAX_START = 980
# A season sync walks one match per throttled call (1.3 s on a dev key), so the
# cap bounds a single player at about half an hour of background budget.
SEASON_MAX_GAMES = 1500
SEASON_RESYNC_S = 600.0
SEASON_RETRY_S = 60.0
SEASON_JOBS_MAX = 64

_client: RiotClient | None = None
_client_lock = threading.Lock()
_profile_cache: dict[tuple[str, str, int], tuple[float, dict]] = {}
_totals_lock = threading.Lock()
_totals: tuple[float, dict] = (0.0, {})
_version: tuple[float, str] = (0.0, "")
_augments_failed_at: dict[str, float] = {}
_account_cache: dict[tuple[str, str], tuple[float, dict]] = {}

# Season sync state. One daemon worker drains the queue; every Riot call it
# makes goes through the same client and `_client_lock` as a lookup, taken per
# call, so an interactive lookup waits at most one request behind it.
_season_lock = threading.Lock()
_season_jobs: dict[tuple[str, str], dict] = {}
_season_queue: deque[tuple[str, str]] = deque()
_season_wake = threading.Event()
_season_worker: threading.Thread | None = None
_season_epoch = 0


class ProfileError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _get_client() -> RiotClient:
    global _client
    if _client is None:
        try:
            _client = RiotClient()
        except RiotError as exc:
            raise ProfileError(str(exc), 503) from exc
    return _client


def parse_riot_id(raw: str) -> tuple[str, str]:
    """Split `Name#TAG`. A missing tag is an error, not a guess."""
    text = (raw or "").strip().replace("＃", "#")
    if "#" not in text:
        raise ProfileError("Escribe el Riot ID completo, con etiqueta: Nombre#TAG.")
    name, _, tag = text.rpartition("#")
    name, tag = name.strip(), tag.strip()
    if not name or not tag:
        raise ProfileError("Escribe el Riot ID completo, con etiqueta: Nombre#TAG.")
    if len(name) > 16 or len(tag) > 5:
        raise ProfileError("Ese Riot ID no tiene una forma valida.")
    return name, tag


def resolve_region(key: str | None) -> dict[str, str]:
    region = REGIONS.get((key or "euw").strip().lower())
    if region is None:
        raise ProfileError(f"Region desconocida: {key!r}.")
    return region


def display_version() -> str:
    """Latest Data Dragon build, for icon URLs only.

    `ddragon.latest_version()` reads a pinned on-disk cache on purpose: the
    pipeline must never reinterpret an old match with new static data. Art is
    not interpretation, and a champion released this patch has no image under
    last patch's build, so the page refreshes its own copy on an hourly TTL and
    falls back to the pinned version whenever the CDN is unreachable.
    """
    global _version
    stamped, version = _version
    if version and time.monotonic() - stamped < VERSION_TTL_S:
        return version
    try:
        response = httpx.get(DDRAGON_VERSIONS, timeout=10.0)
        response.raise_for_status()
        version = str(response.json()[0])
    except Exception:
        version = latest_version()
    _version = (time.monotonic(), version)
    return version


@lru_cache(maxsize=4)
def _champions(version: str) -> dict[int, dict[str, str]]:
    """championId -> Data Dragon id and display name.

    The match DTO's `championName` is not the art filename and not a display
    name: Riot ships "FiddleSticks" against a `Fiddlesticks.png`, and calls
    Wukong "MonkeyKing". Resolving both from the numeric id off champion.json
    is the only spelling that holds.
    """
    data = load_json(
        CACHE_DIR / f"champion-{version}.json",
        f"{DDRAGON_CDN}/{version}/data/en_US/champion.json",
    )
    return {
        int(champ["key"]): {"id": champ["id"], "name": champ["name"]}
        for champ in (data.get("data") or {}).values()
    }


def _champion(champions: dict[int, dict[str, str]], participant: dict, version: str) -> dict:
    """Name and icon for one participant, falling back to the DTO spelling.

    A champion released after this static payload is not in the map; showing
    Riot's internal name with a broken icon beats showing nothing.
    """
    fallback = participant.get("championName") or ""
    champ = champions.get(int(participant.get("championId") or 0))
    key = champ["id"] if champ else fallback
    return {
        "name": champ["name"] if champ else fallback,
        "icon": f"{DDRAGON_CDN}/{version}/img/champion/{key}.png" if key else "",
    }


@lru_cache(maxsize=4)
def _summoner_spells(version: str) -> dict[int, dict[str, str]]:
    data = load_json(
        CACHE_DIR / f"summoner-{version}.json",
        f"{DDRAGON_CDN}/{version}/data/en_US/summoner.json",
    )
    return {
        int(spell["key"]): {
            "name": spell.get("name") or "",
            "icon": f"{DDRAGON_CDN}/{version}/img/spell/{spell['image']['full']}",
        }
        for spell in (data.get("data") or {}).values()
    }


@lru_cache(maxsize=4)
def _runes(version: str) -> dict[int, dict[str, str]]:
    """Flat perk-id -> name/icon index over trees, slots and keystones."""
    data = load_json(
        CACHE_DIR / f"runesReforged-{version}.json",
        f"{DDRAGON_CDN}/{version}/data/en_US/runesReforged.json",
    )
    index: dict[int, dict[str, str]] = {}
    for tree in data:
        # Rune art is version-less on the CDN: /cdn/img/<icon path>.
        index[int(tree["id"])] = {
            "name": tree.get("name") or "",
            "icon": f"{DDRAGON_CDN}/img/{tree['icon']}",
        }
        for slot in tree.get("slots") or []:
            for rune in slot.get("runes") or []:
                index[int(rune["id"])] = {
                    "name": rune.get("name") or "",
                    "icon": f"{DDRAGON_CDN}/img/{rune['icon']}",
                }
    return index


def _augments(version: str) -> dict[int, dict[str, str]]:
    """Augment id -> name/icon/rarity, for Arena and ARAM Mayhem.

    Data Dragon does not ship augments; CommunityDragon does. Augments are
    decoration: if the list cannot be fetched the page shows none, and the
    lookup itself never fails because of it. A failure is not cached (the next
    success still lands), but it is remembered for AUGMENTS_RETRY_S so an
    outage does not add the fetch timeout to every lookup.
    """
    failed = _augments_failed_at.get(version)
    if failed is not None and time.monotonic() - failed < AUGMENTS_RETRY_S:
        return {}
    try:
        return _augment_index(version)
    except Exception:
        _augments_failed_at[version] = time.monotonic()
        return {}


@lru_cache(maxsize=4)
def _augment_index(version: str) -> dict[int, dict[str, str]]:
    # The cache file is keyed by the Data Dragon version so a new patch
    # refetches the list. Raises on failure: lru_cache never stores those.
    data = load_json(
        CACHE_DIR / f"cherry-augments-{version}.json",
        f"{CDRAGON_GAME_DATA}/v1/cherry-augments.json",
    )
    index: dict[int, dict[str, str]] = {}
    for augment in data if isinstance(data, list) else []:
        path = augment.get("augmentSmallIconPath") or ""
        # "/lol-game-data/assets/ASSETS/UX/..." is served lower-cased under the plugin root.
        tail = path.split("/lol-game-data/assets/", 1)[-1].lower() if path else ""
        index[int(augment.get("id") or 0)] = {
            "name": augment.get("nameTRA") or "",
            "icon": f"{CDRAGON_GAME_DATA}/{tail}" if tail else "",
            "rarity": (augment.get("rarity") or "").removeprefix("k").lower(),
        }
    return index


def _augment_list(participant: dict, augments: dict[int, dict[str, str]]) -> list[dict]:
    ids = [int(participant.get(f"playerAugment{n}") or 0) for n in range(1, 7)]
    return [
        {"id": i, **augments.get(i, {"name": "", "icon": "", "rarity": ""})}
        for i in ids
        if i
    ]


def _kda_ratio(kills: int, deaths: int, assists: int) -> float:
    # A deathless game has no ratio; dividing by max(deaths, 1) is the usual
    # convention and the UI labels that case "Perfecta" instead of printing it
    # as if it were a real quotient.
    return round((kills + assists) / max(deaths, 1), 2)


def _perk_ids(participant: dict) -> tuple[int, int]:
    """Keystone perk id and secondary tree id, 0 where Riot left them out."""
    styles = (participant.get("perks") or {}).get("styles") or []
    keystone_id = 0
    secondary_id = 0
    if styles:
        selections = styles[0].get("selections") or []
        if selections:
            keystone_id = int(selections[0].get("perk") or 0)
    if len(styles) > 1:
        secondary_id = int(styles[1].get("style") or 0)
    return keystone_id, secondary_id


def _board_entry(
    p: dict,
    me: dict,
    version: str,
    spells: dict[int, dict[str, str]],
    runes: dict[int, dict[str, str]],
    champions: dict[int, dict[str, str]],
    augments: dict[int, dict[str, str]],
) -> dict:
    """One lobby slot, complete enough to draw the expanded scoreboard.

    Items stay bare ids here: ten players times seven slots of full CDN URLs
    would dominate the payload, and the page already has `ddragon_version`
    to build them.
    """
    champion = _champion(champions, p, version)
    keystone_id, secondary_id = _perk_ids(p)
    return {
        "name": p.get("riotIdGameName") or p.get("summonerName") or "",
        "tag": p.get("riotIdTagline") or "",
        "champion": champion["name"],
        "icon": champion["icon"],
        "team_id": int(p.get("teamId") or 0),
        # Arena's real team; 0 everywhere else.
        "subteam": int(p.get("playerSubteamId") or 0),
        "is_me": p.get("puuid") == me.get("puuid"),
        "participant_id": int(p.get("participantId") or 0),
        "role": ROLE_LABELS.get(p.get("teamPosition") or "", ""),
        "win": bool(p.get("win")),
        "placement": int(p.get("placement") or 0),
        "level": int(p.get("champLevel") or 0),
        "kills": int(p.get("kills") or 0),
        "deaths": int(p.get("deaths") or 0),
        "assists": int(p.get("assists") or 0),
        "cs": int(p.get("totalMinionsKilled") or 0) + int(p.get("neutralMinionsKilled") or 0),
        "gold": int(p.get("goldEarned") or 0),
        "damage": int(p.get("totalDamageDealtToChampions") or 0),
        "damage_taken": int(p.get("totalDamageTaken") or 0),
        "vision": int(p.get("visionScore") or 0),
        "wards_placed": int(p.get("wardsPlaced") or 0),
        "wards_killed": int(p.get("wardsKilled") or 0),
        "control_wards": int(p.get("visionWardsBoughtInGame") or 0),
        "items": [int(p.get(f"item{i}") or 0) for i in range(6)],
        "trinket": int(p.get("item6") or 0),
        "spells": [spells.get(int(p.get(f"summoner{n}Id") or 0)) for n in (1, 2)],
        "keystone": runes.get(keystone_id),
        "secondary_tree": runes.get(secondary_id),
        "augments": _augment_list(p, augments),
    }


def _sides(info: dict) -> list[dict]:
    """Per-team result and objective counts for the scoreboard header."""
    participants = info.get("participants") or []
    out = []
    for team in info.get("teams") or []:
        team_id = int(team.get("teamId") or 0)
        members = [p for p in participants if int(p.get("teamId") or 0) == team_id]
        objectives = team.get("objectives") or {}
        out.append({
            "team_id": team_id,
            "win": bool(team.get("win")),
            "kills": sum(int(p.get("kills") or 0) for p in members),
            "gold": sum(int(p.get("goldEarned") or 0) for p in members),
            "objectives": {
                name: int((objectives.get(name) or {}).get("kills") or 0)
                for name in (
                    "tower", "inhibitor", "dragon", "baron", "riftHerald", "horde", "atakhan",
                )
            },
        })
    out.sort(key=lambda t: t["team_id"])
    return out


def _participant_row(
    match: dict,
    me: dict,
    version: str,
    spells: dict[int, dict[str, str]],
    runes: dict[int, dict[str, str]],
    champions: dict[int, dict[str, str]],
    augments: dict[int, dict[str, str]],
) -> dict:
    info = match.get("info") or {}
    participants = info.get("participants") or []
    duration_s = int(info.get("gameDuration") or 0)
    minutes = max(duration_s / 60.0, 1 / 60.0)

    mode = info.get("gameMode") or ""
    arena = mode == ARENA_MODE
    if arena:
        subteam = me.get("playerSubteamId")
        side = [p for p in participants if p.get("playerSubteamId") == subteam]
        subteams = len({p.get("playerSubteamId") for p in participants})
    else:
        side = [p for p in participants if p.get("teamId") == me.get("teamId")]
        subteams = 0

    team_kills = sum(int(p.get("kills") or 0) for p in side)
    kills = int(me.get("kills") or 0)
    deaths = int(me.get("deaths") or 0)
    assists = int(me.get("assists") or 0)
    cs = int(me.get("totalMinionsKilled") or 0) + int(me.get("neutralMinionsKilled") or 0)

    keystone_id, secondary_id = _perk_ids(me)

    champion = _champion(champions, me, version)
    # Item identity stays numeric: the CDN serves art by id, so nothing here
    # reinterprets an older match with today's static data.
    items = [int(me.get(f"item{i}") or 0) for i in range(6)]
    trinket = int(me.get("item6") or 0)

    return {
        "match_id": (match.get("metadata") or {}).get("matchId"),
        "participant_id": int(me.get("participantId") or 0),
        "queue_id": int(info.get("queueId") or 0),
        "queue": QUEUE_NAMES.get(
            int(info.get("queueId") or 0), MODE_NAMES.get(mode, "Otra cola")
        ),
        "mode": mode,
        "arena": arena,
        "placement": int(me.get("placement") or 0) if arena else 0,
        "teams": subteams,
        "patch": ".".join((info.get("gameVersion") or "").split(".")[:2]),
        "ended_at_ms": int(info.get("gameEndTimestamp") or info.get("gameCreation") or 0),
        "duration_s": duration_s,
        "win": bool(me.get("win")),
        # A remake is not a loss. Riot flags it on the participant, which is
        # more reliable than inferring it from a short game duration.
        "remake": bool(me.get("gameEndedInEarlySurrender")),
        "champion": champion["name"],
        "champion_id": int(me.get("championId") or 0),
        "champion_icon": champion["icon"],
        "level": int(me.get("champLevel") or 0),
        "role": ROLE_LABELS.get(me.get("teamPosition") or "", ""),
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda": _kda_ratio(kills, deaths, assists),
        "kill_participation": round(100 * (kills + assists) / team_kills, 1) if team_kills else 0.0,
        "cs": cs,
        # Arena has no lane phase: a CS line there would be noise, not farming.
        "cs_per_min": 0.0 if arena else round(cs / minutes, 1),
        "gold": int(me.get("goldEarned") or 0),
        "damage": int(me.get("totalDamageDealtToChampions") or 0),
        "vision": int(me.get("visionScore") or 0),
        "wards_placed": int(me.get("wardsPlaced") or 0),
        "wards_killed": int(me.get("wardsKilled") or 0),
        "control_wards": int(me.get("visionWardsBoughtInGame") or 0),
        "damage_taken": int(me.get("totalDamageTaken") or 0),
        "multikill": int(me.get("largestMultiKill") or 0),
        "items": [
            {"id": i, "icon": f"{DDRAGON_CDN}/{version}/img/item/{i}.png" if i else None}
            for i in items
        ],
        "trinket": {
            "id": trinket,
            "icon": f"{DDRAGON_CDN}/{version}/img/item/{trinket}.png" if trinket else None,
        },
        "spells": [spells.get(int(me.get(f"summoner{n}Id") or 0)) for n in (1, 2)],
        "keystone": runes.get(keystone_id),
        "secondary_tree": runes.get(secondary_id),
        "subteam": int(me.get("playerSubteamId") or 0),
        "augments": _augment_list(me, augments),
        # The full lobby in every mode. In Arena that is all the subteams: the
        # page groups them by `subteam` and orders them by placement.
        "board": [
            _board_entry(p, me, version, spells, runes, champions, augments)
            for p in participants
        ],
        # Arena has no towers or dragons, and its two "teams" are not sides.
        "sides": [] if arena else _sides(info),
    }


def _aggregate(rows: list[dict]) -> dict:
    """Totals over the fetched window only - never over a career.

    Arena is held out of the averages. Its kill participation is a share of a
    small subteam rather than of five, and its CS is not farming, so folding it
    into the same mean as a Rift game would produce a number that looks right
    and means nothing. A player whose whole window is Arena still gets a
    summary, flagged as such, instead of an empty card.
    """
    counted = [r for r in rows if not r["remake"]]
    rift = [r for r in counted if not r["arena"]]
    basis = rift or counted
    games = len(basis)
    wins = sum(1 for r in basis if r["win"])
    kills = sum(r["kills"] for r in basis)
    deaths = sum(r["deaths"] for r in basis)
    assists = sum(r["assists"] for r in basis)
    arena_only = not rift and bool(counted)
    counted_ids = {id(r) for r in basis}
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "winrate": round(100 * wins / games, 1) if games else 0.0,
        "kills_avg": round(kills / games, 1) if games else 0.0,
        "deaths_avg": round(deaths / games, 1) if games else 0.0,
        "assists_avg": round(assists / games, 1) if games else 0.0,
        "kda": _kda_ratio(kills, deaths, assists),
        "cs_per_min": (
            0.0 if arena_only or not games
            else round(sum(r["cs_per_min"] for r in basis) / games, 1)
        ),
        "kill_participation": (
            0.0 if arena_only or not games
            else round(sum(r["kill_participation"] for r in basis) / games, 1)
        ),
        "basis": "arena" if arena_only else "rift",
        "excluded_arena": len(counted) - len(rift) if rift else 0,
        # The strip covers the whole window, so it carries which games the
        # averages above were computed from. Ten pills over a "last 6" headline
        # would otherwise invite the reader to recount and get a different
        # record than the one stated.
        "form": [
            {
                "result": "R" if r["remake"] else "V" if r["win"] else "D",
                "counted": id(r) in counted_ids,
                "arena": r["arena"],
                "placement": r["placement"],
            }
            for r in rows
        ],
    }


def _by_champion(rows: list[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for row in rows:
        if row["remake"]:
            continue
        bucket = buckets.setdefault(
            row["champion"],
            {
                # Reuse the row's resolved icon: a display name is not a
                # filename, and rebuilding the URL from it breaks Fiddlesticks.
                "champion": row["champion"],
                "icon": row["champion_icon"],
                "games": 0,
                "wins": 0,
                "kills": 0,
                "deaths": 0,
                "assists": 0,
                "cs_per_min": 0.0,
                "cs_games": 0,
            },
        )
        bucket["games"] += 1
        bucket["wins"] += int(row["win"])
        bucket["kills"] += row["kills"]
        bucket["deaths"] += row["deaths"]
        bucket["assists"] += row["assists"]
        if not row["arena"]:
            bucket["cs_per_min"] += row["cs_per_min"]
            bucket["cs_games"] += 1
    out = []
    for bucket in buckets.values():
        games = bucket["games"]
        out.append({
            "champion": bucket["champion"],
            "icon": bucket["icon"],
            "games": games,
            "wins": bucket["wins"],
            "losses": games - bucket["wins"],
            "winrate": round(100 * bucket["wins"] / games, 1),
            "kda": _kda_ratio(bucket["kills"], bucket["deaths"], bucket["assists"]),
            "cs_per_min": (
                round(bucket["cs_per_min"] / bucket["cs_games"], 1) if bucket["cs_games"] else 0.0
            ),
        })
    out.sort(key=lambda c: (-c["games"], -c["winrate"]))
    return out


def _by_role(rows: list[dict]) -> list[dict]:
    buckets: dict[str, int] = {}
    for row in rows:
        if row["remake"] or not row["role"]:
            continue
        buckets[row["role"]] = buckets.get(row["role"], 0) + 1
    total = sum(buckets.values())
    out = [
        {"role": role, "games": games, "share": round(100 * games / total)}
        for role, games in buckets.items()
    ]
    out.sort(key=lambda r: -r["games"])
    return out


def _ranks(entries: list[dict]) -> list[dict]:
    out = []
    for entry in entries:
        queue_type = entry.get("queueType") or ""
        if queue_type not in QUEUE_TYPE_LABELS:
            continue
        wins = int(entry.get("wins") or 0)
        losses = int(entry.get("losses") or 0)
        played = wins + losses
        out.append({
            "queue": QUEUE_TYPE_LABELS[queue_type],
            "queue_type": queue_type,
            "tier": (entry.get("tier") or "").title(),
            "division": entry.get("rank") or "",
            "lp": int(entry.get("leaguePoints") or 0),
            "wins": wins,
            "losses": losses,
            "winrate": round(100 * wins / played, 1) if played else 0.0,
            "hot_streak": bool(entry.get("hotStreak")),
            "inactive": bool(entry.get("inactive")),
        })
    out.sort(key=lambda r: r["queue_type"] != "RANKED_SOLO_5x5")
    return out


def _corpus_footprint(puuid: str) -> dict:
    """How much of this player Videns has actually reconstructed.

    Read-only and index-bound: `game_players(puuid)` is indexed, and the shop
    count is reached through `matches(puuid)` + `shop_visits(match_id, puuid)`
    because there is no index on `shop_visits.puuid` alone.
    """
    try:
        with db() as conn:
            games = conn.execute(
                "SELECT COUNT(*) FROM game_players WHERE puuid = ?", (puuid,)
            ).fetchone()[0]
            decisions = conn.execute(
                """
                SELECT COUNT(*)
                FROM matches m
                JOIN shop_visits sv ON sv.match_id = m.match_id AND sv.puuid = m.puuid
                WHERE m.puuid = ?
                """,
                (puuid,),
            ).fetchone()[0]
    except Exception:
        # The corpus panel is a bonus, never the reason a lookup fails.
        return {"available": False, "games": 0, "shop_decisions": 0}
    return {"available": True, "games": int(games), "shop_decisions": int(decisions)}


def corpus_totals() -> dict:
    """Corpus size, counted live so the page never quotes a stale claim."""
    global _totals
    with _totals_lock:
        stamped, cached = _totals
        if cached and time.monotonic() - stamped < CORPUS_TTL_S:
            return cached
    try:
        with db() as conn:
            totals = {
                "games": conn.execute("SELECT COUNT(*) FROM games").fetchone()[0],
                "players": conn.execute("SELECT COUNT(*) FROM players").fetchone()[0],
                "shop_decisions": conn.execute("SELECT COUNT(*) FROM shop_visits").fetchone()[0],
                "patches": [
                    row[0]
                    for row in conn.execute("SELECT DISTINCT patch FROM games ORDER BY patch")
                ],
            }
    except Exception:
        return {"games": 0, "players": 0, "shop_decisions": 0, "patches": []}
    with _totals_lock:
        _totals = (time.monotonic(), totals)
    return totals


def compile_profile(
    riot_id: str, region_key: str = "euw", count: int = 10, refresh: bool = False
) -> dict:
    """Resolve a Riot ID and compile its recent-match view.

    Cost is one account + one summoner + one league + one id-list call plus one
    Match-V5 call per game. The client throttles at 1.3 s and archives every raw
    match, so a repeat lookup is served from disk.
    """
    name, tag = parse_riot_id(riot_id)
    region = resolve_region(region_key)
    count = max(1, min(int(count), 20))
    key = (f"{name.lower()}#{tag.lower()}", region["platform"], count)

    cached = _profile_cache.get(key)
    if cached:
        age = time.monotonic() - cached[0]
        if age < PROFILE_TTL_S and not (refresh and age >= PROFILE_REFRESH_S):
            return cached[1]

    version = display_version()
    assets = _assets(version)

    with _client_lock:
        client = _get_client()
        try:
            account = _account(client, region, name, tag)
            puuid = account["puuid"]
            summoner = client.summoner_by_puuid(region["platform"], puuid)
            if not summoner:
                raise ProfileError(
                    f"{name}#{tag} existe, pero no ha jugado en {region['label']}.", 404
                )
            entries = client.league_entries_by_puuid(region["platform"], puuid)
            # One id past the page says whether "Load more" has anything to load.
            match_ids = client.match_ids(region["match"], puuid, None, count + 1)
            rows = _rows_for(client, region, puuid, match_ids[:count], version, assets)
        except RiotError as exc:
            raise ProfileError(str(exc), exc.status or 502) from exc
    profile = {
        "riot_id": f"{account.get('gameName') or name}#{account.get('tagLine') or tag}",
        "game_name": account.get("gameName") or name,
        "tag_line": account.get("tagLine") or tag,
        "region": {
            "key": region_key.lower(),
            "label": region["label"],
            "platform": region["platform"],
        },
        "level": int(summoner.get("summonerLevel") or 0),
        "icon": f"{DDRAGON_CDN}/{version}/img/profileicon/{summoner.get('profileIconId', 29)}.png",
        "ranks": _ranks(entries),
        "totals": _aggregate(rows),
        "champions": _by_champion(rows)[:6],
        "roles": _by_role(rows),
        "matches": rows,
        "has_more": len(match_ids) > count,
        "corpus": _corpus_footprint(puuid),
        "ddragon_version": version,
        "compiled_at_ms": int(time.time() * 1000),
    }
    _profile_cache[key] = (time.monotonic(), profile)
    return profile


def _account(client: RiotClient, region: dict[str, str], name: str, tag: str) -> dict:
    """Riot ID -> account, cached so paging and season polls do not re-resolve.

    The caller holds `_client_lock`. A 404 raises RiotError and is never cached.
    """
    key = (f"{name.lower()}#{tag.lower()}", region["account"])
    cached = _account_cache.get(key)
    if cached and time.monotonic() - cached[0] < ACCOUNT_TTL_S:
        return cached[1]
    account = client.account_by_riot_id(region["account"], name, tag)
    _account_cache[key] = (time.monotonic(), account)
    return account


def _assets(version: str) -> tuple[dict, dict, dict, dict]:
    return _summoner_spells(version), _runes(version), _champions(version), _augments(version)


def _rows_for(
    client: RiotClient,
    region: dict[str, str],
    puuid: str,
    match_ids: list[str],
    version: str,
    assets: tuple[dict, dict, dict, dict],
) -> list[dict]:
    """Match rows for `puuid`, newest first. The caller holds `_client_lock`."""
    spells, runes, champions, augments = assets
    rows = []
    for match_id in match_ids:
        match = client.match(region["match"], match_id)
        if not match:
            continue
        participants = (match.get("info") or {}).get("participants") or []
        me = next((p for p in participants if p.get("puuid") == puuid), None)
        if me:
            rows.append(_participant_row(match, me, version, spells, runes, champions, augments))
    rows.sort(key=lambda r: -r["ended_at_ms"])
    return rows


def compile_matches(
    riot_id: str, region_key: str = "euw", start: int = 20, count: int = 20
) -> dict:
    """One further page of a player's match history, for the page's Load more.

    The id endpoint pages from 0, so this asks for start + count + 1 ids (one
    call per 100) and keeps the slice; the extra id says whether a further page
    exists. If the player finished a game since the first page, the list has
    shifted by one: the page dedupes by match_id.
    """
    name, tag = parse_riot_id(riot_id)
    region = resolve_region(region_key)
    start = max(0, min(int(start), MATCH_PAGE_MAX_START))
    count = max(1, min(int(count), 20))
    version = display_version()
    assets = _assets(version)
    with _client_lock:
        client = _get_client()
        try:
            puuid = _account(client, region, name, tag)["puuid"]
            ids = client.match_ids(region["match"], puuid, None, start + count + 1)
            rows = _rows_for(client, region, puuid, ids[start:start + count], version, assets)
        except RiotError as exc:
            raise ProfileError(str(exc), exc.status or 502) from exc
    return {
        "start": start,
        "count": count,
        "matches": rows,
        "has_more": len(ids) > start + count,
        "ddragon_version": version,
    }


# ---------------------------------------------------------------------------
# Season sync: every game of the current season, for per-champion totals.
#
# Riot has no per-champion season endpoint; the only honest count is to read
# the season's matches. The season is the Data Dragon major version (16.x is
# 2026): ids are requested from 1 January of that year as a lower bound and a
# match is kept only if its gameVersion shares the major, so the exact rollout
# day never has to be guessed. Rows are compact (no lobby) and the list only
# grows while a job lives, so a poller can ask for the rows after an offset. A
# new job gets a new `sync_id`; the page restarts its copy when that changes.
# ---------------------------------------------------------------------------


def _season_major() -> int:
    try:
        return int(display_version().split(".")[0])
    except (ValueError, IndexError):
        return datetime.now(timezone.utc).year - 2010


def _season_row(match: dict, me: dict, version: str, champions: dict) -> dict:
    info = match.get("info") or {}
    mode = info.get("gameMode") or ""
    arena = mode == ARENA_MODE
    champion = _champion(champions, me, version)
    return {
        "match_id": (match.get("metadata") or {}).get("matchId"),
        "champion": champion["name"],
        "icon": champion["icon"],
        "queue_id": int(info.get("queueId") or 0),
        "mode": mode,
        "arena": arena,
        "placement": int(me.get("placement") or 0) if arena else 0,
        "win": bool(me.get("win")),
        "remake": bool(me.get("gameEndedInEarlySurrender")),
        "role": ROLE_LABELS.get(me.get("teamPosition") or "", ""),
        "kills": int(me.get("kills") or 0),
        "deaths": int(me.get("deaths") or 0),
        "assists": int(me.get("assists") or 0),
        "cs": int(me.get("totalMinionsKilled") or 0) + int(me.get("neutralMinionsKilled") or 0),
        "duration_s": int(info.get("gameDuration") or 0),
        "gold": int(me.get("goldEarned") or 0),
        "damage": int(me.get("totalDamageDealtToChampions") or 0),
        "vision": int(me.get("visionScore") or 0),
        "control_wards": int(me.get("visionWardsBoughtInGame") or 0),
        "ended_at_ms": int(info.get("gameEndTimestamp") or info.get("gameCreation") or 0),
    }


def _run_season_job(job: dict) -> None:
    region, puuid, major = job["region"], job["puuid"], job["major"]
    start_s = int(datetime(2010 + major, 1, 1, tzinfo=timezone.utc).timestamp())
    with _client_lock:
        ids = _get_client().match_ids(region["match"], puuid, None, 0, start_time=start_s)
    capped = len(ids) > SEASON_MAX_GAMES
    ids = ids[:SEASON_MAX_GAMES]
    with _season_lock:
        job["total"] = len(ids)
        job["scanned"] = 0
        job["capped"] = capped
    version = display_version()
    champions = _champions(version)
    for match_id in ids:
        # Only this worker writes `seen`, so reading it unlocked is safe.
        if match_id in job["seen"]:
            with _season_lock:
                job["scanned"] += 1
            continue
        with _client_lock:
            match = _get_client().match(region["match"], match_id)
        info = (match or {}).get("info") or {}
        game_major = str(info.get("gameVersion") or "").split(".")[0]
        older = game_major.isdigit() and int(game_major) < major
        row = None
        if match and game_major == str(major):
            participants = info.get("participants") or []
            me = next((p for p in participants if p.get("puuid") == puuid), None)
            if me:
                row = _season_row(match, me, version, champions)
        with _season_lock:
            job["scanned"] += 1
            job["seen"].add(match_id)
            if row:
                job["rows"].append(row)
            job["updated_at_ms"] = int(time.time() * 1000)
            if older:
                # Newest-first: everything after this is last season.
                job["total"] = job["scanned"]
        if older:
            break


def _season_worker_loop() -> None:
    while True:
        _season_wake.wait()
        with _season_lock:
            if not _season_queue:
                _season_wake.clear()
                continue
            job = _season_jobs.get(_season_queue.popleft())
            if job is None or job["status"] != "queued":
                continue
            job["status"] = "running"
        try:
            _run_season_job(job)
            status, error = "done", None
        except (RiotError, ProfileError) as exc:
            status, error = "error", str(exc)
        except Exception as exc:  # the worker must outlive any single job
            status, error = "error", f"{type(exc).__name__}: {exc}"
        with _season_lock:
            job["status"] = status
            job["error"] = error
            job["finished_at"] = time.monotonic()
            job["updated_at_ms"] = int(time.time() * 1000)


def _ensure_season_worker() -> None:
    global _season_worker
    with _season_lock:
        if _season_worker is None or not _season_worker.is_alive():
            _season_worker = threading.Thread(
                target=_season_worker_loop, name="season-sync", daemon=True
            )
            _season_worker.start()


def _enqueue_season(key: tuple[str, str]) -> None:
    """Caller holds `_season_lock`."""
    if key not in _season_queue:
        _season_queue.append(key)
    _season_wake.set()


def season_status(riot_id: str, region_key: str = "euw", after: int = 0) -> dict:
    """Start (or resume) the season sync for a player and report on it.

    Rows come back from `after` on, so a page polling during a sync only
    downloads what is new since its last poll.
    """
    global _season_epoch
    name, tag = parse_riot_id(riot_id)
    region = resolve_region(region_key)
    with _client_lock:
        client = _get_client()
        try:
            puuid = _account(client, region, name, tag)["puuid"]
        except RiotError as exc:
            raise ProfileError(str(exc), exc.status or 502) from exc
    major = _season_major()
    key = (puuid, region["platform"])
    now = time.monotonic()
    with _season_lock:
        job = _season_jobs.get(key)
        if job is None or job["major"] != major:
            _season_epoch += 1
            job = {
                "sync_id": _season_epoch,
                "puuid": puuid,
                "region": region,
                "major": major,
                "status": "queued",
                "rows": [],
                "seen": set(),
                "total": None,
                "scanned": 0,
                "capped": False,
                "error": None,
                "finished_at": 0.0,
                "touched_at": now,
                "updated_at_ms": int(time.time() * 1000),
            }
            _season_jobs[key] = job
            _enqueue_season(key)
        elif job["status"] == "done" and now - job["finished_at"] >= SEASON_RESYNC_S:
            # Incremental: games already read are skipped, new ones append.
            job["status"] = "queued"
            _enqueue_season(key)
        elif job["status"] == "error" and now - job["finished_at"] >= SEASON_RETRY_S:
            job["status"] = "queued"
            job["error"] = None
            _enqueue_season(key)
        job["touched_at"] = now
        if len(_season_jobs) > SEASON_JOBS_MAX:
            idle = sorted(
                (k for k, j in _season_jobs.items() if j["status"] in ("done", "error")),
                key=lambda k: _season_jobs[k]["touched_at"],
            )
            for stale in idle[: len(_season_jobs) - SEASON_JOBS_MAX]:
                del _season_jobs[stale]
        queue = list(_season_queue)
        after = max(0, min(int(after), len(job["rows"])))
        body = {
            "season": 2010 + major,
            "sync_id": job["sync_id"],
            "status": job["status"],
            "queue_position": queue.index(key) + 1 if key in queue else 0,
            "total": job["total"],
            "scanned": job["scanned"],
            "games": len(job["rows"]),
            "capped": job["capped"],
            "max_games": SEASON_MAX_GAMES,
            "offset": after,
            "rows": job["rows"][after:],
            "error": job["error"],
            "updated_at_ms": job["updated_at_ms"],
        }
    _ensure_season_worker()
    return body
