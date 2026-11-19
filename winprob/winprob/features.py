"""Independent, offline OMNI state features from Match-V5 payloads.

This module never opens files and imports no sibling-project code. The outcome
and match metadata are returned beside, never inside, the feature mapping. All
features describe blue (team 100) minus red (team 200). These are retrospective
spectator features: the module makes no live-POV or causal-action-value claim.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


FEATURE_VERSION = "omni-asof-v1"
BASELINE_FEATURES = ("gold_diff", "kills_diff", "towers_diff", "dragons_diff", "time_min")
STATE_FEATURES = (
    "time_min", "gold_diff", "xp_diff", "level_diff", "cs_diff", "jungle_cs_diff",
    "damage_champions_diff", "damage_taken_diff", "kills_diff", "deaths_diff",
    "assists_diff", "towers_diff", "inhibitors_diff", "dragons_diff", "barons_diff",
    "heralds_diff", "grubs_diff",
)
HISTORY_FEATURES = ("gold_diff_slope_per_min", "gold_diff_slope_available")
ALL_FEATURES = STATE_FEATURES + HISTORY_FEATURES

_FRAME_FIELDS = {
    "gold_diff": ("totalGold",),
    "xp_diff": ("xp",),
    "level_diff": ("level",),
    "cs_diff": ("minionsKilled",),
    "jungle_cs_diff": ("jungleMinionsKilled",),
    "damage_champions_diff": ("damageStats", "totalDamageDoneToChampions"),
    "damage_taken_diff": ("damageStats", "totalDamageTaken"),
}
_EVENT_FEATURES = (
    "kills", "deaths", "assists", "towers", "inhibitors", "dragons", "barons", "heralds", "grubs",
)


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("missing or non-numeric measurement")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite measurement")
    return value


def _integer(value: Any) -> int:
    number = _finite_number(value)
    if not number.is_integer():
        raise ValueError("expected integer measurement")
    return int(number)


def _metadata(match: dict, timeline: dict) -> tuple[str, int, str, int, dict[int, int]]:
    game_id = match.get("metadata", {}).get("matchId")
    timeline_id = timeline.get("metadata", {}).get("matchId")
    if not isinstance(game_id, str) or not game_id or game_id != timeline_id:
        raise ValueError("match/timeline game identity is absent or mismatched")
    info = match["info"]
    start = _integer(info.get("gameStartTimestamp"))
    version = info.get("gameVersion")
    parsed = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version) if isinstance(version, str) else None
    if parsed is None:
        raise ValueError("gameVersion must contain major.minor patch")
    patch = f"{int(parsed.group(1))}.{int(parsed.group(2))}"
    participant_teams = {}
    for participant in info.get("participants", []):
        pid = _integer(participant.get("participantId"))
        team = _integer(participant.get("teamId"))
        if pid in participant_teams or team not in (100, 200):
            raise ValueError("duplicate participant or unknown team")
        participant_teams[pid] = team
    if set(participant_teams) != set(range(1, 11)) or Counter(participant_teams.values()) != {100: 5, 200: 5}:
        raise ValueError("require ten unique participants, five on each team")
    outcomes = {}
    for team in info.get("teams", []):
        team_id = _integer(team.get("teamId"))
        if team_id in outcomes or type(team.get("win")) is not bool:
            raise ValueError("require distinct team outcomes with boolean win values")
        outcomes[team_id] = team["win"]
    if set(outcomes) != {100, 200} or outcomes[100] == outcomes[200]:
        raise ValueError("require one winning team and one losing team")
    return game_id, start, patch, int(outcomes[100]), participant_teams


def _frame_diffs(frame: dict, participant_teams: dict[int, int]) -> dict[str, float] | None:
    """Incomplete state frames are excluded, never repaired with invented zeroes."""
    raw = frame.get("participantFrames")
    if not isinstance(raw, dict) or len(raw) != 10:
        return None
    participants = {}
    try:
        for key, record in raw.items():
            pid = _integer(record.get("participantId"))
            if str(pid) != str(key) or pid in participants:
                return None
            participants[pid] = record
        if set(participants) != set(participant_teams):
            return None
        result = {name: 0.0 for name in _FRAME_FIELDS}
        for pid, record in participants.items():
            sign = 1 if participant_teams[pid] == 100 else -1
            for name, path in _FRAME_FIELDS.items():
                value = record
                for part in path:
                    value = value[part]
                result[name] += sign * _finite_number(value)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return result


def _credit_event(event: dict, teams: dict[int, int], totals: dict[int, Counter]) -> None:
    def participant_team(raw: Any) -> int | None:
        try:
            return teams.get(_integer(raw))
        except ValueError:
            return None

    kind = event.get("type")
    if kind == "CHAMPION_KILL":
        killer_team = participant_team(event.get("killerId"))
        victim_team = participant_team(event.get("victimId"))
        if killer_team is not None:
            totals[killer_team]["kills"] += 1
        if victim_team is not None:
            totals[victim_team]["deaths"] += 1
        for pid in set(event.get("assistingParticipantIds") or []):
            team = participant_team(pid)
            if team is not None:
                totals[team]["assists"] += 1
    elif kind == "BUILDING_KILL":
        # Riot's teamId is the DESTROYED building's owner, not its killer.
        defending_team = event.get("teamId")
        building = {"TOWER_BUILDING": "towers", "INHIBITOR_BUILDING": "inhibitors"}.get(event.get("buildingType"))
        if defending_team in (100, 200) and building:
            totals[300 - defending_team][building] += 1
    elif kind == "ELITE_MONSTER_KILL":
        team = event.get("killerTeamId")
        if team not in (100, 200):
            team = participant_team(event.get("killerId"))
        monster = {
            "DRAGON": "dragons", "BARON_NASHOR": "barons",
            "RIFTHERALD": "heralds", "HORDE": "grubs", "VOID_GRUB": "grubs",
            "VOIDGRUB": "grubs",
        }.get(event.get("monsterType"))
        if team in (100, 200) and monster:
            totals[team][monster] += 1


def extract_game_states(match: dict, timeline: dict) -> list[dict]:
    """Return complete as-of frames from 3 through 40 minutes, inclusive.

    Payloads must be unwrapped Match-V5 responses from the same game. The
    containing frame of GAME_END and every later frame are excluded, even when
    that frame's timestamp precedes the GAME_END event. Per-event timestamps
    govern counts: events bundled in a frame are not assumed to have happened
    by that frame's timestamp. No interpolation, final statistics, role labels,
    duration, item state, current wallet, or championStats enter ``x``.

    Gold slope uses the most recent earlier COMPLETE frame (including frames
    before minute 3). If no such frame exists, its numeric placeholder is zero
    and gold_diff_slope_available is zero. All ordinary state fields are required.
    """
    game_id, start, patch, y, teams = _metadata(match, timeline)
    info = timeline["info"]
    if _integer(info.get("frameInterval")) != 60000:
        raise ValueError("feature version requires 60000 ms timeline frameInterval")
    frames = info.get("frames")
    if not isinstance(frames, list):
        raise ValueError("timeline frames must be a list")
    ordered = sorted(((_integer(frame.get("timestamp")), frame) for frame in frames),
                     key=lambda pair: pair[0])
    if len({timestamp for timestamp, _ in ordered}) != len(ordered):
        raise ValueError("duplicate frame timestamps")
    terminal_cutoff = math.inf
    events = []
    for frame_ts, frame in ordered:
        for event in frame.get("events", []):
            event_ts = _integer(event.get("timestamp"))
            if event.get("type") == "GAME_END":
                terminal_cutoff = min(terminal_cutoff, frame_ts, event_ts)
            events.append((event_ts, event))
    if terminal_cutoff == math.inf:
        raise ValueError("complete timeline requires a GAME_END event")
    events.sort(key=lambda pair: pair[0])
    totals = {100: Counter(), 200: Counter()}
    event_index = 0
    previous = None
    rows = []
    for timestamp, frame in ordered:
        if timestamp >= terminal_cutoff or timestamp > 40 * 60000:
            break
        while event_index < len(events) and events[event_index][0] <= timestamp:
            _credit_event(events[event_index][1], teams, totals)
            event_index += 1
        state = _frame_diffs(frame, teams)
        if state is None:
            continue
        slope = 0.0
        available = float(previous is not None)
        if previous is not None:
            prev_ts, prev_gold = previous
            slope = (state["gold_diff"] - prev_gold) / ((timestamp - prev_ts) / 60000.0)
        previous = (timestamp, state["gold_diff"])
        if timestamp < 3 * 60000:
            continue
        x = {"time_min": timestamp / 60000.0, **state}
        x.update({f"{name}_diff": float(totals[100][name] - totals[200][name]) for name in _EVENT_FEATURES})
        x["gold_diff_slope_per_min"] = slope
        x["gold_diff_slope_available"] = available
        assert set(x) == set(ALL_FEATURES)
        rows.append({"game_id": game_id, "timestamp_ms": timestamp, "game_start_ms": start,
                     "patch": patch, "y": y, "x": x})
    return rows
