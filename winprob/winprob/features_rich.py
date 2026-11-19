"""Versioned exact-frame geometry and resource allocation for offline OMNI.

The base feature extractor remains unchanged. No roles, champion identity,
health/championStats, wallet, item inventory, or postmatch totals are consumed.
Map corners are fixed geometric base proxies, not measured fountain hitboxes.
Gold-weighted geometry is a resource-position association, not combat strength.
"""

from __future__ import annotations

import math
from itertools import combinations

from .features import HISTORY_FEATURES, STATE_FEATURES, extract_game_states


FEATURE_VERSION = "omni-rich-asof-v1"
MAP_EXTENT = 14870.0
LOCAL_RADIUS_UNITS = 2000.0  # Fixed before fitting; no radius search.

_GEOMETRY = (
    "pos_x_mean", "pos_y_mean", "pair_distance_mean", "pair_distance_min",
    "own_base_distance_mean", "enemy_base_distance_mean", "nearest_enemy_distance_mean",
    "gold_weighted_enemy_distance_mean", "gold_weighted_own_base_distance_mean",
    "local_gold_balance_min",
)
_ALLOCATION = (
    "gold_std", "gold_q25", "gold_q50", "gold_q75", "gold_max_share",
    "gold_top2_share",
)
_AVAILABILITY = (
    "position_count", "position_any", "pair_geometry_available",
    "gold_share_available",
)
_TEAM_SUMMARIES = _GEOMETRY + _ALLOCATION + _AVAILABILITY
ADDITIONAL_RICH_FEATURES = tuple(
    f"rich_{name}_{combination}" for name in _TEAM_SUMMARIES for combination in ("diff", "sum")
) + ("rich_cross_geometry_available", "rich_gold_total_sum")
STATE_RICH_FEATURES = STATE_FEATURES + ADDITIONAL_RICH_FEATURES
HISTORY_RICH_FEATURES = HISTORY_FEATURES
ALL_RICH_FEATURES = STATE_RICH_FEATURES + HISTORY_RICH_FEATURES
# A focal-red risk model negates these only; it retains sums and availability.
# This is team-relative feature orientation, not an assertion of map symmetry.
SIGNED_RICH_FEATURES = tuple(name for name in ALL_RICH_FEATURES
                             if name.endswith("_diff") or name == "gold_diff_slope_per_min")
INVARIANT_RICH_FEATURES = tuple(name for name in ALL_RICH_FEATURES if name not in SIGNED_RICH_FEATURES)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values]))


def _quantile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    location = (len(values) - 1) * fraction
    low = int(math.floor(location))
    high = int(math.ceil(location))
    return values[low] + (values[high] - values[low]) * (location - low)


def _position(record):
    position = record.get("position")
    if not isinstance(position, dict):
        return None
    x, y = position.get("x"), position.get("y")
    if (isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float))
            or not isinstance(y, (int, float)) or not math.isfinite(x) or not math.isfinite(y)
            or not (0 < x <= MAP_EXTENT and 0 < y <= MAP_EXTENT)):
        return None
    return (float(x) / MAP_EXTENT, float(y) / MAP_EXTENT)


def _distance(left, right):
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Rich allocation requires finite nonnegative resource values")
    return value


def _summarize_team(players, team):
    summary = {name: 0.0 for name in _TEAM_SUMMARIES}
    gold = [player["gold"] for player in players]
    total_gold = sum(gold)
    summary["gold_share_available"] = float(total_gold > 0)
    summary["gold_top2_share"] = sum(sorted(gold)[-2:]) / total_gold if total_gold else 0.0
    summary["gold_std"] = _std(gold)
    for suffix, fraction in (("q25", .25), ("q50", .5), ("q75", .75)):
        summary["gold_" + suffix] = _quantile(gold, fraction)
    if total_gold:
        shares = [value / total_gold for value in gold]
        summary["gold_max_share"] = max(shares)
    known = [player for player in players if player["position"] is not None]
    summary["position_count"] = float(len(known))
    summary["position_any"] = float(bool(known))
    summary["pair_geometry_available"] = float(len(known) >= 2)
    own = (0.0, 0.0) if team == 100 else (1.0, 1.0)
    enemy = (1.0, 1.0) if team == 100 else (0.0, 0.0)
    relative = [player["position"] if team == 100 else (1 - player["position"][0], 1 - player["position"][1])
                for player in known]
    for dimension, index in (("x", 0), ("y", 1)):
        values = [point[index] for point in relative]
        summary[f"pos_{dimension}_mean"] = _mean(values)
    pair_distances = [_distance(left["position"], right["position"]) for left, right in combinations(known, 2)]
    summary["pair_distance_mean"] = _mean(pair_distances)
    summary["pair_distance_min"] = min(pair_distances) if pair_distances else 0.0
    own_distances = [_distance(player["position"], own) for player in known]
    summary["own_base_distance_mean"] = _mean(own_distances)
    summary["enemy_base_distance_mean"] = _mean([_distance(player["position"], enemy) for player in known])
    # Require all own positions for wealth-position coupling, preventing a missing
    # rich player's gold from being redistributed over the remaining positions.
    if len(known) == 5 and total_gold:
        summary["gold_weighted_own_base_distance_mean"] = sum(
            player["gold"] * distance for player, distance in zip(known, own_distances)) / total_gold
    return summary, total_gold


def _cross_geometry(own_players, enemy_players, summary):
    nearest = [min(_distance(player["position"], enemy["position"]) for enemy in enemy_players)
               for player in own_players]
    summary["nearest_enemy_distance_mean"] = _mean(nearest)
    total_gold = sum(player["gold"] for player in own_players)
    if total_gold:
        summary["gold_weighted_enemy_distance_mean"] = sum(
            player["gold"] * distance for player, distance in zip(own_players, nearest)) / total_gold
    radius = LOCAL_RADIUS_UNITS / MAP_EXTENT
    local_balances = []
    for player in own_players:
        def local_weighted_gold(others):
            return sum(other["gold"] * math.exp(-.5 * (_distance(player["position"], other["position"]) / radius) ** 2)
                       for other in others)

        own_gold, enemy_gold = local_weighted_gold(own_players), local_weighted_gold(enemy_players)
        total = own_gold + enemy_gold
        local_balances.append((own_gold - enemy_gold) / total if total else 0.0)
    summary["local_gold_balance_min"] = min(local_balances)


def extract_rich_game_states(match: dict, timeline: dict) -> list[dict]:
    """Preserve base rows/labels/timestamps; augment each exact matching frame.

    Spatial summaries use only valid observed positions, with counts and
    availability flags. Missing, nonfinite, zero or out-of-map coordinates are
    unknown, never clamped or replaced by a plausible position. Cross-team
    geometry requires all ten positions. No alive/dead claim is inferred from
    position or a death timer: stored coordinates can belong to recent deaths.

    Distances are in map-extent units. Own-side position axes and base distances
    explicitly mirror red's coordinate frame. Every team summary emits blue
    minus red (_diff) and blue plus red (_sum); sums preserve absolute context.
    """
    rows = extract_game_states(match, timeline)
    frames = {int(frame["timestamp"]): frame for frame in timeline["info"]["frames"]}
    teams = {int(player["participantId"]): int(player["teamId"]) for player in match["info"]["participants"]}
    for row in rows:
        raw = frames[row["timestamp_ms"]]["participantFrames"]
        players = {100: [], 200: []}
        for pid, team in teams.items():
            record = raw[str(pid)]
            players[team].append({"position": _position(record), "gold": _nonnegative(record["totalGold"])})
        blue, blue_gold = _summarize_team(players[100], 100)
        red, red_gold = _summarize_team(players[200], 200)
        complete = blue["position_count"] == red["position_count"] == 5
        if complete:
            _cross_geometry(players[100], players[200], blue)
            _cross_geometry(players[200], players[100], red)
        for name in _TEAM_SUMMARIES:
            row["x"][f"rich_{name}_diff"] = blue[name] - red[name]
            row["x"][f"rich_{name}_sum"] = blue[name] + red[name]
        row["x"]["rich_cross_geometry_available"] = float(complete)
        row["x"]["rich_gold_total_sum"] = blue_gold + red_gold
        if set(row["x"]) != set(ALL_RICH_FEATURES) or not all(math.isfinite(value) for value in row["x"].values()):
            raise ValueError("Rich feature allowlist or finite-value invariant failed")
    return rows
