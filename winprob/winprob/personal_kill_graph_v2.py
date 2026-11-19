"""Past kill relationships, snapshot parity and degree-preserving controls.

No file, game API, source runtime, labels or outcomes are read on import.
Times are provenance only; the model projection consumes counts and profiles.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def teams_array(teams):
    value = np.asarray(teams)
    require(value.shape == (10,) and np.issubdtype(value.dtype, np.integer)
            and set(value.tolist()) == {100, 200} and np.sum(value == 100) == 5, 'fixed five-versus-five roster required')
    return value


@dataclass(frozen=True)
class Kill:
    timestamp_ms: int
    killer: int
    victim: int
    assisters: tuple[int, ...]

    def validate(self):
        require(type(self.timestamp_ms) is int and self.timestamp_ms >= 0, 'valid event timestamp required')
        require(type(self.killer) is int and 0 <= self.killer <= 10
                and type(self.victim) is int and 1 <= self.victim <= 10, 'valid killer/victim identities required')
        require(isinstance(self.assisters, tuple) and all(type(p) is int and 1 <= p <= 10 for p in self.assisters),
                'valid source assister identities required; preserve original multiplicity')
        return self

    @classmethod
    def from_raw(cls, event):
        require(event.get('type') == 'CHAMPION_KILL', 'only raw champion-kill events allowed')
        return cls(event.get('timestamp'), event.get('killerId') or 0, event.get('victimId'),
                   tuple(event.get('assistingParticipantIds') or [])).validate()


def needed_kills(timeline, maximum_deaths):
    """Original frame/event order; stop before even validating an unused suffix."""
    require(type(maximum_deaths) is int and 0 <= maximum_deaths <= 10000, 'bounded required death prefix')
    if maximum_deaths == 0:
        return []
    frames = timeline.get('info', {}).get('frames')
    require(isinstance(frames, list), 'timeline frames required')
    result = []
    for frame in frames:
        events = frame.get('events') or []
        require(isinstance(events, list), 'frame events must be a list')
        for event in events:
            if event.get('type') == 'CHAMPION_KILL':
                result.append(Kill.from_raw(event))
                if len(result) == maximum_deaths:
                    return result
    raise ValueError('timeline lacks the observed death prefix')


def apply_kill(kill, teams, kda, graph, environment):
    kill.validate(); victim = kill.victim-1
    if kill.killer:
        killer = kill.killer-1
        require(teams[killer] != teams[victim], 'unsupported same-team champion kill')
        kda[killer, 0] += 1; graph[killer, victim] += 1
    else:
        environment[victim] += 1
    kda[victim, 1] += 1
    for pid in kill.assisters:
        kda[pid-1, 2] += 1


def graph_for_observation(events, expected_kda, decision_ms, teams):
    """Recover the unique stored-observation prefix, not all events before a shop."""
    teams = teams_array(teams); expected = np.asarray(expected_kda)
    require(expected.shape == (10, 3) and np.issubdtype(expected.dtype, np.integer) and np.all(expected >= 0), 'integer all-player KDA witness required')
    require(type(decision_ms) is int and decision_ms >= 0, 'valid decision timestamp required')
    count = int(expected[:, 1].sum())
    require(0 <= count <= min(10000, len(events)), 'observed prefix exceeds projected events')
    graph = np.zeros((10, 10), dtype=np.int64); environment = np.zeros(10, dtype=np.int64)
    observed = np.zeros((10, 3), dtype=np.int64); maximum_time = None
    for event in events[:count]:
        require(event.timestamp_ms <= decision_ms, 'future event entered original snapshot')
        apply_kill(event, teams, observed, graph, environment)
        maximum_time = event.timestamp_ms if maximum_time is None else max(maximum_time, event.timestamp_ms)
    require(np.array_equal(observed, expected), 'full KDA parity failed; do not choose a nearby prefix')
    require(np.array_equal(graph.sum(1), observed[:, 0])
            and np.array_equal(graph.sum(0)+environment, observed[:, 1]), 'graph marginal conservation failed')
    return graph, environment, {'kill_prefix_length': count, 'maximum_included_timestamp_ms': maximum_time}


def validate_graph(graph, teams):
    graph = np.asarray(graph); teams = teams_array(teams)
    require(graph.shape == (10, 10) and np.issubdtype(graph.dtype, np.integer) and np.all(graph >= 0)
            and graph.sum() <= 10000, 'bounded nonnegative integer graph required')
    require(not np.any(graph[teams[:, None] == teams[None, :]]), 'same-team edge in opponent graph')
    return graph, teams


def control_seed(entry):
    raw = f"personal-kill-graph-v1/null/{entry['match_id']}|{entry['participant_id']}|{entry['source_event_index']}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], 'big')


def degree_control(graph, teams, seed):
    """Uniform labelled-stub matching; preserves all row/column degrees exactly."""
    graph, teams = validate_graph(graph, teams); rng = np.random.default_rng(seed)
    result = np.zeros((10, 10), dtype=np.int64)
    for side in (100, 200):
        attackers = np.flatnonzero(teams == side); victims = np.flatnonzero(teams != side)
        block = graph[np.ix_(attackers, victims)]
        killer_stubs = np.repeat(attackers, block.sum(1))
        victim_stubs = np.repeat(victims, block.sum(0))
        np.add.at(result, (killer_stubs, rng.permutation(victim_stubs)), 1)
    require(np.array_equal(result.sum(0), graph.sum(0)) and np.array_equal(result.sum(1), graph.sum(1)), 'null graph changed observed degrees')
    return result


def edge_excess(graph, teams, buyer_pid):
    graph, teams = validate_graph(graph, teams)
    require(type(buyer_pid) is int and 1 <= buyer_pid <= 10, 'valid buyer participant required')
    buyer = buyer_pid-1; enemies = np.flatnonzero(teams != teams[buyer])
    global_kills = graph[enemies].sum(1); personal = graph[enemies, buyer]
    total = int(global_kills.sum()); deaths = int(personal.sum())
    numerator = total*personal - deaths*global_kills
    require(int(numerator.sum()) == 0, 'degree-corrected excess must sum to zero')
    denominator = total*(deaths+5) if total else 1
    return enemies, numerator, denominator


def excess_profile(graph, teams, buyer_pid, profiles):
    """17public profile coordinates; exact zero for no excess or common traits."""
    profiles = np.asarray(profiles, dtype=np.float64)
    require(profiles.shape == (10, 17) and np.isfinite(profiles).all(), 'ten finite public enemy profiles required')
    enemies, numerator, denominator = edge_excess(graph, teams, buyer_pid)
    centered = profiles[enemies] - profiles[enemies[0]]
    return numerator @ centered / denominator
