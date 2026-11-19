/* One timestamp owns the board and map. Movement is never interpolated across
   unobserved intervals. A death invalidates every older living position. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.VidensPlayback = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function lastAt(rows, timestamp) {
    let low = 0, high = rows.length - 1, found = null;
    while (low <= high) {
      const index = (low + high) >> 1;
      if (rows[index].ts <= timestamp) { found = rows[index]; low = index + 1; }
      else high = index - 1;
    }
    return found;
  }

  function point(value) {
    return value && Number.isFinite(value.x) && Number.isFinite(value.y)
      ? { x: value.x, y: value.y } : null;
  }

  function create(data) {
    const frames = [...(data.observations || [])].sort((a, b) => a.ts - b.ts);
    const inventories = [...(data.inventory_states || [])].sort((a, b) => a.ts - b.ts);
    const kills = (data.events || []).filter(e => e.kind === "kill").sort((a, b) => a.ts - b.ts);
    const series = new Map(data.players.map(player => {
      const id = player.participant_id;
      const samples = frames.filter(frame => frame.players?.[id]).map(frame => ({
        ...frame.players[id], ts: frame.ts,
      }));
      return [id, {
        samples,
        alive: samples.filter(sample => Number.isFinite(sample.health) && sample.health > 0),
        down: samples.filter(sample => sample.health === 0),
        deaths: kills.filter(event => event.actors?.[1] === id),
      }];
    }));

    function stateAt(requestedTime) {
      const ts = Math.max(0, Math.min(Number(requestedTime) || 0, data.duration_ms));
      const inventory = lastAt(inventories, ts);
      const kda = Object.fromEntries(data.players.map(p => [p.participant_id, [0, 0, 0]]));
      for (const event of kills) {
        if (event.ts > ts) break;
        const [killer, victim] = event.actors || [];
        if (kda[killer]) kda[killer][0]++;
        if (kda[victim]) kda[victim][1]++;
        for (const assist of new Set(event.assists || [])) {
          if (kda[assist] && assist !== killer && assist !== victim) kda[assist][2]++;
        }
      }

      const players = {};
      for (const player of data.players) {
        const id = player.participant_id, history = series.get(id);
        const sample = lastAt(history.samples, ts);
        const alive = lastAt(history.alive, ts);
        const down = lastAt(history.down, ts);
        const death = lastAt(history.deaths, ts);
        const invalidatedAt = Math.max(down?.ts ?? -Infinity, death?.ts ?? -Infinity);
        // At a shared timestamp, the death wins over a positive-health frame.
        const hasLivingObservation = alive && alive.ts > invalidatedAt;
        const unresolvedDeath = death && (!alive || death.ts >= alive.ts);
        players[id] = {
          id, sample_at: sample?.ts ?? null,
          level: sample?.level ?? null, cs: sample?.cs ?? null,
          health: sample?.health ?? null, health_max: sample?.health_max ?? null,
          inventory: [...(inventory?.players?.[id] || [])], kda: kda[id],
          life: hasLivingObservation ? "observed_alive" : unresolvedDeath ? "death_recorded" : "unknown",
          // Returning to the map requires positive-health evidence AFTER death.
          // A kill/assist by this player is insufficient (e.g. Karthus passive).
          position: hasLivingObservation ? point(alive.position) : null,
          position_at: hasLivingObservation ? alive.ts : null,
          position_age_ms: hasLivingObservation ? ts - alive.ts : null,
          death_position: unresolvedDeath ? point(death.position) : null,
          death_at: unresolvedDeath ? death.ts : null,
        };
      }
      const structures = (data.structures || []).map(structure => ({
        ...structure,
        down: (Number.isFinite(structure.destroyed_at) && structure.destroyed_at <= ts)
          || (structure.down_intervals || []).some(interval => interval.from <= ts && ts < interval.to),
      }));
      return { ts, players, structures, inventory_at: inventory?.ts ?? null,
        final_inventory: inventory?.source === "match_final" };
    }
    return { stateAt };
  }
  return { create, lastAt };
});
