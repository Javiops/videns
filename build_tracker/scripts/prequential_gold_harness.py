"""Prequential gold harness — Phase A of the leak-safe inputs plan.

Per docs/experiment_gold_prequential_v3.md (frozen BEFORE the first run).
One pass over TRAIN games of the frozen generation, three reports:

  1. prequential metrics: the REAL production estimator
     (app.reconstruct._attach_gold_estimates, captured inputs via the
     reconstruct_game gold_debug hook — the walk is never re-implemented)
     evaluated at t = frame_k timestamps against the exact currentGold truth,
     plus pre-registered ADDITIVE variant corrections (A1 goldPerSecond with a
     fitted unit, A2 assist gold with prior-window de-double-counting) fitted
     on even-hash games and reported on odd-hash games only.
  2. visit-level over-credit audit (READS THE FUTURE; TRAIN-only diagnostic
     that never feeds an export): wallet(t) <= currentGold(f_next) + purchases
     in [t, f_next), income being non-negative. Counts gold_est > bound + 1.
  3. opening-sale timing: how often a 10s-idle-grouped visit opens with
     ITEM_SOLD and whether the first purchase is strictly later.

TRAIN only. No validation or test reads, no model, no ledger spend.
"""
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from app.config import RAW_MATCH_V5_DIR, SHOP_IDLE_MS
from app.riot import RawMatchArchive
from app.reconstruct import (SHOP_DEAD_TIME_MS, _attach_gold_estimates,
                             reconstruct_game)
from app.ddragon import dragon_for_patch

GPS_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
ASSIST_GRID = tuple(float(v) for v in range(0, 425, 25))
CHECKPOINT_EVERY = 100


def minute_bucket(ts_ms: float) -> str:
    minute = ts_ms / 60000.0
    if minute < 10:
        return '00-09'
    if minute < 20:
        return '10-19'
    return '20+'


def fit_split(match_id: str) -> str:
    digest = int(hashlib.sha256(match_id.encode()).hexdigest(), 16)
    return 'fit' if digest % 2 == 0 else 'report'


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {'n': len(ordered), 'median': pick(0.5), 'p25': pick(0.25),
            'p75': pick(0.75), 'p90': pick(0.90),
            'mean': sum(ordered) / len(ordered)}


class WindowRecord:
    """One frame-boundary prediction with the features every variant needs."""
    __slots__ = ('split', 'err_v2', 'purchase_in_window', 'income_event',
                 'minute', 'gps0', 'dt_s', 'n_assist_gap', 'n_assist_prior',
                 'frac', 'prior_residual')

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw[name])

    def correction(self, c_gps: float, c_assist: float) -> float:
        delta = c_gps * self.gps0 * self.dt_s + c_assist * self.n_assist_gap
        if c_assist and self.prior_residual is not None:
            # A2 must not double count: v2's residual already carried the
            # prior window's assist gold, prorated by frac and floored at 0.
            delta -= self.frac * min(c_assist * self.n_assist_prior,
                                     float(self.prior_residual))
        return delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='smoke: first N games')
    args = parser.parse_args()

    assignment = json.loads((args.dataset / 'split_assignment.json').read_text(encoding='utf-8'))
    train_ids = sorted(mid for mid, fold in assignment.items() if fold == 'train')
    if args.limit:
        train_ids = train_ids[:args.limit]
    archive = RawMatchArchive(RAW_MATCH_V5_DIR)
    dragon = dragon_for_patch('16.18')
    args.out.mkdir(parents=True, exist_ok=True)

    windows: list[WindowRecord] = []
    visit_rows = []       # (split, gold_est, bound, features for corrections)
    timing = Counter()
    gap_ms: list[int] = []
    skipped = Counter()

    for done, mid in enumerate(train_ids, 1):
        match = archive.read('match', mid)
        timeline = archive.read('timeline', mid)
        if not match or not timeline:
            skipped['missing_raw'] += 1
            continue
        debug: dict = {}
        try:
            _game, events = reconstruct_game(match, timeline, dragon, gold_debug=debug)
        except Exception:
            skipped['reconstruct_error'] += 1
            continue
        frame_series = debug['frame_series']
        gold_events = debug['gold_events']
        split = fit_split(mid)

        # Raw per-frame extras (goldPerSecond) and assist events, all timestamped.
        gps_by_pid: dict[int, list[tuple[int, float]]] = {}
        for frame in (timeline.get('info') or {}).get('frames') or []:
            fts = int(frame.get('timestamp') or 0)
            for key, stats in (frame.get('participantFrames') or {}).items():
                gps_by_pid.setdefault(int(key), []).append(
                    (fts, float(stats.get('goldPerSecond') or 0)))
        assists: dict[int, list[int]] = {}
        item_stream: dict[int, list[tuple[int, str]]] = {}
        for frame in (timeline.get('info') or {}).get('frames') or []:
            for event in frame.get('events') or []:
                etype = event.get('type')
                if etype == 'CHAMPION_KILL':
                    for helper in event.get('assistingParticipantIds') or []:
                        assists.setdefault(int(helper), []).append(
                            int(event.get('timestamp') or 0))
                elif etype in ('ITEM_PURCHASED', 'ITEM_SOLD'):
                    pid = int(event.get('participantId') or 0)
                    item_stream.setdefault(pid, []).append(
                        (int(event.get('timestamp') or 0), etype))

        def gps_at(pid: int, ts: int) -> float:
            entries = gps_by_pid.get(pid) or []
            value = 0.0
            for fts, gps in entries:
                if fts > ts:
                    break
                value = gps
            return value

        def assist_count(pid: int, lo: int, hi_exclusive: int) -> int:
            return sum(1 for ats in assists.get(pid, ()) if lo < ats < hi_exclusive)

        for pid, series in frame_series.items():
            timed = gold_events.get(pid) or []
            for k in range(1, len(series)):
                t, truth, _tot = series[k]
                f0_ts = series[k - 1][0]
                synth = {'type': 'shop', 'participant_id': pid, 'ts': t}
                _attach_gold_estimates([synth], frame_series, gold_events)
                if 'gold_est' not in synth:
                    continue
                prior_residual = (synth.get('gold_gap') or {}).get('prior_residual')
                dt_ms = t - f0_ts
                frac = max(0.0, (dt_ms - SHOP_DEAD_TIME_MS)) / max(1, dt_ms) \
                    if prior_residual is not None else 0.0
                # frac in production is over the PRIOR window length; both
                # windows are the frame cadence, and using the prior length
                # exactly mirrors the formula:
                if prior_residual is not None and k >= 2:
                    prior_window = max(1, f0_ts - series[k - 2][0])
                    frac = max(0.0, (dt_ms - SHOP_DEAD_TIME_MS)) / prior_window
                windows.append(WindowRecord(
                    split=split,
                    err_v2=float(synth['gold_est']) - float(truth),
                    purchase_in_window=any(kind == 'spend' and amount > 0
                                           and f0_ts < ets < t
                                           for ets, amount, kind in timed),
                    income_event=(any(kind == 'bounty' and f0_ts < ets < t
                                      for ets, _a, kind in timed)
                                  or assist_count(pid, f0_ts, t) > 0),
                    minute=minute_bucket(t),
                    gps0=gps_at(pid, f0_ts),
                    dt_s=dt_ms / 1000.0,
                    n_assist_gap=assist_count(pid, f0_ts, t),
                    n_assist_prior=(assist_count(pid, series[k - 2][0], f0_ts + 1)
                                    if k >= 2 else 0),
                    frac=frac,
                    prior_residual=prior_residual,
                ))

            # Visit-level over-credit bound for the real shop events of this pid.
            stamps = [f[0] for f in series]
            for ev in events:
                if ev.get('type') != 'shop' or ev.get('participant_id') != pid:
                    continue
                if 'gold_est' not in ev:
                    continue
                t = int(ev.get('ts') or 0)
                nxt = bisect_left(stamps, t + 1)
                if nxt >= len(series):
                    continue
                fN_ts, fN_cur, _ = series[nxt]
                purchases = sum(amount for ets, amount, kind in timed
                                if kind == 'spend' and amount > 0
                                and t <= ets < fN_ts)
                f0_idx = bisect_left(stamps, t) - 1
                f0_ts = series[f0_idx][0] if f0_idx >= 0 else 0
                prior_residual = (ev.get('gold_gap') or {}).get('prior_residual')
                frac = 0.0
                if prior_residual is not None and f0_idx >= 1:
                    prior_window = max(1, f0_ts - series[f0_idx - 1][0])
                    frac = max(0.0, (t - f0_ts - SHOP_DEAD_TIME_MS)) / prior_window
                visit_rows.append({
                    'split': split,
                    'gold_est': float(ev['gold_est']),
                    'bound': float(fN_cur) + purchases,
                    'gps0': gps_at(pid, f0_ts),
                    'dt_s': (t - f0_ts) / 1000.0,
                    'n_assist_gap': assist_count(pid, f0_ts, t),
                    'n_assist_prior': (assist_count(pid, series[f0_idx - 1][0], f0_ts + 1)
                                       if f0_idx >= 1 else 0),
                    'frac': frac,
                    'prior_residual': prior_residual,
                })

            # Opening-sale timing over the raw item stream (10s idle grouping).
            stream = sorted(item_stream.get(pid) or [])
            visit: list[tuple[int, str]] = []
            def flush(batch):
                if not batch:
                    return
                timing['visits'] += 1
                sale_ts = batch[0][0] if batch[0][1] == 'ITEM_SOLD' else None
                buys = [ts for ts, kind in batch if kind == 'ITEM_PURCHASED']
                if sale_ts is None:
                    return
                if not buys:
                    timing['opens_with_sale_no_purchase'] += 1
                    return
                timing['opens_with_sale_with_purchase'] += 1
                if min(buys) > sale_ts:
                    timing['first_purchase_strictly_later'] += 1
                    gap_ms.append(min(buys) - sale_ts)
                else:
                    timing['first_purchase_not_later'] += 1
            for ts, kind in stream:
                if visit and ts - visit[-1][0] > SHOP_IDLE_MS:
                    flush(visit)
                    visit = []
                visit.append((ts, kind))
            flush(visit)

        if done % CHECKPOINT_EVERY == 0:
            print(f'{done}/{len(train_ids)} games; windows={len(windows)} '
                  f'visits={len(visit_rows)} skipped={dict(skipped)}', flush=True)

    # ---- fit on even-hash games, report on odd-hash games ----
    fit_rows = [w for w in windows if w.split == 'fit']
    rep_rows = [w for w in windows if w.split == 'report']

    def fit_constant(rows, grid, kind):
        best, best_err = 0.0, None
        for c in grid:
            if kind == 'gps':
                errs = [abs(w.err_v2 + w.correction(c, 0.0)) for w in rows]
            else:
                errs = [abs(w.err_v2 + w.correction(0.0, c)) for w in rows]
            med = statistics.median(errs) if errs else 0.0
            if best_err is None or med < best_err:
                best, best_err = c, med
        return best, best_err

    c_gps, _ = fit_constant(fit_rows, GPS_GRID, 'gps')
    c_assist, _ = fit_constant(fit_rows, ASSIST_GRID, 'assist')
    variants = {'v2': (0.0, 0.0), 'a1_gps': (c_gps, 0.0),
                'a2_assist': (0.0, c_assist), 'a3_both': (c_gps, c_assist)}

    def describe(rows, cg, ca):
        errs = [w.err_v2 + w.correction(cg, ca) for w in rows]
        out = {'abs_err': quantiles([abs(e) for e in errs]),
               'signed_bias': quantiles(errs),
               'overcredit_rate': (sum(1 for e in errs if e > 0) / len(errs)) if errs else None}
        return out

    metrics = {}
    for name, (cg, ca) in variants.items():
        no_buy = [w for w in rep_rows if not w.purchase_in_window]
        with_buy = [w for w in rep_rows if w.purchase_in_window]
        metrics[name] = {
            'constants': {'c_gps': cg, 'c_assist': ca},
            'report_split_all': describe(rep_rows, cg, ca),
            'report_split_no_purchase_SELECTION_STRATUM': describe(no_buy, cg, ca),
            'report_split_with_purchase_transparency_only': describe(with_buy, cg, ca),
            'by_minute': {b: describe([w for w in no_buy if w.minute == b], cg, ca)
                          for b in ('00-09', '10-19', '20+')},
            'by_income_event': {
                'with_kill_or_assist': describe([w for w in no_buy if w.income_event], cg, ca),
                'quiet': describe([w for w in no_buy if not w.income_event], cg, ca)},
        }

    def visit_overcredit(cg, ca):
        flags = []
        for row in visit_rows:
            delta = cg * row['gps0'] * row['dt_s'] + ca * row['n_assist_gap']
            if ca and row['prior_residual'] is not None:
                delta -= row['frac'] * min(ca * row['n_assist_prior'],
                                           float(row['prior_residual']))
            flags.append(row['gold_est'] + delta > row['bound'] + 1.0)
        return {'n_visits': len(flags),
                'provably_overcredited': sum(flags),
                'rate': (sum(flags) / len(flags)) if flags else None}

    overcredit = {name: visit_overcredit(cg, ca) for name, (cg, ca) in variants.items()}

    common = {'schema': 1, 'protocol': 'docs/experiment_gold_prequential_v3.md',
              'dataset': str(args.dataset), 'split': 'train',
              'games_requested': len(train_ids), 'skipped': dict(skipped),
              'fit_split': 'even sha256(match_id)', 'report_split': 'odd sha256(match_id)',
              'shop_dead_time_frozen_ms': SHOP_DEAD_TIME_MS,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'validation_or_test_reads': False}
    from app.dataset_artifacts import write_json
    write_json(args.out / 'prequential_metrics.json',
               {**common, 'purpose': 'prequential_gold_harness_v1',
                'windows': {'fit': len(fit_rows), 'report': len(rep_rows)},
                'variants': metrics})
    write_json(args.out / 'visit_overcredit_audit.json',
               {**common, 'purpose': 'gold_overcredit_audit_v1',
                'reads_future': True,
                'never_a_feature': 'one-sided bound; TRAIN diagnostic only',
                'variants': overcredit})
    write_json(args.out / 'opening_sale_timing.json',
               {**common, 'purpose': 'opening_sale_timing_v1',
                'counts': dict(timing), 'gap_ms': quantiles(gap_ms)})
    print(json.dumps({
        'windows_report': len(rep_rows),
        'v2_no_purchase': metrics['v2']['report_split_no_purchase_SELECTION_STRATUM'],
        'constants': {'c_gps': c_gps, 'c_assist': c_assist},
        'overcredit': {k: v['rate'] for k, v in overcredit.items()},
        'timing': dict(timing)}, default=str), flush=True)


if __name__ == '__main__':
    main()
