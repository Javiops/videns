"""Shape of the gold-estimation error — diagnostic for the wallet lever.

TRAIN only. No model, no validation/test reads, no ledger spend, no API.

Never re-implements the gold walk: it calls the production estimator
`app.reconstruct._attach_gold_estimates` on synthetic shop events, using the
inputs captured through `reconstruct_game(gold_debug=...)` — the same
discipline as scripts/prequential_gold_harness.py (Phase A).

Phase A reported five quantiles of the signed error. This dumps the RAW
distribution so its shape can be plotted, and adds one thing Phase A could not
answer: how the error responds to staleness.

  1. signed error `gold_est - truth` at frame boundaries under the production
     cadence (the anchor frame is the one immediately before t, so dt ~= 60 s —
     the WORST case: a real shop visit lands at a uniform point inside the
     window and is fresher than this).
  2. the same error with the anchor frame made older by hiding the intervening
     frames (dt ~= 120 s, ~180 s). Truth exists only at frames, so the error at
     dt < 60 s cannot be measured directly; this measures the other direction
     and the curve is anchored by dt = 0 => error identically 0.
  3. over-credit as a function of a mask slack: P(err + delta > 0), the share of
     states where masking at `gold_est + delta` would exceed the true wallet.

Truth is `currentGold` at frame k. Reading it is legal HERE and nowhere near
an export: this is a diagnostic of estimator error, never a feature.
"""
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from app.config import RAW_MATCH_V5_DIR
from app.riot import RawMatchArchive
from app.reconstruct import _attach_gold_estimates, reconstruct_game
from app.ddragon import dragon_for_patch

SKIPS = (0, 1, 2)          # frames hidden before the anchor => dt ~ 60/120/180 s


def minute_code(ts_ms: float) -> int:
    minute = ts_ms / 60000.0
    return 0 if minute < 10 else (1 if minute < 20 else 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--patch', default='16.18')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    assignment = json.loads((args.dataset / 'split_assignment.json').read_text(encoding='utf-8'))
    train_ids = sorted(mid for mid, fold in assignment.items() if fold == 'train')
    if args.limit:
        train_ids = train_ids[:args.limit]
    archive = RawMatchArchive(RAW_MATCH_V5_DIR)
    dragon = dragon_for_patch(args.patch)
    args.out.mkdir(parents=True, exist_ok=True)

    err = {s: array('f') for s in SKIPS}
    dt = {s: array('f') for s in SKIPS}
    minute = array('b')          # skip=0 only
    bought = array('b')          # skip=0 only: a purchase fell inside the window
    income = array('b')          # skip=0 only: a kill bounty fell inside the window
    skipped = Counter()

    for done, mid in enumerate(train_ids, 1):
        match = archive.read('match', mid)
        timeline = archive.read('timeline', mid)
        if not match or not timeline:
            skipped['missing_raw'] += 1
            continue
        debug: dict = {}
        try:
            reconstruct_game(match, timeline, dragon, gold_debug=debug)
        except Exception:
            skipped['reconstruct_error'] += 1
            continue
        frame_series = debug['frame_series']
        gold_events = debug['gold_events']

        for pid, series in frame_series.items():
            timed = gold_events.get(pid) or []
            for k in range(1, len(series)):
                t, truth, _tot = series[k]
                for s in SKIPS:
                    anchor = k - 1 - s
                    if anchor < 0:
                        continue
                    # Hide every frame after the anchor: the estimator then walks
                    # from an older frame, exactly as a coarser cadence would.
                    synth = {'type': 'shop', 'participant_id': pid, 'ts': t}
                    _attach_gold_estimates([synth], {pid: series[:anchor + 1]}, gold_events)
                    if 'gold_est' not in synth:
                        continue
                    err[s].append(float(synth['gold_est']) - float(truth))
                    dt[s].append((t - series[anchor][0]) / 1000.0)
                    if s == 0:
                        a_ts = series[anchor][0]
                        minute.append(minute_code(t))
                        bought.append(1 if any(kind == 'spend' and amount > 0
                                               and a_ts < ets < t
                                               for ets, amount, kind in timed) else 0)
                        income.append(1 if any(kind == 'bounty' and a_ts < ets < t
                                               for ets, _a, kind in timed) else 0)
        if done % 100 == 0:
            print(f'{done}/{len(train_ids)} games; windows={len(err[0])} '
                  f'skipped={dict(skipped)}', flush=True)

    import numpy as np
    payload = {f'err_skip{s}': np.frombuffer(err[s], dtype=np.float32) for s in SKIPS}
    payload.update({f'dt_skip{s}': np.frombuffer(dt[s], dtype=np.float32) for s in SKIPS})
    payload['minute'] = np.frombuffer(minute, dtype=np.int8)
    payload['bought'] = np.frombuffer(bought, dtype=np.int8)
    payload['income'] = np.frombuffer(income, dtype=np.int8)
    np.savez_compressed(args.out / 'gold_error_raw.npz', **payload)

    meta = {
        'purpose': 'gold_error_shape_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': str(args.dataset),
        'split': 'train',
        'games_requested': len(train_ids),
        'skipped': dict(skipped),
        'windows': {f'skip{s}': len(err[s]) for s in SKIPS},
        'truth': 'participant frame currentGold at the predicted frame',
        'sign': 'err = gold_est - truth; negative = the estimate UNDER-credits',
        'bounds': ('frame-boundary windows are the dt~=60s worst case; a real shop '
                   'visit sits at a uniform point inside the window and is fresher. '
                   'Diagnostic only - truth here is post-hoc and never a feature.'),
    }
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
