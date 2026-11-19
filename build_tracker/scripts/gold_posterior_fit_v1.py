"""Fit the gold posterior — the wallet as a range, not a point.

TRAIN only. No labels enter the fit. No validation or test read, no ledger spend.

The correction we must add to `gold_est` to reach the player's true wallet,
`c = currentGold - gold_est`, was measured directly on 653,030 frame-boundary
windows (data/reports/gold_error_shape_v1). Its quantiles have a clear shape:

    c_q(dt) = b * dt + k_q * sqrt(dt)

Bias grows LINEARLY with staleness, uncertainty as its SQUARE ROOT — the
signature of income arriving as a stream of increments, so variance grows with
elapsed time and the standard deviation with its root. Measured fit error 2-5%
per quantile against 26-30% for a straight line through the origin, which is
what made the first sweep's `rate:k` arms the wrong functional form.

This matters most where it cannot be measured. Truth exists only at frames, so
every observation sits at dt >= 60 s, while a real shop visit sits at dt ~ 29 s
median. The sqrt term says the uncertainty there is relatively LARGER per
second than at the boundary, which a linear extrapolation understates.

The censored check (`--shortfalls`) is the one use of the recorded purchases
that is not a leak. A player who bought a basket costing X had at least X, so
each budget-short purchase is a one-sided observation `true wallet >= X`, taken
at a REAL visit timestamp rather than a frame boundary. The fit does not use
them — they are label-SELECTED (we only see them where our estimate was wrong
and the player had gold), so fitting the upper tail to them would bias delta
upward. They are used to ask whether a posterior fitted on unselected data
covers them at the rate it claims, which is a validation the frame data alone
cannot provide.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

# A fine grid across the whole range, not just the upper half: the trainer
# SAMPLES from this posterior (inverse-CDF on k_q), so the lower tail has to
# exist too. The mask only ever reads the upper half.
QUANTILES = tuple(round(0.02 + 0.02 * i, 2) for i in range(49))  # 0.02 .. 0.98


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--errors', type=Path,
                    default=ROOT / 'data/reports/gold_error_shape_v1/gold_error_raw.npz')
    ap.add_argument('--shortfalls', type=Path,
                    default=ROOT / 'data/reports/gold_error_shape_v1/shortfall_raw.npz')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    d = np.load(args.errors)
    # c = truth - gold_est: how much must be ADDED to cover the wallet.
    series = [(-d['err_skip0'], float(np.median(d['dt_skip0']))),
              (-d['err_skip1'], float(np.median(d['dt_skip1']))),
              (-d['err_skip2'], float(np.median(d['dt_skip2'])))]
    dts = np.array([dt for _c, dt in series])

    med = np.array([float(np.median(c)) for c, _dt in series])
    b = float((dts * med).sum() / (dts ** 2).sum())

    # Each quantile gets its OWN drift and diffusion term:
    #     c_q(dt) = a_q*dt + k_q*sqrt(dt)
    # A single shared drift (the median's b) describes the upper half well but
    # fails the lower tail badly (58-75% error below q=0.20). That tail is
    # over-credit, which comes from spending we could not see rather than from
    # income accumulating — lumpy, not a diffusion — so it has no reason to
    # share the income drift. Two parameters on three anchor ages leaves one
    # degree of freedom, and `worst_absolute_error_gold` reports it.
    design = np.stack([dts, np.sqrt(dts)], axis=1)
    fit = {}
    for q in QUANTILES:
        obs = np.array([float(np.percentile(c, q * 100)) for c, _dt in series])
        (a, k), *_ = np.linalg.lstsq(design, obs, rcond=None)
        pred = design @ np.array([a, k])
        fit[f'{q:.2f}'] = {
            'a': float(a), 'k': float(k),
            'worst_absolute_error_gold': float(np.max(np.abs(pred - obs))),
            'worst_relative_error': float(np.max(np.abs(pred - obs) / np.maximum(np.abs(obs), 1.0))),
            'observed': obs.tolist(), 'predicted': pred.tolist()}

    report = {
        'schema_version': 1,
        'purpose': 'gold_posterior_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'train',
        'model': 'c_q(dt) = a_q*dt + k_q*sqrt(dt);  delta = c_q, dt in SECONDS',
        'definition': 'c = currentGold - gold_est (gold to ADD to cover the wallet)',
        'b_gold_per_second': b,
        'quantiles': fit,
        'fitted_on': {'source': str(args.errors),
                      'windows': [int(len(c)) for c, _dt in series],
                      'anchor_ages_s': dts.tolist(),
                      'labels_used': False},
        'bounds': ('every observation sits at dt >= 60 s because truth exists only at '
                   'frames; values below that are an extrapolation of the fitted form, '
                   'anchored by c(0) = 0.'),
    }

    if args.shortfalls.exists():
        s = np.load(args.shortfalls)
        short = s['basket_short']          # cost(basket) - gold_est, > 0 by construction
        # These carry no dt of their own here, so the check is stated at the
        # median real visit staleness and at the frame boundary, bracketing it.
        cover = {}
        for dt in (29.0, 60.0):
            per_q = {}
            for q in [q for q in QUANTILES if q >= 0.50]:
                e = fit[f'{q:.2f}']
                delta = e['a'] * dt + e['k'] * (dt ** 0.5)
                per_q[f'{q:.2f}'] = {'delta': float(delta),
                                     'covered': float(np.mean(short <= delta))}
            cover[f'dt={dt:g}s'] = per_q
        report['censored_coverage_check'] = {
            'n': int(len(short)),
            'source': str(args.shortfalls),
            'question': ('of the purchases our estimate priced out of reach, what share would '
                         'a mask at gold_est + delta have admitted?'),
            'reading': ('NOT a calibration target. This population is label-selected — only '
                        'rows where the estimate was too low AND a purchase happened — so its '
                        'covered share is expected to sit BELOW the posterior quantile. A '
                        'covered share ABOVE q would mean the fitted tail is too wide.'),
            'by_dt': cover,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'posterior.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    print(f'b = {b:.4f} g/s')
    print(f"{'q':>6}{'a (g/s)':>10}{'k (g/sqrt s)':>14}{'worst err (g)':>15}{'delta@29s':>11}{'delta@60s':>11}")
    for q in [q for q in QUANTILES if abs(q*100 % 10) < 1e-6 or q in (0.65,0.75,0.95)]:
        e = fit[f'{q:.2f}']
        print(f"{q:6.2f}{e['a']:10.3f}{e['k']:14.2f}{e['worst_absolute_error_gold']:15.1f}"
              f"{e['a']*29 + e['k']*29**0.5:11.0f}{e['a']*60 + e['k']*60**0.5:11.0f}")
    if 'censored_coverage_check' in report:
        print('\ncensored coverage (label-selected; expected BELOW q):')
        for dt, per_q in report['censored_coverage_check']['by_dt'].items():
            row = '  '.join(f"{q}:{v['covered']:.2f}" for q, v in per_q.items())
            print(f'  {dt:9s} {row}')
    print(f"\nwrote {args.out / 'posterior.json'}")


if __name__ == '__main__':
    main()
