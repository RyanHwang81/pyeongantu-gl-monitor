#!/usr/bin/env python3
"""Holdout test of the preregistered v4 candidates (docs/model-v4-preregistration.md).

Training signals: 1972-01..2005-12, with any forward window that would reach
past 2005-12 dropped. Holdout signals: 2006-01 onward. Asset guidance is fixed
on training data only, then applied once to the holdout.

H6 rescoring uses ALFRED vintages for PERMIT, INDPRO, PAYEMS and UNRATE at
each June and December end from 2006 to 2026; the other inputs are current
data cut at the vintage date. `--no-alfred` skips it.

Usage: python3 scripts/v4_holdout.py [--cache DIR] [--alfred-cache DIR] [--json OUT]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import growth_bias_check as gb  # noqa: E402
import market_regime_check as mr  # noqa: E402
import regime_returns_check as rr  # noqa: E402

TRAIN_END = pd.Timestamp('2005-12-01')
HOLDOUT_START = pd.Timestamp('2006-01-01')
WEATHER_COMPARE_FROM = pd.Timestamp('1980-08-01')
H = 12
THRESHOLD = 1.0  # percentage points
SPREADS = ('growth_value', 'cyc_def')
# What the dashboard's REGIME_ASSETS lists imply for the two spreads (0 = not stated).
C0_GUIDANCE = {
    'expansion': {'growth_value': 0, 'cyc_def': 1},
    'liquidity': {'growth_value': 1, 'cyc_def': 0},
    'adjustment': {'growth_value': -1, 'cyc_def': 0},
    'defense': {'growth_value': 0, 'cyc_def': -1},
}
ALFRED_SERIES = ('PERMIT', 'INDPRO', 'PAYEMS', 'UNRATE')


def months_between(a, b):
    return (b.year - a.year) * 12 + b.month - a.month


def train_mask(index, lag, start=None):
    """Signal months whose forward window ends by TRAIN_END."""
    last = TRAIN_END - pd.DateOffset(months=H + lag)
    mask = index <= last
    if start is not None:
        mask &= index >= start
    return pd.Series(mask, index=index)


def holdout_mask(index):
    return pd.Series(index >= HOLDOUT_START, index=index)


def derive_guidance(fwd, labels, mask, threshold=THRESHOLD):
    """Per quadrant, +1 / −1 / 0 for each spread from training means (in %)."""
    out, means = {}, {}
    for q in rr.QUADRANTS:
        out[q], means[q] = {}, {}
        for col in SPREADS:
            pick = fwd[col][(labels == q) & mask].dropna()
            m = float(pick.mean() * 100) if len(pick) else np.nan
            means[q][col] = None if not np.isfinite(m) else round(m, 2)
            out[q][col] = 0 if not np.isfinite(m) or abs(m) < threshold else int(np.sign(m))
    return out, means


def follow_returns(fwd, labels, guidance):
    """Monthly average of direction × spread over spreads with a stated direction."""
    parts = []
    for col in SPREADS:
        d = labels.map(lambda q: guidance.get(q, {}).get(col, 0) if isinstance(q, str) else 0).astype(float)
        parts.append((d * fwd[col]).where(d != 0))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def hac_mean(series, lags=12):
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 24:
        return {'n': len(y), 'mean': None, 't': None}
    nw = rr.newey_west_means(y, np.zeros(len(y)), [0.0], lags)
    b, cov, n = nw
    return {'n': int(n), 'mean': round(float(b[0]) * 100, 2), 't': round(float(b[0] / np.sqrt(cov[0, 0])), 2)}


def sign_retention(fwd, labels, mask, guidance, min_months=12):
    rows = {}
    for col in SPREADS:
        kept, counted, detail = 0, 0, {}
        for q in rr.QUADRANTS:
            d = guidance[q][col]
            pick = fwd[col][(labels == q) & mask].dropna()
            mean = float(pick.mean() * 100) if len(pick) else None
            detail[q] = {'guide': d, 'holdout_mean': None if mean is None else round(mean, 2), 'months': int(len(pick))}
            if d == 0 or len(pick) < min_months:
                continue
            counted += 1
            kept += int(np.sign(mean) == d)
        rows[col] = {'kept': kept, 'counted': counted, 'detail': detail}
    return rows


def guidance_test(fwd, labels, train, hold, fixed=None):
    guidance, train_means = derive_guidance(fwd, labels, train) if fixed is None else (fixed, None)
    follow = follow_returns(fwd, labels, guidance)
    result = {'guidance': guidance, 'train_means': train_means,
              'train_follow': hac_mean(follow[train]), 'holdout_follow': hac_mean(follow[hold]),
              'retention': sign_retention(fwd, labels, hold, guidance)}
    hf = result['holdout_follow']
    ret_ok = all(r['counted'] == 0 or r['kept'] / r['counted'] >= 0.75 for r in result['retention'].values())
    result['pass'] = bool(hf['mean'] is not None and hf['mean'] > 0 and hf['t'] is not None and hf['t'] >= 1.65 and ret_ok)
    return result


def lead_test(season, weather, frame_g, weather_g):
    """H1 on the holdout: entries from HOLDOUT_START, placebo by circular shifts."""
    def summary(w):
        _, rows = mr.transition_leads(season, w)
        rows = [r for r in rows if pd.Timestamp(r['d'] + '-01') >= HOLDOUT_START]
        leads = [r['lead'] for r in rows if r['lead'] is not None]
        return {'entries': len(rows), 'matched': len(leads),
                'median': None if not leads else float(np.median(leads)),
                'first': None if not leads else round(float(np.mean([x > 0 for x in leads])), 3)}, rows
    real, rows = summary(weather)
    values = weather.to_numpy(dtype=object)
    placebo = [summary(pd.Series(np.roll(values, k), index=weather.index))[0] for k in (36, 60, 84, 108, 132, 156)]
    p_first = float(np.mean([p['first'] for p in placebo if p['first'] is not None]))
    p_median = float(np.mean([p['median'] for p in placebo if p['median'] is not None]))
    xc = mr.cross_correlation(weather_g.loc[HOLDOUT_START:], frame_g.loc[HOLDOUT_START:])
    passed = (real['first'] is not None and real['first'] >= p_first + 0.10 and real['median'] > p_median
              and xc['best'] > 0 and xc['corr_at_best'] >= 0.2)
    return {'real': real, 'rows': rows, 'placebo_first': round(p_first, 3), 'placebo_median': round(p_median, 2),
            'xcorr_best': xc['best'], 'xcorr_at_best': xc['corr_at_best'], 'xcorr_at_0': xc['corr_at_0'],
            'pass': bool(passed)}


def weight_test(returns, frame, weather_frame):
    season_w = rr.target_weights(frame, 'confirmed')
    pos = mr.combine_positions(season_w, mr.weather_weights(weather_frame), returns.index)
    periods = {'훈련': (WEATHER_COMPARE_FROM.strftime('%Y-%m-%d'), TRAIN_END.strftime('%Y-%m-%d')),
               '보류': (HOLDOUT_START.strftime('%Y-%m-%d'), None)}
    table = mr.rule_backtests(returns, pos, periods)
    combo = [k for k in pos if k.startswith('계절 + 날씨')][0]
    h = table['보류'][combo]
    passed = h.get('gap_vs_const', -1) > 0 and h.get('gap_t', 0) >= 1.0
    return {'table': table, 'combo': combo, 'pass': bool(passed)}


# ---------------------------------------------------------------- H6 ALFRED

def vintage_dates(start=2006, end=2026):
    out = []
    for y in range(start, end + 1):
        out.append(f'{y}-06-30')
        if y < end:
            out.append(f'{y}-12-31')
    return out


def alfred_series(series_id, vintage, cache_dir):
    import alfred_backtest as ab
    return ab._alfred_series(series_id, vintage, Path(cache_dir))


def score_at_vintage(current_raw, vintage, cache_dir):
    stamp = pd.Timestamp(vintage)
    raw = {k: v.loc[:stamp] for k, v in current_raw.items()}
    errors = {}
    for sid in ALFRED_SERIES:
        try:
            raw[sid] = alfred_series(sid, vintage, cache_dir).loc[:stamp]
        except Exception as exc:
            errors[sid] = type(exc).__name__
    monthly, netliq = gb.monthly_inputs(raw)
    asof = pd.Timestamp(year=stamp.year, month=stamp.month, day=1)
    index = gb.model_index(monthly, asof)
    liquidity = gb.v3.change_liquidity(monthly, netliq)
    liquidity['NETLIQ']['available_from'] = monthly['WALCL'].first_valid_index()
    frame = gb.regimes(gb.score_axis(gb.base_growth(monthly), index), gb.score_axis(liquidity, index), index)
    last = frame.index[-1]
    regime = frame.loc[last, 'regime']
    return {'vintage': vintage, 'month': last.strftime('%Y-%m'), 'G': round(float(frame.loc[last, 'G']), 3),
            'regime': regime if isinstance(regime, str) else None, 'c': bool(frame.loc[last, 'c']), 'errors': errors}


def realtime_test(cache, alfred_cache, pub):
    current_raw, _ = gb.load_all(cache)
    rows = []
    for vintage in vintage_dates():
        row = score_at_vintage(current_raw, vintage, alfred_cache)
        stamp = pd.Timestamp(row['month'] + '-01')
        if stamp in pub.index:
            rev = pub.loc[stamp]
            row['G_revised'] = round(float(rev['G']), 3)
            row['regime_revised'] = rev['regime'] if isinstance(rev['regime'], str) else None
        rows.append(row)
    usable = [r for r in rows if 'G_revised' in r and not r['errors']]
    g_same = float(np.mean([np.sign(r['G']) == np.sign(r['G_revised']) for r in usable])) if usable else None
    reg_same = float(np.mean([r['regime'] == r['regime_revised'] for r in usable])) if usable else None
    return {'rows': rows, 'usable': len(usable), 'g_sign_agree': None if g_same is None else round(g_same, 3),
            'regime_agree': None if reg_same is None else round(reg_same, 3),
            'pass': bool(g_same is not None and g_same >= 0.8)}


# ---------------------------------------------------------------- run

def run(cache, end_month, alfred_cache=None):
    returns = rr.monthly_returns(cache)
    pub = rr.published_frame()
    seasons = {'C1': pub, 'C2': gb.variant_regimes(cache, end_month, ('a4',))['a4']}
    weather_frame = mr.market_frame(mr.market_signals(cache), 'M1', end_month)
    weather_labels = weather_frame['regime'].fillna('neutral')
    f1 = rr.forward(returns, H, 1)
    f0 = rr.forward(returns, H, 0)
    out = {'train_end': TRAIN_END.strftime('%Y-%m'), 'holdout_start': HOLDOUT_START.strftime('%Y-%m'),
           'returns_end': returns['stock'].dropna().index[-1].strftime('%Y-%m'), 'candidates': {}}

    def masks(index, lag, start=None):
        return train_mask(index, lag, start), holdout_mask(index)

    # C0: the current dashboard list on the published season.
    labels0 = rr.labels_from(pub, 'confirmed').reindex(f1.index)
    tr, ho = masks(f1.index, 1)
    out['C0'] = {'H5': guidance_test(f1, labels0, tr, ho, fixed=C0_GUIDANCE)}
    out['C0']['H5']['pass'] = bool(out['C0']['H5']['holdout_follow']['mean'] is not None
                                   and out['C0']['H5']['holdout_follow']['mean'] > 0
                                   and out['C0']['H5']['holdout_follow']['t'] >= 1.65)

    wl = weather_labels.reindex(f0.index)
    tr0, ho0 = masks(f0.index, 0, WEATHER_COMPARE_FROM)
    h3 = guidance_test(f0, wl, tr0, ho0)
    for key, frame in seasons.items():
        labels = rr.labels_from(frame, 'confirmed').reindex(f1.index)
        season = mr.season_series(frame)
        out['candidates'][key] = {
            'H1': lead_test(season.loc[WEATHER_COMPARE_FROM:], weather_frame['regime'], frame['G'], weather_frame['G']),
            'H2': guidance_test(f1, labels, tr, ho),
            'H3': h3,
            'H4': weight_test(returns, frame, weather_frame),
            'season_stock_p': {
                part: (rr.regime_stats(f1['stock_cash'][m], labels[m], rr.QUADRANTS, H, boot=False,
                                       extra=('unconfirmed',)) or {}).get('p_equal')
                for part, m in (('train', tr), ('holdout', ho))},
        }
    if alfred_cache:
        out['H6'] = realtime_test(cache, alfred_cache, pub)
    return out


def _yn(flag):
    return '통과' if flag else '실패'


def markdown(result):
    lines = [f"train≤{result['train_end']} holdout≥{result['holdout_start']} returns_end={result['returns_end']}", '']
    h5 = result['C0']['H5']
    lines.append(f"C0 H5 현행 목록: train {h5['train_follow']} holdout {h5['holdout_follow']} → {_yn(h5['pass'])}")
    lines.append(f"   retention {json.dumps(h5['retention'], ensure_ascii=False)}")
    for key, c in result['candidates'].items():
        lines.append(f'## {key}')
        h1 = c['H1']
        lines.append(f"H1 {_yn(h1['pass'])}: real {h1['real']} placebo first {h1['placebo_first']} median {h1['placebo_median']} "
                     f"xcorr best {h1['xcorr_best']} ({h1['xcorr_at_best']}, k0 {h1['xcorr_at_0']})")
        lines.append('   rows ' + ' '.join(f"{r['d']}:{r['to']}:{r['lead']}" for r in h1['rows']))
        for hk in ('H2', 'H3'):
            g = c[hk]
            lines.append(f"{hk} {_yn(g['pass'])}: guidance {g['guidance']} train_means {g['train_means']}")
            lines.append(f"   train {g['train_follow']} holdout {g['holdout_follow']}")
            lines.append(f"   retention {json.dumps(g['retention'], ensure_ascii=False)}")
        h4 = c['H4']
        lines.append(f"H4 {_yn(h4['pass'])}")
        for period, rows in h4['table'].items():
            for name, p in rows.items():
                gap = '—' if 'gap_vs_const' not in p else f"{p['gap_vs_const']:+.2f} ({p['gap_t']})"
                lines.append(f"   {period} | {name} | {p['months']} | {p['cagr']:.2f}% | {p['vol']:.1f}% | {p['sharpe']} | "
                             f"{p['mdd']:.1f}% | {p['total']:.1f}% | {p.get('avg_equity', '—')} | {gap}")
        lines.append(f"season stock_cash 12m p {c['season_stock_p']}")
        lines.append('')
    if 'H6' in result:
        h6 = result['H6']
        lines.append(f"H6 {_yn(h6['pass'])}: usable {h6['usable']} G sign agree {h6['g_sign_agree']} regime agree {h6['regime_agree']}")
        for r in h6['rows']:
            lines.append(f"   {r['vintage']} {r['month']} G {r['G']} vs {r.get('G_revised')} | {r['regime']} vs {r.get('regime_revised')} {r['errors'] or ''}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache', default=os.environ.get('GL_CACHE', '/tmp/gl-cache'))
    parser.add_argument('--alfred-cache', default='/tmp/alfred-cache')
    parser.add_argument('--no-alfred', action='store_true')
    parser.add_argument('--end', default=None)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    end = args.end or pd.Timestamp.today().to_period('M').to_timestamp().strftime('%Y-%m-%d')
    result = run(args.cache, end, None if args.no_alfred else args.alfred_cache)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(markdown(result))


if __name__ == '__main__':
    main()
