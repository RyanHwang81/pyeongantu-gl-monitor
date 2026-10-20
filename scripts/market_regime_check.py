#!/usr/bin/env python3
"""A market-price regime ("weather") beside the confirmed v3 regime ("season"). Analysis only.

G_m and L_m are built only from market prices and yields, scored with the same
v3 machinery (median/MAD z, frozen pairwise rho, EMA3, ±0.25/±0.10 hysteresis).
Market inputs are known at the end of the month they describe, so market
signals use no publication lag. The v3 season keeps its one-month lag.

Three candidate specifications, fixed before looking at results:
  M1 core   G: cyclical−defensive 6M (.5), 10Y−3M 6M change (.5)
            L: −2Y 3M change (.5), −Baa−10Y 6M change (.5)
  M2 broad  M1 plus copper/gold 6M change in G and −broad dollar YoY in L
  M3 single G: cyclical−defensive 6M only, L: −2Y 3M change only

Sources: Ken French 10 industry portfolios; FRED GS10, TB3MS, DGS2, BAA10YM,
DTWEXBGS, PCOPPUSDM; gold monthly prices from the datahub gold-prices mirror.
All revised current data, so this is an after-the-fact check.

Usage: python3 scripts/market_regime_check.py [--cache DIR] [--json OUT]
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import growth_bias_check as gb  # noqa: E402
import regime_returns_check as rr  # noqa: E402

GOLD_MIRROR = 'https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv'
QUADRANTS = rr.QUADRANTS
KO = rr.KO
# Book order: 금융 → 실적 → 역금융 → 역실적 → 금융.
NEXT = {'liquidity': 'expansion', 'expansion': 'adjustment', 'adjustment': 'defense', 'defense': 'liquidity'}
PREV = {v: k for k, v in NEXT.items()}
# Which axis flips on the way to the next season, and in which direction.
FLIP = {'liquidity': ('G', 1), 'expansion': ('L', -1), 'adjustment': ('G', -1), 'defense': ('L', 1)}
TILT = 0.5
SPECS = ('M1', 'M2', 'M3')


# ---------------------------------------------------------------- inputs

def load_gold(cache):
    text = gb._cached(cache, 'gold_mirror.csv', lambda: gb._get(GOLD_MIRROR))
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = ['date', 'price']
    frame['date'] = pd.to_datetime(frame['date'])
    return frame.set_index('date')['price'].resample('MS').mean()


def market_signals(cache):
    ind = rr.load_french(cache, '10_Industry_Portfolios')
    cyc_def = ind[list(rr.CYCLICAL)].mean(axis=1) - ind[list(rr.DEFENSIVE)].mean(axis=1)
    m = {sid: gb.to_monthly(gb.load_fred(cache, sid))
         for sid in ('GS10', 'TB3MS', 'DGS2', 'BAA10YM', 'DTWEXBGS', 'PCOPPUSDM')}
    curve = m['GS10'] - m['TB3MS']
    copper_gold = np.log(m['PCOPPUSDM']) - np.log(load_gold(cache).reindex(m['PCOPPUSDM'].index))
    return {
        'CYCDEF': cyc_def.rolling(6).sum() * 100,
        'CURVE': curve - curve.shift(6),
        'CUAU': (copper_gold - copper_gold.shift(6)) * 100,
        'DGS2': -(m['DGS2'] - m['DGS2'].shift(3)),
        'BAA': -(m['BAA10YM'] - m['BAA10YM'].shift(6)),
        'DOLLAR': -gb.yoy(m['DTWEXBGS']),
    }


def spec_components(signals, spec):
    s = signals
    if spec == 'M1':
        g = {'CYCDEF': .5, 'CURVE': .5}
        l = {'DGS2': .5, 'BAA': .5}
    elif spec == 'M2':
        g = {'CYCDEF': .4, 'CURVE': .3, 'CUAU': .3}
        l = {'DGS2': .4, 'BAA': .3, 'DOLLAR': .3}
    elif spec == 'M3':
        g = {'CYCDEF': 1.0}
        l = {'DGS2': 1.0}
    else:
        raise ValueError(spec)
    build = lambda ws: {k: {'t': s[k].dropna(), 'w': w} for k, w in ws.items()}
    return build(g), build(l)


def market_frame(signals, spec, end_month):
    g_comps, l_comps = spec_components(signals, spec)
    starts = [c['t'].first_valid_index() for c in (*g_comps.values(), *l_comps.values())]
    index = pd.date_range(min(starts), pd.Timestamp(end_month), freq='MS')
    g = gb.score_axis(g_comps, index)
    l = gb.score_axis(l_comps, index)
    frame = gb.regimes(g, l, index)
    # Hysteresis quadrant, not the two-month confirmation: weather is meant to be fast.
    frame['regime'] = frame['regime'].where(frame['regime'].apply(lambda v: isinstance(v, str)))
    return frame


# ---------------------------------------------------------------- lead / lag

def cross_correlation(market, model, max_lag=24):
    """corr(market_t, model_{t+k}). Positive k: market leads."""
    out = {}
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat({'m': market, 'v': model.shift(-k)}, axis=1, sort=True).dropna()
        out[k] = round(float(pair.m.corr(pair.v)), 3) if len(pair) > 24 else None
    best = max((k for k in out if out[k] is not None), key=lambda k: out[k])
    return {'best': best, 'corr_at_best': out[best], 'corr_at_0': out[0], 'by_lag': out}


def spells(labels):
    """Consecutive runs of the same non-null label: list of (label, start, end, months)."""
    runs, current, start, prev = [], None, None, None
    for stamp, value in labels.items():
        value = value if isinstance(value, str) else None
        if value != current:
            if current is not None:
                runs.append((current, start, prev))
            current, start = value, stamp
        prev = stamp
    if current is not None:
        runs.append((current, start, prev))
    return [(lab, a, b, (b.year - a.year) * 12 + b.month - a.month + 1) for lab, a, b in runs]


def season_series(frame):
    """Last confirmed regime carried forward. This is the 'season'."""
    regime = frame['regime'].where(frame['regime'].apply(lambda v: isinstance(v, str)))
    return regime.where(frame['c'].astype(bool)).ffill()


def entries(labels):
    """Months where a label starts, with the label before it."""
    out = []
    prev = None
    for stamp, value in labels.items():
        value = value if isinstance(value, str) else None
        if value is not None and value != prev:
            out.append((stamp, value, prev))
        if value is not None:
            prev = value
    return out


def transition_leads(season, weather, before=12, after=12):
    """For each season entry, how many months earlier the weather entered the same quadrant."""
    weather_runs = [r for r in spells(weather) if r[0] in QUADRANTS]
    rows = []
    for stamp, value, prev in entries(season):
        if prev is None:
            continue
        lo, hi = stamp - pd.DateOffset(months=before), stamp + pd.DateOffset(months=after)
        cands = [r for r in weather_runs if r[0] == value and r[1] <= hi and r[2] >= lo]
        if not cands:
            rows.append({'d': stamp.strftime('%Y-%m'), 'to': value, 'from': prev, 'lead': None})
            continue
        best = min(cands, key=lambda r: abs((stamp.year - r[1].year) * 12 + stamp.month - r[1].month))
        lead = (stamp.year - best[1].year) * 12 + stamp.month - best[1].month
        rows.append({'d': stamp.strftime('%Y-%m'), 'to': value, 'from': prev, 'lead': int(lead)})
    leads = [r['lead'] for r in rows if r['lead'] is not None]
    summary = {'entries': len(rows), 'matched': len(leads),
               'median_lead': None if not leads else float(np.median(leads)),
               'q1': None if not leads else float(np.percentile(leads, 25)),
               'q3': None if not leads else float(np.percentile(leads, 75)),
               'share_market_first': None if not leads else round(float(np.mean([x > 0 for x in leads])), 3)}
    return summary, rows


def placebo_leads(season, weather, shifts=(36, 60, 84, 108, 132, 156)):
    """Same matching on a circularly shifted weather series, which has no real link to the season."""
    values = weather.to_numpy(dtype=object)
    rows = []
    for k in shifts:
        shifted = pd.Series(np.roll(values, k), index=weather.index)
        summary, _ = transition_leads(season, shifted)
        rows.append(summary)
    meds = [r['median_lead'] for r in rows if r['median_lead'] is not None]
    firsts = [r['share_market_first'] for r in rows if r['share_market_first'] is not None]
    matched = [r['matched'] / r['entries'] for r in rows if r['entries']]
    return {'median_lead': round(float(np.mean(meds)), 2), 'share_market_first': round(float(np.mean(firsts)), 3),
            'matched_share': round(float(np.mean(matched)), 3), 'shifts': list(shifts)}


# ---------------------------------------------------------------- seasons

def duration_table(season):
    runs = [r for r in spells(season) if r[0] in QUADRANTS]
    # The last run is still open; keep it out of the distribution.
    closed = runs[:-1]
    out = {}
    for q in QUADRANTS:
        months = [r[3] for r in closed if r[0] == q]
        if months:
            out[q] = {'n': len(months), 'median': float(np.median(months)),
                      'q1': float(np.percentile(months, 25)), 'q3': float(np.percentile(months, 75)),
                      'max': int(max(months))}
    open_run = runs[-1] if runs else None
    return out, None if open_run is None else {'regime': open_run[0], 'from': open_run[1].strftime('%Y-%m'),
                                               'months': open_run[3]}


def transition_stats(season):
    s = season.dropna()
    same = float((s == s.shift(1)).iloc[1:].mean())
    changes = [(a, b) for a, b in zip(s.shift(1).iloc[1:], s.iloc[1:]) if a != b]
    forward = sum(NEXT[a] == b for a, b in changes)
    backward = sum(PREV[a] == b for a, b in changes)
    pairs = {}
    for a, b in changes:
        pairs[f'{a}>{b}'] = pairs.get(f'{a}>{b}', 0) + 1
    return {'p_same_next_month': round(same, 3), 'changes': len(changes),
            'book_order': forward, 'reverse': backward, 'diagonal': len(changes) - forward - backward,
            'pairs': pairs}


def false_springs(labels, horizons=(3, 6)):
    """Share of changes X→Y that return to X within N months, by pair."""
    s = labels.where(labels.apply(lambda v: isinstance(v, str))).ffill().dropna()
    stamps = list(s.index)
    values = list(s)
    rows = {}
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if a == b:
            continue
        key = f'{a}>{b}'
        row = rows.setdefault(key, {'n': 0, **{f'back_{h}': 0 for h in horizons}, **{f'known_{h}': 0 for h in horizons}})
        row['n'] += 1
        for h in horizons:
            window = values[i + 1:i + 1 + h]
            if len(window) < h and a not in window:
                continue
            row[f'known_{h}'] += 1
            if a in window:
                row[f'back_{h}'] += 1
    total = {'n': sum(r['n'] for r in rows.values())}
    for h in horizons:
        known = sum(r[f'known_{h}'] for r in rows.values())
        back = sum(r[f'back_{h}'] for r in rows.values())
        total[f'rate_{h}'] = round(back / known, 3) if known else None
    for r in rows.values():
        for h in horizons:
            r[f'rate_{h}'] = round(r[f'back_{h}'] / r[f'known_{h}'], 3) if r[f'known_{h}'] else None
    return total, rows


def next_season_signals(season, weather_frame, v3_frame, horizon=6):
    """Fire = first month a candidate signal turns on inside season X. Hit = season X→NEXT[X] within horizon."""
    idx = season.dropna().index
    weather = weather_frame['regime'].reindex(idx)
    gm, lm = weather_frame['G'].reindex(idx), weather_frame['L'].reindex(idx)
    dg = v3_frame['G'].reindex(idx) - v3_frame['G'].reindex(idx).shift(3)
    dl = v3_frame['L'].reindex(idx) - v3_frame['L'].reindex(idx).shift(3)
    s = season.reindex(idx)
    target = s.map(NEXT)

    def axis_flip(series_g, series_l, threshold):
        out = []
        for stamp, cur in s.items():
            axis, sign = FLIP[cur]
            value = (series_g if axis == 'G' else series_l).loc[stamp]
            out.append(bool(np.isfinite(value) and sign * value > threshold))
        return pd.Series(out, index=idx)

    s1 = (weather == target)
    candidates = {
        'S1 시장 국면 = 다음 계절': s1,
        'S2 시장 국면 = 다음 계절, 2개월 연속': s1 & s1.shift(1, fill_value=False),
        'S3 바뀌어야 할 시장 축의 부호가 이미 바뀜': axis_flip(gm, lm, 0.0),
        'S4 바뀌어야 할 v3 축의 3개월 변화가 그 방향 (>0.05)': axis_flip(dg, dl, 0.05),
    }
    values = list(s)
    future_next = []
    for i in range(len(values)):
        window = values[i + 1:i + 1 + horizon]
        future_next.append(NEXT[values[i]] in window if len(window) == horizon or NEXT[values[i]] in window else None)
    future_next = pd.Series(future_next, index=idx, dtype=object)
    base = future_next.dropna().astype(bool).mean()
    out = {'horizon': horizon, 'base_rate': round(float(base), 3), 'signals': {}}
    transitions = [(stamp, b) for stamp, b, a in entries(s) if a is not None and NEXT[a] == b]
    for name, sig in candidates.items():
        sig = sig.fillna(False).astype(bool)
        fresh = sig & ~(sig.shift(1, fill_value=False) & (s == s.shift(1)))
        fires = [(stamp, future_next.loc[stamp]) for stamp in idx[fresh.to_numpy()] if future_next.loc[stamp] is not None]
        hits = sum(bool(h) for _, h in fires)
        recall_hits = 0
        for stamp, b in transitions:
            lo = stamp - pd.DateOffset(months=horizon)
            window = fresh.loc[lo:stamp - pd.DateOffset(months=1)]
            prev_season = s.loc[lo:stamp - pd.DateOffset(months=1)]
            if (window & (prev_season == PREV[b])).any():
                recall_hits += 1
        months_on = sig[future_next.notna().to_numpy()]
        on_rate = future_next[future_next.notna()][months_on.to_numpy()].astype(bool).mean() if months_on.any() else np.nan
        out['signals'][name] = {'fires': len(fires), 'hits': hits,
                                'precision': round(hits / len(fires), 3) if fires else None,
                                'recall': round(recall_hits / len(transitions), 3) if transitions else None,
                                'transitions': len(transitions),
                                'months_on': int(months_on.sum()),
                                'hit_rate_months_on': None if not np.isfinite(on_rate) else round(float(on_rate), 3)}
    return out


# ---------------------------------------------------------------- returns and rules

def weather_weights(frame):
    regime = frame['regime']
    return regime.map(rr.WEIGHTS).ffill().fillna(rr.START_WEIGHT)


def combine_positions(season_w, weather_w, index, tilt=TILT):
    """Season weight lags two months (published a month late); weather lags one (known at month end)."""
    base = season_w.shift(2).reindex(index).ffill()
    weather = weather_w.shift(1).reindex(index).ffill()
    return {
        '계절만 (v3 확인)': base,
        '날씨만 (시장 국면)': weather,
        f'계절 + 날씨 (차이의 {int(tilt * 100)}%만 이동)': base + tilt * (weather - base),
    }


def rule_backtests(returns, positions, periods):
    out = {}
    for period, (lo, hi) in periods.items():
        rows = {}
        common = None
        for pos in positions.values():
            idx = pos.dropna().index
            common = idx if common is None else common.intersection(idx)
        part = returns.loc[common].dropna(subset=['stock'])
        part = part.loc[lo:hi] if hi else part.loc[lo:]
        rows['주식 100%'] = rr.performance(part['stock'], part['cash'])
        rows['60/40'] = rr.performance(.6 * part['stock'] + .4 * part['bond10'], part['cash'])
        for name, pos in positions.items():
            w = pos.reindex(part.index)
            port = w * part['stock'] + (1 - w) * part['bond10']
            perf = rr.performance(port, part['cash'])
            const = w.mean() * part['stock'] + (1 - w.mean()) * part['bond10']
            gap = (port - const).dropna()
            nw = rr.newey_west_means(gap.to_numpy(), np.zeros(len(gap)), [0.0], 12)
            perf['avg_equity'] = round(float(w.mean()) * 100, 1)
            perf['turnover'] = round(float(w.diff().abs().sum() / (len(w) / 12)) * 100, 1)
            if nw is not None and nw[1][0, 0] > 0:
                perf['gap_vs_const'] = round(float(nw[0][0]) * 1200, 2)
                perf['gap_t'] = round(float(nw[0][0] / np.sqrt(nw[1][0, 0])), 2)
            rows[name] = perf
        out[period] = rows
    return out


def return_tables(returns, labels_by_name, logp, loge):
    out = {}
    for h in (1, 3, 6, 12):
        fwd_by_lag = {}
        for name, (labels, groups, extra, lag) in labels_by_name.items():
            if lag not in fwd_by_lag:
                fwd = rr.forward(returns, h, lag)
                fwd['pe'] = rr.forward_log_change(logp - loge, h, lag)
                fwd['eps'] = rr.forward_log_change(loge, h, lag)
                fwd_by_lag[lag] = fwd
            fwd = fwd_by_lag[lag]
            cols = ('stock_cash', 'stock_bond', 'growth_value', 'cyc_def', 'mom', 'pe', 'eps') if h in (6, 12) else ('stock_cash',)
            for col in cols:
                out[f'{col}/{h}/{name}'] = rr.regime_stats(fwd[col], labels, groups, h, boot=h in (6, 12), extra=extra)
    return out


# ---------------------------------------------------------------- run

def app_example(frame, season, months=12):
    rows = []
    for stamp in frame.index[-months:]:
        r = frame.loc[stamp]
        rows.append({'d': stamp.strftime('%Y-%m'), 'g_m': round(float(r.G), 3), 'l_m': round(float(r.L), 3),
                     'weather': r.regime if isinstance(r.regime, str) else None,
                     'season': season.get(stamp) if isinstance(season.get(stamp), str) else None})
    return rows


def run(cache, end_month):
    signals = market_signals(cache)
    pub = rr.published_frame()
    season = season_series(pub)
    returns = rr.monthly_returns(cache)
    logp, loge = rr.shiller_pe(cache)
    result = {'signal_end': pub.index[-1].strftime('%Y-%m'), 'specs': {}}
    frames = {}
    for spec in SPECS:
        frame = market_frame(signals, spec, end_month)
        frames[spec] = frame
        common = frame.index.intersection(pub.index)
        weather = frame['regime']
        lead_summary, lead_rows = transition_leads(season.loc[common[0]:], weather)
        result['specs'][spec] = {
            'from': frame.index[0].strftime('%Y-%m'), 'months': len(frame),
            'xcorr_g': cross_correlation(frame['G'], pub['G']),
            'xcorr_l': cross_correlation(frame['L'], pub['L']),
            'sign_g': gb.best_shift(np.sign(frame['G']), np.sign(pub['G'])),
            'sign_l': gb.best_shift(np.sign(frame['L']), np.sign(pub['L'])),
            'quadrant_agree_0': round(float((weather.reindex(common) == season.reindex(common)).mean()), 3),
            'leads': lead_summary, 'lead_rows': lead_rows,
            'placebo': placebo_leads(season.loc[common[0]:], weather),
            'durations': duration_table(weather.ffill())[0],
            'false_spring': false_springs(weather)[0],
            'transitions': transition_stats(weather.ffill()),
        }
    # Season-side statistics from the published v3 regime.
    result['season'] = {'durations': duration_table(season)[0], 'open': duration_table(season)[1],
                        'transitions': transition_stats(season),
                        'false_spring_total': false_springs(season)[0],
                        'false_spring_pairs': false_springs(season)[1]}
    for spec in SPECS:
        result['specs'][spec]['next_signals'] = next_season_signals(season.loc[frames[spec].index[0]:], frames[spec], pub)

    labels = {'v3 확인 (지연 1)': (rr.labels_from(pub, 'confirmed'), QUADRANTS, ('unconfirmed',), 1)}
    for spec in SPECS:
        labels[f'{spec} 시장 (지연 0)'] = (frames[spec]['regime'].fillna('neutral'), QUADRANTS, ('neutral',), 0)
    start = max(f.index[0] for f in frames.values())
    trimmed = {k: (v[0].loc[start:], v[1], v[2], v[3]) for k, v in labels.items()}
    trimmed['v3 확인 (지연 1, 전 기간)'] = labels['v3 확인 (지연 1)']
    result['returns'] = return_tables(returns, trimmed, logp, loge)
    result['returns_from'] = start.strftime('%Y-%m')

    season_w = rr.target_weights(pub, 'confirmed')
    periods = {'전 기간': (start.strftime('%Y-%m-%d'), None), '1990–2019': ('1990-01-01', '2019-12-01'),
               '2023-04~': ('2023-04-01', None)}
    result['rules'] = {}
    for spec in SPECS:
        pos = combine_positions(season_w, weather_weights(frames[spec]), returns.index)
        result['rules'][spec] = rule_backtests(returns, pos, periods)
    result['app_example'] = app_example(frames['M1'], season)
    return result


def _pct(x):
    return '—' if x is None else f'{x * 100:.0f}%'


def markdown(result):
    lines = [f"signal_end={result['signal_end']} returns_from={result['returns_from']}", '']
    for spec, r in result['specs'].items():
        lines.append(f"## {spec} (from {r['from']}, {r['months']} months)")
        lines.append(f"xcorr G best k={r['xcorr_g']['best']} corr={r['xcorr_g']['corr_at_best']} (k0 {r['xcorr_g']['corr_at_0']}); "
                     f"L best k={r['xcorr_l']['best']} corr={r['xcorr_l']['corr_at_best']} (k0 {r['xcorr_l']['corr_at_0']})")
        lines.append(f"sign G {r['sign_g']}; sign L {r['sign_l']}; quadrant agree at 0: {r['quadrant_agree_0']}")
        lines.append(f"leads {r['leads']} placebo {r['placebo']}")
        lines.append('lead rows ' + ' '.join(f"{x['d']}:{KO[x['to']]}:{x['lead']}" for x in r['lead_rows']))
        lines.append(f"weather durations {json.dumps(r['durations'], ensure_ascii=False)}")
        lines.append(f"weather false spring {r['false_spring']}; transitions {r['transitions']}")
        lines.append(f"next signals {json.dumps(r['next_signals'], ensure_ascii=False)}")
        lines.append('')
    lines += ['## season', json.dumps(result['season'], ensure_ascii=False), '', '## returns']
    for key, st in result['returns'].items():
        if st is None:
            continue
        lines.append(f"{key} p={st['p_equal']} " + ' | '.join(
            f"{KO[g]} {v['mean']:+.1f}" + (f" [{v['ci'][0]:+.1f},{v['ci'][1]:+.1f}]" if v['ci'] else '') + f" n{v['n']}"
            for g, v in st['groups'].items()))
    lines += ['', '## rules']
    for spec, block in result['rules'].items():
        for period, rows in block.items():
            lines.append(f'### {spec} {period}')
            for name, p in rows.items():
                gap = '—' if 'gap_vs_const' not in p else f"{p['gap_vs_const']:+.2f} ({p['gap_t']})"
                lines.append(f"| {name} | {p['months']} | {p['cagr']:.2f}% | {p['vol']:.1f}% | {p['sharpe']} | {p['mdd']:.1f}% | "
                             f"{p['total']:.1f}% | {p.get('avg_equity', '—')} | {p.get('turnover', '—')} | {gap} |")
    lines += ['', '## app example', json.dumps(result['app_example'], ensure_ascii=False)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache', default=os.environ.get('GL_CACHE', '/tmp/gl-cache'))
    parser.add_argument('--end', default=None)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    end = args.end or pd.Timestamp.today().to_period('M').to_timestamp().strftime('%Y-%m-%d')
    result = run(args.cache, end)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(markdown(result))


if __name__ == '__main__':
    main()
