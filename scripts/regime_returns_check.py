#!/usr/bin/env python3
"""Do the v3 quadrants separate later asset returns? Analysis only.

The signal for month t is known about a month later, so by default the forward
window starts at the end of month t+1 (`--lag 1`): a 3-month result for July is
the return over September–November. `--lag 0` gives the window that starts at
the end of month t, as docs/regime-forward-returns.md does.

Sources, all free and primary:
- Kenneth R. French Data Library: market total return (Mkt-RF + RF), HML,
  RMW, momentum, 10 value-weighted industry portfolios.
- FRED: GS5, GS10 (constant-maturity yields, monthly average) for par-bond
  return approximations, TB3MS for cash.
- S&P 500 price and trailing earnings from the Shiller data mirror on
  datahub (earnings end 2023-06), for the price/earnings split only.

Statistics: means with Newey-West (HAC) standard errors, a joint HAC Wald test
that the four quadrant means are equal, non-overlapping subsamples, and a
moving-block bootstrap. Everything uses the current revised data, so it is an
after-the-fact check, not a real-time record.

Usage: python3 scripts/regime_returns_check.py [--cache DIR] [--json OUT] [--lag 1]
"""
import argparse
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import growth_bias_check as gb  # noqa: E402

FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip'
SHILLER_MIRROR = 'https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv'
QUADRANTS = ('expansion', 'liquidity', 'adjustment', 'defense')
KO = {'expansion': '실적', 'liquidity': '금융', 'adjustment': '역금융', 'defense': '역실적',
      'neutral': '경계·중립', 'unconfirmed': '미확인'}
HORIZONS = (1, 3, 6, 12)
WEIGHTS = {'expansion': .70, 'liquidity': .65, 'adjustment': .60, 'defense': .50}
START_WEIGHT = .60
CYCLICAL = ('Durbl', 'Manuf', 'Shops')
DEFENSIVE = ('NoDur', 'Hlth', 'Utils')
PERIODS = {'1972–2026': ('1972-01-01', None), '1990–2019': ('1990-01-01', '2019-12-01'),
           '2023-04~': ('2023-04-01', None)}
SERIES_KO = {
    'stock': '미국 주식 (CRSP 시가총액가중 총수익)',
    'bond10': '10년 국채 (GS10 par-bond 근사)',
    'bond5': '5년 국채 (GS5 par-bond 근사)',
    'cash': '현금 (TB3MS)',
    'stock_cash': '주식 − 현금',
    'stock_bond': '주식 − 10년 국채',
    'bond_cash': '10년 국채 − 현금',
    'bond5_cash': '5년 국채 − 현금',
    'growth_value': '성장 − 가치 (−HML)',
    'rmw': '고수익성 − 저수익성 (RMW)',
    'mom': '모멘텀 (Mom)',
    'cyc_def': '경기민감 − 방어 업종',
    'pe': 'S&P500 PER 변화 (log, ~2023-06)',
    'eps': 'S&P500 이익 변화 (log, ~2023-06)',
}


# ---------------------------------------------------------------- data

def parse_french(text):
    """First monthly block of a Ken French CSV, as decimals indexed by month start."""
    rows, header, started = [], None, False
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(',')]
        if not started:
            if len(cells) > 1 and cells[0] == '' and cells[1]:
                header = cells[1:]
                started = True
            continue
        if len(cells) < 2 or not cells[0].isdigit() or len(cells[0]) != 6:
            if rows:
                break
            continue
        rows.append([cells[0]] + cells[1:len(header) + 1])
    frame = pd.DataFrame(rows, columns=['ym'] + header)
    frame.index = pd.to_datetime(frame.pop('ym'), format='%Y%m')
    frame = frame.apply(pd.to_numeric, errors='coerce')
    return frame.mask(frame <= -99.99) / 100.0


def load_french(cache, name):
    def fetch():
        raw = _get_bytes(FRENCH.format(name))
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            member = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
            return z.read(member).decode('latin-1')
    return parse_french(gb._cached(cache, f'FF_{name}.csv', fetch))


def _get_bytes(url, timeout=90):
    import ssl
    import urllib.request
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(urllib.request.Request(url, headers=gb.UA), timeout=timeout, context=context) as r:
        return r.read()


def par_bond_returns(yields_pct, years):
    """Monthly return of a par bond bought at last month's yield and repriced at this month's.

    Semiannual coupon, fractional periods allowed. Yields are monthly averages,
    so this is an approximation of a constant-maturity total return.
    """
    y = yields_pct / 100.0
    c = y.shift(1)
    n = 2.0 * (years - 1.0 / 12.0)
    disc = (1 + y / 2) ** (-n)
    price = (c / y) * (1 - disc) + disc
    return (price - 1.0 + c / 12.0).where(c.notna() & y.notna())


def monthly_returns(cache):
    ff = load_french(cache, 'F-F_Research_Data_Factors')
    ff5 = load_french(cache, 'F-F_Research_Data_5_Factors_2x3')
    mom = load_french(cache, 'F-F_Momentum_Factor')
    ind = load_french(cache, '10_Industry_Portfolios')
    gs10 = gb.to_monthly(gb.load_fred(cache, 'GS10'))
    gs5 = gb.to_monthly(gb.load_fred(cache, 'GS5'))
    tb3 = gb.to_monthly(gb.load_fred(cache, 'TB3MS'))
    out = pd.DataFrame({
        'stock': ff['Mkt-RF'] + ff['RF'],
        'bond10': par_bond_returns(gs10, 10),
        'bond5': par_bond_returns(gs5, 5),
        # TB3MS is a monthly average; last month's rate is what a buyer earned this month.
        'cash': tb3.shift(1) / 100.0 / 12.0,
        'growth_value': -ff['HML'],
        'rmw': ff5['RMW'],
        'mom': mom.iloc[:, 0],
        'cyc_def': ind[list(CYCLICAL)].mean(axis=1) - ind[list(DEFENSIVE)].mean(axis=1),
    })
    return out.loc['1960-01-01':]


def shiller_pe(cache):
    text = gb._cached(cache, 'shiller_mirror.csv', lambda: gb._get(SHILLER_MIRROR))
    frame = pd.read_csv(io.StringIO(text), parse_dates=['Date']).set_index('Date')
    frame = frame[frame['Earnings'] > 0]
    return np.log(frame['SP500']), np.log(frame['Earnings'])


# ---------------------------------------------------------------- forward windows

SPREADS = ('growth_value', 'rmw', 'mom', 'cyc_def')


def forward(returns, h, lag):
    """Value at t = outcome over months t+1+lag .. t+h+lag.

    Assets compound. Long-short spreads add, since they are not holdings.
    Pair columns are differences of compounded legs.
    """
    out = {}
    for col in returns.columns:
        r = returns[col]
        if col in SPREADS:
            total = r.rolling(h).sum()
        else:
            total = np.exp(np.log1p(r).rolling(h).sum()) - 1.0
        out[col] = total.shift(-(h + lag))
    out = pd.DataFrame(out)
    out['stock_cash'] = out['stock'] - out['cash']
    out['stock_bond'] = out['stock'] - out['bond10']
    out['bond_cash'] = out['bond10'] - out['cash']
    out['bond5_cash'] = out['bond5'] - out['cash']
    return out


def forward_log_change(level, h, lag):
    level = level.resample('MS').mean()
    return (level.shift(-(h + lag)) - level.shift(-lag))


# ---------------------------------------------------------------- statistics

def newey_west_means(y, labels, groups, lags):
    """Group means and their HAC covariance from y = X·b with group dummies."""
    mask = np.isfinite(y)
    y = np.asarray(y, dtype=float)[mask]
    labels = np.asarray(labels)[mask]
    x = np.column_stack([(labels == g).astype(float) for g in groups])
    keep = x.sum(axis=1) > 0
    x, y = x[keep], y[keep]
    xtx = x.T @ x
    if np.any(np.diag(xtx) == 0):
        return None
    inv = np.linalg.inv(xtx)
    b = inv @ x.T @ y
    u = y - x @ b
    xu = x * u[:, None]
    s = xu.T @ xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xu[lag:].T @ xu[:-lag]
        s += w * (g + g.T)
    return b, inv @ s @ inv, len(y)


def chi2_sf(x, k):
    """Upper tail of a chi-square with integer k degrees of freedom."""
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        term, total = 1.0, 1.0
        for i in range(1, k // 2):
            term *= (x / 2) / i
            total += term
        return math.exp(-x / 2) * total
    total = math.erfc(math.sqrt(x / 2))
    term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    for i in range(1, (k + 1) // 2):
        total += term
        term *= x / (2 * i + 1)
    return min(1.0, total)


def normal_p(t):
    return math.erfc(abs(t) / math.sqrt(2))


def wald_equal(b, cov):
    k = len(b)
    r = np.zeros((k - 1, k))
    for i in range(k - 1):
        r[i, i], r[i, i + 1] = 1.0, -1.0
    d = r @ b
    stat = float(d @ np.linalg.pinv(r @ cov @ r.T) @ d)
    return stat, chi2_sf(stat, k - 1)


def block_bootstrap_means(y, labels, groups, block=24, reps=2000, seed=7):
    """Moving-block bootstrap of the monthly (label, outcome) sequence. Returns 2.5/97.5 pct per group
    and for each pair difference."""
    mask = np.isfinite(y)
    y = np.asarray(y, dtype=float)[mask]
    labels = np.asarray(labels)[mask]
    n = len(y)
    rng = np.random.default_rng(seed)
    starts_max = n - block
    draws = {g: [] for g in groups}
    diffs = {}
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, starts_max + 1, n // block + 1)])[:n]
        yy, ll = y[idx], labels[idx]
        means = {}
        for g in groups:
            pick = yy[ll == g]
            means[g] = pick.mean() if pick.size else np.nan
            draws[g].append(means[g])
        for i, a in enumerate(groups):
            for bkey in groups[i + 1:]:
                diffs.setdefault((a, bkey), []).append(means[a] - means[bkey])
    ci = {g: tuple(np.nanpercentile(v, [2.5, 97.5])) for g, v in draws.items()}
    dci = {k: tuple(np.nanpercentile(v, [2.5, 97.5])) for k, v in diffs.items()}
    return ci, dci


def non_overlapping(y, labels, groups, h, offset=0):
    """Every h-th calendar month, so windows do not overlap. Group means and counts."""
    frame = pd.concat({'y': y, 'g': labels}, axis=1, sort=True).iloc[offset::h].dropna()
    means = {g: (float(frame.y[frame.g == g].mean()), int((frame.g == g).sum())) for g in groups}
    return means


def regime_stats(outcome, labels, groups, h, boot=True, extra=()):
    """`extra` labels stay in the regression and the bootstrap so the monthly
    sequence has no calendar gaps, but they are left out of the Wald test."""
    everything = tuple(groups) + tuple(extra)
    aligned = pd.concat({'y': outcome, 'g': labels}, axis=1, sort=True).dropna()
    aligned = aligned[aligned.g.isin(everything)]
    y, g = aligned.y.to_numpy(dtype=float), aligned.g.to_numpy()
    nw = newey_west_means(y, g, everything, max(h, 6))
    if nw is None:
        return None
    b, cov, _ = nw
    k = len(groups)
    se = np.sqrt(np.diag(cov))
    wald, p = wald_equal(b[:k], cov[:k, :k])
    out = {'n': int(np.isin(g, groups).sum()), 'wald': round(wald, 2), 'p_equal': round(p, 4), 'groups': {}}
    ci, dci = block_bootstrap_means(y, g, list(everything)) if boot else ({}, {})
    dci = {key: val for key, val in dci.items() if key[0] in groups and key[1] in groups}
    nov = non_overlapping(outcome, labels, groups, h)
    for i, name in enumerate(groups):
        pick = y[g == name]
        out['groups'][name] = {
            'n': int(pick.size), 'mean': round(float(b[i]) * 100, 2),
            'median': round(float(np.median(pick)) * 100, 2) if pick.size else None,
            'hit': round(float((pick > 0).mean()), 3) if pick.size else None,
            't_hac': round(float(b[i] / se[i]), 2) if se[i] > 0 else None,
            'ci': None if name not in ci else [round(v * 100, 2) for v in ci[name]],
            'nonoverlap_mean': round(nov[name][0] * 100, 2) if nov[name][1] else None,
            'nonoverlap_n': nov[name][1],
        }
    out['diff_ci'] = {f'{a}-{b_}': [round(v * 100, 2) for v in val] for (a, b_), val in dci.items()}
    return out


# ---------------------------------------------------------------- labels and strategy

def labels_from(frame, mode):
    """'all': regime when set, else 'neutral'. 'confirmed': regime only on confirmed months."""
    regime = frame['regime'].where(frame['regime'].apply(lambda v: isinstance(v, str)))
    if mode == 'confirmed':
        return regime.where(frame['c'].astype(bool), 'unconfirmed')
    return regime.fillna('neutral')


def sign_quadrant(frame):
    return pd.Series([gb.v2.quadrant(g, l) for g, l in zip(frame.G, frame.L)], index=frame.index)


def target_weights(frame, mode):
    """Equity weight decided at the end of each signal month."""
    regime = frame['regime'].where(frame['regime'].apply(lambda v: isinstance(v, str)))
    if mode == 'confirmed':
        chosen = regime.where(frame['c'].astype(bool))
    elif mode == 'all':
        chosen = regime.fillna(sign_quadrant(frame))
    else:
        raise ValueError(mode)
    weights = chosen.map(WEIGHTS).ffill().fillna(START_WEIGHT)
    return weights


def strategy_returns(returns, weights, lag, other='bond10'):
    """Weight of signal month t applies to the return of month t+1+lag."""
    position = weights.shift(1 + lag).reindex(returns.index)
    position = position.ffill()
    port = position * returns['stock'] + (1 - position) * returns[other]
    return port.where(position.notna()), position


def performance(r, cash):
    r = r.dropna()
    if r.empty:
        return None
    cash = cash.reindex(r.index).fillna(0)
    growth = (1 + r).cumprod()
    years = len(r) / 12.0
    excess = r - cash
    vol = float(r.std() * math.sqrt(12))
    return {'months': len(r), 'cagr': round(float(growth.iloc[-1] ** (1 / years) - 1) * 100, 2),
            'vol': round(vol * 100, 2),
            'sharpe': round(float(excess.mean() * 12 / (excess.std() * math.sqrt(12))), 2) if excess.std() > 0 else None,
            'mdd': round(float((growth / growth.cummax() - 1).min()) * 100, 1),
            'total': round(float(growth.iloc[-1] - 1) * 100, 1)}


def backtests(returns, frames, lag):
    """Benchmarks use the same months as the strategies (the first 1+lag months have no position)."""
    out = {}
    first = returns.index[returns.index >= pd.Timestamp(PERIODS['1972–2026'][0])][1 + lag]
    for period, (lo, hi) in PERIODS.items():
        lo = max(pd.Timestamp(lo), first)
        part = (returns.loc[lo:hi] if hi else returns.loc[lo:]).dropna(subset=['stock'])
        rows = {}
        rows['주식 100%'] = performance(part['stock'], part['cash'])
        rows['60/40 (주식/10년 국채)'] = performance(.6 * part['stock'] + .4 * part['bond10'], part['cash'])
        for name, (frame, mode) in frames.items():
            weights = target_weights(frame, mode)
            port, position = strategy_returns(returns, weights, lag)
            port = port.loc[part.index]
            perf = performance(port, part['cash'])
            if perf:
                pos = position.loc[part.index].dropna()
                perf['avg_equity'] = round(float(pos.mean()) * 100, 1)
                const = pos.mean() * part['stock'] + (1 - pos.mean()) * part['bond10']
                gap = (port - const).loc[pos.index].dropna()
                nw = newey_west_means(gap.to_numpy(), np.zeros(len(gap)), [0.0], 12)
                if nw is not None:
                    perf['gap_vs_const'] = round(float(nw[0][0]) * 1200, 2)
                    perf['gap_t'] = round(float(nw[0][0] / math.sqrt(nw[1][0, 0])), 2)
                rows[name] = perf
                rows[name + ' · 같은 평균 비중 고정'] = performance(const.loc[pos.index], part['cash'])
        out[period] = rows
    return out


def recent_defense(returns, frame, lag, start='2023-04-01'):
    """Asset returns over the months whose position came from a confirmed defense signal."""
    regime = frame['regime'].where(frame['regime'].apply(lambda v: isinstance(v, str)))
    confirmed_def = (regime == 'defense') & frame['c'].astype(bool)
    held = confirmed_def.shift(1 + lag).reindex(returns.index).fillna(False).astype(bool)
    part = returns.loc[start:].dropna(subset=['stock'])
    held = held.loc[part.index]
    def cum(s):
        s = s.dropna()
        return round(float((1 + s).prod() - 1) * 100, 1) if len(s) else None
    return {'months': int(held.sum()), 'of': len(part),
            'from': part.index[0].strftime('%Y-%m'), 'to': part.index[-1].strftime('%Y-%m'),
            'stock_in_defense': cum(part['stock'][held]), 'bond10_in_defense': cum(part['bond10'][held]),
            'cash_in_defense': cum(part['cash'][held]),
            'stock_all': cum(part['stock']), 'bond10_all': cum(part['bond10']), 'cash_all': cum(part['cash'])}


# ---------------------------------------------------------------- report

def published_frame():
    _, months = gb.published()
    months = months.rename(columns={'g': 'G', 'l': 'L'})
    months['regime'] = months['regime'].where(months['regime'].apply(lambda v: isinstance(v, str)))
    return months[['G', 'L', 'regime', 'c', 'p']]


def run(cache, end_month, lag, variants=('a4', 'f2')):
    returns = monthly_returns(cache)
    pub = published_frame()
    frames = {'published': pub}
    if variants:
        frames.update(gb.variant_regimes(cache, end_month, variants))
    logp, loge = shiller_pe(cache)

    result = {'lag': lag, 'data_end': returns['stock'].dropna().index[-1].strftime('%Y-%m'),
              'signal_end': pub.index[-1].strftime('%Y-%m'), 'tables': {}, 'label_counts': {}}
    for key, frame in frames.items():
        for mode in ('confirmed', 'all'):
            labels = labels_from(frame, mode)
            result['label_counts'][f'{key}/{mode}'] = {KO.get(k, k): int(v) for k, v in labels.value_counts().items()}
    pub_conf = labels_from(pub, 'confirmed')
    pub_all = labels_from(pub, 'all')
    for h in HORIZONS:
        fwd = forward(returns, h, lag)
        fwd['pe'] = forward_log_change(logp - loge, h, lag)
        fwd['eps'] = forward_log_change(loge, h, lag)
        for col in SERIES_KO:
            for mode, labels in (('confirmed', pub_conf), ('all', pub_all)):
                groups = QUADRANTS if mode == 'confirmed' else QUADRANTS + ('neutral',)
                boot = h in (6, 12) or col in ('stock_cash', 'stock_bond')
                extra = ('unconfirmed',) if mode == 'confirmed' else ()
                stats = regime_stats(fwd[col], labels, groups, h, boot=boot, extra=extra)
                result['tables'][f'{col}/{h}/{mode}/published'] = stats
        for key in variants:
            labels = labels_from(frames[key], 'confirmed')
            for col in ('stock_cash', 'stock_bond'):
                result['tables'][f'{col}/{h}/confirmed/{key}'] = regime_stats(
                    fwd[col], labels, QUADRANTS, h, boot=False, extra=('unconfirmed',))

    strat_frames = {}
    for key, frame in frames.items():
        name = {'published': 'v3 발행', 'a4': 'PR #10 B (a4)', 'f2': 'PR #10 f2'}.get(key, key)
        strat_frames[f'{name} · 확인만'] = (frame, 'confirmed')
        strat_frames[f'{name} · 잠정·경계 포함'] = (frame, 'all')
    result['backtest'] = backtests(returns, strat_frames, lag)
    cash_frames = {k: v for k, v in strat_frames.items() if k.startswith('v3 발행')}
    result['backtest_cash'] = {}
    for period, (lo, hi) in PERIODS.items():
        part = returns.loc[lo:hi] if hi else returns.loc[lo:]
        rows = {}
        for name, (frame, mode) in cash_frames.items():
            port, _ = strategy_returns(returns, target_weights(frame, mode), lag, other='cash')
            rows[name + ' · 나머지 현금'] = performance(port.loc[part.index], part['cash'])
        result['backtest_cash'][period] = rows
    result['recent_defense'] = {key: recent_defense(returns, frame, lag) for key, frame in frames.items()}
    return result


def _f(v, digits=1, sign=True):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    return f'{v:+.{digits}f}' if sign else f'{v:.{digits}f}'


def markdown(result):
    lines = [f"lag={result['lag']} data_end={result['data_end']} signal_end={result['signal_end']}", '',
             '## label counts', json.dumps(result['label_counts'], ensure_ascii=False), '']
    for key, stats in result['tables'].items():
        if stats is None:
            continue
        col, h, mode, src = key.split('/')
        groups = list(stats['groups'])
        lines.append(f"### {SERIES_KO[col]} · {h}개월 · {mode} · {src}  (Wald {stats['wald']}, p={stats['p_equal']}, n={stats['n']})")
        lines.append('| 국면 | n | 평균 | 중위 | 양수 | HAC t | 95% 부트스트랩 | 비중첩 평균 (n) |')
        lines.append('|---|---:|---:|---:|---:|---:|---|---:|')
        for g in groups:
            r = stats['groups'][g]
            ci = '—' if not r['ci'] else f"[{_f(r['ci'][0])}, {_f(r['ci'][1])}]"
            hit = '—' if r['hit'] is None else f"{r['hit'] * 100:.0f}%"
            lines.append(f"| {KO[g]} | {r['n']} | {_f(r['mean'])} | {_f(r['median'])} | {hit} | {_f(r['t_hac'], 2)} | {ci} | "
                         f"{_f(r['nonoverlap_mean'])} ({r['nonoverlap_n']}) |")
        if stats['diff_ci']:
            lines.append('차이 95%: ' + ', '.join(f"{KO[k.split('-')[0]]}−{KO[k.split('-')[1]]} [{_f(v[0])}, {_f(v[1])}]"
                                                for k, v in stats['diff_ci'].items()))
        lines.append('')
    for title, block in (('backtest', result['backtest']), ('backtest_cash', result['backtest_cash'])):
        lines.append(f'## {title}')
        for period, rows in block.items():
            lines.append(f'### {period}')
            lines.append('| 전략 | 개월 | CAGR | 변동성 | 샤프 | MDD | 누적 | 평균 주식 비중 | 고정 대비 연 %p (HAC t) |')
            lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|')
            for name, p in rows.items():
                if p:
                    gap = '—' if 'gap_vs_const' not in p else f"{p['gap_vs_const']:+.2f} ({p['gap_t']})"
                    lines.append(f"| {name} | {p['months']} | {p['cagr']:.2f}% | {p['vol']:.1f}% | {p['sharpe']} | "
                                 f"{p['mdd']:.1f}% | {p['total']:.1f}% | {p.get('avg_equity', '—')} | {gap} |")
            lines.append('')
    lines += ['## recent defense', json.dumps(result['recent_defense'], ensure_ascii=False)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache', default=os.environ.get('GL_CACHE', '/tmp/gl-cache'))
    parser.add_argument('--end', default=None)
    parser.add_argument('--lag', type=int, default=1)
    parser.add_argument('--no-variants', action='store_true')
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    end = args.end or pd.Timestamp.today().to_period('M').to_timestamp().strftime('%Y-%m-%d')
    result = run(args.cache, end, args.lag, () if args.no_variants else ('a4', 'f2'))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(markdown(result))


if __name__ == '__main__':
    main()
