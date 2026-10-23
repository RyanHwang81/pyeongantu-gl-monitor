#!/usr/bin/env python3
"""Check the v3 growth axis against outside activity measures, then vary its inputs.

Analysis only. Nothing here changes model_v3, build.py or dist/.

Two parts:

1. Agreement. The published G and L signs (dist/gl_data.json) against
   outside series, month by month, with the lead or lag that agrees best.
   Positive k means GL leads: GL at t is compared with the comparator at t+k.
2. Sensitivity. The v3 growth score rebuilt from the same public inputs with
   one change at a time. L stays at the replicated v3 liquidity score, so a
   regime change comes from G alone.

Data comes from public CSV endpoints without a key: FRED graph CSV, the OECD
SDMX API for the composite leading indicator, and the DBnomics mirror of ISM
PMIs (2020-05 onward only; obvious scrape errors outside 30–70 are dropped).
Files are cached in --cache (default $GL_CACHE or /tmp/gl-cache). All series
are the current revised vintage, so none of this is a real-time record.

Usage: python3 scripts/growth_bias_check.py [--cache DIR] [--json OUT]
"""
import argparse
import contextlib
import io
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_v2 as v2  # noqa: E402
import model_v3 as v3  # noqa: E402

UA = {'User-Agent': 'pyeongantoo-gl-analysis/1.0'}
FRED_CSV = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={}'
OECD_CLI = ('https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,'
            '/USA.M.LI...AA...H?startPeriod=1960-01&format=csvfilewithlabels')
DBNOMICS = 'https://api.db.nomics.world/v22/series/ISM/{}?observations=1'

MODEL_SERIES = ('PERMIT', 'INDPRO', 'PAYEMS', 'UNRATE', 'ICSA', 'M2SL', 'T10Y3M', 'TOTALSL',
                'BAA10YM', 'WALCL', 'DGS2', 'DTWEXBGS', 'WTREGEN', 'RRPONTSYD')
EXTRA_SERIES = ('GDPC1', 'A191RL1Q225SBEA', 'W875RX1', 'PCESC96', 'NROU', 'CFNAIMA3', 'USSLIND', 'NFCI',
                'FEDFUNDS', 'SAHMREALTIME', 'USREC')
# Same weights as build.G_COMP. tests/test_growth_bias_check.py keeps them in sync.
BASE_WEIGHTS = {'PERMIT': .28, 'INDPRO': .24, 'PAYEMS': .24, 'UNRATE': .14, 'ICSA': .10}
COVID_STATS_GAP = ('2020-03-01', '2022-12-01')
RECENT = ('2023-01-01', '2026-09-01')
HISTORY = ('1990-01-01', '2019-12-01')
KO = {'expansion': '실적', 'liquidity': '금융', 'adjustment': '역금융', 'defense': '역실적', None: '경계·중립'}


# ---------------------------------------------------------------- data

def _get(url, timeout=90):
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout, context=context) as r:
        return r.read().decode('utf-8', 'replace')


def _cached(cache, name, fetcher):
    path = Path(cache) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fetcher(), encoding='utf-8')
    return path.read_text(encoding='utf-8')


def parse_two_column_csv(text):
    frame = pd.read_csv(io.StringIO(text))
    frame = frame.iloc[:, :2]
    frame.columns = ['date', 'value']
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    frame['date'] = pd.to_datetime(frame['date'])
    return frame.dropna().set_index('date')['value'].sort_index()


def load_fred(cache, series_id):
    return parse_two_column_csv(_cached(cache, series_id + '.csv', lambda: _get(FRED_CSV.format(series_id))))


def load_oecd_cli(cache):
    def fetch():
        frame = pd.read_csv(io.StringIO(_get(OECD_CLI)))
        frame = frame[['TIME_PERIOD', 'OBS_VALUE']].rename(columns={'TIME_PERIOD': 'date', 'OBS_VALUE': 'value'})
        return frame.sort_values('date').to_csv(index=False)
    return parse_two_column_csv(_cached(cache, 'OECD_CLI_USA.csv', fetch))


def clean_pmi(series, low=30.0, high=70.0):
    """Drop mirror values no PMI survey could produce. Returns (clean, dropped)."""
    bad = (series < low) | (series > high)
    return series[~bad], series[bad]


def load_ism(cache, dataset):
    def fetch():
        doc = json.loads(_get(DBNOMICS.format(dataset + '/pm')))['series']['docs'][0]
        return pd.DataFrame({'date': [p + '-01' for p in doc['period']], 'value': doc['value']}).to_csv(index=False)
    return clean_pmi(parse_two_column_csv(_cached(cache, f'ISM_{dataset}.csv', fetch)))


def load_all(cache):
    raw = {sid: load_fred(cache, sid) for sid in MODEL_SERIES + EXTRA_SERIES}
    raw['OECD_CLI'] = load_oecd_cli(cache)
    dropped = {}
    for key, dataset in (('ISM_MFG', 'pmi'), ('ISM_SVC', 'nm-pmi')):
        try:
            raw[key], bad = load_ism(cache, dataset)
            dropped[key] = {d.strftime('%Y-%m'): float(v) for d, v in bad.items()}
        except Exception as exc:  # the mirror is optional
            raw[key] = pd.Series(dtype=float)
            dropped[key] = {'error': type(exc).__name__}
    return raw, dropped


# ---------------------------------------------------------------- transforms

def to_monthly(series):
    return series.resample('MS').mean()


def yoy(series):
    return series.pct_change(12, fill_method=None) * 100.0


def annualized(series, months=6):
    return ((series / series.shift(months)) ** (12.0 / months) - 1.0) * 100.0


def spread_quarterly(series):
    """Put a quarterly value on all three months of its quarter. Never past the quarter."""
    series = series.dropna()
    if series.empty:
        return series
    index = pd.date_range(series.index[0], series.index[-1] + pd.DateOffset(months=2), freq='MS')
    return series.reindex(index).ffill(limit=2)


def trailing_median_gap(series, window=120, min_periods=60):
    return series - series.rolling(window, min_periods=min_periods).median()


# ---------------------------------------------------------------- model replication

def monthly_inputs(raw):
    monthly = {k: to_monthly(raw[k]) for k in MODEL_SERIES}
    netliq = v2.net_liquidity(monthly['WALCL'], monthly['WTREGEN'], monthly['RRPONTSYD'])
    return monthly, netliq


def base_growth(monthly):
    """build.G_COMP, copied so build_data does not have to run."""
    return {
        'PERMIT': {'t': yoy(monthly['PERMIT']), 'w': BASE_WEIGHTS['PERMIT']},
        'INDPRO': {'t': yoy(monthly['INDPRO']), 'w': BASE_WEIGHTS['INDPRO']},
        'PAYEMS': {'t': yoy(monthly['PAYEMS']), 'w': BASE_WEIGHTS['PAYEMS']},
        'UNRATE': {'t': -(monthly['UNRATE'] - monthly['UNRATE'].shift(12)), 'w': BASE_WEIGHTS['UNRATE']},
        'ICSA': {'t': -yoy(monthly['ICSA']), 'w': BASE_WEIGHTS['ICSA']},
    }


def _with_start(comps):
    for c in comps.values():
        c.setdefault('available_from', c['t'].first_valid_index())
    return comps


def score_axis(comps, index, method='robust'):
    """build.build_data's two-pass score: draft z, frozen rho, final composite."""
    comps = _with_start(comps)
    draft = v3.score_composite(comps, index, method, 0.0)
    rho = v3.frozen_pairwise_rho(draft.z)
    return v3.score_composite(comps, index, method, rho)


def regimes(g, l, index):
    """Filter like build_data, then run v3.classify. Returns a frame with regime columns."""
    df = pd.DataFrame({'G': g.score, 'L': l.score, 'gr': g.raw, 'lr': l.raw,
                       'owg': g.observed_ratio, 'owl': l.observed_ratio}).dropna()
    df = df[(df.index >= '1972-01-01') & (df.index < index[-1])]
    df = df[(df.owg >= .4 - 1e-10) & (df.owl >= .4 - 1e-10)]
    df['gp'] = v2.expanding_percentile(g.unit).reindex(df.index)
    df['lp'] = v2.expanding_percentile(l.unit).reindex(df.index)
    flags, _ = v3.classify(df)
    df['regime'] = [f['regime'] for f in flags]
    df['c'] = [f['c'] for f in flags]
    df['p'] = [f['p'] for f in flags]
    return df


def model_index(monthly, end):
    dates = [s.first_valid_index() for s in monthly.values() if s.first_valid_index() is not None]
    return pd.date_range(min(dates), pd.Timestamp(end), freq='MS')


@contextlib.contextmanager
def window(months):
    old = v2.WINDOW
    v2.WINDOW = months
    try:
        yield
    finally:
        v2.WINDOW = old


@contextlib.contextmanager
def stats_gap(start, end):
    """Leave [start, end] out of the median/MAD samples. Scores for those months still exist."""
    original = v2.observed_z
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)

    def patched(values, observed, method='classical'):
        excluded = (values.index >= lo) & (values.index <= hi)
        samples = values.where(observed & ~excluded)
        if method == 'robust':
            center, scale = v2._robust_center_scale(samples)
        else:
            center = samples.rolling(v2.WINDOW, min_periods=v2.MIN_OBS).mean()
            scale = samples.rolling(v2.WINDOW, min_periods=v2.MIN_OBS).std()
        center = center.where(observed).ffill()
        scale = scale.where(observed).ffill()
        return ((values - center) / scale).clip(-v2.CLAMP, v2.CLAMP)

    v2.observed_z = patched
    try:
        yield
    finally:
        v2.observed_z = original


@contextlib.contextmanager
def mad_floor(ratio):
    """Robust scale = max(MAD, ratio × standard deviation of the same window)."""
    original = v2._robust_center_scale

    def patched(samples):
        center, scale = original(samples)
        std = samples.rolling(v2.WINDOW, min_periods=v2.MIN_OBS).std()
        return center, np.fmax(scale, ratio * std).where(scale.notna() | std.notna())

    v2._robust_center_scale = patched
    try:
        yield
    finally:
        v2._robust_center_scale = original


# ---------------------------------------------------------------- variants

def growth_variants(monthly, raw):
    base = base_growth(monthly)
    gdp = spread_quarterly(raw['GDPC1'].pct_change(4, fill_method=None) * 100.0)
    income = yoy(to_monthly(raw['W875RX1']))
    services = yoy(to_monthly(raw['PCESC96']))
    unrate = monthly['UNRATE']
    gap120 = -trailing_median_gap(unrate)
    gap_nrou = -(unrate - spread_quarterly(raw['NROU']).reindex(unrate.index))

    def swap(**changes):
        out = {k: dict(v) for k, v in base.items()}
        for key, spec in changes.items():
            if spec is None:
                out.pop(key, None)
            else:
                out[key] = spec
        return out

    def ann(series, smooth=False):
        s = series.rolling(3).mean() if smooth else series
        return annualized(s, 6)

    def six_month(smooth):
        return {
            'PERMIT': {'t': ann(monthly['PERMIT'], smooth), 'w': .28},
            'INDPRO': {'t': ann(monthly['INDPRO'], smooth), 'w': .24},
            'PAYEMS': {'t': ann(monthly['PAYEMS'], smooth), 'w': .24},
            'UNRATE': {'t': -(unrate - unrate.shift(6)), 'w': .14},
            'ICSA': {'t': -ann(monthly['ICSA'], smooth), 'w': .10},
        }

    unrate_split = {'UNRATE': {'t': base['UNRATE']['t'], 'w': .07},
                    'UGAP': {'t': gap120, 'w': .07}}
    variants = {
        'base': ('v3 운영 산식 재현', base, None),
        'a1': ('+ 실질 GDP YoY (.20)', swap(GDP={'t': gdp, 'w': .20}), None),
        'a2': ('+ 실질 GDP YoY (.15) + 이전소득 제외 실질개인소득 YoY (.15)',
               swap(GDP={'t': gdp, 'w': .15}, INCOME={'t': income, 'w': .15}), None),
        'a3': ('+ 실질 서비스 소비 PCESC96 YoY (.20, 2007년~)', swap(SERV={'t': services, 'w': .20}), None),
        'a4': ('+ 실질 GDP .15 + 실질 서비스 소비 .15',
               swap(GDP={'t': gdp, 'w': .15}, SERV={'t': services, 'w': .15}), None),
        'b1': ('실업률 = 12M 변화 .07 + 120개월 중위수 대비 수준 .07', swap(**unrate_split), None),
        'b2': ('실업률 = 120개월 중위수 대비 수준만 .14', swap(UNRATE={'t': gap120, 'w': .14}), None),
        'b3': ('실업률 = 12M 변화 .07 + CBO 자연실업률 대비 .07 (점검용)',
               swap(UNRATE={'t': base['UNRATE']['t'], 'w': .07}, UGAP={'t': gap_nrou, 'w': .07}), None),
        'c1': ('기준 기간 120개월', base, ('window', 120)),
        'c2': ('기준 기간 360개월', base, ('window', 360)),
        'c3': ('2020-03~2022-12를 중위수/MAD 표본에서 제외', base, ('gap', COVID_STATS_GAP)),
        'c4': ('고전 평균/표준편차 z (gc와 같은 산식)', base, ('method', 'classical')),
        'c5': ('중위수/MAD, 단 척도 하한 = 0.5 × 같은 창 표준편차', base, ('floor', 0.5)),
        'c6': ('c5, 단 2020-03~2022-12를 중위수·MAD·표준편차 표본에서 제외', base, ('gapfloor', 0.5)),
        'd1': ('YoY → 6개월 연율 (실업률은 6개월 변화)', six_month(False), None),
        'd2': ('3개월 평균 후 6개월 연율', six_month(True), None),
        'e1': ('a1 + b1', swap(GDP={'t': gdp, 'w': .20}, **unrate_split), None),
        'f1': ('a4 + c5 (서비스 입력 + 척도 하한)',
               swap(GDP={'t': gdp, 'w': .15}, SERV={'t': services, 'w': .15}), ('floor', 0.5)),
        'f2': ('a4 + c6', swap(GDP={'t': gdp, 'w': .15}, SERV={'t': services, 'w': .15}), ('gapfloor', 0.5)),
        'e2': ('a2 + b1', swap(GDP={'t': gdp, 'w': .15}, INCOME={'t': income, 'w': .15}, **unrate_split), None),
    }
    return variants


def score_variant(comps, setting, index):
    method = 'robust'
    ctx = contextlib.nullcontext()
    if setting and setting[0] == 'window':
        ctx = window(setting[1])
    elif setting and setting[0] == 'gap':
        ctx = stats_gap(*setting[1])
    elif setting and setting[0] == 'floor':
        ctx = mad_floor(setting[1])
    elif setting and setting[0] == 'gapfloor':
        ctx = contextlib.ExitStack()
        ctx.enter_context(stats_gap(*COVID_STATS_GAP))
        ctx.enter_context(mad_floor(setting[1]))
    elif setting and setting[0] == 'method':
        method = setting[1]
    with ctx:
        return score_axis(comps, index, method)


# ---------------------------------------------------------------- agreement

def sign_of(series, threshold=0.0):
    out = np.sign(series - threshold)
    return out.where(series.notna())


def agreement(model_sign, other_sign, shift=0, start=None, end=None):
    """Share of months where sign(model_t) equals sign(other_{t+shift}). Zero signs are left out."""
    other = other_sign.shift(-shift)
    frame = pd.concat({'m': model_sign, 'o': other}, axis=1, sort=True).dropna()
    if start is not None:
        frame = frame.loc[pd.Timestamp(start):]
    if end is not None:
        frame = frame.loc[:pd.Timestamp(end)]
    frame = frame[(frame.m != 0) & (frame.o != 0)]
    n = len(frame)
    if n == 0:
        return {'n': 0, 'agree': None, 'kappa': None, 'model_pos': None, 'other_pos': None}
    agree = float((frame.m == frame.o).mean())
    pm, po = float((frame.m > 0).mean()), float((frame.o > 0).mean())
    chance = pm * po + (1 - pm) * (1 - po)
    kappa = None if chance >= 1 - 1e-12 else (agree - chance) / (1 - chance)
    return {'n': n, 'agree': round(agree, 3), 'kappa': None if kappa is None else round(kappa, 3),
            'model_pos': round(pm, 3), 'other_pos': round(po, 3)}


def best_shift(model_sign, other_sign, start=None, end=None, max_shift=12, min_n=24):
    """Shift in [-max_shift, max_shift] with the highest agreement. Positive = model leads."""
    best = None
    for k in range(-max_shift, max_shift + 1):
        row = agreement(model_sign, other_sign, k, start, end)
        if row['n'] < min_n or row['agree'] is None:
            continue
        if best is None or row['agree'] > best[1]['agree'] + 1e-12 or (
                abs(row['agree'] - best[1]['agree']) < 1e-12 and abs(k) < abs(best[0])):
            best = (k, row)
    return None if best is None else {'shift': best[0], **best[1]}


def growth_comparators(raw):
    gdp_q = spread_quarterly(raw['A191RL1Q225SBEA'])
    gdp_med = spread_quarterly(raw['A191RL1Q225SBEA'].rolling(40, min_periods=20).median())
    cli = to_monthly(raw['OECD_CLI'])
    return {
        'CFNAI-MA3 (0=추세 성장)': sign_of(to_monthly(raw['CFNAIMA3'])),
        'ISM 제조업 PMI vs 50 (DBnomics 미러)': sign_of(raw['ISM_MFG'], 50.0),
        'ISM 서비스업 PMI vs 50 (DBnomics 미러)': sign_of(raw['ISM_SVC'], 50.0),
        'OECD CLI 6개월 변화 (LEI 대용)': sign_of(cli - cli.shift(6)),
        'OECD CLI 수준 vs 100': sign_of(cli, 100.0),
        'Philly Fed 선행지수 USSLIND (~2020-02)': sign_of(to_monthly(raw['USSLIND'])),
        '실질 GDP 전기비 연율 vs 0': sign_of(gdp_q),
        '실질 GDP 전기비 연율 vs 2%': sign_of(gdp_q, 2.0),
        '실질 GDP 전기비 연율 vs 직전 40분기 중위수': sign_of(gdp_q - gdp_med.reindex(gdp_q.index)),
    }


def liquidity_comparators(raw):
    nfci = to_monthly(raw['NFCI'])
    m2 = yoy(to_monthly(raw['M2SL']))
    ff = to_monthly(raw['FEDFUNDS'])
    ff3 = -(ff - ff.shift(3))
    return {
        '−NFCI 수준 (평균보다 완화=+)': sign_of(-nfci),
        '−NFCI 6개월 변화 (완화 쪽=+)': sign_of(-(nfci - nfci.shift(6))),
        'M2 YoY vs 0': sign_of(m2),
        'M2 YoY vs 직전 120개월 중위수': sign_of(trailing_median_gap(m2)),
        '정책금리 3개월 방향 (인하=+, ±5bp 이내 제외)': sign_of(ff3.where(ff3.abs() > .05, 0.0)),
    }


def agreement_table(model, comparators):
    sign = sign_of(model)
    rows = []
    for name, other in comparators.items():
        first = other.first_valid_index()
        rows.append({
            'name': name,
            'from': None if first is None else first.strftime('%Y-%m'),
            'full': agreement(sign, other),
            'history': agreement(sign, other, 0, *HISTORY),
            'recent': agreement(sign, other, 0, *RECENT),
            'best': best_shift(sign, other),
        })
    return rows


# ---------------------------------------------------------------- reports

def published(path=ROOT / 'dist' / 'gl_data.json'):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    months = pd.DataFrame(data['months'])
    months.index = pd.to_datetime(months['d'] + '-01')
    return data['meta'], months


def contribution_table(result, comps, start, end):
    """Mean weighted z share of the raw composite, and the robust centre/scale at `end`."""
    z = result.z.loc[start:end]
    weights = pd.Series({k: c['w'] for k, c in comps.items()})
    effw = z.notna().mul(weights, axis=1).sum(axis=1)
    share = z.mul(weights, axis=1).div(effw, axis=0)
    rows = []
    for key, comp in comps.items():
        signal = comp['t']
        samples = signal.dropna().loc[:end].tail(v2.WINDOW)
        med = float(samples.median())
        mad = float((samples - med).abs().median() * v2.MAD_SCALE)
        last = signal.dropna().loc[:end]
        rows.append({'id': key, 'w': comp['w'], 'mean_z': round(float(z[key].mean()), 2),
                     'mean_share': round(float(share[key].mean()), 3),
                     'neg_months': int((z[key] < 0).sum()), 'months': int(z[key].notna().sum()),
                     'median': round(med, 3), 'mad': round(mad, 3),
                     'std': round(float(samples.std()), 3),
                     'last': round(float(last.iloc[-1]), 3), 'last_month': last.index[-1].strftime('%Y-%m')})
    return rows


def negative_runs(g, usrec, top=6):
    """Longest stretches of G<0 with the number of NBER recession months inside each."""
    runs, start, prev = [], None, None
    for stamp, value in g.items():
        if value < 0 and start is None:
            start = stamp
        if value >= 0 and start is not None:
            runs.append((start, prev))
            start = None
        prev = stamp
    if start is not None:
        runs.append((start, prev))
    rec = usrec.reindex(g.index).fillna(0)
    rows = [{'from': a.strftime('%Y-%m'), 'to': b.strftime('%Y-%m'), 'months': len(g.loc[a:b]),
             'recession_months': int(rec.loc[a:b].sum())} for a, b in runs]
    return sorted(rows, key=lambda r: -r['months'])[:top]


def regime_summary(frame, start, end):
    part = frame.loc[start:end]
    counts = {KO[k]: int((part.regime == k).sum()) if k else int(part.regime.isna().sum()) for k in KO}
    confirmed = {KO[k]: int(((part.regime == k) & part.c).sum()) for k in KO if k}
    return {'months': len(part), 'g_pos': int((part.G > 0).sum()), 'counts': counts, 'confirmed': confirmed,
            'path': ''.join({'expansion': 'E', 'liquidity': 'L', 'adjustment': 'A',
                             'defense': 'D'}.get(r, '·') for r in part.regime)}


def guards(frame, usrec, cfnai_sign, ism_sign):
    rec = usrec.reindex(frame.index)
    shift = best_shift(sign_of(frame.G), cfnai_sign, *HISTORY)
    in_rec = frame.G[rec == 1]
    out_rec = frame.G[rec == 0]
    return {
        'recession_g_neg': round(float((in_rec < 0).mean()), 3),
        'expansion_g_pos': round(float((out_rec > 0).mean()), 3),
        'cfnai_history': agreement(sign_of(frame.G), cfnai_sign, 0, *HISTORY)['agree'],
        'cfnai_recent': agreement(sign_of(frame.G), cfnai_sign, 0, *RECENT)['agree'],
        'cfnai_shift': None if shift is None else shift['shift'],
        'ism_recent': agreement(sign_of(frame.G), ism_sign, 0, *RECENT)['agree'],
    }


def _name(value):
    return value if isinstance(value, str) else None


def run(cache, end_month):
    raw, dropped = load_all(cache)
    monthly, netliq = monthly_inputs(raw)
    index = model_index(monthly, end_month)
    meta, pub = published()

    liquidity = v3.change_liquidity(monthly, netliq)
    liquidity['NETLIQ']['available_from'] = monthly['WALCL'].first_valid_index()
    l_axis = score_axis(liquidity, index)
    base_comps = base_growth(monthly)
    g_axis = score_axis(base_comps, index)
    base_frame = regimes(g_axis, l_axis, index)

    joined = base_frame[['G', 'L', 'regime']].join(pub[['g', 'l', 'regime']], rsuffix='_pub', how='inner')
    recent = joined.loc[RECENT[0]:RECENT[1]]
    replication = {
        'months': len(recent),
        'max_abs_dg': round(float((recent.G - recent.g).abs().max()), 4),
        'max_abs_dl': round(float((recent.L - recent.l).abs().max()), 4),
        'regime_mismatch': [d.strftime('%Y-%m') for d, r in recent.iterrows()
                            if _name(r.regime) != _name(r.regime_pub)],
        'published_generated': meta.get('generated'),
    }

    g_pub, l_pub = pub['g'].astype(float), pub['l'].astype(float)
    g_comp = growth_comparators(raw)
    l_comp = liquidity_comparators(raw)
    nfci = to_monthly(raw['NFCI'])
    corr = {}
    for label, (lo, hi) in {'1990–2019': HISTORY, '2023-01–2026-09': RECENT}.items():
        frame = pd.concat({'L': l_pub, 'nfci': -nfci, 'dnfci': -(nfci - nfci.shift(6))}, axis=1, sort=True).loc[lo:hi].dropna()
        corr[label] = {'level': round(float(frame.L.corr(frame.nfci)), 3),
                       'change6': round(float(frame.L.corr(frame.dnfci)), 3), 'n': len(frame)}

    usrec = to_monthly(raw['USREC'])
    cfnai = g_comp['CFNAI-MA3 (0=추세 성장)']
    variants = {}
    for key, (label, comps, setting) in growth_variants(monthly, raw).items():
        g_var = g_axis if key == 'base' else score_variant(comps, setting, index)
        frame = regimes(g_var, l_axis, index)
        variants[key] = {'label': label, 'recent': regime_summary(frame, *RECENT),
                         'last': {'d': frame.index[-1].strftime('%Y-%m'), 'G': round(float(frame.G.iloc[-1]), 3)},
                         'guards': guards(frame, usrec, cfnai, g_comp['ISM 제조업 PMI vs 50 (DBnomics 미러)']),
                         'g_2023_2026': {d.strftime('%Y-%m'): round(float(v), 2)
                                         for d, v in frame.G.loc[RECENT[0]:RECENT[1]].items()}}

    def stamp(series, months):
        out = {}
        for m in months:
            ts = pd.Timestamp(m + '-01')
            out[m] = None if ts not in series.index or pd.isna(series.loc[ts]) else round(float(series.loc[ts]), 2)
        return out

    facts = {
        'gdp_saar': stamp(raw['A191RL1Q225SBEA'], ['2025-01', '2025-04', '2025-07', '2025-10', '2026-01', '2026-04']),
        'sahm_max_recent': round(float(raw['SAHMREALTIME'].loc[RECENT[0]:].max()), 2),
        'sahm_last': {raw['SAHMREALTIME'].index[-1].strftime('%Y-%m'): float(raw['SAHMREALTIME'].iloc[-1])},
        'cfnai_ma3_last': {to_monthly(raw['CFNAIMA3']).dropna().index[-1].strftime('%Y-%m'):
                           float(to_monthly(raw['CFNAIMA3']).dropna().iloc[-1])},
        'nfci_2026': {'min': round(float(nfci.loc['2026-01-01':].min()), 3),
                      'max': round(float(nfci.loc['2026-01-01':].max()), 3)},
        'm2_yoy_2026': stamp(yoy(to_monthly(raw['M2SL'])), ['2026-01', '2026-04', '2026-08']),
        'ism_mfg_last': None if raw['ISM_MFG'].empty else {raw['ISM_MFG'].index[-1].strftime('%Y-%m'): float(raw['ISM_MFG'].iloc[-1])},
        'ism_svc_last': None if raw['ISM_SVC'].empty else {raw['ISM_SVC'].index[-1].strftime('%Y-%m'): float(raw['ISM_SVC'].iloc[-1])},
        'ism_dropped': dropped,
        'oecd_cli_last': {raw['OECD_CLI'].index[-1].strftime('%Y-%m'): float(raw['OECD_CLI'].iloc[-1])},
    }
    return {
        'replication': replication,
        'growth_agreement': agreement_table(g_pub, g_comp),
        'liquidity_agreement': agreement_table(l_pub, l_comp),
        'l_nfci_corr': corr,
        'contribution': contribution_table(g_axis, base_comps, *RECENT),
        'z_by_year': {str(y): {k: round(float(v), 2) for k, v in row.items()} for y, row in
                      g_axis.z.loc[RECENT[0]:RECENT[1]].groupby(lambda d: d.year).mean().iterrows()},
        'g_negative_runs': negative_runs(pub['g'].astype(float), to_monthly(raw['USREC'])),
        'variants': variants,
        'facts': facts,
    }


def _pct(x):
    return '—' if x is None else f'{x * 100:.0f}%'


def markdown(result):
    lines = ['## replication', json.dumps(result['replication'], ensure_ascii=False), '']
    for axis in ('growth_agreement', 'liquidity_agreement'):
        lines += [f'## {axis}', '| 비교 | 시작 | 전체 n | 일치 | κ | 비교 + 비율 | 1990–2019 | 2023–26 n | 2023–26 | 2023–26 비교 + | 최적 시차 | 그때 일치 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in result[axis]:
            b = r['best'] or {}
            lines.append(f"| {r['name']} | {r['from']} | {r['full']['n']} | {_pct(r['full']['agree'])} | "
                         f"{r['full']['kappa']} | {_pct(r['full']['other_pos'])} | {_pct(r['history']['agree'])} | "
                         f"{r['recent']['n']} | {_pct(r['recent']['agree'])} | {_pct(r['recent']['other_pos'])} | {b.get('shift', '—')} | {_pct(b.get('agree'))} |")
        lines.append('')
    lines += ['## l_nfci_corr', json.dumps(result['l_nfci_corr'], ensure_ascii=False), '',
              '## contribution 2023-01..2026-09', '| id | w | 평균 z | 평균 몫 | 음수 달 | 중위수 | MAD | 표준편차 | 마지막 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['contribution']:
        lines.append(f"| {r['id']} | {r['w']} | {r['mean_z']} | {r['mean_share']} | {r['neg_months']}/{r['months']} | "
                     f"{r['median']} | {r['mad']} | {r['std']} | {r['last']} ({r['last_month']}) |")
    lines += ['', '## mean z by year', json.dumps(result['z_by_year'], ensure_ascii=False),
              '', '## longest G<0 runs', json.dumps(result['g_negative_runs'], ensure_ascii=False)]
    lines += ['', '## variants', '| 변형 | 설명 | G>0 달 | 실적 | 금융 | 역금융 | 역실적 | 경계·중립 | 역실적 확인 | 2026-09 G | 침체 G<0 | 확장 G>0 | CFNAI 1990–2019 | CFNAI 2023–26 | CFNAI 최적 시차 | ISM 2023–25 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, v in result['variants'].items():
        c, g = v['recent']['counts'], v['guards']
        lines.append(f"| {key} | {v['label']} | {v['recent']['g_pos']}/{v['recent']['months']} | {c['실적']} | {c['금융']} | "
                     f"{c['역금융']} | {c['역실적']} | {c['경계·중립']} | {v['recent']['confirmed']['역실적']} | {v['last']['G']} | "
                     f"{_pct(g['recession_g_neg'])} | {_pct(g['expansion_g_pos'])} | {_pct(g['cfnai_history'])} | {_pct(g['cfnai_recent'])} | {g['cfnai_shift']} | {_pct(g['ism_recent'])} |")
    lines += ['', '## paths 2023-01..2026-09 (E 실적, L 금융, A 역금융, D 역실적, · 경계·중립)']
    for key, v in result['variants'].items():
        lines.append(f"{key:>4} {v['recent']['path']}")
    lines += ['', '## G 2025-01..2026-09']
    months = [m for m in result['variants']['base']['g_2023_2026'] if m >= '2025-01']
    lines.append('| 변형 | ' + ' | '.join(m[2:] for m in months) + ' |')
    lines.append('|---|' + '---:|' * len(months))
    for key, v in result['variants'].items():
        lines.append(f'| {key} | ' + ' | '.join(f"{v['g_2023_2026'][m]:+.2f}" for m in months) + ' |')
    lines += ['', '## facts', json.dumps(result['facts'], ensure_ascii=False)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--cache', default=os.environ.get('GL_CACHE', '/tmp/gl-cache'))
    parser.add_argument('--end', default=None, help='first month of the model index end (default: this month)')
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    end = args.end or pd.Timestamp.today().to_period('M').to_timestamp().strftime('%Y-%m-%d')
    result = run(args.cache, end)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(markdown(result))


if __name__ == '__main__':
    main()
