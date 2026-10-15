#!/usr/bin/env python3
"""Daily radar backtest against the published monthly v3 G/L regimes.

The radar is a separate daily panel: 6 growth proxies and 8 liquidity proxies.
It never feeds the monthly score. This script rebuilds the radar on a business-day
calendar from FRED and compares it with `dist/gl_data.json` months.

Each input is moved to the date it would have been public (observation date plus
a release lag) before it is used. The monthly reference is the published history,
which already includes later revisions, so the comparison is against a revised
target, not a real-time one.

Usage:
  FRED_API_KEY=... python3 scripts/daily_radar_backtest.py [--start 2003-01-01] [--json out.json]
  GL_CACHE=/path/to/csvs python3 scripts/daily_radar_backtest.py   # <SERIES>.csv, date,value
  python3 scripts/daily_radar_backtest.py --selftest                 # synthetic data, pipeline check only
"""
import argparse
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_v2 import quadrant  # noqa: E402
from model_v3 import ENTER, EXIT  # noqa: E402

Z_WINDOW, Z_MIN = 756, 252          # about 3 years of business days
MIN_EFF_WEIGHT = 0.7
EWMA_SPAN = 5
CONFIRM_DAYS = 2
SWITCH_PERSIST = 10                  # business days a daily switch must hold to count
LEAD_LOOKBACK_WEEKS = 26
FALSE_ALARM_MONTHS = 4
PERIODS = (
    ('전체', None, None),
    ('2008-2012', '2008-01-01', '2012-12-31'),
    ('2013-2019', '2013-01-01', '2019-12-31'),
    ('2020-2021', '2020-01-01', '2021-12-31'),
    ('2022-현재', '2022-01-01', None),
)

# t: transform on the native observation frequency. lag: calendar days from the
# observation date to public availability. stale: business days of forward fill.
# kind: 'd' daily, 'w' weekly, 'q' quarterly.
G_SPEC = {
    'IC4WSA': dict(w=.25, kind='w', lag=5, stale=10, t=lambda s: -s.pct_change(52, fill_method=None) * 100),
    'CCSA': dict(w=.10, kind='w', lag=12, stale=10, t=lambda s: -s.pct_change(52, fill_method=None) * 100),
    'WEI': dict(w=.20, kind='w', lag=5, stale=10, t=lambda s: s),
    'GDPNOW': dict(w=.20, kind='q', lag=30, stale=70, t=lambda s: s),
    'SP500': dict(w=.15, kind='d', lag=1, stale=5, t=lambda s: s.pct_change(63, fill_method=None) * 100,
                  fallback='NASDAQCOM'),
    'T5YIE': dict(w=.10, kind='d', lag=1, stale=5, t=lambda s: s.diff(20)),
}
L_SPEC = {
    'NETLIQ': dict(w=.20, kind='w', lag=1, stale=10, t=lambda s: s.diff(13)),
    'DGS2': dict(w=.18, kind='d', lag=1, stale=5, t=lambda s: -s.diff(20)),
    'POLICY_GAP': dict(w=.10, kind='d', lag=1, stale=5, t=lambda s: s.rolling(20, min_periods=10).mean()),
    'T10Y3M': dict(w=.10, kind='d', lag=1, stale=5, t=lambda s: s.diff(60)),
    'BAMLH0A0HYM2': dict(w=.15, kind='d', lag=1, stale=5, t=lambda s: -s.diff(60), fallback='BAA10Y'),
    'DTWEXBGS': dict(w=.15, kind='d', lag=7, stale=10, t=lambda s: -s.pct_change(65, fill_method=None) * 100),
    'NFCI': dict(w=.07, kind='w', lag=5, stale=10, t=lambda s: -s),
    'VIXCLS': dict(w=.05, kind='d', lag=1, stale=5, t=lambda s: -s.rolling(20, min_periods=10).mean()),
}
FETCH = ('IC4WSA', 'CCSA', 'WEI', 'GDPNOW', 'SP500', 'NASDAQCOM', 'T5YIE', 'WALCL', 'WDTGAL',
         'RRPONTSYD', 'DGS2', 'DFF', 'T10Y3M', 'BAMLH0A0HYM2', 'BAA10Y', 'DTWEXBGS', 'NFCI', 'VIXCLS')
# G-axis variants for the redesign comparison. Fixed before the first real-data run.
# Each changes one thing: the z baseline, the horizon, or the components.
# D/E weights are the monthly v3 G weights mapped onto daily analogues, not fitted:
# WEI <- INDPRO+PAYEMS (.24+.24), CCSA <- UNRATE (.14), IC4WSA <- ICSA (.10); PERMIT has no daily analogue.
_yoy52 = lambda s: -s.pct_change(52, fill_method=None) * 100
_D_W = {'WEI': .48, 'CCSA': .14, 'IC4WSA': .10}
_D_W = {k: round(v / sum(_D_W.values()), 4) for k, v in _D_W.items()}
G_YOY = {
    **{k: dict(v) for k, v in G_SPEC.items()},
    'SP500': {**G_SPEC['SP500'], 't': lambda s: s.pct_change(252, fill_method=None) * 100},
    'T5YIE': {**G_SPEC['T5YIE'], 't': lambda s: s.diff(252)},
}
G_MONTHLY_ANALOG = {k: {**G_SPEC[k], 'w': w} for k, w in _D_W.items()}
G_VARIANTS = {
    'A 기존(6지표·단기·3년 기준)': dict(spec=G_SPEC, z_window=Z_WINDOW),
    'B 6지표·단기·장기 기준': dict(spec=G_SPEC, z_window=None),
    'C 6지표·YoY·장기 기준': dict(spec=G_YOY, z_window=None),
    'D 월간대응 3지표·YoY·장기 기준': dict(spec=G_MONTHLY_ANALOG, z_window=None),
    'E 월간대응 3지표·YoY·3년 기준': dict(spec=G_MONTHLY_ANALOG, z_window=Z_WINDOW),
}
TRAIN = ('학습 2009-2019', None, '2019-12-31')
TEST = ('검증 2020-현재', '2020-01-01', None)
MILLIONS = {'WALCL': 1.0, 'WDTGAL': 1.0, 'RRPONTSYD': 1000.0}   # FRED default units, used without the API


# ---------------------------------------------------------------- data

def _fred_json(path, params, key):
    url = f"https://api.stlouisfed.org/fred/{path}?" + urllib.parse.urlencode({**params, 'api_key': key, 'file_type': 'json'})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'pyeongantoo-radar/1.0'}),
                                    timeout=60) as response:
            return json.load(response)
    except Exception as exc:
        raise RuntimeError(f"FRED API 실패: {params.get('series_id')} ({type(exc).__name__})") from None


def load_series(series_id, start):
    cache = os.environ.get('GL_CACHE', '')
    path = Path(cache) / f'{series_id}.csv' if cache else None
    key = os.environ.get('FRED_API_KEY', '')
    if path and path.exists():
        frame = pd.read_csv(path)
        scale = MILLIONS.get(series_id, 1.0)
    elif key:
        rows = _fred_json('series/observations', {'series_id': series_id, 'observation_start': start}, key)['observations']
        frame = pd.DataFrame(rows)[['date', 'value']]
        units = _fred_json('series', {'series_id': series_id}, key)['seriess'][0].get('units', '')
        scale = 1000.0 if 'Billions' in units else 1.0
        if path:
            frame.to_csv(path, index=False)
            scale = MILLIONS.get(series_id, scale)
    else:
        raise RuntimeError('FRED_API_KEY 또는 GL_CACHE가 필요합니다')
    frame.columns = ['date', 'value']
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    frame['date'] = pd.to_datetime(frame['date'])
    return frame.dropna().set_index('date')['value'].sort_index() * scale


def load_all(start):
    raw, errors = {}, {}
    for sid in FETCH:
        try:
            raw[sid] = load_series(sid, start)
        except Exception as exc:
            errors[sid] = str(exc)
            raw[sid] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return raw, errors


def net_liquidity(raw):
    """Wednesday levels, millions of dollars. RRP is daily, so take the Wednesday value."""
    walcl, tga = raw['WALCL'], raw['WDTGAL']
    rrp = raw['RRPONTSYD'].reindex(walcl.index, method='ffill', tolerance=pd.Timedelta(days=3))
    rrp = rrp.where(walcl.index >= raw['RRPONTSYD'].first_valid_index(), 0.0) if len(raw['RRPONTSYD']) else 0.0
    return (walcl - tga.reindex(walcl.index, method='ffill', tolerance=pd.Timedelta(days=3)) - rrp).dropna()


def inputs(raw, start):
    """Pick fallbacks where the primary series is short (FRED keeps 3 years of ICE and 10 years of S&P)."""
    out, used = {}, {}
    derived = {'NETLIQ': net_liquidity(raw),
               'POLICY_GAP': (raw['DFF'] - raw['DGS2']).dropna()}
    for spec in (G_SPEC, L_SPEC):
        for sid, item in spec.items():
            series = derived.get(sid, raw.get(sid, pd.Series(dtype=float)))
            name = sid
            fb = item.get('fallback')
            if fb and (series.empty or series.index[0] > pd.Timestamp(start) + pd.DateOffset(years=4)):
                series, name = raw[fb], fb
            out[sid] = series
            used[sid] = {'series': name, 'first': None if series.empty else series.index[0].strftime('%Y-%m-%d')}
    return out, used


# ---------------------------------------------------------------- radar

def _window(x, window):
    return x.expanding(min_periods=Z_MIN) if window is None else x.rolling(window, min_periods=Z_MIN)


def robust_z(x, window=Z_WINDOW):
    """Causal median/MAD z. window=None uses all history up to each day."""
    med = _window(x, window).median()
    mad = _window((x - med).abs(), window).median() * 1.4826
    return ((x - med) / mad.replace(0, np.nan)).clip(-3, 3)


def axis_score(spec, series, bdays, include, z_window=Z_WINDOW):
    zs, ws = {}, {}
    for sid, item in spec.items():
        if sid not in include or series[sid].empty:
            continue
        native = item['t'](series[sid]).dropna()
        native.index = native.index + pd.Timedelta(days=item['lag'])
        native = native[~native.index.duplicated(keep='last')]
        aligned = native.reindex(native.index.union(bdays)).ffill(limit=None).reindex(bdays)
        # Stale guard: drop values older than `stale` business days.
        last_obs = pd.Series(native.index, index=native.index).reindex(native.index.union(bdays)).ffill().reindex(bdays)
        age = np.busday_count(last_obs.fillna(bdays[0]).values.astype('datetime64[D]'), bdays.values.astype('datetime64[D]'))
        aligned[(age > item['stale']) | last_obs.isna()] = np.nan
        zs[sid] = robust_z(aligned, z_window)
        ws[sid] = item['w']
    z = pd.DataFrame(zs, index=bdays)
    w = pd.Series(ws)
    avail = z.notna().mul(w, axis=1)
    eff = avail.sum(axis=1)
    raw_score = z.fillna(0).mul(w, axis=1).sum(axis=1) / eff.replace(0, np.nan)
    raw_score[eff < MIN_EFF_WEIGHT * w.sum()] = np.nan
    # Rescale so ±0.25 means the same thing as in v3: divide by a causal 3-year std.
    scale = _window(raw_score, z_window).std()
    score = (raw_score / scale).ewm(span=EWMA_SPAN, ignore_na=True).mean()
    score[raw_score.isna()] = np.nan
    return score, eff, z


def hysteresis(score):
    """v3 enter/exit thresholds, plus a 2-day confirmation before the state changes."""
    state, pending, count, out = 0, None, 0, []
    for value in score.to_numpy():
        if not np.isfinite(value):
            out.append(np.nan)
            continue
        if state == 1:
            target = -1 if value <= -ENTER else (0 if value < EXIT else 1)
        elif state == -1:
            target = 1 if value >= ENTER else (0 if value > -EXIT else -1)
        else:
            target = 1 if value >= ENTER else (-1 if value <= -ENTER else 0)
        if target == state:
            pending, count = None, 0
        elif target == pending:
            count += 1
            if count >= CONFIRM_DAYS:
                state, pending, count = target, None, 0
        else:
            pending, count = target, 1
            if CONFIRM_DAYS <= 1:
                state, pending, count = target, None, 0
        out.append(state)
    return pd.Series(out, index=score.index, dtype=float)


def daily_regime(gs, ls):
    def one(g, l):
        if not (np.isfinite(g) and np.isfinite(l)) or g == 0 or l == 0:
            return None
        return quadrant(g, l)
    return pd.Series([one(g, l) for g, l in zip(gs, ls)], index=gs.index, dtype=object)


# ---------------------------------------------------------------- evaluation

def load_monthly(path):
    months = json.loads(Path(path).read_text(encoding='utf-8'))['months']
    frame = pd.DataFrame([{
        'month': pd.Timestamp(m['d'] + '-01'), 'g': m.get('g'), 'l': m.get('l'),
        'ax_g': (m.get('ax') or {}).get('g'), 'ax_l': (m.get('ax') or {}).get('l'),
        'regime': m.get('regime'), 'c': bool(m.get('c')), 'p': bool(m.get('p')),
    } for m in months])
    return frame[~frame.p].set_index('month')


def announce_date(month):
    """The monthly run that first carries month t: the 16th of t+1 (update.yml runs on the 1st and 16th)."""
    return month + pd.DateOffset(months=1, days=15)


def at_or_before(series, stamp):
    sub = series.loc[:stamp]
    return sub.iloc[-1] if len(sub) else np.nan


def _rate(hits, total):
    return None if total == 0 else round(hits / total * 100, 1)


def direction_agreement(monthly, daily_g, daily_l, gs_raw, ls_raw):
    rows = {'G': [0, 0], 'L': [0, 0], 'G_sign': [0, 0], 'L_sign': [0, 0], 'Q': [0, 0], 'Q_any': [0, 0]}
    for month, row in monthly.iterrows():
        end = month + pd.offsets.MonthEnd(0)
        dg, dl = at_or_before(daily_g, end), at_or_before(daily_l, end)
        rg, rl = at_or_before(gs_raw, end), at_or_before(ls_raw, end)
        for axis, ax, d_state, d_raw, m_val in (('G', row.ax_g, dg, rg, row.g), ('L', row.ax_l, dl, rl, row.l)):
            if ax in (1, -1) and np.isfinite(d_state) and d_state != 0:
                rows[axis][1] += 1
                rows[axis][0] += int(d_state == ax)
            if m_val is not None and np.isfinite(d_raw):
                rows[axis + '_sign'][1] += 1
                rows[axis + '_sign'][0] += int(np.sign(d_raw) == np.sign(m_val))
        if row.regime and np.isfinite(dg) and np.isfinite(dl):
            if dg != 0 and dl != 0:
                rows['Q'][1] += 1
                rows['Q'][0] += int(quadrant(dg, dl) == row.regime)
            if np.isfinite(rg) and np.isfinite(rl):
                rows['Q_any'][1] += 1
                rows['Q_any'][0] += int(quadrant(rg, rl) == row.regime)
    return {k: {'rate': _rate(*v), 'n': v[1]} for k, v in rows.items()}


def monthly_switches(monthly):
    """Confirmed regime changes: the first confirmed month of a new regime."""
    out, last = [], None
    for month, row in monthly.iterrows():
        if row.c and row.regime:
            if last is not None and row.regime != last:
                out.append((month, last, row.regime))
            last = row.regime
    return out


def axis_flips(monthly, col):
    out, last = [], None
    for month, value in monthly[col].items():
        if value in (1, -1):
            if last is not None and value != last:
                out.append((month, value))
            last = value
    return out


def lead_of(target_hit, announce, start_bound):
    """Weeks from the start of the uninterrupted run of `target_hit` that is live at `announce`."""
    window = target_hit.loc[start_bound:announce]
    if window.empty or not window.iloc[-1]:
        reached = bool(window.any())
        return None, reached
    run_start = window.index[-1]
    for stamp, hit in window.iloc[::-1].items():
        if not hit:
            break
        run_start = stamp
    return round((announce - run_start).days / 7, 1), True


def lead_stats(events):
    leads = [e['lead_weeks'] for e in events if e['lead_weeks'] is not None]
    return {
        'events': len(events),
        'caught_before_announce': _rate(len(leads), len(events)),
        'reached_any_time_in_window': _rate(sum(e['reached'] for e in events), len(events)),
        'median_lead_weeks': None if not leads else round(float(np.median(leads)), 1),
        'mean_lead_weeks': None if not leads else round(float(np.mean(leads)), 1),
        'lead_vs_month_end_median_weeks': None if not leads else round(float(np.median([e['lead_vs_month_end'] for e in events if e['lead_weeks'] is not None])), 1),
    }


def leadness(monthly, regime, daily_g, daily_l):
    reg_events = []
    for month, old, new in monthly_switches(monthly):
        ann = announce_date(month)
        lead, reached = lead_of(regime == new, ann, ann - pd.Timedelta(weeks=LEAD_LOOKBACK_WEEKS))
        month_end_gap = (ann - (month + pd.offsets.MonthEnd(0))).days / 7
        reg_events.append({'month': month.strftime('%Y-%m'), 'from': old, 'to': new, 'lead_weeks': lead,
                           'reached': reached,
                           'lead_vs_month_end': None if lead is None else round(lead - month_end_gap, 1)})
    axis_events = {}
    for axis, col, daily in (('G', 'ax_g', daily_g), ('L', 'ax_l', daily_l)):
        rows = []
        for month, value in axis_flips(monthly, col):
            ann = announce_date(month)
            lead, reached = lead_of(daily == value, ann, ann - pd.Timedelta(weeks=LEAD_LOOKBACK_WEEKS))
            month_end_gap = (ann - (month + pd.offsets.MonthEnd(0))).days / 7
            rows.append({'month': month.strftime('%Y-%m'), 'to': int(value), 'lead_weeks': lead, 'reached': reached,
                         'lead_vs_month_end': None if lead is None else round(lead - month_end_gap, 1)})
        axis_events[axis] = rows
    return reg_events, axis_events


def daily_switches(regime):
    """Non-neutral regime changes that hold for SWITCH_PERSIST business days."""
    out, last, idx = [], None, regime.index
    values = regime.to_numpy()
    i = 0
    while i < len(values):
        cur = values[i]
        if cur is not None and cur != last:
            hold = values[i:i + SWITCH_PERSIST]
            if len(hold) == SWITCH_PERSIST and all(v == cur for v in hold):
                if last is not None:
                    out.append((idx[i], cur))
                last = cur
                i += SWITCH_PERSIST
                continue
        i += 1
    return out


def false_alarms(monthly, regime):
    confirmed = monthly[monthly.c & monthly.regime.notna()].regime
    events = []
    for stamp, new in daily_switches(regime):
        lo = stamp - pd.DateOffset(months=1)
        hi = stamp + pd.DateOffset(months=FALSE_ALARM_MONTHS)
        window = confirmed.loc[lo.to_period('M').to_timestamp():hi]
        events.append({'date': stamp.strftime('%Y-%m-%d'), 'to': new, 'confirmed_by_monthly': bool((window == new).any())})
    total = len(events)
    false = sum(not e['confirmed_by_monthly'] for e in events)
    return {'daily_switches': total, 'false_alarms': false, 'false_alarm_rate': _rate(false, total)}, events


def evaluate(monthly, gs_state, ls_state, gs, ls, regime, start, end):
    m = monthly.loc[start:end] if (start or end) else monthly
    sl = slice(start, end)
    dg, dl, rg, rl, reg = gs_state.loc[sl], ls_state.loc[sl], gs.loc[sl], ls.loc[sl], regime.loc[sl]
    # Events come from the full history so a period's first switch keeps its prior regime.
    reg_events, axis_events = leadness(monthly, regime, gs_state, ls_state)
    keep = set(m.index.strftime('%Y-%m'))
    reg_events = [e for e in reg_events if e['month'] in keep]
    axis_events = {k: [e for e in v if e['month'] in keep] for k, v in axis_events.items()}
    fa, fa_events = false_alarms(monthly, reg)
    return {
        'months': int(len(m)),
        'agreement': direction_agreement(m, dg, dl, rg, rl),
        'regime_lead': lead_stats(reg_events),
        'axis_lead': {k: lead_stats(v) for k, v in axis_events.items()},
        'false_alarm': fa,
        'regime_events': reg_events,
        'daily_switch_events': fa_events,
    }


# ---------------------------------------------------------------- report

def fmt(v, suffix=''):
    return '-' if v is None else f'{v}{suffix}'


def markdown(result):
    lines = ['| 기간 | 월수 | G 방향 | L 방향 | 사분면(히스테리시스) | 사분면(부호) | 전환 수 | 전환 선행 중앙값(주) | 사전 포착률 | G축 선행(주) | L축 선행(주) | 일간 전환 | 오경보율 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, r in result['periods'].items():
        a = r['agreement']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name, r['months'], fmt(a['G']['rate'], '%'), fmt(a['L']['rate'], '%'), fmt(a['Q']['rate'], '%'),
            fmt(a['Q_any']['rate'], '%'), r['regime_lead']['events'], fmt(r['regime_lead']['median_lead_weeks']),
            fmt(r['regime_lead']['caught_before_announce'], '%'), fmt(r['axis_lead']['G']['median_lead_weeks']),
            fmt(r['axis_lead']['L']['median_lead_weeks']), r['false_alarm']['daily_switches'],
            fmt(r['false_alarm']['false_alarm_rate'], '%')))
    return '\n'.join(lines)


def run(start, monthly_path, include_gdpnow):
    raw, errors = load_all(start)
    series, used = inputs(raw, start)
    end = pd.Timestamp.today().normalize()
    bdays = pd.bdate_range(start, end)
    g_inc = set(G_SPEC) - (set() if include_gdpnow else {'GDPNOW'})
    gs, g_eff, _ = axis_score(G_SPEC, series, bdays, g_inc)
    ls, l_eff, _ = axis_score(L_SPEC, series, bdays, set(L_SPEC))
    gs_state, ls_state = hysteresis(gs), hysteresis(ls)
    regime = daily_regime(gs_state, ls_state)
    monthly = load_monthly(monthly_path)
    first = max(gs.first_valid_index() or end, ls.first_valid_index() or end)
    monthly = monthly.loc[first:]
    periods = {}
    for name, a, b in PERIODS:
        lo = pd.Timestamp(a) if a else None
        hi = pd.Timestamp(b) if b else None
        if lo is not None and hi is not None and hi < first:
            continue
        periods[name] = evaluate(monthly, gs_state, ls_state, gs, ls, regime, lo, hi)
    latest = gs.last_valid_index()
    return {
        'settings': {'start': start, 'z_window_bdays': Z_WINDOW, 'ewma_span': EWMA_SPAN, 'enter': ENTER, 'exit': EXIT,
                     'confirm_days': CONFIRM_DAYS, 'switch_persist_bdays': SWITCH_PERSIST,
                     'false_alarm_window_months': FALSE_ALARM_MONTHS, 'include_gdpnow': include_gdpnow,
                     'first_scored_day': None if first is end else first.strftime('%Y-%m-%d')},
        'inputs': used, 'source_errors': errors,
        'latest': None if latest is None else {'date': latest.strftime('%Y-%m-%d'), 'G': round(float(gs[latest]), 3),
                                               'L': round(float(ls.get(latest, np.nan)), 3),
                                               'regime': regime.get(latest)},
        'periods': periods,
    }


def selftest():
    """Synthetic random walks plus a synthetic monthly file. Checks the pipeline runs; numbers mean nothing."""
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp(prefix='radar-selftest-'))
    days = pd.bdate_range('2003-01-01', pd.Timestamp.today())
    weeks = pd.date_range('2003-01-04', pd.Timestamp.today(), freq='W-SAT')
    weds = pd.date_range('2003-01-01', pd.Timestamp.today(), freq='W-WED')
    cycle = np.sin(np.arange(len(days)) / 400)
    for sid in FETCH:
        if sid in ('IC4WSA', 'CCSA', 'WEI', 'NFCI'):
            idx = weeks
        elif sid in ('WALCL', 'WDTGAL'):
            idx = weds
        elif sid == 'GDPNOW':
            idx = pd.date_range('2011-03-31', pd.Timestamp.today(), freq='QE')
        else:
            idx = days
        base = np.interp(np.arange(len(idx)), np.linspace(0, len(idx) - 1, len(days)), cycle)
        values = 100 + np.cumsum(rng.normal(0, 1, len(idx))) + 20 * base
        pd.DataFrame({'date': idx.strftime('%Y-%m-%d'), 'value': np.abs(values) + 1}).to_csv(tmp / f'{sid}.csv', index=False)
    months = pd.date_range('2003-01-01', pd.Timestamp.today(), freq='MS')
    mcycle = np.sin(np.arange(len(months)) * 21 / 400)
    fake = {'months': [{'d': m.strftime('%Y-%m'), 'g': float(v), 'l': float(-v),
                        'ax': {'g': int(np.sign(v)) if abs(v) > .25 else 0, 'l': int(-np.sign(v)) if abs(v) > .25 else 0},
                        'regime': quadrant(v, -v) if abs(v) > .25 else None, 'c': bool(abs(v) > .25), 'p': False}
                       for m, v in zip(months, mcycle)]}
    (tmp / 'months.json').write_text(json.dumps(fake))
    os.environ['GL_CACHE'] = str(tmp)
    result = run('2003-01-01', tmp / 'months.json', include_gdpnow=False)
    assert result['periods'], 'no periods scored'
    assert not result['source_errors'], result['source_errors']
    full = result['periods']['전체']
    assert full['agreement']['G']['n'] > 0 and full['agreement']['Q_any']['n'] > 0
    print('selftest OK (synthetic data, numbers are not results)')
    print(markdown(result))


def compare(start, monthly_path):
    """Score every G variant against the same L axis and monthly reference."""
    raw, errors = load_all(start)
    series, used = inputs(raw, start)
    end = pd.Timestamp.today().normalize()
    bdays = pd.bdate_range(start, end)
    ls, _, _ = axis_score(L_SPEC, series, bdays, set(L_SPEC))
    ls_state = hysteresis(ls)
    monthly_all = load_monthly(monthly_path)
    out = {}
    for name, v in G_VARIANTS.items():
        include = set(v['spec']) - {'GDPNOW'}
        gs, _, _ = axis_score(v['spec'], series, bdays, include, v['z_window'])
        gs_state = hysteresis(gs)
        regime = daily_regime(gs_state, ls_state)
        first = max(gs.first_valid_index() or end, ls.first_valid_index() or end)
        monthly = monthly_all.loc[first:]
        periods = {}
        for label, a, b in (*PERIODS, TRAIN, TEST):
            lo = pd.Timestamp(a) if a else None
            hi = pd.Timestamp(b) if b else None
            if hi is not None and hi < first:
                continue
            periods[label] = evaluate(monthly, gs_state, ls_state, gs, ls, regime, lo, hi)
        last = gs.last_valid_index()
        out[name] = {'first_scored_day': first.strftime('%Y-%m-%d'), 'periods': periods,
                     'latest': {'date': last.strftime('%Y-%m-%d'), 'G': round(float(gs[last]), 3),
                                'regime': regime.get(last)}}
    return {'inputs': used, 'source_errors': errors, 'variants': out,
            'rule': '학습 구간 G 방향 일치율 최고안 선택, 단 학습 구간 오경보율이 A안 이하일 것'}


def compare_markdown(result):
    rows = ['| 안 | 구간 | 월수 | G 방향 | L 방향 | 사분면 | 사분면(부호) | 전환 | 사전 포착 | 선행 중앙값(주) | G축 선행(주) | 일간 전환 | 오경보율 |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, v in result['variants'].items():
        for label, r in v['periods'].items():
            a = r['agreement']
            rows.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                name, label, r['months'], fmt(a['G']['rate'], '%'), fmt(a['L']['rate'], '%'),
                fmt(a['Q']['rate'], '%'), fmt(a['Q_any']['rate'], '%'), r['regime_lead']['events'],
                fmt(r['regime_lead']['caught_before_announce'], '%'), fmt(r['regime_lead']['median_lead_weeks']),
                fmt(r['axis_lead']['G']['median_lead_weeks']), r['false_alarm']['daily_switches'],
                fmt(r['false_alarm']['false_alarm_rate'], '%')))
    base = result['variants'][next(iter(result['variants']))]['periods'][TRAIN[0]]['false_alarm']['false_alarm_rate']
    ok = {n: v['periods'][TRAIN[0]] for n, v in result['variants'].items()
          if (v['periods'][TRAIN[0]]['false_alarm']['false_alarm_rate'] or 0) <= (base or 0)}
    pick = max(ok, key=lambda n: ok[n]['agreement']['G']['rate'] or 0) if ok else None
    rows.append('')
    rows.append(f'선택 규칙(사전 고정): {result["rule"]}')
    rows.append(f'규칙상 선택: {pick}')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2003-01-01')
    ap.add_argument('--monthly', default=str(ROOT / 'dist' / 'gl_data.json'))
    ap.add_argument('--with-gdpnow', action='store_true',
                    help='FRED GDPNOW is one final nowcast per quarter, not the daily path; off by default')
    ap.add_argument('--json')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--compare', action='store_true', help='compare the G-axis variants')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.compare:
        result = compare(args.start, args.monthly)
        if args.json:
            Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        print(json.dumps({'inputs': result['inputs'], 'source_errors': result['source_errors'],
                          'latest': {k: v['latest'] for k, v in result['variants'].items()}},
                         ensure_ascii=False, indent=1))
        print(compare_markdown(result))
        return
    result = run(args.start, args.monthly, args.with_gdpnow)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({'inputs': result['inputs'], 'source_errors': result['source_errors'],
                      'latest': result['latest'], 'settings': result['settings']}, ensure_ascii=False, indent=1))
    print(markdown(result))


if __name__ == '__main__':
    main()
