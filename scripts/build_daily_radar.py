#!/usr/bin/env python3
"""Build dist/daily_data.json: the daily G/L radar beside the monthly v3 quadrant.

The radar is a reference signal, not a quadrant call. It reuses the backtested
definitions from scripts/daily_radar_backtest.py so the production numbers and
the validation in docs/daily-radar-l-variants.md come from the same code:
G = variant D (WEI, CCSA, IC4WSA), L = variant LC (monthly v3 L inputs on
weekly/daily series), both on an expanding median/MAD baseline.

Usage:
  FRED_API_KEY=... python3 scripts/build_daily_radar.py [--out dist/daily_data.json]
  GL_CACHE=/path python3 scripts/build_daily_radar.py   # offline, <SERIES>.csv files
Exit code 1 when the latest G or L cannot be scored.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import daily_radar_backtest as bt  # noqa: E402

RADAR_VERSION = 'daily-radar-1'
START = '2003-01-01'
HISTORY_BDAYS = 60
G_SPEC = bt.G_MONTHLY_ANALOG
L_SPEC = bt.L_MONTHLY_ANALOG
LABELS = {
    'WEI': '주간경제지수(댈러스 연은)', 'CCSA': '계속실업수당청구 YoY (역)', 'IC4WSA': '신규실업수당청구 4주평균 YoY (역)',
    'WM2NS': '주간 M2 YoY', 'DGS2': '2년물 금리 3개월 변화 (역)', 'NETLIQ': '순유동성(연준자산−TGA−역레포) YoY',
    'DTWEXBGS': '광의 달러지수 YoY (역)', 'T10Y3M': '10Y-3M 6개월 변화', 'BAA10Y': 'Baa-10Y 6개월 변화 (역)',
}
STATE_KO = {1: '상승', 0: '중립', -1: '하락'}
REGIME_KO = {'expansion': '확장', 'liquidity': '유동성', 'adjustment': '조정', 'defense': '방어', None: '중립'}


def _num(value, digits=3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _state(value):
    return None if value is None or not np.isfinite(value) else int(value)


def components(spec, series, raw, bdays, z):
    """Latest z, weight, last public observation and its age for each input."""
    today = bdays[-1]
    out = {}
    for sid, item in spec.items():
        src = series[sid] if sid in series else raw.get(sid, pd.Series(dtype=float))
        last_obs = None if src.empty else src.index[-1]
        public = None if last_obs is None else last_obs + pd.Timedelta(days=item['lag'])
        age = None if public is None else int(np.busday_count(public.date(), today.date())) if public <= today else 0
        zval = z[sid].iloc[-1] if sid in z.columns else np.nan
        out[sid] = {'label': LABELS.get(sid, sid), 'weight': item['w'], 'z': _num(zval, 2),
                    'last_observation': None if last_obs is None else last_obs.strftime('%Y-%m-%d'),
                    'age_bdays': age, 'stale': bool(age is None or age > item['stale'] or not np.isfinite(zval))}
    return out


def monthly_reference(path):
    """Latest monthly v3 month (provisional or not) and the latest confirmed regime."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    months = data['months']
    last = months[-1]
    confirmed = next((m for m in reversed(months) if m.get('c') and m.get('regime')), None)
    return {
        'model_version': data['meta'].get('model_version'),
        'month': last['d'], 'provisional': bool(last.get('p')),
        'g': last.get('g'), 'l': last.get('l'), 'regime': last.get('regime'),
        'axis': last.get('ax'),
        'last_confirmed': None if confirmed is None else {'month': confirmed['d'], 'regime': confirmed['regime']},
    }


def build(monthly_path, start=START):
    raw, errors = bt.load_all(start)
    series, _ = bt.inputs(raw, start)
    bdays = pd.bdate_range(start, pd.Timestamp.today().normalize())
    gs, g_eff, gz = bt.axis_score(G_SPEC, bt._series_for(G_SPEC, series, raw), bdays, set(G_SPEC), None)
    ls, l_eff, lz = bt.axis_score(L_SPEC, bt._series_for(L_SPEC, series, raw), bdays, set(L_SPEC), None)
    g_state, l_state = bt.hysteresis(gs), bt.hysteresis(ls)
    regime = bt.daily_regime(g_state, l_state)
    scored = gs.dropna().index.intersection(ls.dropna().index)
    if scored.empty:
        raise SystemExit('일간 레이더 점수를 계산할 수 없습니다 (입력 부족). source_errors=' + json.dumps(errors))
    last = scored[-1]
    stale_days = int(np.busday_count(last.date(), bdays[-1].date()))
    history = [{'d': d.strftime('%Y-%m-%d'), 'g': _num(gs[d]), 'l': _num(ls[d]),
                'gs': _state(g_state[d]), 'ls': _state(l_state[d]), 'regime': regime[d]}
               for d in scored[-HISTORY_BDAYS:]]
    ref = monthly_reference(monthly_path)
    latest_regime = regime[last]
    axis = ref.get('axis') or {}
    agree = {
        'g': None if not axis.get('g') or not g_state[last] else bool(int(g_state[last]) == axis['g']),
        'l': None if not axis.get('l') or not l_state[last] else bool(int(l_state[last]) == axis['l']),
    }
    return {
        'meta': {
            'generated': pd.Timestamp.now('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
            'radar_version': RADAR_VERSION,
            'reference_model_version': ref['model_version'],
            'purpose_ko': '월간 v3 사분면을 미리 보는 참고 신호입니다. 사분면 판정이 아닙니다.',
            'method': {
                'g': {k: v['w'] for k, v in G_SPEC.items()}, 'l': {k: v['w'] for k, v in L_SPEC.items()},
                'z': 'expanding median/MAD, clip ±3, weighted mean, scaled by expanding std, 5-day EWMA',
                'enter': bt.ENTER, 'exit': bt.EXIT, 'confirm_bdays': bt.CONFIRM_DAYS,
                'min_effective_weight': bt.MIN_EFF_WEIGHT,
                'validation': 'docs/daily-radar-l-variants.md',
            },
            'source_errors': errors,
            'stale_bdays': stale_days,
        },
        'latest': {
            'date': last.strftime('%Y-%m-%d'),
            'g': _num(gs[last]), 'l': _num(ls[last]),
            'g_state': _state(g_state[last]), 'l_state': _state(l_state[last]),
            'g_state_ko': STATE_KO.get(_state(g_state[last])), 'l_state_ko': STATE_KO.get(_state(l_state[last])),
            'regime': latest_regime, 'regime_ko': REGIME_KO.get(latest_regime),
            'effective_weight': {'g': _num(g_eff[last], 2), 'l': _num(l_eff[last], 2)},
        },
        'monthly': ref,
        'agreement_with_monthly': agree,
        'components': {'g': components(G_SPEC, series, raw, bdays, gz), 'l': components(L_SPEC, series, raw, bdays, lz)},
        'history': history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'dist' / 'daily_data.json'))
    ap.add_argument('--monthly', default=str(ROOT / 'dist' / 'gl_data.json'))
    args = ap.parse_args()
    data = build(args.monthly)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    lt, mo = data['latest'], data['monthly']
    print(f"[daily] {lt['date']} G={lt['g']} ({lt['g_state_ko']}) L={lt['l']} ({lt['l_state_ko']}) "
          f"-> {lt['regime_ko']} | 월간 {mo['month']} {mo['regime']} | stale={data['meta']['stale_bdays']}bd "
          f"| source_errors={list(data['meta']['source_errors'])}")


if __name__ == '__main__':
    main()
