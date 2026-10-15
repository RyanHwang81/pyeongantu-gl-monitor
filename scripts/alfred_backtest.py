#!/usr/bin/env python3
"""Turning points, whipsaws, and a limited ALFRED vintage check.

The default path reads the published v3 months and FRED USREC. Those scores
already include later revisions, so a lead versus an NBER peak is not a
real-time claim.

`--vintage` downloads one ALFRED graph CSV per input series at a few month-ends
around 2001, 2008, and 2020, then rescores v3 on that vintage alone.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VINTAGE_DATES = (
    '2001-03-31', '2001-06-30', '2001-11-30',
    '2007-12-31', '2008-03-31', '2008-09-30', '2009-06-30',
    '2020-02-29', '2020-04-30', '2020-07-31',
)
SERIES = (
    'PERMIT', 'INDPRO', 'PAYEMS', 'UNRATE', 'ICSA', 'M2SL', 'T10Y3M',
    'TOTALSL', 'BAA10YM', 'WALCL', 'DGS2', 'DTWEXBGS', 'WTREGEN', 'RRPONTSYD',
)


def _import_model():
    import sys
    sys.path.insert(0, str(ROOT))
    import model_v2
    import model_v3
    return model_v2, model_v3


def load_usrec(path=None):
    cache = os.environ.get('GL_CACHE', '/tmp/gl-cache')
    csv_path = Path(path) if path else Path(cache) / 'USREC.csv'
    frame = pd.read_csv(csv_path)
    frame.columns = ['date', 'value']
    frame['date'] = pd.to_datetime(frame['date'])
    return frame.dropna().set_index('date')['value'].astype(int).resample('MS').last()


def nber_episodes(usrec):
    """USREC turns on the month after the NBER peak and stays on through the trough."""
    episodes = []
    on = usrec == 1
    started = None
    for stamp, flag in on.items():
        if flag and started is None:
            started = stamp
        elif not flag and started is not None:
            peak = started - pd.DateOffset(months=1)
            trough = stamp - pd.DateOffset(months=1)
            episodes.append({'peak': peak, 'trough': trough})
            started = None
    return [row for row in episodes if row['peak'] >= pd.Timestamp('1972-01-01')]


def _month_index(months):
    frame = pd.DataFrame(months)
    frame['date'] = pd.to_datetime(frame['d'])
    return frame.set_index('date').sort_index()


def _previous_month(stamp):
    return (pd.Timestamp(stamp) - pd.DateOffset(months=1)).to_period('M').to_timestamp()


def defense_lead(frame, peak, confirmed_only=False):
    """Months of defense already in place at the NBER peak.

    Positive means the defense spell that includes the peak started that many
    months earlier. Zero means the peak month is the first defense month.
    Negative means the first defense month arrived after the peak. None means
    there is no defense month in the year after the peak either.
    """
    peak = pd.Timestamp(peak).to_period('M').to_timestamp()
    if peak not in frame.index:
        return None

    def is_defense(stamp):
        row = frame.loc[stamp]
        if row['regime'] != 'defense':
            return False
        return (not confirmed_only) or bool(row['c'])

    if is_defense(peak):
        count = 0
        cursor = peak
        while cursor in frame.index and is_defense(cursor):
            count += 1
            cursor = _previous_month(cursor)
        return count - 1
    cursor = peak
    for _ in range(12):
        cursor = (pd.Timestamp(cursor) + pd.DateOffset(months=1)).to_period('M').to_timestamp()
        if cursor not in frame.index:
            break
        if is_defense(cursor):
            return -int((cursor.to_period('M') - peak.to_period('M')).n)
    return None


def turning_points(frame, episodes):
    rows = []
    for episode in episodes:
        peak = pd.Timestamp(episode['peak']).to_period('M').to_timestamp()
        at_peak = frame.loc[peak] if peak in frame.index else None
        rows.append({
            'peak': peak.strftime('%Y-%m'),
            'trough': pd.Timestamp(episode['trough']).strftime('%Y-%m'),
            'regime_at_peak': None if at_peak is None else (at_peak['regime'] or 'neutral'),
            'confirmed_at_peak': None if at_peak is None else bool(at_peak['c']),
            'g_at_peak': None if at_peak is None else round(float(at_peak['g']), 3),
            'l_at_peak': None if at_peak is None else round(float(at_peak['l']), 3),
            'months_since_last_defense': defense_lead(frame, peak, False),
            'months_since_last_confirmed_defense': defense_lead(frame, peak, True),
        })
    return rows


def revised_report(data, usrec):
    model_v2, model_v3 = _import_model()
    frame = _month_index(data['months'])
    episodes = nber_episodes(usrec)
    leads = turning_points(frame, episodes)
    regimes = [row.get('regime') for row in data['months']]
    confirmed = [bool(row.get('c')) for row in data['months']]
    dates = frame.index
    return {
        'label': 'revised_history_not_vintage',
        'episodes': leads,
        'whipsaw_reversals': model_v3.whipsaw_count(dates, regimes, confirmed),
        'transitions': model_v3.transition_matrix(dates, regimes),
        'confirmed_months': int(sum(confirmed)),
        'months': int(len(frame)),
    }


def _alfred_series(series_id, vintage, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{series_id}_{vintage.replace("-", "")}.csv'
    if not path.exists():
        url = ('https://alfred.stlouisfed.org/graph/alfredgraph.csv?id='
               + series_id + '&vintage_date=' + vintage)
        # urllib hangs on this host. curl with its own user agent returns the vintage CSV.
        partial = path.with_suffix('.part')
        subprocess.run(
            ['curl', '-fs', '--http1.1', '--max-time', '40', '-A', 'curl/8.5.0', '-o', str(partial), url],
            check=True, stderr=subprocess.DEVNULL,
        )
        partial.replace(path)
    frame = pd.read_csv(path)
    frame.columns = ['date', 'value']
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    frame['date'] = pd.to_datetime(frame['date'])
    return frame.dropna().set_index('date')['value']


def _asof_month(vintage):
    stamp = pd.Timestamp(vintage)
    return pd.Timestamp(year=stamp.year, month=stamp.month, day=1)


def score_vintage(vintage, cache_dir):
    model_v2, model_v3 = _import_model()
    raw = {}
    errors = {}
    for series_id in SERIES:
        try:
            raw[series_id] = _alfred_series(series_id, vintage, cache_dir)
        except Exception as exc:
            errors[series_id] = type(exc).__name__
            raw[series_id] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    monthly = {key: series.resample('MS').mean() for key, series in raw.items()}
    rrp = monthly['RRPONTSYD']
    if rrp.dropna().empty and not monthly['WALCL'].dropna().empty:
        rrp = pd.Series(0.0, index=monthly['WALCL'].index)
    netliq = model_v2.net_liquidity(monthly['WALCL'], monthly['WTREGEN'], rrp)
    yoy = lambda series: series.pct_change(12, fill_method=None) * 100.0
    growth = {
        'PERMIT': {'t': yoy(monthly['PERMIT']), 'w': .28},
        'INDPRO': {'t': yoy(monthly['INDPRO']), 'w': .24},
        'PAYEMS': {'t': yoy(monthly['PAYEMS']), 'w': .24},
        'UNRATE': {'t': -(monthly['UNRATE'] - monthly['UNRATE'].shift(12)), 'w': .14},
        'ICSA': {'t': -yoy(monthly['ICSA']), 'w': .10},
    }
    liquidity = model_v3.change_liquidity(monthly, netliq)
    for comps, source in ((growth, raw), (liquidity, {**raw, 'NETLIQ': netliq, 'DOLLAR': raw['DTWEXBGS']})):
        for key, component in comps.items():
            series = source.get(key, component['t'])
            component['available_from'] = series.first_valid_index() if hasattr(series, 'first_valid_index') else None
    asof = _asof_month(vintage)
    pieces = [component['t'] for component in (*growth.values(), *liquidity.values())]
    dates = [stamp for series in pieces for stamp in series.dropna().index]
    if not dates:
        return {'vintage': vintage, 'error': 'no_observations', 'source_errors': errors}
    index = pd.date_range(min(dates), asof, freq='MS')
    draft = model_v3.score_composite(growth, index, 'robust', 0.0)
    rho_g = model_v3.frozen_pairwise_rho(draft.z)
    ga = model_v3.score_composite(growth, index, 'robust', rho_g)
    draft_l = model_v3.score_composite(liquidity, index, 'robust', 0.0)
    rho_l = model_v3.frozen_pairwise_rho(draft_l.z)
    la = model_v3.score_composite(liquidity, index, 'robust', rho_l)
    frame = pd.DataFrame({
        'G': ga.score, 'L': la.score, 'gr': ga.raw, 'lr': la.raw,
        'owg': ga.observed_ratio, 'owl': la.observed_ratio,
    }).dropna(subset=['G', 'L', 'gr', 'lr', 'owg', 'owl'])
    frame = frame[(frame.index >= '1972-01-01') & (frame.index <= asof)]
    frame = frame[(frame.owg >= .4 - 1e-10) & (frame.owl >= .4 - 1e-10)]
    if frame.empty:
        return {'vintage': vintage, 'error': 'insufficient_coverage', 'source_errors': errors}
    gp = model_v2.expanding_percentile(ga.unit).reindex(frame.index)
    lp = model_v2.expanding_percentile(la.unit).reindex(frame.index)
    frame['gp'], frame['lp'] = gp, lp
    flags, _ = model_v3.classify(frame)
    last = frame.index[-1]
    flag = flags[-1]
    return {
        'vintage': vintage,
        'month': last.strftime('%Y-%m'),
        'g': round(float(frame.loc[last, 'G']), 3),
        'l': round(float(frame.loc[last, 'L']), 3),
        'regime': flag['regime'],
        'phase': flag['phase'],
        'c': flag['c'],
        'n': flag['n'],
        'ow': flag['ow'],
        'source_errors': errors,
    }


def _fmt_lead(value):
    if value is None:
        return '정점 전후 12개월 안에 방어 없음'
    if value > 0:
        return f'정점 {value}개월 전부터 방어'
    if value == 0:
        return '정점 당월에 방어 진입'
    return f'정점 이후 {-value}개월 뒤 방어'


def to_markdown(revised, vintages, revised_months):
    lines = [
        '# NBER 전환 · 휩소 · ALFRED 부분 결과',
        '',
        '기본 표는 수정 반영 이력입니다. 점수는 나중에 고쳐진 FRED 수치로 다시 계산한 것이므로, '
        '정점보다 앞섰다는 말은 실시간 예측이 아닙니다.',
        '',
        f"확인된 달 {revised['confirmed_months']} / 전체 {revised['months']}. "
        f"3개월 안에 직전 확인 국면으로 되돌아온 횟수(휩소) {revised['whipsaw_reversals']}.",
        '',
        '## 수정 이력에서 본 NBER 정점',
        '',
        'USREC는 정점 다음 달부터 저점 달까지 1입니다. 정점은 그 직전 달로 둡니다.',
        '',
        '| 정점 | 저점 | 정점의 국면 | G | L | 방어 진입 | 확인된 방어 진입 |',
        '| --- | --- | --- | ---: | ---: | --- | --- |',
    ]
    by_month = {row['d']: row for row in revised_months}
    for row in revised['episodes']:
        if row['peak'] < '1980-01':
            continue
        lines.append(
            f"| {row['peak']} | {row['trough']} | {row['regime_at_peak']} | "
            f"{row['g_at_peak']} | {row['l_at_peak']} | "
            f"{_fmt_lead(row['months_since_last_defense'])} | "
            f"{_fmt_lead(row['months_since_last_confirmed_defense'])} |"
        )
    lines.extend(['', '## 전이 횟수 (수정 이력, 중립 포함)', ''])
    states = ['expansion', 'liquidity', 'adjustment', 'defense', 'neutral']
    ko = {'expansion': '확장', 'liquidity': '유동성', 'adjustment': '조정', 'defense': '방어', 'neutral': '중립'}
    lines.append('| 에서 \\ 으로 | ' + ' | '.join(ko[s] for s in states) + ' |')
    lines.append('| --- | ' + ' | '.join('---:' for _ in states) + ' |')
    for src in states:
        cells = [str(revised['transitions'][src][dst]) for dst in states]
        lines.append('| ' + ko[src] + ' | ' + ' | '.join(cells) + ' |')
    lines.extend(['', '## ALFRED 빈티지', ''])
    if not vintages:
        lines.append('이번 실행은 `--vintage` 없이 수정 이력만 집계했습니다. 실시간 점수는 `python3 scripts/alfred_backtest.py --vintage`로 다시 계산합니다.')
    else:
        lines.append('각 날짜는 그 달 말까지 공개된 ALFRED 관측만 사용합니다. 비교 열은 같은 달의 현재 수정 이력입니다.')
        lines.append('')
        lines.append('| 빈티지 | 계산월 | 빈티지 G | 빈티지 L | 빈티지 국면 | 수정 G | 수정 L | 수정 국면 |')
        lines.append('| --- | --- | ---: | ---: | --- | ---: | ---: | --- |')
        for row in vintages:
            if row.get('error'):
                lines.append(f"| {row['vintage']} | — | — | — | {row['error']} | — | — | — |")
                continue
            revised_row = by_month.get(row['month'], {})
            regime = row['regime'] or 'neutral'
            revised_regime = revised_row.get('regime') or 'neutral'
            lines.append(
                f"| {row['vintage']} | {row['month']} | {row['g']} | {row['l']} | {regime} | "
                f"{revised_row.get('g', '—')} | {revised_row.get('l', '—')} | {revised_regime} |"
            )
        failed = sorted({key for row in vintages for key in row.get('source_errors', {})})
        if failed:
            lines.append('')
            lines.append('일부 달 말에 ALFRED가 404를 돌려준 시계열: ' + ', '.join(failed) + '.')
        lines.append('2001년 말일 빈티지는 금리·스프레드가 비어 유동성 커버리지가 0.4를 넘지 못했습니다. 2007년 이후 날짜는 점수를 만들었습니다.')
        lines.append('2009-04 빈티지의 L은 양수(유동성)이고, 같은 달의 수정 이력은 음수(방어)입니다. 방어 진입만으로 경기 저점을 예고했다고 읽지 않습니다. 2001·2020 정점은 성장이 약해도 유동성 변화가 플러스라 유동성 사분면에 머물렀습니다.')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'dist' / 'gl_data.json'))
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'alfred-backtest.md'))
    parser.add_argument('--vintage', action='store_true')
    parser.add_argument('--cache', default='/tmp/alfred-cache')
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    revised = revised_report(data, load_usrec())
    vintages = []
    if args.vintage:
        cache_dir = Path(args.cache)
        for vintage in VINTAGE_DATES:
            print('[vintage]', vintage, flush=True)
            vintages.append(score_vintage(vintage, cache_dir))
    text = to_markdown(revised, vintages, data['months'])
    Path(args.out).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
