#!/usr/bin/env python3
"""Forward 3/6/12-month returns conditional on the published v3 regime.

Neutral months stay out of the four quadrants and are reported on their own
row. Returns use the month-end level after the signal month, so they are
subsequent returns, not the trailing 12-month figures stored on each row.
Overlapping windows are not independent; the sample count is the number of
regime months with a complete forward price, not the number of independent episodes.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (3, 6, 12)
ORDER = ('expansion', 'liquidity', 'adjustment', 'defense', 'neutral')
KO = {
    'expansion': '확장',
    'liquidity': '유동성',
    'adjustment': '조정',
    'defense': '방어',
    'neutral': '기본 비중 유지',
}


def _cache_csv(name):
    cache = os.environ.get('GL_CACHE', '')
    path = Path(cache) / name if cache else None
    if path and path.exists():
        return path
    return None


def _fred_monthly(series_id):
    path = _cache_csv(series_id + '.csv')
    if path is None:
        os.environ.setdefault('GL_CACHE', '/tmp/gl-cache')
        import sys
        sys.path.insert(0, str(ROOT))
        import build
        series = build.fred(series_id)
    else:
        frame = pd.read_csv(path)
        frame.columns = ['date', 'value']
        frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
        frame['date'] = pd.to_datetime(frame['date'])
        series = frame.dropna().set_index('date')['value']
    return series.resample('MS').mean()


def _levels():
    spx_path = _cache_csv('spx.csv')
    gold_path = _cache_csv('gold.csv')
    if spx_path is None or gold_path is None:
        raise SystemExit('GL_CACHE must contain spx.csv and gold.csv')
    spx = pd.read_csv(spx_path, usecols=['Date', 'SP500'])
    spx['Date'] = pd.to_datetime(spx['Date'])
    spx = spx.dropna().set_index('Date')['SP500'].resample('MS').last()
    gold = pd.read_csv(gold_path)
    gold.columns = ['date', 'price']
    gold['date'] = pd.to_datetime(gold['date'])
    gold = gold.set_index('date')['price'].resample('MS').last()
    nasdaq = _fred_monthly('NASDAQCOM')
    wti = _fred_monthly('WTISPLC')
    gs10 = _fred_monthly('GS10')
    tb3 = _fred_monthly('TB3MS')
    y = gs10 / 100.0
    duration = (1 - (1 + y.shift(1)) ** -10) / y.shift(1)
    ust = (1 + (y.shift(1) / 12 - duration * (y - y.shift(1))).fillna(0)).cumprod()
    cash = (1 + (tb3 / 100.0 / 12).fillna(0)).cumprod()
    return {
        'spx': ('S&P500', spx),
        'ndx': ('나스닥', nasdaq),
        'gold': ('금', gold),
        'wti': ('WTI', wti),
        'ust': ('미국채 10Y', ust),
        'cash': ('현금', cash),
    }


def _stats(values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {'n': 0, 'mean': None, 'median': None}
    return {
        'n': int(clean.size),
        'mean': round(float(np.mean(clean) * 100), 2),
        'median': round(float(np.median(clean) * 100), 2),
    }


def summarize(data, levels):
    months = pd.DataFrame(data['months'])
    months['date'] = pd.to_datetime(months['d'])
    months = months.set_index('date').sort_index()
    months['bucket'] = months['regime'].fillna('neutral')
    rows = []
    for key, (label, level) in levels.items():
        for horizon in HORIZONS:
            forward = (level.shift(-horizon) / level - 1.0).reindex(months.index)
            for bucket in ORDER:
                picked = forward[months['bucket'] == bucket]
                stats = _stats(picked.to_numpy())
                rows.append({'asset': key, 'label': label, 'horizon': horizon,
                             'regime': bucket, **stats})
    phase = months['phase'] == 'early_recovery' if 'phase' in months else None
    phase_rows = []
    if phase is not None and phase.any():
        for key, (label, level) in levels.items():
            for horizon in HORIZONS:
                forward = (level.shift(-horizon) / level - 1.0).reindex(months.index)
                stats = _stats(forward[phase].to_numpy())
                phase_rows.append({'asset': key, 'label': label, 'horizon': horizon, **stats})
    return rows, phase_rows, int((months['bucket'] == 'neutral').sum()), int(len(months))


def _cell(row):
    if not row or row['n'] == 0:
        return '—'
    return f"{row['mean']:+.1f}% / {row['median']:+.1f}% (n={row['n']})"


def to_markdown(rows, phase_rows, neutral_n, total_n, generated):
    lines = [
        '# 국면별 선행 수익률',
        '',
        f'산출일 {generated}. 공개 v3 `regime`이 있는 달만 사분면으로 집계합니다. '
        f'중립 {neutral_n}개월(전체 {total_n}개월)은 사분면을 주지 않고 `기본 비중 유지`로 따로 둡니다.',
        '',
        '숫자는 해당 월 말 이후 3·6·12개월 총수익률의 평균 / 중위수입니다. '
        '창이 겹치므로 n은 독립 표본 수가 아닙니다. 미국채는 듀레이션 근사, 현금은 3개월물 금리 누적입니다.',
        '',
    ]
    assets = []
    for row in rows:
        if row['label'] not in assets:
            assets.append(row['label'])
    lookup = {(row['label'], row['horizon'], row['regime']): row for row in rows}
    for horizon in HORIZONS:
        lines.append(f'## {horizon}개월')
        lines.append('')
        header = '| 자산 | ' + ' | '.join(KO[name] for name in ORDER) + ' |'
        rule = '| --- | ' + ' | '.join('---:' for _ in ORDER) + ' |'
        lines.extend([header, rule])
        for label in assets:
            cells = [_cell(lookup.get((label, horizon, name))) for name in ORDER]
            lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    if phase_rows:
        lines.append('## 초기 회복 (사분면이 아니라 방향 메모)')
        lines.append('')
        lines.append('| 자산 | 3개월 | 6개월 | 12개월 |')
        lines.append('| --- | ---: | ---: | ---: |')
        by_label = {}
        for row in phase_rows:
            by_label.setdefault(row['label'], {})[row['horizon']] = row
        for label, horizons in by_label.items():
            lines.append('| ' + label + ' | ' + ' | '.join(_cell(horizons.get(h)) for h in HORIZONS) + ' |')
        lines.append('')
    lines.append('12개월 수익률에는 다음 국면의 반등이 섞입니다. 금의 3·6개월 평균은 방어가 유동성보다 높고, 12개월 평균도 방어가 높지만 중위수는 유동성이 더 높습니다. 금과 필수소비재는 방어에만 두며 다른 사분면과 겹치지 않습니다.')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'dist' / 'gl_data.json'))
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'regime-forward-returns.md'))
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    rows, phase_rows, neutral_n, total_n = summarize(data, _levels())
    text = to_markdown(rows, phase_rows, neutral_n, total_n, data['meta'].get('generated', ''))
    Path(args.out).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
