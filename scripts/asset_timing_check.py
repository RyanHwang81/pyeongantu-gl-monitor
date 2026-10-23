#!/usr/bin/env python3
"""Is the regime asset list wrong because of timing? Preregistered in
docs/asset-timing-preregistration.md. Analysis only.

The dashboard list (REGIME_ASSETS) is mapped to measurable assets: Ken French
10 industries and book-to-market Lo 30 / Hi 30 portfolios (value weighted),
par-bond approximations from GS10 / GS5, TB3MS cash and monthly gold. The list
score S is the mean favored-asset return minus the mean avoided-asset return
over a window.

A signal month t with publication lag `lag` and shift k uses the window
t+1+lag+k .. t+lag+k+h. Season: lag 1 (k = 0 is the first tradable window).
Weather: lag 0. Same-month baseline: lag −1, h = 1.

Usage: python3 scripts/asset_timing_check.py [--cache DIR] [--json OUT]
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
TRAIN_START = pd.Timestamp('1972-01-01')
H = 6
SHIFTS = range(-18, 19)
QUADRANTS = rr.QUADRANTS
NEXT, PREV = mr.NEXT, mr.PREV
KO = {'expansion': '실적', 'liquidity': '금융', 'adjustment': '역금융', 'defense': '역실적'}

LISTS = {
    'expansion': {'favor': ['Manuf', 'Durbl', 'Shops', 'Other'], 'avoid': ['Utils', 'bond10', 'NoDur']},
    'liquidity': {'favor': ['Lo30', 'HiTec', 'Hlth', 'bond10'], 'avoid': ['Enrgy', 'Other', 'Manuf']},
    'adjustment': {'favor': ['Enrgy', 'Hi30', 'bond5'], 'avoid': ['Lo30', 'bond10']},
    'defense': {'favor': ['NoDur', 'Hlth', 'Utils', 'Telcm', 'cash', 'gold'], 'avoid': ['Durbl', 'Shops', 'Manuf']},
}
ASSET_KO = {'NoDur': '비내구재(필수소비재)', 'Durbl': '내구재', 'Manuf': '제조(산업재·소재)', 'Enrgy': '에너지',
            'HiTec': 'IT', 'Telcm': '통신', 'Shops': '유통', 'Hlth': '헬스케어', 'Utils': '유틸리티',
            'Other': '기타(금융 포함)', 'Lo30': '성장주(B/M 하위 30%)', 'Hi30': '가치주(B/M 상위 30%)',
            'bond10': '10년 국채', 'bond5': '5년 국채', 'cash': '현금', 'gold': '금'}


# ---------------------------------------------------------------- data

def asset_returns(cache):
    ind = rr.load_french(cache, '10_Industry_Portfolios')
    bm = rr.load_french(cache, 'Portfolios_Formed_on_BE-ME')
    base = rr.monthly_returns(cache)
    gold = mr.load_gold(cache).pct_change(fill_method=None)
    frame = pd.concat([ind, bm[['Lo 30', 'Hi 30']].rename(columns={'Lo 30': 'Lo30', 'Hi 30': 'Hi30'}),
                       base[['bond10', 'bond5', 'cash']], gold.rename('gold')], axis=1, sort=True)
    return frame.loc['1960-01-01':, list(ASSET_KO)]


def level_regime(cache):
    """Book-style level quadrant: CFNAI-MA3 sign for growth, −NFCI sign for liquidity."""
    g = gb.to_monthly(gb.load_fred(cache, 'CFNAIMA3'))
    l = -gb.to_monthly(gb.load_fred(cache, 'NFCI'))
    both = pd.concat({'g': g, 'l': l}, axis=1, sort=True).dropna()
    return pd.Series([gb.v2.quadrant(a, b) for a, b in zip(both.g, both.l)], index=both.index)


# ---------------------------------------------------------------- scores

def forward(returns, h, end_offset):
    """Value at t = compounded return over months t+end_offset−h+1 .. t+end_offset."""
    total = np.exp(np.log1p(returns).rolling(h, min_periods=h).sum()) - 1.0
    return total.shift(-end_offset)


def list_score(fwd, labels, mapping=None):
    """S_t = mean favored − mean avoided for the list chosen by mapping(label)."""
    mapping = mapping or (lambda q: q)
    out = pd.Series(np.nan, index=fwd.index)
    for q in QUADRANTS:
        spec = LISTS[mapping(q)]
        s = fwd[spec['favor']].mean(axis=1) - fwd[spec['avoid']].mean(axis=1)
        mask = (labels.reindex(fwd.index) == q).to_numpy()
        out[mask] = s[mask]
    return out


def masks(index, end_offset, start=TRAIN_START):
    end = index + pd.DateOffset(months=end_offset) if end_offset >= 0 else index - pd.DateOffset(months=-end_offset)
    # Signal month and window both stay inside the training years, so no signal is in both sets.
    train = pd.Series((index >= start) & (index <= TRAIN_END) & (end <= TRAIN_END), index=index)
    hold = pd.Series(index >= HOLDOUT_START, index=index)
    return train, hold


def hac(series, lags):
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 24:
        return {'n': int(len(y)), 'mean': None, 't': None}
    b, cov, n = rr.newey_west_means(y, np.zeros(len(y)), [0.0], max(lags, 1))
    return {'n': int(n), 'mean': round(float(b[0]) * 100, 2), 't': round(float(b[0] / np.sqrt(cov[0, 0])), 2)}


def evaluate(returns, labels, lag, k=0, h=H, mapping=None):
    e = lag + k + h
    fwd = forward(returns, h, e)
    s = list_score(fwd, labels, mapping)
    train, hold = masks(s.index, e)
    return {'train': hac(s[train], h), 'holdout': hac(s[hold], h), 'k': k, 'lag': lag, 'h': h}


def fit_matrix(returns, labels, lag, k=0, h=H, part='train'):
    """Mean S of every list on months of every label. Diagonal = own list."""
    e = lag + k + h
    fwd = forward(returns, h, e)
    train, hold = masks(fwd.index, e)
    m = train if part == 'train' else hold
    out = {}
    lab = labels.reindex(fwd.index)
    for q in QUADRANTS:
        rows = {}
        pick = (lab == q) & m
        for L in QUADRANTS:
            spec = LISTS[L]
            s = (fwd[spec['favor']].mean(axis=1) - fwd[spec['avoid']].mean(axis=1))[pick].dropna()
            rows[L] = None if s.empty else round(float(s.mean()) * 100, 2)
        out[q] = {'n': int(pick.sum()), 'lists': rows,
                  'best': max((L for L in rows if rows[L] is not None), key=lambda L: rows[L], default=None)}
    return out


def passes(result, train_t=2.0, hold_t=1.65):
    tr, ho = result['train'], result['holdout']
    return bool(tr['mean'] is not None and tr['mean'] > 0 and tr['t'] >= train_t
                and ho['mean'] is not None and ho['mean'] > 0 and ho['t'] >= hold_t)


def spell_hits(returns, labels):
    """H-b at spell level: mean same-month S per spell of the regime."""
    fwd = forward(returns, 1, 0)
    s = list_score(fwd, labels)
    rows = []
    for lab, a, b, months in mr.spells(labels):
        if lab not in QUADRANTS:
            continue
        mean = s.loc[a:b].mean()
        if np.isfinite(mean):
            rows.append({'regime': lab, 'from': a.strftime('%Y-%m'), 'months': months, 'mean': float(mean)})
    out = {}
    for part, cond in (('train', lambda r: r['from'] <= '2005-12'), ('holdout', lambda r: r['from'] >= '2006-01')):
        pick = [r for r in rows if cond(r)]
        out[part] = {'spells': len(pick), 'hit': None if not pick else round(float(np.mean([r['mean'] > 0 for r in pick])), 3)}
    return out


def item_check(returns, alignments):
    """H-e. Direction × (asset − cross-asset mean) on months of the item's regime."""
    out = {}
    for q in QUADRANTS:
        for side, sign in (('favor', 1), ('avoid', -1)):
            for asset in LISTS[q][side]:
                key = f'{q}:{side}:{asset}'
                row = {'regime': q, 'side': side, 'asset': asset}
                for name, (labels, lag, h, k) in alignments.items():
                    e = lag + k + h
                    fwd = forward(returns, h, e)
                    rel = fwd[asset] - fwd.mean(axis=1)
                    train, hold = masks(fwd.index, e)
                    lab = labels.reindex(fwd.index) == q
                    for part, m in (('train', train), ('holdout', hold)):
                        v = (sign * rel)[lab & m].dropna()
                        row[f'{name}/{part}'] = None if len(v) < 6 else round(float(v.mean()) * 100, 2)
                # Best training shift for the season alignment.
                season_labels, lag, h, _ = alignments['season']
                best = None
                for k in SHIFTS:
                    e = lag + k + h
                    fwd = forward(returns, h, e)
                    rel = fwd[asset] - fwd.mean(axis=1)
                    train, _ = masks(fwd.index, e)
                    v = (sign * rel)[(season_labels.reindex(fwd.index) == q) & train].dropna()
                    if len(v) >= 12 and (best is None or v.mean() > best[1]):
                        best = (k, float(v.mean()))
                row['season_best_k'] = None if best is None else best[0]
                row['season_best_mean'] = None if best is None else round(best[1] * 100, 2)
                trains = [row[f'{n}/train'] for n in alignments]
                row['wrong_everywhere'] = bool(all(v is not None and v < 0 for v in trains)
                                               and row['season/holdout'] is not None and row['season/holdout'] < 0)
                out[key] = row
    return out


def season_age(season):
    """Months since the current season began (0 in its first month)."""
    s = season.dropna()
    ages, start, prev = [], None, None
    for stamp, value in s.items():
        if value != prev:
            start = stamp
        ages.append((stamp.year - start.year) * 12 + stamp.month - start.month)
        prev = value
    return pd.Series(ages, index=s.index)


def age_split(returns, season, k, cut=12):
    """Exploratory: S at shift k for months early (≤ cut) and late (> cut) in their season."""
    e = 1 + k + H
    fwd = forward(returns, H, e)
    s = list_score(fwd, season)
    age = season_age(season).reindex(s.index)
    train, hold = masks(s.index, e)
    out = {}
    for part, m in (('train', train), ('holdout', hold)):
        out[part] = {'early': hac(s[m & (age <= cut)], H), 'late': hac(s[m & (age > cut)], H)}
    return out


# ---------------------------------------------------------------- run

def run(cache, end_month):
    returns = asset_returns(cache)
    pub = rr.published_frame()
    season = mr.season_series(pub)
    regime = pub['regime']
    weather = mr.market_frame(mr.market_signals(cache), 'M1', end_month)['regime']
    level = level_regime(cache)
    result = {'data_end': returns.dropna(how='all').index[-1].strftime('%Y-%m'), 'h': H}

    shift_rows = {k: evaluate(returns, season, 1, k) for k in SHIFTS}
    k_star = max(SHIFTS, key=lambda k: shift_rows[k]['train']['mean'] if shift_rows[k]['train']['mean'] is not None else -1e9)
    ha = shift_rows[k_star]
    result['H-a'] = {'by_k': {str(k): v for k, v in shift_rows.items()}, 'k_star': k_star,
                     'pass': bool(k_star <= -3 and passes(ha)), 'k0': shift_rows[0]}
    result['age_split'] = {str(k): age_split(returns, season, k) for k in (k_star, 0)}
    result['H-a_next'] = {**evaluate(returns, season, 1, 0, mapping=lambda q: NEXT[q])}
    result['H-a_next']['pass'] = passes(result['H-a_next'])
    result['H-a_prev'] = {**evaluate(returns, season, 1, 0, mapping=lambda q: PREV[q])}
    result['H-a_prev']['pass'] = passes(result['H-a_prev'])
    result['H-b'] = {**evaluate(returns, regime, -1, 0, h=1), 'spells': spell_hits(returns, regime)}
    sp = result['H-b']['spells']
    result['H-b']['pass'] = bool(passes(result['H-b']) and sp['train']['hit'] is not None and sp['train']['hit'] >= .6
                                 and sp['holdout']['hit'] is not None and sp['holdout']['hit'] >= .6)
    result['H-c'] = evaluate(returns, weather, 0, 0)
    result['H-c']['pass'] = passes(result['H-c'])
    result['H-d'] = evaluate(returns, level, 1, 0)
    result['H-d']['pass'] = passes(result['H-d'])
    result['h12'] = {name: evaluate(returns, lab, lag, 0, h=12, mapping=mp)
                     for name, (lab, lag, mp) in {'season': (season, 1, None), 'season_next': (season, 1, lambda q: NEXT[q]),
                                                   'weather': (weather, 0, None), 'level': (level, 1, None)}.items()}
    result['matrix'] = {
        'season_k0_train': fit_matrix(returns, season, 1, 0, part='train'),
        'season_k0_holdout': fit_matrix(returns, season, 1, 0, part='holdout'),
        'same_month_train': fit_matrix(returns, regime, -1, 0, h=1, part='train'),
        'same_month_holdout': fit_matrix(returns, regime, -1, 0, h=1, part='holdout'),
        'weather_holdout': fit_matrix(returns, weather, 0, 0, part='holdout'),
    }
    alignments = {'season': (season, 1, H, 0), 'same_month': (regime, -1, 1, 0),
                  'weather': (weather, 0, H, 0), 'level': (level, 1, H, 0)}
    result['H-e'] = item_check(returns, alignments)
    hb, ha_any = result['H-b']['pass'], result['H-a']['pass'] or result['H-a_next']['pass']
    k0_wrong = not passes(result['H-a']['k0'])
    result['cause'] = {
        'timing': bool(hb and k0_wrong and ha_any),
        'definition': bool(result['H-d']['pass'] and not (hb or ha_any or result['H-c']['pass'])),
        'list': bool(not hb),
    }
    remedies = {'1 다음 국면 목록': result['H-a_next'], '2 날씨에 목록': result['H-c']}
    ok = {k: v for k, v in remedies.items() if v['holdout']['mean'] is not None and v['holdout']['mean'] > 0
          and v['holdout']['t'] is not None and v['holdout']['t'] >= 1.65}
    result['recommend'] = max(ok, key=lambda k: ok[k]['holdout']['t']) if ok else None
    return result


def markdown(r):
    f = lambda x: f"{x['mean']:+.2f}% (t {x['t']}, n {x['n']})" if x['mean'] is not None else '—'
    lines = [f"data_end {r['data_end']} h {r['h']}", '', '## H-a by k (season, own list)']
    for k, v in r['H-a']['by_k'].items():
        lines.append(f"k {int(k):+3d} train {f(v['train'])} | holdout {f(v['holdout'])}")
    lines.append(f"k* {r['H-a']['k_star']} pass {r['H-a']['pass']}")
    for k, v in r['age_split'].items():
        lines.append(f"age split k {k}: " + ' | '.join(f"{p} early {f(x['early'])} late {f(x['late'])}" for p, x in v.items()))
    for key in ('H-a_next', 'H-a_prev', 'H-b', 'H-c', 'H-d'):
        v = r[key]
        lines.append(f"{key} pass {v['pass']} train {f(v['train'])} | holdout {f(v['holdout'])}" + (f" spells {v['spells']}" if 'spells' in v else ''))
    lines.append('h12 ' + json.dumps({k: {'train': v['train'], 'holdout': v['holdout']} for k, v in r['h12'].items()}, ensure_ascii=False))
    lines.append('cause ' + json.dumps(r['cause']) + f" recommend {r['recommend']}")
    for name, m in r['matrix'].items():
        lines.append(f'## matrix {name}')
        for q, row in m.items():
            lines.append(f"{KO[q]} n{row['n']} best {KO.get(row['best'], row['best'])} " + ' '.join(
                f"{KO[L]}:{'—' if v is None else f'{v:+.2f}'}" for L, v in row['lists'].items()))
    lines.append('## H-e')
    for key, row in r['H-e'].items():
        cells = ' '.join(f"{k.split('/')[0][:4]}/{k.split('/')[1][:2]} {'—' if v is None else f'{v:+.2f}'}"
                         for k, v in row.items() if '/' in k)
        lines.append(f"{KO[row['regime']]} {row['side']} {ASSET_KO[row['asset']]}: {cells} best_k {row['season_best_k']} "
                     f"({row['season_best_mean']}) WRONG={row['wrong_everywhere']}")
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
