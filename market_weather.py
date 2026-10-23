"""Market-price regime ("weather") shown beside the v3 regime ("season"). Reference only.

G_m: cyclical − defensive industry return over 6 months (.5) and the 6-month
change in 10Y − 3M (.5). L_m: the 3-month change in the 2-year yield, inverted
(.5), and the 6-month change in Baa − 10Y, inverted (.5). Scored like v3:
median/MAD z, pairwise correlation frozen at 2019-12, theoretical variance,
EMA3, ±0.25 / ±0.10 hysteresis. No two-month confirmation, so it moves faster.

Validation (analysis PRs #12 and #14): G_m led the v3 growth score and the
confirmed regime changed after the market in 64% of 2006–2026 cases. Asset
guidance and weight rules built on it did not pass, so it is a display only.
"""
import io
import zipfile

import pandas as pd

import model_v3 as v3
from model_v2 import expanding_percentile

FRENCH_INDUSTRIES = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_CSV.zip'
CYCLICAL = ('Durbl', 'Manuf', 'Shops')
DEFENSIVE = ('NoDur', 'Hlth', 'Utils')
SPEC = 'M1'
LABEL_KO = '시장 국면 (날씨)'


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
    if not rows:
        raise ValueError('no monthly rows in industry file')
    frame = pd.DataFrame(rows, columns=['ym'] + header)
    frame.index = pd.to_datetime(frame.pop('ym'), format='%Y%m')
    frame = frame.apply(pd.to_numeric, errors='coerce')
    return frame.mask(frame <= -99.99) / 100.0


def industries_from_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
        return z.read(member).decode('latin-1')


def components(monthly, industries):
    cyc_def = industries[list(CYCLICAL)].mean(axis=1) - industries[list(DEFENSIVE)].mean(axis=1)
    curve = monthly['GS10'] - monthly['TB3MS']
    signals = {
        'CYCDEF': cyc_def.rolling(6).sum() * 100,
        'CURVE': curve - curve.shift(6),
        'DGS2': -(monthly['DGS2'] - monthly['DGS2'].shift(3)),
        'BAA': -(monthly['BAA10YM'] - monthly['BAA10YM'].shift(6)),
    }
    pick = lambda keys: {k: {'t': signals[k].dropna(), 'w': .5,
                             'available_from': signals[k].first_valid_index()} for k in keys}
    return pick(('CYCDEF', 'CURVE')), pick(('DGS2', 'BAA'))


def _score(comps, index):
    draft = v3.score_composite(comps, index, 'robust', 0.0)
    return v3.score_composite(comps, index, 'robust', v3.frozen_pairwise_rho(draft.z))


def build(monthly, industries, index):
    """Monthly weather rows up to the month before index[-1], and the latest row."""
    g_comps, l_comps = components(monthly, industries)
    g, l = _score(g_comps, index), _score(l_comps, index)
    df = pd.DataFrame({'G': g.score, 'L': l.score, 'gr': g.raw, 'lr': l.raw,
                       'owg': g.observed_ratio, 'owl': l.observed_ratio}).dropna()
    df = df[(df.index >= '1972-01-01') & (df.index < index[-1])]
    df = df[(df.owg >= .4 - 1e-10) & (df.owl >= .4 - 1e-10)]
    if df.empty:
        return None
    df['gp'] = expanding_percentile(g.unit).reindex(df.index)
    df['lp'] = expanding_percentile(l.unit).reindex(df.index)
    flags, _ = v3.classify(df)
    # p marks months where either axis has under 80% observed weight, as in months[].p.
    rows = [{'d': d.strftime('%Y-%m'), 'g': round(float(r.G), 3), 'l': round(float(r.L), 3), 'w': f['regime'],
             'p': bool(min(float(r.owg), float(r.owl)) < .8 - 1e-10)}
            for (d, r), f in zip(df.iterrows(), flags)]
    last = df.index[-1]
    carried = {k: used for result in (g, l) for k, used in result.carried.loc[last].items() if used}
    return {
        'spec': SPEC,
        'label_ko': LABEL_KO,
        'weights': {'g': {k: c['w'] for k, c in g_comps.items()}, 'l': {k: c['w'] for k, c in l_comps.items()}},
        'industries_as_of': industries.dropna(how='all').index[-1].strftime('%Y-%m'),
        'latest': {**rows[-1], 'carried': carried,
                   'ow': {'g': round(float(df.owg.iloc[-1]), 2), 'l': round(float(df.owl.iloc[-1]), 2)}},
        'months': rows,
        'note_ko': '시장 가격만으로 만든 참고 표시입니다. 확정 국면보다 먼저 움직이는 경향이 있지만, '
                   '자산 비중이나 매매 시점의 근거로 검증되지 않았습니다.',
    }
