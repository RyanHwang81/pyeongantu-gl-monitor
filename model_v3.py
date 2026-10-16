"""GL model v3.

Published scores are a single-stage robust (median/MAD) z-score, rescaled by the
theoretical variance of the weighted average. The second 216-month z-score is not
used. A classical single-stage score is retained beside it. v2 itself is unchanged
in model_v2.composite defaults.

Liquidity no longer mixes spread levels with rates of change. L is a 3–6 month
change composite. The old level indicators are a separate financial-conditions
score, which is the one that lines up with the Chicago Fed NFCI.
"""
from model_v2 import composite, expanding_percentile, quadrant

# Published as meta.model_version. scripts/check_dist_model.py blocks a deploy whose dist disagrees.
MODEL_VERSION = '3.0'
ENTER, EXIT, STRONG = 0.25, 0.10, 0.50
CONFIRM_MONTHS = 2
CARRY_LIMIT = 2
DIRECTION_MONTHS = 3
DEADBAND = 0.05
RHO_FREEZE = '2019-12-01'
PCT_ENTER_HI, PCT_ENTER_LO = 80.0, 20.0
PCT_HOLD_HI, PCT_HOLD_LO = 60.0, 40.0
NEUTRAL_ALLOCATION = '기본 비중 유지'

# Favor lists are disjoint. Gold and staples sit only in defense.
REGIME_ASSETS = {
    'expansion': {
        'favor': '산업재 · 소재 · 경기소비재 · 금융',
        'avoid': '유틸리티 · 장기채 · 필수소비재',
        'note': '성장·유동성 방향이 함께 우호. 이익 모멘텀이 있는 시클리컬이 앞서는 구간',
    },
    'liquidity': {
        'favor': '성장주 · IT/플랫폼 · 바이오 · 장기채',
        'avoid': '에너지 · 은행 · 소재',
        'note': '성장 수준은 약해도 유동성 변화는 우호. 듀레이션이 긴 성장주가 앞서는 구간',
    },
    'adjustment': {
        'favor': '에너지 · 배당/가치주 · 단기채',
        'avoid': '고밸류 성장주 · 리츠 · 장기채',
        'note': '성장은 버티지만 유동성은 조여짐. 현금흐름과 짧은 듀레이션이 상대 우위',
    },
    'defense': {
        'favor': '필수소비재 · 헬스케어 · 유틸리티 · 통신 · 현금 · 금',
        'avoid': '경기소비재 · 산업재 · 하이일드',
        'note': '성장과 유동성이 함께 약함. 방어주·현금·금의 상대 방어력이 부각되는 구간',
    },
}
PHASE_KO = {
    'early_recovery': '초기 회복',
    'late_cycle': '확장 후반',
    'expansion': '확장',
    'liquidity': '유동성',
    'adjustment': '조정',
    'defense': '방어',
    'neutral': '중립',
}


def change_liquidity(monthly, netliq):
    """Every L input is a change. Term and credit spreads use a 6-month change."""
    yoy = lambda s: s.pct_change(12, fill_method=None) * 100.0
    return {
        'M2SL': {'t': yoy(monthly['M2SL']), 'w': .18, 'label': 'M2 YoY'},
        'DGS2': {'t': -(monthly['DGS2'] - monthly['DGS2'].shift(3)), 'w': .18,
                 'label': '2년물 금리 3M 변화 (역)'},
        'NETLIQ': {'t': yoy(netliq), 'w': .15, 'label': '순유동성 YoY'},
        'DOLLAR': {'t': -yoy(monthly['DTWEXBGS']), 'w': .15, 'label': '광의 달러지수 YoY (역)'},
        'T10Y3M': {'t': monthly['T10Y3M'].diff(6), 'w': .12, 'label': '10Y-3M 6개월 변화'},
        'BAA10YM': {'t': -monthly['BAA10YM'].diff(6), 'w': .12, 'label': 'Baa-10Y 6개월 변화 (역)'},
        'TOTALSL': {'t': yoy(monthly['TOTALSL']), 'w': .10, 'label': '소비자신용 YoY'},
    }


def financial_conditions(monthly):
    """Level of the curve and the credit spread. Positive means easier conditions."""
    return {
        'T10Y3M': {'t': monthly['T10Y3M'], 'w': .50, 'label': '10Y-3M 기간스프레드 수준'},
        'BAA10YM': {'t': -monthly['BAA10YM'], 'w': .50, 'label': 'Baa-10Y 신용스프레드 수준 (역)'},
    }


def frozen_pairwise_rho(zdf, freeze=RHO_FREEZE, min_periods=60):
    """Causal expanding average pairwise correlation, frozen before the COVID window."""
    import pandas as pd
    import numpy as np
    cols = list(zdf.columns)
    parts = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                parts.append(zdf[cols[i]].expanding(min_periods).corr(zdf[cols[j]]))
    if not parts:
        return pd.Series(0.0, index=zdf.index)
    rho = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    freeze_ts = pd.Timestamp(freeze)
    known = rho.loc[:freeze_ts].dropna()
    if len(known):
        rho.loc[rho.index > freeze_ts] = known.iloc[-1]
    return rho.fillna(0.0).clip(0.0, 0.95)


def score_composite(comps, index, z_method, correlation=0.0):
    return composite(comps, index, carry_limit=CARRY_LIMIT, z_method=z_method,
                     rescale=True, correlation=correlation)


def step_axis(value, percentile, previous):
    """Schmitt trigger. Enter at ±0.25, exit at ±0.10. Percentiles can hold or enter."""
    import numpy as np
    if previous == 1:
        state = -1 if value <= -ENTER else (0 if value < EXIT else 1)
    elif previous == -1:
        state = 1 if value >= ENTER else (0 if value > -EXIT else -1)
    else:
        state = 1 if value >= ENTER else (-1 if value <= -ENTER else 0)
    if state == 0 and percentile is not None and np.isfinite(percentile):
        if previous == 1 and percentile >= PCT_HOLD_HI:
            return 1
        if previous == -1 and percentile <= PCT_HOLD_LO:
            return -1
        if percentile >= PCT_ENTER_HI and value > 0:
            return 1
        if percentile <= PCT_ENTER_LO and value < 0:
            return -1
    return state


def _direction(delta):
    import numpy as np
    if delta is None or not np.isfinite(delta):
        return 0
    if delta > DEADBAND:
        return 1
    if delta < -DEADBAND:
        return -1
    return 0


def phase_of(g_state, l_state, dg, dl, regime):
    if regime is None:
        return 'neutral'
    rising_g = _direction(dg) > 0
    rising_l = _direction(dl) > 0
    if g_state < 0 and rising_g and (l_state > 0 or rising_l):
        return 'early_recovery'
    if g_state > 0 and l_state > 0 and _direction(dg) < 0:
        return 'late_cycle'
    return regime


def classify(frame):
    """Hysteresis regimes. Neutral months reset the streak and do not update the last quadrant.

    `previous` in v2 was replaced on neutral and borderline months, so the next
    month could confirm against a quadrant that was never confirmed. This loop
    updates the confirmed quadrant only when the month itself confirms.
    """
    import numpy as np
    g_state = l_state = 0
    streak_q = None
    streak = 0
    rows = []
    values_g = frame['G'].to_numpy()
    values_l = frame['L'].to_numpy()
    for i, (dt, row) in enumerate(frame.iterrows()):
        gp = row.gp if 'gp' in frame.columns else np.nan
        lp = row.lp if 'lp' in frame.columns else np.nan
        g_state = step_axis(float(row.G), gp, g_state)
        l_state = step_axis(float(row.L), lp, l_state)
        provisional = min(float(row.owg), float(row.owl)) < .8 - 1e-10
        neutral = g_state == 0 and l_state == 0
        borderline = (g_state == 0) ^ (l_state == 0)
        regime = None if g_state == 0 or l_state == 0 else quadrant(
            1 if g_state > 0 else -1, 1 if l_state > 0 else -1)
        sign_r = quadrant(float(row.G), float(row.L))
        dg = float(values_g[i] - values_g[i - DIRECTION_MONTHS]) if i >= DIRECTION_MONTHS else np.nan
        dl = float(values_l[i] - values_l[i - DIRECTION_MONTHS]) if i >= DIRECTION_MONTHS else np.nan
        strong = abs(float(row.G)) >= STRONG and abs(float(row.L)) >= STRONG
        confirmed = False
        if provisional or regime is None and (neutral or borderline):
            if not provisional and neutral:
                streak = 0
                streak_q = None
        elif regime is not None and not provisional:
            if regime == streak_q:
                streak += 1
            else:
                streak_q = regime
                streak = 1
            confirmed = streak >= CONFIRM_MONTHS or (strong and streak >= 1)
        phase = phase_of(g_state, l_state, dg, dl, regime)
        rows.append({
            'r': sign_r,
            'regime': regime,
            'phase': phase,
            'p': bool(provisional),
            'n': bool(neutral),
            'b': bool(borderline),
            'c': bool(confirmed),
            'x': bool(float(row.G) * float(row.gr) < 0 or float(row.L) * float(row.lr) < 0),
            'ow': {'g': round(float(row.owg), 2), 'l': round(float(row.owl), 2)},
            'dg': None if dg != dg else round(dg, 3),
            'dl': None if dl != dl else round(dl, 3),
            'ax': {'g': int(g_state), 'l': int(l_state)},
            'strong': bool(strong),
            'streak': int(streak if regime is not None and not provisional else 0),
            'allocation': REGIME_ASSETS[regime]['favor'] if regime else NEUTRAL_ALLOCATION,
        })
    return rows, {'g': g_state, 'l': l_state}


def favor_tokens(text):
    return [part.strip() for part in text.split('·')]


def favor_sets_disjoint():
    seen = {}
    for regime, spec in REGIME_ASSETS.items():
        for token in favor_tokens(spec['favor']):
            if token in seen:
                return False
            seen[token] = regime
    return True


def transition_matrix(dates, regimes):
    """Month-to-month counts. Neutral is its own state. Dates stay in order."""
    states = ['expansion', 'liquidity', 'adjustment', 'defense', 'neutral']
    counts = {a: {b: 0 for b in states} for a in states}
    prev = None
    for regime in regimes:
        cur = regime or 'neutral'
        if prev is not None:
            counts[prev][cur] += 1
        prev = cur
    return counts


def whipsaw_count(dates, regimes, confirmed):
    """Confirmed flips that return to the prior regime within three months."""
    import pandas as pd
    series = [(pd.Timestamp(dt), reg) for dt, reg, ok in zip(dates, regimes, confirmed)
              if ok and reg]
    count = 0
    for i in range(1, len(series)):
        if series[i][1] == series[i - 1][1]:
            continue
        old = series[i - 1][1]
        start = series[i][0]
        for dt, reg in series[i + 1:]:
            if dt > start + pd.DateOffset(months=3):
                break
            if reg == old:
                count += 1
                break
    return count


def percentile_of(series):
    scored = series.copy()
    return expanding_percentile(scored)
