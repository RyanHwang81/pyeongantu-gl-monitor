#!/usr/bin/env python3
"""Build dist/thermo_data.json: market temperature for indices and sectors.

Temperature = 0.4 sentiment + 0.3 weekly RSI + 0.3 daily RSI, each 0-100.
Sentiment is our own price-based proxy, not the CNN Fear & Greed index:
- US market: S&P 500 vs its 125-day MA, 52-week range position, VIX vs its
  50-day MA (inverted), SPY-IEF and HYG-IEF 20-day returns, RSP/SPY 63-day
  change. Each is a percentile in a trailing 504-day window (no look-ahead).
- KR market and every sector: 125-day MA percentile, 52-week position and
  20-day realized volatility percentile (inverted).
- Sector sentiment is half market, half its own.
Weekly RSI uses last Friday's close (the current week is not counted).

The app shows this as a description of the market mood only. It is not a
buy or sell signal; the app does not say what to do with it.

Usage:
  python3 scripts/build_trading_temp.py [--out dist/thermo_data.json]
  TEMP_CACHE=/path python3 scripts/build_trading_temp.py   # offline, <SYMBOL>.csv (date,close)
Exit code 1 when any index cannot be scored.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TEMP_VERSION = 'trading-temp-1'
WINDOW = 504
HISTORY_DAYS = 120
YEARS = 5  # yearly low/high and weekly line for the current year and the 4 before
UA = {'User-Agent': 'Mozilla/5.0 (pyeongantu-gl-monitor)'}

# (symbol, name, market, kind). Korean sectors are KODEX sector ETFs.
TARGETS = [
    ('^GSPC', 'S&P500', 'US', 'index'),
    ('^NDX', '나스닥100', 'US', 'index'),
    ('^KS11', '코스피', 'KR', 'index'),
    ('^KQ11', '코스닥', 'KR', 'index'),
    ('XLK', '기술', 'US', 'sector'),
    ('XLF', '금융', 'US', 'sector'),
    ('XLE', '에너지', 'US', 'sector'),
    ('XLV', '헬스케어', 'US', 'sector'),
    ('XLI', '산업재', 'US', 'sector'),
    ('XLY', '경기소비재', 'US', 'sector'),
    ('XLP', '필수소비재', 'US', 'sector'),
    ('XLU', '유틸리티', 'US', 'sector'),
    ('XLB', '소재', 'US', 'sector'),
    ('XLRE', '부동산', 'US', 'sector'),
    ('XLC', '커뮤니케이션', 'US', 'sector'),
    ('SMH', '반도체', 'US', 'sector'),
    ('091160.KS', '반도체', 'KR', 'sector'),
    ('091180.KS', '자동차', 'KR', 'sector'),
    ('091170.KS', '은행', 'KR', 'sector'),
    ('305720.KS', '2차전지', 'KR', 'sector'),
    ('117680.KS', '철강', 'KR', 'sector'),
    ('266420.KS', '헬스케어', 'KR', 'sector'),
    ('102960.KS', '기계장비', 'KR', 'sector'),
    ('140710.KS', '운송', 'KR', 'sector'),
]
US_INPUTS = ['^VIX', 'SPY', 'IEF', 'HYG', 'RSP']
MARKET = {'US': '^GSPC', 'KR': '^KS11'}


def yahoo_close(symbol, tries=4):
    """Daily adjusted closes for the last ~10 years from Yahoo's chart endpoint."""
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}'
           '?range=10y&interval=1d&includeAdjustedClose=true')
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                res = json.load(r)['chart']['result'][0]
            ts = pd.to_datetime(res['timestamp'], unit='s', utc=True)
            gmt = res['meta'].get('gmtoffset', 0)
            days = (ts + pd.to_timedelta(gmt, unit='s')).tz_localize(None).normalize()
            # Adjusted close (dividends) where Yahoo has it, so ETF RSI is not hit by payout days.
            adj = (res['indicators'].get('adjclose') or [{}])[0].get('adjclose')
            s = pd.Series(adj or res['indicators']['quote'][0]['close'], index=days, dtype=float)
            # Index bars (e.g. ^KS11) can leave the latest close empty; meta has that day's price.
            meta = res['meta']
            if len(s) and np.isnan(s.iloc[-1]) and meta.get('regularMarketPrice') and meta.get('regularMarketTime'):
                day = pd.to_datetime(meta['regularMarketTime'] + gmt, unit='s').normalize()
                if day == s.index[-1]:
                    s.iloc[-1] = float(meta['regularMarketPrice'])
            s = s.dropna()
            return s[~s.index.duplicated(keep='last')]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f'{symbol} 시세 수집 실패 ({type(last).__name__})')


def load_close(symbol):
    cache = os.environ.get('TEMP_CACHE')
    if cache:
        p = Path(cache, symbol.replace('^', '_') + '.csv')
        df = pd.read_csv(p, parse_dates=['date']).set_index('date')
        return df['close'].astype(float).dropna()
    return yahoo_close(symbol)


def pct(s, inverse=False):
    """Percentile of today's value within the trailing window."""
    r = s.rolling(WINDOW, min_periods=250).rank(pct=True) * 100
    return 100 - r if inverse else r


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def range_pos(s):
    hi, lo = s.rolling(252).max(), s.rolling(252).min()
    return (s - lo) / (hi - lo) * 100


def own_sentiment(s):
    mom = pct(s / s.rolling(125).mean() - 1)
    vol = pct(np.log(s).diff().rolling(20).std(), inverse=True)
    return pd.concat([mom, range_pos(s), vol], axis=1).mean(axis=1, skipna=False)


def us_sentiment(close):
    sp = close['^GSPC']
    idx = sp.index
    get = lambda k: close[k].reindex(idx).ffill() if k in close else pd.Series(np.nan, index=idx)  # noqa: E731
    vix, spy, ief, hyg, rsp = (get(k) for k in US_INPUTS)
    parts = pd.DataFrame({
        'momentum': pct(sp / sp.rolling(125).mean() - 1),
        'range': range_pos(sp),
        'volatility': pct(vix / vix.rolling(50).mean(), inverse=True),
        'safe_haven': pct(spy.pct_change(20) - ief.pct_change(20)),
        'junk': pct(hyg.pct_change(20) - ief.pct_change(20)),
        'breadth': pct((rsp / spy).pct_change(63)),
    })
    return parts.mean(axis=1, skipna=True).where(parts.notna().sum(axis=1) >= 3)


def weekly_rsi(s):
    """RSI on Friday closes, shifted a week so the current week never counts."""
    return rsi(s.resample('W-FRI').last()).shift(1).reindex(s.index, method='ffill')


def temperature(s, market_sent, own_weight):
    mkt = market_sent.reindex(s.index).ffill()
    sent = mkt if own_weight == 0 else (1 - own_weight) * mkt + own_weight * own_sentiment(s)
    wk, dl = weekly_rsi(s), rsi(s)
    df = pd.DataFrame({'sent': sent, 'wk': wk, 'dl': dl})
    df['temp'] = 0.4 * df.sent + 0.3 * df.wk + 0.3 * df.dl
    return df.dropna()


def yearly(temp, now_year):
    """Low/high per calendar year (daily) and a weekly line over the same years.

    Each week shows its Friday value, except the week holding a year's low or
    high shows that extreme, so the marked points sit on the line.
    """
    t = temp[temp.index.year > now_year - YEARS]
    if t.empty:
        return None
    years, extremes = [], {}
    for y, g in t.groupby(t.index.year):
        lo, hi = g.idxmin(), g.idxmax()
        extremes[lo], extremes[hi] = g[lo], g[hi]
        years.append({
            'y': int(y), 'from': g.index[0].strftime('%Y-%m-%d'),
            'lo': round(float(g[lo]), 1), 'lo_d': lo.strftime('%Y-%m-%d'),
            'hi': round(float(g[hi]), 1), 'hi_d': hi.strftime('%Y-%m-%d'),
        })
    week = t.resample('W-FRI')
    line = week.last()
    for d, v in extremes.items():
        line[line.index[line.index >= d][0]] = v
    line = line.ffill()  # a week with no trading (holidays) keeps the last value; spacing stays weekly
    return {
        'start': line.index[0].strftime('%Y-%m-%d'),
        'weeks': [round(float(v), 1) for v in line],
        'years': years,
    }


def build(close, now=None):
    now = now or datetime.now(timezone.utc)
    market = {}
    if '^GSPC' in close:
        market['US'] = us_sentiment(close)
    if '^KS11' in close:
        market['KR'] = own_sentiment(close['^KS11'])
    items, errors = [], {}
    for symbol, name, mkt, kind in TARGETS:
        if symbol not in close or mkt not in market:
            errors[symbol] = 'no prices'
            continue
        df = temperature(close[symbol], market[mkt], 0 if symbol == MARKET[mkt] else 0.5)
        if df.empty:
            errors[symbol] = 'not enough history'
            continue
        last = df.iloc[-1]
        hist = df['temp'].iloc[-HISTORY_DAYS:]
        items.append({
            'symbol': symbol, 'name': name, 'market': mkt, 'kind': kind,
            'asof': df.index[-1].strftime('%Y-%m-%d'),
            'temp': round(float(last.temp), 1), 'sent': round(float(last.sent), 1),
            'wk': round(float(last.wk), 1), 'dl': round(float(last.dl), 1),
            'hist': [round(float(v), 1) for v in hist.iloc[::-2][::-1]],
            'y5': yearly(df['temp'], df.index[-1].year),
        })
    return {
        'meta': {
            'version': TEMP_VERSION,
            'generated': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'formula': 'temp = 0.4*sent + 0.3*weekly_rsi + 0.3*daily_rsi',
            'sentiment': 'price-based proxy, not CNN Fear & Greed',
            'source': 'Yahoo Finance daily adjusted close',
            'history': f'last {HISTORY_DAYS} trading days, every 2nd day',
            'yearly': f'y5: last {YEARS} calendar years, daily low/high and weekly line',
            'source_errors': errors,
        },
        'items': items,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dist/thermo_data.json')
    args = ap.parse_args(argv)
    close, errors = {}, {}
    for symbol in [t[0] for t in TARGETS] + US_INPUTS:
        try:
            close[symbol] = load_close(symbol)
        except Exception as exc:  # noqa: BLE001
            errors[symbol] = str(exc)
    data = build(close)
    data['meta']['source_errors'].update(errors)
    have = {i['symbol'] for i in data['items']}
    missing = [t[0] for t in TARGETS if t[3] == 'index' and t[0] not in have]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('trading temp', len(data['items']), 'items', '| errors:', list(data['meta']['source_errors']) or 'none')
    if missing:
        print('missing index:', missing, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
