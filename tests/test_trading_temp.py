import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_trading_temp', ROOT / 'scripts' / 'build_trading_temp.py')
temp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temp)


def walk(seed, n=1100, drift=0.0):
    """Random walk closes. Shape only; the numbers mean nothing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end='2026-10-27', periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=idx)


def synthetic():
    symbols = [t[0] for t in temp.TARGETS] + temp.US_INPUTS
    return {s: walk(i) for i, s in enumerate(symbols)}


class TradingTempTest(unittest.TestCase):
    def test_rsi_extremes(self):
        up = pd.Series(np.arange(1, 60, dtype=float))
        self.assertGreater(temp.rsi(up).iloc[-1], 99)
        self.assertLess(temp.rsi(up[::-1].reset_index(drop=True)).iloc[-1], 1)

    def test_weekly_rsi_ignores_current_week(self):
        s = walk(1)
        base = temp.weekly_rsi(s)
        bumped = s.copy()
        bumped.iloc[-1] *= 1.5  # 2026-10-27 is a Tuesday: still the current week
        self.assertAlmostEqual(temp.weekly_rsi(bumped).iloc[-1], base.iloc[-1])

    def test_percentile_uses_no_future(self):
        s = walk(2)
        early = temp.pct(s.iloc[:800]).iloc[-1]
        self.assertAlmostEqual(temp.pct(s).iloc[799], early)

    def test_build_shape_and_mix(self):
        close = synthetic()
        data = temp.build(close)
        items = {i['symbol']: i for i in data['items']}
        self.assertEqual(len(items), len(temp.TARGETS))
        self.assertEqual(data['meta']['source_errors'], {})
        for i in items.values():
            for k in ('temp', 'sent', 'wk', 'dl'):
                self.assertTrue(0 <= i[k] <= 100, (i['symbol'], k, i[k]))
            self.assertAlmostEqual(i['temp'], 0.4 * i['sent'] + 0.3 * i['wk'] + 0.3 * i['dl'], delta=0.15)
            self.assertEqual(len(i['hist']), temp.HISTORY_DAYS // 2)
            self.assertEqual(i['hist'][-1], i['temp'])
            self.assertTrue(i['y5']['years'])
            self.assertEqual(i['y5']['years'][-1]['y'], 2026)
        # 코스피는 국내 시장 심리 그대로, 섹터는 시장 반 + 자체 반
        kr = temp.own_sentiment(close['^KS11'])
        self.assertAlmostEqual(items['^KS11']['sent'], round(float(kr.iloc[-1]), 1))
        own = temp.own_sentiment(close['091160.KS'])
        mixed = 0.5 * kr.iloc[-1] + 0.5 * own.iloc[-1]
        self.assertAlmostEqual(items['091160.KS']['sent'], round(float(mixed), 1))

    def test_yearly_low_high_on_weekly_price(self):
        s = walk(3, n=2000)
        t = temp.temperature(s, temp.own_sentiment(s), 0)['temp']
        y5 = temp.yearly(t, s, 2026)
        self.assertEqual([r['y'] for r in y5['years']], [2022, 2023, 2024, 2025, 2026])
        start = pd.Timestamp(y5['start'])
        self.assertEqual(start.dayofweek, 4)  # Friday
        for r in y5['years']:
            g = t[t.index.year == r['y']]
            self.assertAlmostEqual(r['lo'], round(float(g.min()), 1))
            self.assertAlmostEqual(r['hi'], round(float(g.max()), 1))
            self.assertEqual(r['lo_d'], g.idxmin().strftime('%Y-%m-%d'))
            self.assertAlmostEqual(r['lo_p'], s[g.idxmin()], delta=s[g.idxmin()] * 1e-4)
            # the low day's close sits on the weekly price line, in the week that holds it
            week = (pd.Timestamp(r['lo_d']) - start).days + 6
            self.assertEqual(y5['price'][week // 7], r['lo_p'])
        last_friday = t.index[-1] + pd.offsets.Week(weekday=4) * 0
        self.assertEqual(len(y5['price']), (last_friday - start).days // 7 + 1)
        self.assertAlmostEqual(y5['price'][-1], s.iloc[-1], delta=s.iloc[-1] * 1e-4)

    def test_missing_symbol_is_reported_not_fatal(self):
        close = synthetic()
        del close['XLK']
        data = temp.build(close)
        self.assertNotIn('XLK', {i['symbol'] for i in data['items']})
        self.assertIn('XLK', data['meta']['source_errors'])

    def test_main_offline_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            for sym, s in synthetic().items():
                pd.DataFrame({'date': s.index.strftime('%Y-%m-%d'), 'close': s.values}).to_csv(
                    Path(folder, sym.replace('^', '_') + '.csv'), index=False)
            out = Path(folder, 'thermo_data.json')
            os.environ['TEMP_CACHE'] = folder
            try:
                self.assertEqual(temp.main(['--out', str(out)]), 0)
            finally:
                del os.environ['TEMP_CACHE']
            data = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(data['meta']['version'], temp.TEMP_VERSION)
            self.assertEqual(data['items'][0]['asof'], '2026-10-27')


if __name__ == '__main__':
    unittest.main()
