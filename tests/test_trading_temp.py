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
        # 코스피는 국내 시장 심리 그대로, 섹터는 시장 반 + 자체 반
        kr = temp.own_sentiment(close['^KS11'])
        self.assertAlmostEqual(items['^KS11']['sent'], round(float(kr.iloc[-1]), 1))
        own = temp.own_sentiment(close['091160.KS'])
        mixed = 0.5 * kr.iloc[-1] + 0.5 * own.iloc[-1]
        self.assertAlmostEqual(items['091160.KS']['sent'], round(float(mixed), 1))

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
