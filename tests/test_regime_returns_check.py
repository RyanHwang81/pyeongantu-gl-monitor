import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('regime_returns_check', ROOT / 'scripts' / 'regime_returns_check.py')
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

FRENCH_SAMPLE = """This file was created using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RF
202601,   1.00,  -0.50,   2.00,   0.30
202602,  -2.00,   0.10, -99.99,   0.30

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
  2025,  10.00,   1.00,   1.00,   4.00
"""


class RegimeReturnsCheckTests(unittest.TestCase):
    def test_french_parser_reads_only_the_monthly_block(self):
        frame = rr.parse_french(FRENCH_SAMPLE)
        self.assertEqual(list(frame.index.strftime('%Y-%m')), ['2026-01', '2026-02'])
        self.assertAlmostEqual(frame.loc['2026-01-01', 'Mkt-RF'], 0.01)
        self.assertTrue(np.isnan(frame.loc['2026-02-01', 'HML']))

    def test_par_bond_at_a_constant_yield_earns_its_coupon(self):
        index = pd.date_range('2000-01-01', periods=4, freq='MS')
        r = rr.par_bond_returns(pd.Series(6.0, index=index), 10)
        self.assertAlmostEqual(float(r.iloc[-1]), 0.06 / 12, places=3)
        rising = rr.par_bond_returns(pd.Series([6.0, 7.0], index=index[:2]), 10)
        self.assertLess(float(rising.iloc[-1]), -0.05)

    def test_forward_window_skips_the_publication_month(self):
        index = pd.date_range('2000-01-01', periods=8, freq='MS')
        r = pd.DataFrame({c: 0.0 for c in ('stock', 'bond10', 'bond5', 'cash', 'growth_value', 'rmw', 'mom', 'cyc_def')},
                         index=index)
        r.loc['2000-03-01', 'stock'] = 0.10
        lag1 = rr.forward(r, 1, 1)
        lag0 = rr.forward(r, 1, 0)
        # The March return belongs to the January signal with a one-month publication lag.
        self.assertAlmostEqual(lag1.loc['2000-01-01', 'stock'], 0.10)
        self.assertAlmostEqual(lag0.loc['2000-02-01', 'stock'], 0.10)
        self.assertAlmostEqual(lag1.loc['2000-02-01', 'stock'], 0.0)

    def test_chi_square_tail_matches_known_critical_values(self):
        self.assertAlmostEqual(rr.chi2_sf(5.991, 2), 0.05, places=3)
        self.assertAlmostEqual(rr.chi2_sf(7.815, 3), 0.05, places=3)
        self.assertAlmostEqual(rr.chi2_sf(9.488, 4), 0.05, places=3)
        self.assertAlmostEqual(rr.chi2_sf(3.841, 1), 0.05, places=3)

    def test_newey_west_without_lags_gives_group_means(self):
        y = np.array([1.0, 2.0, 3.0, 10.0, 12.0])
        labels = np.array(['a', 'a', 'a', 'b', 'b'])
        b, cov, n = rr.newey_west_means(y, labels, ['a', 'b'], 0)
        self.assertEqual(n, 5)
        np.testing.assert_allclose(b, [2.0, 11.0])
        # White variance of a group mean: sum of squared residuals / n_g^2.
        self.assertAlmostEqual(cov[0, 0], 2.0 / 9.0)

    def test_confirmed_weights_hold_through_unconfirmed_months(self):
        index = pd.date_range('2020-01-01', periods=4, freq='MS')
        frame = pd.DataFrame({'G': [-1, -1, 1, 1], 'L': [-1, -1, 1, 1],
                              'regime': ['defense', 'defense', 'expansion', 'expansion'],
                              'c': [False, True, False, True]}, index=index)
        w = rr.target_weights(frame, 'confirmed')
        self.assertEqual(list(w), [rr.START_WEIGHT, .50, .50, .70])
        frame.loc['2020-03-01', 'regime'] = None
        w_all = rr.target_weights(frame, 'all')
        self.assertEqual(w_all.loc['2020-03-01'], .70)  # sign quadrant fills a borderline month

    def test_strategy_applies_weights_after_the_lag(self):
        index = pd.date_range('2020-01-01', periods=4, freq='MS')
        returns = pd.DataFrame({'stock': [0.1] * 4, 'bond10': [0.0] * 4, 'cash': [0.0] * 4}, index=index)
        weights = pd.Series([.5, .7, .7, .7], index=index)
        port, position = rr.strategy_returns(returns, weights, lag=1)
        self.assertTrue(np.isnan(port.iloc[0]) and np.isnan(port.iloc[1]))
        self.assertAlmostEqual(port.iloc[2], 0.05)  # January's weight sets March
        self.assertAlmostEqual(port.iloc[3], 0.07)

    def test_non_overlapping_sample_steps_through_calendar_months(self):
        index = pd.date_range('2000-01-01', periods=12, freq='MS')
        y = pd.Series(np.arange(12, dtype=float), index=index)
        labels = pd.Series(['a', 'b'] * 6, index=index)
        means = rr.non_overlapping(y, labels, ['a', 'b'], 3)
        # Months 0, 3, 6, 9 -> labels a, b, a, b.
        self.assertEqual(means['a'], (3.0, 2))
        self.assertEqual(means['b'], (6.0, 2))

    def test_extra_labels_stay_in_the_sequence_but_not_in_the_test(self):
        index = pd.date_range('2000-01-01', periods=120, freq='MS')
        rng = np.random.default_rng(1)
        labels = pd.Series(np.where(np.arange(120) % 3 == 0, 'unconfirmed',
                                    np.where(np.arange(120) % 2 == 0, 'a', 'b')), index=index)
        y = pd.Series(rng.standard_normal(120) * 0.01, index=index)
        stats = rr.regime_stats(y, labels, ('a', 'b'), 1, boot=False, extra=('unconfirmed',))
        self.assertEqual(set(stats['groups']), {'a', 'b'})
        self.assertEqual(stats['n'], int(labels.isin(['a', 'b']).sum()))

    @unittest.skipUnless(os.environ.get('GL_BIAS_CACHE'), 'set GL_BIAS_CACHE to a folder of downloaded inputs')
    def test_full_run_produces_tables_and_backtests(self):
        result = rr.run(os.environ['GL_BIAS_CACHE'], pd.Timestamp.today().to_period('M').to_timestamp().strftime('%Y-%m-%d'),
                        1, variants=())
        self.assertIn('stock_cash/12/confirmed/published', result['tables'])
        self.assertIn('v3 발행 · 확인만', result['backtest']['1990–2019'])


if __name__ == '__main__':
    unittest.main()
