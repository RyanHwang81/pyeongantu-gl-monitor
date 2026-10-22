import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v4_holdout', ROOT / 'scripts' / 'v4_holdout.py')
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)


class V4HoldoutTests(unittest.TestCase):
    def test_training_windows_never_reach_the_holdout(self):
        index = pd.date_range('2003-01-01', '2007-01-01', freq='MS')
        lag1 = v4.train_mask(index, 1)
        self.assertTrue(lag1.loc['2004-11-01'])
        self.assertFalse(lag1.loc['2004-12-01'])
        lag0 = v4.train_mask(index, 0)
        self.assertTrue(lag0.loc['2004-12-01'])
        self.assertFalse(lag0.loc['2005-01-01'])
        self.assertTrue(v4.holdout_mask(index).loc['2006-01-01'])
        self.assertFalse(v4.holdout_mask(index).loc['2005-12-01'])

    def test_c0_guidance_matches_the_dashboard_lists(self):
        assets = v4.mr.rr.gb.v3.REGIME_ASSETS
        self.assertIn('산업재', assets['expansion']['favor'])
        self.assertIn('성장주', assets['liquidity']['favor'])
        self.assertIn('가치주', assets['adjustment']['favor'])
        self.assertIn('필수소비재', assets['defense']['favor'])
        self.assertEqual(v4.C0_GUIDANCE['expansion']['cyc_def'], 1)
        self.assertEqual(v4.C0_GUIDANCE['liquidity']['growth_value'], 1)
        self.assertEqual(v4.C0_GUIDANCE['adjustment']['growth_value'], -1)
        self.assertEqual(v4.C0_GUIDANCE['defense']['cyc_def'], -1)

    def test_guidance_uses_the_threshold(self):
        index = pd.date_range('1990-01-01', periods=8, freq='MS')
        labels = pd.Series(['expansion'] * 4 + ['defense'] * 4, index=index)
        fwd = pd.DataFrame({'growth_value': [0.05] * 4 + [0.005] * 4,
                            'cyc_def': [-0.03] * 4 + [-0.02] * 4}, index=index)
        guide, means = v4.derive_guidance(fwd, labels, pd.Series(True, index=index))
        self.assertEqual(guide['expansion'], {'growth_value': 1, 'cyc_def': -1})
        self.assertEqual(guide['defense'], {'growth_value': 0, 'cyc_def': -1})
        self.assertEqual(guide['liquidity'], {'growth_value': 0, 'cyc_def': 0})
        self.assertAlmostEqual(means['expansion']['growth_value'], 5.0)

    def test_follow_returns_average_only_stated_directions(self):
        index = pd.date_range('2000-01-01', periods=2, freq='MS')
        labels = pd.Series(['expansion', 'unconfirmed'], index=index)
        fwd = pd.DataFrame({'growth_value': [0.04, 0.10], 'cyc_def': [-0.02, 0.10]}, index=index)
        guide = {'expansion': {'growth_value': 1, 'cyc_def': 0}}
        follow = v4.follow_returns(fwd, labels, guide)
        self.assertAlmostEqual(follow.iloc[0], 0.04)
        self.assertTrue(np.isnan(follow.iloc[1]))

    def test_persistent_signal_passes_and_noise_fails(self):
        index = pd.date_range('1972-01-01', '2026-01-01', freq='MS')
        rng = np.random.default_rng(4)
        labels = pd.Series(np.where((np.arange(len(index)) // 18) % 2 == 0, 'expansion', 'defense'), index=index)
        signal = np.where(labels == 'expansion', 0.04, -0.04)
        good = pd.DataFrame({'growth_value': signal + rng.normal(0, 0.01, len(index)),
                             'cyc_def': signal + rng.normal(0, 0.01, len(index))}, index=index)
        noise = pd.DataFrame({'growth_value': rng.normal(0, 0.05, len(index)),
                              'cyc_def': rng.normal(0, 0.05, len(index))}, index=index)
        train, hold = v4.train_mask(index, 1), v4.holdout_mask(index)
        self.assertTrue(v4.guidance_test(good, labels, train, hold)['pass'])
        self.assertFalse(v4.guidance_test(noise, labels, train, hold)['pass'])

    def test_vintage_dates_cover_each_half_year(self):
        dates = v4.vintage_dates()
        self.assertEqual(dates[0], '2006-06-30')
        self.assertEqual(dates[-1], '2026-06-30')
        self.assertEqual(len(dates), 41)


if __name__ == '__main__':
    unittest.main()
