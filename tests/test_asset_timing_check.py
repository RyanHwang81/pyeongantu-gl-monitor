import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('asset_timing_check', ROOT / 'scripts' / 'asset_timing_check.py')
at = importlib.util.module_from_spec(spec)
spec.loader.exec_module(at)


def frame(values, cols=None, start='2000-01-01'):
    cols = cols or list(at.ASSET_KO)
    index = pd.date_range(start, periods=len(values), freq='MS')
    return pd.DataFrame({c: values for c in cols}, index=index, dtype=float)


class AssetTimingTests(unittest.TestCase):
    def test_every_listed_asset_is_measured(self):
        for q, spec_ in at.LISTS.items():
            for side in ('favor', 'avoid'):
                for asset in spec_[side]:
                    self.assertIn(asset, at.ASSET_KO, f'{q} {side} {asset}')
        self.assertEqual(set(at.LISTS), set(at.QUADRANTS))

    def test_forward_window_offsets(self):
        returns = frame([0.0] * 10)
        returns.loc['2000-04-01', 'Manuf'] = 0.10
        # Season lag 1, k 0, h 1: the January signal uses March. lag 1, k 1: April.
        self.assertAlmostEqual(at.forward(returns, 1, 1 + 0 + 1).loc['2000-02-01', 'Manuf'], 0.10)
        self.assertAlmostEqual(at.forward(returns, 1, 1 + 1 + 1).loc['2000-01-01', 'Manuf'], 0.10)
        # Same-month baseline: lag −1, h 1 → the label month itself.
        self.assertAlmostEqual(at.forward(returns, 1, 0).loc['2000-04-01', 'Manuf'], 0.10)

    def test_list_score_is_favored_minus_avoided(self):
        returns = frame([0.0] * 3)
        for a in at.LISTS['defense']['favor']:
            returns[a] = 0.02
        for a in at.LISTS['defense']['avoid']:
            returns[a] = -0.01
        labels = pd.Series(['defense', 'expansion', None], index=returns.index, dtype=object)
        s = at.list_score(returns, labels)
        self.assertAlmostEqual(s.iloc[0], 0.03)
        self.assertTrue(np.isnan(s.iloc[2]))
        nxt = at.list_score(returns, labels, mapping=lambda q: at.NEXT[q])
        # Defense months scored with the liquidity list.
        liq = at.LISTS['liquidity']
        expected = returns.iloc[0][liq['favor']].mean() - returns.iloc[0][liq['avoid']].mean()
        self.assertAlmostEqual(nxt.iloc[0], expected)

    def test_training_windows_end_by_2005(self):
        index = pd.date_range('2004-01-01', '2006-12-01', freq='MS')
        train, hold = at.masks(index, 7)
        self.assertTrue(train.loc['2005-05-01'])
        self.assertFalse(train.loc['2005-06-01'])
        self.assertTrue(hold.loc['2006-01-01'])
        train0, _ = at.masks(index, 0)
        self.assertTrue(train0.loc['2005-12-01'])

    def test_pass_rule(self):
        ok = {'train': {'mean': 1.0, 't': 2.1, 'n': 100}, 'holdout': {'mean': 0.5, 't': 1.7, 'n': 50}}
        weak = {'train': {'mean': 1.0, 't': 1.9, 'n': 100}, 'holdout': {'mean': 0.5, 't': 1.7, 'n': 50}}
        self.assertTrue(at.passes(ok))
        self.assertFalse(at.passes(weak))


if __name__ == '__main__':
    unittest.main()
