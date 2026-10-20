import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('market_regime_check', ROOT / 'scripts' / 'market_regime_check.py')
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)


def series(values, start='2000-01-01'):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq='MS'), dtype=object)


class MarketRegimeCheckTests(unittest.TestCase):
    def test_book_order_is_a_cycle(self):
        self.assertEqual(set(mr.NEXT), set(mr.QUADRANTS))
        state = 'liquidity'
        for _ in range(4):
            state = mr.NEXT[state]
        self.assertEqual(state, 'liquidity')
        for cur, (axis, sign) in mr.FLIP.items():
            nxt = mr.NEXT[cur]
            g = {'expansion': 1, 'adjustment': 1, 'liquidity': -1, 'defense': -1}
            l = {'expansion': 1, 'liquidity': 1, 'adjustment': -1, 'defense': -1}
            changed = 'G' if g[cur] != g[nxt] else 'L'
            self.assertEqual(axis, changed)
            self.assertEqual(sign, (g if axis == 'G' else l)[nxt])

    def test_cross_correlation_finds_a_market_lead(self):
        index = pd.date_range('1990-01-01', periods=300, freq='MS')
        rng = np.random.default_rng(0)
        market = pd.Series(rng.standard_normal(300), index=index).rolling(3, min_periods=1).mean()
        model = market.shift(5)  # the model repeats the market five months later
        self.assertEqual(mr.cross_correlation(market, model)['best'], 5)

    def test_spells_and_entries(self):
        s = series(['a', 'a', None, 'b', 'b', 'a'])
        runs = mr.spells(s)
        self.assertEqual([(r[0], r[3]) for r in runs], [('a', 2), ('b', 2), ('a', 1)])
        e = mr.entries(s)
        self.assertEqual([(x[1], x[2]) for x in e], [('a', None), ('b', 'a'), ('a', 'b')])

    def test_transition_lead_counts_months_the_weather_was_early(self):
        season = series(['defense'] * 10 + ['liquidity'] * 10)
        weather = series(['defense'] * 7 + ['liquidity'] * 13)
        summary, rows = mr.transition_leads(season, weather)
        self.assertEqual(rows[-1]['lead'], 3)
        self.assertEqual(summary['share_market_first'], 1.0)

    def test_false_spring_counts_returns_within_the_window(self):
        s = series(['defense'] * 5 + ['liquidity'] * 2 + ['defense'] * 10 + ['liquidity'] * 10)
        total, pairs = mr.false_springs(s, horizons=(3,))
        self.assertEqual(pairs['defense>liquidity']['n'], 2)
        self.assertEqual(pairs['defense>liquidity']['back_3'], 1)
        self.assertEqual(pairs['liquidity>defense']['back_3'], 0)

    def test_transition_stats_split_book_order_and_reverse(self):
        s = series(['liquidity', 'expansion', 'expansion', 'liquidity', 'defense'])
        stats = mr.transition_stats(s)
        self.assertEqual(stats['changes'], 3)
        self.assertEqual(stats['book_order'], 1)   # liquidity -> expansion
        self.assertEqual(stats['reverse'], 2)      # expansion -> liquidity, liquidity -> defense

    def test_durations_leave_the_open_spell_out(self):
        s = series(['defense'] * 4 + ['liquidity'] * 2 + ['defense'] * 9)
        table, open_run = mr.duration_table(s)
        self.assertEqual(table['defense']['n'], 1)
        self.assertEqual(table['defense']['median'], 4.0)
        self.assertEqual(open_run['months'], 9)

    def test_positions_use_two_month_season_lag_and_one_month_weather_lag(self):
        index = pd.date_range('2020-01-01', periods=4, freq='MS')
        season_w = pd.Series([.5, .7, .7, .7], index=index)
        weather_w = pd.Series([.6, .6, .7, .7], index=index)
        pos = mr.combine_positions(season_w, weather_w, index, tilt=.5)
        combo = pos['계절 + 날씨 (차이의 50%만 이동)']
        self.assertTrue(np.isnan(combo.iloc[1]))
        self.assertAlmostEqual(combo.iloc[2], .5 + .5 * (.6 - .5))
        self.assertAlmostEqual(combo.iloc[3], .7)

    @unittest.skipUnless(os.environ.get('GL_BIAS_CACHE'), 'set GL_BIAS_CACHE to a folder of downloaded inputs')
    def test_full_run(self):
        result = mr.run(os.environ['GL_BIAS_CACHE'], pd.Timestamp.today().to_period('M').to_timestamp().strftime('%Y-%m-%d'))
        self.assertEqual(set(result['specs']), set(mr.SPECS))
        self.assertEqual(result['season']['transitions']['changes'],
                         result['season']['transitions']['book_order'] + result['season']['transitions']['reverse']
                         + result['season']['transitions']['diagonal'])


if __name__ == '__main__':
    unittest.main()
