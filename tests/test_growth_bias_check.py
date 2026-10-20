import importlib.util
import os
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('growth_bias_check', ROOT / 'scripts' / 'growth_bias_check.py')
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)


class GrowthBiasCheckTests(unittest.TestCase):
    def test_base_weights_match_build_growth_components(self):
        text = (ROOT / 'build.py').read_text(encoding='utf-8')
        block = text[text.index('G_COMP={'):text.index('}', text.index("'ICSA':{'t'")) + 1]
        weights = {k: float(w) for k, w in re.findall(r"'(\w+)':\{'t':[^\n]*?'w':(\.\d+)", block)}
        self.assertEqual(weights, gb.BASE_WEIGHTS)

    def test_agreement_skips_zero_signs_and_reports_base_rates(self):
        index = pd.date_range('2000-01-01', periods=6, freq='MS')
        model = pd.Series([1, 1, -1, -1, 1, 0], index=index, dtype=float)
        other = pd.Series([1, -1, -1, 0, 1, 1], index=index, dtype=float)
        row = gb.agreement(model, other)
        self.assertEqual(row['n'], 4)
        self.assertAlmostEqual(row['agree'], 0.75)
        self.assertAlmostEqual(row['other_pos'], 0.5)

    def test_best_shift_is_positive_when_the_model_leads(self):
        index = pd.date_range('1990-01-01', periods=240, freq='MS')
        rng = np.random.default_rng(3)
        model = pd.Series(np.sign(rng.standard_normal(240)), index=index)
        other = model.shift(3)  # the comparator repeats the model three months later
        best = gb.best_shift(model, other)
        self.assertEqual(best['shift'], 3)
        self.assertEqual(best['agree'], 1.0)

    def test_quarterly_values_cover_their_quarter_only(self):
        q = pd.Series([1.0, 2.0], index=pd.to_datetime(['2025-01-01', '2025-04-01']))
        spread = gb.spread_quarterly(q)
        self.assertEqual(list(spread.index.strftime('%Y-%m')),
                         ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06'])
        self.assertEqual(list(spread), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def test_six_month_annualized_rate(self):
        index = pd.date_range('2000-01-01', periods=13, freq='MS')
        level = pd.Series(1.01 ** np.arange(13), index=index)
        rate = gb.annualized(level, 6)
        self.assertAlmostEqual(float(rate.iloc[-1]), (1.01 ** 12 - 1) * 100, places=8)
        self.assertTrue(rate.iloc[:6].isna().all())

    def test_mirror_scrape_errors_are_dropped(self):
        index = pd.date_range('2025-06-01', periods=4, freq='MS')
        clean, dropped = gb.clean_pmi(pd.Series([49.0, 48.7, 11.1, 10.0], index=index))
        self.assertEqual(list(clean), [49.0, 48.7])
        self.assertEqual(list(dropped.index.strftime('%Y-%m')), ['2025-08', '2025-09'])

    def test_variant_contexts_restore_model_globals(self):
        window, observed_z, scale = gb.v2.WINDOW, gb.v2.observed_z, gb.v2._robust_center_scale
        with gb.window(120):
            self.assertEqual(gb.v2.WINDOW, 120)
        with gb.stats_gap(*gb.COVID_STATS_GAP):
            self.assertIsNot(gb.v2.observed_z, observed_z)
        with gb.mad_floor(0.5):
            self.assertIsNot(gb.v2._robust_center_scale, scale)
        self.assertEqual(gb.v2.WINDOW, window)
        self.assertIs(gb.v2.observed_z, observed_z)
        self.assertIs(gb.v2._robust_center_scale, scale)

    def test_stats_gap_leaves_the_gap_out_of_the_median(self):
        index = pd.date_range('2010-01-01', '2024-12-01', freq='MS')
        values = pd.Series(0.0, index=index)
        values.loc['2020-03-01':'2022-12-01'] = 10.0
        values += np.sin(np.arange(len(index))) * 0.1
        seen = values.notna()
        with gb.stats_gap(*gb.COVID_STATS_GAP):
            z = gb.v2.observed_z(values, seen, method='robust')
        # A gap month scored against calm samples only is far above zero.
        self.assertGreater(float(z.loc['2022-06-01']), 2.9)

    def test_negative_runs_count_recession_months(self):
        index = pd.date_range('2000-01-01', periods=8, freq='MS')
        g = pd.Series([1, -1, -1, -1, 1, -1, -1, 1], index=index, dtype=float)
        rec = pd.Series([0, 0, 1, 1, 0, 0, 0, 0], index=index)
        runs = gb.negative_runs(g, rec)
        self.assertEqual(runs[0], {'from': '2000-02', 'to': '2000-04', 'months': 3, 'recession_months': 2})
        self.assertEqual(runs[1]['months'], 2)

    @unittest.skipUnless(os.environ.get('GL_BIAS_CACHE'), 'set GL_BIAS_CACHE to a folder of downloaded inputs')
    def test_replication_matches_published_scores(self):
        result = gb.run(os.environ['GL_BIAS_CACHE'], os.environ.get('GL_BIAS_END', '2026-10-01'))
        self.assertLess(result['replication']['max_abs_dg'], 0.02)
        self.assertLess(result['replication']['max_abs_dl'], 0.02)
        self.assertEqual(result['replication']['regime_mismatch'], [])


if __name__ == '__main__':
    unittest.main()
