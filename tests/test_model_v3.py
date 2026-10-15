import unittest
import numpy as np
import pandas as pd
import model_v2 as v2
import model_v3 as v3


class ModelV3Tests(unittest.TestCase):
    def test_default_composite_still_double_standardizes(self):
        dates = pd.date_range('1980-01-01', '2020-01-01', freq='MS')
        series = pd.Series(np.sin(np.arange(len(dates)) / 9) + np.arange(len(dates)) / 800, index=dates)
        result = v2.composite({'A': {'t': series, 'w': 1}})
        values, seen, _ = v2.tail_signal(series, result.score.index)
        raw = v2.observed_z(values, seen)
        second = v2.observed_z(raw, raw.notna())
        expected = second.ewm(span=3, min_periods=1).mean().where(second.notna())
        pd.testing.assert_series_equal(result.score, expected, check_names=False)

    def test_theoretical_scale_respects_correlation(self):
        index = pd.date_range('2000-01-01', periods=3, freq='MS')
        valid = pd.DataFrame({'A': True, 'B': True}, index=index)
        weights = pd.Series({'A': .5, 'B': .5})
        independent = v2.theoretical_scale(valid, weights, 0.0)
        perfect = v2.theoretical_scale(valid, weights, 1.0)
        self.assertAlmostEqual(float(independent.iloc[-1]), np.sqrt(0.5))
        self.assertAlmostEqual(float(perfect.iloc[-1]), 1.0)

    def test_robust_z_ignores_a_single_spike_in_the_scale(self):
        index = pd.date_range('2000-01-01', periods=80, freq='MS')
        calm = pd.Series(np.sin(np.arange(80) / 6), index=index)
        spiked = calm.copy()
        spiked.iloc[40] = 50
        classical = v2.observed_z(spiked, spiked.notna(), method='classical')
        robust = v2.observed_z(spiked, spiked.notna(), method='robust')
        # The spike inflates the classical scale, so a later calm point looks closer to zero.
        self.assertLess(abs(classical.iloc[-1]), abs(robust.iloc[-1]) + 1e-9)
        self.assertGreater(abs(float(robust.iloc[-1])), 0.05)

    def test_carry_limit_can_be_two_months(self):
        dates = pd.date_range('2000-01-01', '2026-08-01', freq='MS')
        series = pd.Series(np.arange(len(dates)), index=dates, dtype=float)
        index = pd.date_range(dates[0], '2026-12-01', freq='MS')
        _, _, carry = v2.tail_signal(series, index, limit=2)
        self.assertEqual('2026-08', carry.loc['2026-10-01'])
        self.assertEqual('', carry.loc['2026-11-01'])

    def test_neutral_month_does_not_hand_its_quadrant_to_the_next_month(self):
        # v2 updated `previous` on the neutral month, so the following month could
        # confirm against a quadrant that was never confirmed.
        index = pd.date_range('2020-01-01', periods=5, freq='MS')
        frame = pd.DataFrame({
            'G': [0.80, 0.05, 0.20, 0.30, 0.40],
            'L': [0.80, 0.05, 0.40, 0.40, 0.40],
            'gr': [0.80, 0.05, 0.20, 0.30, 0.40],
            'lr': [0.80, 0.05, 0.40, 0.40, 0.40],
            'gp': [90, 50, 60, 70, 70],
            'lp': [90, 50, 70, 70, 70],
            'owg': [1, 1, 1, 1, 1],
            'owl': [1, 1, 1, 1, 1],
        }, index=index)
        flags, _ = v3.classify(frame)
        self.assertTrue(flags[0]['c'])
        self.assertTrue(flags[1]['n'])
        self.assertIsNone(flags[1]['regime'])
        self.assertFalse(flags[2]['c'])
        self.assertIsNone(flags[2]['regime'])
        self.assertEqual('기본 비중 유지', flags[1]['allocation'])
        self.assertEqual('기본 비중 유지', flags[2]['allocation'])
        # One expansion month is not enough; the second consecutive month confirms.
        self.assertEqual('expansion', flags[3]['regime'])
        self.assertFalse(flags[3]['c'])
        self.assertEqual('expansion', flags[4]['regime'])
        self.assertTrue(flags[4]['c'])

    def test_hysteresis_holds_inside_the_exit_band_and_resets_after_neutral(self):
        self.assertEqual(1, v3.step_axis(0.30, 50, 0))
        self.assertEqual(1, v3.step_axis(0.12, 50, 1))
        self.assertEqual(0, v3.step_axis(0.05, 50, 1))
        self.assertEqual(-1, v3.step_axis(-0.30, 50, 0))
        self.assertEqual(1, v3.step_axis(0.05, 85, 0))

    def test_early_recovery_needs_negative_growth_and_an_improving_direction(self):
        self.assertEqual('early_recovery', v3.phase_of(-1, 1, 0.20, 0.0, 'liquidity'))
        self.assertEqual('defense', v3.phase_of(-1, -1, -0.20, -0.10, 'defense'))
        self.assertEqual('early_recovery', v3.phase_of(-1, -1, 0.20, 0.10, 'defense'))
        self.assertEqual('neutral', v3.phase_of(0, 0, 0.20, 0.10, None))

    def test_favor_lists_do_not_share_staples_or_gold(self):
        self.assertTrue(v3.favor_sets_disjoint())
        defense = v3.REGIME_ASSETS['defense']['favor']
        liquidity = v3.REGIME_ASSETS['liquidity']['favor']
        adjustment = v3.REGIME_ASSETS['adjustment']['favor']
        self.assertIn('금', defense)
        self.assertNotIn('금', liquidity)
        self.assertIn('필수소비재', defense)
        self.assertNotIn('필수소비재', adjustment)

    def test_quadrant_helper_is_shared(self):
        self.assertIs(v3.quadrant, v2.quadrant)
        import build
        self.assertIs(build.quadrant, v2.quadrant)


if __name__ == '__main__':
    unittest.main()
