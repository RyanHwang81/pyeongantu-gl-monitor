import io
import unittest
import zipfile

import numpy as np
import pandas as pd

import market_weather as mw

SAMPLE = """This file was created using the 202608 CRSP database.

  Average Value Weighted Returns -- Monthly
,NoDur,Durbl,Manuf,Enrgy,HiTec,Telcm,Shops,Hlth,Utils,Other
202607,   1.00,   2.00,   3.00,   0.00,   0.00,   0.00,   4.00,  -1.00,  -2.00,   0.00
202608,   0.50,  -99.99,   1.00,   0.00,   0.00,   0.00,   1.00,   0.50,   0.50,   0.00

  Average Equal Weighted Returns -- Monthly
,NoDur,Durbl,Manuf,Enrgy,HiTec,Telcm,Shops,Hlth,Utils,Other
202607,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00,   9.00
"""


def synthetic_inputs(months=480):
    index = pd.date_range('1985-01-01', periods=months, freq='MS')
    t = np.arange(months)
    rng = np.random.default_rng(2)
    wave = np.sin(t / 14)
    monthly = {
        'GS10': pd.Series(5 + wave + rng.normal(0, .1, months), index=index),
        'TB3MS': pd.Series(4 + np.cos(t / 14) + rng.normal(0, .1, months), index=index),
        'DGS2': pd.Series(4.5 + np.sin(t / 11) + rng.normal(0, .1, months), index=index),
        'BAA10YM': pd.Series(2 + .5 * np.cos(t / 17) + rng.normal(0, .05, months), index=index),
    }
    cols = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']
    ind = pd.DataFrame(rng.normal(.01, .04, (months, len(cols))), index=index, columns=cols)
    ind[['Durbl', 'Manuf', 'Shops']] += (.01 * wave)[:, None]
    return monthly, ind, pd.date_range(index[0], index[-1] + pd.DateOffset(months=1), freq='MS')


class MarketWeatherTests(unittest.TestCase):
    def test_parser_reads_the_value_weighted_block_only(self):
        frame = mw.parse_french(SAMPLE)
        self.assertEqual(list(frame.index.strftime('%Y-%m')), ['2026-07', '2026-08'])
        self.assertAlmostEqual(frame.loc['2026-07-01', 'Manuf'], 0.03)
        self.assertTrue(np.isnan(frame.loc['2026-08-01', 'Durbl']))

    def test_zip_member_is_read(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('10_Industry_Portfolios.csv', SAMPLE)
        self.assertIn('Average Value Weighted', mw.industries_from_zip(buffer.getvalue()))

    def test_components_have_equal_weights_and_signs(self):
        monthly, ind, _ = synthetic_inputs(60)
        g, l = mw.components(monthly, ind)
        self.assertEqual(set(g), {'CYCDEF', 'CURVE'})
        self.assertEqual(set(l), {'DGS2', 'BAA'})
        self.assertTrue(all(c['w'] == .5 for c in (*g.values(), *l.values())))
        # A rise in the 2-year yield is a liquidity negative.
        d = l['DGS2']['t']
        expected = -(monthly['DGS2'] - monthly['DGS2'].shift(3)).dropna()
        pd.testing.assert_series_equal(d, expected, check_names=False)

    def test_build_returns_rows_before_the_current_month(self):
        monthly, ind, index = synthetic_inputs()
        out = mw.build(monthly, ind, index)
        self.assertIsNotNone(out)
        self.assertEqual(out['months'][-1]['d'], index[-2].strftime('%Y-%m'))
        self.assertEqual(out['latest']['d'], out['months'][-1]['d'])
        self.assertTrue({r['w'] for r in out['months']} <= {'expansion', 'liquidity', 'adjustment', 'defense', None})
        self.assertGreater(len({r['w'] for r in out['months']}), 2)


if __name__ == '__main__':
    unittest.main()
