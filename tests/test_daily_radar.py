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
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('build_daily_radar', ROOT / 'scripts' / 'build_daily_radar.py')
radar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(radar)
bt = radar.bt


def synthetic_cache(folder, end):
    """Random walks for every FRED input. Shape only; the numbers mean nothing."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range('2003-01-01', end)
    for sid in bt.FETCH:
        if sid in ('IC4WSA', 'CCSA', 'WEI', 'NFCI', 'WM2NS'):
            idx = pd.date_range('2003-01-04', end, freq='W-SAT')
        elif sid in ('WALCL', 'WDTGAL'):
            idx = pd.date_range('2003-01-01', end, freq='W-WED')
        elif sid == 'GDPNOW':
            idx = pd.date_range('2011-03-31', end, freq='QE')
        else:
            idx = days
        values = np.abs(100 + np.cumsum(rng.normal(0, 1, len(idx)))) + 1
        pd.DataFrame({'date': idx.strftime('%Y-%m-%d'), 'value': values}).to_csv(Path(folder, f'{sid}.csv'), index=False)


def monthly_file(folder):
    months = [{'d': '2026-08', 'g': -0.6, 'l': -0.6, 'regime': 'defense', 'c': True, 'p': False, 'ax': {'g': -1, 'l': -1}},
              {'d': '2026-09', 'g': -0.55, 'l': -0.67, 'regime': 'defense', 'c': False, 'p': True, 'ax': {'g': -1, 'l': -1}}]
    path = Path(folder, 'gl_data.json')
    path.write_text(json.dumps({'meta': {'model_version': '3.0'}, 'months': months}), encoding='utf-8')
    return path


class DailyRadarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        synthetic_cache(cls.tmp.name, pd.Timestamp.today().normalize())
        cls.old_cache = os.environ.get('GL_CACHE')
        os.environ['GL_CACHE'] = cls.tmp.name
        cls.data = radar.build(monthly_file(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        if cls.old_cache is None:
            os.environ.pop('GL_CACHE', None)
        else:
            os.environ['GL_CACHE'] = cls.old_cache
        cls.tmp.cleanup()

    def test_uses_the_validated_variants(self):
        self.assertIs(radar.G_SPEC, bt.G_MONTHLY_ANALOG)
        self.assertIs(radar.L_SPEC, bt.L_MONTHLY_ANALOG)
        self.assertEqual(set(self.data['meta']['method']['g']), {'WEI', 'CCSA', 'IC4WSA'})
        self.assertEqual(set(self.data['meta']['method']['l']),
                         {'WM2NS', 'DGS2', 'NETLIQ', 'DTWEXBGS', 'T10Y3M', 'BAA10Y'})

    def test_schema_and_value_ranges(self):
        d = self.data
        self.assertEqual(set(d), {'meta', 'latest', 'monthly', 'agreement_with_monthly', 'components', 'history'})
        self.assertEqual(d['meta']['reference_model_version'], '3.0')
        self.assertIn('사분면 판정이 아닙니다', d['meta']['purpose_ko'])
        self.assertIn(d['latest']['regime'], (None, 'expansion', 'liquidity', 'adjustment', 'defense'))
        self.assertIn(d['latest']['g_state'], (-1, 0, 1))
        self.assertLessEqual(len(d['history']), radar.HISTORY_BDAYS)
        self.assertEqual(d['history'][-1]['d'], d['latest']['date'])
        self.assertEqual(d['monthly']['last_confirmed'], {'month': '2026-08', 'regime': 'defense'})

    def test_neutral_axis_has_no_regime(self):
        for row in self.data['history']:
            if row['gs'] == 0 or row['ls'] == 0:
                self.assertIsNone(row['regime'])

    def test_components_report_age_and_stale_flag(self):
        for axis in ('g', 'l'):
            for sid, item in self.data['components'][axis].items():
                self.assertIn('age_bdays', item, sid)
                self.assertIsInstance(item['stale'], bool)

    def test_output_is_json_serialisable(self):
        json.dumps(self.data, ensure_ascii=False)


if __name__ == '__main__':
    unittest.main()
