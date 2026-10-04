import unittest
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import pandas as pd
import build
import model_v2 as model


class ModelV2Tests(unittest.TestCase):
    def history(self,end='2026-08-01'):
        dates=pd.date_range('1980-01-01',end,freq='MS')
        return pd.Series(np.sin(np.arange(len(dates))/9)+np.arange(len(dates))/500, index=dates)

    def test_only_tail_three_months_are_carried(self):
        s=self.history();s.iloc[30]=np.nan
        index=pd.date_range(s.index[0],'2026-12-01',freq='MS')
        values,seen,carry=model.tail_signal(s,index)
        self.assertTrue(pd.isna(values.iloc[30]))
        for d in ('2026-09-01','2026-10-01','2026-11-01'):
            self.assertEqual(s.iloc[-1],values.loc[d])
            self.assertFalse(seen.loc[d]);self.assertEqual('2026-08',carry.loc[d])
        self.assertTrue(pd.isna(values.loc['2026-12-01']))
        short=model.tail_signal(s.loc[:'2026-07-01'],pd.date_range(s.index[0],'2026-07-01',freq='MS'))[0]
        pd.testing.assert_series_equal(short,values.loc[:'2026-07-01'])

    def test_carry_does_not_enter_indicator_or_composite_statistics(self):
        s=self.history();index=pd.date_range(s.index[0],'2026-11-01',freq='MS')
        values,seen,_=model.tail_signal(s,index)
        z=model.observed_z(values,seen)
        expected=model.roll_z(s).iloc[-1]
        for d in ('2026-09-01','2026-10-01','2026-11-01'):
            self.assertAlmostEqual(expected,z.loc[d])
        # New actual data may alter the composite value, but carried months do not
        # enter its second-layer parameters either.
        observed=pd.Series(True,index=index);observed.loc['2026-09-01':]=False
        changed=values.copy();changed.loc['2026-09-01':]=[10.,20.,30.]
        z2=model.observed_z(changed,observed)
        mu=s.iloc[-model.WINDOW:].mean();sd=s.iloc[-model.WINDOW:].std()
        self.assertEqual(np.clip((20-mu)/sd,-3,3),z2.loc['2026-10-01'])

    def test_available_universe_starts_at_first_observation(self):
        old=self.history();new=old.loc['2006-01-01':]
        result=model.composite({'OLD':{'t':old,'w':.85},'DOLLAR':{'t':new,'w':.15,'available_from':'2006-01-02'}})
        self.assertEqual(1.,result.observed_ratio.loc['2005-12-01'])
        self.assertEqual(1.,result.observed_ratio.loc['2006-01-01'])
        missing=model.composite({'OLD':{'t':old,'w':.85},'FAILED':{'t':pd.Series(dtype=float,index=pd.DatetimeIndex([])),'w':.15}})
        self.assertAlmostEqual(.85,missing.observed_ratio.loc['2005-12-01'])

    def test_coverage_thresholds_and_boundary_confirmation(self):
        self.assertTrue(model.row_flags(-.4,.4,-.2,.2,.6,1)['p'])
        self.assertFalse(model.row_flags(-.4,.4,-.2,.2,.6,1)['c'])
        self.assertIsNone(model.row_flags(-.4,.4,-.2,.2,.3,1))
        edge=model.row_flags(-.01,.30,-.01,.10,1,1,'liquidity')
        self.assertTrue(edge['b']);self.assertFalse(edge['c']);self.assertFalse(edge['n'])
        neutral=model.row_flags(.01,.01,.01,.01,1,1,'expansion')
        self.assertTrue(neutral['n']);self.assertFalse(neutral['b']);self.assertFalse(neutral['c'])
        self.assertTrue(model.row_flags(.4,.4,.2,.2,.8,.8)['c'])

    def test_net_liquidity_units_and_pre_series_zero_only(self):
        index=pd.date_range('2002-01-01',periods=4,freq='MS')
        assets=pd.Series([10000]*4,index=index);tga=pd.Series([1000]*4,index=index)
        rrp=pd.Series([2,np.nan,3],index=index[1:])
        actual=model.net_liquidity(assets,tga,rrp)
        self.assertEqual(9000,actual.iloc[0]);self.assertEqual(7000,actual.iloc[1])
        self.assertTrue(pd.isna(actual.iloc[2]));self.assertEqual(6000,actual.iloc[3])
        failed=model.net_liquidity(assets,tga,pd.Series(dtype=float,index=pd.DatetimeIndex([])))
        self.assertTrue(failed.isna().all())

    def test_sign_disagreement_and_percentile_are_causal(self):
        self.assertTrue(model.row_flags(-.4,.3,-.2,-.1,1,1)['x'])
        s=self.history();prior=model.expanding_percentile(s)
        extended=pd.concat([s,pd.Series([10000],index=[s.index[-1]+pd.offsets.MonthBegin(1)])])
        pd.testing.assert_series_equal(prior,model.expanding_percentile(extended).loc[s.index],check_freq=False)
        duplicate=pd.Series([1.,1.,2.],index=pd.date_range('1972-01-01',periods=3,freq='MS'))
        self.assertEqual(75.,model.expanding_percentile(duplicate).iloc[1])

    def test_monthly_and_current_estimate_share_score_and_raw(self):
        old=self.history();index=pd.date_range(old.index[0],'2026-09-01',freq='MS')
        fresh=pd.concat([old,pd.Series([1.7],index=index[-1:])])
        c={'OBS':{'t':fresh,'w':.4},'OLD':{'t':old,'w':.6}}
        raw={'OBS':pd.Series([1.7],index=pd.to_datetime(['2026-09-19'])),
             'OLD':pd.Series([old.iloc[-1]],index=pd.to_datetime(['2026-08-01']))}
        point=build.provisional_point({'growth':c,'liquidity':c},raw,{'month':'2026-09','as_of':'2026-09-19'})
        monthly=model.composite(c,index)
        self.assertEqual(round(monthly.score.iloc[-1],3),point['g'])
        self.assertEqual(round(monthly.raw.iloc[-1],3),point['gr'])
        self.assertTrue(point['p']);self.assertFalse(point['c'])

    def test_upgrade_recalculation_covers_all_history(self):
        old={'meta':{},'months':[{'d':'2026-09','g':-.01,'l':.299,'r':'liquidity'}]}
        new={'meta':{'model_version':'2.0'},'months':[{'d':'2026-09','g':-.01,'l':-.3,'r':'defense'}]}
        note=build.recalculation_note(old,new)
        self.assertEqual('all_history',note['scope']);self.assertEqual('model_v1_to_v2',note['reason'])

    def test_failed_source_keeps_build_running_and_records_only_type(self):
        index=pd.date_range('1960-01-01','2026-10-01',freq='MS')
        x=np.arange(len(index))
        def source(key):
            if key=='DGS2':raise RuntimeError('sensitive-sentinel')
            values=100+x/5+10*np.sin(x/9)
            if key=='WALCL':values=1e6+50000*np.sin(x/9)+x*100
            if key=='WTREGEN':values=1e5+10000*np.sin(x/11)
            if key=='RRPONTSYD':values=2+np.sin(x/13)
            return pd.Series(values,index=index)
        spx=pd.DataFrame({'Date':index,'SP500':100+x}).to_csv(index=False)
        gold=pd.DataFrame({'date':index,'price':200+x}).to_csv(index=False)
        with patch('build.fred',side_effect=source),patch('build.fetch',side_effect=lambda u:spx if u==build.SPX_URL else gold),redirect_stdout(io.StringIO()):
            data=build.build_data()
        self.assertEqual({'DGS2':'RuntimeError'},data['meta']['source_errors'])
        self.assertGreater(len(data['months']),100)
        self.assertNotIn('sensitive-sentinel',json.dumps(data))


if __name__=='__main__':unittest.main()
