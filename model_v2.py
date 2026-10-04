"""Causal monthly GL model helpers. Only actual observations enter statistics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

WINDOW, MIN_OBS, CLAMP, MIN_EFF_W = 216, 48, 3.0, 0.4


def roll_z(s):
    return ((s-s.rolling(WINDOW,min_periods=MIN_OBS).mean()) /
            s.rolling(WINDOW,min_periods=MIN_OBS).std()).clip(-CLAMP,CLAMP)


def tail_signal(s, index, limit=3):
    """Carry transformed signals only after the last actual month, never inside gaps."""
    actual=s.sort_index().reindex(index)
    values=actual.copy()
    carried=pd.Series('',index=index,dtype=object)
    valid=actual.dropna()
    if not valid.empty:
        last=valid.index[-1]
        age=(index.year-last.year)*12+index.month-last.month
        mask=(age>0)&(age<=limit)
        values.loc[mask]=valid.iloc[-1]
        carried.loc[mask]=last.strftime('%Y-%m')
    return values, actual.notna(), carried


def observed_z(values, observed):
    """Freeze carry-month parameters at the preceding actual observation's window."""
    samples=values.where(observed)
    mean=samples.rolling(WINDOW,min_periods=MIN_OBS).mean()
    sd=samples.rolling(WINDOW,min_periods=MIN_OBS).std()
    mean=mean.where(observed).ffill()
    sd=sd.where(observed).ffill()
    return ((values-mean)/sd).clip(-CLAMP,CLAMP)


@dataclass
class CompositeResult:
    z: pd.DataFrame
    score: pd.Series
    raw: pd.Series
    observed_ratio: pd.Series
    carried: pd.DataFrame
    effective_weight: pd.Series


def composite(comps,index=None):
    if index is None:
        dates=[d for c in comps.values() for d in c['t'].index]
        if not dates:raise ValueError('No model history')
        index=pd.date_range(min(dates),max(dates),freq='MS')
    weights=pd.Series({k:c['w'] for k,c in comps.items()})
    zs,observed,carried,universe={},{},{},{}
    for key,c in comps.items():
        values,seen,carry=tail_signal(c['t'],index)
        zs[key]=observed_z(values,seen)
        observed[key]=seen
        carried[key]=carry
        start=c.get('available_from',c['t'].first_valid_index())
        # A failed entire source must lower coverage, not disappear from the universe.
        universe[key]=index>=pd.Timestamp(start).to_period('M').to_timestamp() if start is not None else np.ones(len(index),dtype=bool)
    zdf=pd.DataFrame(zs,index=index)
    exists=pd.DataFrame(universe,index=index)
    seen=pd.DataFrame(observed,index=index)&exists
    ratio=seen.mul(weights,axis=1).sum(axis=1)/exists.mul(weights,axis=1).sum(axis=1)
    effw=zdf.notna().mul(weights,axis=1).sum(axis=1)
    raw=zdf.mul(weights,axis=1).sum(axis=1,min_count=1)/effw
    raw=raw.where(effw>=MIN_EFF_W-1e-10)
    carried=pd.DataFrame(carried,index=index)
    # Partial carried composites are also excluded from second-layer statistics.
    score=observed_z(raw,raw.notna()&carried.eq('').all(axis=1))
    score=score.ewm(span=3,min_periods=1).mean().where(score.notna())
    return CompositeResult(zdf,score,raw,ratio,carried,effw)


def composite_v1(comps):
    """Frozen pre-upgrade formula for an honest like-for-like comparison."""
    zdf=pd.DataFrame({k:roll_z(c['t']) for k,c in comps.items()})
    weights=pd.Series({k:c['w'] for k,c in comps.items()})
    effw=zdf.notna().mul(weights,axis=1).sum(axis=1)
    raw=zdf.mul(weights,axis=1).sum(axis=1,min_count=1)/effw
    raw[effw<MIN_EFF_W]=np.nan
    score=roll_z(raw)
    return zdf,score.ewm(span=3,min_periods=1).mean().where(score.notna()),raw


def net_liquidity(walcl,tga,rrp):
    """Monthly means in USD millions; pre-RRPONTSYD-series observations assume RRP=0.

    This is a modeling assumption about this series' history, not proof that no
    reverse-repo transactions or facilities existed before its first observation.
    Internal/post-start RRP gaps are not treated as zero.
    """
    frame=pd.concat({'assets':walcl,'tga':tga,'rrp':rrp},axis=1).sort_index()
    first=rrp.first_valid_index()
    if first is not None:
        frame.loc[frame.index<first,'rrp']=0
    return frame['assets']-frame['tga']-frame['rrp']*1000


def expanding_percentile(s):
    values=s.loc[s.index>='1972-01-01']
    def rank(a):
        a=a[np.isfinite(a)]
        return ((a<a[-1]).sum()+(1+(a==a[-1]).sum())/2)/len(a)*100
    return values.expanding(min_periods=1).apply(rank,raw=True).where(values.notna())


def quadrant(g,l):
    if g>=0 and l>=0:return 'expansion'
    if g<0 and l>=0:return 'liquidity'
    if g<0 and l<0:return 'defense'
    return 'adjustment'


def row_flags(g,l,gr,lr,owg,owl,previous=None):
    if min(owg,owl)<.4-1e-10:return None
    p=min(owg,owl)<.8-1e-10
    n=abs(g)<.15 and abs(l)<.15
    b=(abs(g)<.15 or abs(l)<.15) and not n
    r=quadrant(g,l)
    confirmed=not(p or n or b) and (previous==r or (abs(g)>.25 and abs(l)>.25))
    return {'r':r,'p':bool(p),'n':bool(n),'b':bool(b),'c':bool(confirmed),
            'x':bool(g*gr<0 or l*lr<0),'ow':{'g':round(float(owg),2),'l':round(float(owl),2)}}
