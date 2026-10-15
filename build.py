#!/usr/bin/env python3
"""
평안투 GL 레짐 모니터 — 자동 빌드 스크립트

원천 데이터를 내려받아 G/L 점수를 재계산하고 HTML 2종을 생성합니다.
  - dist/index.html          공개용 (방법론 섹션 제외)
  - dist/gl-internal.html    내부용 (방법론 전체 포함)

사용법:
    python3 build.py                # 기본: ./dist 에 출력
    python3 build.py --out public   # 출력 디렉터리 지정

매월 자동 실행은 .github/workflows/update.yml (GitHub Actions) 참고.
"""
import argparse, copy, io, json, os, sys, time
from datetime import date
from pathlib import Path
import urllib.request
import urllib.parse
import ssl
import certifi
import pandas as pd
import numpy as np
from model_v2 import composite, composite_v1, row_flags, net_liquidity, expanding_percentile, quadrant
import model_v3 as v3

WINDOW, MIN_OBS, CLAMP, MIN_EFF_W = 216, 48, 3.0, 0.4
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SPX_URL  = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
# A Mozilla-prefixed agent is stalled by the FRED edge. A plain product token returns the CSV.
UA = {"User-Agent": "pyeongantoo-gl-builder/1.0"}
CACHE = os.environ.get("GL_CACHE", "")   # 값이 있으면 해당 폴더의 CSV를 우선 사용
BOOK_DATA = Path(__file__).with_name("book_dashboard.json")
BOOK_INDICATORS = {
    "global_manufacturing_pmi": "growth",
    "korea_semiconductor_exports": "growth",
    "leading_industry_earnings_revision": "growth",
    "fed_next_move_expectation": "liquidity",
    "dollar_index_trend": "liquidity",
    "high_yield_spread": "liquidity",
    "usdkrw_position": "liquidity",
}
BOOK_DIRECTIONS = {"up", "down", "flat", "pending"}
BOOK_STATES = {"confirmed", "provisional", "pending"}
BOOK_AUTO_SERIES = {
    "DTWEXBGS": "dollar_index_trend",
    "BAMLH0A0HYM2": "high_yield_spread",
    "DEXKOUS": "usdkrw_position",
}
BOOK_REGIMES = {
    ("up", "up"): "expansion",
    ("up", "down"): "selection",
    ("down", "up"): "liquidity",
    ("down", "down"): "winter",
}


def _axis_direction(indicators, axis):
    effects = [item["effect"] for item in indicators.values() if item["axis"] == axis]
    up, down = effects.count("up"), effects.count("down")
    if up > down:
        return "up"
    if down > up:
        return "down"
    tie_key = "leading_industry_earnings_revision" if axis == "growth" else "high_yield_spread"
    tie = indicators.get(tie_key, {}).get("effect")
    return tie if tie in {"up", "down"} else "flat"


def judge_book_month(month, previous_regime=None):
    indicators = month["indicators"]
    complete_count = sum(
        item["effect"] != "pending" and item["state"] != "pending"
        for item in indicators.values()
    )
    growth = _axis_direction(indicators, "growth")
    liquidity = _axis_direction(indicators, "liquidity")
    candidate = BOOK_REGIMES.get((growth, liquidity))
    unresolved = complete_count < len(BOOK_INDICATORS) or candidate is None
    held_previous = bool(unresolved and previous_regime)
    regime = previous_regime if held_previous else candidate
    status = "confirmed" if (complete_count == len(BOOK_INDICATORS) and candidate and
                              all(item["state"] == "confirmed" for item in indicators.values())) else "provisional"
    return {
        "growth": growth,
        "liquidity": liquidity,
        "regime": regime or "undetermined",
        "status": status,
        "held_previous": held_previous,
        "complete_count": complete_count,
        "total_count": len(BOOK_INDICATORS),
    }


def load_book_dashboard(path=BOOK_DATA):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    months = data.get("months") or []
    if not months:
        raise ValueError("book dashboard needs at least one month")
    previous = None
    for month in months:
        indicators = month.get("indicators") or {}
        if set(indicators) != set(BOOK_INDICATORS):
            raise ValueError("book dashboard must contain the exact seven Chapter 27 indicators")
        for key, expected_axis in BOOK_INDICATORS.items():
            item = indicators[key]
            if item.get("axis") != expected_axis:
                raise ValueError(f"invalid axis for {key}")
            if item.get("direction") not in BOOK_DIRECTIONS:
                raise ValueError(f"invalid direction for {key}")
            if item.get("effect") not in BOOK_DIRECTIONS:
                raise ValueError(f"invalid effect for {key}")
            if item.get("state") not in BOOK_STATES:
                raise ValueError(f"invalid state for {key}")
            if not item.get("source_name") or not item.get("as_of"):
                raise ValueError(f"missing source/as_of for {key}")
        month["judgment"] = judge_book_month(month, previous)
        if month["judgment"]["regime"] != "undetermined":
            previous = month["judgment"]["regime"]
        if month["status"] == "confirmed" and month["judgment"]["status"] == "confirmed":
            data["meta"]["latest_confirmed"] = month["date"]
    data["meta"]["latest_judgment"] = months[-1]["judgment"]
    return data

def refresh_book_dashboard(authored, target_month, sources, as_of=None, errors=None):
    """Derive a dated, sourced gauge snapshot without changing author-confirmed history."""
    as_of = as_of or date.today().isoformat()
    errors = errors or {}
    book = copy.deepcopy(authored)
    latest = book["months"][-1]
    if latest["date"] > target_month:
        raise ValueError("author month is newer than refresh month")
    if latest["date"] != target_month:
        rows = copy.deepcopy(latest["indicators"])
        for item in rows.values():
            item["direction"] = item["effect"] = item["state"] = "pending"
            item["reading_ko"] = "미확인 · 마지막 기록 " + item["as_of"]
            item["reading_en"] = "Unverified · last record " + item["as_of"]
        latest = {"date": target_month, "status": "provisional",
                  "source_basis_ko": "확인된 원자료만 반영 · 미확인 칸은 이전 판단을 이월하지 않음",
                  "source_basis_en": "Verified observations only; prior arrows are not carried forward",
                  "indicators": rows}
        book["months"].append(latest)
    first = pd.Timestamp(target_month + "-01")
    prev = first - pd.offsets.MonthBegin(1)
    cutoff = pd.Timestamp(as_of)
    for series_id, name in BOOK_AUTO_SERIES.items():
        item = latest["indicators"][name]
        if item["state"] == "confirmed" and item["as_of"].startswith(target_month):
            continue  # preserve an explicitly verified authored judgment
        item["direction"] = item["effect"] = item["state"] = "pending"
        item["source_name"] = "FRED " + series_id
        item["source_url"] = "https://fred.stlouisfed.org/series/" + series_id
        series = sources.get(series_id, pd.Series(dtype=float, index=pd.DatetimeIndex([]))).loc[:cutoff].dropna().sort_index()
        previous = series[(series.index >= prev) & (series.index < first)]
        current = series[(series.index >= first) & (series.index <= cutoff)]
        if previous.empty or current.empty:
            last = series.index[-1].strftime("%Y-%m-%d") if not series.empty else "없음"
            reason = errors.get(series_id, "no observations in both comparison months")
            item["as_of"] = last
            item["reading_ko"] = "미확인 · " + last + " · " + reason
            item["reading_en"] = "Unverified · " + last + " · " + reason
            continue
        before, after = float(previous.mean()), float(current.mean())
        direction = "up" if after > before else "down" if after < before else "flat"
        item.update(direction=direction, effect={"up": "down", "down": "up", "flat": "flat"}[direction],
                    state="provisional", as_of=current.index[-1].strftime("%Y-%m-%d"),
                    reading_ko=f"당월 평균 {after:.2f} / 전월 {before:.2f} · 잠정",
                    reading_en=f"MTD mean {after:.2f} / prior {before:.2f} · provisional")
    previous_regime = book["months"][-2]["judgment"]["regime"] if len(book["months"]) > 1 else None
    latest["judgment"] = judge_book_month(latest, previous_regime)
    observed = sum(row["state"] != "pending" and row["as_of"].startswith(target_month)
                   for row in latest["indicators"].values())
    book["meta"].update(refresh_as_of=as_of, observed_count=observed,
                        pending_count=len(BOOK_INDICATORS)-observed, latest_judgment=latest["judgment"])
    return book

def fetch(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"다운로드 실패: {url} ({last})")

def fred_api(series_id, key):
    # Never send the key to the generic fetch/logger: its failure message includes the URL.
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(
        {"series_id": series_id, "api_key": key, "file_type": "json"}
    )
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45,
                                    context=ssl.create_default_context(cafile=certifi.where())) as response:
            rows = json.load(response)["observations"]
        frame = pd.DataFrame(rows)[["date", "value"]]
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.dropna().set_index("date")["value"]
    except Exception as exc:
        raise RuntimeError(f"FRED API 수집 실패: {series_id} ({type(exc).__name__})") from None

def fred(series_id):
    key = os.environ.get("FRED_API_KEY", "")
    if key and not (CACHE and os.path.exists(os.path.join(CACHE, series_id + ".csv"))):
        return fred_api(series_id, key)
    if CACHE:
        p = os.path.join(CACHE, series_id + ".csv")
        if os.path.exists(p):
            txt = open(p, encoding="utf-8").read()
        else:
            txt = fetch(FRED.format(series_id)); open(p, "w", encoding="utf-8").write(txt)
    else:
        txt = fetch(FRED.format(series_id))
    d = pd.read_csv(io.StringIO(txt))
    d.columns = ["date", "value"]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna().set_index("date")["value"]

def pending_observations(raw, latest_month, today, weights=None):
    """Report newer, actually observed inputs; do not produce a score from partial coverage."""
    weights = weights or {
        "growth": {"PERMIT": .28, "INDPRO": .24, "PAYEMS": .24, "UNRATE": .14, "ICSA": .10},
        "liquidity": {"M2SL": .22, "T10Y3M": .18, "FEDFUNDS": .18,
                      "TOTALSL": .14, "BAA10YM": .14, "WALCL": .14},
    }
    dates = [d.date() for series in raw.values() for d in series.index
             if latest_month < d.strftime("%Y-%m") <= today.strftime("%Y-%m") and d.date() <= today]
    if not dates:
        return None
    month = max(dates).strftime("%Y-%m")
    observed = []
    coverage = {}
    for axis, components in weights.items():
        found = []
        for series_id, weight in components.items():
            series = raw.get(series_id, pd.Series(dtype=float))
            valid_dates = [d for d in series.index if d.strftime("%Y-%m") == month and d.date() <= today]
            if valid_dates:
                last = max(valid_dates).strftime("%Y-%m-%d")
                observed.append({"id": series_id, "as_of": last})
                found.append(weight)
        coverage[axis] = {"available": len(found), "total": len(components), "weight": round(sum(found), 2)}
    return {"month": month, "as_of": max(row["as_of"] for row in observed),
            "status": "month_in_progress" if month == today.strftime("%Y-%m") else "incomplete",
            **coverage, "observed": observed}

def recalculation_note(prior, current):
    prior_ver=(prior or {}).get('meta',{}).get('model_version')
    current_ver=current.get('meta',{}).get('model_version')
    if prior and prior_ver!='2.0' and current_ver=='2.0':
        old,new=prior['months'][-1],current['months'][-1]
        return {'month':new['d'],'scope':'all_history','reason':'model_v1_to_v2',
                'before':{k:old[k] for k in ('g','l','r')},'after':{k:new[k] for k in ('g','l','r')},
                'newly_available':[]}
    if prior and prior_ver=='2.0' and current_ver=='3.0' and prior.get('months') and current.get('months'):
        old,new=prior['months'][-1],current['months'][-1]
        return {'month':new['d'],'scope':'all_history','reason':'model_v2_to_v3',
                'before':{k:old[k] for k in ('g','l','r') if k in old},
                'after':{k:new[k] for k in ('g','l','r')},
                'newly_available':[]}
    if not prior or not prior.get("months") or not current["months"]:
        return None
    old, new = prior["months"][-1], current["months"][-1]
    if old["d"] != new["d"]:
        return None
    if all(old[k] == new[k] for k in ("g", "l", "r")):
        existing = prior.get("meta", {}).get("recalculation")
        return existing if existing and existing["month"] == new["d"] else None
    existing = prior.get("meta", {}).get("recalculation")
    before = existing["before"] if existing and existing["month"] == new["d"] else old
    new_inputs = [k for axis in ("gz", "lz") for k, value in new[axis].items()
                  if value is not None and old[axis].get(k) is None]
    if existing and existing["month"] == new["d"]:
        new_inputs = list(dict.fromkeys(existing["newly_available"] + new_inputs))
    return {"month": new["d"],
            "before": {k: before[k] for k in ("g", "l", "r")},
            "after": {k: new[k] for k in ("g", "l", "r")},
            "newly_available": new_inputs}

def to_monthly(s):
    return s.resample("MS").mean()

def provisional_point(axes, raw, pending, carry_limit=3, z_method='classical', rescale=False, correlation=None):
    """Separate current-month estimate using the monthly history's carry/statistics rules."""
    if not pending:return None
    month=pd.Timestamp(pending['month']+'-01')
    as_of=pd.Timestamp(pending['as_of'])
    result={'month':pending['month'],'as_of':pending['as_of'],
            'status':'provisional_carry_forward','p':True,'c':False,'coverage':{},'inputs':{}}
    values={}
    correlation=correlation or {}
    for axis,components in axes.items():
        sliced={k:{**c,'t':c['t'].loc[:month]} for k,c in components.items()}
        dates=[d for c in sliced.values() for d in c['t'].index]
        if not dates:return None
        result_axis=composite(sliced,pd.date_range(min(dates),month,freq='MS'),
                              carry_limit=carry_limit,z_method=z_method,rescale=rescale,
                              correlation=correlation.get(axis,0.0))
        if pd.isna(result_axis.score.loc[month]):return None
        provenance={}
        for sid,c in sliced.items():
            start=c.get('available_from',c['t'].first_valid_index())
            if start is not None and month<pd.Timestamp(start).to_period('M').to_timestamp():continue
            if pd.isna(result_axis.z.loc[month,sid]):return None
            dates=raw.get(sid,pd.Series(dtype=float,index=pd.DatetimeIndex([]))).loc[:as_of].dropna()
            if dates.empty:return None
            carried=result_axis.carried.loc[month,sid]
            provenance[sid]={'source_as_of':dates.index[-1].strftime('%Y-%m-%d'),
                             'used_month':carried or pending['month'],'weight':c['w'],'carried':bool(carried)}
        observed=sum(p['weight'] for p in provenance.values() if not p['carried'])
        result['coverage'][axis]={'observed':sum(not p['carried'] for p in provenance.values()),
                                 'total':len(provenance),'observed_weight':round(observed,2),
                                 'observed_ratio':round(float(result_axis.observed_ratio.loc[month]),2),
                                 'carried_weight':round(sum(p['weight'] for p in provenance.values() if p['carried']),2)}
        result['inputs'][axis]=provenance
        key='g' if axis=='growth' else 'l'
        result[key]=round(float(result_axis.score.loc[month]),3)
        result[key+'r']=round(float(result_axis.raw.loc[month]),3)
        level=result_axis.unit if result_axis.unit is not None else result_axis.raw
        percentile=expanding_percentile(level).loc[month]
        result[key+'p']=round(float(percentile))
        result[key+'z']={sid:round(float(z),2) for sid,z in result_axis.z.loc[month].items() if pd.notna(z)}
        values[key]=result[key]
    result['r']=quadrant(result['g'],result['l'])
    result['n']=abs(result['g'])<.15 and abs(result['l'])<.15
    result['b']=(abs(result['g'])<.15 or abs(result['l'])<.15) and not result['n']
    result['x']=result['g']*result['gr']<0 or result['l']*result['lr']<0
    return result

def build_data():
    log = lambda *a: print("[data]", *a, flush=True)
    log("FRED 지표 수집...")
    source_errors={}
    ids=('PERMIT','INDPRO','PAYEMS','UNRATE','ICSA','M2SL','T10Y3M','FEDFUNDS',
         'TOTALSL','BAA10YM','WALCL','DGS2','DTWEXBGS','WTREGEN','RRPONTSYD',
         'NASDAQCOM','WTISPLC','GS10','TB3MS')
    raw={}
    for sid in ids:
        try:raw[sid]=fred(sid).loc[:pd.Timestamp(date.today())]
        except Exception as exc:
            source_errors[sid]=type(exc).__name__
            raw[sid]=pd.Series(dtype=float,index=pd.DatetimeIndex([]))
            log(sid,'source_error',type(exc).__name__)
    monthly={k:to_monthly(v) for k,v in raw.items()}
    permit,indpro,payems,unrate,icsa=(monthly[k] for k in ('PERMIT','INDPRO','PAYEMS','UNRATE','ICSA'))
    m2sl,t10y3m,fedfunds,totalsl,baa10y,walcl=(monthly[k] for k in ('M2SL','T10Y3M','FEDFUNDS','TOTALSL','BAA10YM','WALCL'))
    nasdaq,wti,gs10,tb3=(monthly[k] for k in ('NASDAQCOM','WTISPLC','GS10','TB3MS'))
    netliq=net_liquidity(walcl,monthly['WTREGEN'],monthly['RRPONTSYD'])
    # Provenance uses the latest common actual input date in each monthly estimate.
    net_dates=[];net_values=[]
    rrp_start=raw['RRPONTSYD'].first_valid_index()
    for month,value in netliq.dropna().items():
        series_ids=['WALCL','WTREGEN']+(['RRPONTSYD'] if rrp_start is not None and month.to_period('M')>=rrp_start.to_period('M') else [])
        last=[v.index[-1] for sid in series_ids if not (v:=raw[sid][raw[sid].index.to_period('M')==month.to_period('M')]).empty]
        if len(last)==len(series_ids):net_dates.append(min(last));net_values.append(value)
    raw['NETLIQ']=pd.Series(net_values,index=pd.DatetimeIndex(net_dates),dtype=float)
    raw['DOLLAR']=raw['DTWEXBGS']
    yoy=lambda s:s.pct_change(12,fill_method=None)*100.0
    G_COMP={
        'PERMIT':{'t':yoy(permit),'w':.28,'label':'건축허가 YoY'},
        'INDPRO':{'t':yoy(indpro),'w':.24,'label':'산업생산 YoY'},
        'PAYEMS':{'t':yoy(payems),'w':.24,'label':'비농업고용 YoY'},
        'UNRATE':{'t':-(unrate-unrate.shift(12)),'w':.14,'label':'실업률 12M 변화 (역)'},
        'ICSA':{'t':-yoy(icsa),'w':.10,'label':'신규실업수당청구 YoY (역)'},
    }
    # v2 keeps spread levels inside L. v3 converts those two inputs to 6-month changes.
    L_V2={
        'M2SL':{'t':yoy(m2sl),'w':.18,'label':'M2 YoY'},
        'DGS2':{'t':-(monthly['DGS2']-monthly['DGS2'].shift(3)),'w':.18,'label':'2년물 금리 3M 변화 (역)'},
        'NETLIQ':{'t':yoy(netliq),'w':.15,'label':'순유동성(연준자산−TGA−역레포) YoY'},
        'DOLLAR':{'t':-yoy(monthly['DTWEXBGS']),'w':.15,'label':'광의 달러지수 YoY (역)'},
        'T10Y3M':{'t':t10y3m,'w':.12,'label':'10Y-3M 기간스프레드'},
        'BAA10YM':{'t':-baa10y,'w':.12,'label':'Baa-10Y 신용스프레드 (역)'},
        'TOTALSL':{'t':yoy(totalsl),'w':.10,'label':'소비자신용 YoY'},
    }
    L_COMP=v3.change_liquidity(monthly, netliq)
    FC_COMP=v3.financial_conditions(monthly)
    # Freeze v1's implicit pct_change fill behavior for the comparison only.
    old_yoy=lambda x:x.ffill().pct_change(12,fill_method=None)*100
    G_V1={k:{**v,'t':old_yoy(monthly[k])*(-1 if k=='ICSA' else 1)} if k!='UNRATE' else dict(v) for k,v in G_COMP.items()}
    L_V1={
        'M2SL':{'t':old_yoy(m2sl),'w':.22},'T10Y3M':{'t':t10y3m,'w':.18},
        'FEDFUNDS':{'t':-(fedfunds-fedfunds.shift(3)),'w':.18},
        'TOTALSL':{'t':old_yoy(totalsl),'w':.14},'BAA10YM':{'t':-baa10y,'w':.14},
        'WALCL':{'t':old_yoy(walcl),'w':.14},
    }
    for comps in (G_COMP,L_V2,L_COMP,FC_COMP):
        for key,c in comps.items():c['available_from']=raw[key].first_valid_index()
    log("G/L 점수 산출...")
    dates=[d for c in (*G_COMP.values(),*L_COMP.values(),*L_V2.values()) for d in c['t'].index]
    index=pd.date_range(min(dates),pd.Timestamp(date.today()).replace(day=1),freq='MS')
    def scored(comps, method):
        draft=v3.score_composite(comps, index, method, 0.0)
        rho=v3.frozen_pairwise_rho(draft.z)
        return v3.score_composite(comps, index, method, rho), rho
    ga,rho_g=scored(G_COMP,'robust')
    la,rho_l=scored(L_COMP,'robust')
    g_classic,_=scored(G_COMP,'classical')
    l_classic,_=scored(L_COMP,'classical')
    fc_result,rho_fc=scored(FC_COMP,'robust')
    ga_v2,la_v2=composite(G_COMP,index),composite(L_V2,index)
    Gz,Lz=ga.z,la.z
    df=pd.DataFrame({'G':ga.score,'L':la.score,'gr':ga.raw,'lr':la.raw,
                     'owg':ga.observed_ratio,'owl':la.observed_ratio,
                     'gc':g_classic.score,'lc':l_classic.score,
                     'fc':fc_result.score,'fcr':fc_result.raw}).dropna(subset=['G','L','gr','lr','owg','owl'])
    df=df[(df.index>='1972-01-01')&(df.index<index[-1])]
    df=df[(df.owg>=.4-1e-10)&(df.owl>=.4-1e-10)]
    if df.empty:raise RuntimeError('Insufficient actual GL coverage')
    gp,lp=expanding_percentile(ga.unit),expanding_percentile(la.unit)
    fcp=expanding_percentile(fc_result.unit)
    df['gp'],df['lp']=gp.reindex(df.index),lp.reindex(df.index)
    flags,axis_state=v3.classify(df)
    v2=pd.DataFrame({'G':ga_v2.score,'L':la_v2.score,'gr':ga_v2.raw,'lr':la_v2.raw,
                     'owg':ga_v2.observed_ratio,'owl':la_v2.observed_ratio}).dropna()
    v2=v2[(v2.index>='1972-01-01')&(v2.index<index[-1])]
    v2=v2[(v2.owg>=.4-1e-10)&(v2.owl>=.4-1e-10)]
    v2_flags=[];previous=None
    for d,row in v2.iterrows():
        flag=row_flags(row.G,row.L,row.gr,row.lr,row.owg,row.owl,previous)
        v2_flags.append(flag);previous=flag['r']
    Gz,Lz=Gz.reindex(df.index).round(2),Lz.reindex(df.index).round(2)
    _,v1g,_=composite_v1(G_V1);_,v1l,_=composite_v1(L_V1)
    v1=pd.DataFrame({'g':v1g,'l':v1l}).dropna()
    v1=v1[(v1.index>='1972-01-01')&(v1.index<index[-1])]
    compare=[{'d':d.strftime('%Y-%m'),'g':round(row.g,3),'l':round(row.l,3),
              'r':quadrant(row.g,row.l)} for d,row in v1.tail(36).iterrows()]
    v2_history=[{'d':d.strftime('%Y-%m'),'g':round(row.G,3),'l':round(row.L,3),
                 'r':flag['r'],'c':flag['c'],'n':flag['n'],'b':flag['b'],'p':flag['p']}
                for (d,row),flag in zip(v2.iterrows(),v2_flags)]

    log("자산군 수익률 산출...")
    sp = pd.read_csv(io.StringIO(fetch(SPX_URL)))[["Date", "SP500"]].dropna()
    sp["Date"] = pd.to_datetime(sp["Date"])
    spx = sp.set_index("Date")["SP500"].resample("MS").last()
    gd = pd.read_csv(io.StringIO(fetch(GOLD_URL)))
    gd.columns = ["date", "price"]; gd["date"] = pd.to_datetime(gd["date"])
    gold = gd.set_index("date")["price"].resample("MS").last()

    y = gs10 / 100.0
    dur = (1 - (1 + y.shift(1)) ** -10) / y.shift(1)
    ust = (1 + (y.shift(1) / 12 - dur * (y - y.shift(1))).fillna(0)).cumprod()
    cash = (1 + (tb3 / 100.0 / 12).fillna(0)).cumprod()

    ASSETS = {"spx": (spx, "미국주식 S&P500"), "ndx": (nasdaq, "나스닥"),
              "gold": (gold, "금"), "wti": (wti, "원유 WTI"),
              "ust": (ust, "미국채 10Y"), "cash": (cash, "현금 (3M T-Bill)")}
    ret12 = {k: (v[0].pct_change(12) * 100).reindex(df.index).round(1) for k, v in ASSETS.items()}

    nn = lambda v: None if pd.isna(v) else v
    pending = pending_observations(
        raw, df.index[-1].strftime("%Y-%m"), date.today(),
        {"growth": {k: v["w"] for k, v in G_COMP.items()},
         "liquidity": {k: v["w"] for k, v in L_COMP.items()}})
    estimate = provisional_point(
        {"growth": G_COMP, "liquidity": L_COMP}, raw, pending,
        carry_limit=v3.CARRY_LIMIT, z_method='robust', rescale=True,
        correlation={'growth': rho_g, 'liquidity': rho_l})
    if estimate:
        gs=v3.step_axis(estimate['g'], estimate.get('gp'), axis_state['g'])
        ls=v3.step_axis(estimate['l'], estimate.get('lp'), axis_state['l'])
        regime=None if gs==0 or ls==0 else quadrant(1 if gs>0 else -1, 1 if ls>0 else -1)
        estimate.update(regime=regime, n=bool(gs==0 and ls==0), b=bool((gs==0)^(ls==0)),
                        c=False, phase=v3.phase_of(gs, ls, None, None, regime),
                        allocation=v3.REGIME_ASSETS[regime]['favor'] if regime else v3.NEUTRAL_ALLOCATION,
                        ax={'g':int(gs),'l':int(ls)})
    def _at(series, stamp):
        ts=pd.Timestamp(stamp)
        if ts not in series.index or pd.isna(series.loc[ts]): return None
        return round(float(series.loc[ts]), 3)
    def _share(mask):
        return None if not len(mask) else round(float(mask.mean()), 3)
    v2_l_std=la_v2.raw.rolling(216, min_periods=48).std()
    classic_g=g_classic.score.reindex(df.index)
    post=df.loc['2022':'2026']
    pre=v2.loc['2010':'2019']
    try:
        nfci=to_monthly(fred('NFCI'))
    except Exception:
        nfci=None
    nfci_diag={'available': nfci is not None}
    if nfci is not None:
        panel=pd.DataFrame({'L':df.L,'FC':df.fc,'easy':-nfci,'d6':-nfci.diff(6)}).loc['1990':'2019'].dropna()
        def _corr(left, right):
            value=left.corr(right)
            return None if pd.isna(value) else round(float(value), 3)
        nfci_diag.update(sample='1990-2019', n=int(len(panel)),
                         l_change_corr_level=_corr(panel.L, panel.easy),
                         l_change_corr_d6=_corr(panel.L, panel.d6),
                         fc_level_corr=_corr(panel.FC, panel.easy))
    regimes=[flag['regime'] for flag in flags]
    diagnostics={
        'z_method':'robust_mad',
        'rescale':'theoretical_correlation',
        'rho_frozen_2019':{
            'g':_at(rho_g,'2019-12-01'),'l':_at(rho_l,'2019-12-01'),'fc':_at(rho_fc,'2019-12-01')},
        'carry_limit':v3.CARRY_LIMIT,
        'hysteresis':{'enter':v3.ENTER,'exit':v3.EXIT,'strong':v3.STRONG,'confirm_months':v3.CONFIRM_MONTHS},
        'covid':{
            'v2_l_second_stage_std_2019_12':_at(v2_l_std,'2019-12-01'),
            'v2_l_second_stage_std_2026_06':_at(v2_l_std,'2026-06-01'),
            'v2_both_abs_lt_0.15_2010_2019':_share((pre.G.abs()<0.15)&(pre.L.abs()<0.15)),
            'v2_both_abs_lt_0.15_2022_2026':_share((v2.loc['2022':'2026'].G.abs()<0.15)&(v2.loc['2022':'2026'].L.abs()<0.15)),
            'v3_classical_abs_g_lt_0.25_2022_2026':_share(classic_g.loc['2022':'2026'].abs()<0.25),
            'v3_robust_abs_g_lt_0.25_2022_2026':_share(post.G.abs()<0.25),
        },
        'nfci':nfci_diag,
        'confirmation':{
            'months':len(flags),
            'neutral':sum(flag['n'] for flag in flags),
            'confirmed':sum(flag['c'] for flag in flags),
            'whipsaw_reversals':v3.whipsaw_count(df.index, regimes, [flag['c'] for flag in flags]),
            'transitions':v3.transition_matrix(df.index, regimes),
        },
    }
    def _month_row(d, row, flag):
        return {"d":d.strftime('%Y-%m'),"g":round(row.G,3),"l":round(row.L,3),**flag,
                "gr":round(row.gr,3),"lr":round(row.lr,3),
                "gp":None if pd.isna(row.gp) else int(round(float(row.gp))),
                "lp":None if pd.isna(row.lp) else int(round(float(row.lp))),
                "gc":None if pd.isna(row.gc) else round(float(row.gc),3),
                "lc":None if pd.isna(row.lc) else round(float(row.lc),3),
                "fc":None if pd.isna(row.fc) else round(float(row.fc),3),
                "fcr":None if pd.isna(row.fcr) else round(float(row.fcr),3),
                "fcp":None if d not in fcp.index or pd.isna(fcp.loc[d]) else int(round(float(fcp.loc[d]))),
                "cf":{k:used for result in (ga,la) for k,used in result.carried.loc[d].items() if used},
                "gz":{k:nn(Gz.loc[d,k]) for k in Gz.columns},"lz":{k:nn(Lz.loc[d,k]) for k in Lz.columns},
                "a":{k:nn(ret12[k].loc[d]) for k in ret12}}
    months=[_month_row(d,row,flag) for (d,row),flag in zip(df.iterrows(),flags)]
    out = {
        "meta": {
            "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
            "model_version":"3.0",
            "v2_model_version":"2.0",
            "model_changes":[
                "Single-stage robust median/MAD z-score with theoretical correlation rescaling; classical single-stage kept as gc/lc",
                "L uses 6-month changes for T10Y3M and BAA10YM; level spreads move to a separate financial-conditions score",
                "Regime is hysteresis plus 3-month direction; neutral months reset confirmation and carry no quadrant",
                "Carry of transformed signals shortened from 3 months to 2",
            ],
            "summary_ko":"v3는 216개월 재표준화를 없애고, 코로나 스파이크에 덜 민감한 중위수/MAD로 점수를 만듭니다. 유동성 축은 변화율만 쓰고, 스프레드 수준은 금융여건으로 따로 둡니다. 중립 달은 사분면을 주지 않습니다.",
            "neutral_allocation":v3.NEUTRAL_ALLOCATION,
            "regime_assets":v3.REGIME_ASSETS,
            "source_errors":source_errors,"v1_compare":compare,
            "v2_compare":v2_history[-36:],
            "v2_history":v2_history,
            "v2_latest":[row for row in v2_history if row['d']>='2026-07'],
            "diagnostics":diagnostics,
            "latest_liquidity_components":[{'id':k,'z':round(float(la.z.loc[df.index[-1],k]),6) if pd.notna(la.z.loc[df.index[-1],k]) else None,
                'weight':v['w'],'contribution':round(float(la.z.loc[df.index[-1],k])*v['w'],6) if pd.notna(la.z.loc[df.index[-1],k]) else None,
                'carried_from':la.carried.loc[df.index[-1],k] or None} for k,v in L_COMP.items()],
            "source_first_observation":{k:v.index[0].strftime('%Y-%m-%d') if not v.empty else None for k,v in raw.items()},
            "latest": df.index[-1].strftime("%Y-%m"),
            "pending_observations": pending,
            "provisional_point": estimate,
            "window": WINDOW, "min_obs": MIN_OBS,
            "g_weights": {k: v["w"] for k, v in G_COMP.items()},
            "l_weights": {k: v["w"] for k, v in L_COMP.items()},
            "g_labels": {k: v["label"] for k, v in G_COMP.items()},
            "l_labels": {k: v["label"] for k, v in L_COMP.items()},
            "fc_labels": {k: v["label"] for k, v in FC_COMP.items()},
            "fc_weights": {k: v["w"] for k, v in FC_COMP.items()},
            "asset_labels": {k: v[1] for k, v in ASSETS.items()},
        },
        "months": months,
    }
    shown=flags[-1]['regime'] or 'neutral'
    log(f"완료 — {len(df)} months, latest {out['meta']['latest']}, "
        f"G={df.G.iloc[-1]:+.3f} L={df.L.iloc[-1]:+.3f} ({shown})")
    return out

def render(template, data, book_data, public, language='ko'):
    if language=='en':
        from localization import english
        template,data,book_data=english(template,data,book_data)
    html = template.replace("__GL_DATA__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__BOOK_DATA__", json.dumps(book_data, ensure_ascii=False))
    if public:
        s, e = html.find("<!--METHOD_START-->"), html.find("<!--METHOD_END-->")
        if s != -1 and e != -1:
            html = html[:s] + html[e + len("<!--METHOD_END-->"):]
    return html

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gl_template.html"))
    a = ap.parse_args()
    tpl = open(a.template, encoding="utf-8").read()
    previous_path = Path(a.out) / "gl_data.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
    data = build_data()
    data["meta"]["recalculation"] = recalculation_note(previous, data)
    book_sources, book_errors = {}, {}
    for series_id in BOOK_AUTO_SERIES:
        try:
            book_sources[series_id] = fred(series_id)
        except Exception as exc:
            # Never publish the original exception: some clients include secrets in URLs.
            book_errors[series_id] = type(exc).__name__
    book_data = refresh_book_dashboard(load_book_dashboard(), date.today().strftime("%Y-%m"),
                                       book_sources, errors=book_errors)
    print("[book]", book_data["months"][-1]["date"],
          f"observed={book_data['meta']['observed_count']}/7",
          f"pending={book_data['meta']['pending_count']}",
          "source_errors=" + ",".join(book_errors))
    os.makedirs(a.out, exist_ok=True)
    for name, pub in [("index.html", True), ("gl-internal.html", False), ("en.html", True)]:
        p = os.path.join(a.out, name)
        open(p, "w", encoding="utf-8").write(render(tpl, data, book_data, pub,language="en" if name=="en.html" else "ko"))
        print(f"[build] {p}  ({os.path.getsize(p)//1024} KB, {'공개용' if pub else '내부용'})")
    with open(os.path.join(a.out, "gl_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("[build] 완료")

if __name__ == "__main__":
    sys.exit(main())
