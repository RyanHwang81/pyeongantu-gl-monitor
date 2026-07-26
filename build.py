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
import argparse, io, json, os, sys, time
import urllib.request
import pandas as pd
import numpy as np

WINDOW, MIN_OBS, CLAMP, MIN_EFF_W = 216, 48, 3.0, 0.4
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SPX_URL  = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
UA = {"User-Agent": "Mozilla/5.0 (compatible; pyeongantoo-gl-builder/1.0)"}
CACHE = os.environ.get("GL_CACHE", "")   # 값이 있으면 해당 폴더의 CSV를 우선 사용

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

def fred(series_id):
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

def to_monthly(s):
    return s.resample("MS").mean()

def roll_z(s):
    m = s.rolling(WINDOW, min_periods=MIN_OBS).mean()
    sd = s.rolling(WINDOW, min_periods=MIN_OBS).std()
    return ((s - m) / sd).clip(-CLAMP, CLAMP)

def composite(comps):
    zdf = pd.DataFrame({k: roll_z(v["t"]) for k, v in comps.items()})
    w = pd.Series({k: v["w"] for k, v in comps.items()})
    effw = zdf.notna().mul(w, axis=1).sum(axis=1)
    raw = zdf.mul(w, axis=1).sum(axis=1, min_count=1) / effw
    raw[effw < MIN_EFF_W] = np.nan
    score = roll_z(raw)                                   # 합성지수 재표준화
    return zdf, score.ewm(span=3, min_periods=1).mean().where(score.notna())

def quadrant(g, l):
    if g >= 0 and l >= 0: return "expansion"
    if g < 0 and l >= 0:  return "liquidity"
    if g < 0 and l < 0:   return "defense"
    return "adjustment"

def build_data():
    log = lambda *a: print("[data]", *a, flush=True)
    log("FRED 지표 수집...")
    permit, indpro, payems = fred("PERMIT"), fred("INDPRO"), fred("PAYEMS")
    unrate, icsa = fred("UNRATE"), fred("ICSA")
    m2sl, t10y3m, fedfunds = fred("M2SL"), fred("T10Y3M"), fred("FEDFUNDS")
    totalsl, baa10y, walcl = fred("TOTALSL"), fred("BAA10YM"), fred("WALCL")
    nasdaq, wti = fred("NASDAQCOM"), fred("WTISPLC")
    gs10, tb3 = fred("GS10"), fred("TB3MS")

    permit, indpro, payems = map(to_monthly, (permit, indpro, payems))
    unrate, icsa = to_monthly(unrate), to_monthly(icsa)
    m2sl, t10y3m, fedfunds = map(to_monthly, (m2sl, t10y3m, fedfunds))
    totalsl, baa10y, walcl = map(to_monthly, (totalsl, baa10y, walcl))
    nasdaq, wti, gs10, tb3 = map(to_monthly, (nasdaq, wti, gs10, tb3))

    yoy = lambda s: s.pct_change(12) * 100.0
    G_COMP = {
        "PERMIT":  {"t": yoy(permit), "w": .28, "label": "건축허가 YoY"},
        "INDPRO":  {"t": yoy(indpro), "w": .24, "label": "산업생산 YoY"},
        "PAYEMS":  {"t": yoy(payems), "w": .24, "label": "비농업고용 YoY"},
        "UNRATE":  {"t": -(unrate - unrate.shift(12)), "w": .14, "label": "실업률 12M 변화 (역)"},
        "ICSA":    {"t": -yoy(icsa),  "w": .10, "label": "신규실업수당청구 YoY (역)"},
    }
    L_COMP = {
        "M2SL":    {"t": yoy(m2sl),   "w": .22, "label": "M2 YoY"},
        "T10Y3M":  {"t": t10y3m,      "w": .18, "label": "10Y-3M 기간스프레드"},
        "FEDFUNDS":{"t": -(fedfunds - fedfunds.shift(3)), "w": .18, "label": "기준금리 3M 변화 (역)"},
        "TOTALSL": {"t": yoy(totalsl),"w": .14, "label": "소비자신용 YoY"},
        "BAA10YM": {"t": -baa10y,     "w": .14, "label": "Baa-10Y 신용스프레드 (역)"},
        "WALCL":   {"t": yoy(walcl),  "w": .14, "label": "연준 총자산 YoY"},
    }
    log("G/L 점수 산출...")
    Gz, G = composite(G_COMP)
    Lz, L = composite(L_COMP)
    df = pd.DataFrame({"G": G, "L": L}).dropna()
    df = df[df.index >= "1972-01-01"]

    df["regime"] = [quadrant(g, l) for g, l in zip(df.G, df.L)]
    conf, neutral = [], []
    for i, (g, l, r) in enumerate(zip(df.G, df.L, df.regime)):
        neutral.append(bool(abs(g) < .15 and abs(l) < .15))
        conf.append(bool((i > 0 and df.regime.iloc[i-1] == r) or (abs(g) > .25 and abs(l) > .25)))
    df["confirmed"], df["neutral"] = conf, neutral
    Gz, Lz = Gz.reindex(df.index).round(2), Lz.reindex(df.index).round(2)

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
    out = {
        "meta": {
            "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            "latest": df.index[-1].strftime("%Y-%m"),
            "window": WINDOW, "min_obs": MIN_OBS,
            "g_weights": {k: v["w"] for k, v in G_COMP.items()},
            "l_weights": {k: v["w"] for k, v in L_COMP.items()},
            "g_labels": {k: v["label"] for k, v in G_COMP.items()},
            "l_labels": {k: v["label"] for k, v in L_COMP.items()},
            "asset_labels": {k: v[1] for k, v in ASSETS.items()},
        },
        "months": [
            {"d": d.strftime("%Y-%m"), "g": round(r.G, 3), "l": round(r.L, 3),
             "r": r.regime, "c": r.confirmed, "n": r.neutral,
             "gz": {k: nn(Gz.loc[d, k]) for k in Gz.columns},
             "lz": {k: nn(Lz.loc[d, k]) for k in Lz.columns},
             "a":  {k: nn(ret12[k].loc[d]) for k in ret12}}
            for d, r in df.iterrows()
        ],
    }
    log(f"완료 — {len(df)}개월, 최신 {out['meta']['latest']}, "
        f"G={df.G.iloc[-1]:+.2f} L={df.L.iloc[-1]:+.2f} ({df.regime.iloc[-1]})")
    return out

def render(template, data, public):
    html = template.replace("__GL_DATA__", json.dumps(data, ensure_ascii=False))
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
    data = build_data()
    os.makedirs(a.out, exist_ok=True)
    for name, pub in [("index.html", True), ("gl-internal.html", False)]:
        p = os.path.join(a.out, name)
        open(p, "w", encoding="utf-8").write(render(tpl, data, pub))
        print(f"[build] {p}  ({os.path.getsize(p)//1024} KB, {'공개용' if pub else '내부용'})")
    with open(os.path.join(a.out, "gl_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("[build] 완료")

if __name__ == "__main__":
    sys.exit(main())
