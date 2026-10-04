"""English reader copy for the same model and single-file public monitor."""
import copy

COPY = dict(line.split('\t',1) for line in '''
평안투 · GL 레짐 모니터	Pyeongantu · GL Regime Monitor
평안투 GL 레짐 모니터	Pyeongantu GL Regime Monitor
평안투 	Pyeongantu 
GL 레짐 모니터	GL Regime Monitor
성장 × 유동성 · FRED 정량 모델 v2.0	Growth × Liquidity · FRED quantitative model v2.0
미국 매크로 정량 보조모델	U.S. macro quantitative reference model
마지막 계산월의 발표분 반영점과 이번 달 월중 잠정점을 구분합니다. 과거 말일 당시의 데이터 재현은 아닙니다.	Completed-month calculations and the current-month estimate are separate. Historical scores use today's data, not month-end vintages.
G·L 점수는 최근 18년 대비 상대 위치입니다. 원합성치는 구성 지표 z의 가중평균이며 경제적 절대 수준과 다릅니다.	G and L scores show relative positions over the past 18 years. The raw composite averages component z-scores; it is not an absolute economic measurement.
현재까지 발표된 해당 월 지표 반영 · 당시 시점 재현 아님	Reflects currently released observations; not a historical real-time vintage
월중 잠정 추정점 / Provisional	Current-month provisional estimate
관측·이월 지표와 재계산 내역 / Inputs &amp; revisions	Observed and carried inputs · Revisions
마지막 계산월 레짐	Latest calculated regime
마지막 계산월	Latest calculated month
레짐 지속	Regime duration
동일 4분면 연속 개월	Consecutive months in the same quadrant
월간 계산 점 선택 · 속 빈 점은 잠정치	Select a monthly point · Hollow points are provisional
축별 확대 · X/Y 축 범위가 서로 다르며 장기 국면 지도와 이동 거리를 직접 비교하지 않습니다	Axes scale independently; distances cannot be compared directly with the long-term map
 · X/Y 축 범위가 서로 다르며 장기 국면 지도와 이동 거리를 직접 비교하지 않습니다.	 · The two axes use different ranges; distances are not directly comparable with the long-term map.
최근 12개월 이동	Recent 12-month path
최근 12개월	Last 12 months
최근 24개월	Last 24 months
최근 이동 기간	Recent path range
장기 GL 국면 지도	Long-term GL regime map
장기 국면 기간	Long-term range
연간 평균 · 고정축 ±3	Annual means · Fixed axes ±3
연간 평균으로 역사적 위치와 국면 전환을 비교합니다	Compare historical positions and transitions with annual means
 · 기간을 바꿔도 축과 거리 기준은 변하지 않습니다. 당해연도는 발표된 최신 월까지의 연중 평균입니다.	 · Axes remain fixed across ranges. The current year averages available non-provisional months.
레짐 히스토리 	Regime history 
클릭해 해당 월 선택	Click to select a month
성장 구성요소 z-score	Growth component z-scores
유동성 구성요소 z-score	Liquidity component z-scores
당시 성과 자산 · 최근 12개월 수익률	Asset performance · Trailing 12-month returns
성장 · 3개 지표 	Growth · 3 indicators 
유동성 · 4개 지표 	Liquidity · 4 indicators 
일곱 지표 계기판	Seven-indicator panel
성장과 자금 흐름의 방향을 확인합니다. 관측 대기는 중립 판정과 다릅니다.	Check growth and funding trends. An unobserved input is not a neutral judgment.
현재 점검월	Current review month
이번 달 관측	Observed this month
최근 일곱 지표 전체 확인	Last fully verified seven-indicator record
정량 모델 자동 갱신 예정: 	Next scheduled quantitative update: 
 · 잠정점은 확정 판정이나 매매 신호가 아닙니다.	 · Provisional points are not confirmed regimes or trading signals.
 · 데이터 산출일 	 · Generated 
 · 일곱 지표 점검월 	 · Seven-indicator review month 
관측 대기	Awaiting observation
확인 완료	Verified
잠정 관측	Provisional observation
관측값 없음	No observation
개선	Improving
악화	Deteriorating
거의 보합	Broadly unchanged
중립(원점 인접)	Neutral (near zero)
전월 평균 	Prior-month mean 
관측 기준 · 	Observed as of · 
마지막 자료 · 	Last available source · 
출처·관측 설명	Sources and observations
이번 달과 비교할 전월 자료가 모두 갖춰지지 않아 방향을 판정하지 않았습니다.	Both comparison months are needed before a direction can be judged.
현재 점검월에 새로 확인된 판독이 없습니다. 이전 기록의 방향을 현재 판단으로 사용하지 않습니다.	There is no newly verified reading for this review month. Earlier directions are not reused as current judgments.
원자료 확인이 완료되지 않아 방향을 판정하지 않았습니다.	Direction remains unjudged until the source observation is verified.
진행 중인 달의 평균입니다. 관측이 추가되면 값과 방향이 달라질 수 있습니다.	This is a month-to-date mean. Additional observations may change its value and direction.
확인된 월간 판독 기록입니다.	A verified monthly reading.
속 빈 점은 이전 신호 이월을 포함한 잠정치입니다. 확정 레짐·자산 성과에는 반영하지 않습니다. / Provisional; excluded from confirmed history and returns.	Hollow points include carried signals. Provisional months are excluded from regime duration, annual means and asset rankings.
현재 자료로 잠정점을 계산할 수 없습니다. / Provisional point unavailable with current inputs.	The current inputs do not support a provisional point.
실제 관측 / Observed: G 	Observed inputs: G 
실제 월중 관측 / Observed this month: 	Observed this month: 
 · 자료 대기 / Data pending	 · Data pending
 · 계산 / Calculated	 · Calculated
 · 잠정 / Provisional	 · Provisional
 · 이월/Carried G	 · Carried G
 이월 / carried: 	 carried: 
 좌표 재계산 / Prior point recalculated: 	 · Prior point recalculated: 
새로 발표된 해당 월 지표 / Newly available inputs: 	Newly released inputs: 
v1→v2 모델 변경으로 전 기간 재계산했습니다. / Model upgrade recalculated the entire history. 	The v1-to-v2 upgrade recalculated the entire history. 
(관측월 재계산; 말일 당시 빈티지 아님 / recalculated observation month, not an as-of month-end vintage).	(Recalculated observation month, not a month-end vintage.)
성장 둔화 속 완화적 유동성 — 금융장세	Slower growth with supportive liquidity
성장·유동성 동반 우호 — 실적장세	Supportive growth and liquidity
성장 둔화 + 긴축 — 역금융·역실적장세	Slower growth and tighter liquidity
성장 견조 속 긴축 — 밸류에이션 조정 국면	Firm growth with tighter liquidity
소비재 · 대형 성장주 (Nifty Fifty)	Consumer businesses · Large growth stocks (Nifty Fifty)
에너지 · 소재 · 산업재 (중국 슈퍼사이클)	Energy · Materials · Industrials (China supercycle)
IT 하드웨어 · 소프트웨어	IT hardware · Software
통신장비 · 인터넷 (닷컴)	Telecom equipment · Internet (dot-com)
필수소비재 · 필수소비재/헬스케어	Consumer staples · Healthcare
방어주 · 필수소비재/헬스케어	Defensive stocks · Consumer staples/Healthcare
방어주 · 필수소비재	Defensive stocks · Consumer staples
경기소비재 · IT 회복	Consumer discretionary · IT recovery
에너지 · 정유/시추	Energy · Refining/Drilling
필수소비재 · 제약	Consumer staples · Pharmaceuticals
금융 · 소비재	Financials · Consumer businesses
헬스케어 · 필수소비재	Healthcare · Consumer staples
IT · 바이오테크	IT · Biotechnology
언택트 성장주 · 반도체	Digital growth stocks · Semiconductors
에너지 · 방산	Energy · Defense
AI 반도체 · 하이퍼스케일러	AI semiconductors · Hyperscalers
경기민감 시클리컬 · 산업재 · 소재 · 경기소비재 · 금융	Cyclicals · Industrials · Materials · Consumer discretionary · Financials
필수소비재 · 유틸리티 · 장기채	Consumer staples · Utilities · Long-duration bonds
성장·유동성 동반 우호. 이익 모멘텀이 높은 시클리컬이 시장을 주도하는 구간	Both axes are supportive; cyclicals with improving earnings often lead.
성장주 · IT/플랫폼 · 바이오 · 금 · 장기채	Growth stocks · IT/Platforms · Biotechnology · Gold · Long-duration bonds
에너지 · 소재 · 은행	Energy · Materials · Banks
실물 성장은 약하나 금리·유동성이 우호. 듀레이션이 긴 성장주와 금이 유리한 구간	Growth is weak but rates and liquidity are supportive; long-duration growth stocks and gold may benefit.
에너지 · 필수소비재 · 헬스케어 · 배당/가치주 · 단기채	Energy · Consumer staples · Healthcare · Dividend/Value stocks · Short-duration bonds
고밸류 성장주 · 리츠 · 장기채	High-valuation growth stocks · REITs · Long-duration bonds
성장은 견조하나 긴축. 밸류에이션 부담이 큰 자산이 조정받고 현금흐름·가치주가 상대 우위	Growth is firm but liquidity is tighter; cash-flow and value stocks may be relatively stronger.
필수소비재 · 헬스케어 · 유틸리티 · 통신 · 현금 · 금	Consumer staples · Healthcare · Utilities · Telecom · Cash · Gold
경기소비재 · 산업재 · 소재 · 하이일드	Consumer discretionary · Industrials · Materials · High yield
성장 둔화 + 긴축. 방어주와 현금성 자산의 상대 방어력이 부각되는 구간	Slower growth and tighter liquidity favor relative resilience in defensive and cash-like assets.
당대 미국 증시 주도주 기록 및 경기 국면별 섹터 로테이션 통념에 기반한 참고 정보이며, 특정 종목의 매수·매도 권유가 아닙니다.	Historical U.S. market leaders and conventional sector patterns are references, not recommendations to trade individual stocks.
잠정 계산월은 성과 자산 순위를 표시하지 않습니다.	Asset rankings are excluded for provisional calculation months.
해당 시점의 자산 수익률 데이터가 없습니다.	Asset returns are unavailable for this date.
주식 내 주도 산업 · 대표 종목	Historical equity leaders · Representative stocks
주식 성과 	Equity return 
국면 섹터 로테이션	regime sector rotation
경계 포함	Includes boundary months
잠정 제외	Excludes provisional months
실측 비중 G 	Observed weight G 
잠정 계산월; 국면 지속·연간 평균·성과 자산 집계 제외	Provisional month; excluded from duration, annual means and asset rankings
원점 인접 — 중립/전환 가능 구간	Near zero — neutral/transition region
한 축이 원점 인접 — 사분면 판정 신뢰도 낮음	One axis is near zero — low confidence in the quadrant assignment
원합성치 	Raw composite 
장기 백분위 	Long-run percentile 
상대·절대 불일치	Relative/raw sign disagreement
표시할 데이터가 없습니다.	No data in this range.
두 달 이상의 자료가 필요합니다.	At least two months are needed.
전체 기간	Full history
축별 확대	Independent axis scaling
성장축	Growth axis
유동성축	Liquidity axis
중립·전환	Neutral/Transition
중립/전환	Neutral/Transition
미확정	Unconfirmed
확정	Confirmed
잠정	Provisional
전월 대비	Month-on-month change
년 연평균	 annual mean
연중 평균(	YTD mean (
까지)	through)
년대	s
개 연도 · 고정축 ±3	years · Fixed axes ±3
계산월	calculated months
월중 잠정 추정	current-month estimate
잠정 추정	provisional estimate
판정 대기	Awaiting judgment
자료 부족	Insufficient observations
마지막 기록	last record
관측	observed
기준축:	Reference axis:
경계(	Boundary (
0 근처)	near zero)
같은 국면	same regime
개월	 months
선택:	Selected:
출처:	Source:
실측	Observed
이월	Carried
성장	Growth
유동성	Liquidity
방어	Defensive
확장	Expansion
조정	Adjustment
중립	Neutral
열위	Less favored
우위	Favored
이동	path
에너지 · 소재	Energy · Materials
평안투	Pyeongantu
월	 month
년	 year
원	 KRW
'''.strip().splitlines())

LABELS={
 'PERMIT':'Building permits YoY','INDPRO':'Industrial production YoY',
 'PAYEMS':'Nonfarm payrolls YoY','UNRATE':'Unemployment 12M change (inverse)',
 'ICSA':'Initial claims YoY (inverse)','M2SL':'M2 YoY','DGS2':'2-year yield 3M change (inverse)',
 'NETLIQ':'Net liquidity (Fed assets − TGA − RRP) YoY','DOLLAR':'Broad dollar YoY (inverse)',
 'T10Y3M':'10Y−3M term spread','BAA10YM':'Baa−10Y credit spread (inverse)','TOTALSL':'Consumer credit YoY',
}

def text(value):
    for ko,en in sorted(COPY.items(),key=lambda item:len(item[0]),reverse=True):value=value.replace(ko,en)
    return value

def english(template,data,book):
    template=template.replace('<html lang="ko">','<html lang="en">')
    # Avoid duplicated Korean/English month labels in the original bilingual UI.
    template=template.replace('pending.month+(inProgress?" · "+monthKo+" 진행 중 / "+monthEn+" in progress":" · 자료 대기 / Data pending")',
                              'pending.month+(inProgress?" · Month in progress":" · Data pending")')
    template=text(template)
    template=template.replace('duration.count+" months"', 'duration.count+(duration.count===1?" month":" months")')
    data=copy.deepcopy(data);book=copy.deepcopy(book)
    for axis in ('g','l'):data['meta'][axis+'_labels']={k:LABELS[k] for k in data['meta'][axis+'_labels']}
    data['meta']['asset_labels']={'spx':'U.S. equities S&P 500','ndx':'Nasdaq','gold':'Gold','wti':'WTI oil','ust':'U.S. Treasury 10Y','cash':'Cash (3M T-Bill)'}
    for month in book['months']:
        for row in month['indicators'].values():
            row['label_ko']=row['label_en'];row['reading_ko']=row.get('reading_en',text(row['reading_ko']))
    return template,data,book
