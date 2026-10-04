# GL model v2 실데이터 검증

검증일: 2026-10-04. FRED 최신 공개 자료를 같은 입력으로 사용한 v1/v2 비교. 월간 평균 및 현재 공개된 개정 자료를 쓰므로 과거 정보집합의 실시간 빈티지 백테스트가 아니다. 가중치·임계값은 지시서대로 유지했다.

## 최근 24개월 비교

|월|v1 G|v1 L|v1 국면|v2 G|v2 L|v2 국면|p|b|x|
|---|---:|---:|---|---:|---:|---|---|---|---|
|2024-10|-0.075|-0.646|defense|-0.075|+0.121|liquidity|False|False|True|
|2024-11|-0.059|-0.324|defense|-0.059|-0.319|defense|False|True|False|
|2024-12|-0.028|-0.247|defense|-0.028|-0.816|defense|False|True|False|
|2025-01|-0.014|-0.244|defense|-0.014|-0.917|defense|False|True|False|
|2025-02|-0.033|-0.365|defense|-0.033|-0.804|defense|False|True|False|
|2025-03|+0.012|-0.483|adjustment|+0.012|-0.586|adjustment|False|True|True|
|2025-04|+0.027|-0.610|adjustment|+0.027|-0.277|adjustment|False|True|True|
|2025-05|+0.020|-0.643|adjustment|+0.020|-0.161|adjustment|False|True|True|
|2025-06|+0.003|-0.641|adjustment|+0.003|-0.086|adjustment|False|False|True|
|2025-07|+0.032|-0.615|adjustment|+0.032|-0.099|adjustment|False|False|True|
|2025-08|-0.025|-0.617|defense|-0.025|-0.014|defense|False|False|False|
|2025-09|+0.036|-0.535|adjustment|+0.036|+0.081|expansion|False|False|True|
|2025-10|+0.049|-0.424|adjustment|+0.049|+0.170|expansion|False|True|True|
|2025-11|-0.001|-0.288|defense|-0.001|+0.133|liquidity|False|False|True|
|2025-12|+0.006|-0.010|adjustment|+0.006|+0.294|expansion|False|True|True|
|2026-01|-0.037|+0.140|liquidity|-0.037|+0.413|liquidity|False|True|False|
|2026-02|+0.019|+0.145|expansion|+0.019|+0.555|expansion|False|True|False|
|2026-03|-0.072|+0.032|liquidity|-0.072|+0.333|liquidity|False|True|True|
|2026-04|-0.045|-0.011|defense|-0.045|+0.210|liquidity|False|True|True|
|2026-05|-0.018|+0.053|liquidity|-0.018|+0.067|liquidity|False|False|True|
|2026-06|-0.031|+0.102|liquidity|-0.031|-0.034|defense|False|False|False|
|2026-07|-0.006|+0.136|liquidity|-0.006|-0.110|defense|False|False|True|
|2026-08|+0.028|+0.223|expansion|+0.028|+0.023|expansion|False|False|True|
|2026-09|-0.010|+0.299|liquidity|+0.044|-0.024|adjustment|True|False|False|

## 2026-09 L 분해

|지표|지표 z|가중치|z×w|이월 원월|
|---|---:|---:|---:|---|
|M2SL|-0.133587|0.18|-0.024046|2026-08|
|DGS2|-1.257275|0.18|-0.226310|실측|
|NETLIQ|-0.361035|0.15|-0.054155|실측|
|DOLLAR|+0.477838|0.15|+0.071676|실측|
|T10Y3M|-0.298463|0.12|-0.035816|실측|
|BAA10YM|+1.240556|0.12|+0.148867|실측|
|TOTALSL|-0.387201|0.10|-0.038720|2026-07|

합산 원합성치 lr=-0.159, 상대 L=-0.024, 장기 백분위 48%. G=+0.044, gr=+0.064, gp=46%. ow={'g': 0.48, 'l': 0.72}; cf={'PERMIT': '2026-08', 'INDPRO': '2026-08', 'M2SL': '2026-08', 'TOTALSL': '2026-07'}; p=True, n=True, b=False, c=False, x=False.

8월 TOTALSL도 7월 신호를 이월하며 원 가중치를 유지한다. 8월 p=False, 9월 p=True. 구성요소 7개 모두 최신월 z가 유효하므로 z 가중치 합은 1.00.

## 1972년 이후 사분면 부호 분포

아래는 동일 657개 계산 행 모두를 사용한다. 잠정·중립·경계까지 포함한 부호 분포이며 확정 국면 빈도라는 뜻이 아니다.

|부호 국면|v1 %|v2 %|
|---|---:|---:|
|expansion|35.77|35.77|
|liquidity|16.74|16.44|
|adjustment|28.16|28.31|
|defense|19.33|19.48|

행 수 v1=657, v2=657. v2 잠정 16, 중립 15, 경계 142. 연간 평균·지속·선택월 자산 순위는 잠정 행 제외.

## 역사 국면 검증

|월|v1 L|v2 G|v2 L|직전월 v2 L|전월 차이|p|b|
|---|---:|---:|---:|---:|---:|---|---|
|2008-10|+1.863|-2.926|+0.711|-0.121|+0.832|False|False|
|2020-04|+2.018|-1.430|+2.381|+1.763|+0.618|False|False|
|2022-06|-0.121|+0.695|-1.261|-0.905|-0.356|False|False|
|2023-03|-2.390|-0.068|-1.588|-1.582|-0.006|False|True|

2008-10과 2020-04는 L이 양수다. 위기와 정책 완화가 동시에 나타날 수 있는 유동성 모형이며 신용위기의 존재를 직접 판정하는 모형이 아니다. 결과를 음수로 만들기 위한 튜닝은 하지 않았다.

## 실제 전제 차이와 보수적 처리

- 첨부는 9월 b=True를 예상했지만 v2는 두 축 모두 |0.15| 미만이므로 기존 n 규칙이 우선한다. n=True, b=False, c=False를 유지했다.
- TOTALSL 최신 실제 관측은 2026-07이므로 이월 원월은 8월로 만들어 쓰지 않았다.
- RRPONTSYD 시작은 2003-02이며 NETLIQ 시작은 2002-12. 시작 전 0은 해당 시계열에 대한 모델 가정이다. 모든 역레포 제도 부재를 뜻하지 않는다. 시작 후 원본 결측은 채우지 않았다.
- 2006년 DOLLAR YoY 초기 12개월과 NETLIQ 원본 결측 때문에 L 실측 비중 0.70으로 월간 잠정 행 12개가 남는다. 잠정 제외 규칙에 따라 2006년 연간 평균 점은 없으며 2000년대 차트는 9개 연도를 표시한다. 전체 월간 행 수는 유지된다.
- 이월값은 지표와 합성치 양쪽의 롤링 통계에서 제외했다. 진행월 별도 추정점은 기존처럼 낮은 관측 비중도 표시하고 월간 역사에는 넣지 않는다.
- 원합성치도 지표 z의 평균이므로 경제적 절대 측정치로 설명하지 않았다. x는 상대 점수와 원합성치의 반대 부호 표시다.
- 기존 September fixture의 v1_compare도 기준월 이전으로 잘라 새 메타데이터가 fixture의 날짜 부재 검사를 오염시키지 않도록 했다.

## 검증 영수증

- Python unittest 29개 PASS: 기존 20개와 신규 9개. task venv pandas 2.3.3 및 Actions와 동일한 /usr/local/bin/python3 pandas 3.0.2에서 확인.
- 실제 CLI `python3 build.py --out dist` PASS, 657 months, latest 2026-09, G=+0.044 L=-0.024, FRED source_errors={} (키는 로그/산출물 비노출).
- KO/EN 360·390·768·1024·1440px 10 화면 PASS. 가로 넘침·스크립트 오류 없음; 잠정 속 빈 점·자산 순위 제외·연간 필터·키보드 선택·경계 fixture·36행 내부 비교 검증. 영문 화면의 한국어 문자 0개.
- 기존 일곱 계기판 5폭 QA PASS. book_dashboard.json 및 workflow 스케줄/러너 원본 바이트 동일.
- 같은 657개월 자산 수익률 값과 기존 수익률 산식 블록은 v1/v2 완전히 동일. v1_compare 최근 36행도 동시 수집 v1 결과와 동일.
- 원합성치·백분위·이월 원월·FRED URL·표 가중치·단위를 보존하며 KR/EN 문구를 읽고 중복 잠정/미확정 문구 및 영문 단수 month를 국소 수정.

공식 자료: [DGS2](https://fred.stlouisfed.org/series/DGS2), [DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS), [WTREGEN](https://fred.stlouisfed.org/series/WTREGEN), [RRPONTSYD](https://fred.stlouisfed.org/series/RRPONTSYD), [WALCL](https://fred.stlouisfed.org/series/WALCL).
