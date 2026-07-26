# 평안투 GL 레짐 모니터

미국 매크로 지표를 성장(Growth) × 유동성(Liquidity) 2축으로 표준화해, 매월 경기·유동성 국면이 4분면 중 어디에 위치하는지 시각화하는 데스크탑 대시보드입니다.

## 산출물

| 파일 | 용도 | 방법론 섹션 |
|---|---|---|
| `dist/index.html` | **공개용** — 외부 배포 | 미포함 |
| `dist/gl-internal.html` | **내부용** — 산식 검토 | 포함 |
| `dist/gl_data.json` | 계산 결과 원본 (재사용·검증용) | — |

두 파일 모두 외부 의존성 없는 단일 HTML입니다. 폰트만 Google Fonts에서 로드하며, 그 외 데이터·차트·로직이 파일 안에 모두 들어 있어 더블클릭만으로 열립니다.

## 자동 갱신 구조

FRED는 브라우저 직접 호출(CORS)을 허용하지 않으므로, **서버 측에서 재빌드 후 배포**하는 방식으로 자동화합니다.

```
GitHub Actions (매월 1일·16일)
   └─ build.py 실행
        ├─ FRED 15개 시계열 + 주가/금 시세 다운로드
        ├─ G/L 점수 재계산 (신규 월 자동 추가)
        └─ index.html / gl-internal.html 재생성
   └─ GitHub Pages 자동 배포
```

신규 월 데이터가 확정되면 차트에 dot이 하나 자동으로 추가되고, 상단 KPI·레짐 히스토리·자산 랭킹이 함께 갱신됩니다.

> **월 2회 실행 이유** — PERMIT·INDPRO·PAYEMS 등 주요 지표는 해당 월 종료 후 2~3주 뒤 발표됩니다. 1일에만 실행하면 직전 월이 아직 미발표 상태일 수 있어, 16일에 한 번 더 실행해 누락을 방지합니다. 신규 데이터가 없으면 커밋 없이 종료합니다.

## 설치 및 실행

### 로컬 실행

```bash
pip install pandas numpy
python3 build.py --out dist
```

옵션:

- `--out DIR` 출력 디렉터리 (기본 `dist`)
- `--template PATH` 템플릿 경로 (기본 `gl_template.html`)
- 환경변수 `GL_CACHE=/path` 지정 시 원본 CSV를 해당 폴더에 캐시 (재실행 시 네트워크 부하 감소)

### GitHub Actions / GitHub Pages 설정

1. 이 폴더의 파일을 저장소 루트에 커밋 (`build.py`, `gl_template.html`, `README.md`)
2. `update.yml` 을 `.github/workflows/update.yml` 경로로 저장
3. 저장소 **Settings → Pages → Source** 를 `GitHub Actions` 로 설정
4. build job은 `self-hosted, macOS, gl-monitor` runner에서 실행합니다.  
   FRED graph CSV가 GitHub-hosted runner IP에서 반복 timeout 되어 owner Mac mini runner를 사용합니다. 계산 로직(`build.py`)은 변경하지 않습니다.
5. **Actions** 탭에서 `Run workflow` 로 첫 실행 → 이후 매월 1일·16일 09:00 UTC 자동 반복

공개 URL은 `https://ryanhwang81.github.io/pyeongantu-gl-monitor/` 이며 `index.html`(공개용)만 iframe/블로그에 노출합니다.  
`gl-internal.html`은 방법론 검토용이며 공개 글/iframe/버튼에서 링크하지 않습니다.

## 데이터 소스

**성장 (G)** — PERMIT 건축허가, INDPRO 산업생산, PAYEMS 비농업고용, UNRATE 실업률, ICSA 신규실업수당청구

**유동성 (L)** — M2SL 통화량, T10Y3M 기간스프레드, FEDFUNDS 기준금리, TOTALSL 소비자신용, BAA10YM 신용스프레드, WALCL 연준 총자산

**자산 수익률** — S&P500(Shiller 월간), 나스닥(NASDAQCOM), 금(LBMA), WTI, 미국채 10Y(GS10 듀레이션 근사), 현금(TB3MS)

모든 FRED 시계열은 St. Louis Fed 공개 데이터이며 API 키가 필요 없습니다.

## 산식 요약

```
z      = clamp((x − μ₂₁₆) / σ₂₁₆, −3, +3)      최소 48개월 확보 시 산출
raw    = Σ wᵢ·zᵢ / Σ wᵢ                        결측 시 가용 가중치로 재정규화
G, L   = EMA₃[ Z₂₁₆(raw) ]                     합성지수 자체를 재표준화 후 3개월 평활
```

레짐은 두 축의 부호로 결정하며, 동일 4분면 2개월 연속 또는 두 축 모두 |0.25| 초과 시 "확정"으로 표기합니다. 원점 인접 구간(|G|<0.15 & |L|<0.15)은 중립/전환으로 별도 표시합니다.

## 고지

본 자료는 정보 제공 목적이며 특정 상품의 매매 권유 또는 투자 자문이 아닙니다. 주도 산업·종목 및 섹터 로테이션 정보는 과거 기록과 일반적 통념에 기반한 참고 자료입니다.
