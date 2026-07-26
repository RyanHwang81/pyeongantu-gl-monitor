# 평안투 GL 레짐 모니터 GitHub Pages 배포 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 첨부된 평안투 GL 레짐 모니터를 RyanHwang81/pyeongantu-gl-monitor Public 저장소와 GitHub Pages에 배포하고, FRED 월간 갱신을 월 1일·16일 09:00 UTC에 자동 반영하며 SignalnFlow 공개 글에 iframe으로 삽입한다.

**Architecture:** 독립 GitHub 저장소의 루트에 제공된 `build.py`, `gl_template.html`, `README.md`를 원본 그대로 보존하고, 제공된 workflow를 `.github/workflows/update.yml`에 배치한다. GitHub Actions가 FRED 데이터로 `dist/`를 재생성·커밋하고 GitHub Pages는 `dist/`를 공개 호스팅한다. SignalnFlow에는 Pages의 HTTPS `index.html`만 iframe으로 넣고 `gl-internal.html` 링크는 만들지 않는다.

**Tech Stack:** Python 3.12, pandas, NumPy, GitHub Actions, GitHub Pages, WordPress REST API.

**Guardrails:** `build.py` 계산 로직(지표·가중치·216개월 롤링 윈도·EMA)과 `gl_template.html` 디자인/레이아웃은 수정하지 않는다. cron `0 9 1,16 * *`를 변경하지 않는다. Public GitHub 저장소와 Pages는 `gl-internal.html`을 완전 비공개로 만들 수 없으므로, 해당 파일의 외부 링크는 만들지 않되 이 제한을 배포 보고에 명시한다.

---

### Task 1: 독립 저장소 및 제공 파일의 무결성 고정

**Files:**
- Create: `build.py`
- Create: `gl_template.html`
- Create: `README.md`
- Create: `.github/workflows/update.yml`
- Create: `dist/index.html`
- Create: `dist/gl-internal.html`
- Create: `docs/superpowers/plans/2026-07-26-gl-regime-monitor-pages-deployment.md`

- [ ] **Step 1: 제공 파일 SHA-256을 기록한다.**

Run:
```bash
shasum -a 256 /Users/hyh/.hermes/webui-mvp/attachments/b3c9762e781d/{build.py,gl_template.html,update.yml,README.md,index.html,gl-internal.html}
```

Expected: `build.py`, `gl_template.html`, `update.yml`, `README.md`의 기준 해시를 확보한다.

- [ ] **Step 2: 제공 파일을 정해진 저장소 경로로 복사한다.**

Run:
```bash
cp /Users/hyh/.hermes/webui-mvp/attachments/b3c9762e781d/build.py build.py
cp /Users/hyh/.hermes/webui-mvp/attachments/b3c9762e781d/gl_template.html gl_template.html
cp /Users/hyh/.hermes/webui-mvp/attachments/b3c9762e781d/README.md README.md
mkdir -p .github/workflows dist
cp /Users/hyh/.hermes/webui-mvp/attachments/b3c9762e781d/update.yml .github/workflows/update.yml
cp /Users/hyh/.hermes/webui-mvp/attachments/b3c9762e781d/index.html dist/index.html
cp /Users/hyh/.hermes/webui-mvp/attachments/b3c9762e781d/gl-internal.html dist/gl-internal.html
```

Expected: 제공된 계산기·템플릿·workflow가 지정 경로에 있고, workflow cron은 `0 9 1,16 * *`이다.

- [ ] **Step 3: 원본 보호 파일의 해시와 workflow schedule을 검증한다.**

Run:
```bash
shasum -a 256 build.py gl_template.html README.md
python3 - <<'PY'
from pathlib import Path
text = Path('.github/workflows/update.yml').read_text(encoding='utf-8')
assert '0 9 1,16 * *' in text
assert 'contents: write' in text
assert 'actions/deploy-pages@v4' in text
print('workflow_contract_ok')
PY
```

Expected: 보호 파일은 제공본과 같고 `workflow_contract_ok`이 출력된다.

- [ ] **Step 4: GitHub Public 저장소를 생성하고 초기 커밋을 push한다.**

Run:
```bash
git init -b main
git add build.py gl_template.html README.md .github/workflows/update.yml dist docs/superpowers/plans
git commit -m "feat: add GL regime monitor dashboard"
gh repo create RyanHwang81/pyeongantu-gl-monitor --public --source=. --remote=origin --push
```

Expected: `https://github.com/RyanHwang81/pyeongantu-gl-monitor`이 Public이고 `main`에 초기 커밋이 존재한다.

### Task 2: 로컬 데이터 빌드 및 출력 계약 검증

**Files:**
- Modify: `dist/index.html` (build.py 생성)
- Modify: `dist/gl-internal.html` (build.py 생성)
- Create: `dist/gl_data.json`

- [ ] **Step 1: 실제 데이터 빌드를 실행한다.**

Run:
```bash
python3 build.py --out dist
```

Expected: `[data] 완료 — NNN개월, 최신 YYYY-MM`와 두 HTML 출력 경로가 나온다.

- [ ] **Step 2: public/internal 분리와 데이터 계약을 확인한다.**

Run:
```bash
python3 - <<'PY'
import json
from pathlib import Path
public = Path('dist/index.html').read_text(encoding='utf-8')
internal = Path('dist/gl-internal.html').read_text(encoding='utf-8')
data = json.loads(Path('dist/gl_data.json').read_text(encoding='utf-8'))
assert '<!--METHOD_START-->' not in public
assert '방법론 · G/L 점수 산식' not in public
assert '방법론 · G/L 점수 산식' in internal
assert data['months'] and data['meta']['latest'] == data['months'][-1]['d']
assert 'gl-height' in public
print('build_contract_ok', len(data['months']), data['meta']['latest'])
PY
```

Expected: `build_contract_ok NNN YYYY-MM`이 출력된다.

- [ ] **Step 3: 생성 결과만 별도 커밋한다.**

Run:
```bash
git add dist
git diff --cached --check
git commit -m "chore: build initial GL dashboard data"
git push origin main
```

Expected: 출력 파일만 포함한 별도 커밋이 remote `main`에 반영된다.

### Task 3: GitHub Pages 및 수동 Actions 실행

**Files:**
- Verify: `.github/workflows/update.yml`
- Verify: `dist/index.html`
- Verify: `dist/gl-internal.html`

- [ ] **Step 1: GitHub Pages를 Actions source로 활성화한다.**

Run:
```bash
gh api --method POST repos/RyanHwang81/pyeongantu-gl-monitor/pages -f build_type=workflow
```

Expected: Pages 설정 응답에 `build_type: workflow`가 포함된다. 이미 활성화된 경우 GET으로 현재 설정을 읽어 검증한다.

- [ ] **Step 2: 워크플로를 수동 실행하고 완료 상태를 기다린다.**

Run:
```bash
gh workflow run 'GL 레짐 모니터 자동 갱신' --repo RyanHwang81/pyeongantu-gl-monitor
gh run list --repo RyanHwang81/pyeongantu-gl-monitor --workflow update.yml --limit 1
gh run watch <run-id> --repo RyanHwang81/pyeongantu-gl-monitor --exit-status
```

Expected: build 및 deploy job이 모두 성공한다.

- [ ] **Step 3: workflow logs와 Git history를 검증한다.**

Run:
```bash
gh run view <run-id> --repo RyanHwang81/pyeongantu-gl-monitor --log
gh api repos/RyanHwang81/pyeongantu-gl-monitor/commits --jq '.[0:5][] | [.sha[0:7], .commit.message] | @tsv'
```

Expected: `[data] 완료 — NNN개월, 최신 YYYY-MM`가 log에 있고, 실제 변경이 있으면 `dist/index.html` 및 `dist/gl-internal.html`을 포함한 자동 커밋이 존재한다. 변경이 없으면 workflow의 `변경 없음` 메시지를 기록한다.

### Task 4: 공개 대시보드 및 iframe 동작 검증

**Files:**
- Verify: `https://ryanhwang81.github.io/pyeongantu-gl-monitor/`

- [ ] **Step 1: HTTPS URL과 Pages artifact를 확인한다.**

Run:
```bash
curl -I --max-time 30 https://ryanhwang81.github.io/pyeongantu-gl-monitor/
```

Expected: HTTPS `200` 및 `text/html` 응답이다.

- [ ] **Step 2: 브라우저로 렌더·차트 점·footer 기준월을 확인한다.**

Expected DOM contract: SVG 4분면 차트가 있고 원형 점이 하나 이상이며 footer의 데이터 기준월이 `dist/gl_data.json`의 `meta.latest`과 일치한다.

- [ ] **Step 3: 상호작용 및 postMessage를 확인한다.**

Expected browser contract: 차트 점 click 뒤 우측 선택월/G·L/자산 랭킹이 변경되고, 레짐 히스토리 ribbon click이 해당 월을 선택한다. 임베드 상위 문서에서 `{type:'gl-height',height:number}` 메시지를 받아 iframe 높이를 변경한다.

### Task 5: SignalnFlow 공개 글 삽입과 공개 검증

**Files:**
- Create: SignalnFlow Korean post `/pyeongantu-gl-regime-monitor/`
- Create: SignalnFlow English counterpart `/en/pyeongantu-gl-regime-monitor/`

- [ ] **Step 1: WordPress REST API 인증·대상 host를 비밀값을 출력하지 않고 확인한다.**

Run:
```bash
python3 - <<'PY'
import os
assert os.environ.get('WORDPRESS_URL') == 'https://signalnflow.com'
assert os.environ.get('WORDPRESS_USERNAME')
assert os.environ.get('WORDPRESS_APP_PASSWORD')
print('wordpress_credentials_present')
PY
```

Expected: `wordpress_credentials_present`가 출력된다.

- [ ] **Step 2: 고객용 짧은 소개문과 public iframe을 KR post 및 EN counterpart에 게시한다.**

Iframe source:
```html
<iframe id="gl-frame" src="https://ryanhwang81.github.io/pyeongantu-gl-monitor/"
        style="width:100%;border:0;height:1700px;display:block;background:#f7f3ec"
        loading="lazy" title="평안투 GL 레짐 모니터"></iframe>
<script>
(function(){
  var f = document.getElementById('gl-frame');
  window.addEventListener('message', function(e){
    if (e.data && e.data.type === 'gl-height' && e.data.height > 400) {
      f.style.height = e.data.height + 'px';
    }
  });
})();
</script>
```

Expected: 글은 공개 상태이고, 내부 파일 직접 링크 또는 `gl-internal.html` 문자열이 공개 본문에 없다.

- [ ] **Step 3: 실제 공개 WordPress DOM에서 iframe HTTPS·script·대시보드 로드 및 모바일 링크 버튼을 확인한다.**

Expected: KR/EN 글은 200, iframe source는 HTTPS, `postMessage` height handler가 존재하고, 모바일 독자가 직접 열 수 있는 public dashboard 링크가 있다.

### Task 6: 운영 기록과 최종 영수증

**Files:**
- Create or Modify: `/Users/hyh/Documents/Obsidian Vault/작업/Hermes운영/SignalnFlow-GL-레짐-모니터-운영.md`

- [ ] **Step 1: 공개 URL, Actions workflow, 월간 schedule, Pages-hosting 결정과 internal-file public limitation을 additive 운영 노트에 기록한다.**

Expected: 서버 cron이 아닌 GitHub Actions가 자동 갱신·배포를 수행한다는 운영 결정이 남는다.

- [ ] **Step 2: 최종 보고에 저장소 URL, workflow run, 기준월, 공개 URL/QA, blog URLs, actual iframe, 다음 실행 일시와 limitation을 포함한다.**

Expected: 모든 완료 주장은 현재 실행의 tool receipt에 근거한다.
