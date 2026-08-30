# Recent and Long-Term GL Charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the quantitative GL monitor into a recent monthly movement view and a long-term annual fixed-axis regime map without changing calculations, URLs, or deployment cadence.

**Architecture:** Keep the single-file HTML architecture and existing Python build pipeline. Replace the mode-switching chart with two cards backed by one shared SVG renderer: recent monthly presets use an equal-unit zoomed range, while annual presets always use `[-3, 3]`. Preserve the existing selection panel, ribbon, public/internal rendering boundary, and iframe height messaging.

**Tech Stack:** Python 3 `unittest`, vanilla HTML/CSS/JavaScript/SVG, headless Google Chrome, GitHub Actions/Pages.

---

### Task 1: Lock the public chart-split contract with a failing test

**Files:**
- Modify: `tests/test_book_dashboard.py`

- [ ] **Step 1: Add a template contract test**

Add a test that requires:

```python
def test_quant_reference_separates_recent_movement_from_long_term_map(self):
    template = TEMPLATE.read_text(encoding="utf-8")
    self.assertIn('id="recent-quad"', template)
    self.assertIn('id="annual-quad"', template)
    self.assertIn("최근 12개월 이동", template)
    self.assertIn("장기 GL 국면 지도", template)
    self.assertIn("확대 보기 · 장기 국면 지도와 이동 거리를 직접 비교하지 않습니다", template)
    self.assertIn("연간 평균 · 고정축 ±3", template)
    self.assertIn('id="recent-period-sel"', template)
    self.assertIn('id="annual-period-sel"', template)
    self.assertNotIn('id="mode-seg"', template)
    self.assertNotIn('id="fit-btn"', template)
    self.assertNotIn('id="full-btn"', template)
```

Also require accessibility and preserved integration markers:

```python
self.assertIn('aria-label="최근 이동 기간"', template)
self.assertIn('aria-label="장기 국면 기간"', template)
self.assertIn('role:"button"', template)
self.assertIn('"gl-height"', template)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
python3 -m unittest tests.test_book_dashboard.BookDashboardContractTests.test_quant_reference_separates_recent_movement_from_long_term_map -v
```

Expected: FAIL because `recent-quad` and the new copy do not exist.

- [ ] **Step 3: Commit the RED test separately**

```bash
git add tests/test_book_dashboard.py
git diff --cached --check
git commit -m "test: define split GL trajectory contract"
```

### Task 2: Implement the two-card UI and shared SVG renderer

**Files:**
- Modify: `gl_template.html:47-204`
- Modify: `gl_template.html:282-344`
- Modify: `gl_template.html:544-715`

- [ ] **Step 1: Replace the single chart controls and markup**

Create:

- `#recent-period-sel` with monthly `12/24개월` presets
- `#movement-summary` with `#delta-g`, `#delta-l`, `#delta-g-note`, `#delta-l-note`
- `#recent-chartbox`, `#recent-quad`, `#recent-tooltip`, `#recent-status`
- `#annual-period-sel` with full-history and decade presets
- `#annual-chartbox`, `#annual-quad`, `#annual-tooltip`, `#annual-status`

Keep the right selection card and history ribbon unchanged.

- [ ] **Step 2: Add purpose-specific CSS**

Add styles for:

```css
.chart-card + .chart-card{margin-top:16px}
.chart-purpose{font-size:12px;color:var(--ink2);line-height:1.6;padding:8px 22px 0}
.movement-summary{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 22px 0}
.delta-card{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:11px 13px}
.delta-card .delta-value{font-family:var(--mono);font-size:18px;font-weight:700}
.chartbox{padding:10px 14px 6px;position:relative}
.quad{width:100%;height:auto;display:block}
.tooltip{position:absolute;pointer-events:none;...}
.neutral-label{font-size:9.5px;fill:var(--ink3);font-family:var(--mono)}
```

At `max-width:720px`, stack `.movement-summary` and keep controls readable. At `max-width:360px`, reduce chart padding and status-bar gaps without creating page overflow.

- [ ] **Step 3: Replace mode state with purpose-specific presets**

Use:

```javascript
const RECENT_PRESETS = [
  {id:"m12",label:"최근 12개월",get:()=>M.slice(-12)},
  {id:"m24",label:"최근 24개월",get:()=>M.slice(-24)}
];
const ANNUAL_PRESETS = [
  {id:"all",label:"전체 기간",get:()=>ANN},
  ...decades.slice().reverse().map(d0=>({
    id:"d"+d0,label:d0+"년대",get:()=>ANN.filter(a=>+a.d>=d0&&+a.d<d0+10)
  }))
];
let state={recentPreset:"m12",annualPreset:"all",selected:latest,ready:false};
```

- [ ] **Step 4: Implement equal-unit recent axes and fixed annual axes**

Retain `niceRange()` and add:

```javascript
function equalUnitRanges(points){
  const gx=niceRange(points.map(m=>m.g),true);
  const ly=niceRange(points.map(m=>m.l),true);
  const span=Math.max(gx[1]-gx[0],ly[1]-ly[0]);
  const expand=range=>{
    const center=(range[0]+range[1])/2;
    return [center-span/2,center+span/2];
  };
  return {gx:expand(gx),ly:expand(ly)};
}
```

The annual renderer must pass `{gx:[-3,3], ly:[-3,3]}`. Both SVGs use a square plot area so one z-score unit has the same visual length on both axes.

- [ ] **Step 5: Implement a shared renderer with explicit label strategies**

Create `renderQuadrant(svgId, tooltipId, points, options)` that:

- draws quadrant tints, grid, zero axes, and the neutral `±0.15` square
- uses a dashed path and age-based dot opacity for monthly points
- uses a solid path for annual points
- labels recent start, quarter-end months, and current month
- labels annual decade views by year and full history every five years plus latest
- selects a month on click, Enter, or Space
- adds `role="button"`, `tabindex="0"`, and a meaningful `aria-label` to each point

- [ ] **Step 6: Implement movement summary**

Use the first and last recent points:

```javascript
function describeDelta(value){
  if(Math.abs(value)<.10)return "거의 보합";
  return value>0?"개선":"악화";
}
```

Show start/end values and deltas. If fewer than two points exist, show `자료 부족` and avoid throwing.

- [ ] **Step 7: Wire both render paths and preserved selection behavior**

Implement `renderRecent()` and `renderAnnual()`. `selectMonth()` rerenders both charts and the ribbon so selection markers stay synchronized. Clicking an annual point selects that year's last available month.

- [ ] **Step 8: Run the focused test and full unit suite**

Run:

```bash
python3 -m unittest tests.test_book_dashboard.BookDashboardContractTests.test_quant_reference_separates_recent_movement_from_long_term_map -v
python3 -m unittest discover -s tests -v
```

Expected: focused PASS; full suite reports zero failures.

- [ ] **Step 9: Commit the template implementation**

```bash
git add gl_template.html tests/test_book_dashboard.py
git diff --cached --check
git commit -m "feat: separate recent and long-term GL charts"
```

### Task 3: Regenerate committed public artifacts and verify build contracts

**Files:**
- Modify: `dist/index.html`
- Modify: `dist/gl-internal.html`
- Modify: `dist/gl_data.json` only if the source build returns changed data

- [ ] **Step 1: Rebuild with the existing local data/cache path**

Run the canonical build:

```bash
python3 build.py --out dist
```

If a transient upstream fetch fails, retry with the configured local `GL_CACHE` used by the self-hosted runner; do not change calculations or fabricate output.

- [ ] **Step 2: Verify generated contracts**

Run a Python verifier that asserts:

```python
assert 'id="recent-quad"' in public
assert 'id="annual-quad"' in public
assert "최근 12개월 이동" in public
assert "장기 GL 국면 지도" in public
assert "방법론 · G/L 점수 산식" not in public
assert "방법론 · G/L 점수 산식" in internal
assert "__GL_DATA__" not in public
assert "__BOOK_DATA__" not in public
assert data["meta"]["latest"] == data["months"][-1]["d"]
```

- [ ] **Step 3: Commit generated artifacts**

```bash
git add dist/index.html dist/gl-internal.html dist/gl_data.json
git diff --cached --check
git commit -m "chore: rebuild split GL monitor views"
```

### Task 4: Local DOM, interaction, and responsive visual QA

**Files:**
- Verify: `dist/index.html`
- Create receipts outside Git under `/tmp/pyeongantu-gl-chart-qa/`

- [ ] **Step 1: Run a local static server**

```bash
python3 -m http.server 8765 --directory dist
```

Run it in the background and verify `http://127.0.0.1:8765/` returns HTTP 200.

- [ ] **Step 2: Run headless DOM assertions**

Use headless Chrome/CDP to open the page, expand the quantitative details, and assert:

- `#recent-quad` and `#annual-quad` each contain path and point nodes
- recent status contains `12개월`
- annual status contains `고정축 ±3`
- changing recent select to `m24` updates status to `24개월`
- changing annual select to a decade updates its status
- keyboard activation of a point updates `#status-sel`
- no console errors

- [ ] **Step 3: Capture desktop and mobile screenshots**

Capture at `1440x1000`, `390x844`, and `360x800`, including the exact quantitative section after opening it.

- [ ] **Step 4: Inspect screenshots and width metrics**

Require:

```javascript
document.documentElement.scrollWidth === document.documentElement.clientWidth
```

Verify visually that:

- recent and annual cards are clearly separate
- purpose/scale copy is readable
- monthly labels do not pile into one unreadable stack
- the long-term chart keeps fixed axes and readable year labels
- selection panel remains legible

- [ ] **Step 5: Run design and source checks**

```bash
cd /Users/hyh/.hermes/designmd-cli
npm exec design.md -- lint /Volumes/T7/Workspace/signalflow/DESIGN.md
cd /Volumes/T7/Workspace/pyeongantu-gl-monitor
python3 -m unittest discover -s tests -v
git diff --check
```

Report existing DESIGN.md warnings separately; require zero lint errors and zero test failures.

### Task 5: Push, deploy, and verify GitHub Pages

**Files:**
- Push: feature branch
- Merge: `main` after checks

- [ ] **Step 1: Review intended diff and repository status**

```bash
git status --short --branch
git diff main...HEAD --stat
git diff main...HEAD --check
```

Only spec, plan, test, template, and generated `dist` artifacts may be included.

- [ ] **Step 2: Push and create a PR**

```bash
git push -u origin feat/separate-gl-trajectories
gh pr create --base main --head feat/separate-gl-trajectories --title "feat: split recent and long-term GL charts" --body-file /tmp/pyeongantu-gl-pr.md
```

Include test and visual QA receipts in the PR body.

- [ ] **Step 3: Inspect checks and merge**

```bash
gh pr checks --watch --fail-fast
gh pr merge --squash --delete-branch
```

If no PR checks exist, state that and rely on the recorded local gates before merging.

- [ ] **Step 4: Run the Pages workflow and wait for success**

```bash
gh workflow run update.yml --repo RyanHwang81/pyeongantu-gl-monitor
RUN_ID=$(gh run list --repo RyanHwang81/pyeongantu-gl-monitor --workflow update.yml --limit 1 --json databaseId --jq '.[0].databaseId')
test -n "$RUN_ID"
gh run watch "$RUN_ID" --repo RyanHwang81/pyeongantu-gl-monitor --exit-status
```

Expected: build and deploy jobs succeed.

### Task 6: Live read-back, blog embed QA, and durable operation note

**Files:**
- Verify: `https://ryanhwang81.github.io/pyeongantu-gl-monitor/`
- Verify: `https://signalnflow.com/pyeongantu-gl-regime-monitor/`
- Modify or Create: `/Users/hyh/Documents/Obsidian Vault/작업/Hermes운영/SignalnFlow-GL-레짐-모니터-운영.md`

- [ ] **Step 1: Verify canonical and cache-busted Pages HTML**

Require HTTP 200 and the new markers/copy on both canonical and `?v=<commit>` URLs. Confirm retired single-chart controls are absent from rendered DOM.

- [ ] **Step 2: Verify the WordPress iframe surface**

Require:

- article HTTP 200
- iframe source remains the canonical Pages URL
- direct monitor renders both chart cards
- iframe height messaging still updates the embedded height
- no `gl-internal` public link

- [ ] **Step 3: Run live desktop/mobile browser QA**

Repeat `1440`, `390`, and `360` width/console/screenshot checks against the canonical live URL. Treat cache or one viewport failure as partial, not PASS.

- [ ] **Step 4: Append the durable decision and receipts**

Record additively:

- the split purpose (`최근 이동` vs `장기 국면`)
- unchanged calculations, URLs, and schedule
- repository/PR/workflow references
- live URLs and checked viewports
- known limitations or DESIGN.md warnings

- [ ] **Step 5: Final verification gate**

Re-run:

```bash
python3 -m unittest discover -s tests -v
git status --short --branch
RUN_ID=$(gh run list --repo RyanHwang81/pyeongantu-gl-monitor --workflow update.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN_ID" --repo RyanHwang81/pyeongantu-gl-monitor --json status,conclusion,url,headSha
```

Only report completion when tests are green, GitHub Pages deployment concludes successfully, and canonical live read-back contains both chart views.
