# Synchronized Seven-Gauge Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On every GL build, refresh a sourced current-month seven-gauge snapshot with explicit pending states instead of presenting June as current.

**Architecture:** Keep `book_dashboard.json` as immutable authored history. Derive a new in-memory current-month snapshot from that history and three FRED credit/FX/dollar series (no proxies for the other four); render with the quantitative model as one release. Unverified inputs never become arrows.

**Tech Stack:** Existing Python 3/pandas/stdlib FRED client, GitHub Actions, inline HTML/JS, unittest, Chrome CDP.

---

### Task 1: Snapshot contract and RED tests

**Files:** Modify `tests/test_book_dashboard.py`, `build.py`.

- [ ] Add a test that constructs an authored June month plus fixture FRED daily series for August/September. Assert a September row with all seven keys, 3 provisional sourced readings with exact as-of and 4 pending readings with no June arrows; assert authored June unchanged.
- [ ] Add boundary cases: no September observation => pending; no preceding month => pending; one source exception => pending plus error receipt; complete and partial judgment states remain distinct.
- [ ] Run `python3 -m unittest discover -s tests -p 'test_book_dashboard.py' -v` and confirm new tests fail because the snapshot function is missing.

### Task 2: Small deterministic collector and publication gate

**Files:** Modify `build.py`, `gl_template.html`, `README.md`, `.github/workflows/update.yml`.

- [ ] In `build.py`, implement `refresh_book_dashboard(book, target_month, sources)` without mutating the authored dict. Source mapping: `DTWEXBGS` -> dollar trend, `BAMLH0A0HYM2` -> HY spread, `DEXKOUS` -> USD/KRW. Resample source series to month-start averages, compare current month to preceding month; increasing sources have `direction='up'`, `effect='down'`; decreasing sources have `direction='down'`, `effect='up'`; equal -> flat. Set `state='provisional'`, `as_of` to last raw sample date, official FRED `source_url`, and actual monthly means in bilingual reading. For missing inputs set `pending` and retain last source/as-of only as explicitly labeled last check.
- [ ] Build quantitative data first, then load authored history, collect the three official FRED series using the existing FRED client, derive snapshot, and render both HTML files atomically to the output directory; log confirmed/provisional/pending counts and errors. Do not write the snapshot to `book_dashboard.json`.
- [ ] In `gl_template.html`, show current `month.date` + observed `N/7` + authored latest confirmed month. Display `확인일` for sourced readings, `미확인 · 마지막 기록` for pending. No selected/confirmed regime claim while pending.
- [ ] In README and workflow test naming, specify synchronous refresh and pending release rule.
- [ ] Run the focused tests, full unittest suite, py_compile, and `git diff --check`.

### Task 3: Build, QA and deploy

**Files:** Generated `dist/index.html`, `dist/gl-internal.html`, `dist/gl_data.json`; vault `log.md` (operating decision only).

- [ ] Preserve existing `dist/` in a timestamped scratch backup. Run a real FRED API build into scratch with certifi CA, validate month, 7 keys, as-of/source URLs, pending count, and unchanged authored June.
- [ ] Render to `dist/`; verify a 360/390/1440px local Chrome CDP matrix: no overflow, all 7 gauge rows, 9월 snapshot, pending vs verified labeling, no console errors. Inspect screenshots.
- [ ] Commit only relevant files, push `main`, dispatch the already-defined committed-dist deployment, wait for green; compare public Pages HTML/JSON SHA-256 and inspect Korean/English WordPress iframe on mobile/desktop.
- [ ] Append the rule, receipts, limitations and rollback backup to vault `log.md`; report observed vs pending indicators rather than calling 7/7 confirmed.
