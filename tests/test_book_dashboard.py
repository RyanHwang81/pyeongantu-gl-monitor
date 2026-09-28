import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import build


ROOT = Path(__file__).resolve().parents[1]
BOOK_DATA = ROOT / "book_dashboard.json"
TEMPLATE = ROOT / "gl_template.html"


class BookDashboardContractTests(unittest.TestCase):
    def test_checked_in_seed_has_exactly_seven_chapter_27_indicators(self):
        dashboard = build.load_book_dashboard(BOOK_DATA)
        latest = dashboard["months"][-1]
        indicators = latest["indicators"]

        growth = [item for item in indicators.values() if item["axis"] == "growth"]
        liquidity = [item for item in indicators.values() if item["axis"] == "liquidity"]

        self.assertEqual(3, len(growth))
        self.assertEqual(4, len(liquidity))
        self.assertEqual(
            {
                "global_manufacturing_pmi",
                "korea_semiconductor_exports",
                "leading_industry_earnings_revision",
                "fed_next_move_expectation",
                "dollar_index_trend",
                "high_yield_spread",
                "usdkrw_position",
            },
            set(indicators),
        )
        for item in indicators.values():
            self.assertIn(item["direction"], {"up", "down", "flat", "pending"})
            self.assertIn(item["effect"], {"up", "down", "flat", "pending"})
            self.assertIn(item["state"], {"confirmed", "provisional", "pending"})
            self.assertTrue(item["source_name"])
            self.assertRegex(item["as_of"], r"^\d{4}-\d{2}")

    def test_manuscript_seed_judges_2026_06_as_selection(self):
        dashboard = build.load_book_dashboard(BOOK_DATA)
        result = build.judge_book_month(dashboard["months"][-1], previous_regime=None)

        self.assertEqual("up", result["growth"])
        self.assertEqual("down", result["liquidity"])
        self.assertEqual("selection", result["regime"])
        self.assertEqual("confirmed", result["status"])
        self.assertFalse(result["held_previous"])
        self.assertEqual(7, result["complete_count"])

    def test_tie_breakers_use_earnings_revision_and_credit_spread(self):
        month = {
            "date": "2026-07",
            "indicators": {
                "global_manufacturing_pmi": {"axis": "growth", "effect": "up", "state": "confirmed"},
                "korea_semiconductor_exports": {"axis": "growth", "effect": "down", "state": "confirmed"},
                "leading_industry_earnings_revision": {"axis": "growth", "effect": "up", "state": "confirmed"},
                "fed_next_move_expectation": {"axis": "liquidity", "effect": "up", "state": "confirmed"},
                "dollar_index_trend": {"axis": "liquidity", "effect": "down", "state": "confirmed"},
                "high_yield_spread": {"axis": "liquidity", "effect": "down", "state": "confirmed"},
                "usdkrw_position": {"axis": "liquidity", "effect": "up", "state": "confirmed"},
            },
        }

        result = build.judge_book_month(month, previous_regime="expansion")
        self.assertEqual("up", result["growth"])
        self.assertEqual("down", result["liquidity"])
        self.assertEqual("selection", result["regime"])

    def test_incomplete_or_unresolved_month_holds_previous_regime(self):
        dashboard = build.load_book_dashboard(BOOK_DATA)
        month = json.loads(json.dumps(dashboard["months"][-1]))
        month["date"] = "2026-07"
        month["indicators"]["leading_industry_earnings_revision"].update(
            {"effect": "pending", "direction": "pending", "state": "pending"}
        )

        result = build.judge_book_month(month, previous_regime="selection")
        self.assertEqual("selection", result["regime"])
        self.assertEqual("provisional", result["status"])
        self.assertTrue(result["held_previous"])
        self.assertEqual(6, result["complete_count"])

    def test_invalid_indicator_direction_is_rejected(self):
        payload = json.loads(BOOK_DATA.read_text(encoding="utf-8"))
        payload["months"][0]["indicators"]["global_manufacturing_pmi"]["effect"] = "strongly_up"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid effect"):
                build.load_book_dashboard(path)

    def test_public_render_leads_with_quant_and_keeps_seven_dated_gauges(self):
        dashboard = build.load_book_dashboard(BOOK_DATA)
        legacy = {
            "meta": {
                "generated": "2026-08-16",
                "latest": "2026-07",
                "g_weights": {},
                "l_weights": {},
                "g_labels": {},
                "l_labels": {},
                "asset_labels": {},
            },
            "months": [],
        }
        html = build.render(TEMPLATE.read_text(encoding="utf-8"), legacy, dashboard, public=True)

        self.assertIn('id="book-dashboard"', html)
        visible = html.split("<script>", 1)[0]
        self.assertNotIn("원고 준거 월간 판독", visible)
        self.assertNotIn("이번 달 세 줄 기록", visible)
        self.assertNotIn("부록 3 · 역사적 대표 이동 경로", visible)
        self.assertNotIn("id=\"book-regime-name\"", visible)
        self.assertIn("마지막 확인", visible)
        self.assertIn("글로벌 제조업 PMI", html)
        self.assertIn("한국 반도체 수출", html)
        self.assertIn("주도 산업 이익 전망치", html)
        self.assertIn("연준의 다음 행보 기대", html)
        self.assertIn("달러인덱스 추세", html)
        self.assertIn("하이일드 스프레드", html)
        self.assertIn("원달러 환율 위치", html)
        self.assertIn("미국 매크로 정량 보조모델", html)
        self.assertLess(html.index('id="quant-reference"'), html.index('id="book-dashboard"'))
        self.assertNotIn("gl-internal", html)
        self.assertNotIn("__BOOK_DATA__", html)
        self.assertNotIn("__GL_DATA__", html)

    def test_narrow_embed_stacks_header_before_title_is_squeezed(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("@media(max-width:360px)", template)
        self.assertIn("header{flex-direction:column", template)
        self.assertIn(".hdr-right{width:100%;flex-direction:row", template)

    def test_quant_reference_separates_recent_movement_from_long_term_map(self):
        template = TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('id="recent-quad"', template)
        self.assertIn('id="annual-quad"', template)
        self.assertIn("최근 12개월 이동", template)
        self.assertIn("장기 GL 국면 지도", template)
        self.assertIn(
            "축별 확대 · X/Y 축 범위가 서로 다르며 장기 국면 지도와 이동 거리를 직접 비교하지 않습니다",
            template,
        )
        self.assertIn("연간 평균 · 고정축 ±3", template)
        self.assertIn("당해연도는 발표된 최신 월까지의 연중 평균입니다", template)
        self.assertIn("function annualPointLabel(point)", template)
        self.assertIn('id="recent-period-sel"', template)
        self.assertIn('id="annual-period-sel"', template)
        self.assertNotIn('id="mode-seg"', template)
        self.assertNotIn('id="fit-btn"', template)
        self.assertNotIn('id="full-btn"', template)
        self.assertIn('aria-label="최근 이동 기간"', template)
        self.assertIn('aria-label="장기 국면 기간"', template)
        self.assertIn('role:"button"', template)
        self.assertIn("function recentAxisRanges(points)", template)
        self.assertNotIn("function equalUnitRanges(points)", template)
        self.assertIn("ranges:{gx:[-3,3],ly:[-3,3]}", template)
        self.assertIn("중립·전환 ±0.15", template)
        self.assertIn('"gl-height"', template)

    def test_quant_reference_is_visible_without_click(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn('id="quant-reference"', template)
        self.assertIn("<h2>미국 매크로 정량 보조모델</h2>", template)
        self.assertIn('class="quant-head"', template)
        self.assertNotIn("보조모델 열기", template)
        self.assertIn('<aside id="pending-observations"', template)
        self.assertIn('id="provisional-card"', template)

    def test_recent_axis_ranges_expand_growth_and_liquidity_independently(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        start = template.index("const MIN_RECENT_SPAN")
        end = template.index("function placeLabel")
        helpers = template[start:end]
        node_program = helpers + """
const points = [
  {g:-0.072,l:-0.616}, {g:-0.025,l:-0.310},
  {g:0.049,l:0.190}, {g:-0.004,l:0.120}
];
console.log(JSON.stringify(recentAxisRanges(points)));
"""
        result = subprocess.run(
            ["node", "-e", node_program],
            check=True,
            capture_output=True,
            text=True,
        )
        ranges = json.loads(result.stdout)
        growth_span = ranges["gx"][1] - ranges["gx"][0]
        liquidity_span = ranges["ly"][1] - ranges["ly"][0]

        self.assertLessEqual(growth_span, 0.5)
        self.assertGreaterEqual(liquidity_span, 1.0)
        self.assertLess(growth_span, liquidity_span)


if __name__ == "__main__":
    unittest.main()
