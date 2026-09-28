import io
import json
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import build


class SeptemberStatusTests(unittest.TestCase):
    def test_recalculated_prior_month_discloses_changed_point_and_new_inputs(self):
        prior = {"months": [{"d": "2026-08", "g": -.016, "l": .275, "r": "liquidity",
                             "gz": {"PERMIT": None, "INDPRO": None}, "lz": {"M2SL": None}}], "meta": {}}
        current = {"months": [{"d": "2026-08", "g": .030, "l": .223, "r": "expansion",
                               "gz": {"PERMIT": .05, "INDPRO": .26}, "lz": {"M2SL": -.13}}], "meta": {}}
        note = build.recalculation_note(prior, current)
        self.assertEqual("2026-08", note["month"])
        self.assertEqual(["PERMIT", "INDPRO", "M2SL"], note["newly_available"])
        self.assertEqual("expansion", note["after"]["r"])
        current["meta"]["recalculation"] = note
        self.assertEqual(note, build.recalculation_note(current, current))

    def test_observations_report_actual_month_and_weight_without_filling_gaps(self):
        raw = {
            "ICSA": pd.Series([198000, 197000], index=pd.to_datetime(["2026-09-12", "2026-09-19"])),
            "T10Y3M": pd.Series([.7], index=pd.to_datetime(["2026-09-25"])),
            "WALCL": pd.Series([6.6], index=pd.to_datetime(["2026-09-23"])),
            "PERMIT": pd.Series([1.4], index=pd.to_datetime(["2026-08-01"])),
        }
        snapshot = build.pending_observations(raw, "2026-08", date(2026, 9, 28))
        self.assertEqual("2026-09", snapshot["month"])
        self.assertEqual("2026-09-25", snapshot["as_of"])
        self.assertEqual("month_in_progress", snapshot["status"])
        self.assertEqual({"available": 1, "total": 5, "weight": .10}, snapshot["growth"])
        self.assertEqual({"available": 2, "total": 6, "weight": .32}, snapshot["liquidity"])
        self.assertEqual({"ICSA", "T10Y3M", "WALCL"}, {o["id"] for o in snapshot["observed"]})
        self.assertEqual("2026-09-19", next(o["as_of"] for o in snapshot["observed"] if o["id"] == "ICSA"))
        self.assertIsNone(build.pending_observations(raw, "2026-09", date(2026, 9, 28)))

    def test_fred_api_parses_missing_values_and_never_leaks_key_on_error(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload
            def __enter__(self):
                return io.BytesIO(self.payload)
            def __exit__(self, *args):
                return False
        payload = json.dumps({"observations": [
            {"date": "2026-09-05", "value": "207000"},
            {"date": "2026-09-12", "value": "."},
        ]}).encode()
        with patch("build.urllib.request.urlopen", return_value=Response(payload)):
            series = build.fred_api("ICSA", "not-a-real-secret")
        self.assertEqual(207000, series.loc["2026-09-05"])
        self.assertEqual(1, len(series))
        with patch("build.urllib.request.urlopen", side_effect=RuntimeError("not-a-real-secret")):
            with self.assertRaises(RuntimeError) as caught:
                build.fred_api("ICSA", "not-a-real-secret")
        self.assertNotIn("not-a-real-secret", str(caught.exception))

    def test_public_html_shows_pending_month_but_keeps_last_point(self):
        data = json.loads((build.Path(__file__).resolve().parents[1] / "dist/gl_data.json").read_text())
        data["meta"]["pending_observations"] = {
            "month": "2026-09", "as_of": "2026-09-25", "status": "month_in_progress",
            "growth": {"available": 1, "total": 5, "weight": .10},
            "liquidity": {"available": 2, "total": 6, "weight": .32},
            "observed": [{"id": "ICSA", "as_of": "2026-09-19"},
                         {"id": "T10Y3M", "as_of": "2026-09-25"},
                         {"id": "WALCL", "as_of": "2026-09-23"}],
        }
        book = build.load_book_dashboard()
        template = build.Path(__file__).resolve().parents[1].joinpath("gl_template.html").read_text()
        html = build.render(template, data, book, public=True)
        self.assertIn('id="pending-observations"', html)
        self.assertIn('const monthEn=', html)
        self.assertIn('" 진행 중 / "+monthEn+" in progress"', html)
        self.assertIn('id="pending-month"', html)
        self.assertEqual("2026-08", data["months"][-1]["d"])
        self.assertNotIn('"d": "2026-09"', html)


if __name__ == "__main__":
    unittest.main()
