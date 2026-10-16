import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('check_dist_model', ROOT / 'scripts' / 'check_dist_model.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def write_dist(folder, version, html_version=None):
    html_version = html_version or version
    data = {'meta': {'generated': '2026-10-16', 'model_version': version}, 'months': []}
    Path(folder, 'gl_data.json').write_text(json.dumps(data), encoding='utf-8')
    html = ('<html><script>const BOOK = {"meta": {}};\nconst DATA = '
            + json.dumps({'meta': {'generated': '2026-10-16', 'model_version': html_version}}) + ';</script></html>')
    for name in ('index.html', 'en.html'):
        Path(folder, name).write_text(html, encoding='utf-8')


class DistModelGuardTest(unittest.TestCase):
    def test_code_version_matches_build_meta(self):
        import model_v3
        self.assertEqual(guard.code_version(), model_v3.MODEL_VERSION)
        self.assertIn('"model_version":v3.MODEL_VERSION', (ROOT / 'build.py').read_text(encoding='utf-8'))

    def test_matching_dist_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dist(tmp, guard.code_version())
            problems, seen = guard.check(tmp, guard.code_version())
            self.assertEqual(problems, [])
            self.assertEqual(set(seen), {'gl_data.json', 'index.html', 'en.html'})

    def test_stale_dist_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dist(tmp, '2.0')
            problems, _ = guard.check(tmp, '3.0')
            self.assertEqual(len(problems), 3)

    def test_mixed_dist_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dist(tmp, '3.0', html_version='2.0')
            problems, _ = guard.check(tmp, '3.0')
            self.assertEqual(sorted(p.split(':')[0] for p in problems), ['en.html', 'index.html'])

    def test_missing_public_file_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dist(tmp, '3.0')
            Path(tmp, 'en.html').unlink()
            problems, _ = guard.check(tmp, '3.0')
            self.assertEqual(problems, ['en.html: 파일 없음'])


if __name__ == '__main__':
    unittest.main()
