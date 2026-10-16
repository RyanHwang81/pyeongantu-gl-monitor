#!/usr/bin/env python3
"""Fail when the dist files were built by a different model version than the code.

A build from a stale checkout writes an older meta.model_version into dist. This
check compares every published file with MODEL_VERSION in model_v3.py and exits 1
on any mismatch, so the workflow stops before commit and deploy.

model_v3.py is read with ast, not imported: the deploy-only workflow has no pandas.

Usage: python3 scripts/check_dist_model.py [dist_dir]
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ('gl_data.json', 'index.html', 'en.html')
OPTIONAL = ('gl-internal.html',)
DATA_RE = re.compile(r'const DATA = (\{"meta": \{.*?"model_version": "([^"]+)")', re.S)


def code_version(path=ROOT / 'model_v3.py'):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'MODEL_VERSION' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('MODEL_VERSION not found in model_v3.py')


def file_version(path):
    text = Path(path).read_text(encoding='utf-8')
    if path.suffix == '.json':
        return json.loads(text).get('meta', {}).get('model_version')
    match = DATA_RE.search(text)
    return match.group(2) if match else None


def check(dist, expected):
    problems, seen = [], {}
    for name in PUBLIC + OPTIONAL:
        path = Path(dist) / name
        if not path.exists():
            if name in PUBLIC:
                problems.append(f'{name}: 파일 없음')
            continue
        found = file_version(path)
        seen[name] = found
        if found != expected:
            problems.append(f'{name}: model_version={found!r}, 코드={expected!r}')
    return problems, seen


def main():
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'dist'
    expected = code_version()
    problems, seen = check(dist, expected)
    print(f'코드 MODEL_VERSION={expected} / dist: ' + ', '.join(f'{k}={v}' for k, v in seen.items()))
    if problems:
        print('배포 중단: dist가 현재 코드와 다른 모델 버전으로 빌드되었습니다. 최신 main에서 다시 빌드하세요.')
        for line in problems:
            print('  -', line)
        raise SystemExit(1)
    print('dist model_version OK')


if __name__ == '__main__':
    main()
