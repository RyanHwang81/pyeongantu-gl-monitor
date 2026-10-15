#!/usr/bin/env python3
"""Compare two gl_data.json builds month by month. Used by the PR build check.

Prints the months that differ and the largest numeric gap per field. Timestamps
(meta.generated) are ignored. Exit code is always 0: a newer FRED vintage can
change recent months legitimately, so the result is for a human to read.
"""
import json
import os
import sys

TOL = 1e-9
IGNORE_META = {'generated'}


def flatten(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f'{prefix}.{key}' if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for i, item in enumerate(value):
            out.update(flatten(item, f'{prefix}[{i}]'))
        return out
    return {prefix: value}


def diff(a, b):
    fa, fb = flatten(a), flatten(b)
    gaps = {}
    for key in sorted(set(fa) | set(fb)):
        x, y = fa.get(key, '<missing>'), fb.get(key, '<missing>')
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if abs(x - y) > TOL:
                gaps[key] = abs(x - y)
        elif x != y:
            gaps[key] = f'{x!r} -> {y!r}'
    return gaps


def main():
    old = json.load(open(sys.argv[1], encoding='utf-8'))
    new = json.load(open(sys.argv[2], encoding='utf-8'))
    om = {m['d']: m for m in old['months']}
    nm = {m['d']: m for m in new['months']}
    common = sorted(set(om) & set(nm))
    changed = {d: diff(om[d], nm[d]) for d in common}
    changed = {d: g for d, g in changed.items() if g}
    lines = [f"기존 {len(om)}개월 / 새 빌드 {len(nm)}개월 / 공통 {len(common)}개월",
             f"새로 추가된 달: {sorted(set(nm) - set(om)) or '없음'}",
             f"사라진 달: {sorted(set(om) - set(nm)) or '없음'}",
             f"값이 달라진 달: {len(changed)}개 {list(changed)[-12:] if changed else ''}"]
    field_max, field_text = {}, {}
    for gaps in changed.values():
        for key, gap in gaps.items():
            field = key.split('.')[0].split('[')[0]
            if isinstance(gap, str):
                field_text[field] = field_text.get(field, 0) + 1
            else:
                field_max[field] = max(field_max.get(field, 0), gap)
    if field_max:
        lines.append('수치 필드 최대 차이: ' + ', '.join(f'{k}={v:.4g}' for k, v in sorted(field_max.items())))
    if field_text:
        lines.append('값/구조가 바뀐 필드(건수): ' + ', '.join(f'{k}={v}' for k, v in sorted(field_text.items())))
    meta_old = {k: v for k, v in old['meta'].items() if k not in IGNORE_META}
    meta_new = {k: v for k, v in new['meta'].items() if k not in IGNORE_META}
    meta_gaps = diff(meta_old, meta_new)
    lines.append(f"meta 차이(generated 제외): {len(meta_gaps)}개 항목"
                 + (f" 예: {list(meta_gaps)[:8]}" if meta_gaps else ''))
    verdict = '동일' if not changed and not meta_gaps and set(om) == set(nm) else '차이 있음 (아래 확인)'
    lines.insert(0, f'판정: {verdict}')
    print('\n'.join(lines))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as fh:
            fh.write('## 커밋된 dist 대비 새 빌드 비교\n\n' + '\n'.join(f'- {line}' for line in lines) + '\n')


if __name__ == '__main__':
    main()
