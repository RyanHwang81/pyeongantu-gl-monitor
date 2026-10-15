#!/usr/bin/env python3
"""Check FRED reachability from the current machine. Prints status and timing only.

The official API needs FRED_API_KEY. The key never appears in output: errors are
reported by exception type and HTTP status, never by URL.

Usage: FRED_API_KEY=... python3 scripts/fred_probe.py DGS2 ICSA [--csv]
Exit code 1 if any API request fails.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {'User-Agent': 'pyeongantoo-gl-probe/1.0'}


def timed_get(url, timeout):
    start = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as response:
            body = response.read()
            return {'ok': True, 'http': response.status, 'seconds': round(time.monotonic() - start, 2),
                    'bytes': len(body)}, body
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'http': exc.code, 'error': 'HTTPError',
                'seconds': round(time.monotonic() - start, 2)}, None
    except Exception as exc:
        return {'ok': False, 'http': None, 'error': type(exc).__name__,
                'seconds': round(time.monotonic() - start, 2)}, None


def probe_api(series_id, key, timeout):
    url = 'https://api.stlouisfed.org/fred/series/observations?' + urllib.parse.urlencode(
        {'series_id': series_id, 'api_key': key, 'file_type': 'json', 'sort_order': 'desc', 'limit': 5})
    result, body = timed_get(url, timeout)
    if body is not None:
        try:
            rows = json.loads(body)['observations']
            result['latest'] = next((f"{r['date']}={r['value']}" for r in rows if r['value'] != '.'), None)
        except Exception as exc:
            result.update(ok=False, error=f'parse:{type(exc).__name__}')
    return result


def probe_csv(series_id, timeout):
    """The graph CSV path that timed out on runners before. Informational only."""
    result, _ = timed_get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}', timeout)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('series', nargs='*', default=['DGS2', 'ICSA'])
    ap.add_argument('--csv', action='store_true', help='also time the keyless graph CSV endpoint')
    ap.add_argument('--timeout', type=float, default=45)
    args = ap.parse_args()
    key = os.environ.get('FRED_API_KEY', '')
    if not key:
        print('FAIL FRED_API_KEY is empty (repository secret missing or not passed)')
        raise SystemExit(1)
    failed = False
    lines = ['| 경로 | 시리즈 | 결과 | HTTP | 응답 시간(초) | 최신 관측 |', '|---|---|---|---|---|---|']
    for sid in args.series:
        r = probe_api(sid, key, args.timeout)
        failed |= not r['ok']
        print(('OK  ' if r['ok'] else 'FAIL'), 'api', sid, json.dumps(r, ensure_ascii=False))
        lines.append(f"| API | {sid} | {'성공' if r['ok'] else '실패 ' + r.get('error', '')} | {r['http']} | {r['seconds']} | {r.get('latest') or '-'} |")
        if args.csv:
            c = probe_csv(sid, args.timeout)
            print(('OK  ' if c['ok'] else 'FAIL'), 'csv', sid, json.dumps(c, ensure_ascii=False), '(informational)')
            lines.append(f"| graph CSV(참고) | {sid} | {'성공' if c['ok'] else '실패 ' + c.get('error', '')} | {c['http']} | {c['seconds']} | - |")
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as fh:
            fh.write('## FRED 접속 시험\n\n' + '\n'.join(lines) + '\n\n')
    print('\n'.join(lines))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
