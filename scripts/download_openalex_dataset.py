#!/usr/bin/env python3
"""Download OpenAlex works for the named-reaction vocabulary into data/openalex/raw/.

Uses the public OpenAlex REST API (polite pool via --mailto). Does not download
the full S3 snapshot (hundreds of GB). See:
https://developers.openalex.org/api-guide/get-started-pages/api-overview
https://developers.openalex.org/download/download-to-machine
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from lddm.reactions.openalex_dataset import (
    DEFAULT_RAW_DIR,
    DEFAULT_VOCAB_PATH,
    load_vocab,
    work_record_from_openalex,
    write_jsonl,
)
from lddm.utils import setup_logging

OPENALEX_WORKS = 'https://api.openalex.org/works'


def _fetch_json(url: str, *, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'lddm-openalex-download (research; mailto in query)'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        import json

        return json.loads(resp.read().decode('utf-8'))


def search_works(
    query: str,
    *,
    mailto: str,
    per_page: int,
    max_retries: int = 5,
) -> list[dict]:
    params = {
        'search': query,
        'per_page': str(per_page),
        'sort': 'cited_by_count:desc',
        'mailto': mailto,
    }
    url = f'{OPENALEX_WORKS}?{urllib.parse.urlencode(params)}'
    delay = 1.0
    for attempt in range(max_retries):
        try:
            payload = _fetch_json(url)
            return list(payload.get('results') or [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                logging.warning(f'OpenAlex HTTP {e.code}; retry in {delay:.1f}s')
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise
        except urllib.error.URLError as e:
            if attempt + 1 < max_retries:
                logging.warning(f'OpenAlex network error {e}; retry in {delay:.1f}s')
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise
    return []


def download_vocab_key(
    entry: dict,
    *,
    out_path: Path,
    mailto: str,
    top_n: int,
    sleep_s: float,
) -> int:
    seen: dict[str, dict] = {}
    for query in entry.get('queries') or []:
        works = search_works(str(query), mailto=mailto, per_page=top_n)
        for work in works:
            rec = work_record_from_openalex(work)
            wid = rec.get('id') or ''
            if not wid or wid in seen:
                continue
            seen[wid] = rec
        time.sleep(sleep_s)
    rows = sorted(
        seen.values(),
        key=lambda r: (-int(r.get('cited_by_count') or 0), str(r.get('year') or '')),
    )[:top_n]
    write_jsonl(out_path, rows)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocab', type=str, default=str(DEFAULT_VOCAB_PATH))
    p.add_argument('--out-dir', type=str, default=str(DEFAULT_RAW_DIR))
    p.add_argument(
        '--mailto',
        type=str,
        default=os.environ.get('OPENALEX_MAILTO', ''),
        help='Email for OpenAlex polite pool (or set OPENALEX_MAILTO)',
    )
    p.add_argument('--top-n', type=int, default=20, help='Max works kept per vocab key')
    p.add_argument('--sleep', type=float, default=0.15, help='Pause between API queries')
    p.add_argument('--force', action='store_true', help='Re-download even if raw JSONL exists')
    p.add_argument(
        '--keys',
        type=str,
        default='',
        help='Comma-separated vocab keys to download (default: all)',
    )
    p.add_argument(
        '--from-snapshot',
        type=str,
        default='',
        help='Reserved: path to local OpenAlex snapshot (not implemented in V1)',
    )
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    setup_logging(args.verbose)

    if args.from_snapshot:
        raise SystemExit(
            '--from-snapshot is reserved for a future local S3 dump filter. '
            'Use the REST download path for V1, or see '
            'https://developers.openalex.org/download/download-to-machine'
        )

    mailto = (args.mailto or '').strip()
    if not mailto or '@' not in mailto:
        raise SystemExit(
            'Provide --mailto you@org.com (or OPENALEX_MAILTO) for the OpenAlex polite pool: '
            'https://developers.openalex.org/api-guide/rate-limits-and-authentication'
        )

    vocab = load_vocab(args.vocab)
    key_filter = {k.strip() for k in args.keys.split(',') if k.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for entry in vocab:
        key = str(entry['key'])
        if key_filter and key not in key_filter:
            continue
        out_path = out_dir / f'{key}.jsonl'
        if out_path.is_file() and not args.force:
            n = sum(1 for _ in out_path.open(encoding='utf-8') if _.strip())
            logging.info(f'skip {key} ({n} works already in {out_path})')
            total += n
            continue
        logging.info(f'downloading {key} …')
        n = download_vocab_key(
            entry,
            out_path=out_path,
            mailto=mailto,
            top_n=args.top_n,
            sleep_s=args.sleep,
        )
        logging.info(f'  wrote {n} works → {out_path}')
        total += n

    logging.info(f'Done. {total} work records under {out_dir}')


if __name__ == '__main__':
    main()
