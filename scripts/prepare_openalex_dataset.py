#!/usr/bin/env python3
"""Normalize data/openalex/raw/*.jsonl into prepared artifacts for offline use."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from lddm.reactions.openalex_dataset import (
    DEFAULT_PREPARED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_VOCAB_PATH,
    load_vocab,
    read_jsonl,
    reference_from_record,
    vocab_hash,
)
from lddm.utils import setup_logging


def prepare(
    *,
    vocab_path: Path,
    raw_dir: Path,
    prepared_dir: Path,
) -> dict:
    vocab = load_vocab(vocab_path)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    reaction_index: dict = {}
    openalex_refs: dict = {}
    missing_raw: list[str] = []
    total_works = 0

    for entry in vocab:
        key = str(entry['key'])
        raw_path = raw_dir / f'{key}.jsonl'
        rows = read_jsonl(raw_path)
        if not raw_path.is_file():
            missing_raw.append(key)
        total_works += len(rows)
        years = sorted({int(r['year']) for r in rows if r.get('year') is not None})
        dois = [str(r.get('doi') or '') for r in rows if r.get('doi')]
        refs = [reference_from_record(r) for r in rows]
        reaction_index[key] = {
            'aliases': list(entry.get('aliases') or []),
            'class_tags': list(entry.get('class_tags') or []),
            'queries': list(entry.get('queries') or []),
            'n_works': len(rows),
            'top_dois': dois[:10],
            'years': years,
            'cited_by_total': sum(int(r.get('cited_by_count') or 0) for r in rows),
        }
        openalex_refs[key] = refs

    manifest = {
        'prepared_at': datetime.now(timezone.utc).isoformat(),
        'vocab_path': str(vocab_path),
        'vocab_hash': vocab_hash(vocab),
        'raw_dir': str(raw_dir),
        'n_vocab_keys': len(vocab),
        'n_keys_with_works': sum(1 for v in reaction_index.values() if v['n_works'] > 0),
        'total_works': total_works,
        'missing_raw_keys': missing_raw,
        'openalex_api': 'https://api.openalex.org/works',
        'docs': 'https://developers.openalex.org/',
    }

    (prepared_dir / 'reaction_index.json').write_text(
        json.dumps(reaction_index, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    (prepared_dir / 'openalex_refs.json').write_text(
        json.dumps(openalex_refs, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    (prepared_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    logging.info(
        f'Prepared {manifest["n_keys_with_works"]}/{manifest["n_vocab_keys"]} keys, '
        f'{total_works} works → {prepared_dir}'
    )
    if missing_raw:
        logging.warning(f'Missing raw JSONL for keys: {missing_raw}')
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocab', type=str, default=str(DEFAULT_VOCAB_PATH))
    p.add_argument('--raw-dir', type=str, default=str(DEFAULT_RAW_DIR))
    p.add_argument('--prepared-dir', type=str, default=str(DEFAULT_PREPARED_DIR))
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    setup_logging(args.verbose)
    prepare(
        vocab_path=Path(args.vocab),
        raw_dir=Path(args.raw_dir),
        prepared_dir=Path(args.prepared_dir),
    )


if __name__ == '__main__':
    main()
