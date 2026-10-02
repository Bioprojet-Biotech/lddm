"""Shared helpers for OpenAlex named-reaction download / prepare / coverage."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB_PATH = REPO_ROOT / 'data' / 'openalex' / 'named_reaction_vocab.json'
DEFAULT_RAW_DIR = REPO_ROOT / 'data' / 'openalex' / 'raw'
DEFAULT_PREPARED_DIR = REPO_ROOT / 'data' / 'openalex' / 'prepared'


def normalize_reaction_key(text: str) -> str:
    """Casefold + collapse separators for fuzzy id/name matching."""
    s = str(text or '').casefold().strip()
    s = s.replace('–', '-').replace('—', '-')
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')


def load_vocab(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path or DEFAULT_VOCAB_PATH)
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError(f'Vocab must be a JSON list: {path}')
    for i, entry in enumerate(data):
        if 'key' not in entry or 'queries' not in entry:
            raise ValueError(f'Vocab entry {i} needs key + queries')
    return data


def vocab_hash(vocab: list[dict[str, Any]]) -> str:
    blob = json.dumps(vocab, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def work_record_from_openalex(work: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAlex work payload to a compact local record."""
    oa = work.get('open_access') or {}
    doi = work.get('doi') or ''
    if isinstance(doi, str) and doi.startswith('https://doi.org/'):
        doi = doi[len('https://doi.org/'):]
    year = work.get('publication_year')
    return {
        'id': work.get('id') or '',
        'doi': doi,
        'title': work.get('display_name') or work.get('title') or '',
        'year': year,
        'publication_date': work.get('publication_date') or '',
        'cited_by_count': int(work.get('cited_by_count') or 0),
        'oa_url': oa.get('oa_url') or '',
        'landing_url': (work.get('primary_location') or {}).get('landing_page_url') or '',
    }


def reference_from_record(rec: dict[str, Any], *, source: str = 'openalex') -> dict[str, Any]:
    doi = rec.get('doi') or ''
    url = ''
    if doi:
        url = f'https://doi.org/{doi}'
    elif rec.get('oa_url'):
        url = str(rec['oa_url'])
    elif rec.get('landing_url'):
        url = str(rec['landing_url'])
    elif rec.get('id'):
        url = str(rec['id'])
    return {
        'doi': doi,
        'title': rec.get('title') or '',
        'url': url,
        'source': source,
        'year': rec.get('year'),
        'cited_by_count': int(rec.get('cited_by_count') or 0),
        'openalex_id': rec.get('id') or '',
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_prepared_refs(prepared_dir: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    path = Path(prepared_dir or DEFAULT_PREPARED_DIR) / 'openalex_refs.json'
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}


def load_reaction_index(prepared_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(prepared_dir or DEFAULT_PREPARED_DIR) / 'reaction_index.json'
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}
