#!/usr/bin/env python3
"""Compare prepared OpenAlex named-reaction index vs retro templates + BB roles.

Writes reaction_coverage_report.json (+ .md) under the chemical-space directory.
Literature-driven complement to missing_chemistry.json (failure-driven).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from lddm.reactions.openalex_dataset import (
    DEFAULT_PREPARED_DIR,
    DEFAULT_VOCAB_PATH,
    load_reaction_index,
    load_vocab,
    normalize_reaction_key,
)
from lddm.reactions.render_pathways import load_reaction_catalog
from lddm.utils import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE = REPO_ROOT / 'data' / 'chemical_spaces' / 'synspace_reasyn'


def _bb_role_counts(role_map: dict, reaction_id: str) -> dict[str, int] | None:
    if reaction_id not in role_map:
        return None
    roles = role_map[reaction_id]
    if not isinstance(roles, dict):
        n = len(roles) if hasattr(roles, '__len__') else 0
        return {'all': int(n)}
    return {str(k): len(v) if hasattr(v, '__len__') else int(v or 0) for k, v in roles.items()}


def _min_role_count(counts: dict[str, int] | None) -> int | None:
    if counts is None:
        return None
    if not counts:
        return 0
    return min(counts.values())


def _match_templates(
    entry: dict,
    norm_to_ids: dict[str, list[str]],
) -> list[str]:
    """Exact/alias normalize match only (no soft class-tag substring matching)."""
    candidates: list[str] = []
    seen: set[str] = set()
    for alias in [entry['key'], *(entry.get('aliases') or [])]:
        nk = normalize_reaction_key(alias)
        for rid in norm_to_ids.get(nk, []):
            if rid not in seen:
                seen.add(rid)
                candidates.append(rid)
    return candidates


def analyze(
    *,
    vocab_path: Path,
    prepared_dir: Path,
    space_dir: Path,
    low_bb_threshold: int,
    opaque_min_works: int,
) -> dict[str, Any]:
    vocab = load_vocab(vocab_path)
    index = load_reaction_index(prepared_dir)
    retro = space_dir / 'reactions_retrosynthesis.json'
    extras = space_dir / 'reactions_retrosynthesis_extra.json'
    paths = [p for p in (retro, extras) if p.is_file()]
    catalog = load_reaction_catalog(paths)

    norm_to_ids: dict[str, list[str]] = defaultdict(list)
    for rid, meta in catalog.items():
        for label in (rid, meta.get('name') or ''):
            nk = normalize_reaction_key(label)
            if nk and rid not in norm_to_ids[nk]:
                norm_to_ids[nk].append(rid)

    role_path = space_dir / 'reaction_to_building_blocks.pkl'
    role_map: dict = {}
    if role_path.is_file():
        with role_path.open('rb') as fh:
            role_map = pickle.load(fh)

    missing_template: list[dict] = []
    template_no_bb: list[dict] = []
    template_low_bb: list[dict] = []
    covered: list[dict] = []
    opaque_reasyn_only: list[dict] = []

    for entry in vocab:
        key = str(entry['key'])
        idx = index.get(key) or {}
        n_works = int(idx.get('n_works') or 0)
        matched = _match_templates(entry, norm_to_ids)

        # Prefer named (non-reasyn) matches when available
        named = [r for r in matched if not str(r).startswith('reasyn_')]
        reasyn_only = bool(matched) and not named
        primary = named[0] if named else (matched[0] if matched else None)

        bb_counts = _bb_role_counts(role_map, primary) if primary else None
        # Also try original SynSpace-style aliases against role map
        if bb_counts is None and matched:
            for rid in matched:
                bb_counts = _bb_role_counts(role_map, rid)
                if bb_counts is not None:
                    primary = rid
                    break
        if bb_counts is None:
            for alias in entry.get('aliases') or []:
                # role map keys may match alias strings exactly
                if alias in role_map:
                    bb_counts = _bb_role_counts(role_map, alias)
                    if primary is None:
                        primary = alias
                    break
                nk = normalize_reaction_key(alias)
                for rid in role_map:
                    if normalize_reaction_key(rid) == nk:
                        bb_counts = _bb_role_counts(role_map, rid)
                        if primary is None:
                            primary = rid
                        break
                if bb_counts is not None:
                    break

        min_bb = _min_role_count(bb_counts)
        row = {
            'key': key,
            'n_works': n_works,
            'class_tags': list(entry.get('class_tags') or []),
            'matched_templates': matched,
            'primary_template': primary,
            'bb_role_counts': bb_counts,
            'bb_min_role_count': min_bb,
            'cited_by_total': int(idx.get('cited_by_total') or 0),
        }

        if not matched:
            if n_works > 0:
                missing_template.append(row)
            continue

        if reasyn_only and n_works >= opaque_min_works:
            opaque_reasyn_only.append(row)

        if bb_counts is None or min_bb == 0:
            row['bb_status'] = 'absent_from_role_map' if bb_counts is None else 'empty'
            template_no_bb.append(row)
        elif min_bb is not None and min_bb < low_bb_threshold:
            row['bb_status'] = 'low'
            template_low_bb.append(row)
        else:
            row['bb_status'] = 'adequate'
            covered.append(row)

    report = {
        'space_dir': str(space_dir),
        'prepared_dir': str(prepared_dir),
        'n_vocab': len(vocab),
        'n_retro_templates': len(catalog),
        'n_role_map_reactions': len(role_map),
        'low_bb_threshold': low_bb_threshold,
        'opaque_min_works': opaque_min_works,
        'counts': {
            'missing_template': len(missing_template),
            'template_no_bb': len(template_no_bb),
            'template_low_bb': len(template_low_bb),
            'covered': len(covered),
            'opaque_reasyn_only': len(opaque_reasyn_only),
        },
        'missing_template': sorted(missing_template, key=lambda r: -r['n_works']),
        'template_no_bb': sorted(template_no_bb, key=lambda r: -r['n_works']),
        'template_low_bb': sorted(template_low_bb, key=lambda r: -r['n_works']),
        'covered': sorted(covered, key=lambda r: -r['n_works']),
        'opaque_reasyn_only': sorted(opaque_reasyn_only, key=lambda r: -r['n_works']),
    }
    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    c = report['counts']
    lines = [
        '# Reaction coverage report (OpenAlex vs retro / BB)',
        '',
        f"- Retro templates: **{report['n_retro_templates']}**",
        f"- Role-map reactions: **{report['n_role_map_reactions']}**",
        f"- Vocab keys: **{report['n_vocab']}**",
        f"- missing_template: **{c['missing_template']}**",
        f"- template_no_bb: **{c['template_no_bb']}**",
        f"- template_low_bb: **{c['template_low_bb']}**",
        f"- covered: **{c['covered']}**",
        f"- opaque_reasyn_only: **{c['opaque_reasyn_only']}**",
        '',
        '## Missing templates (literature hits, no SMARTS match)',
        '',
    ]
    for row in report['missing_template'][:40]:
        lines.append(
            f"- `{row['key']}` — {row['n_works']} works — tags={row['class_tags']}"
        )
    lines += ['', '## Template present but no / empty BB role map', '']
    for row in report['template_no_bb'][:40]:
        lines.append(
            f"- `{row['key']}` → `{row.get('primary_template')}` "
            f"({row.get('bb_status')}, works={row['n_works']})"
        )
    lines += ['', '## Low BB coverage', '']
    for row in report['template_low_bb'][:40]:
        lines.append(
            f"- `{row['key']}` → `{row.get('primary_template')}` "
            f"min_bb={row.get('bb_min_role_count')} works={row['n_works']}"
        )
    lines += ['', '## Covered', '']
    for row in report['covered'][:40]:
        lines.append(
            f"- `{row['key']}` → `{row.get('primary_template')}` "
            f"min_bb={row.get('bb_min_role_count')} works={row['n_works']}"
        )
    lines += ['', '## Opaque ReaSyn-only matches', '']
    for row in report['opaque_reasyn_only'][:40]:
        lines.append(
            f"- `{row['key']}` → {row.get('matched_templates')} works={row['n_works']}"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocab', type=str, default=str(DEFAULT_VOCAB_PATH))
    p.add_argument('--prepared-dir', type=str, default=str(DEFAULT_PREPARED_DIR))
    p.add_argument('--space-dir', type=str, default=str(DEFAULT_SPACE))
    p.add_argument('--low-bb-threshold', type=int, default=10)
    p.add_argument('--opaque-min-works', type=int, default=5)
    p.add_argument(
        '--output',
        type=str,
        default='',
        help='JSON report path (default: <space-dir>/reaction_coverage_report.json)',
    )
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    setup_logging(args.verbose)

    space_dir = Path(args.space_dir)
    report = analyze(
        vocab_path=Path(args.vocab),
        prepared_dir=Path(args.prepared_dir),
        space_dir=space_dir,
        low_bb_threshold=args.low_bb_threshold,
        opaque_min_works=args.opaque_min_works,
    )
    out_json = Path(args.output) if args.output else space_dir / 'reaction_coverage_report.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    out_md = out_json.with_suffix('.md')
    out_md.write_text(report_to_markdown(report), encoding='utf-8')
    logging.info(f'Wrote {out_json} and {out_md}')
    logging.info(f'Counts: {report["counts"]}')


if __name__ == '__main__':
    main()
