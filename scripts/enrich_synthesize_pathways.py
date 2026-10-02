#!/usr/bin/env python3
"""Enrich synthesize / benchmark pathway CSVs with offline procedure cards.

Reads curated named-reaction procedures + prepared OpenAlex refs only
(no runtime network). Optionally writes enriched JSON and/or embedded HTML.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from lddm.reactions.openalex_dataset import DEFAULT_PREPARED_DIR
from lddm.reactions.procedure_enrichment import ProcedureMatcher, enrich_route
from lddm.reactions.render_pathways import write_embedded_retrosynthesis_html
from lddm.utils import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB = REPO_ROOT / 'data' / 'procedures' / 'named_reaction_procedures.json'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True, type=str, help='synthesize / benchmark CSV')
    p.add_argument(
        '--out',
        type=str,
        default='output/enriched_pathways.json',
        help='Enriched JSON path (and .html if --html omitted but out ends with .html)',
    )
    p.add_argument('--html', type=str, default='', help='Optional embedded HTML report path')
    p.add_argument('--kb', type=str, default=str(DEFAULT_KB))
    p.add_argument('--prepared-dir', type=str, default=str(DEFAULT_PREPARED_DIR))
    p.add_argument('--top-k', type=int, default=1)
    p.add_argument('--max-molecules', type=int, default=None)
    p.add_argument('--include-not-found', action='store_true')
    p.add_argument('--title', type=str, default='LDDM enriched retrosynthesis report')
    p.add_argument(
        '--reaction-path',
        action='append',
        default=None,
        help='Reaction JSON for HTML catalog (repeatable)',
    )
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    setup_logging(args.verbose)

    df = pd.read_csv(args.csv)
    required = {'query_smiles', 'found', 'react_trace'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'CSV missing columns: {sorted(missing)}')

    work = df.copy()
    if not args.include_not_found:
        work = work[work['found'].astype(bool)]
    work = work[work['react_trace'].astype(str).str.len() > 0]
    sort_cols = [c for c in ('query_smiles', 'exact', 'similarity', 'n_steps') if c in work.columns]
    if sort_cols:
        ascending = []
        for c in sort_cols:
            if c in ('exact', 'similarity'):
                ascending.append(False)
            elif c == 'n_steps':
                ascending.append(True)
            else:
                ascending.append(True)
        work = work.sort_values(sort_cols, ascending=ascending)
    work = work.groupby('query_smiles', as_index=False).head(args.top_k)
    if args.max_molecules is not None:
        keep = list(dict.fromkeys(work['query_smiles'].tolist()))[: args.max_molecules]
        work = work[work['query_smiles'].isin(keep)]

    matcher = ProcedureMatcher(kb_path=args.kb, prepared_dir=args.prepared_dir)
    routes = []
    for _, row in work.iterrows():
        trace = str(row['react_trace'])
        try:
            enriched = enrich_route(trace, matcher=matcher)
        except Exception as e:
            logging.warning(f'Failed to enrich {row.get("query_smiles")}: {e}')
            continue
        routes.append(
            {
                'query_smiles': str(row['query_smiles']),
                'found': bool(row.get('found', True)),
                'exact': bool(row.get('exact', False)) if 'exact' in row else None,
                'n_steps': int(row['n_steps']) if 'n_steps' in row and pd.notna(row['n_steps']) else len(enriched),
                'react_trace': trace,
                'pathway': str(row.get('pathway') or ''),
                'steps': [s.to_dict() for s in enriched],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_path = Path(args.html) if args.html else None
    if out.suffix.lower() in {'.html', '.htm'} and html_path is None:
        html_path = out
        json_path = out.with_suffix('.json')
    else:
        json_path = out if out.suffix.lower() == '.json' else out.with_suffix('.json')

    payload = {
        'n_routes': len(routes),
        'kb': str(args.kb),
        'prepared_dir': str(args.prepared_dir),
        'offline': True,
        'routes': routes,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    logging.info(f'Wrote enriched JSON → {json_path} ({len(routes)} routes)')

    if html_path is not None:
        reaction_paths = args.reaction_path
        if not reaction_paths:
            default = Path('data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis.json')
            extras = Path(
                'data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis_extra.json'
            )
            reaction_paths = [p for p in (default, extras) if p.is_file()]
        path = write_embedded_retrosynthesis_html(
            work,
            html_path,
            reaction_paths=reaction_paths or None,
            top_k=args.top_k,
            only_found=not args.include_not_found,
            max_molecules=args.max_molecules,
            title=args.title,
            enrich_procedures=True,
            procedure_matcher=matcher,
        )
        logging.info(f'Wrote enriched HTML → {path}')


if __name__ == '__main__':
    main()
