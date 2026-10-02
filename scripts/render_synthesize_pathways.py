"""Render LDDM synthesize CSV pathways to PNG+HTML or a single embedded HTML report.

Default mode writes PNG panels + ``index.html`` (external image refs), matching
ReaSyn-style diagrams. Use ``--embedded`` for a chemist-facing single-file HTML
report with inline SVG drawings (no external assets).

With ``--embedded``, procedure enrichment (curated cards + local OpenAlex refs)
is on by default when the procedure KB is present. Pass ``--no-enrich-procedures``
to show SMARTS-only metadata.

Reaction JSON catalogs only provide SMARTS / name / metadata — not experimental
yields or lab conditions unless enrichment is enabled. See:
https://github.com/MolecularAI/ReaSyn
https://github.com/wenhao-gao/synformer/tree/main/data/rxn_templates
https://developers.openalex.org/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from lddm.reactions.openalex_dataset import DEFAULT_PREPARED_DIR
from lddm.reactions.procedure_enrichment import (
    DEFAULT_PROCEDURE_KB,
    ProcedureMatcher,
    procedures_available,
)
from lddm.reactions.render_pathways import (
    render_routes_dataframe,
    write_embedded_retrosynthesis_html,
)
from lddm.utils import setup_logging


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_csv', type=str, help='synthesize.py / benchmark CSV')
    p.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='PNG mode: output directory. Embedded mode: output .html path '
             '(default: output/rendered_pathways or output/retrosynthesis_report.html)',
    )
    p.add_argument(
        '--embedded',
        action='store_true',
        help='Write one self-contained HTML file with inline SVG drawings',
    )
    p.add_argument(
        '--reaction-path',
        action='append',
        default=None,
        help='Reaction JSON to attach SMARTS/name/source to steps '
             '(repeatable; default for --embedded: synspace_reasyn retrosynthesis JSON)',
    )
    p.add_argument('--top-k', type=int, default=1, help='Routes to render per query')
    p.add_argument('--max-molecules', type=int, default=None)
    p.add_argument('--include-not-found', action='store_true')
    p.add_argument('--width', type=int, default=900, help='PNG panel width')
    p.add_argument('--panel-height', type=int, default=360)
    p.add_argument('--max-panel-height', type=int, default=520)
    p.add_argument('--svg-width', type=int, default=780, help='Embedded reaction SVG width')
    p.add_argument('--svg-height', type=int, default=260)
    p.add_argument('--title', type=str, default='LDDM retrosynthesis report')
    p.add_argument(
        '--enrich-procedures',
        action='store_true',
        default=None,
        help='Attach curated procedure cards + local OpenAlex refs (embedded mode)',
    )
    p.add_argument(
        '--no-enrich-procedures',
        action='store_true',
        help='Disable procedure enrichment in embedded HTML',
    )
    p.add_argument('--procedure-kb', type=str, default=str(DEFAULT_PROCEDURE_KB))
    p.add_argument('--openalex-prepared-dir', type=str, default=str(DEFAULT_PREPARED_DIR))
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    setup_logging(args.verbose)

    df = pd.read_csv(args.input_csv)

    if args.embedded:
        out = Path(args.output or 'output/retrosynthesis_report.html')
        if out.suffix.lower() not in {'.html', '.htm'}:
            out = out / 'retrosynthesis_report.html' if out.suffix == '' else out.with_suffix('.html')
        reaction_paths = args.reaction_path
        if not reaction_paths:
            default = Path('data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis.json')
            extras = Path(
                'data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis_extra.json'
            )
            reaction_paths = [p for p in (default, extras) if p.is_file()]
            if not reaction_paths:
                syn = Path('data/synspace/reactions.json')
                if syn.is_file():
                    reaction_paths = [syn]

        enrich = False
        matcher = None
        if args.no_enrich_procedures:
            enrich = False
        elif args.enrich_procedures or procedures_available(kb_path=args.procedure_kb):
            enrich = True
            matcher = ProcedureMatcher(
                kb_path=args.procedure_kb,
                prepared_dir=args.openalex_prepared_dir,
            )

        path = write_embedded_retrosynthesis_html(
            df,
            out,
            reaction_paths=reaction_paths or None,
            top_k=args.top_k,
            only_found=not args.include_not_found,
            max_molecules=args.max_molecules,
            svg_width=args.svg_width,
            svg_height=args.svg_height,
            title=args.title,
            enrich_procedures=enrich,
            procedure_matcher=matcher,
        )
        logging.info(f'Wrote embedded HTML report to {path}')
        return

    out_dir = Path(args.output or 'output/rendered_pathways')
    index = render_routes_dataframe(
        df,
        out_dir,
        top_k=args.top_k,
        only_found=not args.include_not_found,
        width=args.width,
        panel_height=args.panel_height,
        max_panel_height=args.max_panel_height,
        max_molecules=args.max_molecules,
    )
    logging.info(f'Wrote HTML index to {index}')


if __name__ == '__main__':
    main()
