"""Expand an LDDM chemical space with multi-source building-block catalogs.

Default flow (suggested coverage upgrade):
  1. Load current space BBs (e.g. synspace_reasyn)
  2. Download / load Mcule purchasable building blocks
  3. Dedup by canonical SMILES; keep a **single preferred primary** provider
     (priority order) while recording all origins in ``sources``
  4. Optionally re-run ``prepare_space`` for fingerprints + role maps

Example::

    python scripts/expand_bb_catalog.py \\
        --base-building-blocks data/chemical_spaces/synspace_reasyn/building_blocks.csv \\
        --base-reactions data/chemical_spaces/synspace_reasyn/reactions.json \\
        --download-mcule \\
        --max-mcule-bbs 200000 \\
        --output data/chemical_spaces/synspace_reasyn_mcule \\
        --prepare --drop-empty-roles

Mcule catalog: https://mcule.com/database/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from lddm.reactions.bb_catalog import (
    MCULE_BB_SMI_URL,
    download_url,
    load_bb_table,
    load_smiles_catalog,
    merge_bb_catalogs,
)
from lddm.reactions.prepare_space import prepare_space
from lddm.utils import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--base-building-blocks',
        type=Path,
        required=True,
        help='Existing BB CSV (id,smiles[,provider,sources,...])',
    )
    p.add_argument(
        '--base-reactions',
        type=Path,
        default=None,
        help='Reactions JSON to copy / prepare against (required with --prepare)',
    )
    p.add_argument(
        '--base-reactions-retrosynthesis',
        type=Path,
        default=None,
        help='Optional retrosynthesis reactions JSON to copy into the output space',
    )
    p.add_argument(
        '--mcule-path',
        type=Path,
        default=None,
        help='Local Mcule .smi / .smi.gz (skips download if set)',
    )
    p.add_argument(
        '--download-mcule',
        action='store_true',
        help='Download Mcule purchasable BB SMILES into --cache-dir',
    )
    p.add_argument(
        '--mcule-url',
        type=str,
        default=MCULE_BB_SMI_URL,
        help='Override Mcule SMILES URL (filenames change periodically)',
    )
    p.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('data/benchmarks/catalogs'),
        help='Where to store downloaded catalogs',
    )
    p.add_argument(
        '--max-mcule-bbs',
        type=int,
        default=None,
        help='Cap Mcule rows read (after file order; useful before full prepare)',
    )
    p.add_argument(
        '--extra-bb-csv',
        type=Path,
        action='append',
        default=[],
        help='Additional BB CSV(s) with optional provider column (repeatable)',
    )
    p.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output chemical-space directory',
    )
    p.add_argument(
        '--prepare',
        action='store_true',
        help='Run prepare_space (fps + role membership) on the merged BBs',
    )
    p.add_argument(
        '--drop-empty-roles',
        action='store_true',
        help='Drop reactions with empty roles during prepare (recommended for large catalogs)',
    )
    p.add_argument('--force-download', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    setup_logging(args.verbose)

    tables = [load_bb_table(args.base_building_blocks)]
    logging.info(
        f'Base catalog: {len(tables[0])} BBs from {args.base_building_blocks}'
    )

    mcule_path = args.mcule_path
    if args.download_mcule:
        dest = args.cache_dir / Path(args.mcule_url).name
        mcule_path = download_url(args.mcule_url, dest, force=args.force_download)
    if mcule_path is not None:
        mcule = load_smiles_catalog(
            mcule_path,
            provider='mcule',
            id_prefix='mcule',
            max_rows=args.max_mcule_bbs,
        )
        logging.info(f'Mcule catalog: {len(mcule)} BBs from {mcule_path}')
        tables.append(mcule)

    for extra in args.extra_bb_csv:
        extra_df = load_bb_table(extra)
        logging.info(f'Extra catalog: {len(extra_df)} BBs from {extra}')
        tables.append(extra_df)

    merged, stats = merge_bb_catalogs(tables, skip_invalid=True)
    logging.info(
        f'Merged {stats["input_rows"]} → {stats["union"]} unique SMILES '
        f'(single-source={stats["single_source"]}, multi-source={stats["multi_source"]}, '
        f'collisions={stats["smiles_collisions"]})'
    )
    logging.info(f'Primary providers: {stats["by_provider"]}')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bb_csv = out / 'building_blocks.csv'
    merged.to_csv(bb_csv, index=False)
    (out / 'bb_merge_report.json').write_text(json.dumps(stats, indent=2) + '\n')

    # Copy reaction templates into the new space folder for a self-contained layout.
    if args.base_reactions is not None and args.base_reactions.is_file():
        shutil.copy2(args.base_reactions, out / 'reactions.json')
    if args.base_reactions_retrosynthesis is not None and args.base_reactions_retrosynthesis.is_file():
        shutil.copy2(
            args.base_reactions_retrosynthesis,
            out / 'reactions_retrosynthesis.json',
        )
    elif args.base_reactions is not None:
        # Fall back: look next to base reactions.
        sibling = args.base_reactions.with_name('reactions_retrosynthesis.json')
        if sibling.is_file():
            shutil.copy2(sibling, out / 'reactions_retrosynthesis.json')

    extras_json = None
    if args.base_reactions is not None:
        extras_json = args.base_reactions.with_name('reactions_retrosynthesis_extra.json')
    if extras_json is not None and extras_json.is_file():
        shutil.copy2(extras_json, out / 'reactions_retrosynthesis_extra.json')

    if args.prepare:
        if args.base_reactions is None or not args.base_reactions.is_file():
            raise SystemExit('--prepare requires --base-reactions')
        import json as _json

        reactions = _json.loads(Path(out / 'reactions.json').read_text())
        logging.info(
            f'Preparing space ({len(merged)} BBs × {len(reactions)} bimolecular reactions)…'
        )
        prepare_space(
            merged,
            reactions,
            out,
            drop_empty_roles=args.drop_empty_roles,
        )
        logging.info(f'Prepared artifacts written under {out}')
    else:
        logging.info(
            f'Wrote {bb_csv}. Run with --prepare to build pkl + role maps, '
            f'or point synthesize at a prepared sibling space.'
        )


if __name__ == '__main__':
    main()
