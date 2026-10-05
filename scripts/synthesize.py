"""Retrosynthesis: recover synthesis pathways for query SMILES in a local chemical space.

Optionally attaches offline procedure cards + local OpenAlex literature refs
(``--enrich-procedures``, on by default when the curated KB is present).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from lddm.reactions.openalex_dataset import DEFAULT_PREPARED_DIR
from lddm.reactions.procedure_enrichment import (
    DEFAULT_PROCEDURE_KB,
    ProcedureMatcher,
    attach_procedures_to_dataframe,
    procedures_available,
)
from lddm.reactions.render_pathways import write_embedded_retrosynthesis_html
from lddm.reactions.retrosynthesis import LocalRetrosynthesizer, routes_to_dataframe
from lddm.utils import disable_rdkit_logging, merge_args_and_yaml, set_default, setup_logging


def _load_smiles(args) -> list[str]:
    smiles: list[str] = []
    if getattr(args, 'smiles', None):
        smiles.extend(args.smiles)
    if getattr(args, 'input', None):
        path = Path(args.input)
        if not path.is_file():
            raise FileNotFoundError(f'Input file not found: {path}')
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
            col = 'smiles' if 'smiles' in df.columns else df.columns[0]
            smiles.extend(df[col].astype(str).tolist())
        else:
            # .smi / .txt: one SMILES per line, optional whitespace-separated id
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                smiles.append(line.split()[0])
    if not smiles:
        raise ValueError('Provide --smiles and/or --input with at least one molecule')
    return smiles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', nargs='?', type=str, help='Optional YAML config')
    p.add_argument('--smiles', nargs='+', help='Query SMILES strings')
    p.add_argument('--input', type=str, help='Input .smi/.txt or .csv (smiles column)')
    p.add_argument('--output', type=str, help='Output CSV path')
    p.add_argument('--reaction-path', type=str, help='Path to reactions.json')
    p.add_argument('--building-blocks-path', type=str, help='Path to building_blocks.pkl')
    p.add_argument(
        '--reaction-to-compound-path',
        type=str,
        help='Optional reaction-to-building-blocks.pkl for role membership checks',
    )
    p.add_argument('--max-depth', type=int, default=None, help='Max retrosynthesis depth (default: 5)')
    p.add_argument(
        '--max-routes-per-mol',
        type=int,
        default=None,
        help='Max routes to keep per query molecule (default: 10)',
    )
    p.add_argument(
        '--max-disconnections',
        type=int,
        default=None,
        help='Max ranked disconnections to expand per intermediate (default: 64)',
    )
    p.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help='Process-pool workers for multi-molecule batches (default: 1)',
    )
    p.add_argument(
        '--disconnect-cache',
        type=str,
        default=None,
        help='Optional pickle path to persist / reuse disconnection caches across runs',
    )
    p.add_argument(
        '--reaction-pack',
        type=str,
        default=None,
        help='Optional retrosynthesis_pack.pkl (default: next to reaction_path)',
    )
    p.add_argument(
        '--no-prepared-bb-cache',
        action='store_true',
        default=None,
        help='Disable sidecar prepared building-block cache (*.pkl.prepared.pkl)',
    )
    p.add_argument(
        '--stream-output',
        action='store_true',
        default=None,
        help='Write CSV rows incrementally after each molecule (lower peak memory)',
    )
    p.add_argument(
        '--approximate',
        action='store_true',
        default=None,
        help='Keep near-miss constructions (default similarity threshold 0.7)',
    )
    p.add_argument(
        '--similarity-threshold',
        type=float,
        default=None,
        help='Minimum Tanimoto similarity between query and reconstructed product '
             '(1.0 = exact only; with --approximate default is 0.7)',
    )
    p.add_argument(
        '--no-force-uncharge',
        action='store_true',
        default=None,
        help='Disable ChargeParent/Uncharger preprocess before search',
    )
    p.add_argument(
        '--no-simplify-peroxides',
        action='store_true',
        default=None,
        help='Disable O–O → ether/CH2 simplification before search',
    )
    p.add_argument(
        '--transform-similarity-threshold',
        type=float,
        default=None,
        help='Min Tanimoto(original, reconstructed) when uncharge/peroxide '
             'transforms were applied (default 0.7)',
    )
    p.add_argument(
        '--no-require-role-membership',
        action='store_true',
        default=None,
        help='Do not require precursors to match curated reaction roles',
    )
    p.add_argument(
        '--no-proposed-reagents',
        action='store_true',
        default=None,
        help='Require every leaf to be a catalog building block '
             '(disable reverse-proposed halide partners)',
    )
    p.add_argument(
        '--no-early-skip',
        action='store_true',
        default=None,
        help='Always escalate shallow misses to max_depth '
             '(disable junk/zero-disc prefilter and progress gate)',
    )
    p.add_argument(
        '--no-early-skip-junk-chem',
        action='store_true',
        default=None,
        help='Disable cumulene / weird-S early-skip prefilter',
    )
    p.add_argument(
        '--early-skip-min-useful-discs',
        type=int,
        default=None,
        help='Min non-excluded disconnections required to escalate past depth 2 '
             '(default 1; set 0 to escalate whenever any disc exists after exclusions)',
    )
    p.add_argument(
        '--enrich-procedures',
        action='store_true',
        default=None,
        help='Attach offline procedure cards + local OpenAlex refs to each route '
             '(default: on when data/procedures/named_reaction_procedures.json exists)',
    )
    p.add_argument(
        '--no-enrich-procedures',
        action='store_true',
        default=None,
        help='Disable procedure / OpenAlex enrichment on the output CSV',
    )
    p.add_argument(
        '--procedure-kb',
        type=str,
        default=None,
        help='Curated procedure JSON (default: data/procedures/named_reaction_procedures.json)',
    )
    p.add_argument(
        '--openalex-prepared-dir',
        type=str,
        default=None,
        help='Prepared OpenAlex dir (default: data/openalex/prepared)',
    )
    p.add_argument(
        '--html',
        type=str,
        default=None,
        help='Also write an embedded HTML retrosynthesis report with procedure panels',
    )
    p.add_argument('--verbose', action='store_true', default=None)
    args = p.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    cfg = merge_args_and_yaml(args, config)

    set_default(cfg, 'verbose', False)
    setup_logging(cfg.verbose)
    if not cfg.verbose:
        disable_rdkit_logging()
    set_default(cfg, 'reaction_path', 'data/synspace/reactions.json')
    set_default(cfg, 'building_blocks_path', 'data/synspace/building_blocks.pkl')
    set_default(cfg, 'reaction_to_compound_path', 'data/synspace/reaction_to_building_blocks.pkl')
    set_default(cfg, 'max_depth', 5)
    set_default(cfg, 'max_routes_per_mol', 10)
    set_default(cfg, 'max_disconnections', 64)
    set_default(cfg, 'n_workers', 1)
    set_default(cfg, 'require_role_membership', True)
    set_default(cfg, 'allow_proposed_reagents', True)
    set_default(cfg, 'max_proposed_reagent_atoms', 16)
    set_default(cfg, 'seed_trivial_catalog', True)
    set_default(cfg, 'force_uncharge', True)
    set_default(cfg, 'simplify_peroxides', True)
    set_default(cfg, 'peroxide_mode', 'ether')
    set_default(cfg, 'transform_similarity_threshold', 0.7)
    set_default(cfg, 'approximate', False)
    set_default(cfg, 'similarity_threshold', None)
    set_default(cfg, 'output', 'output/synthesize_pathways.csv')
    set_default(cfg, 'disconnect_cache', None)
    set_default(cfg, 'reaction_pack', None)
    set_default(cfg, 'stream_output', False)
    set_default(cfg, 'use_prepared_bb_cache', True)
    set_default(cfg, 'early_skip', True)
    set_default(cfg, 'early_skip_junk_chem', True)
    set_default(cfg, 'early_skip_exclude_reactions', ['retro_cc_wurtz'])
    set_default(cfg, 'early_skip_min_useful_discs', 1)
    set_default(cfg, 'procedure_kb', str(DEFAULT_PROCEDURE_KB))
    set_default(cfg, 'openalex_prepared_dir', str(DEFAULT_PREPARED_DIR))
    set_default(cfg, 'html', None)
    set_default(cfg, 'enrich_procedures', None)
    if getattr(cfg, 'no_require_role_membership', None):
        cfg.require_role_membership = False
    if getattr(cfg, 'no_proposed_reagents', None):
        cfg.allow_proposed_reagents = False
    if getattr(cfg, 'no_prepared_bb_cache', None):
        cfg.use_prepared_bb_cache = False
    if getattr(cfg, 'no_force_uncharge', None):
        cfg.force_uncharge = False
    if getattr(cfg, 'no_simplify_peroxides', None):
        cfg.simplify_peroxides = False
    if getattr(cfg, 'no_early_skip', None):
        cfg.early_skip = False
    if getattr(cfg, 'no_early_skip_junk_chem', None):
        cfg.early_skip_junk_chem = False
    if getattr(cfg, 'no_enrich_procedures', None):
        cfg.enrich_procedures = False
    elif cfg.enrich_procedures is None:
        # Default on when curated procedure KB is present (offline OpenAlex optional).
        cfg.enrich_procedures = procedures_available(kb_path=cfg.procedure_kb)
    if cfg.approximate and cfg.similarity_threshold is None:
        cfg.similarity_threshold = 0.7
    if cfg.similarity_threshold is None:
        cfg.similarity_threshold = 1.0

    smiles_list = _load_smiles(cfg)
    logging.info(f'Synthesizing pathways for {len(smiles_list)} molecule(s)')
    if cfg.n_workers > 1:
        logging.info(f'Using {cfg.n_workers} process workers')
    if cfg.similarity_threshold < 1.0 - 1e-12:
        logging.info(
            f'Approximate mode enabled (similarity threshold={cfg.similarity_threshold})'
        )
    if cfg.force_uncharge:
        logging.info(
            'Force uncharge enabled (search on ChargeParent; score vs original, '
            f'threshold={cfg.transform_similarity_threshold})'
        )
    if cfg.simplify_peroxides:
        logging.info(
            f'Peroxide simplification enabled (O–O → {cfg.peroxide_mode})'
        )
    if cfg.allow_proposed_reagents:
        logging.info(
            'Proposed reagents enabled (non-catalog halide / small alcohol-thiol '
            f'leaves up to {cfg.max_proposed_reagent_atoms} atoms)'
        )
    if cfg.early_skip:
        logging.info(
            'Early skip enabled (prefilter junk/zero-disc; escalate past depth 2 '
            f'only with catalog/proposed leaf or '
            f'>={cfg.early_skip_min_useful_discs} useful discs; '
            f'exclude={list(cfg.early_skip_exclude_reactions)})'
        )

    matcher = None
    if cfg.enrich_procedures:
        if not procedures_available(kb_path=cfg.procedure_kb):
            logging.warning(
                f'Procedure enrichment requested but KB missing: {cfg.procedure_kb}'
            )
            cfg.enrich_procedures = False
        else:
            matcher = ProcedureMatcher(
                kb_path=cfg.procedure_kb,
                prepared_dir=cfg.openalex_prepared_dir,
            )
            logging.info(
                'Procedure enrichment enabled (curated KB + local OpenAlex prepared cache; '
                'offline)'
            )

    synthesizer = LocalRetrosynthesizer(
        reaction_path=cfg.reaction_path,
        building_blocks_path=cfg.building_blocks_path,
        reaction_to_compound_path=cfg.reaction_to_compound_path,
        max_depth=cfg.max_depth,
        max_routes_per_mol=cfg.max_routes_per_mol,
        require_role_membership=cfg.require_role_membership,
        max_disconnections=cfg.max_disconnections,
        similarity_threshold=cfg.similarity_threshold,
        allow_proposed_reagents=cfg.allow_proposed_reagents,
        max_proposed_reagent_atoms=cfg.max_proposed_reagent_atoms,
        force_uncharge=cfg.force_uncharge,
        simplify_peroxides=cfg.simplify_peroxides,
        peroxide_mode=cfg.peroxide_mode,
        transform_similarity_threshold=cfg.transform_similarity_threshold,
        seed_trivial_catalog=cfg.seed_trivial_catalog,
        n_workers=cfg.n_workers,
        use_prepared_bb_cache=cfg.use_prepared_bb_cache,
        disconnect_cache_path=cfg.disconnect_cache,
        reaction_pack_path=cfg.reaction_pack,
        early_skip=cfg.early_skip,
        early_skip_junk_chem=cfg.early_skip_junk_chem,
        early_skip_exclude_reactions=cfg.early_skip_exclude_reactions,
        early_skip_min_useful_discs=cfg.early_skip_min_useful_discs,
    )

    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _maybe_enrich(chunk: pd.DataFrame) -> pd.DataFrame:
        if not cfg.enrich_procedures or matcher is None:
            return chunk
        return attach_procedures_to_dataframe(chunk, matcher=matcher)

    df_all = None
    if cfg.stream_output:
        header_written = False
        n_rows = 0
        n_found_queries = 0
        chunks: list[pd.DataFrame] = []

        def _on_done(_i, _smi, routes):
            nonlocal header_written, n_rows, n_found_queries
            chunk = _maybe_enrich(routes_to_dataframe(routes))
            chunk.to_csv(out, mode='a' if header_written else 'w', index=False, header=not header_written)
            header_written = True
            n_rows += len(chunk)
            if any(r.found for r in routes):
                n_found_queries += 1
            if cfg.html:
                chunks.append(chunk)

        # Truncate existing file before streaming appends.
        if out.is_file():
            out.unlink()
        synthesizer.synthesize_many(smiles_list, on_molecule_done=_on_done)
        n_queries = len(smiles_list)
        logging.info(
            f'Wrote {n_rows} route row(s) for {n_queries} quer{"y" if n_queries == 1 else "ies"} '
            f'({n_found_queries} with at least one pathway) to {out}'
        )
        if cfg.html and chunks:
            df_all = pd.concat(chunks, ignore_index=True)
    else:
        routes = synthesizer.synthesize_many(smiles_list)
        df_all = _maybe_enrich(routes_to_dataframe(routes))
        df_all.to_csv(out, index=False)
        n_queries = df_all['query_smiles'].nunique() if len(df_all) else 0
        n_found = (
            int(df_all.loc[df_all['found'].astype(bool), 'query_smiles'].nunique())
            if len(df_all)
            else 0
        )
        logging.info(
            f'Wrote {len(df_all)} route row(s) for {n_queries} quer{"y" if n_queries == 1 else "ies"} '
            f'({n_found} with at least one pathway) to {out}'
        )

    if cfg.html and df_all is not None and len(df_all):
        html_path = Path(cfg.html)
        reaction_paths = [Path(cfg.reaction_path)]
        extras = Path(cfg.reaction_path).with_name('reactions_retrosynthesis_extra.json')
        if extras.is_file() and extras not in reaction_paths:
            reaction_paths.append(extras)
        path = write_embedded_retrosynthesis_html(
            df_all,
            html_path,
            reaction_paths=reaction_paths,
            top_k=cfg.max_routes_per_mol,
            only_found=True,
            title='LDDM retrosynthesis report',
            enrich_procedures=bool(cfg.enrich_procedures),
            procedure_matcher=matcher,
        )
        logging.info(f'Wrote embedded HTML report to {path}')


if __name__ == '__main__':
    main()
