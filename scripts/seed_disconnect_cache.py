#!/usr/bin/env python3
"""Seed a persistent disconnect cache from benchmark SMILES (+ Murcko scaffolds).

Runs retrosynthesis (or disconnect-only) so ranked cuts for queries and their
intermediates are stored in ``disconnect_cache`` for later synthesize runs.

Example::

    python scripts/seed_disconnect_cache.py \\
        --config configs/controlled_generation/synthesize_synspace_reasyn.yml \\
        --mode synthesize \\
        --add-murcko
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from lddm.reactions.retrosynthesis import LocalRetrosynthesizer, _canonicalize
from lddm.utils import merge_args_and_yaml, set_default, setup_logging

RDLogger.DisableLog('rdApp.*')

_DEFAULT_BENCHMARKS = [
    'output/benchmark_zinc50/benchmark_routes.csv',
    'output/benchmark_zinc50_v2/benchmark_routes.csv',
    'output/benchmark_enamine100/benchmark_routes.csv',
    'output/benchmark_enamine100_v2/benchmark_routes.csv',
    'output/benchmark_reasyn8/benchmark_routes.csv',
    'output/benchmark_smoke/benchmark_routes.csv',
]


def _smiles_from_csv(path: Path) -> List[str]:
    df = pd.read_csv(path)
    for col in ('query_smiles', 'searched_smiles', 'smiles', 'SMILES', 'target'):
        if col in df.columns:
            return [str(x) for x in df[col].dropna().astype(str).tolist() if str(x).strip()]
    return [str(x) for x in df.iloc[:, 0].dropna().astype(str).tolist() if str(x).strip()]


def _smiles_from_smi(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        out.append(line.split()[0])
    return out


def _collect_smiles(paths: Iterable[Path]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for path in paths:
        if not path.is_file():
            logging.warning(f'Skip missing input: {path}')
            continue
        if path.suffix.lower() == '.csv':
            raw = _smiles_from_csv(path)
        else:
            raw = _smiles_from_smi(path)
        for smi in raw:
            can = _canonicalize(smi) or smi
            if can in seen:
                continue
            seen.add(can)
            ordered.append(smi)
    return ordered


def _add_murcko(smiles_list: List[str]) -> List[str]:
    seen = {_canonicalize(s) or s for s in smiles_list}
    out = list(smiles_list)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            continue
        can = _canonicalize(core)
        if not can or can in seen:
            continue
        seen.add(can)
        out.append(core)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', nargs='?', type=str, help='Optional synthesize YAML config')
    p.add_argument(
        '--input',
        nargs='*',
        default=None,
        help='Benchmark CSV/SMI paths (default: known output/benchmark_* routes)',
    )
    p.add_argument(
        '--mode',
        choices=('synthesize', 'disconnect'),
        default='synthesize',
        help='synthesize: full search (fills intermediate cuts); disconnect: top-level only',
    )
    p.add_argument(
        '--add-murcko',
        action='store_true',
        help='Also seed Murcko scaffolds of each input molecule',
    )
    p.add_argument('--reaction-path', type=str, default=None)
    p.add_argument('--building-blocks-path', type=str, default=None)
    p.add_argument('--reaction-to-compound-path', type=str, default=None)
    p.add_argument('--disconnect-cache', type=str, default=None)
    p.add_argument('--reaction-pack', type=str, default=None)
    p.add_argument('--max-depth', type=int, default=None)
    p.add_argument('--max-routes-per-mol', type=int, default=None)
    p.add_argument('--limit', type=int, default=None, help='Optional cap on molecules')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    cfg = merge_args_and_yaml(args, config)
    setup_logging(cfg.verbose if hasattr(cfg, 'verbose') else args.verbose)

    set_default(cfg, 'reaction_path', 'data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis.json')
    set_default(cfg, 'building_blocks_path', 'data/chemical_spaces/synspace_reasyn/building_blocks.pkl')
    set_default(
        cfg,
        'reaction_to_compound_path',
        'data/chemical_spaces/synspace_reasyn/reaction_to_building_blocks.pkl',
    )
    set_default(
        cfg,
        'disconnect_cache',
        'data/chemical_spaces/synspace_reasyn/disconnect_cache.pkl',
    )
    set_default(cfg, 'reaction_pack', None)
    set_default(cfg, 'max_depth', 5)
    set_default(cfg, 'max_routes_per_mol', 3)
    set_default(cfg, 'allow_proposed_reagents', True)
    set_default(cfg, 'max_proposed_reagent_atoms', 16)
    set_default(cfg, 'require_role_membership', True)
    set_default(cfg, 'early_skip', True)

    inputs = [Path(x) for x in (args.input or _DEFAULT_BENCHMARKS)]
    smiles_list = _collect_smiles(inputs)
    if args.add_murcko:
        before = len(smiles_list)
        smiles_list = _add_murcko(smiles_list)
        logging.info(f'Added Murcko scaffolds: {before} → {len(smiles_list)} molecules')
    if args.limit is not None:
        smiles_list = smiles_list[: max(0, int(args.limit))]
    if not smiles_list:
        raise SystemExit('No SMILES collected to seed')

    pack_path = cfg.reaction_pack
    if pack_path is None:
        pack_path = Path(cfg.reaction_path).with_name('retrosynthesis_pack.pkl')

    logging.info(
        f'Seeding disconnect cache for {len(smiles_list)} molecule(s) '
        f'(mode={args.mode}) → {cfg.disconnect_cache}'
    )
    rs = LocalRetrosynthesizer(
        reaction_path=cfg.reaction_path,
        building_blocks_path=cfg.building_blocks_path,
        reaction_to_compound_path=cfg.reaction_to_compound_path,
        max_depth=int(cfg.max_depth),
        max_routes_per_mol=int(cfg.max_routes_per_mol),
        require_role_membership=bool(cfg.require_role_membership),
        allow_proposed_reagents=bool(cfg.allow_proposed_reagents),
        max_proposed_reagent_atoms=int(cfg.max_proposed_reagent_atoms),
        early_skip=bool(cfg.early_skip),
        disconnect_cache_path=cfg.disconnect_cache,
        reaction_pack_path=pack_path,
    )

    found = 0
    for i, smi in enumerate(smiles_list, 1):
        if args.mode == 'disconnect':
            can = _canonicalize(smi) or smi
            discs = rs._disconnections(can)
            logging.info(f'[{i}/{len(smiles_list)}] disconnects={len(discs)} {can[:60]}')
        else:
            routes = rs.synthesize(smi)
            ok = any(r.found for r in routes)
            found += int(ok)
            logging.info(
                f'[{i}/{len(smiles_list)}] found={ok} routes={len(routes)} '
                f'cache={len(rs._disconnect_cache)} {smi[:50]}'
            )
        if i % 10 == 0:
            rs.save_disconnect_cache()

    rs.save_disconnect_cache()
    logging.info(
        f'Done. disconnect_cache entries={len(rs._disconnect_cache)} '
        f'found={found}/{len(smiles_list)} → {cfg.disconnect_cache}'
    )


if __name__ == '__main__':
    main()
