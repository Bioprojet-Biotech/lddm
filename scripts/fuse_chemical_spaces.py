"""Fuse the current (SynSpace) chemical space with ReaSyn / SynFormer data.

ReaSyn ships building blocks as a SMILES text file and 115 SynFormer reaction
templates (uni-, bi-, and tri-molecular).

LDDM **forward** synthesizable design still requires bimolecular reactions
(two reactants, one product), so ``reactions.json`` stays bimolecular-only.

Local **retrosynthesis** (``scripts/synthesize.py``) is not tied to that
constraint: ``reactions_retrosynthesis.json`` keeps uni-/bi-/trimolecular
single-product templates.

Building blocks are deduplicated by canonical SMILES with provider priority
(synspace preferred over reasyn as primary id; all origins kept in ``sources``).
Pass ``--prepare`` to run fingerprinting and role assignment via
``prepare_space`` on the bimolecular set.

Example
-------
::

    python scripts/fuse_chemical_spaces.py \\
        --current-building-blocks data/synspace/building_blocks.csv \\
        --current-reactions data/synspace/reactions.json \\
        --reasyn-building-blocks path/to/building_blocks.txt \\
        --reasyn-reactions path/to/comprehensive.txt \\
        --output data/chemical_spaces/synspace_reasyn \\
        --prepare

ReaSyn reaction templates:
https://github.com/wenhao-gao/synformer/tree/main/data/rxn_templates
ReaSyn data prep:
https://github.com/NVIDIA-Digital-Bio/reasyn
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from lddm.reactions.bb_catalog import load_bb_table, merge_bb_catalogs, provider_rank
from lddm.reactions.chemical_space_format import (
    is_bimolecular_reaction,
    is_retrosynthesis_reaction,
    reaction_molecularity,
    reactant_count,
)
from lddm.reactions.prepare_space import prepare_space


def load_building_blocks_csv(path: Path) -> pd.DataFrame:
    """Load BB CSV and attach provider / sources / priority metadata."""
    return load_bb_table(path)


def load_building_blocks_txt(path: Path, id_prefix: str = 'reasyn') -> pd.DataFrame:
    """Load ReaSyn-style building blocks (optional SMILES header, one SMILES/line).

    Also accepts an ``id,smiles`` CSV saved with a ``.txt`` extension.
    """
    text = path.read_text()
    first = next((line.strip() for line in text.splitlines() if line.strip()), '')
    if first.lower().replace(' ', '') in {'id,smiles', 'id,smi'} or (
            first.lower().startswith('id,') and 'smiles' in first.lower()):
        return load_bb_table(path, default_provider=id_prefix)

    records = []
    for line in text.splitlines():
        smi = line.strip()
        if not smi or smi.lower() in {'smiles', 'smi', 'canonical_smiles'}:
            continue
        block_id = f'{id_prefix}_{len(records):05d}'
        records.append(
            {
                'id': block_id,
                'smiles': smi,
                'provider': id_prefix,
                'sources': f'{id_prefix}:{block_id}',
                'n_sources': 1,
                'priority': provider_rank(id_prefix),
            }
        )
    if not records:
        raise ValueError(f'No building blocks found in {path}')
    return pd.DataFrame(records)


def load_reactions_json(path: Path) -> list[dict]:
    reactions = json.loads(path.read_text())
    if not isinstance(reactions, list) or not reactions:
        raise ValueError(f'{path} must be a nonempty JSON list of reactions')
    return reactions


def _make_reaction_entry(
    reaction_id: str,
    smarts: str,
    *,
    name: str | None = None,
    explicit_hs: bool = False,
    source: str = 'reasyn',
) -> dict:
    mol = reaction_molecularity(smarts)
    return {
        'id': reaction_id,
        'name': name or reaction_id,
        'reaction': smarts,
        'explicit_hs': explicit_hs,
        'source': source,
        'molecularity': mol,
    }


def load_reasyn_reactions(
    path: Path,
    id_prefix: str = 'reasyn',
    *,
    for_retrosynthesis: bool = False,
) -> tuple[list[dict], Counter]:
    """Parse SynFormer/ReaSyn ``comprehensive.txt``.

    Parameters
    ----------
    for_retrosynthesis
        If True, keep uni-/bi-/trimolecular single-product templates.
        If False (default), keep bimolecular only (LDDM forward spaces).
    """
    kept: list[dict] = []
    stats: Counter = Counter()
    for line in path.read_text().splitlines():
        smarts = line.strip()
        if not smarts or smarts.startswith('#'):
            continue
        n = reactant_count(smarts)
        if n is None:
            stats['invalid'] += 1
            continue
        if for_retrosynthesis:
            if not is_retrosynthesis_reaction(smarts):
                if n == 1:
                    stats['unimolecular_dropped'] += 1
                elif n >= 3:
                    stats['trimolecular_dropped'] += 1
                else:
                    stats['unsupported_dropped'] += 1
                continue
            label = {1: 'unimolecular_kept', 2: 'bimolecular_kept', 3: 'trimolecular_kept'}[n]
            stats[label] += 1
        else:
            if n == 1:
                stats['unimolecular_dropped'] += 1
                continue
            if n >= 3:
                stats['trimolecular_dropped'] += 1
                continue
            if not is_bimolecular_reaction(smarts):
                stats['unsupported_bimolecular_dropped'] += 1
                continue
            stats['bimolecular_kept'] += 1
        reaction_id = f'{id_prefix}_{len(kept):03d}'
        kept.append(_make_reaction_entry(reaction_id, smarts, source='reasyn'))
    return kept, stats


def merge_building_blocks(
    current: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    skip_invalid: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Union building blocks by canonical SMILES with provider priority + origins.

    Prefers the better-priority primary id (synspace before reasyn) while
    recording all catalog origins in ``sources``.
    """
    # Ensure provider columns exist for priority-aware merge.
    for frame, default in ((current, None), (extra, 'reasyn')):
        if 'provider' not in frame.columns:
            from lddm.reactions.bb_catalog import infer_provider

            frame['provider'] = [
                infer_provider(str(i), default) for i in frame['id']
            ]
        if 'priority' not in frame.columns:
            frame['priority'] = [provider_rank(p) for p in frame['provider']]
        if 'sources' not in frame.columns:
            frame['sources'] = [
                f'{p}:{i}' for p, i in zip(frame['provider'], frame['id'])
            ]

    merged, stats = merge_bb_catalogs(
        [current, extra], skip_invalid=skip_invalid
    )
    # Keep legacy keys expected by callers.
    stats = {
        'current': len(current),
        'extra': len(extra),
        'union': stats['union'],
        'skipped_invalid': stats['skipped_invalid'],
        'renamed_id_collisions': stats['renamed_id_collisions'],
        'smiles_collisions': stats['smiles_collisions'],
        'multi_source': stats['multi_source'],
        'single_source': stats['single_source'],
        'by_provider': stats['by_provider'],
    }
    return merged, stats


def merge_reactions(
    current: list[dict],
    reasyn: list[dict],
    *,
    for_retrosynthesis: bool = False,
) -> tuple[list[dict], dict]:
    """Union reactions by SMARTS; prefer current IDs."""
    merged: list[dict] = []
    seen_smarts: set[str] = set()
    used_ids: set[str] = set()
    dropped_current = 0
    accept = is_retrosynthesis_reaction if for_retrosynthesis else is_bimolecular_reaction
    for source_name, reactions in (('current', current), ('reasyn', reasyn)):
        for reaction in reactions:
            if not {'id', 'reaction'}.issubset(reaction):
                raise ValueError(f'Reaction missing id/reaction fields: {reaction}')
            smarts = reaction['reaction']
            if not accept(smarts):
                if source_name == 'current':
                    dropped_current += 1
                continue
            if smarts in seen_smarts:
                continue
            reaction_id = str(reaction['id'])
            if reaction_id in used_ids:
                reaction_id = f'{reaction_id}__{source_name}'
            entry = _make_reaction_entry(
                reaction_id,
                smarts,
                name=reaction.get('name', reaction_id),
                explicit_hs=bool(reaction.get('explicit_hs', False)),
                source=reaction.get('source', source_name),
            )
            merged.append(entry)
            seen_smarts.add(smarts)
            used_ids.add(reaction_id)
    mol_counts = Counter(r.get('molecularity') for r in merged)
    stats = {
        'current_in': len(current),
        'reasyn_in': len(reasyn),
        'union': len(merged),
        'current_dropped': dropped_current,
        'molecularity': dict(mol_counts),
    }
    return merged, stats


def write_fusion_inputs(
    blocks: pd.DataFrame,
    reactions_forward: list[dict],
    reactions_retro: list[dict],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(output / 'building_blocks.csv', index=False)
    (output / 'reactions.json').write_text(json.dumps(reactions_forward, indent=2) + '\n')
    (output / 'reactions_retrosynthesis.json').write_text(
        json.dumps(reactions_retro, indent=2) + '\n'
    )
    mol_counts = Counter(r.get('molecularity') for r in reactions_retro)
    report = {
        'n_building_blocks': len(blocks),
        'n_reactions_forward_bimolecular': len(reactions_forward),
        'n_reactions_retrosynthesis': len(reactions_retro),
        'retrosynthesis_molecularity': dict(mol_counts),
        'forward_reaction_ids': [r['id'] for r in reactions_forward],
        'retrosynthesis_reaction_ids': [r['id'] for r in reactions_retro],
    }
    (output / 'fusion_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'retrosynthesis_filter_report.json').write_text(
        json.dumps(
            {
                'n_retrosynthesis_compatible': len(reactions_retro),
                'molecularity': dict(mol_counts),
                'note': (
                    'Uni-/bi-/trimolecular single-product SMARTS for LocalRetrosynthesizer. '
                    'LDDM forward generation still uses bimolecular reactions.json only.'
                ),
            },
            indent=2,
        )
        + '\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--current-building-blocks', type=Path, required=True,
        help='Current-space building-block CSV (id, smiles), e.g. data/synspace/building_blocks.csv',
    )
    parser.add_argument(
        '--current-reactions', type=Path, required=True,
        help='Current-space reactions JSON, e.g. data/synspace/reactions.json',
    )
    parser.add_argument(
        '--reasyn-building-blocks', type=Path, required=True,
        help='ReaSyn building_blocks.txt (SMILES header + one SMILES per line)',
    )
    parser.add_argument(
        '--reasyn-reactions', type=Path, required=True,
        help='SynFormer/ReaSyn comprehensive.txt reaction templates',
    )
    parser.add_argument(
        '--output', type=Path, default=Path('data/chemical_spaces/synspace_reasyn'),
        help='Output directory for fused CSV/JSON (and prepared artifacts if --prepare)',
    )
    parser.add_argument(
        '--reasyn-id-prefix', default='reasyn',
        help='Prefix for ReaSyn-derived building-block and reaction IDs',
    )
    parser.add_argument(
        '--skip-invalid-bbs', action='store_true',
        help='Skip invalid/disconnected ReaSyn SMILES instead of failing',
    )
    parser.add_argument(
        '--prepare', action='store_true',
        help='Run prepare_space on the fused bimolecular inputs (fps + role mappings)',
    )
    args = parser.parse_args()

    current_blocks = load_building_blocks_csv(args.current_building_blocks)
    current_reactions = load_reactions_json(args.current_reactions)
    reasyn_blocks = load_building_blocks_txt(
        args.reasyn_building_blocks, id_prefix=args.reasyn_id_prefix,
    )
    reasyn_bi, rxn_filter_bi = load_reasyn_reactions(
        args.reasyn_reactions, id_prefix=args.reasyn_id_prefix, for_retrosynthesis=False,
    )
    reasyn_retro, rxn_filter_retro = load_reasyn_reactions(
        args.reasyn_reactions, id_prefix=args.reasyn_id_prefix, for_retrosynthesis=True,
    )

    blocks, block_stats = merge_building_blocks(
        current_blocks, reasyn_blocks, skip_invalid=args.skip_invalid_bbs,
    )
    reactions_forward, reaction_stats_fwd = merge_reactions(
        current_reactions, reasyn_bi, for_retrosynthesis=False,
    )
    # Retrosynthesis file: reuse forward bimolecular IDs (role-map aligned), then
    # append uni/tri (and any extra bi) templates not already present.
    seen_smarts = {r['reaction'] for r in reactions_forward}
    used_ids = {r['id'] for r in reactions_forward}
    reactions_retro = [dict(r) for r in reactions_forward]
    for reaction in reasyn_retro:
        smarts = reaction['reaction']
        if smarts in seen_smarts:
            continue
        rid = str(reaction['id'])
        if rid in used_ids:
            rid = f'{rid}__retro'
        entry = _make_reaction_entry(
            rid,
            smarts,
            name=reaction.get('name', rid),
            explicit_hs=bool(reaction.get('explicit_hs', False)),
            source=reaction.get('source', 'reasyn'),
        )
        reactions_retro.append(entry)
        seen_smarts.add(smarts)
        used_ids.add(rid)
    reaction_stats_retro = {
        'forward_bi_base': len(reactions_forward),
        'retrosynthesis_total': len(reactions_retro),
        'molecularity': dict(Counter(r.get('molecularity') for r in reactions_retro)),
    }

    if blocks.empty:
        raise SystemExit('Fused building-block set is empty')
    if not reactions_forward:
        raise SystemExit('Fused bimolecular reaction set is empty')
    if not reactions_retro:
        raise SystemExit('Fused retrosynthesis reaction set is empty')

    write_fusion_inputs(blocks, reactions_forward, reactions_retro, args.output)

    print('ReaSyn filter (forward / bimolecular):', dict(rxn_filter_bi))
    print('ReaSyn filter (retrosynthesis):', dict(rxn_filter_retro))
    print('Building-block merge:', block_stats)
    print('Reaction merge (forward):', reaction_stats_fwd)
    print('Reaction merge (retrosynthesis):', reaction_stats_retro)
    print(
        f'Wrote fused inputs: {len(blocks)} building blocks, '
        f'{len(reactions_forward)} forward reactions, '
        f'{len(reactions_retro)} retrosynthesis reactions -> {args.output}'
    )

    if args.prepare:
        # Role maps only for bimolecular LDDM forward space.
        prepare_space(blocks, reactions_forward, args.output, drop_empty_roles=True)
        print(f'Prepared chemical space artifacts in {args.output}')
    else:
        print(
            'Skipped prepare; run:\n'
            f'  python scripts/prepare_chemical_space.py '
            f'--building-blocks {args.output / "building_blocks.csv"} '
            f'--reactions {args.output / "reactions.json"} '
            f'--output {args.output}'
        )


if __name__ == '__main__':
    main()
