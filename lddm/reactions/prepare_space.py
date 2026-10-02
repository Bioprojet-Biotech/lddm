"""Prepare building-block fingerprints and reaction-role mappings for LDDM."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from lddm.reactions.chemical_space_format import is_bimolecular_reaction
from lddm.reactions.process_reactions import ReactionsProcessor


def prepare_space(blocks, reactions, output, memberships=None, drop_empty_roles=False):
    if not {'id', 'smiles'}.issubset(blocks.columns):
        raise ValueError('Building-block CSV requires id and smiles columns')
    # Preserve optional multi-source metadata columns when present.
    meta_cols = [
        c
        for c in ('provider', 'sources', 'n_sources', 'priority')
        if c in blocks.columns
    ]
    keep_cols = ['id', 'smiles', *meta_cols]
    blocks = blocks[keep_cols].copy()
    if blocks.empty or blocks[['id', 'smiles']].isna().any().any():
        raise ValueError('Building blocks must contain nonempty IDs and SMILES')
    blocks['id'] = blocks['id'].astype(str)
    if blocks.id.duplicated().any() or blocks.id.str.strip().eq('').any():
        raise ValueError('Building-block IDs must be unique and nonempty')
    fpgen = AllChem.GetMorganGenerator(2, fpSize=2048)
    molecules = []
    for row in blocks.itertuples():
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(f'Invalid or disconnected building block: {row.id}')
        molecules.append(Chem.RemoveHs(mol))
    blocks['smiles'] = [Chem.MolToSmiles(mol) for mol in molecules]
    if blocks.smiles.duplicated().any():
        raise ValueError('Duplicate canonical SMILES: keep one ID per building block')
    blocks['synthon_smiles'] = blocks.smiles
    blocks['completed'] = False
    blocks['fp'] = [fpgen.GetFingerprint(mol) for mol in molecules]

    definitions = []
    for reaction in reactions:
        if not {'id', 'reaction'}.issubset(reaction):
            raise ValueError('Each reaction requires id and reaction (SMARTS)')
        if not is_bimolecular_reaction(reaction['reaction']):
            raise ValueError('Custom spaces require two reactants and one product per reaction')
        definitions.append({
            'id': str(reaction['id']),
            'name': reaction.get('name', str(reaction['id'])),
            'reaction': reaction['reaction'],
            'explicit_hs': bool(reaction.get('explicit_hs', False)),
            **({'source': reaction['source']} if 'source' in reaction else {}),
        })
    ids = [r['id'] for r in definitions]
    if not ids or len(set(ids)) != len(ids) or any(not name.strip() for name in ids):
        raise ValueError('Provide reactions with unique, nonempty IDs')
    # The processor adds runtime objects to its definitions; keep the JSON serializable.
    processor = ReactionsProcessor(
        [{'id': r['id'], 'reaction': r['reaction'], 'explicit_hs': r['explicit_hs'],
          'name': r.get('name', r['id'])} for r in definitions],
        blocks,
        fp_size=2048,
    )
    if memberships is None:
        processor.load_reactant_to_building_blocks()
    else:
        required = ['reaction_id', 'reactant_role', 'building_block_id']
        if not set(required).issubset(memberships.columns):
            raise ValueError(f'Membership CSV requires {", ".join(required)}')
        if memberships[required].isna().any().any():
            raise ValueError('Membership CSV contains missing values')
        for reaction in processor.reactions.values():
            for reactant in reaction['reactants']:
                reactant.allowed_building_blocks = set()
        for row in memberships[required].drop_duplicates().itertuples(index=False):
            reaction = processor.reactions.get(str(row.reaction_id))
            if reaction is None or row.reactant_role not in (0, 1):
                raise ValueError(f'Unknown reaction or role: {row.reaction_id}, {row.reactant_role}')
            block_id = str(row.building_block_id)
            reactant = reaction['reactants'][int(row.reactant_role)]
            if (block_id not in processor.bb_id_to_smi or
                    not reactant.matches_mol(processor.bb_id_to_smi[block_id])):
                raise ValueError(f'Building block {block_id} does not match the specified reaction role')
            reactant.allowed_building_blocks.add(block_id)

    mapping, rows = {}, []
    kept_definitions = []
    dropped_empty = []
    definition_by_id = {d['id']: d for d in definitions}
    for reaction_id, reaction in processor.reactions.items():
        role_map = {}
        empty_roles = []
        for reactant in reaction['reactants']:
            allowed = reactant.allowed_building_blocks
            if not allowed:
                empty_roles.append(reactant.id)
            else:
                role_map[reactant.id] = set(allowed)
        if empty_roles:
            if drop_empty_roles:
                dropped_empty.append(reaction_id)
                continue
            raise ValueError(f'No building blocks for {reaction_id}, role {empty_roles[0]}')
        mapping[reaction_id] = role_map
        kept_definitions.append(definition_by_id[reaction_id])
        for role, allowed in role_map.items():
            rows.extend({'reaction_id': reaction_id, 'reactant_role': role,
                         'building_block_id': block_id} for block_id in sorted(allowed))
    if not kept_definitions:
        raise ValueError('No reactions remain after role assignment')
    if dropped_empty:
        print(f'Dropped {len(dropped_empty)} reactions with empty reactant roles: '
              f'{", ".join(dropped_empty[:10])}{"..." if len(dropped_empty) > 10 else ""}')

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    blocks.to_pickle(output / 'building_blocks.pkl')
    blocks.drop(columns='fp').to_csv(output / 'building_blocks.csv', index=False)
    (output / 'reactions.json').write_text(json.dumps(kept_definitions, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(output / 'reaction_to_building_blocks.csv', index=False)
    with (output / 'reaction_to_building_blocks.pkl').open('wb') as handle:
        pickle.dump(mapping, handle, protocol=4)
    print(f'Wrote {len(blocks)} building blocks, {len(kept_definitions)} reactions '
          f'and {len(rows)} role memberships to {output}')
