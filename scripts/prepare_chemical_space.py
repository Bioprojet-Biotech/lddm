"""Prepare building-block fingerprints and reaction-role mappings for LDDM."""
import argparse
import json
from pathlib import Path

import pandas as pd

from lddm.reactions.prepare_space import prepare_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--building-blocks', type=Path, required=True, help='CSV with id and smiles columns')
    parser.add_argument('--reactions', type=Path, required=True, help='JSON list of reaction SMARTS definitions')
    parser.add_argument('--memberships', type=Path, help='Optional CSV of curated reaction-role assignments')
    parser.add_argument('--output', type=Path, default=Path('data/chemical_spaces/custom'))
    parser.add_argument(
        '--drop-empty-roles', action='store_true',
        help='Drop reactions with no building blocks for a reactant role instead of failing',
    )
    args = parser.parse_args()
    memberships = (pd.read_csv(args.memberships, dtype={'reaction_id': str, 'building_block_id': str})
                   if args.memberships else None)
    prepare_space(pd.read_csv(args.building_blocks, dtype={'id': str, 'smiles': str}),
                  json.loads(args.reactions.read_text()), args.output, memberships,
                  drop_empty_roles=args.drop_empty_roles)


if __name__ == '__main__':
    main()
