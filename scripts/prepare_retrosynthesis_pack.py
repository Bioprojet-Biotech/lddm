#!/usr/bin/env python3
"""Build an offline retrosynthesis pack (reverse SMARTS + feature→reaction index).

Example::

    python scripts/prepare_retrosynthesis_pack.py \\
        --reaction-path data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lddm.reactions.retrosynthesis_pack import (
    build_pack,
    default_pack_path,
    save_pack,
)
from lddm.utils import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--reaction-path',
        type=str,
        default='data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis.json',
        help='Primary retrosynthesis reactions JSON (extras sidecar auto-merged)',
    )
    p.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output pickle (default: <reaction_dir>/retrosynthesis_pack.pkl)',
    )
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    setup_logging(args.verbose)

    reaction_path = Path(args.reaction_path)
    out = Path(args.output) if args.output else default_pack_path(reaction_path)
    pack = build_pack(reaction_path)
    save_pack(pack, out)
    n = len(pack['reactions'])
    n_always = len(pack['always_try'])
    n_gated = n - n_always
    logging.info(
        f'Wrote retrosynthesis pack: {out} '
        f'({n} reactions, {n_gated} feature-gated, {n_always} always-try, '
        f'{pack["n_skipped"]} skipped)'
    )


if __name__ == '__main__':
    main()
