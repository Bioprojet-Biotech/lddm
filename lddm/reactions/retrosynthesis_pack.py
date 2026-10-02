"""Offline retrosynthesis reaction pack: reverse SMARTS + feature→reaction index.

Build once with ``scripts/prepare_retrosynthesis_pack.py``. At search time the
pack lets ``LocalRetrosynthesizer`` skip reverse-SMARTS discovery and restrict
``_disconnections`` to templates whose product motifs are present on the mol.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from lddm.reactions.chemical_space_format import reaction_molecularity

PACK_VERSION = 1

# Motifs detected on query molecules. Templates declare a subset as
# ``required_features`` (necessary conditions inferred from product SMARTS).
FEATURE_SMARTS: Dict[str, str] = {
    'aryl_alkyl_ether': '[c,n;a][O,S][#6;!$(C=[O,S,N])]',
    'diaryl_ether': '[c,n;a][O,S][c,n;a]',
    'amide': '[#6]C(=O)[#7]',
    'sulfonamide': '[#16](=O)(=O)[#7]',
    'sulfinamide': '[#16](=O)[#7]',
    'ester': '[#6]C(=O)O[#6]',
    'imine': '[CX3]=[NX2]',
    'enamine': '[C]=[C][N]',
    'disulfide': '[#16]-[#16]',
    'benzoxazinone_13': '[c]1[c]C(=O)N[CH2]O1',
    'benzoxazine_14': '[c]1[c]C(=O)N[C][CH2]O1',
    'urea': '[#7]C(=O)[#7]',
    'nitro': '[N+](=O)[O-]',
    'boronate': '[#5]',
    'alkyne': 'C#C',
    'thiol_or_sulfide': '[#16]',
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def reaction_sources_fingerprint(reaction_path: Path) -> str:
    """Hash main retrosynthesis JSON + optional extras sidecar."""
    paths = [reaction_path]
    extras = reaction_path.with_name('reactions_retrosynthesis_extra.json')
    if extras.is_file():
        paths.append(extras)
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(b'\0')
        h.update(_file_sha256(path).encode())
        h.update(b'\0')
    return h.hexdigest()


def default_pack_path(reaction_path: Path | str) -> Path:
    return Path(reaction_path).with_name('retrosynthesis_pack.pkl')


def load_reaction_definitions(reaction_path: Path) -> List[dict]:
    with open(reaction_path) as f:
        reaction_defs = json.load(f)
    extras_path = reaction_path.with_name('reactions_retrosynthesis_extra.json')
    if not extras_path.is_file():
        return reaction_defs
    with open(extras_path) as f:
        extras = json.load(f)
    seen_ids = {str(r['id']) for r in reaction_defs}
    seen_smarts = {r['reaction'] for r in reaction_defs}
    for reaction in extras:
        rid = str(reaction['id'])
        smarts = reaction['reaction']
        if rid in seen_ids or smarts in seen_smarts:
            continue
        reaction_defs.append(reaction)
        seen_ids.add(rid)
        seen_smarts.add(smarts)
    return reaction_defs


def _compile_feature_queries() -> Dict[str, Chem.Mol]:
    out: Dict[str, Chem.Mol] = {}
    for name, smarts in FEATURE_SMARTS.items():
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            logging.warning(f'Invalid feature SMARTS for {name}: {smarts}')
            continue
        out[name] = mol
    return out


def mol_feature_names(
    mol: Chem.Mol, feature_queries: Optional[Dict[str, Chem.Mol]] = None
) -> Tuple[str, ...]:
    queries = feature_queries if feature_queries is not None else _compile_feature_queries()
    hits: List[str] = []
    for name, q in queries.items():
        try:
            if mol.HasSubstructMatch(q):
                hits.append(name)
        except Exception:
            continue
    return tuple(hits)


def infer_required_features(product_smarts: str) -> Tuple[str, ...]:
    """Conservative necessary motifs inferred from the forward product SMARTS."""
    prod = product_smarts.strip()
    req: List[str] = []

    def add(name: str) -> None:
        if name not in req:
            req.append(name)

    # Prefer concrete ring / FG patterns before generic ether/amide.
    if re.search(r'C\(=O\)N.+O1|C\(=O\)\[N.+\[O:|C\(=O\)\[N:3]\[CH2:4]\[O', prod):
        if 'CH2' in prod or '[CH2' in prod:
            add('benzoxazinone_13')
    if re.search(
        r'C\(=O\).{0,40}O:5\]1|\[c:1\]1\[c:2\].{0,30}\[CH2:4]\[O:5\]1', prod
    ):
        if '[C:7]' in prod or '[C:3]' in prod:
            add('benzoxazine_14')

    if 'S(=O)(=O)' in prod or '$(S(=O)(=O)' in prod:
        add('sulfonamide')
    elif '[S' in prod and prod.count('=[O') >= 2:
        add('sulfonamide')
    elif 'S(=O)' in prod or '$(S=O)' in prod:
        add('sulfinamide')

    # Amide / urea via simple substring / safe regexes (avoid nested char classes).
    has_n_c_o = '[N' in prod and '][C' in prod and '(=[O' in prod
    has_c_o_n = (
        'C(=O)N' in prod
        or 'C(=O)[N' in prod
        or ('](=[O' in prod and ('][N' in prod or '][#7' in prod))
    )
    if has_n_c_o or has_c_o_n:
        if re.search(r'\[N[^\]]*\]\[C[^\]]*\]\(=[O[^\]]*\]\)\[(?:N|#7)', prod):
            add('urea')
        else:
            add('amide')

    if 'C(=O)O' in prod or 'C(=O)[O' in prod or ('](=[O' in prod and '][O' in prod):
        if 'amide' not in req and 'urea' not in req:
            add('ester')

    if re.search(r'\[c,n;a:1\]\[O|\[c:1\]\[O|\]\[O,S:2\]\[c|\]\[O:2\]\[c', prod):
        if re.search(r'\]\[c,n;a|\]\[c:', prod):
            add('diaryl_ether')
        else:
            add('aryl_alkyl_ether')
    elif re.search(r'\[#6:1\]\[O,S:2\]\[#6|\[O,S:2\]\[#6', prod):
        add('aryl_alkyl_ether')

    if ('=[' in prod and '[N' in prod) and 'C(=O)' not in prod and '(=[O' not in prod:
        add('imine')
    if '[S:2][S:' in prod or '][S:2][S:' in prod or 'S][S' in prod:
        add('disulfide')
    if 'C#C' in prod or '[C]#[C]' in prod:
        add('alkyne')
    if '[#5' in prod or '[B' in prod:
        add('boronate')
    if '[N+](=O)[O-]' in prod or 'N(=O)=O' in prod:
        add('nitro')
    if (
        ('[#16' in prod or ']S[' in prod or prod.startswith('[S') or '[S:' in prod)
        and 'sulfonamide' not in req
        and 'disulfide' not in req
    ):
        add('thiol_or_sulfide')

    return tuple(req)


def _build_entry(reaction: dict) -> Optional[dict]:
    # Late import avoids a circular dependency with retrosynthesis.py.
    from lddm.reactions.retrosynthesis import (
        _explicit_halogen_reverse_variants,
        _fallback_reverse_smarts,
        _reverse_smarts,
    )

    rid = str(reaction['id'])
    smarts = reaction['reaction']
    explicit_hs = bool(reaction.get('explicit_hs', False))
    molecularity = reaction_molecularity(smarts)
    if molecularity is None:
        return None
    primary: List[str] = []
    try:
        base_rev = _reverse_smarts(smarts)
        primary.append(base_rev)
        primary.extend(_explicit_halogen_reverse_variants(base_rev))
    except ValueError:
        primary = []
    fallbacks = _fallback_reverse_smarts(smarts)
    # Dedup while preserving order.
    seen: Set[str] = set()
    primary_u: List[str] = []
    for s in primary:
        if s not in seen:
            seen.add(s)
            primary_u.append(s)
    fallback_u: List[str] = []
    for s in fallbacks:
        if s not in seen:
            seen.add(s)
            fallback_u.append(s)
    if not primary_u and not fallback_u:
        return None
    product_smarts = smarts.split('>>')[1].strip()
    return {
        'id': rid,
        'name': reaction.get('name', rid),
        'forward_smarts': smarts,
        'explicit_hs': explicit_hs,
        'molecularity': molecularity,
        'primary_reverse_smarts': primary_u,
        'fallback_reverse_smarts': fallback_u,
        'product_smarts': product_smarts,
        'required_features': list(infer_required_features(product_smarts)),
        'source': reaction.get('source'),
        'covers': reaction.get('covers'),
    }


def build_pack(reaction_path: Path | str) -> dict:
    reaction_path = Path(reaction_path)
    defs = load_reaction_definitions(reaction_path)
    reactions: Dict[str, dict] = {}
    feature_to_reactions: Dict[str, List[str]] = {k: [] for k in FEATURE_SMARTS}
    always_try: List[str] = []
    skipped = 0
    for reaction in defs:
        entry = _build_entry(reaction)
        if entry is None:
            skipped += 1
            continue
        rid = entry['id']
        reactions[rid] = entry
        req = entry['required_features']
        if not req:
            always_try.append(rid)
        else:
            for feat in req:
                if feat in feature_to_reactions:
                    feature_to_reactions[feat].append(rid)
                else:
                    # Unknown feature name — treat as always-try to stay safe.
                    always_try.append(rid)
                    break
    pack = {
        'version': PACK_VERSION,
        'source_fp': reaction_sources_fingerprint(reaction_path),
        'reaction_path': str(reaction_path),
        'feature_smarts': dict(FEATURE_SMARTS),
        'reactions': reactions,
        'feature_to_reactions': feature_to_reactions,
        'always_try': always_try,
        'n_skipped': skipped,
    }
    return pack


def save_pack(pack: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_pack(path: Path | str) -> Optional[dict]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path, 'rb') as f:
            pack = pickle.load(f)
    except Exception as e:
        logging.warning(f'Ignoring unreadable retrosynthesis pack {path}: {e}')
        return None
    if not isinstance(pack, dict) or pack.get('version') != PACK_VERSION:
        return None
    return pack


def pack_is_fresh(pack: dict, reaction_path: Path | str) -> bool:
    return pack.get('source_fp') == reaction_sources_fingerprint(Path(reaction_path))


def candidate_reaction_ids(
    mol_features: Iterable[str],
    *,
    always_try: Sequence[str],
    required_features_by_rid: Dict[str, Sequence[str]],
    reaction_order: Sequence[str],
) -> List[str]:
    """Ordered reaction ids whose required motifs are all present on the molecule."""
    feats = set(mol_features)
    selected: Set[str] = set(always_try)
    for rid in reaction_order:
        if rid in selected:
            continue
        req = required_features_by_rid.get(rid) or ()
        if req and set(req).issubset(feats):
            selected.add(rid)
    return [rid for rid in reaction_order if rid in selected]
