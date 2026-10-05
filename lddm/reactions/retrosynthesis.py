"""Local template-based retrosynthesis against SynSpace / custom chemical spaces.

Supports uni-, bi-, and trimolecular single-product templates (retrosynthesis is
not limited to LDDM's bimolecular forward-generation constraint).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, DataStructs, rdChemReactions

RDLogger.DisableLog('rdApp.*')
try:
    rdBase.DisableLog('rdApp.*')
except Exception:
    pass

from lddm.reactions.bb_catalog import infer_provider, provider_rank
from lddm.reactions.chemical_space_format import reaction_molecularity
from lddm.reactions.reaction_utils import (
    ReactionTree,
    get_react_trace_bimolecular,
    get_react_trace_building_block,
    get_react_trace_trimolecular,
    get_react_trace_unimolecular,
    run_compiled_reaction,
    run_reaction_smarts,
)

# Process-pool worker synthesizer (set by initializer).
_WORKER_SYNTH: Optional['LocalRetrosynthesizer'] = None

# Cross-process disconnect cache (Manager.dict proxy); set during synthesize_many.
_SHARED_DISCONNECT_CACHE = None

# Compiled once for proposed-reagent checks.
_OH_SH_QUERY = Chem.MolFromSmarts('[OH,SH]')
_SS_QUERY = Chem.MolFromSmarts('[#16]-[#16]')
_IMINE_QUERY = Chem.MolFromSmarts('[C,c]=[N,n]')
_HETEROARENE_QUERY = Chem.MolFromSmarts('a1aaaaa1')  # 6-member heteroaryl approx via aromatic

# Early-skip junk chemistry (denovo artifacts that burn deep search).
_CUMULENE_QUERY = Chem.MolFromSmarts('[#6]=[#6]=[#6]')
_WEIRD_S_QUERY = Chem.MolFromSmarts('[S;X3,X4]([#8;X2])([#8;X2])')
_SULFONE_QUERY = Chem.MolFromSmarts('[S](=O)(=O)')
_DEFAULT_EARLY_SKIP_EXCLUDE_REACTIONS = ('retro_cc_wurtz',)

# Common small reagents often absent from ZINC BB dumps but needed as leaves.
_TRIVIAL_CATALOG_SMILES: Tuple[str, ...] = (
    'c1ccncc1',
    'Cc1ccncc1',
    'Nc1ccncc1',
    'Nc1ccccn1',
    'Nc1cccnc1',
    'Oc1ccncc1',
    'Oc1ccccn1',
    'Sc1ccncc1',
    'Sc1ccccn1',
    'Clc1ccncc1',
    'Clc1ccccn1',
    'Brc1ccncc1',
    'Brc1ccccn1',
    'Fc1ccncc1',
    'Fc1ccccn1',
    'O=Cc1ccncc1',
    'O=Cc1ccccn1',
    'ClCc1ccncc1',
    'BrCc1ccncc1',
    'ICc1ccncc1',
    'BrCCCC(=O)O',
    'O=C(O)CCCBr',
    'FC(F)(F)c1ccccn1',
    'FC(F)(F)c1ccncc1',
    'NCCO',
    # Glyoxylic acid (alkylidene / Knoevenagel partner for indolenones).
    'O=CC(=O)O',
    # 2-halo-3H-indol-3-ones for thioindolenine acrylic acids.
    'O=C1C(Cl)=Nc2ccccc21',
    'O=C1C(Br)=Nc2ccccc21',
    'Cc1ccc2c(c1)C(=O)C(Cl)=N2',
    'Cc1ccc2c(c1)C(=O)C(Br)=N2',
)

_PREPARED_BB_CACHE_VERSION = 2
# Priority penalty for proposed__ leaves (worse than any catalog provider).
_PROPOSED_BB_PRIORITY = 200


def _canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def _atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol is not None else 0


def _prepared_bb_cache_path(building_blocks_path: Path) -> Path:
    return building_blocks_path.with_name(building_blocks_path.name + '.prepared.pkl')


def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    payload = f'{path.resolve()}|{st.st_mtime_ns}|{st.st_size}'
    return hashlib.sha1(payload.encode()).hexdigest()


def _serialize_disconnects(discs: Sequence['_Disconnect']) -> list:
    return [
        {
            'sort_key': list(d.sort_key),
            'reaction_id': d.reaction_id,
            'precursors': list(d.precursors),
            'product_smi': d.product_smi,
            'similarity': d.similarity,
        }
        for d in discs
    ]


def _deserialize_disconnects(rows: Sequence[dict]) -> List['_Disconnect']:
    return [
        _Disconnect(
            sort_key=tuple(row['sort_key']),
            reaction_id=row['reaction_id'],
            precursors=tuple(row['precursors']),
            product_smi=row['product_smi'],
            similarity=row['similarity'],
        )
        for row in rows
    ]


def _init_synth_worker(config: dict, shared_disconnect_cache=None) -> None:
    global _WORKER_SYNTH, _SHARED_DISCONNECT_CACHE
    RDLogger.DisableLog('rdApp.*')
    try:
        rdBase.DisableLog('rdApp.*')
    except Exception:
        pass
    _SHARED_DISCONNECT_CACHE = shared_disconnect_cache
    _WORKER_SYNTH = LocalRetrosynthesizer(**config)
    _WORKER_SYNTH._shared_disconnect_cache = shared_disconnect_cache


def _worker_synthesize(smiles: str):
    assert _WORKER_SYNTH is not None
    before = set(_WORKER_SYNTH._disconnect_cache.keys())
    try:
        routes = _WORKER_SYNTH.synthesize(smiles)
    finally:
        _WORKER_SYNTH._flush_shared_disconnect_cache()
    # Ship only newly computed disconnections back to the parent process.
    updates = {
        smi: _serialize_disconnects(_WORKER_SYNTH._disconnect_cache[smi])
        for smi in _WORKER_SYNTH._disconnect_cache.keys() - before
    }
    return routes, updates


def _merge_disconnect_updates(
    synth: 'LocalRetrosynthesizer', updates: dict
) -> None:
    if not updates:
        return
    for smi, rows in updates.items():
        if smi in synth._disconnect_cache:
            continue
        try:
            synth._disconnect_cache[smi] = _deserialize_disconnects(rows)
        except Exception:
            continue


def _largest_fragment(smiles: str) -> str:
    """Keep the heaviest organic component from salt / mixture SMILES."""
    if '.' not in smiles:
        return smiles
    best = smiles
    best_n = -1
    for part in smiles.split('.'):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        n = mol.GetNumAtoms()
        if n > best_n:
            best_n = n
            best = part
    return best


def _neutralize_smiles(smiles: str) -> Optional[str]:
    """Protonate carboxylates / deprotonate ammoniums so templates can match."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    changed = False
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge == 0:
            continue
        sym = atom.GetSymbol()
        if sym == 'O' and charge == -1:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
            changed = True
        elif sym == 'N' and charge == 1 and atom.GetTotalDegree() < 4:
            hs = atom.GetTotalNumHs()
            if hs > 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hs - 1)
                changed = True
    if not changed:
        return _canonicalize(smiles)
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return _canonicalize(smiles)


def _force_uncharge_smiles(smiles: str) -> Optional[str]:
    """Stronger charge stripping via RDKit MolStandardize ChargeParent + Uncharger.

    Used so retrosynthesis can run on a neutral parent while the reported
    Tanimoto is computed against the original (possibly charged) query.

    Returns a SMILES with **no** formal charges, or None if neutralization fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    def _still_charged(m: Chem.Mol) -> bool:
        return any(a.GetFormalCharge() != 0 for a in m.GetAtoms())

    if not _still_charged(mol):
        return _canonicalize(smiles)

    candidates: List[Chem.Mol] = []
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        parent = rdMolStandardize.ChargeParent(Chem.Mol(mol))
        uncharged = rdMolStandardize.Uncharger().uncharge(parent)
        candidates.append(uncharged)
    except Exception:
        pass
    mild = _neutralize_smiles(smiles)
    if mild is not None:
        m_mild = Chem.MolFromSmiles(mild)
        if m_mild is not None:
            candidates.append(m_mild)

    # Aggressive: zero every remaining formal charge.
    base = candidates[-1] if candidates else Chem.Mol(mol)
    rw = Chem.RWMol(base)
    for atom in rw.GetAtoms():
        chg = atom.GetFormalCharge()
        if chg == 0:
            continue
        atom.SetFormalCharge(0)
        if atom.GetSymbol() in ('N', 'O', 'S') and chg < 0:
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + (-chg))
        elif atom.GetSymbol() == 'N' and chg > 0 and atom.GetTotalNumHs() >= chg:
            atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - chg))
    candidates.append(rw)

    for cand in candidates:
        try:
            Chem.SanitizeMol(cand)
            if _still_charged(cand):
                continue
            return Chem.MolToSmiles(Chem.RemoveHs(cand))
        except Exception:
            continue
    return None


def _substitute_peroxides(smiles: str, mode: str = 'ether') -> Optional[str]:
    """Replace O–O with a single O (ether) or CH2 to get a synthesizable analog.

    Applies replacements iteratively until no peroxide remains. Prefer connected
    single-component products; return None if substitution fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    oo = Chem.MolFromSmarts('[O][O]')
    if oo is None:
        return _canonicalize(smiles)
    repl = Chem.MolFromSmiles('O' if mode == 'ether' else 'C')
    if repl is None:
        return None

    current = Chem.Mol(mol)
    changed = False
    for _ in range(8):
        if not current.HasSubstructMatch(oo):
            break
        try:
            products = Chem.ReplaceSubstructs(current, oo, repl, replaceAll=False)
        except Exception:
            break
        next_mol = None
        best_n = -1
        for prod in products:
            try:
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(Chem.RemoveHs(prod))
            except Exception:
                continue
            if not smi or '.' in smi:
                # Keep heaviest fragment if replacement disconnected the molecule.
                parts = [p for p in smi.split('.') if p]
                if not parts:
                    continue
                heaviest = max(parts, key=lambda p: (
                    Chem.MolFromSmiles(p).GetNumAtoms()
                    if Chem.MolFromSmiles(p) is not None else 0
                ))
                frag = Chem.MolFromSmiles(heaviest)
                if frag is None:
                    continue
                smi = Chem.MolToSmiles(frag)
                prod = frag
            n = prod.GetNumAtoms()
            if n > best_n:
                best_n = n
                next_mol = prod
        if next_mol is None:
            break
        current = next_mol
        changed = True

    if not changed:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(current))
    except Exception:
        return None


@dataclass
class PreparedQuery:
    """Query after optional uncharge / peroxide simplification for search."""

    input_smiles: str
    reference_smiles: str
    search_smiles: str
    transforms: List[str] = field(default_factory=list)

    @property
    def transformed(self) -> bool:
        return bool(self.transforms) and self.search_smiles != self.reference_smiles


def _prepare_query(
    smiles: str,
    *,
    force_uncharge: bool = True,
    simplify_peroxides: bool = True,
    peroxide_mode: str = 'ether',
) -> Optional[PreparedQuery]:
    """Largest fragment → optional force-uncharge → optional peroxide simplify.

    ``reference_smiles`` keeps the user-facing fragment (charges intact) for
    final Tanimoto; ``search_smiles`` is what the planner expands.
    """
    frag = _largest_fragment(smiles.strip())
    reference = _canonicalize(frag)
    if reference is None:
        return None
    search = reference
    transforms: List[str] = []

    # Peroxide first so subsequent uncharge sees the simplified scaffold.
    if simplify_peroxides:
        mol = Chem.MolFromSmiles(search)
        if mol is not None and mol.HasSubstructMatch(Chem.MolFromSmarts('[O][O]')):
            simplified = _substitute_peroxides(search, mode=peroxide_mode)
            if simplified is not None and simplified != search:
                transforms.append(f'peroxide_to_{peroxide_mode}')
                search = simplified

    if force_uncharge:
        uncharged = _force_uncharge_smiles(search)
        if uncharged is not None and uncharged != search:
            transforms.append('force_uncharge')
            search = uncharged

    search = _canonicalize(search) or search
    return PreparedQuery(
        input_smiles=smiles,
        reference_smiles=reference,
        search_smiles=search,
        transforms=transforms,
    )


def _prepare_query_smiles(smiles: str) -> Optional[str]:
    """Largest fragment → neutralize → canonicalize (legacy helper)."""
    prep = _prepare_query(smiles, force_uncharge=True, simplify_peroxides=False)
    return prep.search_smiles if prep is not None else None


def _reverse_smarts(forward_smarts: str) -> str:
    """Flip a uni-/bi-/trimolecular single-product SMARTS for retrosynthesis."""
    if forward_smarts.count('>>') != 1:
        raise ValueError(f'Expected a single >> in reaction SMARTS: {forward_smarts}')
    reactants, products = forward_smarts.split('>>')
    products = products.strip()
    # Parenthetical multi-product SMARTS, e.g. >>(A.B), are not supported.
    if products.startswith('(') or '.' in products:
        raise ValueError(
            'Retrosynthesis requires a single product component '
            f'(got {forward_smarts})'
        )
    n_reac = reactants.count('.') + 1
    if n_reac < 1 or n_reac > 3:
        raise ValueError(
            'Retrosynthesis supports 1–3 reactants with a single product '
            f'(got {forward_smarts})'
        )
    return f'{products}>>{reactants}'


def _explicit_halogen_reverse_variants(reverse_smarts: str) -> List[str]:
    """Replace OR-halogen product atoms with explicit F/Cl/Br/I.

    RDKit cannot create `[Cl,Br,I]` query atoms as reaction products, so a naive
    reverse of halide-consuming templates yields dummy `*` atoms. Explicit
    halogen variants restore usable electrophile precursors.

    Handles any bracketed permutation of F/Cl/Br/I (e.g. ``[Cl,F,Br,I]``,
    ``[Cl,Br,I,F]``), not only the ``[F,Cl,Br,I]`` / ``[Cl,Br,I]`` spellings.
    ``F`` is emitted only when present in an OR group (SNAr templates).
    """
    bracket_pat = re.compile(r'\[((?:F|Cl|Br|I)(?:,(?:F|Cl|Br|I))+)]')
    variants: List[str] = []
    for halogen in ('I', 'Br', 'Cl', 'F'):  # prefer I > Br > Cl > F
        if '>>' not in reverse_smarts:
            continue
        left, right = reverse_smarts.split('>>', 1)

        def _sub_brackets(side: str, halogen: str = halogen) -> str:
            def repl(match: re.Match) -> str:
                members = match.group(1).split(',')
                if set(members) <= {'F', 'Cl', 'Br', 'I'} and halogen in members:
                    return f'[{halogen}]'
                return match.group(0)

            return bracket_pat.sub(repl, side)

        new_right = _sub_brackets(right)
        for pattern in ('F,Cl,Br,I', 'Cl,F,Br,I', 'Cl,Br,I,F', 'Cl,Br,I', 'Br,I'):
            if pattern in new_right and halogen in pattern.split(','):
                new_right = new_right.replace(pattern, halogen)
        v = f'{left}>>{new_right}'
        if v != reverse_smarts:
            variants.append(v)
    seen = set()
    unique: List[str] = []
    for smarts in variants:
        if smarts not in seen:
            seen.add(smarts)
            unique.append(smarts)
    return unique

def _fallback_reverse_smarts(forward_smarts: str) -> List[str]:
    """Extra reverse templates when the naive SMARTS flip fails to sanitize.

    Complex reactant-side H / recursive constraints often break RDKit's reverse
    application. Map-minimal bond disconnections recover those routes; forward
    verification keeps them honest.

    Aliphatic-only `[C][N]` cuts miss diarylamines / diaryl ethers — aromatic
    `c`/`n` variants are required for typical medicinal-chemistry scaffolds.
    """
    if forward_smarts.count('>>') != 1:
        return []
    reactants, product = forward_smarts.split('>>')
    product = product.strip()
    if '.' in product or product.startswith('('):
        return []
    n_reac = reactants.count('.') + 1
    # Generic bond-cut fallbacks are for bimolecular (and some uni) templates.
    if n_reac > 2:
        return []
    fallbacks: List[str] = []
    # Carbonyl / acid + amine → C–N (amide, reductive amination, ...).
    if n_reac == 2 and '[N' in reactants and (
            '=O' in reactants or '=[OD' in reactants or 'C=O' in reactants
    ):
        fallbacks.append('[C:1][N:2]>>[C:1]=O.[N:2]')
        fallbacks.append('[C:1](=[O:3])[N:2]>>[C:1](=[O:3])[OH].[N:2]')
    # Sulfonamide / sulfinamide: sulfonyl(sulfinyl) halide + amine → S–N.
    # SynSpace sulfon_amide excludes aryl amines (!$(N[c,O])); extras + these
    # fallbacks recover ArSO2NHR after forward verification.
    if n_reac == 2 and '[N' in reactants and (
            'S(=O)(=O)' in reactants or 'S(=O)' in reactants or '$(S(=O)' in reactants
    ):
        fallbacks.extend(
            [
                '[S;$(S(=O)(=O)):1][N:2]>>[S:1][Cl].[N:2]',
                '[S;$(S(=O)(=O)[#6]):1][N:2]>>[S:1][Cl].[N:2]',
                '[S;$(S=O):1][N:2]>>[S:1][Cl].[N:2]',
                '[#16:1]-[#7:2]>>[#16:1][Cl].[#7:2]',
            ]
        )    # Condensation / Knoevenagel-like: activated C + aldehyde → C=C.
    if n_reac == 2 and (
            'CH=O' in reactants or '[CH:5]=O' in reactants or '[CH]=O' in reactants or (
            '=O' in reactants and ('[CH3' in reactants or '[CH2' in reactants or 'CH2' in reactants)
    )):
        fallbacks.append('[C:2]=[C:5]>>[C:2].[C:5]=O')
        fallbacks.append('[C:2]=[CH:5]>>[CH2:2].[CH:5]=O')
        fallbacks.append('[C:2]=[C:5]>>[CH3:2].[CH:5]=O')
    # Esterification / etherification: acid/alcohol + halide → C–O–C.
    if n_reac == 2 and (
            '[OH' in reactants or 'O-' in reactants or '[OH,O-]' in reactants or '[OH,SH' in reactants
    ) and (
            'Cl,Br,I' in reactants or '[Cl,Br,I]' in reactants or 'F,Cl,Br,I' in reactants
    ):
        fallbacks.append('[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][Cl]')
        fallbacks.append('[C:1][O:2][C:3]>>[C:1][OH].[C:3][Cl]')
        fallbacks.append('[c:1][O:2][C:3]>>[c:1][OH].[C:3][Cl]')
        fallbacks.append('[c:1][O:2][c:3]>>[c:1][OH].[c:3][Cl]')
        fallbacks.append('[#6:1][O:2][#6:3]>>[#6:1][OH].[#6:3][Cl]')
        fallbacks.append('[C:1][S:2][C:3]>>[C:1][SH].[C:3][Cl]')
        fallbacks.append('[c:1][S:2][C:3]>>[c:1][SH].[C:3][Cl]')
        fallbacks.append('[c:1][S:2][c:3]>>[c:1][SH].[c:3][Cl]')
        fallbacks.append('[#6:1][S:2][#6:3]>>[#6:1][SH].[#6:3][Cl]')
    # Aryl / alkyl halide + nucleophile: aliphatic AND aromatic C–hetero cuts.
    # Explicit Cl (not [Cl,Br,I]) so RDKit emits a real electrophile fragment.
    if n_reac == 2 and (
            'Cl,Br,I' in reactants or '[Cl,Br,I]' in reactants or 'F,Cl,Br,I' in reactants
            or '[Br,I]' in reactants
    ):
        fallbacks.extend(
            [
                '[c:1][N;H0,H1:2]>>[c:1][Cl].[N:2]',
                '[c:1][NH2:2]>>[c:1][Cl].[NH2:2]',
                '[C:1][N:2]>>[C:1][Cl].[N:2]',
                '[N:1][#6:2]>>[N:1].[#6:2][Cl]',
                # Diaryl / alkyl–aryl ethers & thioethers (need both carbons mapped).
                '[c:1][O:2][c:3]>>[c:1][OH].[c:3][Cl]',
                '[c:1][O:2][C:3]>>[c:1][OH].[C:3][Cl]',
                '[C:1][O:2][c:3]>>[C:1][OH].[c:3][Cl]',
                '[#6:1][O:2][#6:3]>>[#6:1][OH].[#6:3][Cl]',
                '[c:1][S:2][c:3]>>[c:1][SH].[c:3][Cl]',
                '[c:1][S:2][C:3]>>[c:1][SH].[C:3][Cl]',
                '[#6:1][S:2][#6:3]>>[#6:1][SH].[#6:3][Cl]',
                # Broader C–C (Wurtz-like) — explicit Br.
                '[c:1][CH2:2][c:3]>>[c:1][Br].[c:3][CH2][Br]',
                '[c:1][CH2:2][CH2:3][c:4]>>[c:1][Br].[c:4][CH2][CH2][Br]',
                '[#6:1][#6:2]>>[#6:1][Br].[#6:2][Br]',
            ]
        )
    # Disulfide R–S–S–R' ← 2× thiol (oxidation).
    if n_reac == 2 and ('[SH' in reactants or 'SH:' in reactants or 'SS' in product):
        fallbacks.extend(
            [
                '[#6:1][S:2][S:3][#6:4]>>[#6:1][SH:2].[#6:4][SH:3]',
                '[c:1][S:2][S:3][c:4]>>[c:1][SH].[c:4][SH]',
                '[c:1][S:2][S:3][C:4]>>[c:1][SH].[C:4][SH]',
            ]
        )
    # Imine / hydrazone C=N ← aldehyde + amine / hydrazine.
    if n_reac == 2 and (
            ']=[N' in product or '=[N:' in product or 'CH]=[N' in product or ']=N' in product
            or ('=O' in reactants and '[N' in reactants)
    ):
        fallbacks.extend(
            [
                '[#6:1][CH:2]=[N:3]>>[#6:1][CH:2]=O.[N:3]',
                '[C:1]=[N:2]>>[C:1]=O.[N:2]',
                '[c:1][CH:2]=[N:3]>>[c:1][CH]=O.[N:3]',
            ]
        )
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for smarts in fallbacks:
        if smarts not in seen:
            seen.add(smarts)
            unique.append(smarts)
    return unique

def format_pathway(react_trace: str) -> str:
    """Human-readable pathway string from a react_trace."""
    tree = ReactionTree(react_trace).tree

    def _fmt(node: dict) -> str:
        product = node.get('product') or 'NA'
        educts = node.get('educts')
        if not educts:
            return product
        left = ' + '.join(_fmt(e) for e in educts)
        rid = node.get('react_id') or '?'
        return f'({left}) -[{rid}]-> {product}'

    return _fmt(tree)


@dataclass
class SynthesisRoute:
    query_smiles: str
    found: bool
    n_steps: int = 0
    reaction_ids: List[str] = field(default_factory=list)
    building_block_ids: List[str] = field(default_factory=list)
    building_block_smiles: List[str] = field(default_factory=list)
    building_block_providers: List[str] = field(default_factory=list)
    building_block_sources: List[str] = field(default_factory=list)
    bb_priority: int = 0
    react_trace: Optional[str] = None
    pathway: Optional[str] = None
    exact: bool = False
    similarity: float = 0.0
    reconstructed_smiles: Optional[str] = None
    searched_smiles: Optional[str] = None
    query_transforms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'query_smiles': self.query_smiles,
            'searched_smiles': self.searched_smiles or self.query_smiles,
            'query_transforms': ';'.join(self.query_transforms),
            'found': self.found,
            'exact': self.exact,
            'similarity': round(self.similarity, 4) if self.found else '',
            'reconstructed_smiles': self.reconstructed_smiles or '',
            'n_steps': self.n_steps,
            'reaction_ids': ';'.join(self.reaction_ids),
            'building_block_ids': ';'.join(self.building_block_ids),
            'building_block_smiles': ';'.join(self.building_block_smiles),
            'building_block_providers': ';'.join(self.building_block_providers),
            'building_block_sources': ';'.join(self.building_block_sources),
            'bb_priority': self.bb_priority if self.found else '',
            'react_trace': self.react_trace or '',
            'pathway': self.pathway or '',
        }


@dataclass
class _PartialRoute:
    react_trace: str
    reaction_ids: List[str]
    building_block_ids: List[str]
    building_block_smiles: List[str]
    n_steps: int
    reconstructed_smiles: str
    min_similarity: float
    exact: bool


@dataclass(order=True)
class _Disconnect:
    """Ranked retrosynthetic disconnection (exact / similar / buyable first)."""

    sort_key: Tuple
    reaction_id: str = field(compare=False)
    precursors: Tuple[str, ...] = field(compare=False)
    product_smi: str = field(compare=False)
    similarity: float = field(compare=False, default=1.0)

    @property
    def smi0(self) -> str:
        return self.precursors[0]

    @property
    def smi1(self) -> str:
        return self.precursors[1] if len(self.precursors) > 1 else ''

    @property
    def molecularity(self) -> int:
        return len(self.precursors)


def _halogen_generic_smiles(smiles: str) -> str:
    """Map Cl/Br/I atom tokens to X for near-duplicate detection.

    Proposed-reagent halogen variants (CNCBr vs CNCI, Ar–Cl vs Ar–Br, …) then
    share one identity so we keep a single chemical plan.
    """
    s = str(smiles or '')
    # Order matters: replace Cl before bare C-patterns; Br/I are unambiguous.
    return s.replace('Cl', 'X').replace('Br', 'X').replace('I', 'X')


def _disconnect_branch_key(reaction_id: str, precursors: Sequence[str]) -> Tuple:
    """Order- and halogen-insensitive branch identity.

    Used to prune (A,B) vs (B,A) and Cl/Br/I proposed-partner twins before
    walking children.
    """
    normed = tuple(sorted(_halogen_generic_smiles(p) for p in precursors))
    return (str(reaction_id), normed)


def _partial_route_signature(route: '_PartialRoute') -> Tuple:
    """Chemical-plan identity: same reactions + same reconstructed product.

    Ignores building-block halogen / Williamson-direction variants that rebuild
    the identical molecule with the same template sequence.
    """
    return (
        int(route.n_steps),
        tuple(sorted(str(r) for r in route.reaction_ids)),
        str(route.reconstructed_smiles or ''),
    )


class LocalRetrosynthesizer:
    """Depth-limited multi-step retrosynthesis via reverse SMARTS + forward checks."""

    def __init__(
        self,
        reaction_path: Path | str,
        building_blocks_path: Path | str,
        reaction_to_compound_path: Path | str | None = None,
        max_depth: int = 5,
        max_routes_per_mol: int = 10,
        require_role_membership: bool = True,
        max_disconnections: int = 64,
        prefer_buyable: bool = True,
        similarity_threshold: float | None = None,
        allow_proposed_reagents: bool = True,
        max_proposed_reagent_atoms: int = 16,
        normalize_query: bool = True,
        force_uncharge: bool = True,
        simplify_peroxides: bool = True,
        peroxide_mode: str = 'ether',
        transform_similarity_threshold: float = 0.7,
        seed_trivial_catalog: bool = True,
        n_workers: int = 1,
        max_leave_one_out_partners: int = 64,
        bb_fp_prefilter: float = 0.12,
        use_prepared_bb_cache: bool = True,
        disconnect_cache_path: Path | str | None = None,
        reaction_pack_path: Path | str | None = None,
        early_skip: bool = True,
        early_skip_junk_chem: bool = True,
        early_skip_exclude_reactions: Sequence[str] | None = None,
        early_skip_min_useful_discs: int = 1,
    ):
        self.reaction_path = Path(reaction_path)
        self.building_blocks_path = Path(building_blocks_path)
        self.reaction_to_compound_path = (
            Path(reaction_to_compound_path) if reaction_to_compound_path else None
        )
        self.max_depth = max_depth
        self.max_routes_per_mol = max_routes_per_mol
        self.require_role_membership = require_role_membership
        self.max_disconnections = max_disconnections
        self.prefer_buyable = prefer_buyable
        # Accept reverse-proposed electrophiles (aryl/alkyl halides, …) that are
        # missing from the catalog as terminal leaves. Catalog nucleophiles still
        # have to match; pathways flag these with building_block_ids proposed__*.
        self.allow_proposed_reagents = allow_proposed_reagents
        self.max_proposed_reagent_atoms = max_proposed_reagent_atoms
        self.normalize_query = normalize_query
        self.force_uncharge = force_uncharge
        self.simplify_peroxides = simplify_peroxides
        self.peroxide_mode = peroxide_mode
        self.transform_similarity_threshold = float(transform_similarity_threshold)
        self.seed_trivial_catalog = seed_trivial_catalog
        self.n_workers = max(1, int(n_workers))
        self.max_leave_one_out_partners = max_leave_one_out_partners
        self.bb_fp_prefilter = bb_fp_prefilter
        self.use_prepared_bb_cache = use_prepared_bb_cache
        self.disconnect_cache_path = (
            Path(disconnect_cache_path) if disconnect_cache_path else None
        )
        self.reaction_pack_path = (
            Path(reaction_pack_path)
            if reaction_pack_path is not None
            else self.reaction_path.with_name('retrosynthesis_pack.pkl')
        )
        # Avoid escalating shallow misses to max_depth when the query is hopeless.
        self.early_skip = bool(early_skip)
        self.early_skip_junk_chem = bool(early_skip_junk_chem)
        if early_skip_exclude_reactions is None:
            self.early_skip_exclude_reactions = set(_DEFAULT_EARLY_SKIP_EXCLUDE_REACTIONS)
        else:
            self.early_skip_exclude_reactions = {
                str(x) for x in early_skip_exclude_reactions if x
            }
        self.early_skip_min_useful_discs = max(0, int(early_skip_min_useful_discs))
        # None or >=1.0 → exact match only. E.g. 0.7 keeps near-miss constructions.
        if similarity_threshold is None:
            self.similarity_threshold = 1.0
        else:
            self.similarity_threshold = float(similarity_threshold)
        self.approximate = self.similarity_threshold < 1.0 - 1e-12

        self.reactions: Dict[str, dict] = {}
        # Primary / fallback reverse ChemicalReaction objects per forward reaction id.
        self.reverse_rxns: Dict[str, Dict[str, List[rdChemReactions.ChemicalReaction]]] = {}
        self.forward_rxns: Dict[str, rdChemReactions.ChemicalReaction] = {}
        self.bb_smi_to_id: Dict[str, str] = {}
        self._bb_n_atoms: Dict[str, int] = {}
        self._bb_fps: Dict[str, object] = {}
        self._bb_mols: Dict[str, Chem.Mol] = {}
        self._bb_priority: Dict[str, int] = {}
        self._bb_provider: Dict[str, str] = {}
        self._bb_sources: Dict[str, str] = {}
        self._bbs_by_size: List[str] = []
        self.fpgen = AllChem.GetMorganGenerator(2, fpSize=2048)
        self.role_map: Optional[Dict[str, Dict[int, Set[str]]]] = None
        self._bb_id_to_smi: Dict[str, str] = {}
        self._fp_cache: Dict[str, object] = {}
        self._mol_cache: Dict[str, Optional[Chem.Mol]] = {}
        self._atom_count_cache: Dict[str, int] = {}
        self._memo: Dict[Tuple, List[_PartialRoute]] = {}
        self._disconnect_cache: Dict[str, List[_Disconnect]] = {}
        # Optional multiprocessing.Manager dict shared by all workers.
        self._shared_disconnect_cache = None
        # Buffer shared writes and flush once per molecule (Manager.dict is costly).
        self._pending_shared_disconnects: Dict[str, List[_Disconnect]] = {}
        # Feature→reaction inverted index from retrosynthesis_pack.pkl (optional).
        self._feature_queries: Dict[str, Chem.Mol] = {}
        self._feature_to_reactions: Dict[str, List[str]] = {}
        self._always_try_reactions: List[str] = []
        self._reaction_order: List[str] = []
        self._use_feature_index: bool = False
        self._mol_feature_cache: Dict[str, Tuple[str, ...]] = {}

        self._load_chemical_space()
        self._load_disconnect_cache()

    def _worker_config(self) -> dict:
        """Picklable constructor kwargs for process-pool workers."""
        return {
            'reaction_path': self.reaction_path,
            'building_blocks_path': self.building_blocks_path,
            'reaction_to_compound_path': self.reaction_to_compound_path,
            'max_depth': self.max_depth,
            'max_routes_per_mol': self.max_routes_per_mol,
            'require_role_membership': self.require_role_membership,
            'max_disconnections': self.max_disconnections,
            'prefer_buyable': self.prefer_buyable,
            'similarity_threshold': self.similarity_threshold,
            'allow_proposed_reagents': self.allow_proposed_reagents,
            'max_proposed_reagent_atoms': self.max_proposed_reagent_atoms,
            'normalize_query': self.normalize_query,
            'force_uncharge': self.force_uncharge,
            'simplify_peroxides': self.simplify_peroxides,
            'peroxide_mode': self.peroxide_mode,
            'transform_similarity_threshold': self.transform_similarity_threshold,
            'seed_trivial_catalog': self.seed_trivial_catalog,
            'n_workers': 1,
            'max_leave_one_out_partners': self.max_leave_one_out_partners,
            'bb_fp_prefilter': self.bb_fp_prefilter,
            'use_prepared_bb_cache': self.use_prepared_bb_cache,
            'disconnect_cache_path': None,
            'reaction_pack_path': self.reaction_pack_path,
            'early_skip': self.early_skip,
            'early_skip_junk_chem': self.early_skip_junk_chem,
            'early_skip_exclude_reactions': sorted(self.early_skip_exclude_reactions),
            'early_skip_min_useful_discs': self.early_skip_min_useful_discs,
        }

    def _get_mol(self, smiles: str) -> Optional[Chem.Mol]:
        if smiles in self._mol_cache:
            return self._mol_cache[smiles]
        if smiles in self._bb_mols:
            mol = self._bb_mols[smiles]
            self._mol_cache[smiles] = mol
            return mol
        mol = Chem.MolFromSmiles(smiles)
        self._mol_cache[smiles] = mol
        return mol

    def _cached_atom_count(self, smiles: str) -> int:
        n = self._atom_count_cache.get(smiles)
        if n is not None:
            return n
        if smiles in self._bb_n_atoms:
            n = self._bb_n_atoms[smiles]
        else:
            mol = self._get_mol(smiles)
            n = mol.GetNumAtoms() if mol is not None else 0
        self._atom_count_cache[smiles] = n
        return n

    def _try_load_prepared_bbs(self, source_fp: str) -> bool:
        cache_path = _prepared_bb_cache_path(self.building_blocks_path)
        if not cache_path.is_file():
            return False
        try:
            with open(cache_path, 'rb') as f:
                payload = pickle.load(f)
        except Exception as e:
            logging.warning(f'Ignoring unreadable prepared BB cache {cache_path}: {e}')
            return False
        if (
            not isinstance(payload, dict)
            or payload.get('version') != _PREPARED_BB_CACHE_VERSION
            or payload.get('source_fp') != source_fp
        ):
            return False
        self.bb_smi_to_id = payload['bb_smi_to_id']
        self._bb_id_to_smi = payload['_bb_id_to_smi']
        self._bb_n_atoms = payload['_bb_n_atoms']
        self._bb_fps = payload.get('_bb_fps', {})
        self._bb_priority = payload.get('_bb_priority', {})
        self._bb_provider = payload.get('_bb_provider', {})
        self._bb_sources = payload.get('_bb_sources', {})
        self._bbs_by_size = payload['_bbs_by_size']
        self._atom_count_cache.update(self._bb_n_atoms)
        # Backfill priority from IDs when older caches omit metadata.
        if not self._bb_priority:
            for smi, bb_id in self.bb_smi_to_id.items():
                prov = infer_provider(bb_id)
                self._bb_provider[smi] = prov
                self._bb_priority[smi] = provider_rank(prov)
                self._bb_sources.setdefault(smi, f'{prov}:{bb_id}')
        for smi in self.bb_smi_to_id:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                self._bb_mols[smi] = mol
                self._mol_cache[smi] = mol
                if smi not in self._bb_fps:
                    self._bb_fps[smi] = self.fpgen.GetFingerprint(mol)
        logging.info(
            f'Loaded prepared building-block cache ({len(self.bb_smi_to_id)} BBs) '
            f'from {cache_path}'
        )
        return True

    def _save_prepared_bbs(self, source_fp: str) -> None:
        cache_path = _prepared_bb_cache_path(self.building_blocks_path)
        payload = {
            'version': _PREPARED_BB_CACHE_VERSION,
            'source_fp': source_fp,
            'bb_smi_to_id': self.bb_smi_to_id,
            '_bb_id_to_smi': self._bb_id_to_smi,
            '_bb_n_atoms': self._bb_n_atoms,
            '_bb_fps': self._bb_fps,
            '_bb_priority': self._bb_priority,
            '_bb_provider': self._bb_provider,
            '_bb_sources': self._bb_sources,
            '_bbs_by_size': self._bbs_by_size,
        }
        try:
            tmp = cache_path.with_suffix(cache_path.suffix + '.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
            logging.info(f'Wrote prepared building-block cache to {cache_path}')
        except Exception as e:
            logging.warning(f'Could not write prepared BB cache {cache_path}: {e}')

    def _reactions_fingerprint(self) -> str:
        """Fingerprint of reaction JSON + extras (+ pack if present) for cache invalidation."""
        from lddm.reactions.retrosynthesis_pack import reaction_sources_fingerprint

        fp = reaction_sources_fingerprint(self.reaction_path)
        if self.reaction_pack_path.is_file():
            fp = hashlib.sha256(
                (fp + '|' + _file_fingerprint(self.reaction_pack_path)).encode()
            ).hexdigest()
        return fp

    def _load_disconnect_cache(self) -> None:
        if self.disconnect_cache_path is None or not self.disconnect_cache_path.is_file():
            return
        try:
            with open(self.disconnect_cache_path, 'rb') as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or 'disconnects' not in payload:
                return
            expected = self._reactions_fingerprint()
            got = payload.get('reactions_fp')
            if got is not None and got != expected:
                logging.info(
                    f'Disconnect cache stale vs reactions/pack '
                    f'({self.disconnect_cache_path}); ignoring'
                )
                return
            for smi, rows in payload['disconnects'].items():
                self._disconnect_cache[smi] = _deserialize_disconnects(rows)
            logging.info(
                f'Loaded {len(self._disconnect_cache)} cached disconnections '
                f'from {self.disconnect_cache_path}'
            )
        except Exception as e:
            logging.warning(f'Ignoring disconnect cache {self.disconnect_cache_path}: {e}')

    def save_disconnect_cache(self) -> None:
        if self.disconnect_cache_path is None:
            return
        # Prefer shared cache contents when present (workers may have added entries).
        disconnects = self._export_disconnect_cache()
        payload = {
            'disconnects': disconnects,
            'reactions_fp': self._reactions_fingerprint(),
        }
        try:
            self.disconnect_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.disconnect_cache_path.with_suffix(
                self.disconnect_cache_path.suffix + '.tmp'
            )
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.disconnect_cache_path)
            logging.info(
                f'Wrote {len(disconnects)} disconnections to '
                f'{self.disconnect_cache_path}'
            )
        except Exception as e:
            logging.warning(f'Could not write disconnect cache: {e}')

    def _export_disconnect_cache(self) -> Dict[str, list]:
        """Merge local + shared disconnect caches into a serializable dict."""
        out: Dict[str, list] = {
            smi: _serialize_disconnects(discs)
            for smi, discs in self._disconnect_cache.items()
        }
        shared = self._shared_disconnect_cache
        if shared is not None:
            try:
                for smi in list(shared.keys()):
                    out[smi] = shared[smi]
            except Exception as e:
                logging.debug(f'Could not read shared disconnect cache: {e}')
        return out

    def _sync_shared_disconnect_cache_into_local(self) -> None:
        """Pull worker-filled shared entries back into the local dict."""
        shared = self._shared_disconnect_cache
        if shared is None:
            return
        try:
            for smi in list(shared.keys()):
                if smi not in self._disconnect_cache:
                    self._disconnect_cache[smi] = _deserialize_disconnects(shared[smi])
        except Exception as e:
            logging.warning(f'Could not sync shared disconnect cache: {e}')

    def _lookup_disconnect_cache(self, smiles: str) -> Optional[List[_Disconnect]]:
        cached = self._disconnect_cache.get(smiles)
        if cached is not None:
            return cached
        pending = self._pending_shared_disconnects.get(smiles)
        if pending is not None:
            self._disconnect_cache[smiles] = pending
            return pending
        shared = self._shared_disconnect_cache
        if shared is None:
            return None
        try:
            rows = shared[smiles]  # single IPC; KeyError on miss
        except KeyError:
            return None
        except Exception:
            return None
        discs = _deserialize_disconnects(rows)
        self._disconnect_cache[smiles] = discs
        return discs

    def _store_disconnect_cache(self, smiles: str, discs: List[_Disconnect]) -> None:
        self._disconnect_cache[smiles] = discs
        # Defer Manager writes — flush once per molecule in synthesize().
        if self._shared_disconnect_cache is not None:
            self._pending_shared_disconnects[smiles] = discs

    def _flush_shared_disconnect_cache(self) -> None:
        """Push pending local disconnections into the cross-worker Manager dict."""
        shared = self._shared_disconnect_cache
        pending = self._pending_shared_disconnects
        if shared is None or not pending:
            self._pending_shared_disconnects = {}
            return
        try:
            for smi, discs in pending.items():
                # Avoid redundant overwrite IPC when another worker already stored.
                if smi in shared:
                    continue
                shared[smi] = _serialize_disconnects(discs)
        except Exception as e:
            logging.debug(f'Could not flush shared disconnect cache: {e}')
        finally:
            self._pending_shared_disconnects = {}

    def _try_load_reaction_pack(self) -> Optional[dict]:
        from lddm.reactions.retrosynthesis_pack import load_pack, pack_is_fresh

        pack = load_pack(self.reaction_pack_path)
        if pack is None:
            return None
        if not pack_is_fresh(pack, self.reaction_path):
            logging.info(
                f'Retrosynthesis pack stale vs reactions JSON; ignoring {self.reaction_pack_path}'
            )
            return None
        return pack

    def _compile_reverse_lists(
        self, primary: Sequence[str], fallbacks: Sequence[str]
    ) -> Tuple[List[rdChemReactions.ChemicalReaction], List[rdChemReactions.ChemicalReaction]]:
        primary_rxns: List[rdChemReactions.ChemicalReaction] = []
        fallback_rxns: List[rdChemReactions.ChemicalReaction] = []
        seen_rev: Set[str] = set()
        for reverse in primary:
            if reverse in seen_rev:
                continue
            seen_rev.add(reverse)
            rxn = rdChemReactions.ReactionFromSmarts(reverse)
            if rxn is not None:
                primary_rxns.append(rxn)
        for reverse in fallbacks:
            if reverse in seen_rev:
                continue
            seen_rev.add(reverse)
            rxn = rdChemReactions.ReactionFromSmarts(reverse)
            if rxn is not None:
                fallback_rxns.append(rxn)
        return primary_rxns, fallback_rxns

    def _register_reaction(
        self,
        *,
        rid: str,
        name: str,
        smarts: str,
        explicit_hs: bool,
        molecularity: int,
        primary_rev: Sequence[str],
        fallback_rev: Sequence[str],
        required_features: Sequence[str] = (),
    ) -> bool:
        primary_rxns, fallback_rxns = self._compile_reverse_lists(primary_rev, fallback_rev)
        if not primary_rxns and not fallback_rxns:
            return False
        forward_rxn = rdChemReactions.ReactionFromSmarts(smarts)
        if forward_rxn is None:
            return False
        product_smarts = smarts.split('>>')[1].strip()
        product_query = Chem.MolFromSmarts(product_smarts)
        self.reactions[rid] = {
            'id': rid,
            'name': name,
            'forward_smarts': smarts,
            'reverse_smarts': (list(primary_rev) or list(fallback_rev))[0],
            'explicit_hs': explicit_hs,
            'product_query': product_query,
            'molecularity': molecularity,
            'required_features': tuple(required_features),
        }
        self.reverse_rxns[rid] = {'primary': primary_rxns, 'fallback': fallback_rxns}
        self.forward_rxns[rid] = forward_rxn
        return True

    def _install_feature_index_from_pack(self, pack: dict) -> None:
        from lddm.reactions.retrosynthesis_pack import FEATURE_SMARTS

        feature_smarts = pack.get('feature_smarts') or FEATURE_SMARTS
        self._feature_queries = {}
        for name, smarts in feature_smarts.items():
            q = Chem.MolFromSmarts(smarts)
            if q is not None:
                self._feature_queries[name] = q
        self._feature_to_reactions = {
            str(k): [str(x) for x in v]
            for k, v in (pack.get('feature_to_reactions') or {}).items()
        }
        self._always_try_reactions = [str(x) for x in pack.get('always_try') or []]
        self._reaction_order = list(self.reactions.keys())
        self._use_feature_index = bool(self._feature_queries) and bool(self.reactions)
        n_gated = sum(
            1
            for rid, info in self.reactions.items()
            if info.get('required_features')
        )
        logging.info(
            f'Loaded retrosynthesis pack ({len(self.reactions)} rxns, '
            f'{n_gated} feature-gated, {len(self._always_try_reactions)} always-try) '
            f'from {self.reaction_pack_path}'
        )

    def _install_feature_index_inline(self) -> None:
        """Build a light feature index when no pack is present."""
        from lddm.reactions.retrosynthesis_pack import (
            FEATURE_SMARTS,
            infer_required_features,
        )

        self._feature_queries = {}
        for name, smarts in FEATURE_SMARTS.items():
            q = Chem.MolFromSmarts(smarts)
            if q is not None:
                self._feature_queries[name] = q
        self._feature_to_reactions = {k: [] for k in self._feature_queries}
        self._always_try_reactions = []
        for rid, info in self.reactions.items():
            req = tuple(info.get('required_features') or ())
            if not req:
                req = infer_required_features(info['forward_smarts'].split('>>')[1])
                info['required_features'] = req
            if not req:
                self._always_try_reactions.append(rid)
            else:
                for feat in req:
                    self._feature_to_reactions.setdefault(feat, []).append(rid)
        self._reaction_order = list(self.reactions.keys())
        self._use_feature_index = bool(self._feature_queries) and bool(self.reactions)

    def _install_reactions_from_pack(self, pack: dict) -> int:
        skipped = 0
        for rid, entry in (pack.get('reactions') or {}).items():
            ok = self._register_reaction(
                rid=str(rid),
                name=str(entry.get('name') or rid),
                smarts=entry['forward_smarts'],
                explicit_hs=bool(entry.get('explicit_hs', False)),
                molecularity=int(entry['molecularity']),
                primary_rev=entry.get('primary_reverse_smarts') or [],
                fallback_rev=entry.get('fallback_reverse_smarts') or [],
                required_features=entry.get('required_features') or [],
            )
            if not ok:
                skipped += 1
        self._install_feature_index_from_pack(pack)
        return skipped

    def _install_reactions_from_json(self) -> int:
        from lddm.reactions.retrosynthesis_pack import load_reaction_definitions

        reaction_defs = load_reaction_definitions(self.reaction_path)
        extra_path = self.reaction_path.with_name('reactions_retrosynthesis_extra.json')
        if extra_path.is_file():
            n_extra = sum(
                1
                for r in reaction_defs
                if str(r.get('source', '')).startswith('retrosynthesis_extra')
                or str(r.get('id', '')).startswith('retro_')
            )
            # Count only true extras file load for the log message.
            try:
                with open(extra_path) as f:
                    n_file = len(json.load(f))
                logging.info(
                    f'Loaded extras from {extra_path} ({n_file} entries in file; '
                    f'merged space has {len(reaction_defs)} reactions)'
                )
            except Exception:
                logging.info(f'Loaded reaction definitions including {extra_path}')
        skipped = 0
        for reaction in reaction_defs:
            rid = str(reaction['id'])
            smarts = reaction['reaction']
            explicit_hs = bool(reaction.get('explicit_hs', False))
            molecularity = reaction_molecularity(smarts)
            if molecularity is None:
                skipped += 1
                continue
            primary: List[str] = []
            try:
                base_rev = _reverse_smarts(smarts)
                primary.append(base_rev)
                primary.extend(_explicit_halogen_reverse_variants(base_rev))
            except ValueError as e:
                logging.debug(f'Skipping primary reverse for {rid}: {e}')
            fallbacks = _fallback_reverse_smarts(smarts)
            ok = self._register_reaction(
                rid=rid,
                name=reaction.get('name', rid),
                smarts=smarts,
                explicit_hs=explicit_hs,
                molecularity=molecularity,
                primary_rev=primary,
                fallback_rev=fallbacks,
            )
            if not ok:
                skipped += 1
        self._install_feature_index_inline()
        return skipped

    def _mol_features(self, smiles: str, mol: Optional[Chem.Mol] = None) -> Tuple[str, ...]:
        cached = self._mol_feature_cache.get(smiles)
        if cached is not None:
            return cached
        if mol is None:
            mol = self._get_mol(smiles)
        if mol is None or not self._feature_queries:
            self._mol_feature_cache[smiles] = ()
            return ()
        hits: List[str] = []
        for name, q in self._feature_queries.items():
            try:
                if mol.HasSubstructMatch(q):
                    hits.append(name)
            except Exception:
                continue
        feats = tuple(hits)
        self._mol_feature_cache[smiles] = feats
        return feats

    def _candidate_reaction_ids(self, smiles: str, mol: Optional[Chem.Mol]) -> List[str]:
        if not self._use_feature_index:
            return list(self.reactions.keys())
        from lddm.reactions.retrosynthesis_pack import candidate_reaction_ids

        feats = self._mol_features(smiles, mol)
        return candidate_reaction_ids(
            feats,
            always_try=self._always_try_reactions,
            required_features_by_rid={
                rid: (info.get('required_features') or ())
                for rid, info in self.reactions.items()
            },
            reaction_order=self._reaction_order or list(self.reactions.keys()),
        )

    def _load_chemical_space(self) -> None:
        if not self.reaction_path.is_file():
            raise FileNotFoundError(f'Reactions file not found: {self.reaction_path}')
        if not self.building_blocks_path.is_file():
            raise FileNotFoundError(
                f'Building blocks file not found: {self.building_blocks_path}'
            )

        pack = self._try_load_reaction_pack()
        if pack is not None:
            skipped = self._install_reactions_from_pack(pack)
        else:
            skipped = self._install_reactions_from_json()
        if skipped:
            logging.info(f'Skipped {skipped} reactions incompatible with reverse SMARTS')
        mol_counts = {1: 0, 2: 0, 3: 0}
        for info in self.reactions.values():
            mol_counts[info['molecularity']] = mol_counts.get(info['molecularity'], 0) + 1
        logging.info(
            f'Reaction molecularity: uni={mol_counts.get(1, 0)} '
            f'bi={mol_counts.get(2, 0)} tri={mol_counts.get(3, 0)}'
        )

        source_fp = _file_fingerprint(self.building_blocks_path)
        loaded_prepared = False
        if self.use_prepared_bb_cache:
            loaded_prepared = self._try_load_prepared_bbs(source_fp)

        if not loaded_prepared:
            with open(self.building_blocks_path, 'rb') as f:
                blocks = pickle.load(f)
            if not {'id', 'smiles'}.issubset(blocks.columns):
                raise ValueError('Building blocks pickle must contain id and smiles columns')
            has_fp = 'fp' in blocks.columns
            has_priority = 'priority' in blocks.columns
            has_provider = 'provider' in blocks.columns
            has_sources = 'sources' in blocks.columns
            for row in blocks.itertuples(index=False):
                can = _canonicalize(str(row.smiles))
                if can is None:
                    continue
                bb_id = str(row.id)
                mol = Chem.MolFromSmiles(can)
                if mol is None:
                    continue
                self.bb_smi_to_id[can] = bb_id
                self._bb_id_to_smi[bb_id] = can
                n_atoms = mol.GetNumAtoms()
                self._bb_n_atoms[can] = n_atoms
                self._atom_count_cache[can] = n_atoms
                self._bb_mols[can] = mol
                self._mol_cache[can] = mol
                if has_fp and getattr(row, 'fp', None) is not None:
                    self._bb_fps[can] = row.fp
                else:
                    self._bb_fps[can] = self.fpgen.GetFingerprint(mol)
                if has_provider:
                    prov = str(getattr(row, 'provider'))
                else:
                    prov = infer_provider(bb_id)
                self._bb_provider[can] = prov
                if has_priority and getattr(row, 'priority', None) is not None:
                    self._bb_priority[can] = int(row.priority)
                else:
                    self._bb_priority[can] = provider_rank(prov)
                if has_sources and getattr(row, 'sources', None):
                    self._bb_sources[can] = str(row.sources)
                else:
                    self._bb_sources[can] = f'{prov}:{bb_id}'
            self._bbs_by_size = sorted(
                self.bb_smi_to_id.keys(), key=lambda s: self._bb_n_atoms[s], reverse=True
            )
            if self.use_prepared_bb_cache:
                self._save_prepared_bbs(source_fp)

        logging.info(
            f'Loaded {len(self.reactions)} reactions and '
            f'{len(self.bb_smi_to_id)} building blocks'
        )
        if self.seed_trivial_catalog:
            self._seed_trivial_catalog()

        if self.reaction_to_compound_path is not None:
            if not self.reaction_to_compound_path.is_file():
                raise FileNotFoundError(
                    f'Reaction-to-compound mapping not found: {self.reaction_to_compound_path}'
                )
            with open(self.reaction_to_compound_path, 'rb') as f:
                mapping = pickle.load(f)
            self.role_map = {}
            for rid, roles in mapping.items():
                self.role_map[str(rid)] = {
                    int(role): {str(x) for x in bbs} for role, bbs in roles.items()
                }
            logging.info(f'Loaded role membership for {len(self.role_map)} reactions')

    def _seed_trivial_catalog(self) -> None:
        """Add common small reagents missing from ZINC-style BB dumps."""
        added = 0
        for raw in _TRIVIAL_CATALOG_SMILES:
            can = _canonicalize(raw)
            if can is None or can in self.bb_smi_to_id:
                continue
            mol = Chem.MolFromSmiles(can)
            if mol is None:
                continue
            bb_id = f'trivial__{can}'
            self.bb_smi_to_id[can] = bb_id
            self._bb_id_to_smi[bb_id] = can
            n_atoms = mol.GetNumAtoms()
            self._bb_n_atoms[can] = n_atoms
            self._atom_count_cache[can] = n_atoms
            self._bb_mols[can] = mol
            self._mol_cache[can] = mol
            self._bb_fps[can] = self.fpgen.GetFingerprint(mol)
            self._bb_provider[can] = 'trivial'
            self._bb_priority[can] = provider_rank('trivial')
            self._bb_sources[can] = f'trivial:{bb_id}'
            added += 1
        if added:
            self._bbs_by_size = sorted(
                self.bb_smi_to_id.keys(), key=lambda s: self._bb_n_atoms[s], reverse=True
            )
            logging.info(f'Seeded {added} trivial catalog reagents')

    def _fp(self, smiles: str):
        fp = self._fp_cache.get(smiles)
        if fp is not None:
            return fp
        if smiles in self._bb_fps:
            fp = self._bb_fps[smiles]
        else:
            mol = self._get_mol(smiles)
            if mol is None:
                return None
            fp = self.fpgen.GetFingerprint(mol)
        self._fp_cache[smiles] = fp
        return fp

    def _tanimoto(self, smi_a: str, smi_b: str) -> float:
        if smi_a == smi_b:
            return 1.0
        fp_a, fp_b = self._fp(smi_a), self._fp(smi_b)
        if fp_a is None or fp_b is None:
            return 0.0
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

    def _forward_match(
        self, reaction_id: str, precursors: Sequence[str], target: str
    ) -> Optional[Tuple[Tuple[str, ...], str, float]]:
        """Best forward product vs target.

        Returns (ordered_precursors, product_smiles, similarity) so reactant
        order matches the forward SMARTS.
        """
        info = self.reactions[reaction_id]
        n = info['molecularity']
        if len(precursors) != n:
            return None
        forward_rxn = self.forward_rxns.get(reaction_id)
        tried: Set[Tuple[str, ...]] = set()
        for ordered in itertools.permutations(precursors):
            if ordered in tried:
                continue
            tried.add(ordered)
            mols = [self._get_mol(s) for s in ordered]
            if any(m is None for m in mols):
                continue
            if forward_rxn is not None:
                products = run_compiled_reaction(
                    forward_rxn, mols, explicit_hs=info['explicit_hs']
                )
            else:
                products = run_reaction_smarts(
                    info['forward_smarts'], ordered, info['explicit_hs']
                )
            if not products:
                continue
            if target in products:
                return ordered, target, 1.0
            if not self.approximate:
                continue
            best_smi = None
            best_sim = -1.0
            for product in products:
                sim = self._tanimoto(target, product)
                if sim > best_sim:
                    best_sim = sim
                    best_smi = product
            if best_smi is not None and best_sim >= self.similarity_threshold:
                return ordered, best_smi, best_sim
        return None

    def _forward_regenerates(
        self, reaction_id: str, precursors: Sequence[str], product: str
    ) -> bool:
        return self._forward_match(reaction_id, precursors, product) is not None

    def _fragment_to_canonical(self, frag: Chem.Mol) -> Optional[str]:
        try:
            frag = Chem.RemoveAllHs(frag)
        except Exception:
            return None
        try:
            Chem.SanitizeMol(frag)
            return Chem.MolToSmiles(frag)
        except Exception:
            # Some reverse applications leave transient valence errors that clear
            # after a SMILES round-trip.
            try:
                raw = Chem.MolToSmiles(frag)
            except Exception:
                return None
            return _canonicalize(raw)

    def _apply_reverse(
        self, reaction_id: str, smiles: str
    ) -> List[Tuple[Tuple[str, ...], str, float]]:
        """Return (ordered_precursors, forward_product, similarity) cuts."""
        base_mol = self._get_mol(smiles)
        if base_mol is None:
            return []
        info = self.reactions[reaction_id]
        n_expected = info['molecularity']

        def _run(
            rxns: List[rdChemReactions.ChemicalReaction],
        ) -> List[Tuple[Tuple[str, ...], str, float]]:
            pairs: List[Tuple[Tuple[str, ...], str, float]] = []
            seen: Set[Tuple[str, ...]] = set()
            parent_atoms = base_mol.GetNumAtoms()
            for rxn in rxns:
                mol = Chem.Mol(base_mol)
                if info['explicit_hs']:
                    try:
                        mol = Chem.AddHs(mol)
                    except Exception:
                        continue
                try:
                    outcomes = rxn.RunReactants((mol,))
                except Exception as e:
                    logging.debug(f'Reverse reaction {reaction_id} failed on {smiles}: {e}')
                    continue
                for outcome in outcomes:
                    if len(outcome) != n_expected:
                        continue
                    precursors: List[str] = []
                    ok = True
                    for frag in outcome:
                        can = self._fragment_to_canonical(frag)
                        if not can or '.' in can or can == smiles:
                            ok = False
                            break
                        if '*' in can or '[#0]' in can:
                            ok = False
                            break
                        n_atoms = self._cached_atom_count(can)
                        if n_atoms <= 0:
                            ok = False
                            break
                        # Bimolecular+ cuts must shrink each piece. Unimolecular
                        # ring closures may grow (halide / leaving-group leaf).
                        if n_expected >= 2 and n_atoms >= parent_atoms:
                            ok = False
                            break
                        if n_expected == 1 and n_atoms > parent_atoms + 6:
                            ok = False
                            break
                        precursors.append(can)
                    if not ok:
                        continue
                    match = self._forward_match(reaction_id, precursors, smiles)
                    if match is None:
                        continue
                    ordered, product_smi, sim = match
                    atom_budget = parent_atoms + (2 * n_expected)
                    if sum(self._cached_atom_count(s) for s in ordered) > atom_budget:
                        continue
                    # Collapse reactant-order and Cl/Br/I proposed-partner twins
                    # before they enter the disconnect beam / search.
                    key = tuple(sorted(_halogen_generic_smiles(s) for s in ordered)) + (
                        product_smi,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((ordered, product_smi, sim))
            return pairs

        rxn_sets = self.reverse_rxns[reaction_id]
        product_query = info.get('product_query')
        pairs: List[Tuple[Tuple[str, ...], str, float]] = []
        if (
            self.approximate
            or product_query is None
            or base_mol.HasSubstructMatch(product_query)
        ):
            pairs = _run(rxn_sets['primary'])
        if not pairs:
            pairs = _run(rxn_sets['fallback'])
        return pairs

    def _membership_ok(self, reaction_id: str, precursors: Sequence[str]) -> bool:
        if self.role_map is None or not self.require_role_membership:
            return True
        roles = self.role_map.get(reaction_id)
        if roles is None:
            return True
        ids = [self.bb_smi_to_id.get(s) for s in precursors]
        n = len(precursors)
        role_sets = [roles.get(i, set()) for i in range(n)]
        if all(i is not None for i in ids):
            for perm in itertools.permutations(ids):
                if all(perm[i] in role_sets[i] for i in range(n)):
                    return True
            return False
        allowed: Set[str] = set()
        for rs in role_sets:
            allowed |= rs
        return all(i in allowed for i in ids if i is not None)

    def _is_proposed_reagent(self, smiles: str) -> bool:
        """Non-catalog electrophile / small nucleophile acceptable as a leaf."""
        if not self.allow_proposed_reagents:
            return False
        if smiles in self.bb_smi_to_id:
            return False
        mol = self._get_mol(smiles)
        if mol is None:
            return False
        n = mol.GetNumAtoms()
        if n <= 0 or n > self.max_proposed_reagent_atoms:
            return False
        # Typical reverse-proposed aryl/alkyl halides.
        if any(atom.GetSymbol() in ('F', 'Cl', 'Br', 'I') for atom in mol.GetAtoms()):
            return True
        # Small alcohols / thiols (ether & thioether partners often absent from ZINC BB sets).
        if n <= 12 and _OH_SH_QUERY is not None and mol.HasSubstructMatch(_OH_SH_QUERY):
            return True
        # Small aldehydes (imine partners).
        if n <= 12 and mol.HasSubstructMatch(Chem.MolFromSmarts('[CH]=O')):
            return True
        # Isocyanates from cyclic-urea opens (intramolecular NCO + amine).
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[N]=C=O')):
            return True
        # Tiny heteroarenes (pyridine etc.) when not already trivial-seeded.
        if n <= 8 and mol.GetNumHeavyAtoms() <= 8:
            aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
            if aromatic >= 5:
                return True
        return False

    @staticmethod
    def _is_junk_chem(smiles: str) -> bool:
        """Denovo artifacts (cumulene / non-sulfone S(OR)2) that should not deepen."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        if _CUMULENE_QUERY is not None and mol.HasSubstructMatch(_CUMULENE_QUERY):
            return True
        if (
            _WEIRD_S_QUERY is not None
            and mol.HasSubstructMatch(_WEIRD_S_QUERY)
            and (_SULFONE_QUERY is None or not mol.HasSubstructMatch(_SULFONE_QUERY))
        ):
            return True
        return False

    def _early_skip_prefilter_reason(self, smiles: str) -> Optional[str]:
        """Return pathway tag if the query should fail before any depth search."""
        if not self.early_skip:
            return None
        if self.early_skip_junk_chem and self._is_junk_chem(smiles):
            return 'early_skip:junk_chem'
        if not self._disconnections(smiles):
            return 'early_skip:no_disconnections'
        return None

    def _has_shallow_progress(self, smiles: str) -> bool:
        """True if depth>2 search is warranted after a shallow miss.

        Escalate when a non-excluded disconnection has a catalog / proposed
        precursor leaf, or when enough non-excluded (useful) disconnections
        exist. Excluded reactions (default: ``retro_cc_wurtz``) never count —
        otherwise every Wurtz halide precursor would look like progress.
        """
        discs = self._disconnections(smiles)
        if not discs:
            return False
        useful = 0
        for disc in discs:
            if disc.reaction_id in self.early_skip_exclude_reactions:
                continue
            useful += 1
            for prec in disc.precursors:
                if prec in self.bb_smi_to_id or self._is_proposed_reagent(prec):
                    return True
        return useful >= self.early_skip_min_useful_discs

    def _leaf_route(self, smiles: str) -> Optional[_PartialRoute]:
        if smiles in self.bb_smi_to_id:
            bb_id = self.bb_smi_to_id[smiles]
            return _PartialRoute(
                react_trace=get_react_trace_building_block(smiles, id=bb_id),
                reaction_ids=[],
                building_block_ids=[bb_id],
                building_block_smiles=[smiles],
                n_steps=0,
                reconstructed_smiles=smiles,
                min_similarity=1.0,
                exact=True,
            )
        if self._is_proposed_reagent(smiles):
            bb_id = f'proposed__{smiles}'
            return _PartialRoute(
                react_trace=get_react_trace_building_block(smiles, id=bb_id),
                reaction_ids=[],
                building_block_ids=[bb_id],
                building_block_smiles=[smiles],
                n_steps=0,
                reconstructed_smiles=smiles,
                min_similarity=1.0,
                exact=True,
            )
        return None

    def _precursor_priority(self, smiles: str) -> int:
        """Lower = preferred provider. Proposed reagents rank worse than catalog."""
        if smiles in self._bb_priority:
            return self._bb_priority[smiles]
        if smiles in self.bb_smi_to_id:
            return provider_rank(infer_provider(self.bb_smi_to_id[smiles]))
        if self._is_proposed_reagent(smiles):
            return _PROPOSED_BB_PRIORITY
        return provider_rank('other')

    def _route_bb_priority(self, building_block_smiles: Sequence[str]) -> int:
        return sum(self._precursor_priority(s) for s in building_block_smiles)

    def _bb_meta_for_smiles(self, smiles: str) -> Tuple[str, str]:
        """Return (provider, sources) for a leaf SMILES."""
        if smiles in self._bb_provider:
            return (
                self._bb_provider[smiles],
                self._bb_sources.get(smiles, f'{self._bb_provider[smiles]}:{self.bb_smi_to_id.get(smiles, "")}'),
            )
        if smiles in self.bb_smi_to_id:
            bb_id = self.bb_smi_to_id[smiles]
            prov = infer_provider(bb_id)
            return prov, f'{prov}:{bb_id}'
        if str(smiles).startswith('proposed__') or self._is_proposed_reagent(smiles):
            return 'proposed', f'proposed:{smiles}'
        return 'other', f'other:{smiles}'

    def _enrich_route_meta(
        self,
        building_block_ids: Sequence[str],
        building_block_smiles: Sequence[str],
    ) -> Tuple[List[str], List[str], int]:
        providers: List[str] = []
        sources: List[str] = []
        for smi, bb_id in zip(building_block_smiles, building_block_ids):
            if str(bb_id).startswith('proposed__'):
                providers.append('proposed')
                sources.append(f'proposed:{bb_id}')
            else:
                prov, src = self._bb_meta_for_smiles(smi)
                providers.append(prov)
                sources.append(src)
        return providers, sources, self._route_bb_priority(building_block_smiles)

    def _rank_disconnect(
        self,
        reaction_id: str,
        precursors: Sequence[str],
        parent_atoms: int,
        product_smi: str,
        similarity: float,
    ) -> _Disconnect:
        buy_flags = [
            int(s in self.bb_smi_to_id or self._is_proposed_reagent(s)) for s in precursors
        ]
        n_buy = sum(buy_flags)
        sizes = [self._cached_atom_count(s) for s in precursors]
        leftover = sum(sz for sz, b in zip(sizes, buy_flags) if not b)
        balance = (max(sizes) - min(sizes)) if sizes else 0
        exact = int(similarity >= 1.0 - 1e-12)
        # Prefer better-priority catalog BBs among buyable precursors.
        buy_priority = sum(
            self._precursor_priority(s) for s, b in zip(precursors, buy_flags) if b
        )
        # Prefer disulfide → 2× thiol cuts when the parent has an S–S bond.
        ss_cut = 0
        prod = self._get_mol(product_smi)
        if prod is not None and _SS_QUERY is not None and prod.HasSubstructMatch(_SS_QUERY):
            thiols_ok = True
            for s in precursors:
                m = self._get_mol(s)
                if (
                    m is None
                    or _OH_SH_QUERY is None
                    or not m.HasSubstructMatch(_OH_SH_QUERY)
                ):
                    thiols_ok = False
                    break
            if thiols_ok:
                ss_cut = 1
        sort_key = (
            -exact,
            -ss_cut,
            -similarity,
            -n_buy,
            buy_priority,
            leftover,
            balance,
            -(parent_atoms - max(sizes, default=0)),
            len(precursors),
        )
        return _Disconnect(
            sort_key, reaction_id, tuple(precursors), product_smi, similarity
        )

    def _bb_substructure_hits(self, product_smi: str, max_hits: int = 80) -> List[str]:
        """Largest catalog BBs that are substructures of the product."""
        prod = self._get_mol(product_smi)
        if prod is None:
            return []
        prod_n = prod.GetNumAtoms()
        prod_fp = self.fpgen.GetFingerprint(prod)
        # Bulk fingerprint prefilter for larger BBs, then substructure confirm.
        large_smis: List[str] = []
        large_fps: List[object] = []
        small_candidates: List[str] = []
        for smi in self._bbs_by_size:
            n = self._bb_n_atoms[smi]
            if n >= prod_n:
                continue
            if n < 2:
                break
            if n > 8:
                fp = self._bb_fps.get(smi)
                if fp is None:
                    continue
                large_smis.append(smi)
                large_fps.append(fp)
            else:
                small_candidates.append(smi)

        hits: List[str] = []
        # Prefer largest BBs first (already size-sorted in _bbs_by_size).
        if large_fps:
            sims = DataStructs.BulkTanimotoSimilarity(prod_fp, large_fps)
            for smi, sim in zip(large_smis, sims):
                if sim < self.bb_fp_prefilter:
                    continue
                mol = self._bb_mols.get(smi) or self._get_mol(smi)
                if mol is not None and prod.HasSubstructMatch(mol):
                    hits.append(smi)
                    if len(hits) >= max_hits:
                        return hits
        for smi in small_candidates:
            mol = self._bb_mols.get(smi) or self._get_mol(smi)
            if mol is not None and prod.HasSubstructMatch(mol):
                hits.append(smi)
                if len(hits) >= max_hits:
                    break
        return hits

    def _leave_one_out_disconnections(
        self, smiles: str, known_bb: str, max_partners: int | None = None
    ) -> List[_Disconnect]:
        """Given one buyable substructure, search complementary role BBs."""
        if max_partners is None:
            max_partners = self.max_leave_one_out_partners
        if self.role_map is None or known_bb not in self.bb_smi_to_id:
            return []
        bb_id = self.bb_smi_to_id[known_bb]
        prod = self._get_mol(smiles)
        if prod is None:
            return []
        prod_n = prod.GetNumAtoms()
        bb_n = self._bb_n_atoms[known_bb]
        expected = max(2, prod_n - bb_n + 3)
        ranked: List[_Disconnect] = []
        for rid, roles in self.role_map.items():
            if rid not in self.reactions:
                continue
            info = self.reactions[rid]
            if info.get('molecularity', 2) != 2:
                continue
            pq = info.get('product_query')
            if pq is not None and not prod.HasSubstructMatch(pq):
                continue
            if bb_id in roles.get(0, set()):
                other_ids = roles.get(1, set())
                order = lambda other: (known_bb, other)
            elif bb_id in roles.get(1, set()):
                other_ids = roles.get(0, set())
                order = lambda other: (other, known_bb)
            else:
                continue
            partners = []
            for oid in other_ids:
                osmi = self._bb_id_to_smi.get(str(oid))
                if osmi is None:
                    continue
                on = self._bb_n_atoms[osmi]
                if abs(on - expected) <= 4 or on <= 5:
                    partners.append(osmi)
            partners.sort(
                key=lambda s: (
                    self._precursor_priority(s),
                    abs(self._bb_n_atoms[s] - expected),
                )
            )
            partners = partners[:max_partners]
            for osmi in partners:
                smi0, smi1 = order(osmi)
                match = self._forward_match(rid, (smi0, smi1), smiles)
                if match is None:
                    continue
                ordered, product_smi, sim = match
                ranked.append(
                    self._rank_disconnect(
                        rid, ordered, prod_n, product_smi, sim
                    )
                )
                if len(ranked) >= 8:
                    return ranked
        return ranked

    def _bb_pair_disconnections(self, smiles: str) -> List[_Disconnect]:
        """1-step recovery: both precursors are catalog BBs and substructures."""
        if self.role_map is None:
            return []
        prod = self._get_mol(smiles)
        if prod is None:
            return []
        # Only bimolecular reactions (role maps are 2-role).
        relevant = []
        for rid, info in self.reactions.items():
            if info.get('molecularity', 2) != 2:
                continue
            pq = info.get('product_query')
            if pq is None or prod.HasSubstructMatch(pq):
                relevant.append(rid)
        if not relevant:
            return []
        hits = self._bb_substructure_hits(smiles, max_hits=48)
        if len(hits) < 2:
            return []
        hit_ids = {self.bb_smi_to_id[s]: s for s in hits}
        parent_atoms = prod.GetNumAtoms()
        ranked: List[_Disconnect] = []
        seen: Set[Tuple[str, str, str]] = set()
        for rid in relevant:
            roles = self.role_map.get(rid)
            if not roles:
                continue
            role0 = [hit_ids[i] for i in roles.get(0, set()) if i in hit_ids]
            role1 = [hit_ids[i] for i in roles.get(1, set()) if i in hit_ids]
            if not role0 or not role1:
                continue
            role0 = sorted(role0, key=lambda s: self._bb_n_atoms[s], reverse=True)[:8]
            role1 = sorted(role1, key=lambda s: self._bb_n_atoms[s], reverse=True)[:8]
            for smi0 in role0:
                for smi1 in role1:
                    key = (rid, smi0, smi1)
                    if key in seen:
                        continue
                    seen.add(key)
                    match = self._forward_match(rid, (smi0, smi1), smiles)
                    if match is None:
                        continue
                    ordered, product_smi, sim = match
                    ranked.append(
                        self._rank_disconnect(
                            rid, ordered, parent_atoms, product_smi, sim
                        )
                    )
                    if len(ranked) >= self.max_disconnections:
                        return ranked
        ranked.sort()
        return ranked

    def _disconnections(self, smiles: str) -> List[_Disconnect]:
        cached = self._lookup_disconnect_cache(smiles)
        if cached is not None:
            return cached
        parent_atoms = self._cached_atom_count(smiles)
        prod = self._get_mol(smiles)
        ranked: List[_Disconnect] = []
        for reaction_id in self._candidate_reaction_ids(smiles, prod):
            info = self.reactions[reaction_id]
            # Skip primary-impossible reactions early when not approximate;
            # _apply_reverse still runs fallbacks when primary product_query fails.
            if (
                not self.approximate
                and prod is not None
                and info.get('product_query') is not None
                and not self.reverse_rxns[reaction_id]['fallback']
                and not prod.HasSubstructMatch(info['product_query'])
            ):
                continue
            for precursors, product_smi, sim in self._apply_reverse(reaction_id, smiles):
                if not self._membership_ok(reaction_id, precursors):
                    continue
                ranked.append(
                    self._rank_disconnect(
                        reaction_id, precursors, parent_atoms, product_smi, sim
                    )
                )
        # Exact 1-step BB+BB recovery when reverse found no fully buyable cut.
        buyable_before = sum(1 for d in ranked if d.sort_key[3] <= -2)
        if buyable_before == 0:
            ranked.extend(self._bb_pair_disconnections(smiles))
            hits = self._bb_substructure_hits(smiles, max_hits=12)
            for known in hits[:2]:
                ranked.extend(self._leave_one_out_disconnections(smiles, known))
        ranked.sort()
        uniq: List[_Disconnect] = []
        seen: Set[Tuple] = set()
        for disc in ranked:
            # Collapse reactant-order permutations of the same chemical cut.
            key = _disconnect_branch_key(disc.reaction_id, disc.precursors)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(disc)
        ranked = uniq
        if self.prefer_buyable:
            with_buyable = [d for d in ranked if d.sort_key[3] < 0]
            without = [d for d in ranked if d.sort_key[3] == 0]
            ranked = with_buyable[: self.max_disconnections] + without[
                : max(8, self.max_disconnections // 4)
            ]
        else:
            ranked = ranked[: self.max_disconnections]
        self._store_disconnect_cache(smiles, ranked)
        return ranked

    def _search(
        self,
        smiles: str,
        depth_left: int,
        visiting: Set[str],
        allow_proposed_leaf: bool = True,
    ) -> List[_PartialRoute]:
        can = _canonicalize(smiles)
        if can is None:
            return []

        memo_key = (can, depth_left, allow_proposed_leaf)
        if memo_key in self._memo:
            return self._memo[memo_key]

        if can in self.bb_smi_to_id:
            leaf = self._leaf_route(can)
            result = [leaf] if leaf is not None else []
            self._memo[memo_key] = result
            return result

        # Proposed reagents are only valid as precursors, never as the query itself.
        if allow_proposed_leaf and self._is_proposed_reagent(can):
            leaf = self._leaf_route(can)
            result = [leaf] if leaf is not None else []
            self._memo[memo_key] = result
            return result

        # Cycle cut: do NOT memoize — the same node may be reachable without this path.
        if can in visiting:
            return []
        if depth_left <= 0:
            self._memo[memo_key] = []
            return []

        visiting = set(visiting)
        visiting.add(can)
        routes: List[_PartialRoute] = []
        seen_route_sigs: Set[Tuple] = set()
        # Prune chemically identical cuts (reactant-order permutations) BEFORE
        # recursing into precursor searches.
        seen_branches: Set[Tuple] = set()

        for disc in self._disconnections(can):
            branch_key = _disconnect_branch_key(disc.reaction_id, disc.precursors)
            if branch_key in seen_branches:
                continue
            seen_branches.add(branch_key)

            child_route_lists: List[List[_PartialRoute]] = []
            ok = True
            for prec in disc.precursors:
                child = self._search(
                    prec, depth_left - 1, visiting, allow_proposed_leaf=True
                )
                if not child:
                    ok = False
                    break
                child_route_lists.append(child)
            if not ok:
                continue
            for combo in itertools.product(*child_route_lists):
                product_smi = disc.product_smi
                traces = [c.react_trace for c in combo]
                if len(combo) == 1:
                    trace = get_react_trace_unimolecular(
                        traces[0], disc.reaction_id, product_smi
                    )
                elif len(combo) == 2:
                    trace = get_react_trace_bimolecular(
                        traces[0], traces[1], disc.reaction_id, product_smi
                    )
                else:
                    trace = get_react_trace_trimolecular(
                        traces[0],
                        traces[1],
                        traces[2],
                        disc.reaction_id,
                        product_smi,
                    )
                min_sim = min(
                    [c.min_similarity for c in combo] + [disc.similarity]
                )
                bb_ids: List[str] = []
                bb_smis: List[str] = []
                rids: List[str] = []
                n_steps = 1
                for c in combo:
                    bb_ids.extend(c.building_block_ids)
                    bb_smis.extend(c.building_block_smiles)
                    rids.extend(c.reaction_ids)
                    n_steps += c.n_steps
                partial = _PartialRoute(
                    react_trace=trace,
                    reaction_ids=rids + [disc.reaction_id],
                    building_block_ids=bb_ids,
                    building_block_smiles=bb_smis,
                    n_steps=n_steps,
                    reconstructed_smiles=product_smi,
                    min_similarity=min_sim,
                    exact=min_sim >= 1.0 - 1e-12,
                )
                sig = _partial_route_signature(partial)
                if sig in seen_route_sigs:
                    continue
                seen_route_sigs.add(sig)
                routes.append(partial)
            if sum(1 for r in routes if r.n_steps <= 2 and r.exact) >= self.max_routes_per_mol:
                break
            if (
                self.approximate
                and sum(1 for r in routes if r.n_steps <= 2) >= self.max_routes_per_mol * 2
            ):
                break

        routes.sort(
            key=lambda r: (
                0 if r.exact else 1,
                -r.min_similarity,
                r.n_steps,
                sum(1 for i in r.building_block_ids if str(i).startswith('proposed__')),
                self._route_bb_priority(r.building_block_smiles),
                len(r.building_block_ids),
            )
        )
        result = routes[: self.max_routes_per_mol]
        self._memo[memo_key] = result
        return result

    def synthesize(self, smiles: str) -> List[SynthesisRoute]:
        if self.normalize_query:
            prep = _prepare_query(
                smiles,
                force_uncharge=self.force_uncharge,
                simplify_peroxides=self.simplify_peroxides,
                peroxide_mode=self.peroxide_mode,
            )
        else:
            can0 = _canonicalize(smiles)
            prep = (
                PreparedQuery(smiles, can0, can0, [])
                if can0 is not None
                else None
            )
        if prep is None:
            return [
                SynthesisRoute(
                    query_smiles=smiles,
                    found=False,
                    pathway='invalid_smiles',
                )
            ]

        # Search on the (possibly uncharged / peroxide-simplified) form.
        can = prep.search_smiles
        # Final Tanimoto is against the user-facing fragment (charges kept).
        reference = prep.reference_smiles
        transforms = list(prep.transforms)
        transformed = prep.transformed

        # Fresh search memo per query so depth schedule / visiting state cannot
        # leak poisoned empties across molecules.
        self._memo.clear()

        def _route(
            *,
            found: bool,
            n_steps: int = 0,
            reaction_ids=None,
            building_block_ids=None,
            building_block_smiles=None,
            react_trace=None,
            pathway=None,
            exact: bool = False,
            similarity: float = 0.0,
            reconstructed_smiles=None,
        ) -> SynthesisRoute:
            bb_ids = building_block_ids or []
            bb_smis = building_block_smiles or []
            providers, sources, priority = (
                self._enrich_route_meta(bb_ids, bb_smis) if found and bb_smis else ([], [], 0)
            )
            return SynthesisRoute(
                query_smiles=reference,
                found=found,
                n_steps=n_steps,
                reaction_ids=reaction_ids or [],
                building_block_ids=bb_ids,
                building_block_smiles=bb_smis,
                building_block_providers=providers,
                building_block_sources=sources,
                bb_priority=priority,
                react_trace=react_trace,
                pathway=pathway,
                exact=exact,
                similarity=similarity,
                reconstructed_smiles=reconstructed_smiles,
                searched_smiles=can,
                query_transforms=transforms,
            )

        # Depth-0 catalog / proposed hit without searching.
        leaf = self._leaf_route(can)
        # Catalog leaves always; for transformed queries also accept a proposed
        # reagent leaf (e.g. peroxide → small alcohol that is itself terminal).
        if leaf is not None and (
                can in self.bb_smi_to_id
                or (transformed and self._is_proposed_reagent(can))
        ):
            sim_ref = self._tanimoto(reference, can)
            accept_leaf = (
                (not transformed and sim_ref >= 1.0 - 1e-12)
                or (
                    transformed
                    and (
                        'peroxide_to_' in ''.join(transforms)
                        or sim_ref >= self.transform_similarity_threshold
                    )
                )
            )
            if accept_leaf:
                return [
                    _route(
                        found=True,
                        n_steps=0,
                        building_block_ids=leaf.building_block_ids,
                        building_block_smiles=leaf.building_block_smiles,
                        react_trace=leaf.react_trace,
                        pathway=format_pathway(leaf.react_trace),
                        exact=not transformed and sim_ref >= 1.0 - 1e-12,
                        similarity=sim_ref,
                        reconstructed_smiles=can,
                    )
                ]

        # Cheap prefilter: junk chem / zero templates never pay for deep search.
        skip_reason = self._early_skip_prefilter_reason(can)
        if skip_reason is not None:
            logging.debug('Early skip %s for %s', skip_reason, can)
            return [_route(found=False, pathway=skip_reason)]

        exact_partials: List[_PartialRoute] = []
        approx_partials: List[_PartialRoute] = []
        seen_route_sigs: Set[Tuple] = set()
        # Prefer shallow routes first (depth 1, then 2), then max_depth only if
        # nothing acceptable was found — skip the deep downdraft once a hit exists.
        if self.max_depth <= 2:
            depth_schedule = list(range(1, self.max_depth + 1))
        else:
            depth_schedule = [1, 2, self.max_depth]
        for depth in depth_schedule:
            # After shallow miss, only escalate if there is catalog/proposed
            # progress or non-excluded (useful) disconnections.
            if self.early_skip and depth > 2 and not self._has_shallow_progress(can):
                logging.debug(
                    'Early skip early_skip:no_shallow_progress for %s', can
                )
                return [_route(found=False, pathway='early_skip:no_shallow_progress')]
            for partial in self._search(can, depth, set(), allow_proposed_leaf=False):
                sig = _partial_route_signature(partial)
                if sig in seen_route_sigs:
                    continue
                seen_route_sigs.add(sig)
                # Exactness of the chemical plan is vs the searched (simplified) form.
                sim_search = self._tanimoto(can, partial.reconstructed_smiles)
                # User-facing score is vs the original fragment (may be charged / peroxide).
                sim_ref = self._tanimoto(reference, partial.reconstructed_smiles)
                if transformed:
                    # Require an exact route to the simplified molecule.
                    if sim_search < 1.0 - 1e-12 or not partial.exact:
                        continue
                    # Peroxide → analog: accept any exact plan for the simplified
                    # form (Tanimoto to original is often low after ring opening).
                    # Uncharge-only: require similarity to the charged input.
                    peroxide_xf = any(t.startswith('peroxide_to_') for t in transforms)
                    if (
                        not peroxide_xf
                        and sim_ref < self.transform_similarity_threshold
                    ):
                        continue
                    enriched = _PartialRoute(
                        react_trace=partial.react_trace,
                        reaction_ids=partial.reaction_ids,
                        building_block_ids=partial.building_block_ids,
                        building_block_smiles=partial.building_block_smiles,
                        n_steps=partial.n_steps,
                        reconstructed_smiles=partial.reconstructed_smiles,
                        min_similarity=sim_ref,
                        exact=False,
                    )
                    approx_partials.append(enriched)
                else:
                    min_keep = (
                        1.0 - 1e-12 if not self.approximate else self.similarity_threshold
                    )
                    if sim_ref < min_keep:
                        continue
                    enriched = _PartialRoute(
                        react_trace=partial.react_trace,
                        reaction_ids=partial.reaction_ids,
                        building_block_ids=partial.building_block_ids,
                        building_block_smiles=partial.building_block_smiles,
                        n_steps=partial.n_steps,
                        reconstructed_smiles=partial.reconstructed_smiles,
                        min_similarity=min(partial.min_similarity, sim_ref),
                        exact=sim_ref >= 1.0 - 1e-12 and partial.exact,
                    )
                    if enriched.exact:
                        exact_partials.append(enriched)
                    else:
                        approx_partials.append(enriched)
            # Stop early only when we already have a good shallow hit that does
            # not rely on excluded templates (default: retro_cc_wurtz). Pure
            # Wurtz / junk shallow hits should not block deeper chemistry
            # (e.g. acyliminium + cyclic urea).
            good_exact = [
                p
                for p in exact_partials
                if not any(
                    rid in self.early_skip_exclude_reactions for rid in p.reaction_ids
                )
            ]
            if good_exact and depth >= 2:
                break
            if (exact_partials or approx_partials) and depth >= self.max_depth:
                break
            if approx_partials and not exact_partials and depth >= 2:
                break

        partials = exact_partials if exact_partials else approx_partials
        if not partials:
            return [_route(found=False)]

        partials.sort(
            key=lambda r: (
                0 if r.exact else 1,
                -r.min_similarity,
                r.n_steps,
                sum(1 for i in r.building_block_ids if str(i).startswith('proposed__')),
                self._route_bb_priority(r.building_block_smiles),
                len(r.building_block_ids),
            )
        )
        results = []
        for partial in partials[: self.max_routes_per_mol]:
            results.append(
                _route(
                    found=True,
                    n_steps=partial.n_steps,
                    reaction_ids=partial.reaction_ids,
                    building_block_ids=partial.building_block_ids,
                    building_block_smiles=partial.building_block_smiles,
                    react_trace=partial.react_trace,
                    pathway=format_pathway(partial.react_trace),
                    exact=partial.exact,
                    similarity=partial.min_similarity,
                    reconstructed_smiles=partial.reconstructed_smiles,
                )
            )
        return results

    def synthesize_many(
        self,
        smiles_list: Sequence[str],
        n_workers: int | None = None,
        on_molecule_done=None,
    ) -> List[SynthesisRoute]:
        """Synthesize pathways for many molecules.

        Parameters
        ----------
        smiles_list
            Query SMILES.
        n_workers
            Process-pool size. Defaults to ``self.n_workers``. Use 1 for sequential.
        on_molecule_done
            Optional callback ``(index, smiles, routes)`` invoked after each
            molecule finishes (useful for streaming CSV writes).
        """
        workers = self.n_workers if n_workers is None else max(1, int(n_workers))
        smiles_list = list(smiles_list)
        if not smiles_list:
            return []

        if workers <= 1 or len(smiles_list) == 1:
            rows: List[SynthesisRoute] = []
            for i, smi in enumerate(smiles_list):
                try:
                    routes = self.synthesize(smi)
                finally:
                    self._flush_shared_disconnect_cache()
                if on_molecule_done is not None:
                    on_molecule_done(i, smi, routes)
                rows.extend(routes)
            self.save_disconnect_cache()
            return rows

        chunksize = max(1, len(smiles_list) // (workers * 4))
        rows: List[SynthesisRoute] = []
        # Prefer fork so workers inherit the already-loaded chemical space
        # (avoids multi-second BB reload per process). Fall back to spawn+init
        # on platforms where fork is unavailable / unsafe.
        use_fork = sys.platform != 'win32' and 'fork' in mp.get_all_start_methods()
        ctx = mp.get_context('fork' if use_fork else 'spawn')
        # NOTE: multiprocessing.Manager.dict for live cross-worker sharing causes
        # severe futex contention under RDKit-heavy loads. Keep per-worker local
        # caches; persist via disconnect_cache_path so sequential batches share.
        logging.info(
            f'Parallel retrosynthesis with {workers} workers '
            f'({len(smiles_list)} molecules, chunksize={chunksize}, '
            f'start={"fork" if use_fork else "spawn"})'
        )
        try:
            if use_fork:
                global _WORKER_SYNTH, _SHARED_DISCONNECT_CACHE
                _SHARED_DISCONNECT_CACHE = None
                self._shared_disconnect_cache = None
                _WORKER_SYNTH = self
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                    for i, (routes, updates) in enumerate(
                        pool.map(_worker_synthesize, smiles_list, chunksize=chunksize)
                    ):
                        _merge_disconnect_updates(self, updates)
                        if on_molecule_done is not None:
                            on_molecule_done(i, smiles_list[i], routes)
                        rows.extend(routes)
                _WORKER_SYNTH = None
            else:
                config = self._worker_config()
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=ctx,
                    initializer=_init_synth_worker,
                    initargs=(config, None),
                ) as pool:
                    for i, (routes, updates) in enumerate(
                        pool.map(_worker_synthesize, smiles_list, chunksize=chunksize)
                    ):
                        _merge_disconnect_updates(self, updates)
                        if on_molecule_done is not None:
                            on_molecule_done(i, smiles_list[i], routes)
                        rows.extend(routes)
        finally:
            self.save_disconnect_cache()
        return rows


def routes_to_dataframe(routes: Iterable[SynthesisRoute]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in routes])
