"""Helpers for chemical-space reaction SMARTS (LDDM forward vs retrosynthesis)."""
from __future__ import annotations

from rdkit.Chem import rdChemReactions


def reactant_count(smarts: str) -> int | None:
    """Return RDKit reactant-template count, or None if SMARTS is invalid."""
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
    except Exception:
        return None
    if rxn is None:
        return None
    return rxn.GetNumReactantTemplates()


def product_count(smarts: str) -> int | None:
    """Return RDKit product-template count, or None if SMARTS is invalid."""
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
    except Exception:
        return None
    if rxn is None:
        return None
    return rxn.GetNumProductTemplates()


def _single_product_side(smarts: str) -> bool:
    """True if the product side is a single component (no parenthetical A.B)."""
    if '>>' not in smarts:
        return False
    product = smarts.split('>>', 1)[1].strip()
    if product.startswith('(') or '.' in product:
        return False
    return True


def is_bimolecular_reaction(smarts: str) -> bool:
    """Return True if SMARTS has exactly two reactants and one product.

    Matches the constraints enforced by ``prepare_chemical_space.py`` / LDDM
    synthesizable design. Templates with parenthetical multi-component patterns
    are rejected even when RDKit reports two reactant templates.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
    except Exception:
        return False
    if rxn is None:
        return False
    if rxn.GetNumReactantTemplates() != 2 or rxn.GetNumProductTemplates() != 1:
        return False
    reactant_side = smarts.split('>>')[0]
    return len(reactant_side.split('.')) == 2


def is_retrosynthesis_reaction(smarts: str) -> bool:
    """True for uni-/bi-/trimolecular single-product templates.

    Used by local retrosynthesis (``LocalRetrosynthesizer``), which is not
    limited to LDDM's bimolecular forward-generation constraint.
    """
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
    except Exception:
        return False
    if rxn is None:
        return False
    n_reac = rxn.GetNumReactantTemplates()
    n_prod = rxn.GetNumProductTemplates()
    if n_prod != 1 or n_reac < 1 or n_reac > 3:
        return False
    if not _single_product_side(smarts):
        return False
    reactant_side = smarts.split('>>')[0]
    # Dot-count must match RDKit reactant templates (reject weird embeddings).
    return len(reactant_side.split('.')) == n_reac


def reaction_molecularity(smarts: str) -> int | None:
    """1 / 2 / 3 if retrosynthesis-compatible, else None."""
    if not is_retrosynthesis_reaction(smarts):
        return None
    return reactant_count(smarts)
