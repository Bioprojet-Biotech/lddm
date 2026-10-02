"""Independent forward verification of LDDM retrosynthesis routes.

Mirrors the spirit of ReaSyn ``eval_recon.py`` (reconstruction = product SMILES
equals the query target), but also checks that every step in a ``react_trace``
actually fires under the recorded reaction SMARTS.

Refs:
https://github.com/NVIDIA-Digital-Bio/reasyn/blob/main/scripts/eval_recon.py
https://github.com/wenhao-gao/synformer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from lddm.reactions.reaction_utils import ReactionTree, run_compiled_reaction, run_reaction_smarts


def canonicalize(smiles: str | None) -> Optional[str]:
    if smiles is None or smiles == '' or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _tanimoto(a: str, b: str, fpgen=None) -> float:
    if a == b:
        return 1.0
    mol_a, mol_b = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if mol_a is None or mol_b is None:
        return 0.0
    if fpgen is None:
        fpgen = AllChem.GetMorganGenerator(2, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(mol_a), fpgen.GetFingerprint(mol_b))
    )


@dataclass
class StepVerify:
    react_id: str
    reactants: List[str]
    claimed_product: str
    forward_ok: bool
    forward_products: List[str] = field(default_factory=list)
    error: str = ''


@dataclass
class RouteVerify:
    """Quality verdict for one route row."""

    query_smiles: str
    searched_smiles: str
    reconstructed_smiles: str
    found: bool
    # Independent forward replay of react_trace.
    forward_replay_ok: bool
    replayed_product: Optional[str]
    # Does replayed / claimed product equal the molecule we actually searched?
    reconstructs_searched: bool
    # Does it equal the original query SMILES (may fail after charge normalize)?
    reconstructs_query: bool
    # Claimed reconstructed_smiles equals searched (string-level, no replay).
    claimed_matches_searched: bool
    catalog_only: bool
    n_proposed_bbs: int
    n_steps: int
    reaction_ids: List[str]
    react_trace: str = ''
    pathway: str = ''
    building_block_ids: str = ''
    building_block_smiles: str = ''
    step_checks: List[StepVerify] = field(default_factory=list)
    similarity_to_searched: float = 0.0
    similarity_to_query: float = 0.0

    @property
    def good_route(self) -> bool:
        """True reconstruction: forward-valid and rebuilds the searched molecule."""
        return self.forward_replay_ok and self.reconstructs_searched

    @property
    def good_catalog_route(self) -> bool:
        return self.good_route and self.catalog_only

    def to_dict(self) -> dict:
        d = asdict(self)
        d['good_route'] = self.good_route
        d['good_catalog_route'] = self.good_catalog_route
        d['reaction_ids'] = ';'.join(self.reaction_ids)
        d['step_checks'] = [asdict(s) for s in self.step_checks]
        return d


def _forward_products(
    reaction_id: str,
    reactants: Sequence[str],
    reactions: Dict[str, dict],
    forward_rxns: Optional[Dict[str, Any]] = None,
) -> List[str]:
    info = reactions.get(reaction_id)
    if info is None:
        return []
    # Try all reactant orderings (bi/tri).
    from itertools import permutations

    seen: set[str] = set()
    products: List[str] = []
    n = info.get('molecularity') or len(reactants)
    if len(reactants) != n:
        # Still try as given.
        orders = [tuple(reactants)]
    else:
        orders = list(dict.fromkeys(permutations(reactants)))
    compiled = (forward_rxns or {}).get(reaction_id)
    for ordered in orders:
        mols = [Chem.MolFromSmiles(s) for s in ordered]
        if any(m is None for m in mols):
            continue
        if compiled is not None:
            outs = run_compiled_reaction(compiled, mols, explicit_hs=info.get('explicit_hs', False))
        else:
            outs = run_reaction_smarts(
                info['forward_smarts'], ordered, explicit_hs=info.get('explicit_hs', False)
            )
        if not outs:
            continue
        for p in outs:
            can = canonicalize(p)
            if can and can not in seen:
                seen.add(can)
                products.append(can)
    return products


def replay_react_trace(
    react_trace: str,
    reactions: Dict[str, dict],
    forward_rxns: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], List[StepVerify]]:
    """Bottom-up forward execution of a react_trace.

    Returns (final_product_or_None, per-step checks).
    A step is forward_ok iff the claimed product is among RDKit products of the
    recorded reaction applied to the *replayed* (or leaf) reactants.
    """
    if not react_trace:
        return None, []
    tree = ReactionTree(react_trace)
    checks: List[StepVerify] = []

    def walk(node: dict) -> Optional[str]:
        eds = node.get('educts') or []
        claimed = canonicalize(node.get('product'))
        if not eds:
            return claimed
        reactant_smis: List[str] = []
        for e in eds:
            child = walk(e)
            if child is None:
                checks.append(
                    StepVerify(
                        react_id=str(node.get('react_id') or '?'),
                        reactants=[],
                        claimed_product=claimed or '',
                        forward_ok=False,
                        error='child_failed',
                    )
                )
                return None
            reactant_smis.append(child)
        rid = str(node.get('react_id') or '')
        if rid not in reactions:
            checks.append(
                StepVerify(
                    react_id=rid or '?',
                    reactants=reactant_smis,
                    claimed_product=claimed or '',
                    forward_ok=False,
                    error='unknown_reaction',
                )
            )
            return None
        products = _forward_products(rid, reactant_smis, reactions, forward_rxns)
        ok = claimed is not None and claimed in products
        checks.append(
            StepVerify(
                react_id=rid,
                reactants=reactant_smis,
                claimed_product=claimed or '',
                forward_ok=ok,
                forward_products=products[:8],
                error='' if ok else ('product_mismatch' if products else 'no_products'),
            )
        )
        if not ok:
            return None
        return claimed

    final = walk(tree.tree)
    return final, checks


def verify_route_row(
    row: pd.Series | dict,
    reactions: Dict[str, dict],
    forward_rxns: Optional[Dict[str, Any]] = None,
    fpgen=None,
) -> RouteVerify:
    """Evaluate one synthesize / benchmark CSV row for reconstruction quality."""
    if not isinstance(row, dict):
        row = row.to_dict()
    query = str(row.get('query_smiles') or '')
    searched = str(row.get('searched_smiles') or query)
    claimed_recon = str(row.get('reconstructed_smiles') or '')
    found = bool(row.get('found'))
    trace = str(row.get('react_trace') or '')
    bb_ids = str(row.get('building_block_ids') or '')
    n_proposed = bb_ids.count('proposed__')
    catalog_only = found and n_proposed == 0 and bool(bb_ids)
    rids = [
        x for x in str(row.get('reaction_ids') or '').split(';')
        if x and x.lower() != 'nan'
    ]
    n_steps = int(row.get('n_steps') or 0) if str(row.get('n_steps')) not in ('', 'nan', 'None') else 0

    q_can = canonicalize(query)
    s_can = canonicalize(searched) or q_can
    c_can = canonicalize(claimed_recon)

    replayed, checks = (None, [])
    if found and trace and trace.lower() != 'nan':
        replayed, checks = replay_react_trace(trace, reactions, forward_rxns)

    forward_ok = bool(found and trace and trace.lower() != 'nan' and replayed is not None and all(c.forward_ok for c in checks))
    # Reconstruction only counts when forward replay succeeded.
    reconstructs_searched = bool(forward_ok and replayed and s_can and replayed == s_can)
    reconstructs_query = bool(forward_ok and replayed and q_can and replayed == q_can)
    claimed_matches = bool(c_can and s_can and c_can == s_can)

    product_for_sim = replayed or c_can
    sim_s = _tanimoto(product_for_sim, s_can, fpgen) if product_for_sim and s_can else 0.0
    sim_q = _tanimoto(product_for_sim, q_can, fpgen) if product_for_sim and q_can else 0.0

    return RouteVerify(
        query_smiles=query,
        searched_smiles=searched,
        reconstructed_smiles=claimed_recon,
        found=found,
        forward_replay_ok=forward_ok,
        replayed_product=replayed,
        reconstructs_searched=reconstructs_searched,
        reconstructs_query=reconstructs_query,
        claimed_matches_searched=claimed_matches,
        catalog_only=catalog_only,
        n_proposed_bbs=n_proposed,
        n_steps=n_steps,
        reaction_ids=rids,
        react_trace=trace,
        pathway=str(row.get('pathway') or ''),
        building_block_ids=bb_ids,
        building_block_smiles=str(row.get('building_block_smiles') or ''),
        step_checks=checks,
        similarity_to_searched=sim_s,
        similarity_to_query=sim_q,
    )


def evaluate_routes_dataframe(
    df: pd.DataFrame,
    reactions: Dict[str, dict],
    forward_rxns: Optional[Dict[str, Any]] = None,
    *,
    pick_best: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Score all routes; optionally keep the best route per query.

    Best = good_catalog > good_route > reconstructs_searched > forward_ok >
    similarity > fewer proposed > fewer steps.
    """
    fpgen = AllChem.GetMorganGenerator(2, fpSize=2048)
    verified: List[RouteVerify] = []
    for _, row in df.iterrows():
        verified.append(verify_route_row(row, reactions, forward_rxns, fpgen=fpgen))

    vdf = pd.DataFrame([v.to_dict() for v in verified])
    # Expand without nested step_checks in the summary CSV.
    if 'step_checks' in vdf.columns:
        vdf = vdf.drop(columns=['step_checks'])

    if pick_best and len(vdf):
        def _key(r):
            return (
                int(r.good_catalog_route),
                int(r.good_route),
                int(r.reconstructs_searched),
                int(r.forward_replay_ok),
                float(r.similarity_to_searched),
                -int(r.n_proposed_bbs),
                -int(r.n_steps),
            )

        best_idx = []
        for q, g in vdf.groupby('query_smiles', sort=False):
            scores = g.apply(_key, axis=1)
            best_idx.append(scores.idxmax())
        best = vdf.loc[best_idx].reset_index(drop=True)
    else:
        best = vdf

    n = best['query_smiles'].nunique() if len(best) else 0
    def _rate(col: str) -> float:
        if n == 0:
            return 0.0
        return float(best[col].astype(bool).sum()) / n

    stats = {
        'n_queries': int(n),
        # ReaSyn-style rates
        'success_rate': _rate('found'),  # any pathway returned
        'reconstruction_rate': _rate('good_route'),  # forward-valid + rebuilds searched
        'reconstruction_rate_vs_query': _rate('reconstructs_query'),
        'forward_valid_rate': _rate('forward_replay_ok'),
        'catalog_route_rate': _rate('good_catalog_route'),
        'claimed_match_searched_rate': _rate('claimed_matches_searched'),
        'n_success': int(best['found'].astype(bool).sum()) if n else 0,
        'n_reconstructed': int(best['good_route'].astype(bool).sum()) if n else 0,
        'n_reconstructed_vs_query': int(best['reconstructs_query'].astype(bool).sum()) if n else 0,
        'n_forward_valid': int(best['forward_replay_ok'].astype(bool).sum()) if n else 0,
        'n_catalog_routes': int(best['good_catalog_route'].astype(bool).sum()) if n else 0,
        'mean_similarity_to_searched': float(
            best.loc[best['found'].astype(bool), 'similarity_to_searched'].mean()
        ) if n and best['found'].astype(bool).any() else 0.0,
        'mean_n_steps_reconstructed': float(
            best.loc[best['good_route'].astype(bool), 'n_steps'].mean()
        ) if n and best['good_route'].astype(bool).any() else 0.0,
        'fraction_routes_with_proposed_bbs': float(
            (best['n_proposed_bbs'] > 0).mean()
        ) if n else 0.0,
    }
    return best, stats


def evaluate_with_synthesizer(
    df: pd.DataFrame,
    synthesizer,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper using a loaded ``LocalRetrosynthesizer``."""
    return evaluate_routes_dataframe(
        df,
        reactions=synthesizer.reactions,
        forward_rxns=getattr(synthesizer, 'forward_rxns', None),
        **kwargs,
    )
