from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from typing import Collection, Set

from rdkit import Chem
from rdkit.Chem import Draw
import logging


#####################################################################
########################## Enamine reactions ########################
#####################################################################

def clean_smi(smi: str, replacement_atomic_num: int = 1) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")

    rw_mol = Chem.RWMol(mol)

    replacement_map = {
        92: replacement_atomic_num,  # U -> H (default replacement)
        93: replacement_atomic_num,  # Np -> H (default replacement)
        94: replacement_atomic_num,  # Pu -> H (default replacement)
    }

    for atom in rw_mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in replacement_map:
            atom.SetAtomicNum(replacement_map[atomic_num])

    # Convert back to mol, remove Hs, sanitize
    mol = rw_mol.GetMol()
    mol = Chem.RemoveAllHs(mol)
    Chem.SanitizeMol(mol)
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))

def find_connectors(smi: str) -> Set[str]:
    connectors = set()
    for conn in ['[U]', '[Np]', '[Pu]']:
        if conn in smi:
            connectors.add(conn)
    return connectors


def connect_synthons(
    smi1: str, 
    smi2: str, 
    connectors: Collection[str] = ['[U]', '[Np]', '[Pu]']
) -> str:
    # Replacing connectors with dummy atoms to enable RDKit functionality
    for i, conn in enumerate(connectors):
        repl = f'[*:{i+1}]'
        smi1 = smi1.replace(conn, repl)
        smi2 = smi2.replace(conn, repl)

    synthon1 = Chem.MolFromSmiles(smi1)
    synthon2 = Chem.MolFromSmiles(smi2)

    # Combining molecules based on the exit vectors
    combo = Chem.molzip(synthon1, synthon2)
    combo = Chem.RemoveAllHs(combo)
    return Chem.CanonSmiles(Chem.MolToSmiles(combo))

#####################################################################
######################### Explicit reactions ########################
#####################################################################

def run_compiled_reaction(
    rxn: rdChemReactions.ChemicalReaction,
    reactant_mols: Collection[Chem.Mol],
    explicit_hs: bool = False,
) -> Collection[str] | None:
    """Run a precompiled RDKit reaction on reactant mols; return unique product SMILES."""
    mols = list(reactant_mols)
    if not mols or rxn is None:
        return None
    prepared = []
    for mol in mols:
        if mol is None:
            return None
        prepared.append(Chem.AddHs(mol) if explicit_hs else Chem.Mol(mol))
    try:
        products = rxn.RunReactants(tuple(prepared))
    except Exception:
        return None

    uniqps = {}
    for p in products:
        try:
            smi = Chem.MolToSmiles(Chem.RemoveAllHs(p[0]))
            uniqps[smi] = p[0]
        except Exception:
            continue
    if not uniqps:
        return None
    return sorted(uniqps.keys())


def run_reaction_smarts(
    reaction_smarts: str,
    reactant_smiles: Collection[str],
    explicit_hs: bool = False,
    compiled_rxn: rdChemReactions.ChemicalReaction | None = None,
    reactant_mols: Collection[Chem.Mol] | None = None,
) -> Collection[str] | None:
    """Run a SMARTS reaction with 1–N reactant SMILES; return unique product SMILES.

    Pass ``compiled_rxn`` / ``reactant_mols`` to avoid recompiling SMARTS or
    re-parsing SMILES on hot paths (e.g. retrosynthesis forward verification).
    """
    if compiled_rxn is not None and reactant_mols is not None:
        return run_compiled_reaction(compiled_rxn, reactant_mols, explicit_hs=explicit_hs)

    smiles_list = list(reactant_smiles)
    if not smiles_list:
        return None
    rxn = compiled_rxn
    if rxn is None:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        except Exception:
            return None
    if rxn is None:
        return None
    if reactant_mols is not None:
        return run_compiled_reaction(rxn, reactant_mols, explicit_hs=explicit_hs)
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mols.append(mol)
    return run_compiled_reaction(rxn, mols, explicit_hs=explicit_hs)


def run_bimolecular_reaction_smarts(
    reaction_smarts: str,
    bb_smiles_x: str,
    bb_smiles_y: str,
    explicit_hs: bool = False
) -> Collection[str]:
    return run_reaction_smarts(
        reaction_smarts, (bb_smiles_x, bb_smiles_y), explicit_hs=explicit_hs
    )


def run_unimolecular_reaction_smarts(
    reaction_smarts: str,
    bb_smiles: str,
    explicit_hs: bool = False
) -> Collection[str]:
    return run_reaction_smarts(reaction_smarts, (bb_smiles,), explicit_hs=explicit_hs)


def run_trimolecular_reaction_smarts(
    reaction_smarts: str,
    bb_smiles_x: str,
    bb_smiles_y: str,
    bb_smiles_z: str,
    explicit_hs: bool = False,
) -> Collection[str]:
    return run_reaction_smarts(
        reaction_smarts,
        (bb_smiles_x, bb_smiles_y, bb_smiles_z),
        explicit_hs=explicit_hs,
    )


class ReactionTree:
    def __init__(self, react_trace: str):
        self.react_trace = react_trace
        self.tree = self._parse_trace(react_trace)

    def _parse_trace(self, trace: str) -> dict:
        if trace.startswith('<') and trace.endswith('>'):
            inner = trace[1:-1]
        else:
            inner = trace
        if ':' not in inner:
            educts_part, react_id, product_part = None, None, inner
        else:
            educts_part, react_id, product_part = inner.rsplit(':', 2)
        # SMILES may contain '-' (branches); id is always the suffix after the last '-'.
        if '-' in product_part:
            product, prod_id = product_part.rsplit('-', 1)
        else:
            product, prod_id = product_part, None
        if product == 'NA':
            product = None
        if prod_id == 'NA':
            prod_id = None
        if educts_part is not None:
            educts = []
            buf = ''
            d = 0
            for ch in educts_part:
                if ch == '<':
                    d += 1
                elif ch == '>':
                    d -= 1
                if ch == ';' and d == 0:
                    educts.append(buf)
                    buf = ''
                else:
                    buf += ch
            if buf:
                educts.append(buf)
            nodes = []
            for ed in educts:
                ed = ed.strip()
                nodes.append(self._parse_trace(ed))
        else:
            nodes = None
        return {'educts': nodes, 'react_id': react_id, 'product': product, 'prod_id': prod_id}

    def depth(self) -> int:
        def d(node):
            eds = node.get('educts') or []
            if not eds:
                return 1
            return 1 + max(d(e) for e in eds)
        return d(self.tree)

    def reaction_steps(self) -> list[dict]:
        """Bottom-up reaction steps: reactants → product with react_id.

        Each step is ``{'reactants': [smiles, ...], 'product': smiles, 'react_id': str}``.
        Building-block leaves are omitted.
        """
        steps: list[dict] = []

        def collect(n: dict) -> None:
            eds = n.get('educts') or []
            for e in eds:
                collect(e)
            if eds:
                reactants = [e.get('product') for e in eds]
                if any(r is None for r in reactants) or n.get('product') is None:
                    return
                steps.append(
                    {
                        'reactants': reactants,
                        'product': n['product'],
                        'react_id': n.get('react_id') or '?',
                    }
                )

        collect(self.tree)
        return steps

    def draw(self, output_path: str | None = None, show: bool = True):
        """Render reaction steps.

        Prefer Cairo/PNG via ``lddm.reactions.render_pathways`` when saving.
        Falls back to matplotlib only for interactive ``show=True`` without a path.
        """
        if output_path is not None:
            from lddm.reactions.render_pathways import render_react_trace

            render_react_trace(self.react_trace, output_path)
            return
        if not show:
            return
        import matplotlib.pyplot as plt

        steps = self.reaction_steps()
        for i, step in enumerate(steps, 1):
            ed_mols = [Chem.MolFromSmiles(s) for s in step['reactants']]
            prod_mol = Chem.MolFromSmiles(step['product'])
            n = len(ed_mols)
            fig, axs = plt.subplots(1, n + 2, figsize=(4 * (n + 2), 4))
            if n + 2 == 1:
                axs = [axs]
            for j, mol in enumerate(ed_mols):
                axs[j].imshow(Draw.MolToImage(mol, size=(300, 300)))
                axs[j].axis('off')
                axs[j].set_title(f'Educt {j + 1}')
            axs[n].text(0.5, 0.5, '→', fontsize=40, ha='center', va='center')
            axs[n].axis('off')
            axs[n + 1].imshow(Draw.MolToImage(prod_mol, size=(300, 300)))
            axs[n + 1].axis('off')
            axs[n + 1].set_title('Product')
            plt.suptitle(f"Step {i} (ID: {step['react_id']})")
            plt.tight_layout()
            plt.show()


def get_react_trace_building_block(smi, id=None):
    if smi is None:
        smi = 'NA'
    if id is None:
        id = 'NA'
    return f'<{smi}-{id}>'

def get_react_trace_unimolecular(tr, reaction_id, product_smi, prod_id=None):
    if prod_id is None:
        prod_id = 'NA'
    return f'<{tr}:{reaction_id}:{product_smi}-{prod_id}>'

def get_react_trace_bimolecular(tr1, tr2, reaction_id, product_smi, prod_id=None):
    if prod_id is None:
        prod_id = 'NA'
    educts = f'{tr1};{tr2}'
    return f'<{educts}:{reaction_id}:{product_smi}-{prod_id}>'

def get_react_trace_trimolecular(tr1, tr2, tr3, reaction_id, product_smi, prod_id=None):
    if prod_id is None:
        prod_id = 'NA'
    educts = f'{tr1};{tr2};{tr3}'
    return f'<{educts}:{reaction_id}:{product_smi}-{prod_id}>'
