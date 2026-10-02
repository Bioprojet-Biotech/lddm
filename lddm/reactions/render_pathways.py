"""Render LDDM ``react_trace`` pathways as reaction diagrams (PNG + HTML).

Style follows ReaSyn's ``scripts/render_pathways.py`` (RDKit Cairo reaction
drawing + stacked panels + HTML index), adapted to LDDM nested react traces:
https://github.com/NVIDIA-Digital-Bio/reasyn

Also provides a chemist-facing **all-embedded** single-file HTML report
(``write_embedded_retrosynthesis_html``) with inline SVG drawings — no external
PNG assets. Reaction templates in SynSpace / ReaSyn spaces expose SMARTS + name
only; optional offline procedure enrichment (curated cards + local OpenAlex
refs) can attach typical conditions via ``enrich_procedures=True``.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

from lddm.reactions.reaction_utils import ReactionTree

# SynSpace / ReaSyn reaction JSON fields we can surface to chemists.
# Yields, solvents, temperatures, catalysts, and literature refs are NOT present.
_REACTION_META_KEYS = (
    'id',
    'name',
    'reaction',
    'explicit_hs',
    'source',
    'molecularity',
    'covers',
)


@dataclass(frozen=True)
class RenderStep:
    react_id: str
    reactants: tuple[str, ...]
    product: str


def react_trace_to_steps(react_trace: str) -> List[RenderStep]:
    """Convert an LDDM react_trace into ordered reaction steps."""
    tree = ReactionTree(react_trace)
    steps: List[RenderStep] = []
    for step in tree.reaction_steps():
        steps.append(
            RenderStep(
                react_id=str(step['react_id']),
                reactants=tuple(step['reactants']),
                product=str(step['product']),
            )
        )
    return steps


def _prepare_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return mol


def _build_reaction(step: RenderStep) -> rdChemReactions.ChemicalReaction:
    reactant_mols = [_prepare_mol(s) for s in step.reactants]
    product_mol = _prepare_mol(step.product)
    if all(m is not None for m in reactant_mols) and product_mol is not None:
        rxn = rdChemReactions.ChemicalReaction()
        for mol in reactant_mols:
            rxn.AddReactantTemplate(mol)
        rxn.AddProductTemplate(product_mol)
        rdChemReactions.Compute2DCoordsForReaction(rxn)
        return rxn
    # Fallback empty reaction with whatever parsed.
    rxn = rdChemReactions.ChemicalReaction()
    for mol in reactant_mols:
        if mol is not None:
            rxn.AddReactantTemplate(mol)
    if product_mol is not None:
        rxn.AddProductTemplate(product_mol)
    if rxn.GetNumReactantTemplates() or rxn.GetNumProductTemplates():
        rdChemReactions.Compute2DCoordsForReaction(rxn)
    return rxn


def _reaction_atom_counts(rxn: rdChemReactions.ChemicalReaction) -> list[int]:
    counts: list[int] = []
    for i in range(rxn.GetNumReactantTemplates()):
        counts.append(rxn.GetReactantTemplate(i).GetNumAtoms())
    for i in range(rxn.GetNumProductTemplates()):
        counts.append(rxn.GetProductTemplate(i).GetNumAtoms())
    return counts


def _panel_height(
    atom_counts: list[int],
    panel_height: int,
    max_panel_height: int,
    n_extra_components: int = 0,
) -> int:
    if not atom_counts:
        return panel_height
    max_atoms = max(atom_counts)
    n_components = len(atom_counts) + n_extra_components
    suggested = (
        panel_height
        + max(0, max_atoms - 28) * 3
        + max(0, n_components - 2) * 18
    )
    return min(max_panel_height, max(panel_height, suggested))


def _draw_reaction_step(
    step: RenderStep,
    max_width: int,
    panel_height: int,
    max_panel_height: int,
) -> Image.Image:
    rxn = _build_reaction(step)
    atom_counts = _reaction_atom_counts(rxn)
    height = _panel_height(atom_counts, panel_height, max_panel_height)
    if not atom_counts:
        canvas = Image.new('RGB', (max_width, height), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f'Unrenderable step ({step.react_id})', fill='black')
        return canvas

    draw_opts = rdMolDraw2D.MolDrawOptions()
    draw_opts.padding = 0.02
    draw_opts.fixedBondLength = -1
    draw_opts.fixedScale = -1
    drawer = rdMolDraw2D.MolDraw2DCairo(max_width, height)
    drawer.SetDrawOptions(draw_opts)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def _wrap_text(text: str, max_chars: int) -> list[str]:
    if not text:
        return ['']
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _annotate_image(image: Image.Image, title: str | list[str], width: int) -> Image.Image:
    lines = [title] if isinstance(title, str) else title
    font = ImageFont.load_default()
    line_height = 14
    padding = 8
    header_h = padding * 2 + line_height * len(lines)
    canvas = Image.new('RGB', (width, image.height + header_h), 'white')
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill='black', font=font)
        y += line_height
    return canvas


def _stack_images(images: list[Image.Image], width: int) -> Image.Image:
    height = sum(image.height for image in images)
    canvas = Image.new('RGB', (width, height), 'white')
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def _draw_overview(
    query: str,
    reconstructed: str | None,
    n_steps: int,
    exact: bool,
    similarity: float,
    width: int,
    panel_height: int,
    max_panel_height: int,
) -> Image.Image:
    mols = [_prepare_mol(query)]
    legends = ['Query']
    if reconstructed and reconstructed != query:
        mols.append(_prepare_mol(reconstructed))
        legends.append(
            f'Reconstructed (exact={exact}, sim={similarity:.3f}, steps={n_steps})'
        )
    else:
        legends[0] = f'Query (exact={exact}, steps={n_steps})'

    valid = [(mol, legend) for mol, legend in zip(mols, legends) if mol is not None]
    if not valid:
        return Image.new('RGB', (width, 120), 'white')

    atom_counts = [mol.GetNumAtoms() for mol, _ in valid]
    height = _panel_height(atom_counts, panel_height, max_panel_height)
    per_mol_width = width // len(valid)
    image = Draw.MolsToGridImage(
        [mol for mol, _ in valid],
        legends=[legend for _, legend in valid],
        molsPerRow=len(valid),
        subImgSize=(per_mol_width, height),
    )
    if image.width != width:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return _annotate_image(image, 'Query vs reconstructed product', width)


def _safe_stem(smiles: str, index: int) -> str:
    digest = hashlib.md5(smiles.encode(), usedforsecurity=False).hexdigest()[:10]
    return f'pathway_{index:03d}_{digest}'


def render_react_trace(
    react_trace: str,
    output_path: str | Path,
    *,
    query_smiles: str | None = None,
    reconstructed_smiles: str | None = None,
    n_steps: int | None = None,
    exact: bool = True,
    similarity: float = 1.0,
    width: int = 900,
    panel_height: int = 360,
    max_panel_height: int = 520,
) -> Path:
    """Render one react_trace to a PNG path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = react_trace_to_steps(react_trace)
    wrap_chars = max(40, width // 8)
    images: list[Image.Image] = []

    query = query_smiles or (steps[-1].product if steps else '')
    recon = reconstructed_smiles or query
    images.append(
        _draw_overview(
            query=query,
            reconstructed=recon,
            n_steps=n_steps if n_steps is not None else len(steps),
            exact=exact,
            similarity=similarity,
            width=width,
            panel_height=panel_height,
            max_panel_height=max_panel_height,
        )
    )
    for step_idx, step in enumerate(steps, start=1):
        reaction_image = _draw_reaction_step(
            step, max_width=width, panel_height=panel_height, max_panel_height=max_panel_height
        )
        caption = [
            f'Step {step_idx}: {step.react_id} ({len(step.reactants)} reactant'
            f'{"s" if len(step.reactants) != 1 else ""})',
            *_wrap_text(' + '.join(step.reactants) + f' >> {step.product}', wrap_chars),
        ]
        images.append(_annotate_image(reaction_image, caption, width=width))

    combined = _stack_images(images, width=width)
    combined.save(output_path)
    return output_path


def _write_html_index(rows: list[dict], output_dir: Path) -> Path:
    index_path = output_dir / 'index.html'
    cards = []
    for item in rows:
        steps_html = ''
        if item.get('step_details'):
            steps_html = '<h3>Reaction steps</h3><ol>' + ''.join(
                f'<li><code>{html.escape(step)}</code></li>' for step in item['step_details']
            ) + '</ol>'
        cards.append(
            '<section>'
            f'<h2>Query</h2><p><code>{html.escape(item["query"])}</code></p>'
            f'<h3>Found={html.escape(str(item["found"]))} '
            f'exact={html.escape(str(item["exact"]))} '
            f'steps={html.escape(str(item["n_steps"]))} '
            f'time_s={html.escape(str(item.get("time_s", "")))}</h3>'
            f'{steps_html}'
            f'<img src="{html.escape(item["image"])}" alt="pathway" />'
            f'<details><summary>react_trace</summary>'
            f'<pre>{html.escape(item.get("react_trace", ""))}</pre></details>'
            f'<details><summary>pathway</summary>'
            f'<pre>{html.escape(item.get("pathway", ""))}</pre></details>'
            '</section>'
        )
    index_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        '<title>LDDM pathway render</title>'
        '<style>'
        'body{font-family:sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;}'
        'section{border:1px solid #ddd;border-radius:8px;padding:1rem;margin-bottom:1.5rem;}'
        'img{max-width:100%;height:auto;border:1px solid #eee;}'
        'code,pre{word-break:break-all;white-space:pre-wrap;}'
        '</style></head><body>'
        '<h1>LDDM retrosynthesis pathway renders</h1>'
        + '\n'.join(cards)
        + '</body></html>',
        encoding='utf-8',
    )
    return index_path


def _sanitize_inline_svg(svg: str) -> str:
    """Strip XML decls so the SVG can sit inline in HTML."""
    svg = re.sub(r"<\?xml[^>]*\?>", '', svg, count=1).strip()
    svg = re.sub(r"<!DOCTYPE[^>]*>", '', svg, count=1).strip()
    # Prefer utf-8 for HTML embedding.
    svg = svg.replace("encoding='iso-8859-1'", "encoding='utf-8'")
    svg = svg.replace('encoding="iso-8859-1"', 'encoding="utf-8"')
    return svg


def mol_to_svg(
    smiles: str,
    *,
    width: int = 220,
    height: int = 160,
    legend: str | None = None,
) -> str:
    """Return an inline-ready SVG for a molecule (empty string if unparseable)."""
    mol = _prepare_mol(smiles)
    if mol is None:
        return ''
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.padding = 0.08
    opts.clearBackground = True
    try:
        if legend:
            drawer.DrawMolecule(mol, legend=legend)
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return _sanitize_inline_svg(drawer.GetDrawingText())
    except Exception as e:
        logging.debug(f'mol_to_svg failed for {smiles}: {e}')
        return ''


def reaction_step_to_svg(
    step: RenderStep,
    *,
    width: int = 780,
    height: int = 260,
) -> str:
    """Return an inline-ready SVG reaction diagram for one step."""
    rxn = _build_reaction(step)
    if not _reaction_atom_counts(rxn):
        return ''
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.padding = 0.02
    try:
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        return _sanitize_inline_svg(drawer.GetDrawingText())
    except Exception as e:
        logging.debug(f'reaction_step_to_svg failed for {step.react_id}: {e}')
        return ''


def load_reaction_catalog(
    paths: Sequence[str | Path] | str | Path | None,
) -> Dict[str, dict]:
    """Load reaction metadata keyed by id from one or more JSON files.

    Available fields in SynSpace / ReaSyn spaces: ``id``, ``name``, ``reaction``
    (SMARTS), ``explicit_hs``, optionally ``source`` / ``molecularity`` /
    ``covers``. Experimental yields and lab conditions are **not** stored.
    """
    if paths is None:
        return {}
    if isinstance(paths, (str, Path)):
        path_list: List[Path] = [Path(paths)]
    else:
        path_list = [Path(p) for p in paths]

    catalog: Dict[str, dict] = {}
    for path in path_list:
        if not path.is_file():
            logging.warning(f'Reaction catalog not found: {path}')
            continue
        with open(path) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            logging.warning(f'Unexpected reaction JSON shape in {path}')
            continue
        for row in rows:
            rid = str(row.get('id') or row.get('name') or '')
            if not rid:
                continue
            meta = {k: row[k] for k in _REACTION_META_KEYS if k in row}
            meta.setdefault('id', rid)
            meta.setdefault('name', rid)
            catalog[rid] = meta
        logging.info(f'Loaded {len(rows)} reaction defs from {path}')
    return catalog


def _bb_badge(bb_id: str) -> tuple[str, str]:
    """Return (css_class, short_label) for a building-block id."""
    bid = str(bb_id)
    if bid.startswith('proposed__'):
        return 'bb-proposed', 'proposed'
    if bid.startswith('trivial__'):
        return 'bb-trivial', 'trivial'
    return 'bb-catalog', 'catalog'


def _split_semi(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    return [p for p in text.split(';') if p]


def _embedded_css() -> str:
    return """
:root {
  --ink: #1a2332;
  --muted: #5a6a7a;
  --line: #d5dde6;
  --bg: #f7f9fb;
  --card: #ffffff;
  --accent: #0b6e6e;
  --accent-soft: #e6f3f3;
  --warn: #8a5a00;
  --warn-bg: #fff6e0;
  --ok: #1b6b3a;
  --bad: #8b2e2e;
  --proposed: #6b4f1d;
  --catalog: #1b4f72;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font: 15px/1.45 "Source Sans 3", "Segoe UI", "Helvetica Neue", sans-serif;
}
header.app {
  position: sticky; top: 0; z-index: 5;
  background: linear-gradient(180deg, #12343a 0%, #0b6e6e 100%);
  color: #f4fbfb;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 0 rgba(0,0,0,.12);
}
header.app h1 { margin: 0 0 .25rem; font-size: 1.35rem; font-weight: 650; }
header.app p { margin: 0; opacity: .9; font-size: .92rem; }
.layout { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 88px); }
nav.toc {
  border-right: 1px solid var(--line);
  background: #eef3f5;
  padding: 1rem;
  overflow: auto;
  position: sticky; top: 88px; height: calc(100vh - 88px);
}
nav.toc h2 { margin: 0 0 .75rem; font-size: .85rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
nav.toc a {
  display: block; padding: .45rem .55rem; margin-bottom: .25rem;
  border-radius: 6px; color: var(--ink); text-decoration: none; font-size: .9rem;
  border: 1px solid transparent;
}
nav.toc a:hover, nav.toc a:focus { background: var(--accent-soft); border-color: #b7d9d9; }
nav.toc .meta { display:block; color: var(--muted); font-size: .78rem; margin-top: .1rem; }
main { padding: 1.25rem; max-width: 1100px; }
.note {
  background: var(--warn-bg); color: var(--warn);
  border: 1px solid #f0d89a; border-radius: 8px;
  padding: .75rem 1rem; margin-bottom: 1.25rem; font-size: .92rem;
}
.route {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem 1.1rem 1.25rem;
  margin-bottom: 1.5rem;
  scroll-margin-top: 100px;
}
.route h2 { margin: 0 0 .35rem; font-size: 1.15rem; }
.badges { display: flex; flex-wrap: wrap; gap: .4rem; margin: .6rem 0 1rem; }
.badge {
  display: inline-flex; align-items: center; gap: .3rem;
  border: 1px solid var(--line); border-radius: 999px;
  padding: .15rem .6rem; font-size: .78rem; background: #f3f6f8; color: var(--muted);
}
.badge.ok { background: #e8f6ee; color: var(--ok); border-color: #b9dfc6; }
.badge.bad { background: #f8eaea; color: var(--bad); border-color: #e2b6b6; }
.badge.accent { background: var(--accent-soft); color: var(--accent); border-color: #9fcfcf; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.panel {
  border: 1px solid var(--line); border-radius: 8px; padding: .75rem; background: #fbfcfd;
}
.panel h3 { margin: 0 0 .5rem; font-size: .85rem; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
.smiles { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .82rem; word-break: break-all; }
.svg-box { display:flex; justify-content:center; align-items:center; min-height: 120px; overflow:auto; }
.svg-box svg { max-width: 100%; height: auto; }
.bb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: .75rem; }
.bb-card {
  border: 1px solid var(--line); border-radius: 8px; padding: .5rem; background: #fff; text-align: center;
}
.bb-card .tag {
  display:inline-block; font-size: .7rem; border-radius: 4px; padding: .1rem .4rem; margin-bottom: .35rem;
}
.bb-catalog .tag { background: #e8f1f8; color: var(--catalog); }
.bb-proposed .tag { background: #f7efd8; color: var(--proposed); }
.bb-trivial .tag { background: #ececec; color: #555; }
.step {
  border: 1px solid var(--line); border-radius: 8px; margin-top: 1rem; overflow: hidden;
}
.step-head {
  display:flex; flex-wrap:wrap; justify-content:space-between; gap:.5rem;
  padding: .65rem .85rem; background: #f0f5f6; border-bottom: 1px solid var(--line);
}
.step-head strong { color: var(--accent); }
.step-body { padding: .75rem .85rem 1rem; }
.meta-table { width: 100%; border-collapse: collapse; font-size: .86rem; margin-top: .75rem; }
.meta-table th, .meta-table td { text-align: left; vertical-align: top; padding: .35rem .4rem; border-bottom: 1px solid #eef2f5; }
.meta-table th { width: 9rem; color: var(--muted); font-weight: 600; }
.na { color: var(--muted); font-style: italic; }
.procedure {
  margin-top: .85rem; padding: .75rem .85rem; border-radius: 8px;
  border: 1px solid #c5ddd8; background: var(--accent-soft);
}
.procedure.none { border-color: #e6d7a8; background: var(--warn-bg); }
.proc-title { font-weight: 650; color: var(--accent); margin-bottom: .35rem; }
.procedure.none .proc-title { color: var(--warn); }
.proc-refs { margin-top: .6rem; font-size: .86rem; }
.proc-refs ul { margin: .35rem 0 0; padding-left: 1.1rem; }
.proc-refs a { color: var(--accent); }
.disclaimer {
  margin: .55rem 0 0; font-size: .78rem; color: var(--muted); font-style: italic;
}
details { margin-top: .75rem; }
details summary { cursor: pointer; color: var(--accent); }
pre.raw {
  background: #111827; color: #e5e7eb; padding: .75rem; border-radius: 6px;
  overflow: auto; font-size: .75rem; white-space: pre-wrap; word-break: break-all;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .grid-2 { grid-template-columns: 1fr; }
}
"""


def _route_section_html(
    *,
    route_id: str,
    row: pd.Series,
    steps: List[RenderStep],
    catalog: Dict[str, dict],
    svg_width: int,
    svg_height: int,
    mol_size: tuple[int, int],
    procedure_cards: Optional[Sequence] = None,
) -> str:
    query = str(row.get('query_smiles') or '')
    recon = str(row.get('reconstructed_smiles') or '') or query
    found = bool(row.get('found', True))
    exact = bool(row.get('exact', False))
    n_steps = int(row.get('n_steps') or len(steps))
    sim_raw = row.get('similarity', '')
    try:
        similarity = float(sim_raw) if sim_raw not in ('', None) and not (
            isinstance(sim_raw, float) and pd.isna(sim_raw)
        ) else None
    except (TypeError, ValueError):
        similarity = None
    transforms = _split_semi(row.get('query_transforms'))
    searched = str(row.get('searched_smiles') or '') or query
    pathway = str(row.get('pathway') or '')
    react_trace = str(row.get('react_trace') or '')
    bb_ids = _split_semi(row.get('building_block_ids'))
    bb_smis = _split_semi(row.get('building_block_smiles'))
    if len(bb_smis) < len(bb_ids):
        bb_smis = bb_smis + [''] * (len(bb_ids) - len(bb_smis))

    badges = [
        f'<span class="badge {"ok" if found else "bad"}">{"found" if found else "not found"}</span>',
        f'<span class="badge {"ok" if exact else ""}">{"exact" if exact else "approximate"}</span>',
        f'<span class="badge accent">{n_steps} step{"s" if n_steps != 1 else ""}</span>',
    ]
    if similarity is not None:
        badges.append(f'<span class="badge">Tanimoto {similarity:.3f}</span>')
    if transforms:
        badges.append(
            f'<span class="badge">transforms: {html.escape(", ".join(transforms))}</span>'
        )

    query_svg = mol_to_svg(query, width=mol_size[0], height=mol_size[1], legend='Query')
    recon_svg = mol_to_svg(
        recon, width=mol_size[0], height=mol_size[1], legend='Reconstructed'
    )

    bb_cards = []
    for bb_id, bb_smi in zip(bb_ids, bb_smis):
        css, label = _bb_badge(bb_id)
        svg = mol_to_svg(bb_smi, width=150, height=110) if bb_smi else ''
        display_id = bb_id
        if bb_id.startswith('proposed__'):
            display_id = bb_id[len('proposed__'):]
        elif bb_id.startswith('trivial__'):
            display_id = bb_id[len('trivial__'):]
        structure = svg if svg else '<span class="na">no structure</span>'
        bb_cards.append(
            f'<div class="bb-card {css}">'
            f'<span class="tag">{html.escape(label)}</span>'
            f'<div class="svg-box">{structure}</div>'
            f'<div class="smiles">{html.escape(bb_smi or display_id)}</div>'
            f'<div class="smiles" style="color:var(--muted);font-size:.72rem">'
            f'{html.escape(str(bb_id))}</div>'
            '</div>'
        )

    step_blocks = []
    for i, step in enumerate(steps, start=1):
        meta = catalog.get(step.react_id, {})
        name = str(meta.get('name') or step.react_id)
        smarts = str(meta.get('reaction') or '')
        source = str(meta.get('source') or '')
        molecularity = meta.get('molecularity', len(step.reactants))
        explicit_hs = meta.get('explicit_hs', '')
        covers = meta.get('covers', '')
        svg = reaction_step_to_svg(step, width=svg_width, height=svg_height)
        rxn_drawing = svg if svg else '<span class="na">Could not draw reaction</span>'
        eq = ' + '.join(step.reactants) + f' → {step.product}'
        card = None
        if procedure_cards is not None and i - 1 < len(procedure_cards):
            card = procedure_cards[i - 1]
        yield_val = None
        conditions_val = None
        solvent_val = None
        if card is not None and getattr(card, 'match_level', 'none') != 'none':
            yield_val = card.typical_yield
            conditions_val = ' · '.join(
                p for p in (card.temperature, card.time, card.stoichiometry) if p
            ) or None
            solvent_val = ', '.join(card.solvents) if card.solvents else None
            if card.reagents_catalysts:
                extra = ', '.join(card.reagents_catalysts)
                solvent_val = f'{solvent_val}; {extra}' if solvent_val else extra
        rows_meta = [
            ('Reaction id', step.react_id),
            ('Name', name),
            ('Molecularity', molecularity if molecularity not in ('', None) else len(step.reactants)),
            ('Source', source or '—'),
            ('Explicit H', explicit_hs if explicit_hs != '' else '—'),
            ('SMARTS', smarts or '— (not in catalog)'),
            ('Yield', yield_val),
            ('Conditions', conditions_val),
            ('Solvent / T / cat.', solvent_val),
        ]
        if covers:
            rows_meta.insert(5, ('Covers', covers))
        meta_rows = []
        for label, value in rows_meta:
            if value is None:
                if label in ('Yield', 'Conditions', 'Solvent / T / cat.'):
                    if card is not None and getattr(card, 'match_level', 'none') == 'none':
                        cell = (
                            '<span class="na">no procedure card for this step — '
                            'expand data/procedures/ or template_class_map.json</span>'
                        )
                    elif card is None:
                        cell = (
                            '<span class="na">enable procedure enrichment '
                            '(--enrich-procedures / synthesize defaults)</span>'
                        )
                    else:
                        cell = '<span class="na">not listed on matched procedure card</span>'
                else:
                    cell = '<span class="na">—</span>'
            else:
                cell = f'<code class="smiles">{html.escape(str(value))}</code>'
            meta_rows.append(
                f'<tr><th>{html.escape(label)}</th><td>{cell}</td></tr>'
            )
        procedure_html = ''
        if card is not None:
            from lddm.reactions.procedure_enrichment import procedure_card_html

            procedure_html = procedure_card_html(card)
        step_blocks.append(
            f'<article class="step">'
            f'<div class="step-head"><div><strong>Step {i}</strong> — '
            f'{html.escape(name)}</div>'
            f'<div class="badge">{html.escape(str(step.react_id))}</div></div>'
            f'<div class="step-body">'
            f'<div class="smiles">{html.escape(eq)}</div>'
            f'<div class="svg-box" style="margin-top:.6rem">'
            f'{rxn_drawing}</div>'
            f'<table class="meta-table">{"".join(meta_rows)}</table>'
            f'{procedure_html}'
            f'</div></article>'
        )

    searched_note = ''
    if searched and searched != query:
        searched_note = (
            f'<p class="smiles" style="color:var(--muted)">Searched as '
            f'<code>{html.escape(searched)}</code></p>'
        )

    bb_grid = ''.join(bb_cards) if bb_cards else '<p class="na">None</p>'
    pathway_block = (
        f'<details><summary>Human pathway</summary><pre class="raw">'
        f'{html.escape(pathway)}</pre></details>'
        if pathway
        else ''
    )
    trace_block = (
        f'<details><summary>react_trace</summary><pre class="raw">'
        f'{html.escape(react_trace)}</pre></details>'
        if react_trace
        else ''
    )

    return (
        f'<section class="route" id="{html.escape(route_id)}">'
        f'<h2>Query</h2>'
        f'<p class="smiles"><code>{html.escape(query)}</code></p>'
        f'{searched_note}'
        f'<div class="badges">{"".join(badges)}</div>'
        f'<div class="grid-2">'
        f'<div class="panel"><h3>Query</h3><div class="svg-box">{query_svg}</div></div>'
        f'<div class="panel"><h3>Reconstructed</h3><div class="svg-box">{recon_svg}</div>'
        f'<p class="smiles"><code>{html.escape(recon)}</code></p></div>'
        f'</div>'
        f'<h3 style="margin:1.1rem 0 .5rem;font-size:.95rem">Building blocks</h3>'
        f'<div class="bb-grid">{bb_grid}</div>'
        f'<h3 style="margin:1.1rem 0 .25rem;font-size:.95rem">Forward steps</h3>'
        f'{"".join(step_blocks)}'
        f'{pathway_block}'
        f'{trace_block}'
        '</section>'
    )


def write_embedded_retrosynthesis_html(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    reaction_paths: Sequence[str | Path] | str | Path | None = None,
    reaction_catalog: Dict[str, dict] | None = None,
    top_k: int = 3,
    only_found: bool = True,
    max_molecules: int | None = None,
    svg_width: int = 780,
    svg_height: int = 260,
    mol_width: int = 280,
    mol_height: int = 180,
    title: str = 'LDDM retrosynthesis report',
    enrich_procedures: bool = False,
    procedure_matcher=None,
) -> Path:
    """Write a single self-contained HTML file with inline SVG route drawings.

    Parameters
    ----------
    df
        synthesize.py / benchmark CSV columns (needs ``query_smiles``,
        ``found``, ``react_trace`` at minimum).
    output_path
        Destination ``.html`` path (parent dirs created as needed).
    reaction_paths
        Optional reaction JSON file(s) to attach SMARTS / name / source /
        molecularity to each step. Yields and lab conditions are not present
        in these files unless ``enrich_procedures`` is enabled (curated KB +
        local OpenAlex cache; offline).
    enrich_procedures
        When True, attach curated procedure cards and local OpenAlex refs.
    procedure_matcher
        Optional ``ProcedureMatcher`` instance; built from defaults if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = {'query_smiles', 'found', 'react_trace'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'CSV missing columns: {sorted(missing)}')

    catalog = dict(reaction_catalog or {})
    if reaction_paths is not None:
        catalog.update(load_reaction_catalog(reaction_paths))

    matcher = procedure_matcher
    if enrich_procedures and matcher is None:
        from lddm.reactions.procedure_enrichment import ProcedureMatcher

        matcher = ProcedureMatcher()

    work = df.copy()
    if only_found:
        work = work[work['found'].astype(bool)]
    work = work[work['react_trace'].astype(str).str.len() > 0]
    sort_cols = [c for c in ('query_smiles', 'exact', 'similarity', 'n_steps') if c in work.columns]
    if sort_cols:
        ascending = []
        for c in sort_cols:
            if c in ('exact', 'similarity'):
                ascending.append(False)
            elif c == 'n_steps':
                ascending.append(True)
            else:
                ascending.append(True)
        work = work.sort_values(sort_cols, ascending=ascending)
    work = work.groupby('query_smiles', as_index=False).head(top_k)
    if max_molecules is not None:
        keep = list(dict.fromkeys(work['query_smiles'].tolist()))[:max_molecules]
        work = work[work['query_smiles'].isin(keep)]

    toc_items: list[str] = []
    sections: list[str] = []
    for i, (_, row) in enumerate(work.iterrows()):
        query = str(row['query_smiles'])
        trace = str(row['react_trace'])
        try:
            steps = react_trace_to_steps(trace)
        except Exception as e:
            logging.warning(f'Failed to parse react_trace for {query}: {e}')
            continue
        route_id = f'route-{i:04d}'
        n_steps = int(row.get('n_steps') or len(steps))
        exact = bool(row.get('exact', False))
        toc_items.append(
            f'<a href="#{html.escape(route_id)}">'
            f'<span class="smiles">{html.escape(query[:42])}{"…" if len(query) > 42 else ""}</span>'
            f'<span class="meta">{n_steps} steps · {"exact" if exact else "approx"}</span>'
            f'</a>'
        )
        cards = None
        if matcher is not None:
            cards = [e.procedure for e in matcher.enrich_steps(steps)]
        sections.append(
            _route_section_html(
                route_id=route_id,
                row=row,
                steps=steps,
                catalog=catalog,
                svg_width=svg_width,
                svg_height=svg_height,
                mol_size=(mol_width, mol_height),
                procedure_cards=cards,
            )
        )

    n_routes = len(sections)
    if enrich_procedures or matcher is not None:
        note = (
            '<div class="note"><strong>Procedure enrichment (offline):</strong> '
            'curated named-reaction cards plus local OpenAlex literature refs '
            '(<code>data/openalex/prepared/</code>). Cards are textbook-typical '
            'ranges — <strong>not molecule-specific SOPs</strong>. SMARTS templates '
            'remain connectivity-only. Refresh refs with '
            '<code>scripts/download_openalex_dataset.py</code> + '
            '<code>prepare_openalex_dataset.py</code>. See '
            '<a href="https://developers.openalex.org/">OpenAlex</a> / '
            '<a href="https://github.com/MolecularAI/ReaSyn">ReaSyn</a>.</div>'
        )
    else:
        note = (
            '<div class="note"><strong>Template metadata:</strong> SynSpace / ReaSyn '
            'reaction JSON entries expose <em>id, name, SMARTS, explicit_hs</em> '
            '(plus optional <em>source / molecularity</em>). '
            '<strong>Experimental yields, solvents, temperatures, catalysts, and '
            'literature references are not available</strong> in the current chemical '
            'spaces — treat each transform as a connectivity template, not a lab SOP. '
            'See '
            '<a href="https://github.com/MolecularAI/ReaSyn">ReaSyn</a> / '
            '<a href="https://github.com/wenhao-gao/synformer/tree/main/data/rxn_templates">'
            'SynFormer templates</a>.</div>'
        )
    toc_html = ''.join(toc_items) if toc_items else '<p class="na">No routes</p>'
    main_html = ''.join(sections) if sections else '<p class="na">No pathways to show.</p>'
    routes_label = 'routes' if n_routes != 1 else 'route'

    doc = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{html.escape(title)}</title>'
        f'<style>{_embedded_css()}</style></head><body>'
        f'<header class="app"><h1>{html.escape(title)}</h1>'
        f'<p>{n_routes} {routes_label} · all drawings embedded '
        '(open offline, no external assets)</p></header>'
        '<div class="layout">'
        f'<nav class="toc"><h2>Routes</h2>{toc_html}</nav>'
        f'<main>{note}{main_html}</main>'
        '</div></body></html>'
    )
    output_path.write_text(doc, encoding='utf-8')
    logging.info(f'Wrote embedded retrosynthesis HTML ({n_routes} routes) to {output_path}')
    return output_path


def render_routes_dataframe(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    top_k: int = 1,
    only_found: bool = True,
    width: int = 900,
    panel_height: int = 360,
    max_panel_height: int = 520,
    max_molecules: int | None = None,
) -> Path:
    """Render synthesize / benchmark CSV rows to PNG + HTML index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {'query_smiles', 'found', 'react_trace'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'CSV missing columns: {sorted(missing)}')

    work = df.copy()
    if only_found:
        work = work[work['found'].astype(bool)]
    work = work[work['react_trace'].astype(str).str.len() > 0]
    if 'similarity' in work.columns:
        work = work.sort_values(
            ['query_smiles', 'exact', 'similarity', 'n_steps'],
            ascending=[True, False, False, True],
        )
    else:
        work = work.sort_values(['query_smiles', 'n_steps'], ascending=[True, True])
    work = work.groupby('query_smiles', as_index=False).head(top_k)
    if max_molecules is not None:
        keep = list(dict.fromkeys(work['query_smiles'].tolist()))[:max_molecules]
        work = work[work['query_smiles'].isin(keep)]

    html_rows: list[dict] = []
    for i, (_, row) in enumerate(work.iterrows()):
        query = str(row['query_smiles'])
        trace = str(row['react_trace'])
        image_path = output_dir / f'{_safe_stem(query, i)}.png'
        try:
            render_react_trace(
                trace,
                image_path,
                query_smiles=query,
                reconstructed_smiles=str(row.get('reconstructed_smiles') or '') or None,
                n_steps=int(row.get('n_steps') or 0),
                exact=bool(row.get('exact', True)),
                similarity=float(row['similarity']) if row.get('similarity') not in ('', None) else 1.0,
                width=width,
                panel_height=panel_height,
                max_panel_height=max_panel_height,
            )
        except Exception as e:
            logging.warning(f'Failed to render {query}: {e}')
            continue
        steps = react_trace_to_steps(trace)
        html_rows.append(
            {
                'query': query,
                'found': row.get('found', True),
                'exact': row.get('exact', True),
                'n_steps': row.get('n_steps', len(steps)),
                'time_s': row.get('time_s', ''),
                'image': image_path.name,
                'react_trace': trace,
                'pathway': str(row.get('pathway') or ''),
                'step_details': [
                    f'{s.react_id}: {" + ".join(s.reactants)} >> {s.product}' for s in steps
                ],
            }
        )
        logging.info(f'Wrote {image_path}')

    return _write_html_index(html_rows, output_dir)
