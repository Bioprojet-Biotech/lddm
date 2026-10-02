"""Offline procedure enrichment for retrosynthesis route steps.

Matches reaction ids / aliases / class tags to curated ProcedureCards and
attaches literature references from a prepared OpenAlex cache. Never calls
the network at runtime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from lddm.reactions.openalex_dataset import (
    DEFAULT_PREPARED_DIR,
    load_prepared_refs,
    normalize_reaction_key,
)
from lddm.reactions.render_pathways import RenderStep, react_trace_to_steps

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCEDURE_KB = REPO_ROOT / 'data' / 'procedures' / 'named_reaction_procedures.json'
DEFAULT_CLASS_MAP = REPO_ROOT / 'data' / 'procedures' / 'template_class_map.json'


def _strip_retro_suffix(react_id: str) -> str:
    rid = str(react_id or '')
    if rid.endswith('__retro'):
        return rid[: -len('__retro')]
    return rid


@dataclass
class Reference:
    doi: str = ''
    title: str = ''
    url: str = ''
    source: str = 'openalex'
    year: int | None = None
    cited_by_count: int = 0
    openalex_id: str = ''

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Reference':
        year = d.get('year')
        try:
            year_i = int(year) if year is not None and year != '' else None
        except (TypeError, ValueError):
            year_i = None
        return cls(
            doi=str(d.get('doi') or ''),
            title=str(d.get('title') or ''),
            url=str(d.get('url') or ''),
            source=str(d.get('source') or 'openalex'),
            year=year_i,
            cited_by_count=int(d.get('cited_by_count') or 0),
            openalex_id=str(d.get('openalex_id') or ''),
        )


@dataclass
class ProcedureCard:
    reaction_id: str
    match_level: str  # exact | alias | class | none
    title: str = ''
    solvents: list[str] = field(default_factory=list)
    temperature: str | None = None
    time: str | None = None
    stoichiometry: str | None = None
    reagents_catalysts: list[str] = field(default_factory=list)
    workup: str | None = None
    typical_yield: str | None = None
    notes: str | None = None
    confidence: str = 'none'  # high | medium | low | none
    references: list[Reference] = field(default_factory=list)
    provenance: str = 'none'  # curated | openalex_prepared | none
    gap_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class EnrichedStep:
    step_index: int
    react_id: str
    reactants: tuple[str, ...]
    product: str
    procedure: ProcedureCard

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'react_id': self.react_id,
            'reactants': list(self.reactants),
            'product': self.product,
            'procedure': self.procedure.to_dict(),
        }


def _card_from_kb_row(row: dict[str, Any], *, match_level: str, react_id: str) -> ProcedureCard:
    return ProcedureCard(
        reaction_id=react_id,
        match_level=match_level,
        title=str(row.get('title') or row.get('reaction_id') or react_id),
        solvents=list(row.get('solvents') or []),
        temperature=row.get('temperature'),
        time=row.get('time'),
        stoichiometry=row.get('stoichiometry'),
        reagents_catalysts=list(row.get('reagents_catalysts') or []),
        workup=row.get('workup'),
        typical_yield=row.get('typical_yield'),
        notes=row.get('notes'),
        confidence=str(row.get('confidence') or 'low'),
        provenance='curated',
    )


def _empty_card(react_id: str, reason: str) -> ProcedureCard:
    return ProcedureCard(
        reaction_id=react_id,
        match_level='none',
        title='',
        confidence='none',
        provenance='none',
        gap_reason=reason,
        notes='No curated procedure card matched this template. '
        'Treat the SMARTS transform as connectivity-only — verify any lab plan independently.',
    )


class ProcedureMatcher:
    """Match route step reaction ids to curated cards + prepared OpenAlex refs."""

    def __init__(
        self,
        kb_path: str | Path | None = None,
        refs_cache_path: str | Path | None = None,
        prepared_dir: str | Path | None = None,
        class_map_path: str | Path | None = None,
        allow_network: bool = False,
        max_refs: int = 5,
    ):
        if allow_network:
            logging.warning(
                'allow_network=True is ignored: enrichment is offline-only. '
                'Refresh refs with scripts/download_openalex_dataset.py + prepare.'
            )
        self.max_refs = max_refs
        self.kb_path = Path(kb_path or DEFAULT_PROCEDURE_KB)
        self.cards: list[dict[str, Any]] = []
        if self.kb_path.is_file():
            self.cards = json.loads(self.kb_path.read_text(encoding='utf-8'))
        else:
            logging.warning(f'Procedure KB not found: {self.kb_path}')

        if refs_cache_path is not None:
            refs_path = Path(refs_cache_path)
            if refs_path.is_file():
                self.refs_by_key = json.loads(refs_path.read_text(encoding='utf-8'))
            else:
                logging.warning(f'OpenAlex refs cache not found: {refs_path}')
                self.refs_by_key = {}
        else:
            self.refs_by_key = load_prepared_refs(prepared_dir or DEFAULT_PREPARED_DIR)

        map_path = Path(class_map_path or DEFAULT_CLASS_MAP)
        self.class_map: dict[str, str] = {}
        if map_path.is_file():
            raw = json.loads(map_path.read_text(encoding='utf-8'))
            if isinstance(raw, dict):
                self.class_map = {str(k): str(v) for k, v in raw.items()}
        else:
            logging.warning(f'Template class map not found: {map_path}')

        self._exact: dict[str, dict[str, Any]] = {}
        self._alias: dict[str, dict[str, Any]] = {}
        self._class: dict[str, dict[str, Any]] = {}
        for row in self.cards:
            rid = str(row.get('reaction_id') or '')
            if rid.startswith('class:'):
                class_name = rid.split(':', 1)[1]
                self._class[normalize_reaction_key(class_name)] = row
                for tag in row.get('class_tags') or []:
                    self._class[normalize_reaction_key(tag)] = row
                for alias in row.get('aliases') or []:
                    self._alias[normalize_reaction_key(alias)] = row
                continue
            if rid:
                self._exact[normalize_reaction_key(rid)] = row
            for alias in row.get('aliases') or []:
                self._alias[normalize_reaction_key(alias)] = row
            for tag in row.get('class_tags') or []:
                # Named cards also register as class fallbacks (lower priority than class: rows
                # only if not already set — class: rows win when registered first above).
                nk = normalize_reaction_key(tag)
                self._class.setdefault(nk, row)

    def _refs_for_row(self, row: dict[str, Any]) -> list[Reference]:
        refs: list[Reference] = []
        seen: set[str] = set()
        keys = list(row.get('openalex_keys') or [])
        keys.append(normalize_reaction_key(row.get('reaction_id') or ''))
        for key in keys:
            if not key or key.startswith('class_'):
                continue
            for rec in self.refs_by_key.get(key) or []:
                ref = Reference.from_dict(rec)
                dedupe = ref.doi or ref.openalex_id or ref.url or ref.title
                if not dedupe or dedupe in seen:
                    continue
                seen.add(dedupe)
                refs.append(ref)
                if len(refs) >= self.max_refs:
                    return refs
        return refs

    def _lookup_class(self, class_name: str) -> dict[str, Any] | None:
        return self._class.get(normalize_reaction_key(class_name))

    def match(self, react_id: str, *, class_hints: Sequence[str] | None = None) -> ProcedureCard:
        base_id = _strip_retro_suffix(react_id)
        nk = normalize_reaction_key(base_id)
        row = self._exact.get(nk)
        level = 'exact'
        if row is None:
            row = self._alias.get(nk)
            level = 'alias'
        if row is None:
            # template_class_map: reasyn_006 → n_alkylation → class card
            mapped = self.class_map.get(base_id) or self.class_map.get(str(react_id))
            if mapped:
                row = self._lookup_class(mapped) or self._alias.get(normalize_reaction_key(mapped))
                if row is not None:
                    level = 'class'
        if row is None:
            for alias_key, candidate in self._alias.items():
                if alias_key and (alias_key in nk or nk in alias_key):
                    row = candidate
                    level = 'alias'
                    break
        if row is None:
            hints = [normalize_reaction_key(h) for h in (class_hints or [])]
            hints.extend(nk.split('_'))
            for h in hints:
                if h and h in self._class:
                    row = self._class[h]
                    level = 'class'
                    break
        if row is None:
            # Last resort: opaque ReaSyn → generic_coupling class card
            if nk.startswith('reasyn_') or base_id.startswith('reasyn_'):
                row = self._lookup_class('generic_coupling')
                if row is not None:
                    level = 'class'
        if row is None:
            return _empty_card(
                react_id,
                reason='no_exact_alias_or_class_match',
            )

        card = _card_from_kb_row(row, match_level=level, react_id=react_id)
        refs = self._refs_for_row(row)
        if refs:
            card.references = refs
            if card.provenance == 'curated':
                card.provenance = 'curated+openalex_prepared'
        return card

    def enrich_steps(self, steps: Iterable[RenderStep]) -> list[EnrichedStep]:
        out: list[EnrichedStep] = []
        for i, step in enumerate(steps, start=1):
            card = self.match(step.react_id)
            out.append(
                EnrichedStep(
                    step_index=i,
                    react_id=step.react_id,
                    reactants=step.reactants,
                    product=step.product,
                    procedure=card,
                )
            )
        return out


def enrich_route(
    react_trace: str,
    *,
    kb_path: str | Path | None = None,
    refs_cache_path: str | Path | None = None,
    prepared_dir: str | Path | None = None,
    allow_network: bool = False,
    matcher: ProcedureMatcher | None = None,
) -> list[EnrichedStep]:
    """Enrich a react_trace with procedure cards (offline)."""
    m = matcher or ProcedureMatcher(
        kb_path=kb_path,
        refs_cache_path=refs_cache_path,
        prepared_dir=prepared_dir,
        allow_network=allow_network,
    )
    steps = react_trace_to_steps(react_trace)
    return m.enrich_steps(steps)


def procedures_available(
    *,
    kb_path: str | Path | None = None,
    prepared_dir: str | Path | None = None,
) -> bool:
    """True when curated KB exists (OpenAlex prepared cache optional)."""
    kb = Path(kb_path or DEFAULT_PROCEDURE_KB)
    return kb.is_file()


def attach_procedures_to_dataframe(
    df,
    *,
    matcher: ProcedureMatcher | None = None,
    kb_path: str | Path | None = None,
    prepared_dir: str | Path | None = None,
):
    """Add offline procedure columns to a synthesize / benchmark routes DataFrame.

    New columns:
    - ``procedures``: JSON list of per-step procedure card dicts
    - ``procedure_match_levels``: semicolon-separated match levels
    - ``procedure_titles``: semicolon-separated card titles (or ``none``)
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return df
    work = df.copy()
    m = matcher or ProcedureMatcher(kb_path=kb_path, prepared_dir=prepared_dir)

    procedures_col: list[str] = []
    levels_col: list[str] = []
    titles_col: list[str] = []

    for _, row in work.iterrows():
        trace = str(row.get('react_trace') or '')
        if not trace or trace.lower() == 'nan':
            procedures_col.append('[]')
            levels_col.append('')
            titles_col.append('')
            continue
        try:
            enriched = enrich_route(trace, matcher=m)
        except Exception as e:
            logging.warning(f'Procedure enrich failed: {e}')
            procedures_col.append('[]')
            levels_col.append('')
            titles_col.append('')
            continue
        procedures_col.append(json.dumps([s.to_dict() for s in enriched], ensure_ascii=False))
        levels_col.append(';'.join(s.procedure.match_level for s in enriched))
        titles_col.append(
            ';'.join(
                (s.procedure.title or 'none').replace(';', ',') for s in enriched
            )
        )

    work['procedures'] = procedures_col
    work['procedure_match_levels'] = levels_col
    work['procedure_titles'] = titles_col
    return work


def procedure_card_html(card: ProcedureCard) -> str:
    """HTML fragment for a procedure card (used by embedded report)."""
    import html as html_mod

    if card.match_level == 'none':
        reason = html_mod.escape(card.gap_reason or 'not available')
        note = html_mod.escape(card.notes or '')
        return (
            f'<div class="procedure none">'
            f'<div class="proc-title">Procedure: not available</div>'
            f'<p class="na">{reason}. {note}</p>'
            f'<p class="disclaimer">Verify independently before any lab use.</p>'
            f'</div>'
        )

    def row(label: str, value: str | None) -> str:
        if value is None or value == '':
            cell = '<span class="na">not available</span>'
        else:
            cell = html_mod.escape(str(value))
        return f'<tr><th>{html_mod.escape(label)}</th><td>{cell}</td></tr>'

    solvents = ', '.join(card.solvents) if card.solvents else None
    reagents = ', '.join(card.reagents_catalysts) if card.reagents_catalysts else None
    refs_html = ''
    if card.references:
        items = []
        for ref in card.references:
            title = html_mod.escape(ref.title or ref.doi or 'reference')
            year = f' ({ref.year})' if ref.year else ''
            if ref.url:
                items.append(
                    f'<li><a href="{html_mod.escape(ref.url)}" rel="noopener noreferrer">'
                    f'{title}</a>{year}</li>'
                )
            else:
                items.append(f'<li>{title}{year}</li>')
        refs_html = (
            '<div class="proc-refs"><strong>Literature (local OpenAlex cache)</strong>'
            f'<ul>{"".join(items)}</ul></div>'
        )
    else:
        refs_html = (
            '<p class="na">No local OpenAlex refs for this card '
            '(run download + prepare to refresh).</p>'
        )

    rows = [
        row('Match', f'{card.match_level} · confidence={card.confidence} · {card.provenance}'),
        row('Solvents', solvents),
        row('Temperature', card.temperature),
        row('Time', card.time),
        row('Stoichiometry', card.stoichiometry),
        row('Reagents / catalysts', reagents),
        row('Workup', card.workup),
        row('Typical yield', card.typical_yield),
        row('Notes', card.notes),
    ]
    return (
        f'<div class="procedure">'
        f'<div class="proc-title">{html_mod.escape(card.title)}</div>'
        f'<table class="meta-table">{"".join(rows)}</table>'
        f'{refs_html}'
        f'<p class="disclaimer">Textbook-typical ranges only — not molecule-specific. '
        f'Verify before use.</p>'
        f'</div>'
    )
