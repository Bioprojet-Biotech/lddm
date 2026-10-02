"""Multi-source building-block catalogs with provider priority and origin tracking.

Deduplicates by canonical SMILES. When the same molecule appears in several
catalogs, one **primary** provider/id is chosen (best priority) while all
origins are retained in ``sources`` (``provider:id`` joined by ``;``).

Default priority (lower = preferred primary)::

    synspace < enamine < molport < mcule < reasyn < zinc < trivial < other

Refs:
https://mcule.com/database/
https://enamine.net/building-blocks/building-blocks-catalog
https://www.molport.com/shop/libraries
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from rdkit import Chem

# Lower rank = preferred as the single primary source when many match.
PROVIDER_PRIORITY: dict[str, int] = {
    'synspace': 10,
    'enamine': 20,
    'molport': 30,
    'mcule': 40,
    'reasyn': 50,
    'zinc': 55,
    'trivial': 90,
    'other': 100,
}

# Mcule S3 filenames change; override with --mcule-url if needed.
# https://mcule.com/database/
MCULE_BB_SMI_URL = (
    'https://mcule.s3.amazonaws.com/database/'
    'mcule_purchasable_building_blocks_260802.smi.gz'
)


def provider_rank(provider: str) -> int:
    key = (provider or 'other').strip().lower()
    return PROVIDER_PRIORITY.get(key, PROVIDER_PRIORITY['other'])


def canonicalize_bb_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def infer_provider(block_id: str, explicit: str | None = None) -> str:
    if explicit:
        return str(explicit).strip().lower()
    bid = str(block_id)
    lower = bid.lower()
    if lower.startswith('synspace'):
        return 'synspace'
    if lower.startswith('reasyn') or lower.startswith('zinc'):
        return 'reasyn'
    if lower.startswith('mcule') or bid.upper().startswith('MCULE'):
        return 'mcule'
    if lower.startswith('enamine') or lower.startswith('en300'):
        return 'enamine'
    if lower.startswith('molport'):
        return 'molport'
    if lower.startswith('trivial'):
        return 'trivial'
    if lower.startswith('proposed'):
        return 'proposed'
    return 'other'


def _origin_token(provider: str, block_id: str) -> str:
    return f'{provider}:{block_id}'


def _parse_sources(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [p for p in text.split(';') if p]


def dataframe_from_records(records: Iterable[dict]) -> pd.DataFrame:
    rows = list(records)
    if not rows:
        return pd.DataFrame(
            columns=['id', 'smiles', 'provider', 'sources', 'n_sources', 'priority']
        )
    return pd.DataFrame(rows)


def load_bb_table(path: Path, *, default_provider: str | None = None) -> pd.DataFrame:
    """Load id/smiles CSV (optional provider/sources columns)."""
    path = Path(path)
    frame = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in frame.columns}
    if 'id' not in cols or 'smiles' not in cols:
        raise ValueError(f'{path} must contain id and smiles columns')
    out = pd.DataFrame(
        {
            'id': frame[cols['id']].astype(str),
            'smiles': frame[cols['smiles']].astype(str),
        }
    )
    if 'provider' in cols:
        out['provider'] = frame[cols['provider']].astype(str)
    else:
        out['provider'] = [
            infer_provider(i, default_provider) for i in out['id']
        ]
    if 'sources' in cols:
        out['sources'] = frame[cols['sources']].fillna('').astype(str)
    else:
        out['sources'] = [
            _origin_token(p, i) for p, i in zip(out['provider'], out['id'])
        ]
    if 'priority' in cols:
        out['priority'] = pd.to_numeric(frame[cols['priority']], errors='coerce')
    else:
        out['priority'] = [provider_rank(p) for p in out['provider']]
    if 'n_sources' in cols:
        out['n_sources'] = pd.to_numeric(frame[cols['n_sources']], errors='coerce').fillna(1).astype(int)
    else:
        out['n_sources'] = out['sources'].map(lambda s: max(1, len(_parse_sources(s))))
    return out


def load_smiles_catalog(
    path: Path,
    *,
    provider: str,
    id_prefix: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load SMILES text / smi.gz (optional ``smiles[\\t ]id`` per line)."""
    path = Path(path)
    open_fn = path.open
    if path.suffix == '.gz' or str(path).endswith('.smi.gz'):
        import gzip

        open_fn = lambda: gzip.open(path, 'rt')  # noqa: E731

    prefix = id_prefix or provider
    records: list[dict] = []
    with open_fn() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.lower() in {'smiles', 'smi', 'canonical_smiles', 'smiles id'}:
                continue
            parts = line.replace(',', '\t').split()
            if not parts:
                continue
            smi = parts[0]
            if len(parts) >= 2 and not parts[1].lower() in {'smiles'}:
                raw_id = parts[1]
                # Mcule IDs look like MCULE-1234567890
                if raw_id.upper().startswith('MCULE'):
                    block_id = raw_id
                elif provider == 'mcule':
                    block_id = f'{prefix}_{raw_id}'
                else:
                    block_id = raw_id if raw_id.startswith(prefix) else f'{prefix}_{raw_id}'
            else:
                block_id = f'{prefix}_{len(records):07d}'
            prov = infer_provider(block_id, provider)
            records.append(
                {
                    'id': block_id,
                    'smiles': smi,
                    'provider': prov,
                    'sources': _origin_token(prov, block_id),
                    'n_sources': 1,
                    'priority': provider_rank(prov),
                }
            )
            if max_rows is not None and len(records) >= max_rows:
                break
    if not records:
        raise ValueError(f'No building blocks found in {path}')
    return pd.DataFrame(records)


def merge_bb_catalogs(
    tables: Sequence[pd.DataFrame],
    *,
    skip_invalid: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Dedup by canonical SMILES; keep best-priority primary + all origins.

    First occurrence of a better (lower) priority wins the primary ``id`` /
    ``provider``. Later hits only append to ``sources`` unless they beat the
    current priority (then they become primary and previous primary moves into
    sources).
    """
    # smiles -> mutable record
    by_smi: dict[str, dict] = {}
    skipped_invalid = 0
    input_rows = 0
    collisions = 0

    for table in tables:
        if table is None or len(table) == 0:
            continue
        for row in table.itertuples(index=False):
            input_rows += 1
            canon = canonicalize_bb_smiles(str(row.smiles))
            if canon is None:
                if skip_invalid:
                    skipped_invalid += 1
                    continue
                raise ValueError(f'Invalid BB {row.id}: {row.smiles}')
            provider = infer_provider(str(row.id), getattr(row, 'provider', None))
            block_id = str(row.id)
            origin = _origin_token(provider, block_id)
            rank = int(getattr(row, 'priority', provider_rank(provider)))
            existing_sources = _parse_sources(getattr(row, 'sources', '') or '')
            origins = list(dict.fromkeys([origin, *existing_sources]))

            if canon not in by_smi:
                by_smi[canon] = {
                    'id': block_id,
                    'smiles': canon,
                    'provider': provider,
                    'sources': ';'.join(origins),
                    'n_sources': len(origins),
                    'priority': rank,
                }
                continue

            collisions += 1
            cur = by_smi[canon]
            cur_sources = _parse_sources(cur['sources'])
            for o in origins:
                if o not in cur_sources:
                    cur_sources.append(o)
            if rank < int(cur['priority']):
                # New primary — demote previous primary into sources (already there).
                cur['id'] = block_id
                cur['provider'] = provider
                cur['priority'] = rank
            cur['sources'] = ';'.join(cur_sources)
            cur['n_sources'] = len(cur_sources)

    # Ensure unique primary IDs (rare cross-provider id clash after priority swap).
    used_ids: set[str] = set()
    renamed = 0
    ordered: list[dict] = []
    for smi in sorted(by_smi.keys(), key=lambda s: (by_smi[s]['priority'], s)):
        rec = dict(by_smi[smi])
        bid = rec['id']
        if bid in used_ids:
            base = bid
            n = 1
            while f'{base}__{n}' in used_ids:
                n += 1
            bid = f'{base}__{n}'
            rec['id'] = bid
            # Update primary token in sources.
            prov = rec['provider']
            tokens = _parse_sources(rec['sources'])
            primary = _origin_token(prov, base)
            tokens = [_origin_token(prov, bid) if t == primary else t for t in tokens]
            if _origin_token(prov, bid) not in tokens:
                tokens.insert(0, _origin_token(prov, bid))
            rec['sources'] = ';'.join(dict.fromkeys(tokens))
            rec['n_sources'] = len(_parse_sources(rec['sources']))
            renamed += 1
        used_ids.add(bid)
        ordered.append(rec)

    stats = {
        'input_rows': input_rows,
        'union': len(ordered),
        'skipped_invalid': skipped_invalid,
        'smiles_collisions': collisions,
        'renamed_id_collisions': renamed,
        'by_provider': (
            pd.Series([r['provider'] for r in ordered]).value_counts().to_dict()
            if ordered
            else {}
        ),
        'multi_source': sum(1 for r in ordered if r['n_sources'] > 1),
        'single_source': sum(1 for r in ordered if r['n_sources'] == 1),
    }
    return pd.DataFrame(ordered), stats


def download_url(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` if missing (or ``force``)."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        logging.info(f'Using cached download {dest}')
        return dest
    logging.info(f'Downloading {url} → {dest}')
    urllib.request.urlretrieve(url, dest)
    return dest


def provider_priority_map(blocks: pd.DataFrame) -> dict[str, int]:
    """Map canonical SMILES → priority rank (for retrosynthesis ranking)."""
    out: dict[str, int] = {}
    for row in blocks.itertuples(index=False):
        out[str(row.smiles)] = int(row.priority)
    return out
