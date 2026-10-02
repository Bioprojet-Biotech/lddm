"""Benchmark LDDM retrosynthesis reconstruction quality (SynFormer / ReaSyn sets).

Primary question (same spirit as ReaSyn ``eval_recon.py``):
  Do we recover a pathway that **forward-replays** to the query molecule?

Metrics (best route per query):
  - success_rate              — any pathway returned (``found``)
  - reconstruction_rate       — forward-valid react_trace AND product == searched SMILES
  - reconstruction_rate_vs_query — product == original query (fails if charge-normalized)
  - forward_valid_rate        — every reaction step fires under recorded SMARTS
  - catalog_route_rate        — reconstruction with only catalog BBs (no ``proposed__``)

Speed is reported as a secondary wall-time / throughput summary.

Test sets:
- ``zinc`` / ``enamine`` / ``chembl`` / ``reasyn`` (see data/benchmarks/README.md)

Refs:
https://github.com/NVIDIA-Digital-Bio/reasyn
https://github.com/wenhao-gao/synformer
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
import urllib.request
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import yaml
from rdkit import RDLogger

from lddm.reactions.render_pathways import render_routes_dataframe
from lddm.reactions.retrosynthesis import LocalRetrosynthesizer
from lddm.reactions.route_eval import evaluate_with_synthesizer
from lddm.utils import merge_args_and_yaml, set_default, setup_logging

RDLogger.DisableLog('rdApp.*')

SYNFORMER_RAW = 'https://raw.githubusercontent.com/wenhao-gao/synformer/main/data'
TEST_URLS = {
    'enamine': f'{SYNFORMER_RAW}/enamine_smiles_1k.txt',
    'chembl': f'{SYNFORMER_RAW}/chembl_filtered_1k.txt',
}

DEFAULT_REASYN_ROOT = Path('/home/nicolaswd/repos/ReaSyn')


def _read_smiles_file(path: Path, limit: int | None = None) -> List[str]:
    smiles: List[str] = []
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        for col in ('smiles', 'target', 'SMILES', 'query_smiles'):
            if col in df.columns:
                smiles = df[col].astype(str).tolist()
                break
        else:
            smiles = df.iloc[:, 0].astype(str).tolist()
    else:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or line.upper() == 'SMILES':
                continue
            smiles.append(line.split()[0].split(',')[0])
    seen = set()
    out: List[str] = []
    for smi in smiles:
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
        if limit is not None and len(out) >= limit:
            break
    return out


def _ensure_testset(name: str, cache_dir: Path, reasyn_root: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if name == 'zinc':
        local = reasyn_root / 'data' / 'test_zinc250k.txt'
        if local.is_file():
            return local
        raise FileNotFoundError(
            f'ZINC test set not found at {local}. Clone ReaSyn or pass --input.'
        )
    if name == 'reasyn':
        for candidate in (
            reasyn_root / 'input_smiles.txt',
            reasyn_root / 'test_smiles.csv',
            reasyn_root / 'data' / 'test_zinc250k.txt',
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f'No ReaSyn test file under {reasyn_root}')
    if name in TEST_URLS:
        dest = cache_dir / f'{name}_smiles_1k.txt'
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        url = TEST_URLS[name]
        logging.info(f'Downloading SynFormer {name} test set from {url}')
        urllib.request.urlretrieve(url, dest)
        return dest
    raise ValueError(f'Unknown testset {name!r}; use zinc|enamine|chembl|reasyn or --input')


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summarize_times(times: Sequence[float]) -> dict:
    if not times:
        return {
            'n': 0, 'mean_s': 0.0, 'median_s': 0.0, 'p95_s': 0.0,
            'min_s': 0.0, 'max_s': 0.0, 'std_s': 0.0,
        }
    return {
        'n': len(times),
        'mean_s': float(statistics.mean(times)),
        'median_s': float(statistics.median(times)),
        'p95_s': float(_percentile(times, 0.95)),
        'min_s': float(min(times)),
        'max_s': float(max(times)),
        'std_s': float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def run_search(
    synthesizer: LocalRetrosynthesizer,
    smiles_list: Sequence[str],
    n_workers: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Run retrosynthesis; return raw route rows + wall-clock speed stats."""
    rows: list[dict] = []
    per_mol_times: List[float] = []

    if n_workers <= 1 or len(smiles_list) == 1:
        wall0 = time.perf_counter()
        for smi in smiles_list:
            t0 = time.perf_counter()
            routes = synthesizer.synthesize(smi)
            dt = time.perf_counter() - t0
            per_mol_times.append(dt)
            for route in routes:
                d = route.to_dict()
                d['time_s'] = round(dt, 4)
                rows.append(d)
        wall = time.perf_counter() - wall0
        timed = True
    else:
        wall0 = time.perf_counter()
        routes = synthesizer.synthesize_many(smiles_list, n_workers=n_workers)
        wall = time.perf_counter() - wall0
        share = wall / max(len(smiles_list), 1)
        per_mol_times = [share] * len(smiles_list)
        for route in routes:
            d = route.to_dict()
            d['time_s'] = round(share, 4)
            rows.append(d)
        timed = False

    df = pd.DataFrame(rows)
    speed = {
        'n_workers': n_workers,
        'wall_time_s': wall,
        'throughput_mol_per_s': len(smiles_list) / wall if wall > 0 else 0.0,
        'per_molecule': _summarize_times(per_mol_times),
        'per_molecule_timed': timed,
        'n_reactions': len(synthesizer.reactions),
        'n_building_blocks': len(synthesizer.bb_smi_to_id),
    }
    return df, speed


def _log_quality(stats: dict) -> None:
    n = stats['n_queries']
    logging.info(
        f"Quality (best route / query, n={n}): "
        f"success={stats['n_success']}/{n} ({100 * stats['success_rate']:.1f}%)  "
        f"reconstructed={stats['n_reconstructed']}/{n} "
        f"({100 * stats['reconstruction_rate']:.1f}%)  "
        f"forward_valid={stats['n_forward_valid']}/{n} "
        f"({100 * stats['forward_valid_rate']:.1f}%)  "
        f"catalog_only={stats['n_catalog_routes']}/{n} "
        f"({100 * stats['catalog_route_rate']:.1f}%)"
    )
    logging.info(
        f"vs original query SMILES: "
        f"{stats['n_reconstructed_vs_query']}/{n} "
        f"({100 * stats['reconstruction_rate_vs_query']:.1f}%)  "
        f"(gap usually = charge/salt normalization)"
    )
    logging.info(
        f"mean sim(searched)={stats['mean_similarity_to_searched']:.3f}  "
        f"mean steps(reconstructed)={stats['mean_n_steps_reconstructed']:.2f}  "
        f"fraction with proposed BBs={100 * stats['fraction_routes_with_proposed_bbs']:.1f}%"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', nargs='?', type=str, help='Optional YAML (e.g. synthesize_synspace_reasyn.yml)')
    p.add_argument(
        '--testset',
        choices=['zinc', 'enamine', 'chembl', 'reasyn'],
        default=None,
        help='Named SynFormer / ReaSyn reconstruction test set',
    )
    p.add_argument('--input', type=str, help='Custom .smi/.txt/.csv of query SMILES')
    p.add_argument(
        '--eval-csv',
        type=str,
        default=None,
        help='Skip search; only forward-verify an existing synthesize/benchmark CSV',
    )
    p.add_argument('--limit', type=int, default=None, help='Cap number of molecules')
    p.add_argument('--output', type=str, default='output/benchmark_retrosynthesis')
    p.add_argument('--reasyn-root', type=str, default=str(DEFAULT_REASYN_ROOT))
    p.add_argument('--cache-dir', type=str, default='data/benchmarks')
    p.add_argument('--reaction-path', type=str, default=None)
    p.add_argument('--building-blocks-path', type=str, default=None)
    p.add_argument('--reaction-to-compound-path', type=str, default=None)
    p.add_argument('--max-depth', type=int, default=None)
    p.add_argument('--max-routes-per-mol', type=int, default=None)
    p.add_argument('--max-disconnections', type=int, default=None)
    p.add_argument('--n-workers', type=int, default=None)
    p.add_argument(
        '--disconnect-cache',
        type=str,
        default=None,
        help='Shared/persisted disconnection cache pickle (speeds multi-run batches)',
    )
    p.add_argument('--render', action='store_true', help='Render best good routes to PNG/HTML')
    p.add_argument('--render-top-k', type=int, default=1)
    p.add_argument('--render-max-molecules', type=int, default=50)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    cfg = merge_args_and_yaml(args, config)
    setup_logging(getattr(cfg, 'verbose', False))

    set_default(cfg, 'reaction_path', 'data/chemical_spaces/synspace_reasyn/reactions_retrosynthesis.json')
    set_default(cfg, 'building_blocks_path', 'data/chemical_spaces/synspace_reasyn/building_blocks.pkl')
    set_default(cfg, 'reaction_to_compound_path', 'data/chemical_spaces/synspace_reasyn/reaction_to_building_blocks.pkl')
    set_default(cfg, 'max_depth', 5)
    set_default(cfg, 'max_routes_per_mol', 5)
    set_default(cfg, 'max_disconnections', 64)
    set_default(cfg, 'n_workers', 1)
    set_default(cfg, 'disconnect_cache', None)
    set_default(cfg, 'output', 'output/benchmark_retrosynthesis')
    set_default(cfg, 'reasyn_root', str(DEFAULT_REASYN_ROOT))
    set_default(cfg, 'cache_dir', 'data/benchmarks')
    set_default(cfg, 'limit', None)
    set_default(cfg, 'render', False)
    set_default(cfg, 'render_top_k', 1)
    set_default(cfg, 'render_max_molecules', 50)
    set_default(cfg, 'eval_csv', None)
    if not getattr(cfg, 'input', None) and not getattr(cfg, 'eval_csv', None):
        set_default(cfg, 'testset', 'zinc')
    else:
        set_default(cfg, 'testset', None)

    out_dir = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    synthesizer = LocalRetrosynthesizer(
        reaction_path=cfg.reaction_path,
        building_blocks_path=cfg.building_blocks_path,
        reaction_to_compound_path=cfg.reaction_to_compound_path,
        max_depth=cfg.max_depth,
        max_routes_per_mol=cfg.max_routes_per_mol,
        max_disconnections=cfg.max_disconnections,
        n_workers=cfg.n_workers if not cfg.eval_csv else 1,
        disconnect_cache_path=cfg.disconnect_cache,
    )
    load_s = time.perf_counter() - t_load0
    logging.info(f'Chemical space loaded in {load_s:.2f}s')

    speed: dict = {
        'load_time_s': load_s,
        'n_reactions': len(synthesizer.reactions),
        'n_building_blocks': len(synthesizer.bb_smi_to_id),
    }

    if cfg.eval_csv:
        csv_path = Path(cfg.eval_csv)
        df = pd.read_csv(csv_path)
        logging.info(f'Evaluating existing routes from {csv_path} ({len(df)} rows)')
        test_path = str(csv_path)
    else:
        if getattr(cfg, 'input', None):
            test_path = Path(cfg.input)
        else:
            test_path = _ensure_testset(
                cfg.testset, Path(cfg.cache_dir), Path(cfg.reasyn_root)
            )
        smiles_list = _read_smiles_file(Path(test_path), limit=cfg.limit)
        if not smiles_list:
            raise SystemExit(f'No SMILES loaded from {test_path}')
        logging.info(f'Loaded {len(smiles_list)} SMILES from {test_path}')
        df, speed_run = run_search(synthesizer, smiles_list, n_workers=cfg.n_workers)
        speed.update(speed_run)
        raw_csv = out_dir / 'benchmark_routes.csv'
        df.to_csv(raw_csv, index=False)
        logging.info(f'Wrote raw routes → {raw_csv}')

    # --- Reconstruction quality (primary) ---
    quality_df, quality = evaluate_with_synthesizer(df, synthesizer, pick_best=True)
    quality_csv = out_dir / 'benchmark_quality.csv'
    quality_df.to_csv(quality_csv, index=False)

    stats = {
        **speed,
        **quality,
        'testset_path': str(test_path),
        'testset': getattr(cfg, 'testset', None),
        'max_depth': cfg.max_depth,
        'max_disconnections': cfg.max_disconnections,
        # Keep legacy aliases for older dashboards.
        'n_found': quality['n_success'],
        'n_exact': quality['n_reconstructed'],
        'recovery_rate': quality['success_rate'],
        'exact_rate': quality['reconstruction_rate'],
    }
    stats_path = out_dir / 'benchmark_stats.json'
    stats_path.write_text(json.dumps(stats, indent=2))

    _log_quality(quality)
    if 'wall_time_s' in speed:
        pm = speed['per_molecule']
        logging.info(
            f"Speed: wall={speed['wall_time_s']:.2f}s  "
            f"throughput={speed['throughput_mol_per_s']:.2f} mol/s  "
            f"per-mol mean={pm['mean_s']:.3f}s median={pm['median_s']:.3f}s "
            f"p95={pm['p95_s']:.3f}s"
        )
    logging.info(f'Wrote {quality_csv} and {stats_path}')

    if cfg.render and len(quality_df):
        # Render best quality routes (prefer good reconstructions).
        render_df = quality_df.copy()
        render_df['exact'] = render_df['good_route']
        render_df['similarity'] = render_df['similarity_to_searched']
        good = render_df[render_df['good_route'].astype(bool)]
        if len(good):
            render_df = good
        render_dir = out_dir / 'rendered'
        index = render_routes_dataframe(
            render_df,
            render_dir,
            top_k=cfg.render_top_k,
            only_found=True,
            max_molecules=cfg.render_max_molecules,
        )
        stats['render_index'] = str(index)
        stats_path.write_text(json.dumps(stats, indent=2))
        logging.info(f'Rendered pathways → {index}')


if __name__ == '__main__':
    main()
