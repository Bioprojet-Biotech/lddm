# Large Drug Discovery Model (LDDM)

![](docs/lddm.png)

Official code repository for "A Unified 3D Generative Model for Synthesizable Structure-Based Drug Design".
LDDM is a pocket-conditioned generative model that supports a broad range of molecular modeling tasks within a single model:
- De novo design
- Fragment growing
- Fragment linking
- Docking
- Partial docking (e.g. for covalent ligands)
- Locally programmable design
- Synthesizable design in virtual chemical spaces

We have tested many of these capabilities in prospective ligand design case studies. Read more about our experimental results on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.09.15.751537v1).

## Setup

### Environment 

Clone the repository:
```bash
git clone https://github.com/LPDI-EPFL/lddm.git
```
Then, install the environment:
```bash
uv sync
```
which will install packages in `.venv` in the work directory.

To use the environment, run scripts with:
```bash
uv run path/to/script.py
```
or activate the environment before running python:
```bash
source .venv/bin/activate
python path/to/script.py
```

### Docker container

In case you don't have [`uv`](https://docs.astral.sh/uv/) installed, we also provide a lightweight [Docker](https://www.docker.com/) container, which can be used as a starting working environment.
In addition to `uv`, [Gnina](https://github.com/gnina/gnina) and [Reduce](https://github.com/rlabduke/reduce) are already pre-installed, which are required for the programmable design workflows. The Python packages must be installed separately via `uv sync`, as described above.

You can pull the image from Docker Hub:
```bash
docker pull schneuing/lddm:0.1.0
```

When using the container, make sure that it has access to your system's GPU as well as the `.venv` folder.

### Checkpoint, geometry reference

Download a pretrained checkpoint from [Zenodo](https://zenodo.org/records/22754501):
```bash
wget -P checkpoints/ https://zenodo.org/records/22754501/files/<name>.ckpt
```

We provide two checkpoints with different licenses. The main checkpoint (`CD+BB+BN`) was partially trained on [BindingNet](http://bindingnetv2.huanglab.org.cn/documentation) which was published with a more restrictive [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. The checkpoint without BindingNet is released under MIT License.

| Model | Link | Description | License |
|---|---|---|---|
| `CD+BB+BN` | https://zenodo.org/records/22754501/files/lddm.ckpt | Model used for experiments in the paper | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `CD+BB` | https://zenodo.org/records/22754501/files/lddm_CDBB.ckpt | Model trained without BindingNet | MIT |

For programmable design, synthesizable design, or 3D validity evaluation, download the geometry reference from Zenodo:
```bash
wget -P data/validity3d/ https://zenodo.org/records/22754501/files/ligands.sdf
```

## Basic usage examples

### De novo design

```bash
python scripts/sample.py design \
    --protein examples/kras.pdb \
    --ref_ligand examples/kras_ref_ligand.sdf \
    --checkpoint checkpoints/lddm.ckpt \
    --output examples/de_novo_samples.sdf
```

### Fragment-based design

```bash
python scripts/sample.py design \
    --protein examples/kras.pdb \
    --ref_ligand examples/kras_ref_ligand.sdf \
    --ligand examples/kras_frag.sdf \
    --checkpoint checkpoints/lddm.ckpt \
    --output examples/grown_samples.sdf
```

For **synthesizable** fragment-based design (grow a fixed fragment inside a
virtual chemical space), use the programmable interface with
`starting_fragments` and `synthesizable: true` — see
[Synthesizable design](#synthesizable-design) below and
`configs/controlled_generation/synthesizable_fragment_design.yml`.

### Docking

```bash
python scripts/sample.py dock \
    --protein examples/kras.pdb \
    --ref_ligand examples/kras_ref_ligand.sdf \
    --ligand examples/kras_ref_ligand.sdf \
    --checkpoint checkpoints/lddm.ckpt \
    --output examples/docked_samples.sdf
```

### Partial docking

```bash
python scripts/sample.py dock \
    --protein examples/kras.pdb \
    --ref_ligand examples/kras_ref_ligand.sdf \
    --ligand examples/kras_ref_ligand.sdf \
    --atoms_to_dock 16 17 18 19 20 21 24 25 26 27 28 29 30 31 \
    --checkpoint checkpoints/lddm.ckpt \
    --output examples/partially_docked_samples.sdf
```

### Important parameters

| Parameter | Mode | Default | Description |
|---|---|---|---|
| `--n_samples` | both | `10` | Number of molecules to generate. |
| `--molecule_size` | design | histogram | Target ligand size: an integer, `uniform_<low>_<high>`, or omit to sample from the pocket-conditioned size histogram. In `dock` the size is fixed to the input molecule. |
| `--atoms_to_dock` | dock | whole molecule | Atom indices whose positions are generated (partial docking); all other atoms keep their input coordinates. |
| `--n_steps` | both | `100` | Number of integration steps of the sampling process (higher = slower, usually better). |
| `--sampler` | both | `ForwardEuler` | ODE sampler: `ForwardEuler` or `HeunSampler`. |
| `--sampling_noise` | both | `5.0` | Scale of the stochastic noise injected into atom coordinates during sampling. |
| `--batch_size` | both | `--n_samples` | Number of samples generated per forward pass (lower it if you hit OOM). |
| `--n_frames` | both | `None` | If set, save a sampling trajectory with this many frames (one sample only) instead of a batch of final molecules. |
| `--return_projected_final` | both | off | With `--n_frames`, project each frame onto the clean final prediction instead of the raw noisy state. |
| `--seed` | both | `None` | Random seed for reproducible sampling. |
| `--device` | both | `cuda:0` | Device to run on, e.g. `cuda:0` or `cpu`. |

## Advanced sampling

![](docs/synthgen.png)

> [!NOTE]
> On first use, the geometry evaluator compiles reference distributions and caches
> them beside `data/validity3d/ligands.sdf`. This can take substantial time before sampling starts.

### Programmable design

```bash
python scripts/generate_programmable_design.py configs/controlled_generation/programmable_design.yml
```

Estimated runtime: **8 minutes** for the default KRAS example on one NVIDIA H100 with four CPU cores.

### Synthesizable design

[Enamine REAL](https://enamine.net/compound-collections/real-compounds/real-space-navigator) reactions and building blocks require a license and cannot be
redistributed here. For this demonstration, we provide a smaller chemical space
of 44,944 building blocks and three reactions derived from
[SynSpace](https://github.com/whitead/synspace), with precomputed reaction-to-building-block
mappings. Please contact us if you want to use synthesizable design with Enamine REAL.

Download the SynSpace data from Zenodo:
```bash
wget -P data/synspace/ https://zenodo.org/records/22754501/files/building_blocks.csv
wget -P data/synspace/ https://zenodo.org/records/22754501/files/building_blocks.pkl
wget -P data/synspace/ https://zenodo.org/records/22754501/files/reactions.json
wget -P data/synspace/ https://zenodo.org/records/22754501/files/reaction_to_building_blocks.csv
wget -P data/synspace/ https://zenodo.org/records/22754501/files/reaction_to_building_blocks.pkl
```

For your own chemical space, provide a building-block CSV with unique `id` and
`smiles` columns and a reaction JSON list. Each reaction must have two reactants
and one product, for example:

```json
[
    {
        "id": "amide", 
        "reaction": "[C:1](=[O:2])[O;H1].[N;H1,H2:3]>>[C:1](=[O:2])[N:3]", 
        "explicit_hs": false
    }
]
```

```bash
python scripts/prepare_chemical_space.py \
    --building-blocks path/to/building_blocks.csv \
    --reactions path/to/reactions.json \
    --output data/chemical_spaces/custom
```

To expand the demo SynSpace with [ReaSyn](https://github.com/NVIDIA-Digital-Bio/reasyn) /
[SynFormer](https://github.com/wenhao-gao/synformer/tree/main/data/rxn_templates)
building blocks and reaction templates, fuse them first (uni- and tri-molecular
ReaSyn templates are dropped from forward `reactions.json`; retrosynthesis keeps
uni/bi/tri in `reactions_retrosynthesis.json`). Building blocks are **deduped by
canonical SMILES** with a single preferred primary provider
(`synspace` > `enamine` > `molport` > `mcule` > `reasyn`) while all origins are
kept in a `sources` column (`provider:id;…`):

```bash
python scripts/fuse_chemical_spaces.py \
    --current-building-blocks data/synspace/building_blocks.csv \
    --current-reactions data/synspace/reactions.json \
    --reasyn-building-blocks path/to/building_blocks.txt \
    --reasyn-reactions path/to/comprehensive.txt \
    --output data/chemical_spaces/synspace_reasyn \
    --skip-invalid-bbs \
    --prepare
```

To broaden buyable coverage further with [Mcule purchasable building
blocks](https://mcule.com/database/) (same priority-aware dedup):

```bash
python scripts/expand_bb_catalog.py \
    --base-building-blocks data/chemical_spaces/synspace_reasyn/building_blocks.csv \
    --base-reactions data/chemical_spaces/synspace_reasyn/reactions.json \
    --download-mcule \
    --max-mcule-bbs 200000 \
    --output data/chemical_spaces/synspace_reasyn_mcule \
    --prepare --drop-empty-roles
```

Omit `--max-mcule-bbs` for the full Mcule dump (~millions of BBs; prepare is
expensive). Then synthesize with
`configs/controlled_generation/synthesize_synspace_reasyn_mcule.yml`. Route CSV
columns include `building_block_providers`, `building_block_sources`, and
`bb_priority` (lower is better).

The script validates the molecules, computes fingerprints, and assigns blocks to
reaction roles by SMARTS matching. To preserve curated assignments, add
`--memberships path/to/memberships.csv` with columns `reaction_id`, `reactant_role`
(`0` or `1`, in SMARTS order), and `building_block_id`. Set the resulting paths in
`configs/controlled_generation/synthesizable_design.yml`:

```yaml
itergen_params:
  reaction_path: data/chemical_spaces/custom/reactions.json
  building_blocks_path: data/chemical_spaces/custom/building_blocks.pkl
  reaction_to_compound_path: data/chemical_spaces/custom/reaction_to_building_blocks.pkl
```

Run synthesizable design with the following command:

```bash
python scripts/generate_programmable_design.py configs/controlled_generation/synthesizable_design.yml
```

Estimated runtime: **21 minutes** for the default KRAS example on one NVIDIA H100 with four CPU cores.

To grow from a fixed starting fragment instead of designing de novo, set
`starting_fragments` (or pass `--starting_fragments`) to an SDF whose molecule
matches a reactant role in your chemical space:

```bash
python scripts/generate_programmable_design.py \
    configs/controlled_generation/synthesizable_fragment_design.yml
```

### Retrosynthesis (synthesize mode)

Given query SMILES, recover **multi-step** synthesis pathways by reversing the
same local reaction templates until every leaf is a catalog building block
(SynSpace, fused SynSpace+ReaSyn, or a custom space from
`prepare_chemical_space.py`). The search uses forward-verified reverse SMARTS
for **uni-, bi-, and trimolecular** single-product templates (retrosynthesis is
not limited to LDDM’s bimolecular forward-generation constraint), aromatic
C–hetero cuts, explicit-halogen reverse variants, charge/salt normalization,
buyable-first beam expansion, iterative deepening, and BB-substructure pair
recovery when templates alone are too noisy.

By default, reverse-proposed **halide / small alcohol–thiol partners** missing
from the catalog are accepted as leaves (`building_block_ids` tagged
`proposed__…`). Pass `--no-proposed-reagents` for strict catalog-only membership.

```bash
python scripts/synthesize.py configs/controlled_generation/synthesize.yml \
    --smiles 'CC1(NC(=O)C2CC2)CC1' \
    --output output/synthesize_pathways.csv
```

For the fused SynSpace + ReaSyn space (uni/bi/tri retrosynthesis templates in
`reactions_retrosynthesis.json`; bimolecular `reactions.json` still used for
forward synthesizable design):

```bash
python scripts/synthesize.py configs/controlled_generation/synthesize_synspace_reasyn.yml \
    --input molecules.smi \
    --max-depth 5 \
    --n-workers 4 \
    --output output/synthesize_pathways.csv \
    --html output/synthesize_pathways.html
```

Procedure enrichment is **on by default** when
`data/procedures/named_reaction_procedures.json` exists: each route row gains
`procedures` (JSON), `procedure_match_levels`, and `procedure_titles` from the
curated KB + local [OpenAlex](https://developers.openalex.org/) prepared cache
(`data/openalex/prepared/`). Pass `--no-enrich-procedures` to skip. Use `--html`
to write an embedded report with procedure panels in one step.

Or pass a `.smi` / `.csv` file with `--input`. For large batches, `--n-workers`
runs molecules in a process pool; `--stream-output` writes CSV rows
incrementally; `--disconnect-cache PATH` reuses ranked disconnections across
runs. A prepared building-block sidecar (`*.pkl.prepared.pkl`) is written on
first load to speed cold starts. Build a compiled reaction pack + feature index
with `scripts/prepare_retrosynthesis_pack.py`, then seed the disconnect cache
from benchmarks / Murcko scaffolds via `scripts/seed_disconnect_cache.py`.

Output columns include `react_trace` (compatible with synthesizable design;
uni/tri use the matching trace helpers) and a human-readable `pathway`.

To keep **near-miss** constructions (forward product close but not identical to
the query), enable approximate mode with a Tanimoto threshold:

```bash
python scripts/synthesize.py configs/controlled_generation/synthesize.yml \
    --smiles 'CC1(NC(=O)C2CC2)CC1' \
    --approximate \
    --similarity-threshold 0.7 \
    --output output/synthesize_approx.csv
```

Exact routes are preferred when available. Approximate hits report `exact=false`,
`similarity`, and `reconstructed_smiles` (the molecule the route actually builds).

### Benchmark reconstruction quality (SynFormer / ReaSyn test sets)

Primary metrics match ReaSyn’s reconstruction eval
([`eval_recon.py`](https://github.com/NVIDIA-Digital-Bio/reasyn/blob/main/scripts/eval_recon.py)):
does a returned pathway **forward-replay** to the query?

| Metric | Meaning |
|--------|---------|
| `success_rate` | Any pathway returned |
| `reconstruction_rate` | Every step fires under recorded SMARTS **and** product == searched SMILES |
| `catalog_route_rate` | Reconstruction using only catalog BBs (no `proposed__`) |
| `forward_valid_rate` | Steps verify even if product is only a near-miss |

```bash
# Search + quality on ReaSyn ZINC-1k
python scripts/benchmark_retrosynthesis.py \
    configs/controlled_generation/synthesize_synspace_reasyn.yml \
    --testset zinc --limit 50 --n-workers 4 --render \
    --output output/benchmark_zinc50

# Re-score an existing CSV (no re-search)
python scripts/benchmark_retrosynthesis.py \
    configs/controlled_generation/synthesize_synspace_reasyn.yml \
    --eval-csv output/benchmark_zinc50/benchmark_routes.csv \
    --output output/benchmark_zinc50_eval

# SynFormer Enamine / ChEMBL 1k (downloaded into data/benchmarks/)
python scripts/benchmark_retrosynthesis.py \
    configs/controlled_generation/synthesize_synspace_reasyn.yml \
    --testset enamine --limit 100 --n-workers 4 \
    --output output/benchmark_enamine100
```

Render an existing `synthesize.py` CSV:

```bash
# PNG panels + index.html (external images)
python scripts/render_synthesize_pathways.py output/synthesize_pathways.csv \
    -o output/rendered_pathways --top-k 1

# Chemist-facing single-file HTML (all SVG drawings embedded inline)
python scripts/render_synthesize_pathways.py output/synthesize_pathways.csv \
    --embedded -o output/retrosynthesis_report.html --top-k 3
```

Open `…/rendered/index.html` or the embedded `.html` offline (no external assets).
Reaction JSON catalogs expose SMARTS / name / optional source — **not** experimental
yields or lab conditions
([ReaSyn](https://github.com/MolecularAI/ReaSyn) /
[SynFormer templates](https://github.com/wenhao-gao/synformer/tree/main/data/rxn_templates)).

### Procedure enrichment + OpenAlex (offline)

After a route CSV exists, attach **curated procedure cards** (solvent / T / stoich /
workup / typical yield ranges) and **local literature refs** from a prepared
[OpenAlex](https://developers.openalex.org/) cache. Runtime enrichment never
calls the network.

```bash
# 1) Download query-scoped works (not the full multi-hundred-GB S3 dump)
#    https://developers.openalex.org/download/download-to-machine
export OPENALEX_MAILTO=you@org.com
python scripts/download_openalex_dataset.py --mailto "$OPENALEX_MAILTO"

# 2) Normalize into data/openalex/prepared/ (openalex_refs.json, reaction_index.json)
python scripts/prepare_openalex_dataset.py

# 3) Literature-driven coverage vs synspace_reasyn retro templates + BB role map
python scripts/analyze_reaction_coverage.py
# → data/chemical_spaces/synspace_reasyn/reaction_coverage_report.{json,md}

# 4) Enrich a synthesize / benchmark CSV (JSON + optional embedded HTML)
python scripts/enrich_synthesize_pathways.py \
    --csv output/synthesize_pathways.csv \
    --out output/enriched_pathways.json \
    --html output/enriched_retrosynthesis.html \
    --top-k 3
```

Vocab: `data/openalex/named_reaction_vocab.json`. Curated cards:
`data/procedures/named_reaction_procedures.json`. Prepared refs are small enough
to keep under `data/openalex/prepared/`; raw JSONL under `data/openalex/raw/` is
gitignored and regenerable.

Docs for reaction templates:
[https://github.com/MolecularAI/ReaSyn](https://github.com/MolecularAI/ReaSyn) /
[SynFormer templates](https://github.com/wenhao-gao/synformer/tree/main/data/rxn_templates)
and SynSpace reactions under `data/synspace/`.
<!-- ## Citing this work

TODO -->
