# Labor Market Structure in Argentina
### A Network Analysis of Occupational Mobility Using the ENES Dataset

**Author:** Aaron Bernal Huanca — Licenciatura en Ciencia de Datos, FCEyN, UBA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Overview

This repository contains the full Snakemake pipeline used to construct, filter, and analyze
labor-market networks from the ENES (Encuesta Nacional sobre Estructura Social) dataset.
The pipeline builds bipartite graphs linking workers to occupational (ISCO-08) and sectoral
(CAES-01) categories, applies the Hidalgo Disparity and disparity filter to extract a statistically significant
backbone, detects community structure, and produces all figures reported in the thesis.

**Key outputs:**
- Bipartite and unipartite (backbone) graphs in `.gexf` format
- Community-colored and gradient-colored projection plots
- Topological data analysis (persistence diagrams, UMAP)
- Alpha-sensitivity and resolution-sensitivity diagnostics
- EPH time-series analysis (preferential attachment, betweenness centrality)

---

## Data Availability

> *The raw microdata from ENES 2021 (ESAyPP) are not publicly available.
> To ensure topological and algorithmic reproducibility, the processed bipartite graphs
> (`data/graphs/enes_all/bipartite.gexf`) and node lists (`data/raw/nodelist_ciuo.csv`,
> `data/raw/nodelist_caes.csv`) are provided in this repository.
> The ENES 2019 microdata are publicly available at the URL below and will be
> downloaded automatically if missing.*

| Dataset | Availability | Source |
|---|---|---|
| ENES PISAC 2019 | 🟢 Public | [datos.gob.ar](https://datos.gob.ar/sq/dataset/mincyt-pisac---programa-investigacion-sobre-sociedad-argentina-contemporanea) |
| ESAyPP 2021 | 🔴 Private | Available upon request |
| EPH (INDEC) | 🟢 Public | [indec.gob.ar](https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/) |
| Bipartite graphs | 🟢 Included | `data/graphs/enes_all/` |
| Node lists | 🟢 Included | `data/raw/nodelist_*.csv` |

---

## Project Layout

```text
labor_market_structure_arg/
├── Snakefile              ← Pipeline orchestration
├── config.yaml            ← Configuration + experimentation panel
├── requirements.txt
├── rules/
│   ├── 00_prepare.smk     ← Data preparation
│   ├── 01_bipartite.smk   ← Bipartite graph construction
│   ├── 02_projection.smk  ← Disparity filter + projection
│   ├── 03_communities.smk ← Community detection
│   └── 04_diagrams.smk    ← Topological data analysis
├── scripts/               ← Python scripts (one per rule)
├── src/                   ← Shared library (graph utils, plotting)
├── data/
│   ├── raw/               ← Base node lists + raw surveys (see above)
│   ├── graphs/            ← Generated .gexf graphs
│   └── processed/         ← Intermediate node lists and tables
└── images/                ← All output figures
```

---

## 1. Environment Setup

Python 3.12+ is recommended. Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Reproducing the Manuscript Results

The pipeline defaults to the exact parameters used in the thesis:
**Disparity Filter α = 0.05**, **Infomap** community detection, **ISCO-08** classification.

Run the default pipeline with:

```bash
snakemake --cores all
```

This generates the key projection plots (`03_projection_plot_by_groups` and
`03_projection_plot_gradient`) for the parameters configured in `config.yaml`
under `run_experiments`.

To reproduce **all** outputs from every chapter of the thesis (one representative
figure per pipeline rule), run:

```bash
snakemake thesis_all --cores all
```

> **Note:** `thesis_all` requires access to the EPH time-series data for the
> dynamic network chapters (7+). ENES-only chapters will run without EPH data.

---

## 3. Experimenting with Variants

The pipeline is fully parameterized. To explore alternative network topologies,
**edit only the `run_experiments` section** at the bottom of `config.yaml`:

```yaml
# config.yaml — run_experiments panel
run_experiments:
  datasets_to_build:   ["enes_all"]
  classes_to_build:    ["ciuo"]
  alphas_to_build:     ["0.05"]          # ← change threshold here
  algorithms_to_build: ["infomap"]       # ← change algorithm here
  discrete_features:   ["grupo", "community"]
  continuous_features: ["female_pct", "income_mean", "nivel_ed_mean"]
```

**Example:** Add a stricter threshold (α = 0.25) and compare Leiden vs Infomap:

```yaml
run_experiments:
  datasets_to_build:   ["enes_all"]
  classes_to_build:    ["ciuo"]
  alphas_to_build:     ["0.05", "0.25"]
  algorithms_to_build: ["infomap", "leiden"]
  discrete_features:   ["community"]
  continuous_features: ["female_pct"]
```

Then run `snakemake --cores all`. Snakemake detects the missing targets and
builds only what is needed — no recomputation of existing outputs.

---

## 4. Interactive CLI Exploration

Individual outputs can be requested directly without modifying `config.yaml`.
Snakemake resolves the full upstream dependency chain automatically.

**Example:** generate a projection plot for α = 0.10 with the Leiden algorithm:

```bash
snakemake --cores 1 "images/enes_all/ciuo/03_projection_plot_by_groups/_hidalgo_0.10_pos_leiden_community.png"
```

Snakemake will automatically:
1. Filter the raw bipartite graph with the Disparity Filter at α = 0.10
2. Run Leiden to detect communities
3. Compute graph layout positions
4. Render and save the figure

**Valid parameter values for CLI exploration:**

| Wildcard | Valid values |
|---|---|
| `{alpha}` | Any decimal, e.g. `0.01`–`1.00` |
| `{algorithm}` | `infomap`, `leiden`, `louvain` |
| `{class_}` | `ciuo` (occupations), `caes` (sectors) |
| `{dataset}` | `enes_all`, `enes_2019`, `enes_2021` |
| `{weight_function}` | `hidalgo`, `dot_product`, `cosine` |
| `{discrete_feature}` | `grupo`, `community` |
| `{continuous_feature}` | `female_pct`, `income_mean`, `nivel_ed_mean`, `age_mean`, … |

---

## 5. Pipeline Utilities

**Visualize the execution DAG:**
```bash
snakemake --dag | dot -Tpng > images/dag.png
```

**Dry run (show what would be built without executing):**
```bash
snakemake --dryrun --cores all
```

**Force rebuild of a specific output:**
```bash
snakemake --cores 1 -F "images/enes_all/ciuo/07_alpha_sensitivity/_hidalgo.png"
```

**GPU acceleration (optional, for large graphs):**
```bash
sudo apt install nvidia-cuda-toolkit
pip install nx-cugraph-cu13
export CUDA_PATH=/usr
```

---

## Citation

If you use this pipeline or data in your research, please cite:

```bibtex
@misc{bernal2025labormarket,
  author    = {Bernal Huanca, Aaron},
  title     = {Labor Market Structure in Argentina: A Network Approach},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```
