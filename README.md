# Labor Market Structure in Argentina

### Authors
* **Aaron Bernal Huanca**¹

1. **Licenciatura en Ciencia de Datos, FCEyN** Universidad de Buenos Aires, Email: [ahuanca@dc.uba.ar](mailto:ahuanca@dc.uba.ar)

**Subject:** Social Network Science

## General Description


## Project Structure

```text
labor_market_structure_arg/
├── requirements.txt                   # Replaces requirements.txt
├── Snakefile                          # Master orchestrator
├── config.yaml                        # Dynamic paths, colors, thresholds
├── rules/                             # Fragmented Snakemake logic
│   ├── 00_prepare.smk                 # Rules for merging data and EDA
│   ├── 01_bipartite.smk               # Rules for bipartite graphs
│   └── 02_projection.smk              # Rules for projections graphs and community detection
├── data/
│   ├── raw/                           # Raw survey data
|   |   ├── base_enespersonas.csv      # Autonomously downloaded from ENES PISAC 2019
│   │   ├── base_enespersonas_2021.csv # Must be manually placed here
│   │   ├── nodelist_caes.csv
│   │   └── nodelist_ciuo.csv
│   └── processed/                     # Snakemake writes datasets AND intermediate files here
├── images/                            # Final visualizations written by Snakemake
│   └── <script_name>/                 # Subfolders for each script's output
│       └── <script_name>_<parameters>*.png # Parameterized filenames
├── scripts/                           # Connect data to src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── prepare_data.py
│   │   ├── build_bipartite.py
│   │   └── build_projection.py
│   ├── plot/
│   │   ├── 00_top_workers.py
│   │   ├── 01_bipartite_by_groups.py
│   │   ├── 02_projection_by_groups.py
|   |   ├── 03_projection_by_gradient.py
│   │   └── 04_edge_correlation.py
│   └── report/
│       └── 00_walt_test.py
└── src/                               # The core Python package, independent of Snakemake
	└── occupational_networks/
		├── __init__.py
		├── data/
		│   ├── loader.py              # Loading and merging raw datasets
		│   └── features.py            # Feature engineering functions
		├── graph/
		│   ├── builder.py             # Functions to build bipartite and projection graphs
		│   ├── metrics.py             # Functions to compute degree, centrality, etc.
		│   └── topology.py            # Functions to analyze graph structure
		├── plotting/
		│   ├── __init__.py
		│   ├── eda.py                 # Plotting functions for exploratory data analysis
		│   ├── bipartite.py           # Plotting functions for bipartite graphs
		│   ├── projection.py          # Plotting functions for projection graphs
		│   └── styles/
		│       └── publication.mplstyle # Matplotlib style for publication-quality figures
		└── utils.py                   # Utility functions
```

## Methodology


## Key Dependencies

- **Network Analysis**: NetworkX
- **Scientific Computing**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn
- **File I/O**: openpyxl

## Installation

Python 3.12.3 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Usage Workflow

## Data Description

- **ENES PISAC 2019**: https://datos.gob.ar/sq/dataset/mincyt-pisac---programa-investigacion-sobre-sociedad-argentina-contemporanea
- **ESAyPP 2021**: Encuesta sobre Estructura Social y Políticas Publicas 2021
- **CAES Node List**: Classification of Economic Activities (branch of activity codes and labels)
- **CIUO Node List**: Classification of Occupations 1.0 Argentina (occupation codes and labels)

## References
