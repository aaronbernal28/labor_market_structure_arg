import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Any

import umap.umap_ as umap
from scripts import *

try:
    from ggsci import pal_observable
except ImportError:
    pal_observable = None

snakemake: Any


def main():
    inputs = [Path(p) for p in snakemake.input]
    output_path = Path(snakemake.output[0])

    # Extract wildcards for the title
    wf = getattr(snakemake.wildcards, "weight_function", "Unknown Weight")
    tm = getattr(snakemake.wildcards, "topo_method", "Unknown Topo")

    # 1. Parse inputs and load persistence diagrams
    print("Parsing inputs and loading diagrams...")
    data = []
    diagrams_cache = []

    for p in inputs:
        parts = p.parts

        # Determine dataset type and dataset name based on path structure
        if "eph" in parts:
            dataset_type = "eph"
            dataset_name = parts[parts.index("eph") + 1]
            class_ = parts[parts.index("eph") + 2]
        else:
            dataset_type = "enes"
            dataset_name = parts[parts.index("diagrams") + 1]
            class_ = parts[parts.index("diagrams") + 2]

        # Determine if it is a null model or empirical (real)
        null_model_name = p.parent.name
        if null_model_name == "_persistence_diagram":
            is_null = False
            null_model = "empirical"
        else:
            is_null = True
            null_model = null_model_name

        # Load diagram points by dimension
        by_dim = topo.load_diagrams_by_dimension(str(p))
        diagrams_cache.append(by_dim)

        data.append({
            "path": str(p),
            "dataset_type": dataset_type,
            "dataset_name": dataset_name,
            "class_": class_,
            "is_null": is_null,
            "null_model": null_model,
        })

    df = pd.DataFrame(data)
    n = len(df)

    if n < 2:
        print("Not enough diagrams to perform UMAP. Exiting.")
        return

    # 2 & 3. Compute pairwise Wasserstein distances
    dist_matrix = np.zeros((n, n))
    total_comps = n * (n - 1) // 2
    print(f"Computing pairwise Wasserstein distances for {n} diagrams ({total_comps} pairs)...")

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dgms_i, dgms_j = topo.align_diagrams(diagrams_cache[i], diagrams_cache[j])
            total_dist = 0.0

            # Sum distances across all homological dimensions
            for dim in range(len(dgms_i)):
                if dgms_i[dim].size == 0 and dgms_j[dim].size == 0:
                    continue
                dist = topo.wasserstein_distance(dgms_i[dim], dgms_j[dim])
                total_dist += dist

            dist_matrix[i, j] = total_dist
            dist_matrix[j, i] = total_dist
            k += 1
            if k % max(1, total_comps // 10) == 0:
                print(f"Computed {k}/{total_comps} distances ({(k/total_comps)*100:.0f}%)")

    # 4. Apply UMAP to the precomputed distance matrix
    print("Running UMAP...")
    # Cap neighbors by dataset size for safety
    n_neighbors = min(15, n - 1)
    reducer = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors, n_components=2, random_state=snakemake.config.get("seed", 42), n_jobs=1)
    embedding = reducer.fit_transform(dist_matrix)

    df["umap_x"] = embedding[:, 0]
    df["umap_y"] = embedding[:, 1]

    # 5. Plotting
    print("Generating plot...")
    pub_style = Path("src/styles/publication.mplstyle")
    if pub_style.exists():
        plt.style.use(str(pub_style))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Assign distinct colors to datasets
    unique_datasets = df["dataset_name"].unique()
    dataset_colors = {}
    eph_idx = 0
    enes_idx = 0

    for ds in unique_datasets:
        ds_type = df[df["dataset_name"] == ds]["dataset_type"].iloc[0]
        if ds_type == "eph":
            if pal_observable is not None:
                pal = pal_observable("observable10", alpha=1.0)(10)
                dataset_colors[ds] = pal[eph_idx % len(pal)]
            else:
                dataset_colors[ds] = plt.cm.tab10(eph_idx % 10)
            eph_idx += 1
        else:
            # Standard palette for non-EPH datasets
            dataset_colors[ds] = plt.cm.Set1(enes_idx % 9)
            enes_idx += 1

    # Assign markers to classes
    unique_classes = df["class_"].unique()
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    class_markers = {cls: markers[i % len(markers)] for i, cls in enumerate(unique_classes)}

    # Plot null models first so they stay in the background
    for ds in unique_datasets:
        for cls in unique_classes:
            subset = df[(df["dataset_name"] == ds) & (df["class_"] == cls) & (df["is_null"] == True)]
            if not subset.empty:
                ax.scatter(
                    subset["umap_x"],
                    subset["umap_y"],
                    color=dataset_colors[ds],
                    marker=class_markers[cls],
                    alpha=0.3,
                    s=50,
                    edgecolors="none"
                )

    # Plot empirical (real) models on top with higher alpha, bigger size, and outlines
    for ds in unique_datasets:
        for cls in unique_classes:
            subset = df[(df["dataset_name"] == ds) & (df["class_"] == cls) & (df["is_null"] == False)]
            if not subset.empty:
                ax.scatter(
                    subset["umap_x"],
                    subset["umap_y"],
                    color=dataset_colors[ds],
                    marker=class_markers[cls],
                    alpha=1.0,
                    s=180,
                    edgecolors="black",
                    linewidths=1.5
                )

    # Build custom legends
    legend_elements = []

    # Sort datasets: EPH files sorted by quarter/year, then others
    eph_datasets = []
    other_datasets = []
    for ds in unique_datasets:
        ds_type = df[df["dataset_name"] == ds]["dataset_type"].iloc[0]
        if ds_type == "eph":
            eph_datasets.append(ds)
        else:
            other_datasets.append(ds)

    eph_datasets_sorted = utils.sort_eph_files(eph_datasets)
    sorted_datasets = eph_datasets_sorted + sorted(other_datasets)

    # Legend section for Datasets
    legend_elements.append(Line2D([0], [0], color='none', label='Datasets:', lw=0))
    for ds in sorted_datasets:
        ds_type = df[df["dataset_name"] == ds]["dataset_type"].iloc[0]
        if ds_type == "eph":
            label = pl._format_eph_series_label(ds)
        else:
            label = ds
        legend_elements.append(
            Line2D([0], [0], marker='o', color='none', label=label,
                   markerfacecolor=dataset_colors[ds], markersize=10)
        )

    # Legend section for Classes
    legend_elements.append(Line2D([0], [0], color='none', label='\nClasses:', lw=0))
    for cls in unique_classes:
        legend_elements.append(
            Line2D([0], [0], marker=class_markers[cls], color='none', label=cls,
                   markerfacecolor='gray', markersize=10)
        )

    # Legend section for Empirical vs Null
    legend_elements.append(Line2D([0], [0], color='none', label='\nModel Type:', lw=0))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='none', label='Empirical (Real)',
               markerfacecolor='gray', markeredgecolor='black', markersize=12, alpha=1.0, lw=1.5)
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='none', label='Null Model',
               markerfacecolor='gray', markeredgecolor='none', markersize=8, alpha=0.3)
    )

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., frameon=False)

    # Setup Titles and Layout
    ax.set_title("UMAP of Persistence Diagram Distances")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
