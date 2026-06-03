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

# Map dimension index to output index and human-readable label
DIM_CONFIG = {
	0: {"output_idx": 0, "label": "H0 (Connected Components)"},
	1: {"output_idx": 1, "label": "H1 (Loops)"},
	2: {"output_idx": 2, "label": "H2 (Voids)"},
}


def _build_dist_matrix_for_dim(diagrams_cache: list, dim: int, n: int) -> np.ndarray:
	"""Compute pairwise Wasserstein distances using only homological dimension `dim`."""
	dist_matrix = np.zeros((n, n))
	total_comps = n * (n - 1) // 2
	print(f"  Computing {total_comps} pairwise distances for H{dim}...")

	k = 0
	for i in range(n):
		for j in range(i + 1, n):
			dgms_i, dgms_j = topo.align_diagrams(diagrams_cache[i], diagrams_cache[j])

			# Only use the requested dimension
			if dim >= len(dgms_i) or dim >= len(dgms_j):
				dist = 0.0
			elif dgms_i[dim].size == 0 and dgms_j[dim].size == 0:
				dist = 0.0
			else:
				dist = topo.wasserstein_distance(dgms_i[dim], dgms_j[dim])

			dist_matrix[i, j] = dist
			dist_matrix[j, i] = dist
			k += 1
			if k % max(1, total_comps // 10) == 0:
				print(f"    {k}/{total_comps} ({(k / total_comps) * 100:.0f}%)")

	return dist_matrix


def _has_data_for_dim(diagrams_cache: list, dim: int) -> bool:
	"""Return True if at least one diagram has non-empty points at dimension `dim`."""
	for by_dim in diagrams_cache:
		if dim < len(by_dim) and by_dim[dim].size > 0:
			return True
	return False


def _save_empty_plot(output_path: Path, dim: int, dim_label: str) -> None:
	"""Save a blank placeholder image indicating no data for this dimension."""
	fig, ax = plt.subplots(figsize=(14, 10))
	ax.text(
		0.5,
		0.5,
		f"No data available for {dim_label}",
		ha="center",
		va="center",
		fontsize=18,
		color="gray",
		transform=ax.transAxes,
	)
	ax.set_axis_off()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, bbox_inches="tight")
	plt.close()
	print(f"  Empty placeholder saved to: {output_path}")


def _plot_umap(
	df: pd.DataFrame,
	embedding: np.ndarray,
	output_path: Path,
	dim_label: str,
	dataset_colors: dict,
	class_markers: dict,
	sorted_datasets: list,
) -> None:
	"""Generate and save a UMAP scatter plot for a single homological dimension."""
	df = df.copy()
	df["umap_x"] = embedding[:, 0]
	df["umap_y"] = embedding[:, 1]

	pub_style = Path("src/styles/publication.mplstyle")
	if pub_style.exists():
		plt.style.use(str(pub_style))

	fig, ax = plt.subplots(figsize=(14, 10))

	unique_datasets = df["dataset_name"].unique()
	unique_classes = df["class_"].unique()

	# Null models in background
	for ds in unique_datasets:
		for cls in unique_classes:
			subset = df[
				(df["dataset_name"] == ds)
				& (df["class_"] == cls)
				& (df["is_null"] == True)
			]
			if not subset.empty:
				ax.scatter(
					subset["umap_x"],
					subset["umap_y"],
					color=dataset_colors[ds],
					marker=class_markers[cls],
					alpha=0.3,
					s=50,
					edgecolors="none",
				)

	# Empirical points on top
	for ds in unique_datasets:
		for cls in unique_classes:
			subset = df[
				(df["dataset_name"] == ds)
				& (df["class_"] == cls)
				& (df["is_null"] == False)
			]
			if not subset.empty:
				ax.scatter(
					subset["umap_x"],
					subset["umap_y"],
					color=dataset_colors[ds],
					marker=class_markers[cls],
					alpha=1.0,
					s=180,
					edgecolors="black",
					linewidths=1.5,
				)

	# Legend
	legend_elements = [Line2D([0], [0], color="none", label="Datasets:", lw=0)]
	for ds in sorted_datasets:
		ds_type = df[df["dataset_name"] == ds]["dataset_type"].iloc[0]
		label = pl._format_eph_series_label(ds) if ds_type == "eph" else ds
		legend_elements.append(
			Line2D(
				[0],
				[0],
				marker="o",
				color="none",
				label=label,
				markerfacecolor=dataset_colors[ds],
				markersize=10,
			)
		)

	legend_elements.append(Line2D([0], [0], color="none", label="\nClasses:", lw=0))
	for cls in unique_classes:
		legend_elements.append(
			Line2D(
				[0],
				[0],
				marker=class_markers[cls],
				color="none",
				label=cls,
				markerfacecolor="gray",
				markersize=10,
			)
		)

	legend_elements.append(Line2D([0], [0], color="none", label="\nModel Type:", lw=0))
	legend_elements.append(
		Line2D(
			[0],
			[0],
			marker="o",
			color="none",
			label="Empirical (Real)",
			markerfacecolor="gray",
			markeredgecolor="black",
			markersize=12,
			alpha=1.0,
			lw=1.5,
		)
	)
	legend_elements.append(
		Line2D(
			[0],
			[0],
			marker="o",
			color="none",
			label="Null Model",
			markerfacecolor="gray",
			markeredgecolor="none",
			markersize=8,
			alpha=0.3,
		)
	)

	ax.legend(
		handles=legend_elements,
		loc="center left",
		bbox_to_anchor=(1.02, 0.5),
		borderaxespad=0.0,
		frameon=False,
	)

	ax.set_title(f"UMAP of Persistence Diagram Distances — {dim_label}")
	ax.set_xlabel("UMAP Dimension 1")
	ax.set_ylabel("UMAP Dimension 2")
	ax.grid(True, linestyle="--", alpha=0.5)

	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, bbox_inches="tight")
	plt.close()
	print(f"  Plot saved to: {output_path}")


def main():
	inputs = [Path(p) for p in snakemake.input]

	# Expect exactly 3 outputs: H0, H1, H2
	output_paths = [Path(snakemake.output[i]) for i in range(3)]

	# 1. Parse inputs and load persistence diagrams
	print("Parsing inputs and loading diagrams...")
	data = []
	diagrams_cache = []

	for p in inputs:
		parts = p.parts

		if "eph" in parts:
			dataset_type = "eph"
			dataset_name = parts[parts.index("eph") + 1]
			class_ = parts[parts.index("eph") + 2]
		else:
			dataset_type = "enes"
			dataset_name = parts[parts.index("diagrams") + 1]
			class_ = parts[parts.index("diagrams") + 2]

		null_model_name = p.parent.name
		if null_model_name == "_persistence_diagram":
			is_null = False
			null_model = "empirical"
		else:
			is_null = True
			null_model = null_model_name

		by_dim = topo.load_diagrams_by_dimension(str(p))
		diagrams_cache.append(by_dim)

		data.append(
			{
				"path": str(p),
				"dataset_type": dataset_type,
				"dataset_name": dataset_name,
				"class_": class_,
				"is_null": is_null,
				"null_model": null_model,
			}
		)

	df = pd.DataFrame(data)
	n = len(df)

	if n < 2:
		print("Not enough diagrams to perform UMAP. Saving empty plots.")
		for dim, cfg in DIM_CONFIG.items():
			_save_empty_plot(output_paths[cfg["output_idx"]], dim, cfg["label"])
		return

	# 2. Build shared color / marker mappings (consistent across all dimension plots)
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
			dataset_colors[ds] = plt.cm.Set1(enes_idx % 9)
			enes_idx += 1

	unique_classes = df["class_"].unique()
	markers = ["o", "s", "^", "D", "v", "p", "*"]
	class_markers = {
		cls: markers[i % len(markers)] for i, cls in enumerate(unique_classes)
	}

	eph_datasets = [
		ds
		for ds in unique_datasets
		if df[df["dataset_name"] == ds]["dataset_type"].iloc[0] == "eph"
	]
	other_datasets = [
		ds
		for ds in unique_datasets
		if df[df["dataset_name"] == ds]["dataset_type"].iloc[0] != "eph"
	]
	sorted_datasets = utils.sort_eph_files(eph_datasets) + sorted(other_datasets)

	seed = snakemake.config.get("seed", 42)
	n_neighbors = min(15, n - 1)

	# 3. Process each homological dimension independently
	for dim, cfg in DIM_CONFIG.items():
		out_path = output_paths[cfg["output_idx"]]
		dim_label = cfg["label"]

		print(f"\n--- Processing {dim_label} ---")

		if not _has_data_for_dim(diagrams_cache, dim):
			print(f"  No data found for {dim_label}. Saving empty plot.")
			_save_empty_plot(out_path, dim, dim_label)
			continue

		dist_matrix = _build_dist_matrix_for_dim(diagrams_cache, dim, n)

		# Check if the distance matrix is effectively all zeros (degenerate case)
		if np.allclose(dist_matrix, 0):
			print(f"  Distance matrix is all zeros for {dim_label}. Saving empty plot.")
			_save_empty_plot(out_path, dim, dim_label)
			continue

		print(f"  Running UMAP for {dim_label}...")
		reducer = umap.UMAP(
			metric="precomputed",
			n_neighbors=n_neighbors,
			n_components=2,
			random_state=seed,
			n_jobs=1,
		)
		embedding = reducer.fit_transform(dist_matrix)

		_plot_umap(
			df=df,
			embedding=embedding,
			output_path=out_path,
			dim_label=dim_label,
			dataset_colors=dataset_colors,
			class_markers=class_markers,
			sorted_datasets=sorted_datasets,
		)

	print("\nAll dimension plots complete.")


if __name__ == "__main__":
	main()
