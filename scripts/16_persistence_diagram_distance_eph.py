
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from scripts import *

import src.utils as utils

import src.topology as topo

snakemake: Any

MAX_YEAR_GAP = 10
MIN_DIMENSION_PRESENCE = 0.5


def _extract_eph_wave_label(path: str | Path) -> str:
	path_obj = Path(path)
	key = utils.parse_eph_file_key(path_obj.parents[2].name)
	if key is not None:
		return str(key.year)
	return path_obj.parents[2].name


def _wave_sort_key(label: str) -> tuple[int, float | int | str]:
	key = utils.parse_eph_file_key(label)
	if key is not None:
		return 0, key.time_numeric
	if label.isdigit():
		return 1, int(label)
	return 2, label


def _wave_year(label: str) -> int | None:
	if label.isdigit():
		return int(label)
	key = utils.parse_eph_file_key(label)
	if key is not None:
		return key.year
	return None


def _load_waves() -> tuple[list[str], list[dict[int, np.ndarray]]]:
	paths = [Path(path) for path in snakemake.input]
	wave_items = sorted(
		((_extract_eph_wave_label(path), path) for path in paths),
		key=lambda item: _wave_sort_key(item[0]),
	)
	wave_labels = [label for label, _ in wave_items]
	wave_diagrams = [topo.load_diagrams_by_dimension(str(path)) for _, path in wave_items]
	return wave_labels, wave_diagrams


def _compute_distance_matrices(
	wave_diagrams: list[dict[int, np.ndarray]],
	wave_labels: list[str],
	metric: str,
) -> dict[int, np.ndarray]:
	if not wave_diagrams:
		return {}

	wave_years = [_wave_year(label) for label in wave_labels]
	n_waves = len(wave_diagrams)
	if n_waves == 0:
		return {}

	max_dim = max((max(diagrams.keys(), default=-1) for diagrams in wave_diagrams), default=-1)
	if max_dim < 0:
		return {}

	matrices: dict[int, np.ndarray] = {}
	for dim in range(max_dim + 1):
		present_count = sum(1 for diagrams in wave_diagrams if dim in diagrams)
		if present_count / n_waves < MIN_DIMENSION_PRESENCE:
			continue
		matrix = np.full((n_waves, n_waves), np.nan, dtype=float)
		for i in range(n_waves):
			if dim in wave_diagrams[i]:
				matrix[i, i] = 0.0
			for j in range(i + 1, n_waves):
				year_i = wave_years[i]
				year_j = wave_years[j]
				if year_i is None or year_j is None:
					continue
				if abs(year_i - year_j) > MAX_YEAR_GAP:
					continue
				if dim not in wave_diagrams[i] or dim not in wave_diagrams[j]:
					continue
				left_diagrams, right_diagrams = topo.align_diagrams(
					wave_diagrams[i],
					wave_diagrams[j],
				)
				left_dgm, right_dgm = left_diagrams[dim], right_diagrams[dim]
				if metric == "bottleneck":
					value = topo.bottleneck_distance(left_dgm, right_dgm)
				elif metric == "wasserstein":
					value = topo.wasserstein_distance(left_dgm, right_dgm)
				else:
					raise ValueError(f"Unsupported metric: {metric}")
				matrix[i, j] = value
				matrix[j, i] = value
		matrices[dim] = matrix

	return matrices


def _plot_faceted_heatmaps(
	matrices: dict[int, np.ndarray],
	wave_labels: list[str],
	metric: str,
	output_path: Path,
	class_: str,
	weight_function: str,
	topo_method: str,
	translation: dict[str, str] | None = None,
) -> None:
	dimensions = sorted(matrices)
	if not dimensions:
		raise ValueError("No persistence diagram dimensions were found in the input files.")

	n_cols = min(3, len(dimensions))
	n_rows = ceil(len(dimensions) / n_cols)
	base_side = max(5.5, 0.65 * len(wave_labels))
	fig, axes = plt.subplots(
		n_rows,
		n_cols,
		figsize=(base_side * n_cols, base_side * n_rows * 0.76),
		constrained_layout=False,
	)
	axes = np.atleast_1d(axes).ravel()
	plot_axes = axes[: len(dimensions)]
	for ax in axes[len(dimensions) :]:
		ax.remove()

	valid_maxima = [float(np.nanmax(matrix)) for matrix in matrices.values() if np.isfinite(matrix).any()]
	vmax = max(valid_maxima) if valid_maxima else 1.0
	if not np.isfinite(vmax) or vmax <= 0:
		vmax = 1.0

	cmap = sns.color_palette("mako", as_cmap=True)

	def _t(label: str) -> str:
		return utils.translate_label(label, translation) if translation else label

	for ax, dim in zip(plot_axes, dimensions):
		df = pd.DataFrame(matrices[dim], index=wave_labels, columns=wave_labels)
		mask = df.isna()
		local_max = float(np.nanmax(matrices[dim])) if np.isfinite(matrices[dim]).any() else 1.0
		if not np.isfinite(local_max) or local_max <= 0:
			local_max = 1.0
		local_norm = Normalize(vmin=0.0, vmax=local_max)
		local_mappable = ScalarMappable(norm=local_norm, cmap=cmap)
		sns.heatmap(
			df,
			ax=ax,
			cmap=cmap,
			vmin=0.0,
			vmax=local_max,
			square=True,
			cbar=False,
			mask=mask,
			linewidths=0.2,
			linecolor="#f2f2f2",
		)
		ax.set_title(f"{_t('Dimension')} {dim}")
		ax.set_xlabel(_t("Year"))
		ax.set_ylabel(_t("Year"))
		ax.tick_params(axis="x", rotation=45)
		ax.tick_params(axis="y", rotation=0)

		# Unique colorbar per subplot
		colorbar = fig.colorbar(
			local_mappable,
			ax=ax,
			fraction=0.046,
			pad=0.04,
		)
		if ax == plot_axes[-1]:  # Only label the last colorbar for cleanliness
			colorbar.set_label(
				f"{_t(metric.title())} {_t('distance')} (3er Trim)"
			)

	fig.suptitle(
		f"{_t('EPH Persistence Diagram Distance')} - {metric.title()} ({class_}, {weight_function}, {topo_method})",
		y=0.99,
	)

	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	class_ = snakemake.wildcards["class_"]
	weight_function = snakemake.wildcards["weight_function"]
	topo_method = snakemake.wildcards["topo_method"]
	metric = snakemake.wildcards["distance_diagrams"]
	translation = snakemake.config.get("translation", {})

	wave_labels, wave_diagrams = _load_waves()
	print(f"Loaded EPH waves in order: {', '.join(wave_labels)}")
	detected_dims = sorted(
		{
			dim
			for diagrams in wave_diagrams
			for dim in diagrams.keys()
		}
	)
	print(f"Detected persistence dimensions: {', '.join(str(dim) for dim in detected_dims) if detected_dims else 'none'}")
	for label, diagrams_by_dim in zip(wave_labels, wave_diagrams):
		missing_dims = [dim for dim in detected_dims if dim not in diagrams_by_dim]
		if missing_dims:
			print(f"Wave {label} is missing dimensions: {', '.join(str(dim) for dim in missing_dims)}")

	matrices = _compute_distance_matrices(wave_diagrams, wave_labels, metric)
	if not matrices:
		raise ValueError(
			"No persistence diagram heatmaps remain after applying the 5-year window and sparse-dimension filter."
		)
	_plot_faceted_heatmaps(
		matrices,
		wave_labels,
		metric,
		Path(snakemake.output[0]),
		class_,
		weight_function,
		topo_method,
		translation,
	)
	print(f"Saved {metric} heatmap to {snakemake.output[0]}")


if __name__ == "__main__":
	main()