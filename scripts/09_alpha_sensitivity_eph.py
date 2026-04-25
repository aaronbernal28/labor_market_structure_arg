from pathlib import Path
from typing import Any

from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from src.seeding import initialize_seeds, get_seed_from_config

snakemake: Any


def _reference_alpha_from_lcc(
	alphas: np.ndarray, nodes_largest_cc: np.ndarray
) -> float:
	ref = None
	for i, a in enumerate(alphas):
		if nodes_largest_cc[i] > 0.95:
			ref = float(a)
			break
	if ref is None:
		ref = float(alphas[-1])
	return ref


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	seed = get_seed_from_config(snakemake.config)
	initialize_seeds(seed)

	algorithm = snakemake.wildcards["algorithm"]
	class_ = snakemake.wildcards["class_"]
	weight_function = snakemake.wildcards["weight_function"]

	seed = int(snakemake.config["seed"])
	alphas = np.logspace(-10, 0, 60)
	alphas.sort()

	input_paths = [Path(p) for p in snakemake.input]
	eph_files_raw = [utils.extract_eph_file_from_path(p) for p in input_paths]

	# Ensure unique mapping eph_file -> path (keep first occurrence)
	eph_to_path = {}
	for eph_file, p in zip(eph_files_raw, input_paths, strict=False):
		eph_to_path.setdefault(eph_file, p)

	eph_files_sorted = utils.sort_eph_files(list(eph_to_path.keys()))

	n_series = len(eph_files_sorted)
	n_alphas = len(alphas)

	nodes_with_edges_all = np.empty((n_series, n_alphas), dtype=float)
	edge_counts_all = np.empty((n_series, n_alphas), dtype=float)
	clustering_all = np.empty((n_series, n_alphas), dtype=float)
	modularities_all = np.empty((n_series, n_alphas), dtype=float)
	nodes_lcc_all = np.empty((n_series, n_alphas), dtype=float)
	reference_alphas = np.empty(n_series, dtype=float)

	graph_metrics: dict[str, dict] = {}

	for i, eph_file in enumerate(eph_files_sorted):
		# NOTE: this block must be ruled to improve efficiency and take advantage of parallelization
		projection_path = eph_to_path[eph_file]
		projection = nx.read_gexf(projection_path, node_type=int)
		graph_metrics[eph_file] = metrics.summarize_graph(projection)

		(
			nodes_with_edges,
			edge_counts,
			clustering_coeffs,
			modularities,
			nodes_largest_cc,
		) = gc.compute_sweep_alpha(projection, alphas, seed, algorithm=algorithm)

		nodes_with_edges_all[i, :] = nodes_with_edges
		edge_counts_all[i, :] = edge_counts
		clustering_all[i, :] = clustering_coeffs
		modularities_all[i, :] = modularities
		nodes_lcc_all[i, :] = nodes_largest_cc
		reference_alphas[i] = _reference_alpha_from_lcc(alphas, nodes_largest_cc)

	title = f"EPH - {class_} - {weight_function} ({algorithm})"
	pl.plot_alpha_sensitivity_multi_series(
		alphas=alphas,
		series_labels=eph_files_sorted,
		nodes_with_edges=nodes_with_edges_all,
		edge_counts=edge_counts_all,
		clustering_coefficients=clustering_all,
		modularities=modularities_all,
		nodes_largest_cc=nodes_lcc_all,
		title=title,
		output_path=Path(snakemake.output[0]),
		reference_alphas=reference_alphas,
		save=True,
		logscale=True,
	)

	print(f"Saved EPH alpha sensitivity plot to {Path(snakemake.output[0])}.")

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("ALPHA SENSITIVITY (EPH - ALL FILES)")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SWEEP SETTINGS",
		[
			f"Algorithm: {algorithm}",
			f"Alpha min: {alphas.min():.6f}",
			f"Alpha max: {alphas.max():.6f}",
			f"Alpha count: {len(alphas)}",
			f"EPH series count: {n_series}",
		],
	)
	log_lines.append("")
	log_lines.append("EPH FILES (chronological order):")
	for j, eph_file in enumerate(eph_files_sorted):
		log_lines.append(
			f"  {j:>3d}. {eph_file} | reference_alpha={reference_alphas[j]:.3g} | "
			f"modularity_range=[{modularities_all[j, :].min():.4f}, {modularities_all[j, :].max():.4f}]"
		)

	# Add a small per-file graph summary (first few keys) to keep logs readable
	log_lines.append("")
	log_lines.append("PER-FILE GRAPH METRICS (projection summaries):")
	for eph_file in eph_files_sorted:
		gm = graph_metrics.get(eph_file, {})
		items = []
		for k in ["n_nodes", "n_edges", "density", "avg_degree"]:
			if k in gm:
				items.append(f"{k}={gm[k]}")
		log_lines.append(
			f"  - {eph_file}: " + (", ".join(items) if items else "(no metrics)")
		)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
