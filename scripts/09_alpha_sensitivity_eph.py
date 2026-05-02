from pathlib import Path
from typing import Any
import json
from scripts import *
import matplotlib.pyplot as plt
import numpy as np

snakemake: Any


def _extract_eph_file(metrics: dict[str, Any], input_path: Path) -> str:
	eph_file = metrics.get("eph_file")
	if isinstance(eph_file, str) and eph_file:
		return eph_file

	# Expected path pattern:
	# data/processed/{dataset}/{class_}/_alpha_sensitivity/_{weight_function}_{algorithm}.json
	parts = list(input_path.parts)
	if "processed" in parts:
		idx = parts.index("processed")
		if idx + 1 < len(parts):
			return parts[idx + 1]

	return input_path.stem


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	class_ = snakemake.wildcards["class_"]
	weight_function = snakemake.wildcards["weight_function"]

	input_paths = [Path(p) for p in snakemake.input]

	# Load all JSON results and assign an eph_file key for ordering.
	results_with_keys: list[tuple[str, dict[str, Any]]] = []
	for p in input_paths:
		with open(p, "r") as f:
			metrics_data = json.load(f)
		eph_file = _extract_eph_file(metrics_data, p)
		results_with_keys.append((eph_file, metrics_data))

	# Ensure unique mapping eph_file -> data (keep first occurrence)
	# and sort chronologically
	eph_to_data: dict[str, dict[str, Any]] = {}
	for eph_file, metrics_data in results_with_keys:
		eph_to_data.setdefault(eph_file, metrics_data)

	eph_files_sorted = utils.sort_eph_files(list(eph_to_data.keys()))
	results_sorted = [eph_to_data[f] for f in eph_files_sorted]

	n_series = len(results_sorted)
	if n_series == 0:
		raise ValueError("No input metrics found.")

	# Assume alphas are consistent across all runs
	alphas = np.array(results_sorted[0]["alphas"])
	n_alphas = len(alphas)

	nodes_with_edges_all = np.empty((n_series, n_alphas), dtype=float)
	edge_counts_all = np.empty((n_series, n_alphas), dtype=float)
	clustering_all = np.empty((n_series, n_alphas), dtype=float)
	nodes_lcc_all = np.empty((n_series, n_alphas), dtype=float)
	reference_alphas = np.empty(n_series, dtype=float)

	graph_metrics: dict[str, dict] = {}

	for i, (eph_file, metrics_data) in enumerate(zip(eph_files_sorted, results_sorted)):
		series_alphas = np.array(metrics_data["alphas"])
		if len(series_alphas) != n_alphas or not np.allclose(series_alphas, alphas):
			raise ValueError(
				f"Alpha grid mismatch for {eph_file}: expected {n_alphas} values from first series."
			)

		nodes_with_edges = np.array(metrics_data["nodes_with_edges"])
		edge_counts = np.array(metrics_data["edge_counts"])
		clustering_coeffs = np.array(metrics_data["clustering_coeffs"])
		nodes_largest_cc = np.array(metrics_data["nodes_largest_cc"])
		reference_alpha = round(float(metrics_data["reference_alpha"]), 4)

		nodes_with_edges_all[i, :] = nodes_with_edges
		edge_counts_all[i, :] = edge_counts
		clustering_all[i, :] = clustering_coeffs
		nodes_lcc_all[i, :] = nodes_largest_cc
		reference_alphas[i] = reference_alpha

		gm = metrics_data.get("graph_metrics")
		if isinstance(gm, dict):
			graph_metrics[eph_file] = gm

	title = f"EPH - {class_} - {weight_function}"
	pl.plot_alpha_sensitivity_multi_series(
		alphas=alphas,
		series_labels=eph_files_sorted,
		nodes_with_edges=nodes_with_edges_all,
		edge_counts=edge_counts_all,
		clustering_coefficients=clustering_all,
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
	log_lines.append("ALPHA SENSITIVITY (EPH - ALL FILES - PARALLEL)")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SWEEP SETTINGS",
		[
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
			f"  {j:>3d}. {eph_file} | reference_alpha={reference_alphas[j]:.3g}"
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
