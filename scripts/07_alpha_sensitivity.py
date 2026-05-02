from typing import Any
from pathlib import Path
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	with open(snakemake.input[0], "r") as f:
		metrics = json.load(f)

	alphas = np.array(metrics["alphas"])
	nodes_with_edges = np.array(metrics["nodes_with_edges"])
	edge_counts = np.array(metrics["edge_counts"])
	clustering_coeffs = np.array(metrics["clustering_coeffs"])
	modularities = np.array(metrics["modularities"])
	nodes_largest_cc = np.array(metrics["nodes_largest_cc"])
	reference_alpha = metrics["reference_alpha"]

	title = (
		f"{snakemake.wildcards['dataset']} - "
		f"{snakemake.wildcards['class_']} - "
		f"{snakemake.wildcards['weight_function']}"
	)

	pl.plot_alpha_sensitivity(
		alphas=alphas,
		nodes_with_edges=nodes_with_edges,
		edge_counts=edge_counts,
		clustering_coefficients=clustering_coeffs,
		title=title,
		output_path=Path(snakemake.output[0]),
		modularities=modularities,
		nodes_largest_cc=nodes_largest_cc,
		reference_alpha=reference_alpha,
		save=True,
		logscale=True,
	)

	print(f"Saved alpha sensitivity plot to {Path(snakemake.output[0])}.")

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("ALPHA SENSITIVITY")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SWEEP SETTINGS",
		[
			f"Reference alpha: {reference_alpha}",
			f"Alpha min: {alphas.min():.6f}",
			f"Alpha max: {alphas.max():.6f}",
			f"Alpha count: {len(alphas)}",
			f"Modularity min: {modularities.min():.4f}",
			f"Modularity max: {modularities.max():.4f}",
		],
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
