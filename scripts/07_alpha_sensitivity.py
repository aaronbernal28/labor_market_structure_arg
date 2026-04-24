from pathlib import Path

from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	projection = nx.read_gexf(snakemake.input[0], node_type=int)
	graph_metrics = metrics.summarize_graph(projection)
	algorithm = snakemake.wildcards["algorithm"]

	seed = int(snakemake.config["seed"])
	alphas = np.logspace(-10, 0, 60)
	alphas.sort()

	(
		nodes_with_edges,
		edge_counts,
		clustering_coeffs,
		modularities,
		nodes_largest_cc,
	) = gc.compute_sweep_alpha(projection, alphas, seed, algorithm=algorithm)

	# Find the minimum alpha where nodes in largest CC > 95%
	reference_alpha = None
	for i, a in enumerate(alphas):
		if nodes_largest_cc[i] > 0.95:
			reference_alpha = a
			break
	if reference_alpha is None:
		reference_alpha = alphas[-1]

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
	log.add_graph_metrics(log_lines, "Projection metrics", graph_metrics)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
