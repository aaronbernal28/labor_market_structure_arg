from pathlib import Path

from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

snakemake: any


def _sweep_alpha(
	projection: nx.Graph,
	alphas: np.ndarray,
	seed: int,
	algorithm: str = "louvain",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Return arrays of (nodes_with_edges, edge_counts, clustering, modularity, nodes_largest_cc) for each alpha."""
	if algorithm == "louvain":
		algorithm_func = comm.best_louvain_partition_random
	elif algorithm == "leiden":
		algorithm_func = comm.best_leiden_partition_random
	elif algorithm == "infomap":
		algorithm_func = comm.best_infomap_partition_random
	else:
		raise NotImplementedError(
			"Unsupported algorithm. Use one of: louvain, leiden, infomap."
		)

	nodes_with_edges = np.empty(len(alphas), dtype=float)
	edge_counts = np.empty(len(alphas), dtype=float)
	clustering_coeffs = np.empty(len(alphas), dtype=float)
	modularities = np.empty(len(alphas), dtype=float)
	nodes_largest_cc = np.empty(len(alphas), dtype=float)

	disparity_graph = gc.get_disparity_graph(
		projection
	)  # Precompute disparity graph once for efficiency

	for i, alpha in enumerate(alphas):
		# FIXME: This is inefficient since it recomputes the disparity graph each time. Refactor to compute once and pass in.
		backbone = gc.disparity_filter_backbone(
			disparity_graph=disparity_graph,
			alpha=float(alpha),
			mode="or",
			keep_isolates=True,
		)

		n_edges_base = max(1, projection.number_of_edges())
		n_nodes_base = max(1, projection.number_of_nodes())
		edge_counts[i] = backbone.number_of_edges() / n_edges_base
		nodes_with_edges[i] = (
			sum(1 for n in backbone.nodes() if backbone.degree(n) > 0) / n_nodes_base
		)

		if backbone.number_of_nodes() > 0 and backbone.number_of_edges() > 0:
			largest_cc = max(nx.connected_components(backbone), key=len)
			nodes_largest_cc[i] = len(largest_cc) / n_nodes_base
		else:
			nodes_largest_cc[i] = 0.0

		if backbone.number_of_edges() > 0:
			clustering_coeffs[i] = nx.average_clustering(backbone, weight="weight")
			_, mod, _ = algorithm_func(backbone, seed=seed, n_samples=10)
			modularities[i] = mod
		else:
			clustering_coeffs[i] = 0.0
			modularities[i] = 0.0

	return (
		nodes_with_edges,
		edge_counts,
		clustering_coeffs,
		modularities,
		nodes_largest_cc,
	)


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
	) = _sweep_alpha(projection, alphas, seed, algorithm=algorithm)

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
