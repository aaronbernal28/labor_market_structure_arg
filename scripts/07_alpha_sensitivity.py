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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Return arrays of (nodes_with_edges, edge_counts, clustering, modularity, nodes_largest_cc) for each alpha."""
	nodes_with_edges = np.empty(len(alphas), dtype=float)
	edge_counts = np.empty(len(alphas), dtype=float)
	clustering_coeffs = np.empty(len(alphas), dtype=float)
	modularities = np.empty(len(alphas), dtype=float)
	nodes_largest_cc = np.empty(len(alphas), dtype=float)

	for i, alpha in enumerate(alphas):
		backbone = gc.disparity_filter_backbone(
			projection,
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
			_, mod = comm.louvain_partition(backbone, seed=seed)
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
	output_path = Path(snakemake.output[0])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	graph_metrics = metrics.summarize_graph(projection)

	reference_alpha = float(snakemake.params.get("alpha", 0.05))
	seed = int(snakemake.config["seed"])
	alphas = np.logspace(-4, 0, 30)
	alphas = np.unique(np.append(alphas, reference_alpha))
	alphas.sort()

	(
		nodes_with_edges,
		edge_counts,
		clustering_coeffs,
		modularities,
		nodes_largest_cc,
	) = _sweep_alpha(projection, alphas, seed)

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
		output_path=output_path,
		modularities=modularities,
		nodes_largest_cc=nodes_largest_cc,
		reference_alpha=reference_alpha,
		save=True,
		logscale=True,
	)

	print(f"Saved alpha sensitivity plot to {output_path}.")

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
