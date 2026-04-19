from pathlib import Path
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import src.topology as topo

snakemake: any


def _build_betas() -> np.ndarray:
	"""Return a strictly increasing beta grid including 0 and 1."""
	betas = np.concatenate(([0.0], np.logspace(-10, 0, 100)))
	betas = np.unique(betas)
	betas.sort()
	if betas[0] != 0.0:
		betas = np.concatenate(([0.0], betas))
	# Ensure 1.0 is present and is the last endpoint
	if betas[-1] != 1.0:
		betas = np.concatenate((betas, [1.0]))
		betas = np.unique(betas)
		betas.sort()
	betas[-1] = 1.0
	return betas


def _edge_distance_from_alpha(alpha_min: float, betas: np.ndarray) -> int:
	"""Compute d(u,v) = min{k : alpha < beta_k} with inclusive last beta."""
	k_max = len(betas) - 1
	# First index where beta_k > alpha_min
	k = int(np.searchsorted(betas, float(alpha_min), side="right"))
	# Inclusive behavior at beta_K = 1 for alpha=1
	return min(k, k_max)


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	betas = _build_betas()
	k_max = len(betas) - 1
	default_distance = float(k_max + 1)

	nodes = sorted(graph.nodes())
	node_index = {node: i for i, node in enumerate(nodes)}
	n_nodes = len(nodes)

	# Build a fully finite matrix (per preference: non-edges use a large constant)
	distance_matrix = np.full((n_nodes, n_nodes), default_distance, dtype=float)
	np.fill_diagonal(distance_matrix, 0.0)

	disparity_graph = gc.get_disparity_graph(graph)

	for u, v in graph.edges():
		a_uv = float(disparity_graph.edges[u, v].get("alpha", 1.0))
		a_vu = float(disparity_graph.edges[v, u].get("alpha", 1.0))
		alpha_min = min(a_uv, a_vu)
		d_uv = float(_edge_distance_from_alpha(alpha_min, betas))
		i = node_index[u]
		j = node_index[v]
		distance_matrix[i, j] = d_uv
		distance_matrix[j, i] = d_uv

	class_ = snakemake.wildcards.get("class_", "")
	weight_function = snakemake.wildcards.get("weight_function", "")
	dataset = snakemake.wildcards.get("dataset", "")
	main_title = " - ".join([x for x in [dataset, class_, weight_function] if x])

	# Sparse labels to keep heatmap readable
	n_labels = min(20, n_nodes)
	label_indices = np.linspace(0, n_nodes - 1, n_labels, dtype=int)
	labels = [""] * n_nodes
	max_caes_id = int(snakemake.config.get("max_caes_id", 10000))
	for idx in label_indices:
		node_id = int(nodes[idx])
		if class_ == "ciuo":
			labels[idx] = str(utils.original_ciuo_id(node_id, max_caes_id=max_caes_id))
		else:
			labels[idx] = str(node_id)

	# Compute persistence diagrams (thresh matches discrete scale)
	diagrams = topo.compute_persistence(
		distance_matrix,
		maxdim=2,
		thresh=float(default_distance),
		coeff=2,
	)

	fig, axs = plt.subplots(2, 2, figsize=(16, 12))
	fig.suptitle(main_title or "Persistence Diagram", y=0.98)

	pl.plot_distance_histogram(
		distance_matrix,
		title=f"Distance histogram d(u,v)\n(K={k_max}, non-edge={default_distance:g})",
		include_infinite=False,
		ax=axs[0, 0],
		save=False,
	)

	pl.plot_distance_heatmap(
		distance_matrix,
		title="Distance matrix d(u,v) (beta-index)",
		labels=labels,
		ax=axs[0, 1],
		save=False,
	)

	pl.plot_persistence_diagrams(
		diagrams,
		title="Persistence diagrams",
		ax=axs[1, 0],
		save=False,
	)

	pl.plot_persistence_barcodes(
		diagrams,
		title="Persistence barcodes",
		ax=axs[1, 1],
		save=False,
	)

	plt.tight_layout()
	fig.savefig(Path(snakemake.output[0]), bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	main()
