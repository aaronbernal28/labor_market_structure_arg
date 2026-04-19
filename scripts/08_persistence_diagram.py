from pathlib import Path
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import src.topology as topo

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	topo_method = snakemake.wildcards.get("topo_method", "disparity_filtration")

	distance_matrix = gc.compute_distance_matrix(graph, method=topo_method)
	nodes = sorted(graph.nodes())
	n_nodes = len(nodes)

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

	default_distance = (
		np.max(distance_matrix) + 1
	)  # Use max distance as default threshold for persistence computation

	# Compute persistence diagrams (thresh matches discrete scale)
	diagrams = topo.compute_persistence(
		distance_matrix,
		maxdim=2,
		thresh=np.inf,
		coeff=2,
	)

	fig, axs = plt.subplots(2, 2, figsize=(16, 12))
	fig.suptitle("Persistence Diagram", y=0.98)

	pl.plot_distance_histogram(
		distance_matrix,
		title=f"Distance histogram d(u,v)\n(non-edge={default_distance:g})",
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
