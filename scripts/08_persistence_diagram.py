from typing import Any
from pathlib import Path
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import src.topology as topo

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	topo_method = snakemake.wildcards.get("topo_method", "disparity_filtration")
	translation = snakemake.config.get("translation", {})

	def _t(label: str) -> str:
		return utils.translate_label(label, translation)

	distance_matrix = gc.compute_distance_matrix(
		graph, method=topo_method, resolution=100
	)
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

	diameter = np.max(distance_matrix[np.isfinite(distance_matrix)])
	print(f"[DEBUG] Distance matrix computed with method '{topo_method}'.")
	print(f"[DEBUG] Distance matrix shape: {distance_matrix.shape}")
	print(f"[DEBUG] Distance matrix diameter: {diameter:.4f}")
	thresh = diameter
	print(f"[DEBUG] Using threshold: {thresh:.4f}")

	# Compute persistence diagrams (thresh matches discrete scale)
	diagrams = topo.compute_persistence(
		distance_matrix,
		maxdim=2,
		thresh=thresh,
		coeff=2,
		n_perm=None,
	)

	fig, axs = plt.subplots(1, 2, figsize=(12, 6))
	fig.suptitle("")

	# pl.plot_distance_histogram(
	# 	distance_matrix,
	# 	title="Distance histogram d(u,v)",
	# 	include_infinite=False,
	# 	ax=axs[0],
	# 	save=False,
	# )

	pl.plot_persistence_diagrams(
		diagrams,
		title="",
		ax=axs[0],
		save=False,
		translation=translation,
	)

	pl.plot_distance_heatmap(
		distance_matrix,
		title="",
		labels=labels,
		x_label=_t(class_),
		y_label=_t(class_),
		ax=axs[1],
		save=False,
		translation=translation,
	)

	# pl.plot_persistence_barcodes(
	# 	diagrams,
	# 	title="Persistence barcodes",
	# 	ax=axs[1],
	# 	save=False,
	# )

	plt.tight_layout()
	fig.savefig(Path(snakemake.output[0]), bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	main()
