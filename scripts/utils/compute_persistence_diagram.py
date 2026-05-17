from typing import Any
from pathlib import Path
from scripts import *
import networkx as nx
import numpy as np
import pandas as pd

import src.topology as topo

snakemake: Any


def main() -> None:
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

	# Compute persistence diagrams (thresh matches discrete scale)
	diagrams = topo.compute_persistence(
		distance_matrix,
		maxdim=1,
		thresh=100,
		coeff=2,
	)
	feature_counts = [len(dgm) for dgm in diagrams]
	message_parts = [
		f"{count} {dim}-dimensional features" for dim, count in enumerate(feature_counts)
	]
	print(f"Computed persistence diagram with {', '.join(message_parts)}.")

	loader = topo.DGMS_loader(snakemake.output[0])
	loader.export(diagrams)


if __name__ == "__main__":
	main()
