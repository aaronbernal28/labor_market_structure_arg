from typing import Any
from scripts import *
import networkx as nx
import pandas as pd

snakemake: Any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	dataset = snakemake.wildcards["dataset"]
	alpha = float(snakemake.wildcards.get("alpha", 0.05))
	seed = int(snakemake.config["seed"])
	resolution = float(
		snakemake.config["community"]["resolution"].get(f"{alpha:.2f}", {}).get(class_, 1.0)
	)

	id_col = snakemake.config[class_]["id"]
	nodelist_df = pd.read_csv(snakemake.input[1], dtype={id_col: int})
	if id_col not in nodelist_df.columns:
		raise KeyError(f"Missing '{id_col}' column in {class_}_{dataset}.csv.")
	graph = graph.subgraph(nodelist_df[id_col].unique()).copy()

	algorithm = snakemake.wildcards["algorithm"].lower()
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

	# Sort graph nodes by ID to ensure consistent ordering across runs
	graph = gc.graph_sort_nodes_by_id(graph)

	communities, modularity = algorithm_func(
		graph, seed=seed, n_samples=100, resolution=resolution
	)

	communities = utils.filter_communities_by_size(communities, min_size=1)
	num_communities = len(set(communities.values()))

	communities_int = {
		int(node): utils.label_fn(comm, len(str(num_communities)))
		for node, comm in communities.items()
	}
	nodelist_df["community"] = nodelist_df[id_col].astype(int).map(communities_int)

	# Save only the global ID column and community
	output_df = nodelist_df[[id_col, "community"]].copy()
	output_df.to_csv(snakemake.output[0], index=False)
	print(f"Saved lightweight {class_}_{dataset} communities to {snakemake.output[0]}.")


if __name__ == "__main__":
	main()
