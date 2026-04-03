from scripts import *
import networkx as nx
import pandas as pd

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])
	class_ = snakemake.wildcards["class_"]
	dataset = snakemake.wildcards["dataset"]

	algorithm = snakemake.params["algorithm"].lower()

	if algorithm == "louvain":
		algorithm_func = comm.best_louvain_partition_random
	else:
		raise NotImplementedError(
			"Leiden is not implemented in src.communities yet. Use 'louvain'."
		)

	communities, modularity, best_resolution = algorithm_func(graph)
	num_communities = len(set(communities.values()))
	print(f"Modularity score: {modularity:.4f}")
	print(f"Best resolution: {best_resolution:.3f}")
	print(f"Detected communities: {num_communities}")

	id_col = snakemake.config[class_]["id"]
	nodelist_df = pd.read_csv(snakemake.input[1])
	if id_col not in nodelist_df.columns:
		raise KeyError(f"Missing '{id_col}' column in {class_}_{dataset}.csv.")

	communities_str = {str(node): int(comm) for node, comm in communities.items()}
	nodelist_df["community"] = (
		nodelist_df[id_col].astype(str).map(communities_str).fillna(-1).astype(int)
	)
	nodelist_df.to_csv(snakemake.output[0], index=False)
	print(f"Saved {class_}_{dataset} communities to {snakemake.output[0]}.")


if __name__ == "__main__":
	main()
