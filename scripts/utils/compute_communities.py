from scripts import *
import networkx as nx
import pandas as pd

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
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
	nodelist_df = pd.read_csv(snakemake.input[1], dtype={id_col: int})
	if id_col not in nodelist_df.columns:
		raise KeyError(f"Missing '{id_col}' column in {class_}_{dataset}.csv.")

	communities_int = {int(node): int(comm) for node, comm in communities.items()}
	nodelist_df[algorithm] = (
		nodelist_df[id_col].astype(int).map(communities_int).fillna(-1).astype(int)
	)
	nodelist_df.to_csv(snakemake.output[0], index=False)
	print(f"Saved {class_}_{dataset} communities to {snakemake.output[0]}.")

	# TODO: Add community distribution plot
	# pl.plot_stacked_by_group(
	# nodelist_ciuo_df,
	# group_col=group_col,
	# community_map=communities_ciuo,
	# title=None,
	# output_path=snakemake.output[0],
	# group_color_map=group_color_map,
	# legend_title="1D categorias CIUO",
	# figsize=cfg.STACKED_FIGSIZE,
	# font_size=cfg.PLOT_FONT_SIZE,
	# save=True,
	# percentage=False
	# )


if __name__ == "__main__":
	main()
