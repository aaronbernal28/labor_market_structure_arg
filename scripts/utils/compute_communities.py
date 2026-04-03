from scripts import *
import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	dataset = snakemake.wildcards["dataset"]
	seed = int(snakemake.config["seed"])

	algorithm = snakemake.params["algorithm"].lower()

	if algorithm == "louvain":
		algorithm_func = comm.best_louvain_partition_random
	else:
		raise NotImplementedError(
			"Leiden is not implemented in src.communities yet. Use 'louvain'."
		)

	communities, modularity, best_resolution = algorithm_func(graph, seed=seed)
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

	group_col = snakemake.config[class_].get("letra")
	group_color_col = snakemake.config[class_].get("letra_color")

	nodelist_idx = nodelist_df.set_index(id_col)

	group_color_map = None
	if group_color_col and group_color_col in nodelist_idx.columns:
		raw_group_color_map = (
			nodelist_idx.groupby(group_col)[group_color_col].first().to_dict()
		)
		group_color_map = {}
		for group_name, color_value in raw_group_color_map.items():
			try:
				parsed_color = utils.parse_color(color_value)
				group_color_map[group_name] = mcolors.to_hex(parsed_color)
			except (ValueError, SyntaxError, TypeError):
				group_color_map[group_name] = "gray"

	pl.plot_stacked_by_group(
		df_index=nodelist_idx,
		group_col=group_col,
		community_map=communities_int,
		title=f"{class_.upper()} - Distribucion por comunidad ({algorithm})",
		output_path=snakemake.output[1],
		group_color_map=group_color_map,
		legend_title=group_col,
		figsize=tuple(snakemake.config["figsizes"]["stacked"]),
		font_size=int(snakemake.config["plot_font_size"]),
		save=True,
		percentage=False,
	)


if __name__ == "__main__":
	main()
