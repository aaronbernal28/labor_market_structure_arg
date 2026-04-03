from scripts import *
import networkx as nx
import pandas as pd

snakemake: any


def main() -> None:
	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]
	pos_df = pd.read_csv(snakemake.input[0], dtype={id_col: int})
	graph = nx.read_gexf(snakemake.input[1], node_type=int)

	feature = getattr(snakemake.wildcards, "continuous_feature", None)
	if feature is None:
		raise ValueError("No feature wildcard found for projection gradient plot.")
	if feature not in pos_df.columns:
		raise ValueError(f"Feature '{feature}' not found in positions dataframe.")

	graph_nodes = set(graph.nodes())
	plot_df = pos_df[pos_df[id_col].astype(int).isin(graph_nodes)].copy()
	if plot_df.empty:
		raise ValueError("No overlap between nodelist ids and projection graph nodes.")

	pos = dl.load_positions(pos_df, id_col)
	node_values = pd.to_numeric(
		plot_df.set_index(id_col)[feature], errors="coerce"
	).to_dict()
	worker_counts = plot_df.set_index(id_col)["n_obs"].to_dict()

	FACTOR_NODE_SIZE = 0.6
	NODE_SIZE_EXPONENT = 0.8
	EDGE_ALPHA = 0.1
	NODE_ALPHA = 0.7

	pl.plot_projection_gradient(
		graph,
		pos=pos,
		node_values=node_values,
		title=None,
		colorbar_label=feature,
		cmap="viridis",
		figsize=snakemake.config["figsizes"]["projection"],
		font_size=snakemake.config["plot_font_size"],
		output_path=snakemake.output[0],
		save=True,
		node_size_map=worker_counts,
		factor_node_size=FACTOR_NODE_SIZE,
		node_size_exponent=NODE_SIZE_EXPONENT,
		edge_alpha=EDGE_ALPHA,
		node_alpha=NODE_ALPHA,
	)


if __name__ == "__main__":
	main()
