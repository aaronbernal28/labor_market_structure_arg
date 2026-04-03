from scripts import *
import networkx as nx
import pandas as pd

snakemake: any


def _resolve_community_column(df: pd.DataFrame, preferred: str | None) -> str:
	if preferred and preferred in df.columns:
		return preferred
	for candidate in ["community", "louvain", "leiden", "infomap"]:
		if candidate in df.columns:
			return candidate
	raise KeyError(
		"No community column found. Tried: "
		+ ", ".join([c for c in [preferred, "community", "louvain", "leiden", "infomap"] if c])
	)


def main() -> None:
	class_ = snakemake.wildcards["class_"]
	continuous_feature = snakemake.wildcards["continuous_feature"]
	algorithm = snakemake.wildcards.get("algorithm", None)

	id_col = snakemake.config[class_]["id"]
	pos_df = pd.read_csv(snakemake.input[0], dtype={id_col: int})
	graph = nx.read_gexf(snakemake.input[1], node_type=int)

	if continuous_feature not in pos_df.columns:
		raise KeyError(
			f"Continuous feature '{continuous_feature}' not found in {snakemake.input[0]}."
		)

	community_col = _resolve_community_column(pos_df, algorithm)
	if "n_obs" not in pos_df.columns:
		raise KeyError(f"Column 'n_obs' not found in {snakemake.input[0]}.")

	graph_nodes = set(graph.nodes())
	plot_df = pos_df[pos_df[id_col].astype(int).isin(graph_nodes)].copy()
	if plot_df.empty:
		raise ValueError("No overlap between nodelist ids and projection graph nodes.")

	feature_map = pd.to_numeric(
		plot_df.set_index(id_col)[continuous_feature], errors="coerce"
	).to_dict()
	community_map = (
		pd.to_numeric(plot_df.set_index(id_col)[community_col], errors="coerce")
		.fillna(-1)
		.astype(int)
		.to_dict()
	)
	node_size_map = plot_df.set_index(id_col)["n_obs"].to_dict()

	palette = snakemake.config["colors"]["community_palette"]
	color_map = {
		node_id: palette[community % len(palette)] if community >= 0 else "gray"
		for node_id, community in community_map.items()
	}

	figsize = snakemake.config["figsizes"]["edge_correlation"]
	font_size = snakemake.config["plot_font_size"]
	default_title = f"{class_.upper()} - {continuous_feature}"

	pl.compute_and_plot_edge_correlation(
		G=graph,
		feature_map=feature_map,
		color_map=color_map,
		community_map=community_map,
		node_size_map=node_size_map,
		title=default_title,
		output_path=snakemake.output[0],
		save=True,
		perfect_line=False,
		factor_node_size=1.2,
		node_size_exponent=0.8,
		figsize=figsize,
		font_size=font_size,
	)


if __name__ == "__main__":
	main()
