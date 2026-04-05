from scripts import *
import networkx as nx
import matplotlib.pyplot as plt
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
		+ ", ".join(
			[c for c in [preferred, "community", "louvain", "leiden", "infomap"] if c]
		)
	)


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	class_ = snakemake.wildcards["class_"]
	feature_name = snakemake.wildcards.get("feature") or snakemake.wildcards.get(
		"continuous_feature"
	)
	algorithm = snakemake.wildcards.get("algorithm", None)

	if not feature_name:
		raise KeyError(
			"No feature wildcard found (expected 'feature' or 'continuous_feature')."
		)

	id_col = snakemake.config[class_]["id"]
	pos_df = pd.read_csv(snakemake.input[0], dtype={id_col: int})

	# Cast nodes to int instantly to prevent string mismatch bugs
	graph = nx.read_gexf(snakemake.input[1], node_type=int)
	graph_metrics = metrics.summarize_graph(graph)

	if feature_name not in pos_df.columns:
		raise KeyError(
			f"Continuous feature '{feature_name}' not found in {snakemake.input[0]}."
		)

	community_col = _resolve_community_column(pos_df, algorithm)
	if "n_obs" not in pos_df.columns:
		raise KeyError(f"Column 'n_obs' not found in {snakemake.input[0]}.")

	graph_nodes = set(graph.nodes())
	plot_df = pos_df[pos_df[id_col].astype(int).isin(graph_nodes)].copy()
	if plot_df.empty:
		raise ValueError("No overlap between nodelist ids and projection graph nodes.")

	feature_map = pd.to_numeric(
		plot_df.set_index(id_col)[feature_name], errors="coerce"
	).to_dict()
	community_map = (
		pd.to_numeric(plot_df.set_index(id_col)[community_col], errors="coerce")
		.fillna(-1)
		.astype(int)
		.to_dict()
	)
	node_size_map = plot_df.set_index(id_col)["n_obs"].to_dict()

	from seaborn import hls_palette

	palette = hls_palette(len(set(community_map.values())), l=0.6).as_hex()
	color_map = {
		node_id: palette[community % len(palette)] if community >= 0 else "gray"
		for node_id, community in community_map.items()
	}

	default_title = f"{class_.upper()} - {feature_name}"

	pl.compute_and_plot_edge_correlation(
		G=graph,
		feature_map=feature_map,
		color_map=color_map,
		community_map=community_map,
		node_size_map=node_size_map,
		highlight_communities=set(community_map.values()),
		title=default_title,
		output_path=snakemake.output[0],
		save=True,
		perfect_line=False,
		factor_node_size=1.2,
		node_size_exponent=0.8,
		figsize=snakemake.config["figsizes"]["edge_correlation"],
	)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("EDGE WEIGHT CORRELATION")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PLOT SETTINGS",
		[
			f"Class: {class_}",
			f"Feature: {feature_name}",
			f"Community column: {community_col}",
			f"Communities: {len(set(community_map.values()))}",
		],
	)
	log.add_dataframe_info(
		log_lines,
		"NODELIST POSITIONS",
		row_count=len(pos_df),
		column_count=len(pos_df.columns),
	)
	log.add_graph_metrics(log_lines, "Projection metrics", graph_metrics)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	plt.close("all")


if __name__ == "__main__":
	main()
