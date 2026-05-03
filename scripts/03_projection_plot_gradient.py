from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]
	pos_df = pd.read_csv(snakemake.input[0], dtype={id_col: int})
	graph = nx.read_gexf(snakemake.input[1], node_type=int)
	graph_metrics = metrics.summarize_graph(graph)

	feature = getattr(snakemake.wildcards, "continuous_feature", None)
	if feature is None:
		feature = getattr(snakemake.wildcards, "discrete_feature", None)
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

	cmaps = snakemake.config.get("cmaps", {})
	cmap_name = cmaps.get(feature, cmaps.get("default", "viridis"))

	pl.plot_projection_gradient(
		graph,
		pos=pos,
		node_values=node_values,
		title=None,
		colorbar_label=feature,
		cmap=str(cmap_name),
		figsize=snakemake.config["figsizes"]["projection"],
		output_path=snakemake.output[0],
		save=True,
		node_size_map=worker_counts,
		factor_node_size=snakemake.config["FACTOR_NODE_SIZE"],
		node_size_exponent=snakemake.config["NODE_SIZE_EXPONENT"],
		edge_alpha=snakemake.config["EDGE_ALPHA"][class_],
		node_alpha=snakemake.config["NODE_ALPHA"],
	)

	values_series = pd.Series(list(node_values.values()))
	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PROJECTION PLOT GRADIENT")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PLOT SETTINGS",
		[
			f"Class: {class_}",
			f"Feature: {feature}",
			f"Non-null values: {int(values_series.notna().sum())}",
			f"Min value: {values_series.min():.4f}"
			if values_series.notna().any()
			else "Min value: N/A",
			f"Max value: {values_series.max():.4f}"
			if values_series.notna().any()
			else "Max value: N/A",
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


if __name__ == "__main__":
	main()
