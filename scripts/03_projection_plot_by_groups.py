from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]
	pos_df = pd.read_csv(snakemake.input[0], dtype={id_col: int})
	max_group_code = snakemake.config["community"]["max"][class_]
	pos_df = pos_df[pos_df["community"] < utils.label_fn(max_group_code, len(str(max_group_code)))].copy()

	graph = nx.read_gexf(snakemake.input[1], node_type=int)
	graph_metrics = metrics.summarize_graph(graph)
	graph_nodes = set(graph.nodes())
	plot_df = pos_df[pos_df[id_col].astype(int).isin(graph_nodes)].copy()
	if plot_df.empty:
		raise ValueError("No overlap between nodelist ids and projection graph nodes.")
	pos = dl.load_positions(pos_df, id_col)
	discrete_feature = snakemake.wildcards["discrete_feature"]

	if discrete_feature in plot_df.columns:
		pass
	elif snakemake.config[class_].get(discrete_feature, None) in plot_df.columns:
		discrete_feature = snakemake.config[class_][discrete_feature]
	else:
		raise ValueError(
			f"Discrete feature '{discrete_feature}' not found in positions dataframe."
		)
	group_map = {
		int(node): group
		for node, group in plot_df.set_index(id_col)[discrete_feature].dropna()
		.to_dict()
		.items()
	}
	group_color_map = None
	preferred_color_col = f"{discrete_feature}_color"
	if preferred_color_col in plot_df.columns:
		color_col = preferred_color_col
	else:
		color_col = next(
			(
				column
				for column in plot_df.columns
				if discrete_feature in column and column.endswith("_color")
			),
			None,
		)

	if color_col:
		pairs = plot_df[[discrete_feature, color_col]].dropna().drop_duplicates()
		raw_group_color_map = dict(zip(pairs[discrete_feature], pairs[color_col]))
		group_color_map = {}
		for group_name, color_value in raw_group_color_map.items():
			try:
				parsed_color = utils.parse_color(color_value)
				group_color_map[group_name] = mcolors.to_hex(parsed_color)
			except (ValueError, SyntaxError, TypeError):
				group_color_map[group_name] = "gray"

	print(f"Using discrete feature '{discrete_feature}' for grouping.")
	print(f"Group mapping: {set(group_map.values())}")
	print(
		f"Group color mapping: {set(group_color_map.values()) if group_color_map else 'None'}"
	)

	if not group_color_map:
		print(
			f"Warning: No color mapping found for discrete feature '{discrete_feature}'. Using default colors."
		)
		from seaborn import hls_palette

		unique_groups = sorted(set(group_map.values()))
		palette = hls_palette(len(unique_groups), l=0.6).as_hex()
		group_color_map = {
			group: palette[i % len(palette)] for i, group in enumerate(unique_groups)
		}

	if pos:
		graph = nx.subgraph(graph, set(pos.keys()))
	else:
		print("Warning: No positions found; plotting without position filter.")

	worker_counts = {
		int(node): count
		for node, count in plot_df.set_index(id_col)["n_obs"].to_dict().items()
	}

	_ = pl.plot_projection_by_group(
		graph,
		group_map=group_map,
		group_color_map=group_color_map,
		title=None,
		legend_title=discrete_feature,
		figsize=snakemake.config["figsizes"]["projection"],
		output_path=snakemake.output[0],
		save=True,
		method="energy",
		node_size_map=worker_counts,
		factor_node_size=snakemake.config["FACTOR_NODE_SIZE"],
		pos=pos,
		node_size_exponent=snakemake.config["NODE_SIZE_EXPONENT"],
		edge_alpha=snakemake.config["EDGE_ALPHA"][class_],
		node_alpha=snakemake.config["NODE_ALPHA"],
	)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PROJECTION PLOT BY GROUPS")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PLOT SETTINGS",
		[
			f"Class: {class_}",
			f"Discrete feature: {discrete_feature}",
			f"Groups: {len(set(group_map.values()))}",
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
