from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors
from networkx.algorithms.community import modularity

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]
	translation = snakemake.config.get("translation", {})
	pos_df = pd.read_csv(snakemake.input[0], dtype={id_col: int})

	graph = nx.read_gexf(snakemake.input[1], node_type=int)
	#graph_metrics = metrics.summarize_graph(graph)
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

	if discrete_feature == "community":
		plot_df = dl.filter_communities(
			plot_df,
			feature_col=discrete_feature,
			max_code=snakemake.config["community"]["max"].get(class_, 98),
		)

	group_map = {
		int(node): group
		for node, group in plot_df.set_index(id_col)[discrete_feature].to_dict().items()
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

	if discrete_feature == "community" and group_color_map is not None:
		# Always render non-community nodes in gray.
		group_color_map.setdefault("Other", "gray")

	print(f"Using discrete feature '{discrete_feature}' for grouping.")
	print(f"Group mapping: {set(group_map.values())}")
	print(
		f"Group color mapping: {set(group_color_map.values()) if group_color_map else 'None'}"
	)

	if not group_color_map:
		print(
			f"Warning: No color mapping found for discrete feature '{discrete_feature}'. Using default colors."
		)
		unique_groups = sorted(set(group_map.values()))
		group_color_map = utils.build_community_color_map(
			unique_groups,
			other_label="Other",
		)

	if pos:
		graph = nx.subgraph(graph, set(pos.keys()))
	else:
		print("Warning: No positions found; plotting without position filter.")

	# Compute group-based modularity metrics on the same graph used for plotting.
	group_to_nodes: dict[str, set[int]] = {}
	for node, group in group_map.items():
		if node in graph:
			group_to_nodes.setdefault(str(group), set()).add(int(node))
	communities_list = [nodes for nodes in group_to_nodes.values() if nodes]
	classic_modularity = None
	if communities_list:
		classic_modularity = modularity(graph, communities_list, weight="weight")

	exclude_group_label = "2. Trabajadores de servicios y ventas"
	classic_modularity_filtered = None
	if discrete_feature == "ciuo3cat":
		# Compute classic modularity excluding a specific CIUO3CAT group.
		normalized_exclude_label = " ".join(exclude_group_label.split())
		filtered_group_to_nodes: dict[str, set[int]] = {}
		for group_label, nodes in group_to_nodes.items():
			normalized_label = " ".join(str(group_label).split())
			if normalized_label != normalized_exclude_label:
				filtered_group_to_nodes[group_label] = nodes
		filtered_nodes = (
			set().union(*filtered_group_to_nodes.values())
			if filtered_group_to_nodes
			else set()
		)
		filtered_graph = (
			graph.subgraph(filtered_nodes).copy() if filtered_nodes else nx.Graph()
		)
		filtered_communities_list = [
			nodes for nodes in filtered_group_to_nodes.values() if nodes
		]
		if filtered_communities_list and filtered_graph.number_of_nodes() > 0:
			classic_modularity_filtered = modularity(
				filtered_graph,
				filtered_communities_list,
				weight="weight",
			)

	local_modularity_rows: list[str] = []
	for group_label in sorted(group_to_nodes.keys()):
		nodes = group_to_nodes[group_label]
		local_modularity = comm.local_modularity_weighted(
			graph, nodes, gamma=1.0
		)
		local_modularity_rows.append(
			f"{group_label}: n={len(nodes)}, local_modularity={local_modularity:.6f}"
		)

	_ = pl.plot_projection_by_group(
		graph,
		group_map=group_map,
		group_color_map=group_color_map,
		title=None,
		legend_title=utils.translate_label(discrete_feature, translation),
		figsize=snakemake.config["figsizes"]["projection"],
		output_path=snakemake.output[0],
		save=True,
		method="energy",
		factor_node_size=snakemake.config["FACTOR_NODE_SIZE"][class_],
		pos=pos,
		edge_alpha=snakemake.config["EDGE_ALPHA"][class_],
		node_alpha=snakemake.config["NODE_ALPHA"],
		translation=translation,
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
	#log.add_graph_metrics(log_lines, "Projection metrics", graph_metrics)
	log.add_notes(
		log_lines,
		"GROUP MODULARITY (WEIGHTED)",
		[
			f"Discrete feature: {discrete_feature}",
			f"Modularity (partition): {classic_modularity:.6f}"
			if classic_modularity is not None
			else "Classic modularity (partition): N/A",
			f"Modularity (partition, exclude '{exclude_group_label}'): {classic_modularity_filtered:.6f}"
			if discrete_feature == "ciuo3cat" and classic_modularity_filtered is not None
			else f"Classic modularity (partition, exclude '{exclude_group_label}'): N/A",
		],
	)
	if local_modularity_rows:
		log.add_notes(
			log_lines,
			"LOCAL MODULARITY BY GROUP",
			local_modularity_rows,
		)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
