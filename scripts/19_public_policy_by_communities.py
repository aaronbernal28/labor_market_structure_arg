from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from scripts import *

snakemake: Any


def _format_label(
	value: object,
	class_index: int,
	max_caes_id: int | None,
) -> str:
	if value is None:
		return ""
	if isinstance(value, float) and pd.isna(value):
		return ""
	if isinstance(value, (int, float)):
		try:
			return str(utils.original_id(int(value), class_index, max_caes_id))
		except Exception:
			return str(value)
	value_str = str(value).strip()
	if value_str.isdigit():
		try:
			return str(utils.original_id(int(value_str), class_index, max_caes_id))
		except Exception:
			return value_str
	return value_str


def _build_label_map(
	nodelist: pd.DataFrame,
	id_col: str,
	label_col: str | None,
	class_index: int,
	max_caes_id: int | None,
) -> dict[int, str]:
	labels: dict[int, str] = {}
	for _, row in nodelist.iterrows():
		try:
			node_id = int(row[id_col])
		except Exception:
			continue
		value = (
			row[label_col]
			if label_col and label_col in nodelist.columns
			else row[id_col]
		)
		labels[node_id] = _format_label(value, class_index, max_caes_id) or str(node_id)
	return labels


def main() -> None:
	import matplotlib.colors as mcolors
	import numpy as np

	plt.style.use("src/styles/publication.mplstyle")
	translation = snakemake.config.get("translation", {})

	def _t(label: str) -> str:
		return utils.translate_label(label, translation)

	dataset = snakemake.wildcards["dataset"]
	class_ = snakemake.wildcards["class_"]
	c1 = snakemake.wildcards["c1"]
	c2 = snakemake.wildcards["c2"]
	if c1 is None or c2 is None:
		raise ValueError("Both community wildcards (c1, c2) must be provided.")

	id_col = snakemake.config[class_]["id"]
	label_col = snakemake.config[class_].get("label")
	class_index = int(snakemake.config[class_].get("partition", 1))
	max_caes_id = snakemake.config.get("max_caes_id")

	nodelist = pd.read_csv(snakemake.input[1], dtype={id_col: int})

	community_col = "community"
	nodelist = dl.filter_communities(
		nodelist,
		feature_col=community_col,
		max_code=snakemake.config["community"]["max"].get(class_, 98),
	)

	if community_col not in nodelist.columns:
		raise KeyError(
			f"Community column '{community_col}' not found in {snakemake.input[1]}"
		)

	filtered_nodes = nodelist[nodelist[community_col].isin({c1, c2})].copy()
	if filtered_nodes.empty:
		raise ValueError(
			f"No nodes found for communities {c1} and {c2} in {snakemake.input[1]}"
		)

	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	filtered_node_ids = set(filtered_nodes[id_col].astype(int))
	subgraph = graph.subgraph(filtered_node_ids).copy()
	if subgraph.number_of_nodes() == 0:
		raise ValueError("Filtered subgraph contains no nodes.")

	use_weighted = any("weight" in data for _, _, data in subgraph.edges(data=True))
	if use_weighted:
		cost_graph = gc.convert_weights_to_costs(subgraph)
		betweenness = nx.betweenness_centrality(cost_graph, weight="cost")
	else:
		raise NotImplementedError(
			"Unweighted betweenness centrality is not implemented in this script."
		)

	label_col_preferred = (
		"as_display" if "as_display" in filtered_nodes.columns else label_col
	)
	label_map = _build_label_map(
		filtered_nodes,
		id_col,
		label_col_preferred,
		class_index,
		max_caes_id,
	)
	community_map = filtered_nodes.set_index(id_col)[community_col].to_dict()

	results = pd.DataFrame(
		[
			{
				"node_id": node_id,
				"community": community_map.get(node_id),
				"betweenness": score,
				"label": label_map.get(node_id, str(node_id)),
			}
			for node_id, score in betweenness.items()
		]
	)
	results = results.dropna(subset=["community"]).copy()
	results = results.sort_values("betweenness", ascending=False)
	if results.empty:
		raise ValueError("No betweenness results computed for filtered communities.")

	top_nodes = results.head(5).copy()

	# Plot 1: Betweenness Centrality Histogram (existing logic)
	color_community_1 = utils.get_community_color(
		c1, communities=nodelist["community"].unique()
	)
	color_community_2 = utils.get_community_color(
		c2, communities=nodelist["community"].unique()
	)

	figsize_hist = tuple(snakemake.config.get("figsizes", {}).get("histogram", (10, 8)))
	fig_hist, ax_hist = plt.subplots(figsize=figsize_hist)
	sns.histplot(
		data=results,
		x="betweenness",
		hue="community",
		bins="auto",
		alpha=0.6,
		kde=True,
		ax=ax_hist,
		multiple="dodge",
		palette=[color_community_1, color_community_2],
		hue_order=[c1, c2],
	)
	ax_hist.set_title("")
	ax_hist.set_xlabel(_t("Betweenness centrality"))
	ax_hist.set_ylabel(_t("node_count"))
	legend = ax_hist.get_legend()
	if legend is not None:
		legend.set_title(_t("community"))

	y_max = ax_hist.get_ylim()[1] if ax_hist.get_ylim()[1] > 0 else 1.0
	for i, (_, row) in enumerate(top_nodes.iterrows()):
		x = row["betweenness"]
		y = y_max * (0.92 - i * 0.07)
		ax_hist.axvline(x, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
		ax_hist.annotate(
			f"{row['label']} ({row['community']})",
			xy=(x, y),
			xytext=(x, y),
			rotation=90,
			va="top",
			ha="right",
			fontsize=9,
		)

	output_path_hist = Path(snakemake.output[0])
	utils.ensure_parent_dir(output_path_hist)
	fig_hist.tight_layout()
	fig_hist.savefig(output_path_hist, bbox_inches="tight")
	plt.close(fig_hist)

	# Plot 2: Community Zoom Projection Plot (NEW)
	pos = dl.load_positions(nodelist, id_col)
	if pos:
		subgraph = nx.subgraph(subgraph, set(pos.keys()))
	group_map = nodelist.set_index(id_col)[community_col].to_dict()

	# Load original community colors map
	discrete_feature = "community"
	color_col = next(
		(col for col in nodelist.columns if discrete_feature in col and col.endswith("_color")),
		None
	)
	group_color_map = {}
	if color_col:
		pairs = nodelist[[discrete_feature, color_col]].dropna().drop_duplicates()
		raw_group_color_map = dict(zip(pairs[discrete_feature], pairs[color_col]))
		for group_name, color_value in raw_group_color_map.items():
			try:
				parsed_color = utils.parse_color(color_value)
				group_color_map[group_name] = mcolors.to_hex(parsed_color)
			except Exception:
				group_color_map[group_name] = "gray"
	if not group_color_map:
		unique_groups = sorted(set(group_map.values()))
		group_color_map = utils.build_community_color_map(unique_groups, other_label="Otros")
	group_color_map.setdefault("Otros", "gray")

	# Prepare node colors (high alpha for top 5 target nodes, 0.1 for other nodes in c1/c2)
	node_colors = []
	node_color_by_node = {}
	node_alpha = snakemake.config.get("NODE_ALPHA", 0.6)
	top_node_ids = set(top_nodes["node_id"].astype(int))
	for node in subgraph.nodes():
		comm = group_map.get(node, "Otros")
		color = group_color_map.get(comm, "gray")
		if node in top_node_ids:
			rgba = mcolors.to_rgba(color, alpha=0.9)
		else:
			rgba = mcolors.to_rgba(color, alpha=0.1)
		node_colors.append(rgba)
		node_color_by_node[node] = rgba

	dataset_cfg = snakemake.config["datasets"].get(dataset, {})
	node_size_metric = dataset_cfg.get("node_size", None)
	if node_size_metric is None and "_unweighted" in dataset:
		node_size_metric = "n_obs"

	node_size_map = None
	if node_size_metric and node_size_metric in nodelist.columns:
		node_size_map = nodelist.set_index(id_col)[node_size_metric].to_dict()
		node_size_map = {int(k): float(v) for k, v in node_size_map.items()}

	factor_node_size = snakemake.config["FACTOR_NODE_SIZE"].get(class_, 0.5)
	if node_size_map is not None:
		max_val = max(node_size_map.values()) if node_size_map else 1.0
		if max_val <= 0.0:
			max_val = 1.0
		size_map = {
			node: max(
				10.0,
				(float(node_size_map.get(node, 1.0)) / max_val) * 100.0 * float(factor_node_size),
			)
			for node in subgraph.nodes()
		}
	else:
		size_map = utils.compute_node_sizes(subgraph, factor=factor_node_size, min_size=10.0, weight_attr="weight")
	node_sizes = [size_map.get(node, 10.0) for node in subgraph.nodes()]

	# Prepare edge alphas
	edge_alpha = snakemake.config["EDGE_ALPHA"].get(class_, 0.1)
	edges = list(subgraph.edges())
	edge_widths = 0.3
	if len(edges) > 0:
		edge_data = next(iter(subgraph.edges(data=True)))[-1]
		if "weight" in edge_data:
			weights = [subgraph[u][v].get("weight", 0.0) for u, v in edges]
			max_weight = max(weights) if max(weights) > 0 else 1.0
			edge_widths = [0.1 + 1.9 * (w / max_weight) for w in weights]

		edge_alphas = []
		for u, v in edges:
			u_top = u in top_node_ids
			v_top = v in top_node_ids
			if u_top and v_top:
				edge_alphas.append(0.9)
			elif u_top or v_top:
				edge_alphas.append(edge_alpha * 0.6)
			else:
				edge_alphas.append(edge_alpha * 0.3)

	# Bounding Box calculations for zooming
	x_coords = [pos[node][0] for node in subgraph.nodes() if node in pos]
	y_coords = [pos[node][1] for node in subgraph.nodes() if node in pos]

	if not x_coords or not y_coords:
		raise ValueError("No positions found for targeted communities.")

	min_x, max_x = min(x_coords), max(x_coords)
	min_y, max_y = min(y_coords), max(y_coords)

	margin_x = 0.08 * (max_x - min_x) if max_x > min_x else 1.0
	margin_y = 0.08 * (max_y - min_y) if max_y > min_y else 1.0

	xlim = (min_x - margin_x, max_x + margin_x)
	ylim = (min_y - margin_y, max_y + margin_y)

	figsize_proj = tuple(snakemake.config.get("figsizes", {}).get("projection", (8, 8)))
	fig_proj, ax_proj = plt.subplots(figsize=figsize_proj)

	nx.draw_networkx_nodes(
		subgraph,
		pos,
		node_color=node_colors,
		node_size=node_sizes,
		alpha=None,
		edgecolors="#000000",
		linewidths=0.2,
		ax=ax_proj,
	)

	if len(edges) > 0:
		edge_colors = [
			pl._edge_rgba_from_node_colors(node_color_by_node[u], node_color_by_node[v], a)
			for (u, v), a in zip(edges, edge_alphas)
		]
		nx.draw_networkx_edges(
			subgraph,
			pos,
			edgelist=edges,
			edge_color=edge_colors,
			width=edge_widths,
			alpha=None,
			ax=ax_proj,
		)

	# Direct labels for top 5 nodes with separate directions
	range_x = max_x - min_x if max_x > min_x else 1.0
	range_y = max_y - min_y if max_y > min_y else 1.0
	directions = [
		(0.12, 0.12),
		(-0.12, 0.12),
		(0.12, -0.12),
		#(-0.12, -0.12),
		(0.0, 0.18),
	]
	for idx, (_, row) in enumerate(top_nodes.iterrows()):
		node_id = int(row["node_id"])
		if node_id in pos:
			x, y = pos[node_id]
			dx, dy = directions[idx % len(directions)]
			ax_proj.annotate(
				row["label"],
				xy=(x, y),
				xytext=(x + dx * range_x, y + dy * range_y),
				arrowprops=dict(arrowstyle="->", color="black", lw=0.7, ls="-"),
				fontweight="bold",
				bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="gray"),
				ha="center",
				va="center",
			)

	ax_proj.set_xlim(xlim)
	ax_proj.set_ylim(ylim)
	ax_proj.axis("off")

	# Build legend for c1 and c2
	for group in sorted([c1, c2], key=lambda value: str(value).lower()):
		if group in group_color_map:
			group_label = utils.translate_label(group, translation) if translation else group
			plt.scatter([], [], color=group_color_map[group], label=group_label)

	legend_title_display = (
		utils.translate_label(discrete_feature, translation) if discrete_feature else ""
	)
	plt.legend(
		title=legend_title_display,
		loc="lower left",
		borderaxespad=2.0,
		framealpha=0.7,
	)

	output_path_proj = Path(snakemake.output[1])
	utils.ensure_parent_dir(output_path_proj)
	fig_proj.savefig(output_path_proj, bbox_inches="tight")
	plt.close(fig_proj)

	# Existing logging logic
	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PUBLIC POLICY BY COMMUNITIES")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SETTINGS",
		[
			f"Dataset: {dataset}",
			f"Class: {class_}",
			f"Communities: {c1}, {c2}",
			f"Community column: {community_col}",
			f"Weighted betweenness: {use_weighted}",
			f"Subgraph nodes: {subgraph.number_of_nodes()}",
			f"Subgraph edges: {subgraph.number_of_edges()}",
		],
	)
	log.add_dataframe_info(
		log_lines,
		"FILTERED NODELIST",
		row_count=len(filtered_nodes),
		column_count=len(filtered_nodes.columns),
	)
	graph_metrics = metrics.summarize_graph(subgraph)
	log.add_graph_metrics(log_lines, "Filtered subgraph metrics", graph_metrics)

	if not top_nodes.empty:
		top_lines = [
			f"{row['label']} ({row['community']}): {row['betweenness']:.6f}"
			for _, row in top_nodes.iterrows()
		]
		log.add_notes(log_lines, "Top 5 betweenness", top_lines)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
