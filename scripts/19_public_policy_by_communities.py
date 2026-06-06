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

	color_community_1 = utils.get_community_color(
		c1, communities=nodelist["community"].unique()
	)

	color_community_2 = utils.get_community_color(
		c2, communities=nodelist["community"].unique()
	)

	figsize = tuple(snakemake.config.get("figsizes", {}).get("histogram", (10, 8)))
	fig, ax = plt.subplots(figsize=figsize)
	sns.histplot(
		data=results,
		x="betweenness",
		hue="community",
		bins="auto",
		alpha=0.6,
		kde=True,
		ax=ax,
		multiple="dodge",
		palette=[color_community_1, color_community_2],
		hue_order=[c1, c2],
	)
	ax.set_title("")
	ax.set_xlabel(_t("Betweenness centrality"))
	ax.set_ylabel(_t("node_count"))
	legend = ax.get_legend()
	if legend is not None:
		legend.set_title(_t("community"))

	y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
	for i, (_, row) in enumerate(top_nodes.iterrows()):
		x = row["betweenness"]
		y = y_max * (0.92 - i * 0.07)
		ax.axvline(x, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
		ax.annotate(
			f"{row['label']} ({row['community']})",
			xy=(x, y),
			xytext=(x, y),
			rotation=90,
			va="top",
			ha="right",
			fontsize=9,
		)

	output_path = Path(snakemake.output[0])
	utils.ensure_parent_dir(output_path)
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches="tight")
	plt.close(fig)

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
