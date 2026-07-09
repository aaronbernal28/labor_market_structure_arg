from typing import Any, Iterable
from pathlib import Path
import logging

from scripts import *
import networkx as nx
import numpy as np

snakemake: Any


def _write_graph(graph: nx.Graph, output_path: str | Path) -> None:
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	nx.write_gexf(graph, path)


def _write_empty(outputs: Iterable[str]) -> None:
	empty = nx.Graph()
	for output_path in outputs:
		_write_graph(empty, output_path)


def _strength_sequence(graph: nx.Graph, nodes: list[int], scale: float) -> list[int]:
	strengths = dict(graph.degree(weight="weight"))
	seq = [max(0, int(round(strengths.get(node, 0.0) * scale))) for node in nodes]
	if seq and sum(seq) % 2 != 0:
		seq[0] += 1
	return seq


def _generate_wcm(graph: nx.Graph, scale: float, seed: int) -> nx.Graph:
	"""Weighted configuration model with discretized strengths."""
	nodes = list(graph.nodes())
	strength_seq = _strength_sequence(graph, nodes, scale)
	graph_multi = nx.configuration_model(strength_seq, seed=seed)

	graph_wcm = nx.Graph()
	for u, v in graph_multi.edges():
		node_u, node_v = nodes[u], nodes[v]
		if node_u == node_v:
			continue
		if graph_wcm.has_edge(node_u, node_v):
			graph_wcm[node_u][node_v]["weight"] += 1.0 / scale
		else:
			graph_wcm.add_edge(node_u, node_v, weight=1.0 / scale)
	return graph_wcm


def _generate_ecm(graph: nx.Graph, seed: int) -> nx.Graph:
	"""Enhanced configuration model (two-step / CReMb variant)."""
	nodes = list(graph.nodes())
	degrees = dict(graph.degree())
	strengths = dict(graph.degree(weight="weight"))

	m = graph.number_of_edges()
	weight_total = sum(strengths.values()) / 2.0
	degree_seq = [degrees[node] for node in nodes]

	graph_multi = nx.configuration_model(degree_seq, seed=seed)
	graph_ecm = nx.Graph()
	if m == 0 or weight_total == 0.0:
		return graph_ecm

	rng = np.random.default_rng(seed + 1)
	for u, v in graph_multi.edges():
		node_u, node_v = nodes[u], nodes[v]
		strength_u = strengths.get(node_u, 0.0)
		strength_v = strengths.get(node_v, 0.0)
		if strength_u == 0.0 or strength_v == 0.0:
			continue
		rate = (weight_total * degrees[node_u] * degrees[node_v]) / (
			m * strength_u * strength_v
		)
		if rate <= 0.0:
			continue
		sampled_weight = float(rng.exponential(scale=1.0 / rate))
		if graph_ecm.has_edge(node_u, node_v):
			graph_ecm[node_u][node_v]["weight"] += sampled_weight
		else:
			graph_ecm.add_edge(node_u, node_v, weight=sampled_weight)

	return graph_ecm


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	alpha = float(snakemake.wildcards.get("alpha", 0.05))
	seed_base = int(snakemake.config["seed"])
	resolution = float(
		snakemake.config["community"]["resolution"].get(f"{alpha:.2f}", {}).get(class_, 1.0)
	)
	wcm_scale = float(
		snakemake.config.get("random_graphs", {}).get("wcm_scale", 1000.0)
	)

	graph = gc.clean_graph(graph)
	# metrics_empirical = metrics.summarize_graph(graph)
	outputs_cm = [snakemake.output["cm"]]
	outputs_ecm = [snakemake.output["ecm"]]
	num_realizations = 1

	node_count, edge_count, _, _, _, avg_degree = metrics.get_empirical_stats(graph)

	strength_seq = _strength_sequence(graph, list(graph.nodes()), wcm_scale)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("RANDOM GRAPH GENERATION")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PARAMETERS",
		[
			f"Class: {class_}",
			f"Alpha: {alpha}",
			f"Seed base: {seed_base}",
			f"Resolution: {resolution}",
			f"WCM scale: {wcm_scale}",
			f"Realizations per model: {num_realizations}",
			f"Avg degree: {avg_degree:.2f}",
			f"WCM stub sum: {sum(strength_seq)}",
		],
	)
	# log.add_graph_metrics(log_lines, "Empirical backbone metrics", metrics_empirical)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None

	if node_count == 0:
		log.add_notes(
			log_lines,
			"NOTES",
			["Empirical graph is empty; writing empty graphs for all models."],
		)
		_write_empty(outputs_cm + outputs_ecm)
		log.write_log(log_lines, log_path)
		return

	for idx in range(num_realizations):
		seed = seed_base + int(snakemake.wildcards.get("i", idx))
		print(
			f"Generating random graphs (realization {idx + 1}/{num_realizations}). Seed = {seed}."
		)

		graph_wcm = _generate_wcm(graph, wcm_scale, seed=seed)
		_write_graph(graph_wcm, outputs_cm[idx])

		graph_ecm = _generate_ecm(graph, seed=seed)
		_write_graph(graph_ecm, outputs_ecm[idx])

	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
