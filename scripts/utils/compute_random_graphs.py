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


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	alpha = float(snakemake.wildcards.get("alpha", 0.05))
	seed_base = int(snakemake.config["seed"])
	resolution = float(snakemake.config["community"]["resolution"][class_])

	graph = gc.clean_graph(graph)
	metrics_empirical = metrics.summarize_graph(graph)
	empirical_weights = [
		data.get("weight", 0.0) for _, _, data in graph.edges(data=True)
	]

	outputs_er = [snakemake.output["er"]]
	outputs_cm = [snakemake.output["cm"]]
	outputs_ws = [snakemake.output["ws"]]
	outputs_ba = [snakemake.output["ba"]]
	outputs_sbm = [snakemake.output["sbm"]]
	num_realizations = 1

	node_count, edge_count, degree_sequence, k_ws, m_ba, avg_degree = (
		metrics.get_empirical_stats(graph)
	)
	sbm_sizes, sbm_probs, sbm_blocks, sbm_modularity = gc.compute_sbm_parameters(
		graph, resolution=resolution, seed=seed_base
	)

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
			f"Realizations per model: {num_realizations}",
			f"Avg degree: {avg_degree:.2f}",
			f"WS k: {k_ws}",
			f"BA m: {m_ba}",
			f"SBM blocks: {sbm_blocks}",
			f"SBM modularity: {sbm_modularity}",
		],
	)
	log.add_graph_metrics(log_lines, "Empirical backbone metrics", metrics_empirical)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None

	if node_count == 0:
		log.add_notes(
			log_lines,
			"NOTES",
			["Empirical graph is empty; writing empty graphs for all models."],
		)
		_write_empty(outputs_er + outputs_cm + outputs_ws + outputs_ba + outputs_sbm)
		log.write_log(log_lines, log_path)
		return

	for idx in range(num_realizations):
		seed = seed_base + int(snakemake.wildcards.get("i", idx))

		graph_er = nx.gnm_random_graph(node_count, edge_count, seed=seed)
		np.random.seed(seed + 1)
		graph_er = gc.assign_resampled_weights(graph_er, empirical_weights)
		_write_graph(graph_er, outputs_er[idx])

		graph_cm_multi = nx.configuration_model(degree_sequence, seed=seed)
		graph_cm = gc.clean_graph(graph_cm_multi)
		np.random.seed(seed + 2)
		graph_cm = gc.assign_resampled_weights(graph_cm, empirical_weights)
		_write_graph(graph_cm, outputs_cm[idx])

		if 0 < k_ws < node_count and k_ws % 2 == 0:
			graph_ws = nx.watts_strogatz_graph(
				node_count, k=k_ws, p=0.1, seed=seed
			)
			np.random.seed(seed + 3)
			graph_ws = gc.assign_resampled_weights(graph_ws, empirical_weights)
			_write_graph(graph_ws, outputs_ws[idx])
		else:
			logging.warning("Skipping WS %s: invalid k=%s for N=%s", idx, k_ws, node_count)

		if 0 < m_ba < node_count:
			graph_ba = nx.barabasi_albert_graph(node_count, m=m_ba, seed=seed)
			np.random.seed(seed + 4)
			graph_ba = gc.assign_resampled_weights(graph_ba, empirical_weights)
			_write_graph(graph_ba, outputs_ba[idx])
		else:
			logging.warning("Skipping BA %s: invalid m=%s for N=%s", idx, m_ba, node_count)

		if sbm_sizes and sbm_probs:
			graph_sbm = nx.stochastic_block_model(sbm_sizes, sbm_probs, seed=seed)
			np.random.seed(seed + 5)
			graph_sbm = gc.assign_resampled_weights(graph_sbm, empirical_weights)
			_write_graph(graph_sbm, outputs_sbm[idx])
		else:
			_write_graph(nx.empty_graph(node_count), outputs_sbm[idx])

	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
