from typing import Any
from pathlib import Path
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

snakemake: Any


def _sample_pairs(nodes: list[int], sample_size: int | None, seed: int) -> list[tuple[int, int]]:
	if len(nodes) < 2:
		return []

	rng = np.random.default_rng(seed)
	total_pairs = len(nodes) * (len(nodes) - 1) // 2
	target = min(sample_size, total_pairs) if sample_size is not None else total_pairs
	pairs: set[tuple[int, int]] = set()
	max_rounds = max(10, target // 2)
	for _ in range(max_rounds):
		remaining = target - len(pairs)
		if remaining <= 0:
			break

		idx = rng.integers(0, len(nodes), size=(remaining * 2, 2))
		for i, j in idx:
			if i == j:
				continue
			u = nodes[i]
			v = nodes[j]
			if u > v:
				u, v = v, u
			pairs.add((u, v))
			if len(pairs) >= target:
				break

	return list(pairs)


def _build_neighbor_maps(
	graph: nx.Graph,
	nodes: list[int],
) -> tuple[dict[int, set[int]], dict[int, dict[int, float]]]:
	neighbors: dict[int, set[int]] = {}
	weights: dict[int, dict[int, float]] = {}
	for node in nodes:
		nb = graph[node]
		neighbors[node] = set(nb)
		weights[node] = {nbr: float(data.get("weight", 1.0)) for nbr, data in nb.items()}
	return neighbors, weights


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	dataset = snakemake.wildcards["dataset"]
	class_ = snakemake.wildcards["class_"]
	seed = int(snakemake.config.get("seed", 28))
	sample_size = None

	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	graph_metrics = metrics.summarize_graph(graph)

	partition = int(snakemake.config[class_]["partition"])
	other_partition = 1 - partition

	nodes = [
		node
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == partition
	]
	other_nodes = [
		node
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == other_partition
	]
	industries_count = len(other_nodes)

	total_edges = graph.number_of_edges()
	frobenius_sq = sum(d.get("weight", 1.0) ** 2 for _, _, d in graph.edges(data=True))

	pairs = _sample_pairs(nodes, sample_size, seed)
	neighbors, weights = _build_neighbor_maps(graph, nodes)

	rows: list[dict[str, float]] = []
	for u, v in pairs:
		nb_u = neighbors.get(u, set())
		nb_v = neighbors.get(v, set())
		if not nb_u or not nb_v:
			continue

		if len(nb_u) <= len(nb_v):
			shared = nb_u & nb_v
		else:
			shared = nb_v & nb_u

		shared_count = len(shared)
		deg_u = len(nb_u)
		deg_v = len(nb_v)
		max_deg = max(deg_u, deg_v)

		weights_u = weights[u]
		weights_v = weights[v]
		dot = sum(weights_u[k] * weights_v[k] for k in shared)
		norm_u = sum(w * w for w in weights_u.values())
		norm_v = sum(w * w for w in weights_v.values())
		max_norm = max(norm_u, norm_v)

		puv_unweighted = shared_count / total_edges if total_edges > 0 else 0.0
		max_p_unweighted = max_deg / total_edges if total_edges > 0 else 0.0
		phi_unweighted = shared_count / max_deg if max_deg > 0 else 0.0

		puv_weighted = dot / frobenius_sq if frobenius_sq > 0 else 0.0
		max_p_weighted = max_norm / frobenius_sq if frobenius_sq > 0 else 0.0
		phi_weighted = dot / max_norm if max_norm > 0 else 0.0

		rows.append(
			{
				"u": u,
				"v": v,
				"phi_unweighted": phi_unweighted,
				"phi_weighted": phi_weighted,
				"puv_unweighted": puv_unweighted,
				"max_p_unweighted": max_p_unweighted,
				"puv_weighted": puv_weighted,
				"max_p_weighted": max_p_weighted,
			}
		)

	results_df = pd.DataFrame(rows)
	if results_df.empty:
		raise ValueError("No valid node pairs found for phi proximity plots.")

	figsize = tuple(snakemake.config.get("figsizes", {}).get("edge_correlation", (6, 6)))

	fig, ax = plt.subplots(figsize=figsize)
	sns.scatterplot(
		data=results_df,
		x="phi_unweighted",
		y="phi_weighted",
		alpha=0.5,
		s=14,
		edgecolor="none",
		ax=ax,
	)
	ax.set_title("Phi weighted vs unweighted")
	ax.set_xlabel("phi_unweighted")
	ax.set_ylabel("phi_weighted")
	phi_out = Path(snakemake.output[0])
	utils.ensure_parent_dir(phi_out)
	plt.savefig(phi_out, bbox_inches="tight")
	plt.close(fig)

	long_df = pd.concat(
		[
			results_df.assign(
				weight_type="weighted",
				puv=results_df["puv_weighted"],
				max_p=results_df["max_p_weighted"],
			),
			results_df.assign(
				weight_type="unweighted",
				puv=results_df["puv_unweighted"],
				max_p=results_df["max_p_unweighted"],
			),
		],
		ignore_index=True,
	)

	fig, ax = plt.subplots(figsize=figsize)
	sns.scatterplot(
		data=long_df,
		x="max_p",
		y="puv",
		hue="weight_type",
		alpha=0.5,
		s=14,
		edgecolor="none",
		ax=ax,
	)
	ax.set_title("P(u,v) vs max(P(u), P(v))")
	ax.set_xlabel("max(P(u), P(v))")
	ax.set_ylabel("P(u,v)")
	puv_out = Path(snakemake.output[1])
	utils.ensure_parent_dir(puv_out)
	plt.savefig(puv_out, bbox_inches="tight")
	plt.close(fig)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PHI PROXIMITY")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SETTINGS",
		[
			f"Dataset: {dataset}",
			f"Class: {class_}",
			f"Sample size requested: {sample_size}",
			f"Pairs computed: {len(results_df)}",
			f"Industries count: {industries_count}",
		],
	)
	log.add_graph_metrics(log_lines, "Bipartite metrics", graph_metrics)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
