"""Graph metric helpers for logging network summaries."""

from typing import Dict, Optional, Union, Iterable, Any
from statistics import mean, median
import math
import src.graph_construction as gc

import networkx as nx

MetricValue = Union[int, float, None, Dict[str, Any]]
BipartiteStats = Dict[str, MetricValue]


def average_degree(graph: nx.Graph, weight: Optional[str] = None) -> float:
	"""Return the average (weighted) degree of the provided graph."""
	node_count = graph.number_of_nodes()
	if node_count == 0:
		res = 0.0
	elif weight is not None:
		total_degree = sum(dict(graph.degree(weight=weight)).values())
		res = total_degree / node_count
	else:
		res = 2 * graph.number_of_edges() / node_count
	return res


def diameter_of_largest_component(graph: nx.Graph) -> Optional[int]:
	"""Return the diameter of the largest connected component, or None if the graph is empty."""
	if graph.number_of_nodes() == 0:
		return None
	if nx.is_connected(graph):
		return nx.diameter(graph)
	largest_component = max(nx.connected_components(graph), key=len, default=set())
	if not largest_component:
		return None
	subgraph = graph.subgraph(largest_component).copy()
	return nx.diameter(subgraph) if subgraph.number_of_nodes() > 0 else None


def _summary_stats(values: Iterable[float]) -> BipartiteStats:
	values_list = list(values)
	if not values_list:
		return {"mean": None, "median": None, "min": None, "max": None}
	return {
		"mean": float(mean(values_list)),
		"median": float(median(values_list)),
		"min": float(min(values_list)),
		"max": float(max(values_list)),
	}


def _partition_nodes(graph: nx.Graph) -> Dict[int, list]:
	partitions: Dict[int, list] = {}
	for node, attrs in graph.nodes(data=True):
		partition = attrs.get("bipartite")
		if partition is None:
			continue
		partitions.setdefault(int(partition), []).append(node)
	return partitions


def _largest_component_subgraph(graph: nx.Graph) -> nx.Graph:
	if graph.number_of_nodes() == 0:
		return graph
	if nx.is_connected(graph):
		return graph.copy()
	largest_component = max(nx.connected_components(graph), key=len, default=set())
	return graph.subgraph(largest_component).copy()


def summarize_graph(graph: nx.Graph) -> Dict[str, MetricValue]:
	"""Compute a handful of descriptive metrics for the graph."""
	# basic counts
	node_count = graph.number_of_nodes()
	edge_count = graph.number_of_edges()
	self_loops = nx.number_of_selfloops(graph)
	density = nx.density(graph)

	# degrees
	avg_deg = average_degree(graph)
	avg_wdeg = average_degree(graph, weight="weight")

	# clustering: guard weighted call against division-by-zero in networkx
	avg_clust = nx.average_clustering(graph) if node_count > 0 else 0.0
	try:
		avg_wclust = (
			nx.average_clustering(graph, weight="weight") if node_count > 0 else 0.0
		)
	except ZeroDivisionError:
		# fallback to unweighted clustering when weights are all zero
		avg_wclust = avg_clust

	return {
		"node_count": node_count,
		"edge_count": edge_count,
		"self_loops": self_loops,
		"density": density,
		"avg_degree": avg_deg,
		"avg_weighted_degree": avg_wdeg,
		"avg_clustering": avg_clust,
		"avg_weighted_clustering": avg_wclust,
		"connected_components": nx.number_connected_components(graph),
		"diameter": diameter_of_largest_component(graph),
	}


def summarize_bipartite_graph(
	graph: nx.Graph, partition_labels: Optional[Dict[int, str]] = None
) -> Dict[str, MetricValue]:
	"""Compute bipartite-specific metrics plus the generic graph summary."""
	metrics = summarize_graph(graph)
	partition_labels = partition_labels or {}
	partitions = _partition_nodes(graph)
	bipartite_partitions: Dict[str, Dict[str, object]] = {}
	for partition_id, nodes in partitions.items():
		label = partition_labels.get(partition_id, f"partition_{partition_id}")
		degrees = [deg for _, deg in graph.degree(nodes)]
		strengths = [deg for _, deg in graph.degree(nodes, weight="weight")]
		bipartite_partitions[label] = {
			"partition": partition_id,
			"size": len(nodes),
			"degree_stats": _summary_stats(degrees),
			"strength_stats": _summary_stats(strengths),
		}

	partition_sizes = [len(nodes) for nodes in partitions.values()]
	if len(partition_sizes) == 2 and partition_sizes[0] > 0 and partition_sizes[1] > 0:
		bipartite_density = metrics["edge_count"] / (
			partition_sizes[0] * partition_sizes[1]
		)
	else:
		bipartite_density = None

	assortativity = None
	if metrics["edge_count"] > 0:
		assortativity_value = nx.degree_assortativity_coefficient(graph, x=0, y=1)
		assortativity_value_weighted = nx.degree_assortativity_coefficient(graph, x=0, y=1, weight="weight")
		if math.isfinite(assortativity_value):
			assortativity = float(assortativity_value)
		if math.isfinite(assortativity_value_weighted):
			assortativity_weighted = float(assortativity_value_weighted)

	lcc = _largest_component_subgraph(graph)
	lcc_size = lcc.number_of_nodes()
	if metrics["node_count"] > 0:
		lcc_percent = 100.0 * lcc_size / metrics["node_count"]
	else:
		lcc_percent = 0.0

	if lcc_size > 1:
		lcc_with_cost = gc.convert_weights_to_costs(lcc, prob_weight=False)
		avg_path_length = float(nx.average_shortest_path_length(lcc_with_cost, weight="cost"))
	else:
		avg_path_length = None

	metrics.update(
		{
			"bipartite_density": bipartite_density,
			"degree_assortativity": assortativity,
			"degree_assortativity_weighted": assortativity_weighted,
			"lcc_size": lcc_size,
			"lcc_percent": lcc_percent,
			"avg_path_length": avg_path_length,
			"bipartite_partitions": bipartite_partitions,
		}
	)
	return metrics


def log_graph_metrics(label: str, metrics: Dict[str, MetricValue]) -> None:
	"""Emit formatted metrics for the named graph in English."""
	print(f"{label} metrics:")
	print(f"Node count: {metrics['node_count']}")
	print(f"Edge count: {metrics['edge_count']}")
	print(f"Loop count: {metrics['self_loops']}")
	print(f"Density: {metrics['density']:.4f}")
	if metrics.get("bipartite_density") is not None:
		print(f"Bipartite density: {metrics['bipartite_density']:.6f}")
	print(f"Average degree: {metrics['avg_degree']:.2f}")
	print(f"Average weighted degree: {metrics['avg_weighted_degree']:.2f}")
	print(f"Average clustering coefficient ponderado: {metrics['avg_clustering']:.4f}")
	print(
		f"Average weighted clustering coefficient: {metrics['avg_weighted_clustering']:.4f}"
	)
	diameter = metrics.get("diameter")
	diameter_display = diameter if diameter is not None else "N/A"
	print(f"Diameter (largest component): {diameter_display}")
	if metrics.get("avg_path_length") is not None:
		print(f"Average path length (largest component): {metrics['avg_path_length']:.4f}")
	print(f"Connected components: {metrics['connected_components']}")
	if metrics.get("lcc_size") is not None:
		lcc_percent = metrics.get("lcc_percent")
		lcc_percent_display = f"{lcc_percent:.2f}%" if lcc_percent is not None else "N/A"
		print(f"Largest component size: {metrics['lcc_size']} ({lcc_percent_display})")
	if metrics.get("degree_assortativity") is not None:
		print(f"Degree assortativity: {metrics['degree_assortativity']:.4f}")
	if metrics.get("degree_assortativity_weighted") is not None:
		print(f"Degree assortativity (weighted): {metrics['degree_assortativity_weighted']:.4f}")
