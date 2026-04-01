"""Graph metric helpers for logging network summaries."""

from typing import Dict, Optional, Union

import networkx as nx

MetricValue = Union[int, float, None]


def fceyn_average_degree(graph: nx.Graph, weight: Optional[str] = None) -> float:
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


def fceyn_diameter_of_largest_component(graph: nx.Graph) -> Optional[int]:
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


def fceyn_summarize_graph(graph: nx.Graph) -> Dict[str, MetricValue]:
	"""Compute a handful of descriptive metrics for the graph."""
	return {
		"node_count": graph.number_of_nodes(),
		"edge_count": graph.number_of_edges(),
		"self_loops": nx.number_of_selfloops(graph),
		"avg_degree": fceyn_average_degree(graph),
		"avg_weighted_degree": fceyn_average_degree(graph, weight="weight"),
		"avg_clustering": nx.average_clustering(graph, weight="weight")
		if graph.number_of_nodes() > 0
		else 0.0,
		"connected_components": nx.number_connected_components(graph),
		"diameter": fceyn_diameter_of_largest_component(graph),
	}


def fceyn_log_graph_metrics(label: str, metrics: Dict[str, MetricValue]) -> None:
	"""Emit formatted metrics for the named graph in English."""
	print(f"{label} metrics:")
	print(f"Node count: {metrics['node_count']}")
	print(f"Edge count: {metrics['edge_count']}")
	print(f"Loop count: {metrics['self_loops']}")
	print(f"Average degree: {metrics['avg_degree']:.2f}")
	print(f"Average weighted degree: {metrics['avg_weighted_degree']:.2f}")
	print(f"Average clustering coefficient: {metrics['avg_clustering']:.4f}")
	diameter = metrics.get("diameter")
	diameter_display = diameter if diameter is not None else "N/A"
	print(f"Diameter (largest component): {diameter_display}")
	print(f"Connected components: {metrics['connected_components']}")
