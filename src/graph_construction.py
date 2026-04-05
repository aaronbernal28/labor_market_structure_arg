"""
Graph construction helpers for bipartite and projected networks.
"""

from typing import Dict
import networkx as nx
import pandas as pd
import numpy as np


def get_weight_function(weight_function_name: str):
	"""Map weight function name to actual function."""
	weight_function_name = weight_function_name.lower()
	if weight_function_name == "hidalgo":
		return weighted_hidalgo_weight
	if weight_function_name == "unweighted_hidalgo":
		return unweighted_hidalgo_weight
	if weight_function_name == "dot_product":
		return dot_product_weight
	if weight_function_name == "cosine":
		return cosine_similarity_weight
	raise ValueError(f"Unknown weight function '{weight_function_name}'.")


def build_bipartite_graph(
	enes_df: pd.DataFrame,
	caes_id: str,
	ciuo_id: str,
	logscale: bool = True,
	caes_partition: int = 1,
	ciuo_partition: int = 0,
) -> nx.Graph:
	"""
	Build the bipartite graph from the merged ENES dataframe.
	"""
	# Define node sets
	caes_nodes = set(enes_df[caes_id].unique())
	ciuo_nodes = set(enes_df[ciuo_id].unique())

	assert caes_nodes & ciuo_nodes == set(), "CAES and CIUO IDs must be disjoint."

	# Build bipartite graph
	graph = nx.Graph()
	graph.add_nodes_from(caes_nodes, bipartite=caes_partition)
	graph.add_nodes_from(ciuo_nodes, bipartite=ciuo_partition)

	edges = (
		enes_df.groupby([caes_id, ciuo_id])
		.size()
		.apply(lambda x: x if not logscale else float(np.log1p(x)))
		.reset_index(name="weight")
	)

	# Add edges with weights using vectorization
	graph.add_weighted_edges_from(
		edges[[caes_id, ciuo_id, "weight"]].itertuples(index=False, name=None)
	)

	assert nx.is_bipartite(graph), "Constructed graph is not bipartite."
	return graph


def build_biadjacency(
	enes_df: pd.DataFrame,
	caes_col: str,
	ciuo_col: str,
	logscale: bool = False,
	rownames=None,
	colnames=None,
) -> pd.DataFrame:
	"""Return the CAES-by-CIUO biadjacency matrix (counts)."""
	matrix = pd.crosstab(enes_df[caes_col], enes_df[ciuo_col])

	# Reindex to include all specified row/column names if provided
	if rownames is not None:
		matrix = matrix.reindex(index=rownames, fill_value=0)
	if colnames is not None:
		matrix = matrix.reindex(columns=colnames, fill_value=0)

	matrix = np.log1p(matrix) if logscale else matrix

	if type(matrix) is not pd.DataFrame:
		raise ValueError(
			"Biadjacency matrix construction failed; result is not a DataFrame."
		)
	return matrix


def generic_weighted_projected_graph(
	graph: nx.Graph,
	target_partition: int = None,
	weight_function=None,
	class_name: str = None,
) -> nx.Graph:
	"""Weighted projection using either a partition index or class name."""
	target_partition = _resolve_partition(
		target_partition=target_partition, class_name=class_name
	)
	nodes = [
		node
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == target_partition
	]
	return nx.bipartite.generic_weighted_projected_graph(graph, nodes, weight_function)


def _resolve_partition(target_partition: int = None, class_name: str = None) -> int:
	"""Support both partition-index and class-name APIs."""
	if target_partition is not None:
		return target_partition
	if class_name is None:
		raise ValueError("Provide either target_partition or class_name.")
	class_name = class_name.lower()
	if class_name == "caes":
		return 1
	if class_name == "ciuo":
		return 0
	raise ValueError("class_name must be 'caes' or 'ciuo'.")


def projected_graph(graph: nx.Graph, class_name: str) -> nx.Graph:
	"""Unweighted projection onto caes or ciuo node sets."""
	target_partition = _resolve_partition(class_name=class_name)
	nodes = [
		node
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == target_partition
	]
	return nx.bipartite.projected_graph(graph, nodes)


def weighted_projected_graph(graph: nx.Graph, class_name: str) -> nx.Graph:
	"""Weighted projection onto caes or ciuo node sets."""
	target_partition = _resolve_partition(class_name=class_name)
	nodes = [
		node
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == target_partition
	]
	return nx.bipartite.weighted_projected_graph(graph, nodes)


def generic_weighted_projected_graph_by_name(
	graph: nx.Graph, class_name: str, weight_function=None
) -> nx.Graph:
	"""Backward-compatible class-name variant of generic weighted projection."""
	target_partition = _resolve_partition(class_name=class_name)
	return generic_weighted_projected_graph(
		graph, target_partition=target_partition, weight_function=weight_function
	)


def unweighted_hidalgo_weight(G: nx.Graph, u: int, v: int) -> float:
	"""Minimum conditional probability proximity as in Hidalgo et al. (2007)."""
	shared_features_len = len(set(G[u]).intersection(G[v]))
	if shared_features_len == 0:
		return 0.0

	degree_u = G.degree[u]
	degree_v = G.degree[v]
	if degree_u == 0 or degree_v == 0:
		return 0.0

	prob_u_given_v = shared_features_len / degree_v
	prob_v_given_u = shared_features_len / degree_u
	return min(prob_u_given_v, prob_v_given_u)


def dot_product_weight(G: nx.Graph, u: int, v: int) -> float:
	"""Newman, M. E. J. (2001). Scientific collaboration networks. II. Shortest paths, weighted networks, and centrality.
	Zhou, T., Ren, J., Medo, M., & Zhang, Y. C. (2007). Bipartite network projection and personal recommendation.
	"""
	shared_nodes = set(G[u]).intersection(G[v])
	return sum(
		G[u][node].get("weight", 1) * G[v][node].get("weight", 1)
		for node in shared_nodes
	)


def cosine_similarity_weight(G: nx.Graph, u: int, v: int) -> float:
	"""Cosine similarity using edge weights on shared neighbors."""
	norm_weight_u = 0.0
	norm_weight_v = 0.0
	shared_nodes = set(G[u]) & set(G[v])

	for node in shared_nodes:
		w_u = G[u][node].get("weight", 1)
		w_v = G[v][node].get("weight", 1)
		norm_weight_u += w_u**2
		norm_weight_v += w_v**2

	if norm_weight_u <= 0 or norm_weight_v <= 0:
		return 0.0

	return dot_product_weight(G, u, v) / (
		np.sqrt(norm_weight_u) * np.sqrt(norm_weight_v)
	)


def weighted_hidalgo_weight(
	G: nx.Graph, u: int, v: int, weight: str = "weight"
) -> float:
	"""
	Calculates the 'Weighted Hidalgo' proximity (Minimum Conditional Probability) preserving intensity.
	"""
	# Get the shared neighbors
	# Note: For very dense graphs, iterating the smaller neighborhood is faster
	nb_u = set(G[u])
	nb_v = set(G[v])
	shared_neighbors = list(nb_u & nb_v)

	# If no overlap, return 0 to save time
	if len(shared_neighbors) == 0:
		return 0.0

	# 1. Calculate Weighted Overlap (Dot Product)
	# Sum of (weight_u_k * weight_v_k) for all shared neighbors k
	dot_product = sum(
		G[u][k].get(weight, 1.0) * G[v][k].get(weight, 1.0) for k in shared_neighbors
	)

	# 2. Calculate "Weighted Degrees" (Squared Norms)
	# We must use squared weights so the denominator matches the scale of the numerator (weights * weights)
	# This ensures the probability never exceeds 1.0.
	norm_sq_u = sum(d.get(weight, 1.0) ** 2 for _, d in G[u].items())
	norm_sq_v = sum(d.get(weight, 1.0) ** 2 for _, d in G[v].items())

	# Avoid division by zero
	if norm_sq_u == 0 or norm_sq_v == 0:
		return 0.0

	# 3. Calculate Conditional Probabilities (Weighted)
	# "Given v, how much of its total 'energy' overlaps with u?"
	prob_u_given_v = dot_product / norm_sq_v
	prob_v_given_u = dot_product / norm_sq_u

	# Return the Minimum (The Hidalgo/Hausmann standard)
	return min(prob_u_given_v, prob_v_given_u)


def degree_sequences(
	graph: nx.Graph, caes_partition: int = 1, ciuo_partition: int = 0
) -> Dict[str, list]:
	"""Return degree lists for all nodes and each partition."""
	degrees_all = list(dict(graph.degree()).values())
	degrees_caes = [
		graph.degree(node)
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == caes_partition
	]
	degrees_ciuo = [
		graph.degree(node)
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == ciuo_partition
	]
	return {
		"all": degrees_all,
		"caes": degrees_caes,
		"ciuo": degrees_ciuo,
	}


def get_projection_positions(
	graph: nx.Graph,
	seed: int = 42,
	spring_layout_iterations: int = 1000,
	spring_layout_k: float = None,
	rotate: bool = False,
	method: str = "auto",
) -> dict:
	"""Calculate node positions for the projection graph using a force-directed layout."""
	if not nx.is_connected(graph):
		# Get the largest connected component for layout
		largest_cc = max(nx.connected_components(graph), key=len)
		graph = graph.subgraph(largest_cc)
		print(
			"Warning: Projection graph is not connected; using largest connected component for layout."
		)
		print(
			f"Original graph had {graph.number_of_nodes()} nodes; largest component has {len(graph.nodes())} nodes."
		)

	if method == "kamada_kawai":
		pos = nx.kamada_kawai_layout(graph)
	else:
		pos = nx.spring_layout(
			graph,
			seed=seed,
			k=spring_layout_k,
			iterations=spring_layout_iterations,
			threshold=1e-3,
			method=method,
		)

	if rotate:
		pos = {node: (-y, x) for node, (x, y) in pos.items()}

	return pos


def disparity_filter_backbone(
	graph: nx.Graph,
	alpha: float = 0.05,
	mode: str = "or",
	keep_isolates: bool = True,
) -> nx.Graph:
	"""Return a disparity-filtered backbone for a weighted undirected graph.

	An edge is kept when its significance is below alpha in at least one endpoint
	("or") or in both endpoints ("and").
	"""
	if mode not in {"or", "and"}:
		raise ValueError("mode must be 'or' or 'and'.")
	if alpha <= 0 or alpha > 1:
		raise ValueError("alpha must be in (0, 1].")

	if graph.number_of_nodes() == 0:
		return nx.Graph()

	# Use a plain graph and preserve existing graph-level attributes.
	backbone = nx.Graph()
	backbone.graph.update(graph.graph)

	if keep_isolates:
		backbone.add_nodes_from(graph.nodes(data=True))

	strength = {
		n: sum(float(d.get("weight", 1.0)) for _, _, d in graph.edges(n, data=True))
		for n in graph.nodes()
	}
	degree = dict(graph.degree())

	for u, v, data in graph.edges(data=True):
		w = float(data.get("weight", 1.0))

		def _alpha_endpoint(node: int, weight: float) -> float:
			k = degree.get(node, 0)
			s = strength.get(node, 0.0)
			if k <= 1 or s <= 0:
				return 0.0
			p = max(0.0, min(1.0, weight / s))
			return float((1.0 - p) ** (k - 1))

		a_u = _alpha_endpoint(u, w)
		a_v = _alpha_endpoint(v, w)

		keep_edge = (
			(a_u < alpha or a_v < alpha)
			if mode == "or"
			else (a_u < alpha and a_v < alpha)
		)
		if keep_edge:
			backbone.add_edge(u, v, **data)

	if not keep_isolates:
		backbone.add_nodes_from(graph.nodes(data=True))
		isolates = [n for n in backbone.nodes() if backbone.degree(n) == 0]
		backbone.remove_nodes_from(isolates)

	return backbone
