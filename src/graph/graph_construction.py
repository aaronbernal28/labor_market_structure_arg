"""
Graph construction helpers for bipartite and projected networks.
"""

from typing import Dict
import networkx as nx
import pandas as pd
import numpy as np


def fceyn_build_bipartite_graph(
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


def fceyn_build_biadjacency(
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


def fceyn_generic_weighted_projected_graph(
	graph: nx.Graph, target_partition: int, weight_function=None
) -> nx.Graph:
	"""Weighted projection onto a bipartite partition using a custom weight function."""
	nodes = [
		node
		for node in graph.nodes
		if graph.nodes[node].get("bipartite") == target_partition
	]
	return nx.bipartite.generic_weighted_projected_graph(graph, nodes, weight_function)


def fceyn_dot_product_weight(G: nx.Graph, u: int, v: int) -> float:
	"""Newman, M. E. J. (2001). Scientific collaboration networks. II. Shortest paths, weighted networks, and centrality.
	Zhou, T., Ren, J., Medo, M., & Zhang, Y. C. (2007). Bipartite network projection and personal recommendation.
	"""
	shared_nodes = set(G[u]).intersection(G[v])
	return sum(
		G[u][node].get("weight", 1) * G[v][node].get("weight", 1)
		for node in shared_nodes
	)


def fceyn_weighted_hidalgo_proximity_weight(
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


def fceyn_degree_sequences(
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
