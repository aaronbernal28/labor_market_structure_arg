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
	"""Calculate node positions for the projection graph using a force-directed layout.

	Handles both undirected and directed graphs (converts directed to undirected as needed).
	"""
	# Convert to undirected if needed (e.g., for directed graphs from disparity_filter_backbone)
	if isinstance(graph, nx.DiGraph):
		graph = nx.to_undirected(graph)

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


def disparity_alpha_endpoint(k: int, strength: float, weight: float) -> float:
	"""Compute disparity-filter endpoint significance.

	This is the endpoint-level significance used by the disparity filter for weighted
	undirected graphs.
	"""
	if k <= 1 or strength <= 0:
		return 0.0
	p = max(0.0, min(1.0, weight / strength))
	return float((1.0 - p) ** (k - 1))


def get_disparity_graph(graph_in: nx.Graph) -> nx.DiGraph:
	"""Return a disparity backbone as a directed graph."""
	if graph_in.number_of_nodes() == 0:
		return nx.DiGraph()

	# Use a directed graph and preserve existing graph-level attributes.
	graph_out = nx.DiGraph()
	graph_out.graph.update(graph_in.graph)

	strength = {
		n: sum(float(d.get("weight", 1.0)) for _, _, d in graph_in.edges(n, data=True))
		for n in graph_in.nodes()
	}
	degree = dict(graph_in.degree())

	for u, v, data in graph_in.edges(data=True):
		w_uv = float(data.get("weight", 0.0))
		a_uv = disparity_alpha_endpoint(degree.get(u, 0), strength.get(u, 0.0), w_uv)
		a_vu = disparity_alpha_endpoint(degree.get(v, 0), strength.get(v, 0.0), w_uv)

		graph_out.add_edge(u, v, alpha=a_uv, weight=w_uv)
		graph_out.add_edge(v, u, alpha=a_vu, weight=w_uv)

	return graph_out


def disparity_filter_backbone(
	original_graph: nx.Graph = None,
	disparity_graph: nx.DiGraph = None,
	alpha: float = 0.05,
	mode: str = "or",
	keep_isolates: bool = True,
) -> nx.Graph:
	"""Return a disparity-filtered backbone as an undirected graph.

	For each edge in the input undirected graph, the significance of
	individual endpoints is checked:
	- Edge u -> v is significant if a_uv < alpha
	- Edge v -> u is significant if a_vu < alpha

	With mode="or", at least one direction must pass.
	With mode="and", both directions must pass.
	"""
	if mode not in {"or", "and"}:
		raise ValueError("mode must be 'or' or 'and'.")
	if alpha <= 0 or alpha > 1:
		raise ValueError("alpha must be in (0, 1].")

	if original_graph is None and disparity_graph is None:
		raise ValueError("Provide at least one of original_graph or disparity_graph.")

	if disparity_graph is None:
		disparity_graph = get_disparity_graph(original_graph)

	if disparity_graph.number_of_nodes() == 0:
		return nx.Graph()

	# Start with all nodes, but no edges
	backbone = nx.Graph()
	backbone.add_nodes_from(disparity_graph.nodes(data=True))

	for u, v in disparity_graph.to_undirected().edges():
		# Determine which directions pass the threshold
		u_to_v_passes = (
			disparity_graph.get_edge_data(u, v, default={}).get("alpha", 1) < alpha
		)
		v_to_u_passes = (
			disparity_graph.get_edge_data(v, u, default={}).get("alpha", 1) < alpha
		)

		if mode == "or":
			# Keep edge if at least one direction passes
			if u_to_v_passes or v_to_u_passes:
				w_uv = disparity_graph.get_edge_data(u, v, default={}).get("weight", 1)
				# print(f"Keeping edge ({u}, {v}) with data w_uv={w_uv}, a_uv={disparity_graph.get_edge_data(u, v, default={}).get('alpha', 'N/A')}, a_vu={disparity_graph.get_edge_data(v, u, default={}).get('alpha', 'N/A')}")
				backbone.add_edge(u, v, weight=w_uv)
		else:  # mode == "and"
			# Keep edge only if both directions pass
			if u_to_v_passes and v_to_u_passes:
				w_uv = disparity_graph.get_edge_data(u, v, default={}).get("weight", 1)
				backbone.add_edge(u, v, weight=w_uv)

	if not keep_isolates:
		isolates = [n for n in backbone.nodes() if backbone.degree(n) == 0]
		backbone.remove_nodes_from(isolates)

	return backbone


def compute_distance_matrix(graph: nx.Graph, method: str) -> np.ndarray:
	"""Compute a distance matrix for the given graph and method."""
	if method == "disparity_filtration":
		return get_disparity_distance_matrix(graph)
	elif method == "shortest_path":
		return get_shortest_path_distance_matrix(graph)
	else:
		raise ValueError(f"Unknown distance matrix method '{method}'.")


def _build_betas(size: int = 100) -> np.ndarray:
	"""Return a strictly increasing beta grid including 0 and 1."""
	betas = np.concatenate(([0.0], np.logspace(-6, 0, size-2)))
	betas = np.unique(betas)
	betas.sort()
	if betas[0] != 0.0:
		betas = np.concatenate(([0.0], betas))
	# Ensure 1.0 is present and is the last endpoint
	if betas[-1] != 1.0:
		betas = np.concatenate((betas, [1.0]))
		betas = np.unique(betas)
		betas.sort()
	betas[-1] = 1.0
	return betas


def _edge_distance_from_alpha(alpha_min: float, betas: np.ndarray) -> int:
	"""Compute d(u,v) = min{k : alpha < beta_k} with inclusive last beta."""
	k_max = len(betas) - 1
	# First index where beta_k > alpha_min
	k = int(np.searchsorted(betas, float(alpha_min), side="right"))
	# Inclusive behavior at beta_K = 1 for alpha=1
	return min(k, k_max)


def get_disparity_distance_matrix(graph: nx.Graph) -> np.ndarray:
	"""Compute a distance matrix for the given graph and method."""
	betas = _build_betas(size=100)
	k_max = len(betas) - 1
	default_distance = float(k_max + 1)

	nodes = sorted(graph.nodes())
	node_index = {node: i for i, node in enumerate(nodes)}
	n_nodes = len(nodes)

	distance_matrix = np.full((n_nodes, n_nodes), default_distance, dtype=float)
	np.fill_diagonal(distance_matrix, 0.0)

	disparity_graph = get_disparity_graph(graph)

	for u, v in graph.edges():
		a_uv = float(disparity_graph.edges[u, v].get("alpha", 1.0))
		a_vu = float(disparity_graph.edges[v, u].get("alpha", 1.0))
		alpha_min = min(a_uv, a_vu)
		d_uv = float(_edge_distance_from_alpha(alpha_min, betas))
		i = node_index[u]
		j = node_index[v]
		distance_matrix[i, j] = d_uv
		distance_matrix[j, i] = d_uv

	#distance_matrix = np.log1p(distance_matrix)  # Log-transform to compress scale and handle infinite distances

	return distance_matrix


def convert_weights_to_costs(graph: nx.Graph) -> nx.Graph:
	"""Convert edge weights to costs for distance calculations."""
	cost_graph = graph.copy()
	for u, v, data in graph.edges(data=True):
		weight = float(data.get("weight", 0.0))
		cost = float(-np.log(weight)) if weight > 0 else float("inf")
		# NOTE: if weight is zero, cost becomes infinite
		# If weight is one, cost becomes zero (as expected for a perfect match)
		cost_graph.add_edge(u, v, cost=cost)
	return cost_graph


def get_shortest_path_distance_matrix(graph: nx.Graph) -> np.ndarray:
	"""Compute a distance matrix for the given graph and method."""
	cost_graph = convert_weights_to_costs(graph)
	nodes = sorted(cost_graph.nodes())
	distance_matrix = nx.floyd_warshall_numpy(cost_graph, nodelist=nodes, weight="cost")
	print("Shortest path distance matrix computed.")
	print(f"Distance matrix shape: {distance_matrix.shape}")
	print(f"Distance matrix sample:\n{distance_matrix[:5, :5]}")
	return distance_matrix
