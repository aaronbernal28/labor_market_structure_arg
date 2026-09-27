"""
Community detection utilities.
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
from networkx.algorithms.community import (
	girvan_newman,
	louvain_communities,
	modularity,
)
import numpy as np
import igraph as ig
import leidenalg as la
from infomap import Infomap
import src.utils as ut


def louvain_partition(
	graph: nx.Graph,
	resolution: float = 1.0,
	seed: int = 28,
	markov_time: float | None = None,
) -> Tuple[Dict[int, int], float]:
	"""
	Run Louvain and return the partition map plus modularity.
	"""
	try:
		communities_list = louvain_communities(
			graph,
			weight="weight",
			resolution=resolution,
			seed=seed,  # backend="cugraph"
		)
	except NotImplementedError as exc:
		print("Falling back to CPU-based Louvain implementation (may be slower)")
		communities_list = louvain_communities(
			graph, weight="weight", resolution=resolution, seed=seed
		)

	communities = {node: i for i, comm in enumerate(communities_list) for node in comm}

	try:
		score = modularity(graph, communities_list, weight="weight", resolution=1.0)
	except ZeroDivisionError:
		score = 0.0
	return communities, score


def best_louvain_partition_random(
	graph: nx.Graph,
	seed: int = 28,
	n_samples: int = 20,
	resolution: float = 1.0,
) -> Tuple[Dict[int, int], float]:
	"""
	Find the best Louvain partition by random sampling of resolution values.
	Args:
		graph: The graph to partition
		seed: Random seed for reproducibility
		n_samples: Number of random resolution values to sample (default 20)
		min_resolution: Minimum resolution value (default 0.5)
		max_resolution: Maximum resolution value (default 5.0)

	Returns:
		Tuple of (best_partition, best_modularity, best_resolution)
	"""
	rng = np.random.RandomState(seed)

	# Sample resolution values uniformly
	seeds = rng.randint(0, 10000, n_samples).tolist()

	best_partition = None
	best_score = -1.0

	for seed in seeds:
		# Use deterministic seed derived from base seed
		partition, score = louvain_partition(graph, resolution=resolution, seed=seed)

		if score > best_score:
			best_partition = partition
			best_score = score

	return best_partition, best_score


def best_louvain_partition_search(
	graph: nx.Graph, seed: int = 28, max_iter: int = 8
) -> Tuple[Dict[int, int], float, float]:
	"""
	Find the best Louvain partition by exploring resolution values.
	"""
	base_resolution = 1.0
	step = 0.1

	best_partition, best_score = louvain_partition(
		graph, resolution=base_resolution, seed=seed
	)
	best_resolution = base_resolution

	lower_res = base_resolution - step
	upper_res = base_resolution + step
	lower_partition, lower_score = louvain_partition(
		graph, resolution=lower_res, seed=seed
	)
	upper_partition, upper_score = louvain_partition(
		graph, resolution=upper_res, seed=seed
	)

	if lower_score > best_score and lower_score >= upper_score:
		direction = -1
		best_partition, best_score, best_resolution = (
			lower_partition,
			lower_score,
			lower_res,
		)
	elif upper_score > best_score:
		direction = 1
		best_partition, best_score, best_resolution = (
			upper_partition,
			upper_score,
			upper_res,
		)
	else:
		direction = 0

	if direction != 0:
		current_resolution = best_resolution
		iter_count = 0
		while iter_count < max_iter:
			next_resolution = current_resolution + (direction * step)
			if next_resolution <= 0:
				break

			next_partition, next_score = louvain_partition(
				graph, resolution=next_resolution, seed=seed
			)
			if next_score > best_score:
				best_partition, best_score, best_resolution = (
					next_partition,
					next_score,
					next_resolution,
				)
				current_resolution = next_resolution
				iter_count += 1
			else:
				break

	fine_step = 0.02
	fine_direction = -direction
	if fine_direction != 0:
		temp_resolution = best_resolution
		iter_count = 0
		while iter_count < max_iter:
			next_resolution = temp_resolution + (fine_direction * fine_step)
			if next_resolution <= 0:
				break

			next_partition, next_score = louvain_partition(
				graph, resolution=next_resolution, seed=seed
			)
			if next_score > best_score:
				best_partition, best_score, best_resolution = (
					next_partition,
					next_score,
					next_resolution,
				)
				temp_resolution = next_resolution
				iter_count += 1
			else:
				break

	return best_partition, best_score, best_resolution


def leiden_partition(
	graph: nx.Graph,
	resolution: float = 1.0,
	seed: int = 28,
	markov_time: float | None = None,
) -> Tuple[Dict[int, int], float]:
	"""
	Run Leiden using leidenalg and return the partition map plus modularity.
	"""
	# Convert NetworkX graph to iGraph
	nodes = sorted(list(graph.nodes()))
	node_to_idx = {node: i for i, node in enumerate(nodes)}

	ig_graph = ig.Graph(len(nodes), directed=False)
	edges = []
	weights = []
	for u, v, data in graph.edges(data=True):
		edges.append((node_to_idx[u], node_to_idx[v]))
		weights.append(data.get("weight", 1.0))

	ig_graph.add_edges(edges)
	ig_graph.es["weight"] = weights

	# Run Leiden
	partition = la.find_partition(
		ig_graph,
		la.RBConfigurationVertexPartition,
		resolution_parameter=resolution,
		weights="weight",
		seed=seed,
	)

	# Convert back to dict format: node -> community_id
	communities = {}
	communities_list = []
	for i, comm in enumerate(partition):
		community_nodes = [nodes[idx] for idx in comm]
		communities_list.append(set(community_nodes))
		for node in community_nodes:
			communities[node] = i

	# Calculate modularity score using NetworkX utility for consistency
	try:
		score = modularity(graph, communities_list, weight="weight", resolution=1.0)
	except ZeroDivisionError:
		score = 0.0
	return communities, score


def best_leiden_partition_random(
	graph: nx.Graph,
	seed: int = 28,
	n_samples: int = 20,
	resolution: float = 1.0,
) -> Tuple[Dict[int, int], float]:
	"""
	Find the best Leiden partition by random sampling of resolution values.
	"""
	rng = np.random.RandomState(seed)

	# Sample resolution values uniformly
	seeds = rng.randint(0, 10000, n_samples).tolist()

	best_partition = None
	best_score = -1.0

	for seed in seeds:
		# Use deterministic seed derived from base seed
		partition, score = leiden_partition(graph, resolution=resolution, seed=seed)

		if score > best_score:
			best_partition = partition
			best_score = score

	return best_partition, best_score


def infomap_partition(
	graph: nx.Graph,
	seed: int = 28,
	markov_time: float | None = None,
	num_trials: int = 20,
	resolution: float = 1.0,
) -> Tuple[Dict[int, int], float]:
	"""
	Run Infomap and return the partition map plus modularity.
	"""
	if Infomap is None:
		raise ImportError(
			"Infomap is not installed. Install it with: pip install infomap"
		)

	nodes = list(graph.nodes())
	node_to_id = {node: idx for idx, node in enumerate(nodes)}
	id_to_node = {idx: node for node, idx in node_to_id.items()}

	if markov_time is None:
		markov_time = ut.get_markov_time(resolution)

	im = Infomap(
		silent=True,
		seed=seed,
		num_trials=num_trials,
		markov_time=markov_time,
		flow_model="undirected",
	)

	for node_id in node_to_id.values():
		im.add_node(node_id)

	for u, v, data in graph.edges(data=True):
		weight = float(data.get("weight", 0.0))
		im.add_link(node_to_id[u], node_to_id[v], weight)

	im.run()

	communities: Dict[int, int] = {}
	for tree_node in im.tree:
		if tree_node.is_leaf:
			original_node = id_to_node.get(tree_node.node_id)
			if original_node is not None:
				communities[original_node] = int(tree_node.module_id)

	# Ensure isolated nodes are still assigned to a valid singleton community.
	next_singleton_comm = max(communities.values(), default=-1) + 1
	for node in nodes:
		if node not in communities:
			communities[node] = next_singleton_comm
			next_singleton_comm += 1

	communities_list_dict: Dict[int, set] = {}
	for node, community_id in communities.items():
		communities_list_dict.setdefault(community_id, set()).add(node)
	communities_list = list(communities_list_dict.values())

	try:
		score = modularity(graph, communities_list, weight="weight", resolution=1.0)
	except ZeroDivisionError:
		score = 0.0
	return communities, score


def best_infomap_partition_random(
	graph: nx.Graph,
	seed: int = 28,
	n_samples: int = 20,
	resolution: float = 1.0,
) -> Tuple[Dict[int, int], float]:
	"""
	Find the best Infomap partition by random sampling of markov_time values.
	"""

	best_partition, best_score = infomap_partition(
		graph, resolution=resolution, seed=seed, num_trials=n_samples
	)

	return best_partition, best_score


def girvan_newman_partition(
	graph: nx.Graph, max_levels: int = 20, resolution: float = 1.0
) -> Tuple[Dict[int, int], float]:
	"""
	Run Girvan-Newman and return the best partition up to max_levels plus modularity.
	"""
	best_partition = None
	best_score = -1.0

	for level, communities in enumerate(girvan_newman(graph)):
		if level >= max_levels:
			break
		communities_list = [set(c) for c in communities]
		try:
			score = modularity(
				graph, communities_list, weight="weight", resolution=resolution
			)
		except ZeroDivisionError:
			score = 0.0
		if score > best_score:
			best_score = score
			best_partition = {
				node: community_id
				for community_id, community_nodes in enumerate(communities_list)
				for node in community_nodes
			}

	if best_partition is None:
		best_partition = {node: 0 for node in graph.nodes}
		best_score = 0.0

	return best_partition, best_score


def best_partition(
	graph: nx.Graph, algorithm: Any, parameters: List[Dict[str, Any]]
) -> Tuple[Dict[int, int], float]:
	"""
	Compute the best partition using the provided community detection algorithm.
	"""
	best_score = -1.0
	best_partition = None
	for params in parameters:
		partition, score = algorithm(graph, **params)
		if score > best_score:
			best_score = score
			best_partition = partition
	return best_partition, best_score


def local_modularity_weighted(
	graph: nx.Graph,
	nodes: set[int],
	gamma: float = 1.0,
) -> float:
	strength_total = sum(dict(graph.degree(weight="weight")).values())
	if strength_total == 0:
		return 0.0

	strength_community = 0.0
	for node in nodes:
		strength_community += graph.degree(node, weight="weight")

	internal_weight = 0.0
	for node in nodes:
		for neighbor in graph.neighbors(node):
			if neighbor in nodes:
				internal_weight += graph[node][neighbor].get("weight", 1.0)

	fraction_real = internal_weight / strength_total
	fraction_expected = (strength_community / strength_total) ** 2
	return fraction_real - (gamma * fraction_expected)


# Community-persistence analysis helpers

def build_co_occurrence_matrix(
	sub_df: "pd.DataFrame", layers: list[str]
) -> "np.ndarray":
	"""Compute pairwise node co-occurrence frequency matrix M across alpha layers.

	M[i, j] is the fraction of layers in which nodes i and j were assigned to
	the same community (NaN assignments count as absent). Diagonal is set to 1.
	"""
	import pandas as pd

	n = len(sub_df)
	co = np.zeros((n, n), dtype=float)
	for col in layers:
		labels = sub_df[col].values
		not_na = pd.notna(labels)
		eq = (labels[:, None] == labels[None, :]) & (not_na[:, None]) & (not_na[None, :])
		co += eq.astype(float)
	M = co / float(len(layers))
	np.fill_diagonal(M, 1.0)
	return M


def detect_persistence_isolates(
	sub_df: "pd.DataFrame", last_col: str
) -> "tuple[list, list]":
	"""Return (core_nodes, isolate_nodes) based on the strictest filtration layer.

	Nodes with NaN or singleton-community assignments at *last_col* are isolates.
	"""
	vc = sub_df[last_col].value_counts()
	singleton_comms = vc[vc == 1].index
	mask = sub_df[last_col].isna() | sub_df[last_col].isin(singleton_comms)
	return sub_df[~mask].index.tolist(), sub_df[mask].index.tolist()


def compute_persistence_meta_groups(
	M: "np.ndarray",
	core_nodes: list,
	valid_nodes: list,
	*,
	max_meta_groups: int = 6,
	meta_group_label: str = "G",
) -> "tuple":
	"""Complete-linkage clustering on core nodes with tie-breaking.

	Returns:
		(Z_core, meta_clusters, cluster_counts, cluster_to_g, node_to_meta_core)
	"""
	import scipy.cluster.hierarchy as sch
	import scipy.spatial.distance as ssd
	import pandas as pd

	core_idx = [valid_nodes.index(n) for n in core_nodes]
	M_core = M[np.ix_(core_idx, core_idx)]
	D_core = np.clip(1.0 - M_core, 0.0, 1.0)
	np.fill_diagonal(D_core, 0.0)

	Z_core = sch.linkage(ssd.squareform(D_core, checks=False), method="complete")

	# Tie-break: ensure strictly monotone distances so fcluster maxclust works
	for i in range(1, len(Z_core)):
		if Z_core[i, 2] <= Z_core[i - 1, 2]:
			Z_core[i, 2] = Z_core[i - 1, 2] + 1e-7

	k = min(max_meta_groups, len(core_nodes))
	meta_clusters = sch.fcluster(Z_core, t=k, criterion="maxclust")

	cluster_counts = pd.Series(meta_clusters).value_counts()
	cluster_order = cluster_counts.index.tolist()
	cluster_to_g = {c: f"{meta_group_label}{i + 1}" for i, c in enumerate(cluster_order)}
	node_to_meta_core = {n: cluster_to_g[meta_clusters[i]] for i, n in enumerate(core_nodes)}

	return Z_core, meta_clusters, cluster_counts, cluster_to_g, node_to_meta_core


def build_persistence_full_linkage(
	Z_core: "np.ndarray",
	n_core: int,
	n_isolates: int,
) -> "np.ndarray":
	"""Graft isolate nodes onto the core linkage matrix at artificially high distances.

	Keeps isolates out of the core dendrogram while still allowing clustermap
	to plot them as a visually separated group.
	"""
	if n_isolates == 0:
		return Z_core
	if n_core == 0:
		return np.array([])

	offset = n_isolates  # isolates sit at indices n_core .. n_core+n_isolates-1
	new_Z: list = []
	for c1, c2, d, s in Z_core:
		c1 = int(c1) + (offset if int(c1) >= n_core else 0)
		c2 = int(c2) + (offset if int(c2) >= n_core else 0)
		new_Z.append([c1, c2, d, s])

	current_root = (n_core + n_isolates) + n_core - 2
	for i in range(n_isolates):
		new_Z.append([current_root, n_core + i, 1.05 + 0.01 * i, new_Z[-1][3] + 1])
		current_root += 1

	return np.array(new_Z)
