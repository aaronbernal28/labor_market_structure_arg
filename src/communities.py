"""
Community detection utilities.
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
from networkx.algorithms.community import girvan_newman, louvain_communities, modularity
import numpy as np


def louvain_partition(
	graph: nx.Graph, resolution: float = 1.0, seed: int = 28
) -> Tuple[Dict[int, int], float]:
	"""
	Run Louvain and return the partition map plus modularity.
	"""
	communities_list = louvain_communities(
		graph, weight="weight", resolution=resolution, seed=seed
	)

	# Convert to dict format: node -> community_id
	communities = {node: i for i, comm in enumerate(communities_list) for node in comm}

	# Calculate modularity score
	score = modularity(graph, communities_list, weight="weight")
	return communities, score


def best_louvain_partition_random(
	graph: nx.Graph,
	seed: int = 28,
	n_samples: int = 20,
	min_resolution: float = 0.5,
	max_resolution: float = 5.0,
) -> Tuple[Dict[int, int], float, float]:
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
	np.random.seed(seed)

	# Sample resolution values uniformly
	resolutions = np.random.uniform(min_resolution, max_resolution, n_samples)

	best_partition = None
	best_score = -1.0
	best_resolution = None

	for resolution in resolutions:
		seed += 1  # Increment seed for variability
		partition, score = louvain_partition(graph, resolution=resolution, seed=seed)

		if score > best_score:
			best_partition = partition
			best_score = score
			best_resolution = resolution

	return best_partition, best_score, best_resolution


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


def girvan_newman_partition(
	graph: nx.Graph, max_levels: int = 20
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
		score = modularity(graph, communities_list, weight="weight")
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
