"""
Community detection utilities.
"""

from typing import Dict, Tuple

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import numpy as np


def fceyn_louvain_partition(
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


def fceyn_best_louvain_partition_random(
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
		partition, score = fceyn_louvain_partition(
			graph, resolution=resolution, seed=seed
		)

		if score > best_score:
			best_partition = partition
			best_score = score
			best_resolution = resolution

	return best_partition, best_score, best_resolution
