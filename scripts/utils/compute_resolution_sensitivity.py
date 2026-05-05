from typing import Any
from scripts import *
import networkx as nx
import pandas as pd
import numpy as np
import random
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

nx.config.warnings_to_ignore.add("cache")

snakemake: Any

RESOLUTIONS = np.geomspace(0.1, 25, num=40)
TRYS = 25


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	algorithms = list(snakemake.params["algorithms"])

	rng = np.random.default_rng(snakemake.config["seed"])
	seeds = rng.integers(low=0, high=2**16 - 1, size=TRYS).tolist()

	# Create empty DataFrame with explicit dtypes for columns
	df = pd.DataFrame(
		{
			"algorithm": pd.Series(dtype="object"),
			"resolution": pd.Series(dtype="float64"),
			"num_communities": pd.Series(dtype="int64"),
			"seed": pd.Series(dtype="int64"),
		},
		index=pd.RangeIndex(start=0, stop=0, step=1),
	)

	df_scores = pd.DataFrame(
		{
			"algorithm": pd.Series(dtype="object"),
			"resolution": pd.Series(dtype="float64"),
			"score": pd.Series(dtype="float64"),
			"score_type": pd.Series(dtype="object"),
		},
		index=pd.RangeIndex(start=0, stop=0, step=1),
	)

	# Generate partitions and collect results
	for algorithm in algorithms:
		print(f"Running {algorithm}...")
		for resolution in RESOLUTIONS:
			print(f"Running {algorithm} with resolution {resolution:.2f}...")
			nodes_sorted_communities_labels = []

			for seed in seeds:
				G_perturbed = graph.copy()

				edge_drop_fraction = 0.05
				edges = list(G_perturbed.edges())
				num_to_remove = int(len(edges) * edge_drop_fraction)
				edges_to_remove = random.sample(edges, num_to_remove)

				G_perturbed.remove_edges_from(edges_to_remove)

				if algorithm == "louvain":
					communities, _ = comm.louvain_partition(
						G_perturbed, seed=seed, resolution=resolution
					)
				elif algorithm == "leiden":
					communities, _ = comm.leiden_partition(
						G_perturbed, seed=seed, resolution=resolution
					)
				elif algorithm == "infomap":
					communities, _ = comm.infomap_partition(
						G_perturbed,
						seed=seed,
						resolution=resolution,
						markov_time=resolution,
					)
				else:
					raise NotImplementedError(
						f"Unsupported algorithm: {algorithm}. Use one of: louvain, leiden, infomap."
					)

				num_communities = len(set(communities.values()))
				df = pd.concat(
					[
						df,
						pd.DataFrame(
							{
								"seed": [seed],
								"algorithm": [algorithm],
								"resolution": [resolution],
								"num_communities": [num_communities],
							}
						),
					],
					ignore_index=True,
				)
				communities = utils.filter_communities_by_size(
					communities, min_size=3
				)
				# Store the full community dict, not just values, so we can align nodes later
				nodes_sorted_communities_labels.append(communities)

			# Compute pairwise AMI/NMI between all partitions for this algorithm/resolution
			print(
				f"Computing pairwise AMI/NMI for {algorithm} at resolution {resolution:.2f}..."
			)
			for i in range(len(nodes_sorted_communities_labels)):
				for j in range(i + 1, len(nodes_sorted_communities_labels)):
					# Get community dicts
					communities_i = nodes_sorted_communities_labels[i]
					communities_j = nodes_sorted_communities_labels[j]

					# Find common nodes between the two partitions
					common_nodes = sorted(set(communities_i.keys()) & set(communities_j.keys()))

					if len(common_nodes) == 0:
						continue  # Skip if no common nodes

					# Extract labels only for common nodes
					labels_i = np.array([communities_i[node] for node in common_nodes])
					labels_j = np.array([communities_j[node] for node in common_nodes])

					ami = adjusted_mutual_info_score(
						labels_i,
						labels_j,
					)

					nmi = normalized_mutual_info_score(
						labels_i,
						labels_j,
						average_method="geometric",
					)

					df_scores = pd.concat(
						[
							df_scores,
							pd.DataFrame(
								{
									"algorithm": [algorithm, algorithm],
									"resolution": [resolution, resolution],
									"score": [ami, nmi],
									"score_type": ["AMI", "NMI"],
								}
							),
						],
						ignore_index=True,
					)

	# Save results to CSV
	df.to_csv(snakemake.output[0], index=False)
	df_scores.to_csv(snakemake.output[1], index=False)


if __name__ == "__main__":
	main()
