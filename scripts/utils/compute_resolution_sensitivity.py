from typing import Any
from scripts import *
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

nx.config.warnings_to_ignore.add("cache")

snakemake: Any

RESOLUTIONS = np.geomspace(0.1, 30, num=40)
TRYS = 15


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	algorithms = sorted(list(snakemake.params["algorithms"]))

	rng = np.random.default_rng(snakemake.config["seed"])
	seeds = rng.integers(low=0, high=2**16 - 1, size=TRYS).tolist()

	# Create empty DataFrame with explicit dtypes for columns
	df = pd.DataFrame(
		{
			"algorithm": pd.Series(dtype="object"),
			"resolution": pd.Series(dtype="float64"),
			"num_communities": pd.Series(dtype="int64"),
			"seed": pd.Series(dtype="int64"),
			"modularity": pd.Series(dtype="float64"),
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
				edges = sorted([tuple(sorted(e)) for e in G_perturbed.edges()])
				num_to_remove = int(len(edges) * edge_drop_fraction)

				# Use seeded numpy RNG to guarantee deterministic sub-sampling
				rng_seed = np.random.default_rng(seed)
				edges_arr = np.array(edges, dtype=object)
				sampled_indices = rng_seed.choice(len(edges_arr), size=num_to_remove, replace=False)
				edges_to_remove = [tuple(edges_arr[i]) for i in sampled_indices]

				G_perturbed.remove_edges_from(edges_to_remove)

				# Apply node shuffling to avoid graph dict internal order bias
				original_nodes = sorted(list(G_perturbed.nodes()))
				shuffled_nodes = original_nodes.copy()
				rng_shuffle = np.random.default_rng(seed + 1)
				rng_shuffle.shuffle(shuffled_nodes)
				mapping = {old: new for old, new in zip(original_nodes, shuffled_nodes)}
				inverse_mapping = {new: old for old, new in mapping.items()}
				G_perturbed = nx.relabel_nodes(G_perturbed, mapping)

				if algorithm == "louvain":
					communities_raw, modularity = comm.louvain_partition(
						G_perturbed, seed=seed, resolution=resolution
					)
				elif algorithm == "leiden":
					communities_raw, modularity = comm.leiden_partition(
						G_perturbed, seed=seed, resolution=resolution
					)
				elif algorithm == "infomap":
					communities_raw, modularity = comm.infomap_partition(
						G_perturbed,
						seed=seed,
						resolution=resolution,
					)
				else:
					raise NotImplementedError(
						f"Unsupported algorithm: {algorithm}. Use one of: louvain, leiden, infomap."
					)

				communities = {inverse_mapping[node]: c for node, c in communities_raw.items()}

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
								"modularity": [modularity],
							}
						),
					],
					ignore_index=True,
				)
				#communities = utils.relabel_communities_by_observations(
				#	communities, order="desc"
				#)
				nodes_sorted = sorted(list(graph.nodes()))
				nodes_sorted_communities_labels.append([communities[n] for n in nodes_sorted])

			# Compute pairwise AMI/NMI between all partitions for this algorithm/resolution
			print(
				f"Computing pairwise AMI/NMI for {algorithm} at resolution {resolution:.2f}..."
			)
			for i in range(len(nodes_sorted_communities_labels)):
				for j in range(i + 1, len(nodes_sorted_communities_labels)):
					ami = adjusted_mutual_info_score(
						list(nodes_sorted_communities_labels[i]),
						list(nodes_sorted_communities_labels[j]),
					)

					nmi = normalized_mutual_info_score(
						list(nodes_sorted_communities_labels[i]),
						list(nodes_sorted_communities_labels[j]),
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
