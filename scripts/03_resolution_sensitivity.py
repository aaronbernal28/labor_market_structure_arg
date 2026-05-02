from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import random
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

nx.config.warnings_to_ignore.add("cache")

snakemake: Any

RESOLUTIONS = np.linspace(0.5, 2.0, num=50)
TRYS = 20


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
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
				nodes = list(graph.nodes())
				random.seed(seed)
				np.random.seed(seed)
				random.shuffle(nodes)
				mapping = {old: new for old, new in zip(graph.nodes(), nodes)}
				G_shuffled = nx.relabel_nodes(graph, mapping)

				if algorithm == "louvain":
					communities, _ = comm.louvain_partition(
						G_shuffled, seed=seed, resolution=resolution
					)
				elif algorithm == "leiden":
					communities, _ = comm.leiden_partition(
						G_shuffled, seed=seed, resolution=resolution
					)
				elif algorithm == "infomap":
					communities, _ = comm.infomap_partition(
						G_shuffled,
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
				communities = utils.relabel_communities_by_size(
					communities, order="desc"
				)
				nodes_sorted_communities_labels.append(communities.values())

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

	print("Data collection complete. Sample of results:")
	print(df.head())
	print(df.info())

	print("Summary of scores:")
	print(df_scores)
	print(df_scores.info())

	# Plotting general trends in number of communities
	figsize = snakemake.config["figsizes"]["catplot"]
	sns.catplot(
		data=df,
		x="resolution",
		y="num_communities",
		hue="algorithm",
		hue_order=["infomap", "louvain", "leiden"],
		palette=["#4B8BBE", "#4CB391", "#F8766D"],
		kind="strip",
		native_scale=True,
		zorder=1,
		alpha=0.5,
		height=figsize[1],
		aspect=figsize[0] / figsize[1],
	)
	sns.regplot(
		data=df,
		x="resolution",
		y="num_communities",
		scatter=False,
		truncate=False,
		order=2,
		color=".2",
		ax=plt.gca(),
	)
	# TODO: Add vertical lines for each algorithm's "best" resolution
	plt.grid()
	plt.savefig(snakemake.output[0], bbox_inches="tight")

	# Plotting scores — create separate plots for AMI and NMI using jointplot
	for score_type in ["AMI", "NMI"]:
		# Filter data for just this score type (keep all algorithms!)
		data_subset = df_scores[df_scores["score_type"] == score_type]

		# Create the jointplot using hue
		g = sns.jointplot(
			data=data_subset,
			x="resolution",
			y="score",
			hue="algorithm",
			hue_order=["infomap", "louvain", "leiden"],
			palette=["#4B8BBE", "#4CB391", "#F8766D"],
			alpha=0.5,  # Make points transparent so overlapping is visible
		)
		# TODO: Add vertical lines for each algorithm's "best" resolution

		g.fig.suptitle(f"{score_type} by Resolution", y=1.02)
		g.set_axis_labels("Resolution", "Score")
		output_path = str(snakemake.output[0]).replace(".png", f"_{score_type}.png")

		plt.savefig(output_path, bbox_inches="tight")
		plt.close()


if __name__ == "__main__":
	main()
