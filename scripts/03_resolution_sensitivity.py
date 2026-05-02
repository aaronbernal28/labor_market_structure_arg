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
TRYS = 15
algorithm_order = ["infomap", "louvain", "leiden"]
color_map = {"infomap": "#4B8BBE", "louvain": "#4CB391", "leiden": "#F8766D"}
marker_map = {"infomap": "+", "louvain": "X", "leiden": "s"}


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
	fig, ax = plt.subplots(figsize=figsize)
	for algorithm in algorithm_order:
		algorithm_data = df[df["algorithm"] == algorithm]
		ax.scatter(
			algorithm_data["resolution"],
			algorithm_data["num_communities"],
			label=algorithm,
			marker=marker_map[algorithm],
			color=color_map[algorithm],
			alpha=0.5,
			zorder=2,
		)
	sns.regplot(
		data=df,
		x="resolution",
		y="num_communities",
		scatter=False,
		truncate=False,
		order=2,
		color=".2",
		ax=ax,
	)
	ax.legend(title="algorithm")
	# TODO: Add vertical lines for each algorithm's "best" resolution
	ax.grid(True)
	fig.savefig(snakemake.output[0], bbox_inches="tight")

	# Plotting scores — create separate plots for AMI and NMI using jointplot
	for score_type in ["AMI", "NMI"]:
		# Filter data for just this score type (keep all algorithms!)
		data_subset = df_scores[df_scores["score_type"] == score_type]

		g = sns.JointGrid(
			data=data_subset,
			x="resolution",
			y="score",
			height=figsize[1],
			space=0.1,
		)
		for algorithm in algorithm_order:
			algorithm_data = data_subset[data_subset["algorithm"] == algorithm]
			g.ax_joint.scatter(
				algorithm_data["resolution"],
				algorithm_data["score"],
				label=algorithm,
				marker=marker_map[algorithm],
				color=color_map[algorithm],
				alpha=0.5,
			)
			sns.histplot(
				algorithm_data,
				x="resolution",
				ax=g.ax_marg_x,
				color=color_map[algorithm],
				alpha=0.25,
				stat="density",
				element="step",
				common_norm=False,
			)
			sns.histplot(
				algorithm_data,
				y="score",
				ax=g.ax_marg_y,
				color=color_map[algorithm],
				alpha=0.25,
				stat="density",
				element="step",
				common_norm=False,
			)
		legend_handles, legend_labels = g.ax_joint.get_legend_handles_labels()
		label_to_handle = dict(zip(legend_labels, legend_handles))
		g.ax_joint.legend(
			[label_to_handle[algorithm] for algorithm in algorithm_order],
			algorithm_order,
			title="algorithm",
		)
		# TODO: Add vertical lines for each algorithm's "best" resolution

		g.fig.suptitle(f"{score_type} by Resolution", y=1.02)
		g.set_axis_labels("Resolution", "Score")
		output_path = str(snakemake.output[0]).replace(".png", f"_{score_type}.png")

		plt.savefig(output_path, bbox_inches="tight")
		plt.close()


if __name__ == "__main__":
	main()
