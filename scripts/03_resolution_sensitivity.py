from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

nx.config.warnings_to_ignore.add("cache")

snakemake: Any

algorithm_order = ["infomap", "louvain", "leiden"]
color_map = {"infomap": "#4B8BBE", "louvain": "#4CB391", "leiden": "#F8766D"}
marker_map = {"infomap": "+", "louvain": "X", "leiden": "s"}


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	df = pd.read_csv(snakemake.input[0])
	df_scores = pd.read_csv(snakemake.input[1])
	class_ = str(snakemake.wildcards["class_"])
	reference_resolution = float(snakemake.config["community"]["resolution"][class_])

	# Plotting general trends in number of communities
	figsize = snakemake.config["figsizes"]["catplot"]
	fig, ax = plt.subplots()
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
	# TODO: Add vertical lines for each algorithm's "best" resolution
	ax.vlines(
		x=reference_resolution,
		ymin=df["num_communities"].min(),
		ymax=df["num_communities"].max(),
		linestyles="dashed",
		color="gray",
		label="reference resolution",
		zorder=1,
	)
	ax.legend(title="algorithm")
	ax.grid(True)
	ax.set_xscale("log")
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
		g.ax_joint.vlines(
			x=reference_resolution,
			ymin=data_subset["score"].min(),
			ymax=data_subset["score"].max(),
			linestyles="dashed",
			color="gray",
			label="reference resolution",
		)
		legend_handles, legend_labels = g.ax_joint.get_legend_handles_labels()
		label_to_handle = dict(zip(legend_labels, legend_handles))
		g.ax_joint.legend(
			[
				label_to_handle[label]
				for label in algorithm_order + ["reference resolution"]
			],
			algorithm_order + ["reference resolution"],
			title="algorithm",
		)

		g.fig.suptitle(f"{score_type} by Resolution", y=1.02)
		g.set_axis_labels("Resolution", "Score")
		g.ax_joint.set_xscale("log")
		g.ax_joint.grid(True)
		output_path = str(snakemake.output[0]).replace(".png", f"_{score_type}.png")

		plt.savefig(output_path, bbox_inches="tight")
		plt.close()


if __name__ == "__main__":
	main()
