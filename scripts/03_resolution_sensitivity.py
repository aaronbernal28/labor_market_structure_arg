from typing import Any
from pathlib import Path
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

nx.config.warnings_to_ignore.add("cache")

snakemake: Any

algorithm_order = ["louvain","leiden", "infomap"]
color_map = {"infomap": "#4B8BBE", "louvain": "#4CB391", "leiden": "#F8766D"}
marker_map = {"infomap": "+", "louvain": "X", "leiden": "s"}


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	raw_inputs = list(snakemake.input)
	if raw_inputs and isinstance(raw_inputs[0], (list, tuple)):
		df_paths = list(raw_inputs[0])
		df_scores_paths = list(raw_inputs[1])
	else:
		df_paths = []
		df_scores_paths = []
		for path in raw_inputs:
			name = Path(str(path)).name
			if name.startswith("_df_scores_"):
				df_scores_paths.append(str(path))
			elif name.startswith("_df_"):
				df_paths.append(str(path))

	if not df_paths or not df_scores_paths:
		raise ValueError(
			"Expected both _df_*.csv and _df_scores_*.csv inputs for "
			"resolution sensitivity"
		)

	df = pd.read_csv(df_paths[0])
	for path in df_paths[1:]:
		df = pd.concat([df, pd.read_csv(path)], ignore_index=True)

	df_scores = pd.read_csv(df_scores_paths[0])
	for path in df_scores_paths[1:]:
		df_scores = pd.concat([df_scores, pd.read_csv(path)], ignore_index=True)

	class_ = str(snakemake.wildcards["class_"])
	reference_resolution = float(snakemake.config["community"]["resolution"][class_])
	translation = snakemake.config.get("translation", {})
	def _t(label: str) -> str:
		return utils.translate_label(label, translation)

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
		label=_t("reference resolution"),
		zorder=1,
	)
	ax.legend(title=_t("algorithm"))
	ax.grid(True)
	ax.set_xscale("log")
	ax.set_xlabel(_t("Resolution"))
	ax.set_ylabel(_t("Communities"))
	fig.savefig(snakemake.output[0], bbox_inches="tight")

	# Plotting scores — create separate plots for AMI and NMI using jointplot
	for i, score_type in enumerate((["AMI", "NMI"])):
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
			label=_t("reference resolution"),
		)
		legend_handles, legend_labels = g.ax_joint.get_legend_handles_labels()
		label_to_handle = dict(zip(legend_labels, legend_handles))
		g.ax_joint.legend(
			[
				label_to_handle[label]
				for label in algorithm_order + ["reference resolution"]
			],
			algorithm_order + [_t("reference resolution")],
			title=_t("algorithm"),
		)

		g.set_axis_labels(_t("Resolution"), _t("Score"))
		g.ax_joint.set_xscale("log")
		g.ax_joint.grid(True)
		output_path = str(snakemake.output[i+1])

		plt.savefig(output_path, bbox_inches="tight")
		plt.close()

	# Modularity violin plot by algorithm
	fig, ax = plt.subplots()
	for algorithm in algorithm_order:
		algorithm_data = df[df["algorithm"] == algorithm]
		ax.scatter(
			algorithm_data["resolution"],
			algorithm_data["modularity"],
			label=algorithm,
			marker=marker_map[algorithm],
			color=color_map[algorithm],
			alpha=0.5,
			zorder=2,
		)
	# Add label with all negatives values omitted, vertically separated
	group_neg = df.groupby("algorithm")["modularity"].apply(lambda x: x[x < 0].count())
	label_y_base = 0.05
	label_y_step = 0.06
	for idx, algorithm in enumerate(algorithm_order):
		count_neg = group_neg.get(algorithm, 0)
		if count_neg > 0:
			ax.text(
				x=reference_resolution,
				y=label_y_base + label_y_step * idx,
				s=f"{count_neg} {_t('negative values')}",
				color=color_map[algorithm],
				horizontalalignment="center",
				verticalalignment="bottom",
				fontsize="small",
			)
	ax.grid(True)
	ax.set_xscale("log")
	ax.set_xlabel(_t("Resolution"))
	ax.set_ylabel(_t("Modularity"))
	ax.set_ylim(-0.1, 1.1)
	plt.savefig(snakemake.output[3], bbox_inches="tight")
	plt.close()


if __name__ == "__main__":
	main()
