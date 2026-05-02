from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns

nx.config.warnings_to_ignore.add("cache")

snakemake: Any

RESOLUTIONS = np.linspace(0.5, 2.0, num=40)
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

	# Generate partitions and collect results
	for algorithm in algorithms:
		print(f"Running {algorithm}...")
		for resolution in RESOLUTIONS:
			print(f"Running {algorithm} with resolution {resolution:.2f}...")
			for seed in seeds:
				if algorithm == "louvain":
					communities, _ = comm.louvain_partition(
						graph, seed=seed, resolution=resolution
					)
				elif algorithm == "leiden":
					communities, _ = comm.leiden_partition(
						graph, seed=seed, resolution=resolution
					)
				elif algorithm == "infomap":
					communities, _ = comm.infomap_partition(
						graph, seed=seed, resolution=resolution, markov_time=resolution
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
	print(df.head())
	print(df.tail())
	print(df.info())

	# Plotting
	figsize = snakemake.config["figsizes"]["catplot"]
	sns.catplot(
		data=df,
		x="resolution",
		y="num_communities",
		hue="algorithm",
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
	plt.grid()
	plt.savefig(snakemake.output[0], bbox_inches="tight")


if __name__ == "__main__":
	main()
