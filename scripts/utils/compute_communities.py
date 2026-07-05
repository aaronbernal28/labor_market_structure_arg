import random
from typing import Any

from narwhals import col
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	dataset = snakemake.wildcards["dataset"]
	alpha = float(snakemake.wildcards.get("alpha", 0.05))
	seed = int(snakemake.config["seed"])
	resolution = float(
		snakemake.config["community"]["resolution"][f"{alpha:.2f}"][class_]
	)

	id_col = snakemake.config[class_]["id"]
	nodelist_df = pd.read_csv(snakemake.input[1], dtype={id_col: int})
	if id_col not in nodelist_df.columns:
		raise KeyError(f"Missing '{id_col}' column in {class_}_{dataset}.csv.")
	graph = graph.subgraph(nodelist_df[id_col].unique()).copy()

	algorithm = snakemake.wildcards["algorithm"].lower()
	# utils.setup_networkx_backend(algorithm=algorithm)

	if algorithm == "louvain":
		algorithm_func = comm.best_louvain_partition_random
	elif algorithm == "leiden":
		algorithm_func = comm.best_leiden_partition_random
	elif algorithm == "infomap":
		algorithm_func = comm.best_infomap_partition_random
	else:
		raise NotImplementedError(
			"Unsupported algorithm. Use one of: louvain, leiden, infomap."
		)

	# Sort graph nodes by ID to ensure consistent ordering across runs
	graph = gc.graph_sort_nodes_by_id(graph)

	communities, modularity = algorithm_func(
		graph, seed=seed, n_samples=100, resolution=resolution
	)

	print(f"Raw communities detected: {len(set(communities.values()))}")
	# n_obs = nodelist_df.set_index(id_col)["n_obs"].to_dict()
	# communities = utils.relabel_communities_by_observations(
	# communities,
	# n_obs,
	# order="desc",
	# num_communities=None # snakemake.config["community"]["max"][class_],
	# )
	communities = utils.filter_communities_by_size(communities, min_size=3)
	num_communities = len(set(communities.values()))

	print(f"Modularity score: {modularity:.4f}")
	print(f"Detected communities: {num_communities}")

	graph_metrics = metrics.summarize_graph(graph)
	translation = snakemake.config.get("translation", {})
	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("COMMUNITY DETECTION")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PARAMETERS",
		[
			f"Class: {class_}",
			f"Algorithm: {algorithm}",
			f"Seed: {seed}",
			f"Modularity: {modularity:.4f}",
			f"Detected communities: {num_communities}",
		],
	)
	log.add_graph_metrics(log_lines, "Projection metrics", graph_metrics)

	communities_int = {
		int(node): utils.label_fn(comm, len(str(num_communities)))
		for node, comm in communities.items()
	}
	nodelist_df["community"] = nodelist_df[id_col].astype(int).map(communities_int)
	nodelist_df.to_csv(snakemake.output[0], index=False)
	print(f"Saved {class_}_{dataset} communities to {snakemake.output[0]}.")
	group_col = snakemake.config[class_].get("letra" if class_ == "ciuo" else "grupo")

	def _fmt_number(value: float | None) -> str:
		if value is None or pd.isna(value):
			return "NA"
		return f"\\num{{{value:.2f}}}"

	def _mean_or_none(df: pd.DataFrame, col: str) -> float | None:
		if col not in df.columns:
			return None
		if "n_obs" in df.columns:
			valid_df = df[[col, "n_obs"]].dropna()
			if valid_df.empty:
				return None
			total_obs = valid_df["n_obs"].sum()
			if total_obs == 0:
				return None
			return float((valid_df[col] * valid_df["n_obs"]).sum() / total_obs)
		else:
			series = df[col].dropna()
			if series.empty:
				return None
			return float(series.mean())

	def _median_or_none(df: pd.DataFrame, col: str) -> float | None:
		if col not in df.columns:
			return None

		if "n_obs" in df.columns:
			valid_df = df[[col, "n_obs"]].dropna()
			if valid_df.empty:
				return None

			total_obs = valid_df["n_obs"].sum()
			if total_obs == 0:
				return None

			# Sort by the column values and reset index to ensure clean positional lookups
			valid_df = valid_df.sort_values(col).reset_index(drop=True)

			# Find the cumulative sum of weights and the halfway cutoff
			cumsum = valid_df["n_obs"].cumsum()
			cutoff = total_obs / 2.0

			# If the cumulative sum hits exactly the midpoint, average it with the next value
			if (cumsum == cutoff).any():
				pos = cumsum[cumsum == cutoff].index[0]
				val1 = valid_df.loc[pos, col]
				val2 = valid_df.loc[pos + 1, col]
				return float((val1 + val2) / 2.0)
			else:
				# Otherwise, take the first value that pushes the cumulative sum over 50%
				return float(valid_df.loc[cumsum > cutoff, col].iloc[0])

		else:
			series = df[col].dropna()
			if series.empty:
				return None
			return float(series.median())

	rows: list[str] = []
	rows.append(
		"Community & Dominant groups (by count) & Mean Female % & Mean Public Sector % & Age median & Income median & Modularity & Workers (millions) \\\\"  # noqa: E501
	)
	for comm_id, group in nodelist_df.groupby("community"):
		if len(group) <= 1:
			continue

		community_nodes = set(group[id_col].astype(int).tolist())
		local_modularity = comm.local_modularity_weighted(
			graph, community_nodes, gamma=1.0
		)

		dominant_groups_str = "NA"
		if group_col and group_col in group.columns:
			dominant_groups = group[group_col].value_counts().head(3)
			dominant_groups_items = [
				f"{idx} (\\num{{{count}}})" for idx, count in dominant_groups.items()
			]
			dominant_groups_str = (
				"\\makecell[l]{" + " \\\\ ".join(dominant_groups_items) + "}"
			)

		female_mean = _mean_or_none(group, "female_pct")
		public_mean = _mean_or_none(group, "public_sector_pct")
		age_median = _median_or_none(group, "age_median")
		income_median = _median_or_none(group, "income_median")
		workers_millions = None
		if "total_workers_weighted" in group.columns:
			workers_series = group["total_workers_weighted"].dropna()
			if not workers_series.empty:
				workers_millions = float(workers_series.sum()) / 1_000_000

		rows.append(  # Formato para latex table
			f"\\texttt{{{comm_id}}} & {dominant_groups_str} & {_fmt_number(female_mean)} & {_fmt_number(public_mean)} & {_fmt_number(age_median)} & {_fmt_number(income_median)} & \\num{{{local_modularity:.4f}}} & \\num{{{workers_millions:.4f}}} \\\\ \\hline"
		)

	log.add_notes(log_lines, "NODELIST WITH COMMUNITIES", rows)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	cols_to_boxplot = {
		"income_mean": "Income",
		"nivel_ed_mean": "Education",
		"age_mean": "Age",
	}

	nodelist_df = nodelist_df.dropna(subset=["community"])
	# Filtering greater that Cxx groups:
	nodelist_df = dl.filter_communities(
		nodelist_df,
		feature_col="community",
		max_code=snakemake.config["community"]["max"].get(class_, 98),
	)
	community_map = nodelist_df.set_index(id_col)["community"].to_dict()

	pl.plot_community_boxplots(
		df_nodes=nodelist_df,
		metrics_dict=cols_to_boxplot,
		class_=class_,
		algorithm=algorithm,
		output_path=snakemake.output[2],
		translation=translation,
	)

	group_color_col = snakemake.config[class_].get(
		"letra_color" if class_ == "ciuo" else "grupo_color"
	)

	nodelist_idx = nodelist_df.set_index(id_col)

	group_color_map = None
	if group_color_col and group_color_col in nodelist_idx.columns:
		raw_group_color_map = (
			nodelist_idx.groupby(group_col)[group_color_col].first().to_dict()
		)
		group_color_map = {}
		for group_name, color_value in raw_group_color_map.items():
			try:
				parsed_color = utils.parse_color(color_value)
				group_color_map[group_name] = mcolors.to_hex(parsed_color)
			except (ValueError, SyntaxError, TypeError):
				group_color_map[group_name] = "gray"

	pl.plot_stacked_by_group(
		df_index=nodelist_idx,
		group_col=group_col,
		community_map=community_map,
		title="",
		output_path=snakemake.output[1],
		group_color_map=group_color_map,
		legend_title=group_col,
		figsize=tuple(snakemake.config["figsizes"]["stacked"].get(class_, (6, 4))),
		save=True,
		percentage=False,
		translation=translation,
	)


if __name__ == "__main__":
	main()
