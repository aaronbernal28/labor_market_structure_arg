from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	dataset = snakemake.wildcards["dataset"]
	seed = int(snakemake.config["seed"])

	id_col = snakemake.config[class_]["id"]
	nodelist_df = pd.read_csv(snakemake.input[1], dtype={id_col: int})
	if id_col not in nodelist_df.columns:
		raise KeyError(f"Missing '{id_col}' column in {class_}_{dataset}.csv.")
	graph = graph.subgraph(nodelist_df[id_col].unique()).copy()

	algorithm = snakemake.wildcards["algorithm"].lower()

	if algorithm == "louvain":
		algorithm_func = comm.best_louvain_partition_random
		param_label = "resolution"
	elif algorithm == "leiden":
		algorithm_func = comm.best_leiden_partition_random
		param_label = "resolution"
	elif algorithm == "infomap":
		algorithm_func = comm.best_infomap_partition_random
		param_label = "markov_time"
	else:
		raise NotImplementedError(
			"Unsupported algorithm. Use one of: louvain, leiden, infomap."
		)

	communities, modularity, best_parameter = algorithm_func(
		graph, seed=seed, n_samples=50
	)
	num_communities = len(set(communities.values()))
	print(f"Modularity score: {modularity:.4f}")
	print(f"Best {param_label}: {best_parameter:.3f}")
	print(f"Detected communities: {num_communities}")

	graph_metrics = metrics.summarize_graph(graph)
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
			f"Best {param_label}: {best_parameter:.3f}",
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
	nodelist_df = nodelist_df.dropna(subset=["community"])
	nodelist_df.to_csv(snakemake.output[0], index=False)
	print(f"Saved {class_}_{dataset} communities to {snakemake.output[0]}.")
	log.add_dataframe_info(
		log_lines,
		"NODELIST WITH COMMUNITIES",
		row_count=len(nodelist_df),
		column_count=len(nodelist_df.columns),
	)

	group_col = snakemake.config[class_].get("letra" if class_ == "ciuo" else "grupo")

	for comm_id, group in nodelist_df.groupby("community"):
		metrics_list = []
		metrics_list.append(f"Size: {len(group)}")

		# Total workers weighted
		if "total_workers_weighted" in group.columns:
			tot_workers = group["total_workers_weighted"].sum()
			metrics_list.append(f"Total workers (weighted): {tot_workers:,.0f}")
		else:
			tot_workers = None

		# Dominant groups
		if group_col and group_col in group.columns:
			dominant_groups = group[group_col].value_counts().head(3)
			group_str = ", ".join(
				[f"{idx} ({count})" for idx, count in dominant_groups.items()]
			)
			metrics_list.append(f"Dominant groups (by count): {group_str}")

			if tot_workers is not None:
				dominant_groups_w = (
					group.groupby(group_col)["total_workers_weighted"]
					.sum()
					.sort_values(ascending=False)
					.head(3)
				)
				group_w_str = ", ".join(
					[f"{idx} ({val:,.0f})" for idx, val in dominant_groups_w.items()]
				)
				metrics_list.append(f"Dominant groups (by workers): {group_w_str}")

		def _compute_means(col):
			if col not in group.columns:
				return None, None
			valid = group[group[col].notna()]
			if valid.empty:
				return None, None

			unweighted = valid[col].mean()

			weighted = None
			if tot_workers is not None and "total_workers_weighted" in valid.columns:
				valid_weight = valid["total_workers_weighted"]
				if valid_weight.sum() > 0:
					weighted = np.average(valid[col], weights=valid_weight)

			return unweighted, weighted

		cols_to_avg = {
			"female_pct": "Female %",
			"public_sector_pct": "Public Sector %",
		}

		for col, prefix in cols_to_avg.items():
			unw, w = _compute_means(col)
			if unw is not None:
				w_str = f" | Weighted: {w:.2f}" if w is not None else ""
				metrics_list.append(f"Mean {prefix}: {unw:.2f}{w_str}")

		log.add_notes(log_lines, f"COMMUNITY {comm_id} METRICS", metrics_list)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	cols_to_boxplot = {
		"income_mean": "Income",
		"nivel_ed_mean": "Education",
		"age_mean": "Age",
	}

	pl.plot_community_boxplots(
		df_nodes=nodelist_df,
		metrics_dict=cols_to_boxplot,
		class_=class_,
		algorithm=algorithm,
		output_path=snakemake.output[2],
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
		community_map=communities_int,
		title=f"{class_.upper()} - Distribucion por comunidad ({algorithm})",
		output_path=snakemake.output[1],
		group_color_map=group_color_map,
		legend_title=group_col,
		figsize=tuple(snakemake.config["figsizes"]["stacked"]),
		save=True,
		percentage=False,
	)


if __name__ == "__main__":
	main()
