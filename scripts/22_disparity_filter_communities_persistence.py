import re
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts import *

# Tunable parameters
MAX_META_GROUPS: int = 6  # Maximum number of hierarchical meta-groups
META_GROUP_LABEL: str = "G"  # Meta-group label prefix  → G1, G2, … Gk
NOISE_LABEL: str = "G0"  # Label for isolate / noise nodes

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	class_ = snakemake.wildcards["class_"]
	dataset = snakemake.wildcards["dataset"]
	weight_function = snakemake.wildcards["weight_function"]
	algorithm = snakemake.wildcards["algorithm"]

	id_col = snakemake.config[class_]["id"]
	group_col = snakemake.config[class_].get("letra" if class_ == "ciuo" else "grupo")

	light_csv_paths = list(dict.fromkeys(snakemake.input[:-1]))
	nodelist_df = pd.read_csv(snakemake.input[-1], dtype={id_col: int})

	# Parse alpha inputs
	alpha_inputs = []
	for p_str in light_csv_paths:
		m = re.search(r"_([0-9]+\.[0-9]+)_pos_", str(p_str))
		if m:
			alpha_inputs.append(
				{"path": p_str, "val": float(m.group(1)), "str": m.group(1)}
			)
	alpha_inputs.sort(key=lambda x: x["val"], reverse=True)
	layers = [x["str"] for x in alpha_inputs]
	n_layers = len(layers)

	# Build community-assignment matrix
	node_ids = sorted(nodelist_df[id_col].unique())
	comm_matrix_df = pd.DataFrame(index=node_ids)
	for item in alpha_inputs:
		df_alpha = pd.read_csv(item["path"], dtype={id_col: int})
		comm_map = df_alpha.set_index(id_col)["community"].dropna().to_dict()
		comm_matrix_df[item["str"]] = comm_matrix_df.index.map(comm_map)

	sub_df = comm_matrix_df.dropna(how="all").copy()
	valid_nodes = sub_df.index.tolist()
	n_nodes = len(valid_nodes)

	if n_nodes == 0 or n_layers == 0:
		raise ValueError("No valid community data found across the alpha sweep.")

	# Co-occurrence matrix
	M = comm.build_co_occurrence_matrix(sub_df, layers)

	# Isolate / noise detection
	core_nodes, isolate_nodes = comm.detect_persistence_isolates(sub_df, layers[-1])
	n_core, n_isolates = len(core_nodes), len(isolate_nodes)

	# Hierarchical clustering → meta-groups
	Z_core, meta_clusters, cluster_counts, cluster_to_g, node_to_meta_core = (
		comm.compute_persistence_meta_groups(
			M,
			core_nodes,
			valid_nodes,
			max_meta_groups=MAX_META_GROUPS,
			meta_group_label=META_GROUP_LABEL,
		)
	)
	k_clusters = len(cluster_to_g)

	node_to_meta = {**node_to_meta_core, **{n: NOISE_LABEL for n in isolate_nodes}}

	# Meta-group color map (for clustermap sidebars)
	mg_palette = sns.color_palette("tab10", k_clusters)
	meta_color_map = {
		f"{META_GROUP_LABEL}{i + 1}": mcolors.to_hex(mg_palette[i])
		for i in range(k_clusters)
	}
	meta_color_map[NOISE_LABEL] = "#d3d3d3"

	# Full linkage matrix (core + isolates grafted)
	Z_full = comm.build_persistence_full_linkage(Z_core, n_core, n_isolates)

	# Ordered matrix + side colors for clustermap
	ordered_nodes = core_nodes + isolate_nodes
	ordered_idx = [valid_nodes.index(n) for n in ordered_nodes]
	M_ordered = M[np.ix_(ordered_idx, ordered_idx)]
	ordered_meta_labels = [node_to_meta[n] for n in ordered_nodes]

	row_colors_meta = pd.Series(ordered_meta_labels, index=ordered_nodes).map(
		meta_color_map
	)
	side_colors = pd.DataFrame(
		{f"Meta-group ({META_GROUP_LABEL}k)": row_colors_meta}, index=ordered_nodes
	)
	if group_col and group_col in nodelist_df.columns:
		node_to_group = nodelist_df.set_index(id_col)[group_col].to_dict()
		groups_s = pd.Series(
			[node_to_group.get(n) for n in ordered_nodes], index=ordered_nodes
		)
		unique_groups = sorted(groups_s.dropna().unique())
		g_pal = sns.color_palette("Set2", len(unique_groups))
		g_color_map = {g: mcolors.to_hex(g_pal[i]) for i, g in enumerate(unique_groups)}
		side_colors["Group"] = groups_s.map(g_color_map).fillna("#d3d3d3")

	# Output 1: Alluvial diagram
	out_path_diagram = Path(snakemake.output[0])
	utils.ensure_parent_dir(out_path_diagram)

	comm_colors = pl.propagate_community_colors(sub_df, layers)
	layer_comms = pl.build_alluvial_layout(sub_df, layers, n_nodes)
	ribbons = pl.build_alluvial_ribbons(sub_df, layers, layer_comms, comm_colors)

	pl.plot_alluvial_diagram(
		layer_comms,
		ribbons,
		comm_colors,
		layers,
		n_nodes,
		title=rf"Community Persistence ($\alpha$: {layers[0]} $\to$ {layers[-1]})",
		output_path=out_path_diagram,
	)

	# Output 2: Hierarchical clustering clustermap
	out_path_hier = Path(snakemake.output[1])
	utils.ensure_parent_dir(out_path_hier)

	pl.plot_persistence_clustermap(
		M_ordered,
		ordered_nodes,
		Z_full,
		side_colors,
		k_clusters=k_clusters,
		meta_group_label=META_GROUP_LABEL,
		noise_label=NOISE_LABEL,
		output_path=out_path_hier,
	)

	# ARI transition log
	ari_entries = [
		f"  {layers[k]} -> {layers[k + 1]}: "
		f"{metrics.ari_between_columns(sub_df, layers[k], layers[k + 1]):.4f}"
		for k in range(n_layers - 1)
	]

	# Logging
	log_lines: list[str] = [
		"=" * 60,
		"DISPARITY FILTER COMMUNITIES PERSISTENCE",
		"=" * 60,
	]
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SWEEP SETTINGS",
		[
			f"Alpha steps ({n_layers}, descending): {layers}",
			f"Total valid nodes: {n_nodes}",
			f"Core nodes: {n_core}  |  Isolate/Noise ({NOISE_LABEL}): {n_isolates}",
			f"Meta-groups: {k_clusters} (MAX_META_GROUPS={MAX_META_GROUPS})",
		],
	)
	cluster_order = cluster_counts.index.tolist()
	log.add_notes(
		log_lines,
		"META-GROUP SIZES",
		[
			f"  {cluster_to_g[lbl_id]}: {cluster_counts[lbl_id]} nodes "
			f"({100 * cluster_counts[lbl_id] / n_nodes:.1f}%)"
			for lbl_id in cluster_order
		]
		+ (
			[f"  {NOISE_LABEL}: {n_isolates} nodes ({100 * n_isolates / n_nodes:.1f}%)"]
			if n_isolates > 0
			else []
		),
	)
	log.add_notes(log_lines, "ARI TRANSITION METRICS (step-by-step)", ari_entries)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
