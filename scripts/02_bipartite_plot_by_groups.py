from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	bigraph = nx.read_gexf(snakemake.input[0], node_type=int)
	caes_meta = snakemake.config["caes"]
	ciuo_meta = snakemake.config["ciuo"]

	caes_df = pd.read_csv(snakemake.input[1], dtype={caes_meta["id"]: int})
	ciuo_df = pd.read_csv(snakemake.input[2], dtype={ciuo_meta["id"]: int})

	color_map_global: dict[int, str] = {}
	label_map_groups: dict[int, str] = {}
	node_size_map_workers: dict[int, float] = {}

	caes_id_col = caes_meta["id"]
	ciuo_id_col = ciuo_meta["id"]
	caes_group_col = caes_meta["grupo"]
	ciuo_group_col = ciuo_meta["grupo"]
	caes_group_color_col = caes_meta["grupo_color"]
	ciuo_group_color_col = ciuo_meta["grupo_color"]

	for _, row in caes_df.iterrows():
		node_id = int(row[caes_id_col])
		group_label = row.get(caes_group_col, None)
		label_map_groups[node_id] = (
			"Unknown" if pd.isna(group_label) else str(group_label)
		)
		color_value = row.get(caes_group_color_col, None)
		if pd.notna(color_value):
			color_map_global[node_id] = utils.parse_color(color_value)

	for _, row in ciuo_df.iterrows():
		node_id = int(row[ciuo_id_col])
		group_label = row.get(ciuo_group_col, None)
		label_map_groups[node_id] = (
			"Unknown" if pd.isna(group_label) else str(group_label)
		)
		color_value = row.get(ciuo_group_color_col, None)
		if pd.notna(color_value):
			color_map_global[node_id] = utils.parse_color(color_value)

	for node_id in bigraph.nodes():
		if node_id not in color_map_global: #TODO: remove uknown label in the plot
			color_map_global[node_id] = pl.LIGTHGRAY
		if node_id not in label_map_groups:
			label_map_groups[node_id] = "Unknown"

	caes_worker_counts = caes_df.set_index(caes_id_col)["n_obs"].to_dict()
	ciuo_worker_counts = ciuo_df.set_index(ciuo_id_col)["n_obs"].to_dict()
	node_size_map_workers = {**caes_worker_counts, **ciuo_worker_counts}

	pl.draw_bipartite_by_color(
		graph=bigraph,
		color_map=color_map_global,
		label_map=label_map_groups,
		top_n=None,
		output_path=snakemake.output[0],
		title=None,
		save=True,
		figsize=snakemake.config["figsizes"]["bipartite"],
		node_size_map=node_size_map_workers,
		factor_node_size=0.6,
		node_size_exponent=0.8,
	)

	degrees = gc.degree_sequences(
		bigraph,
		caes_partition=caes_meta.get("partition", 1),
		ciuo_partition=ciuo_meta.get("partition", 0),
	)

	colors = {"all": "#4c4c4c", "caes": "#2c7bb6", "ciuo": "#d95f0e"}

	pl.plot_degree_histograms(
		degrees=degrees,
		colors=colors,
		output_path=snakemake.output[1],
		logscale=utils._as_bool(snakemake.wildcards["logscale"]),
	)


if __name__ == "__main__":
	main()
