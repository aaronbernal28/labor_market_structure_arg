from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	bigraph = nx.read_gexf(snakemake.input[0], node_type=int)
	caes_meta = snakemake.config["caes"]
	ciuo_meta = snakemake.config["ciuo"]

	caes_df = pd.read_csv(snakemake.input[1], dtype={caes_meta["id"]: int})
	ciuo_df = pd.read_csv(snakemake.input[2], dtype={ciuo_meta["id"]: int})

	caes_id_col = caes_meta["id"]
	ciuo_id_col = ciuo_meta["id"]
	caes_worker_counts = caes_df.set_index(caes_id_col)["n_obs"].to_dict()
	ciuo_worker_counts = ciuo_df.set_index(ciuo_id_col)["n_obs"].to_dict()
	node_size_map_workers = {**caes_worker_counts, **ciuo_worker_counts}

	factor_node_size_caes = snakemake.config["FACTOR_NODE_SIZE"]["caes"] / 15
	factor_node_size_ciuo = snakemake.config["FACTOR_NODE_SIZE"]["ciuo"] / 15

	pl.draw_bipartite_simple(
		graph=bigraph,
		caes_df=caes_df,
		ciuo_df=ciuo_df,
		caes_meta=caes_meta,
		ciuo_meta=ciuo_meta,
		output_path=snakemake.output[0],
		figsize=snakemake.config["figsizes"]["bipartite_simple"],
		node_size_map=node_size_map_workers,
		factor_node_size_caes=factor_node_size_caes,
		factor_node_size_ciuo=factor_node_size_ciuo,
	)

	# Logging
	log_lines = []
	log_lines.append("=" * 60)
	log_lines.append("SIMPLE BIPARTITE PLOT")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_dataframe_info(
		log_lines,
		"CAES NODELIST",
		row_count=len(caes_df),
		column_count=len(caes_df.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"CIUO NODELIST",
		row_count=len(ciuo_df),
		column_count=len(ciuo_df.columns),
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
