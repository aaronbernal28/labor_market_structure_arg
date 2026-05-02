from typing import Any
from pathlib import Path
from scripts import *
import networkx as nx
import pandas as pd

snakemake: Any


def main() -> None:
	enes_df = pd.read_csv(
		snakemake.input[0],
		dtype={
			snakemake.config["caes"]["id"]: int,
			snakemake.config["ciuo"]["id"]: int,
		},
	)

	caes_id = snakemake.config["caes"]["id"]
	ciuo_id = snakemake.config["ciuo"]["id"]

	graph = gc.build_bipartite_graph(
		enes_df,
		caes_id,
		ciuo_id,
		logscale=False,
		caes_partition=snakemake.config["caes"]["partition"],
		ciuo_partition=snakemake.config["ciuo"]["partition"],
	)

	metric_results = metrics.summarize_graph(graph)
	metrics.log_graph_metrics("Bipartite graph", metric_results)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("BIPARTITE GRAPH")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_dataframe_info(
		log_lines,
		"ENES DATA",
		row_count=len(enes_df),
		column_count=len(enes_df.columns),
	)
	log.add_graph_metrics(log_lines, "Bipartite graph metrics", metric_results)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	graph_path = Path(snakemake.output[0])
	nx.write_gexf(graph, graph_path)


if __name__ == "__main__":
	main()
