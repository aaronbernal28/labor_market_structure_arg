from pathlib import Path
from scripts import *
import networkx as nx
import pandas as pd

snakemake: any


def main() -> None:
	enes_df = pd.read_csv(
		snakemake.input[0],
		dtype={snakemake.config["id_caes"]: int, snakemake.config["id_ciuo"]: int},
	)

	caes_id = snakemake.config["id_caes"]
	ciuo_id = snakemake.config["id_ciuo"]

	graph = gc.build_bipartite_graph(
		enes_df,
		caes_id,
		ciuo_id,
		logscale=utils._as_bool(snakemake.wildcards["logscale"]),
		caes_partition=snakemake.config["caes"]["partition"],
		ciuo_partition=snakemake.config["ciuo"]["partition"],
	)

	metric_results = metrics.summarize_graph(graph)
	metrics.log_graph_metrics("Bipartite graph", metric_results)

	graph_path = Path(snakemake.output[0])
	nx.write_gexf(graph, graph_path)


if __name__ == "__main__":
	main()
