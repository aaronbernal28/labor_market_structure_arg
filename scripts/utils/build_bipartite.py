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
		max_caes_id=snakemake.config["max_caes_id"],
	)

	translation = utils.get_config_section(snakemake.config, "translation")
	label_caes = utils.translate_label("Sector", translation)
	label_ciuo = utils.translate_label("Ocupacion", translation)
	partition_labels = {
		snakemake.config["caes"]["partition"]: label_caes,
		snakemake.config["ciuo"]["partition"]: label_ciuo,
	}
	metric_results = metrics.summarize_bipartite_graph(
		graph, partition_labels=partition_labels
	)
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
	log.add_bipartite_degree_strength_latex(
		log_lines,
		"LATEX RESUMEN GRADOS Y FUERZAS",
		metric_results,
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	graph_path = Path(snakemake.output[0])
	nx.write_gexf(graph, graph_path)

	dataset = snakemake.wildcards.get("dataset", "unknown")
	dataset_label = snakemake.config["datasets"]["labels"].get(dataset, dataset)
	caes_partition = snakemake.config["caes"]["partition"]
	ciuo_partition = snakemake.config["ciuo"]["partition"]

	description = (
		f"Bipartite graph of sectors and occupations from '{dataset_label}' dataset. "
		f"Parameters: CAES ID field = '{caes_id}', CIUO ID field = '{ciuo_id}', "
		f"CAES partition = {caes_partition}, CIUO partition = {ciuo_partition}."
	)

	utils.update_gexf_metadata(
		filepath=snakemake.output[0],
		creator="NetworkX + Labor Market Structure Pipeline",
		description=description,
	)


if __name__ == "__main__":
	main()
