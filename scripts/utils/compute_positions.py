from typing import Any
from scripts import *
import networkx as nx
import pandas as pd

snakemake: Any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]
	seed = int(snakemake.config["seed"])
	dataset_df = pd.read_csv(snakemake.input[1], dtype={id_col: int})
	input_metrics = metrics.summarize_graph(graph)

	pos = gc.get_projection_positions(
		graph,
		seed=seed,
		spring_layout_iterations=1000,
		spring_layout_k=None,
		rotate=False,
		method="auto",
	)
	dataset_df = dl.insert_positions(dataset_df, pos, id_col=id_col)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("COMPUTE POSITIONS")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PARAMETERS",
		[
			f"Class: {class_}",
			f"Seed: {seed}",
			f"Position count: {len(pos)}",
		],
	)
	log.add_dataframe_info(
		log_lines,
		"NODELIST WITH POSITIONS",
		row_count=len(dataset_df),
		column_count=len(dataset_df.columns),
	)
	log.add_graph_metrics(log_lines, "Input projection metrics", input_metrics)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	dataset_df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
