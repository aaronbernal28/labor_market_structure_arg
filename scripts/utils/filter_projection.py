from pathlib import Path
from scripts import *
import networkx as nx
from scripts import *

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	alpha = float(snakemake.wildcards["alpha"])
	print(f"Filtering projection graph with alpha={alpha}...")
	input_metrics = metrics.summarize_graph(graph)

	if alpha < 1.0:
		backbone = gc.disparity_filter_backbone(graph, alpha=alpha)
	else:
		print("Alpha >= 1.0, skipping filtering and using original projection.")
		backbone = graph

	backbone_metrics = metrics.summarize_graph(backbone)
	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("FILTER PROJECTION")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PARAMETERS",
		[
			f"Alpha: {alpha}",
			f"Filtering applied: {alpha < 1.0}",
		],
	)
	log.add_graph_metrics(log_lines, "Input projection metrics", input_metrics)
	log.add_graph_metrics(log_lines, "Backbone metrics", backbone_metrics)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	output_path = Path(snakemake.output[0])
	nx.write_gexf(backbone, output_path)

	pl.plot_backbone_weight_histogram(
		original_weights=[d["weight"] for _, _, d in graph.edges(data=True)],
		backbone_weights=[d["weight"] for _, _, d in backbone.edges(data=True)],
		alpha=alpha,
		title_prefix=None,
		output_path=Path(snakemake.output[1]),
		save=True,
		figsize=tuple(snakemake.config["figsizes"]["histogram"]),
	)


if __name__ == "__main__":
	main()
