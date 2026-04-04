from pathlib import Path
from scripts import *
import networkx as nx
from scripts import *

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0], node_type=int)
	alpha = float(snakemake.wildcards["alpha"])
	print(f"Filtering projection graph with alpha={alpha}...")

	if alpha < 1.0:
		backbone = gc.disparity_filter_backbone(graph, alpha=alpha)
	else:
		print("Alpha >= 1.0, skipping filtering and using original projection.")
		backbone = graph

	output_path = Path(snakemake.output[0])
	nx.write_gexf(backbone, output_path)

	pl.plot_backbone_weight_histogram(
		original_weights=[d['weight'] for _, _, d in graph.edges(data=True)],
		backbone_weights=[d['weight'] for _, _, d in backbone.edges(data=True)],
		alpha=alpha,
		title_prefix=None,
		output_path=Path(snakemake.output[1]),
		save=True,
		figsize=tuple(snakemake.config["figsizes"]["histogram"]),
	)

if __name__ == "__main__":
	main()
