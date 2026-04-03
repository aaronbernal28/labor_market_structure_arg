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


if __name__ == "__main__":
	main()
