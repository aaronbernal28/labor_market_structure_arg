from pathlib import Path
from scripts import *
import networkx as nx
from scripts import *

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])
	alpha = float(snakemake.wildcards["alpha"])
	print(f"Filtering projection graph with alpha={alpha}...")

	if alpha < 1.0:
		raise NotImplementedError("Filtering with alpha < 1.0 is not implemented yet.")

	output_path = Path(snakemake.output[0])
	nx.write_gexf(graph, output_path)

	# fig = lcd_plot_alpha_sensitivity(graph, title="Alpha sensitivity")
	# fig.savefig(snakemake.output[1], bbox_inches="tight")


if __name__ == "__main__":
	main()
