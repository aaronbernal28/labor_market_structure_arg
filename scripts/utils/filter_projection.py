from snakemake.script import snakemake
from src import fceyn_plot_alpha_sensitivity
import networkx as nx


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])
	nx.write_gexf(graph, snakemake.output[0])

	fig = fceyn_plot_alpha_sensitivity(graph, title="Alpha sensitivity")
	fig.savefig(snakemake.output[1], bbox_inches="tight")


if __name__ == "__main__":
	main()
