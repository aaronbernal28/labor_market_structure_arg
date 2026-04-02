from scripts import *
import networkx as nx


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])

	fig = lcd_plot_edge_weight_correlation(graph, title="Edge weight correlation")
	fig.savefig(snakemake.output[0], bbox_inches="tight")


if __name__ == "__main__":
	main()
