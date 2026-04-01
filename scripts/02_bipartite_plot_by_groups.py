from snakemake.script import snakemake
from src import (
	fceyn_plot_bipartite_degree_distribution,
	fceyn_plot_bipartite_layout_by_groups,
)
import networkx as nx


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])

	fig = fceyn_plot_bipartite_layout_by_groups(
		graph, title="Bipartite layout by groups"
	)
	fig.savefig(snakemake.output[0], bbox_inches="tight")

	fig = fceyn_plot_bipartite_degree_distribution(
		graph, title="Bipartite degree distribution"
	)
	fig.savefig(snakemake.output[1], bbox_inches="tight")


if __name__ == "__main__":
	main()
