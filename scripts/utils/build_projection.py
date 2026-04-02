from scripts import *
import networkx as nx


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])
	projection = graph.copy()
	nx.write_gexf(projection, snakemake.output[0])


if __name__ == "__main__":
	main()
