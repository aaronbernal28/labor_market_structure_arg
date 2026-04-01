from snakemake.script import snakemake
import networkx as nx


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])
	nx.write_gexf(graph, snakemake.output[0])


if __name__ == "__main__":
	main()
