from snakemake.script import snakemake
import networkx as nx
import pandas as pd


def main() -> None:
	_ = pd.read_csv(snakemake.input[0])
	_ = pd.read_csv(snakemake.input[1])
	_ = pd.read_csv(snakemake.input[2])

	graph = nx.Graph()
	nx.write_gexf(graph, snakemake.output[0])


if __name__ == "__main__":
	main()
