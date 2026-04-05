from scripts import *
import networkx as nx

snakemake: any


def main() -> None:
	graph = nx.read_gexf(snakemake.input[0])
	class_name = snakemake.wildcards["class_"]
	weight_function_name = snakemake.wildcards["weight_function"]

	print(
		f"Building projection for class '{class_name}' using weight function '{weight_function_name}'..."
	)
	weight_function = gc.get_weight_function(weight_function_name)

	projection = gc.generic_weighted_projected_graph(
		graph,
		target_partition=snakemake.config[class_name]["partition"],
		weight_function=weight_function,
	)
	nx.write_gexf(projection, snakemake.output[0])


if __name__ == "__main__":
	main()
