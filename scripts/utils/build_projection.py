from scripts import *
import networkx as nx
from src.seeding import initialize_seeds, get_seed_from_config

snakemake: any


def main() -> None:
	initialize_seeds(get_seed_from_config(snakemake.config))
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

	input_metrics = metrics.summarize_graph(graph)
	projection_metrics = metrics.summarize_graph(projection)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PROJECTION GRAPH")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"PARAMETERS",
		[
			f"Class: {class_name}",
			f"Weight function: {weight_function_name}",
		],
	)
	log.add_graph_metrics(log_lines, "Input graph metrics", input_metrics)
	log.add_graph_metrics(log_lines, "Projection graph metrics", projection_metrics)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	nx.write_gexf(projection, snakemake.output[0])


if __name__ == "__main__":
	main()
