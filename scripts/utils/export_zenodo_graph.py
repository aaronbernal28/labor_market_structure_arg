import pandas as pd
import networkx as nx
from typing import Any
import src.logging_utils as log
import datetime
from src.utils import update_gexf_metadata

snakemake: Any


def main() -> None:
	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]
	label_col = snakemake.config[class_]["label"]

	print(f"Loading graph from {snakemake.input.graph}...")
	G = nx.read_gexf(snakemake.input.graph, node_type=int)

	print(f"Loading nodelist from {snakemake.input.nodelist}...")
	df_nodes = pd.read_csv(snakemake.input.nodelist)
	df_nodes[id_col] = df_nodes[id_col].astype(int)
	df_nodes.set_index(id_col, inplace=True)

	print("Merging metadata into graph nodes...")
	for node in list(G.nodes()):
		node_id = int(node)
		if node_id in df_nodes.index:
			attrs = df_nodes.loc[node_id].to_dict()

			# Replace node label with the readable label name if present
			if label_col in attrs and not pd.isna(attrs[label_col]):
				G.nodes[node]["label"] = str(attrs[label_col])

			# Inject all other attributes from the nodelist
			for key, val in attrs.items():
				# Filter out NaN values to ensure valid GEXF format
				if pd.isna(val):
					continue
				G.nodes[node][key] = val
		else:
			print(f"Warning: Node {node_id} not found in nodelist index.")

	print(f"Saving merged graph to {snakemake.output.graph}...")
	nx.write_gexf(G, snakemake.output.graph)

	# Post-process XML metadata
	print("Writing Zenodo-compliant GEXF metadata attributes...")
	dataset = snakemake.wildcards.get("dataset", "unknown_dataset")
	weight_function = snakemake.wildcards.get("weight_function", "unknown_weight")
	alpha = snakemake.wildcards.get("alpha", "unknown_alpha")
	algorithm = snakemake.wildcards.get("algorithm", "unknown_algorithm")

	class_desc = ""
	if class_ == "caes":
		class_desc = (
			"Nodes represent economic activity sectors (CAES - Clasificación de Actividades Económicas para Encuestas Sociodemográficas). "
			"Edges represent statistically significant occupational similarity between sectors (i.e., sectors sharing a similar occupational composition of workers), "
			f"extracted from the bipartite sector-occupation graph using the '{weight_function}' disparity filter at significance level alpha={alpha}."
		)
	elif class_ in ["ciuo", "cno"]:
		class_desc = (
			f"Nodes represent occupational categories ({class_.upper()} - Clasificación {'Internacional Uniforme' if class_ == 'ciuo' else 'Nacional'} de Ocupaciones). "
			"Edges represent statistically significant sectoral similarity between occupations (i.e., occupations sharing a similar distribution across economic activity sectors), "
			f"extracted from the bipartite sector-occupation graph using the '{weight_function}' disparity filter at significance level alpha={alpha}."
		)
	else:
		class_desc = (
			f"Nodes represent {class_} categories. "
			f"Edges represent statistically significant similarity between these categories, "
			f"extracted using the '{weight_function}' disparity filter at significance level alpha={alpha}."
		)

	dataset_label = snakemake.config["datasets"]["labels"].get(dataset, dataset)
	description = (
		f"This unipartite projection network was generated from the '{dataset_label}' dataset of the labor market in Argentina. "
		f"{class_desc} "
		f"Node colors, positions, and community assignments were computed using the '{algorithm}' community detection algorithm."
	)

	keywords_list = [
		"labor market",
		"Argentina",
		"network analysis",
		dataset,
		class_,
		weight_function,
		f"alpha-{alpha}",
		"disparity filter",
		"backbone",
		"community detection",
		algorithm,
		"zenodo",
	]
	keywords = ", ".join(keywords_list)

	today_date = datetime.date.today().strftime("%Y-%m-%d")
	creator_info = "NetworkX + Labor Market Structure Pipeline (Aaron Bernal Huanca)"

	update_gexf_metadata(
		filepath=snakemake.output.graph,
		creator=creator_info,
		description=description,
		keywords=keywords,
		lastmodifieddate=today_date,
	)

	print("Zenodo graph export complete.")

	# Document the run via logging
	log_lines = []
	log_lines.append("=" * 60)
	log_lines.append("EXPORT ZENODO GRAPH")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"EXPORT SUMMARY",
		[
			f"Class: {class_}",
			f"Input Graph: {snakemake.input.graph}",
			f"Input Nodelist: {snakemake.input.nodelist}",
			f"Output Graph: {snakemake.output.graph}",
			f"Merged Nodes: {len(G.nodes())}",
			f"Merged Edges: {len(G.edges())}",
		],
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
