import pandas as pd
import networkx as nx
from typing import Any
import src.logging_utils as log

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
				G.nodes[node]['label'] = str(attrs[label_col])
			
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
			f"Merged Edges: {len(G.edges())}"
		]
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

if __name__ == "__main__":
	main()
