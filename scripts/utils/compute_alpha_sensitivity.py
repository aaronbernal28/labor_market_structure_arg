import json
from typing import Any

from scripts import *
import networkx as nx
import numpy as np

snakemake: Any


def _reference_alpha_from_lcc(
	alphas: np.ndarray, nodes_largest_cc: np.ndarray
) -> float:
	ref = None
	for i, a in enumerate(alphas):
		if nodes_largest_cc[i] > 0.95:
			ref = float(a)
			break
	if ref is None:
		ref = float(alphas[-1])
	return round(float(ref), 4)


def main() -> None:
	projection = nx.read_gexf(snakemake.input[0], node_type=int)

	seed = int(snakemake.config["seed"])
	alphas = np.logspace(-10, 0, 60)
	alphas.sort()

	(
		nodes_with_edges,
		edge_counts,
		clustering_coeffs,
		nodes_largest_cc,
	) = gc.compute_sweep_alpha(projection, alphas, seed)
	#NOTE: Modularity is not computed here because it is not used for determining the reference alpha.

	# Find the minimum alpha where nodes in largest CC > 95%
	reference_alpha = _reference_alpha_from_lcc(alphas, nodes_largest_cc)

	output_data = {
		"projection_file": str(snakemake.input[0]),
		"class_": str(snakemake.wildcards["class_"]),
		"weight_function": str(snakemake.wildcards["weight_function"]),
		"alphas": alphas.tolist(),
		"nodes_with_edges": nodes_with_edges.tolist(),
		"edge_counts": edge_counts.tolist(),
		"clustering_coeffs": clustering_coeffs.tolist(),
		"nodes_largest_cc": nodes_largest_cc.tolist(),
		"reference_alpha": reference_alpha,
	}

	output_path = snakemake.output[0]
	with open(output_path, "w") as f:
		json.dump(output_data, f, indent=4)

	print(f"Saved alpha sensitivity calculation to {output_path}.")

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("ALPHA SENSITIVITY")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SWEEP SETTINGS",
		[
			f"Reference alpha: {reference_alpha}",
			f"Alpha min: {alphas.min():.6f}",
			f"Alpha max: {alphas.max():.6f}",
			f"Alpha count: {len(alphas)}",
		],
	)


if __name__ == "__main__":
	main()
