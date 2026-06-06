from typing import Any
from pathlib import Path
from scripts import *
import networkx as nx
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

snakemake: Any


def preferential_attachment_func(k, A, alpha):
	"""
	Mathematical model to fit the empirical attachment probability.
	Pi(k) = A * k^alpha
	"""
	return A * (k**alpha)


def main() -> None:
	# 1. Inputs: time-ordered list of backbone projections
	# Snakemake should provide them in chronological order
	projection_paths = snakemake.input.projections
	if not isinstance(projection_paths, list):
		projection_paths = [projection_paths]

	# Track degree evolution across all time windows
	cumulative_delta_k = {}  # Sum of new edges connecting to nodes of degree k
	cumulative_nodes_k = {}  # Total number of existing nodes that had degree k

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("PREFERENTIAL ATTACHMENT ANALYSIS")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)

	# 2. Iterate through consecutive time periods T and T+1
	for i in range(len(projection_paths) - 1):
		path_t0 = projection_paths[i]
		path_t1 = projection_paths[i + 1]

		G_T0 = nx.read_gexf(path_t0)
		G_T1 = nx.read_gexf(path_t1)

		# Ensure they are undirected for degree calculation
		G_T0_und = nx.to_undirected(G_T0)
		G_T1_und = nx.to_undirected(G_T1)

		nodes_T0 = set(G_T0_und.nodes())
		degrees_T0 = dict(G_T0_und.degree())

		edges_T0 = set(frozenset(e) for e in G_T0_und.edges())
		edges_T1 = set(frozenset(e) for e in G_T1_und.edges())
		new_edges = edges_T1 - edges_T0

		# Count how many new edges attach to the existing nodes
		new_edge_counts_per_node = {node: 0 for node in nodes_T0}
		for edge in new_edges:
			u, v = list(edge)
			if u in nodes_T0:
				new_edge_counts_per_node[u] += 1
			if v in nodes_T0:
				new_edge_counts_per_node[v] += 1

		# Aggregate data: Group by the initial degree k
		for node in nodes_T0:
			k = degrees_T0[node]
			new_links_acquired = new_edge_counts_per_node[node]

			if k not in cumulative_nodes_k:
				cumulative_nodes_k[k] = 0
				cumulative_delta_k[k] = 0

			cumulative_nodes_k[k] += 1
			cumulative_delta_k[k] += new_links_acquired

	# 3. Calculate empirical probability Pi(k)
	k_vals = []
	Pi_k_vals = []

	for k in sorted(cumulative_nodes_k.keys()):
		if cumulative_nodes_k[k] > 0 and k > 0:  # Ignore k=0 for log-log fit
			empirical_rate = cumulative_delta_k[k] / cumulative_nodes_k[k]
			k_vals.append(k)
			Pi_k_vals.append(empirical_rate)

	if not k_vals:
		print("No data points for fitting.")
		log_lines.append("ERROR: No data points for fitting.")
		log.write_log(log_lines, snakemake.log[0])
		return

	# Normalize
	k_vals = np.array(k_vals)
	Pi_k_vals = np.array(Pi_k_vals)
	total_pi = sum(Pi_k_vals)
	if total_pi > 0:
		Pi_k_vals = Pi_k_vals / total_pi
	else:
		print("Sum of attachment rates is zero.")
		log_lines.append("ERROR: Sum of attachment rates is zero.")
		log.write_log(log_lines, snakemake.log[0])
		return

	# 4. Curve fitting
	try:
		popt, pcov = curve_fit(
			preferential_attachment_func, k_vals, Pi_k_vals, p0=[1, 1], maxfev=10000
		)
		A_fit, alpha_fit = popt
		log_lines.append(f"Estimated alpha: {alpha_fit:.4f}")
		log_lines.append(f"Estimated A: {A_fit:.4e}")
	except Exception as e:
		log_lines.append(f"Fitting failed: {str(e)}")
		log.write_log(log_lines, snakemake.log[0])
		return

	# 5. Serialization & Logging
	log.write_log(log_lines, snakemake.log[0])

	# 6. Visualization
	plt.figure(figsize=(10, 6))
	plt.loglog(k_vals, Pi_k_vals, "o", alpha=0.6, label="Empirical EPH Data")

	# Fit line
	fit_y = preferential_attachment_func(k_vals, A_fit, alpha_fit)
	plt.loglog(
		k_vals, fit_y, "r--", label=f"Fit: $\\Pi(k) \propto k^{{{alpha_fit:.2f}}}$"
	)

	plt.xlabel("Degree $k$ at $T_0$")
	plt.ylabel("Attachment Probability $\Pi(k)$")
	plt.title("")
	plt.legend()
	plt.grid(True, which="both", ls="-", alpha=0.2)

	plt.savefig(snakemake.output.plot, bbox_inches="tight")
	plt.close()


if __name__ == "__main__":
	main()
