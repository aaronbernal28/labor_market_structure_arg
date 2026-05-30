from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np
import itertools
from scipy.spatial import ConvexHull
from scripts import *

snakemake: Any


def main() -> None:
	print("[DEBUG] Starting script 18_disparity_filtration_subgraph.py")

	# Use standard publication style if available
	try:
		plt.style.use("src/styles/publication.mplstyle")
		print("[DEBUG] Applied custom matplotlib style.")
	except OSError:
		print("[DEBUG] Custom matplotlib style not found, using default.")

	nodelist_path = snakemake.input["nodelist"]
	graph_path = snakemake.input["graph"]
	output_path = snakemake.output[0]

	# 1. Load the projection graph and corresponding nodelist
	print(f"[DEBUG] Loading nodelist from: {nodelist_path}")
	df = pd.read_csv(nodelist_path)
	print(f"[DEBUG] Nodelist shape: {df.shape}")

	print(f"[DEBUG] Loading graph from: {graph_path}")
	G = nx.read_gexf(graph_path, node_type=int)
	print(
		f"[DEBUG] Original graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
	)

	class_ = snakemake.wildcards["class_"]
	id_col = snakemake.config[class_]["id"]

	# 2. Filter the graph using one community as the focal class
	if "community" in df.columns:
		comm_counts = df["community"].value_counts()
		if 12 in comm_counts:
			target_comm = 12
		elif "C12" in comm_counts:
			target_comm = "C12"
		else:
			target_comm = comm_counts.idxmax()

		print(f"[DEBUG] Focal community determined: {target_comm}")
		focal_nodes = df[df["community"] == target_comm][id_col].tolist()
	else:
		target_comm = "All"
		focal_nodes = list(G.nodes())
		print("[DEBUG] No community column found, using all nodes.")

	# Extract induced subgraph for the focal class
	focal_nodes = [n for n in focal_nodes if n in G.nodes()]
	subgraph = G.subgraph(focal_nodes).copy()
	print(
		f"[DEBUG] Focal subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges."
	)

	# 3. Apply the disparity filtration to the subgraph
	print("[DEBUG] Calculating disparity graph constraints on the focal subgraph...")
	disp_G = gc.get_disparity_graph(subgraph)
	alphas = np.logspace(-3, 0, num=6).round(3)

	# Set up plotting
	fig, axes = plt.subplots(1, len(alphas), figsize=(3 * len(alphas), 5))
	fig.suptitle(
		f"Disparity Filtration (Focal Community: {target_comm})", y=1.05, fontsize=16
	)

	# Fixed layout across all subgraphs
	pos = {}
	for _, row in df.iterrows():
		if "pos_x" in row and "pos_y" in row:
			pos[int(row[id_col])] = (row["pos_x"], row["pos_y"])

	# Fallback if coordinates are missing in nodelist
	if not pos:
		print("[DEBUG] Nodelist lacks 'pos_x'/'pos_y', computing spring layout...")
		pos = nx.spring_layout(subgraph, seed=snakemake.config.get("seed", 42))

	# Setup fixed styles (color, constant sizes, specific labels)
	color_community = utils.get_community_color(
		target_comm, communities=df["community"].unique()
	)
	color_map = {n: color_community for n in subgraph.nodes()}

	label_col = "as_display" if "as_display" in df.columns else id_col
	labels_map = {int(row[id_col]): str(row[label_col]) for _, row in df.iterrows()}

	prev_nodes = set()

	# 4. Plot resulting subgraphs side-by-side
	for i, a in enumerate(alphas):
		print(f"[DEBUG] Processing alpha = {a}")
		ax = axes[i]

		# Alpha = 1.0 implies no filtering (None)
		if a == 1.0:
			filtered_G = subgraph.copy()
			title = r"$\alpha = None$ (Full Subgraph)"
		else:
			filtered_G = gc.disparity_filter_backbone(
				original_graph=subgraph,
				disparity_graph=disp_G,
				alpha=a,
				mode="or",
				keep_isolates=False,
			)
			title = rf"$\alpha = {a}$"

		current_nodelist = list(filtered_G.nodes())
		current_nodes_set = set(current_nodelist)

		# Determine newly introduced nodes vs the previous tighter alpha limit
		new_nodes = current_nodes_set - prev_nodes

		# TOPOLOGICAL SIMPLICES (Triangles & Tetrahedra)
		all_3_cliques = set()
		all_4_cliques = set()

		# Find all maximal cliques to extract 3-cliques and 4-cliques
		for c in nx.find_cliques(filtered_G):
			if len(c) >= 3:
				for sub_c in itertools.combinations(c, 3):
					all_3_cliques.add(tuple(sorted(sub_c)))
			if len(c) >= 4:
				for sub_c in itertools.combinations(c, 4):
					all_4_cliques.add(tuple(sorted(sub_c)))

		# Filter out 3-cliques that are faces of 4-cliques to avoid visual double-filling
		faces_of_4_cliques = set()
		for c4 in all_4_cliques:
			for face in itertools.combinations(c4, 3):
				faces_of_4_cliques.add(tuple(sorted(face)))

		independent_3_cliques = all_3_cliques - faces_of_4_cliques

		print(
			f"[DEBUG] Alpha {a}: {len(independent_3_cliques)} Triangles, {len(all_4_cliques)} Tetrahedra"
		)

		# Draw 4-cliques (3-Simplices / Tetrahedra) - Background Z-order 1
		for c4 in all_4_cliques:
			if all(n in pos for n in c4):
				pts = np.array([pos[n] for n in c4])
				try:
					hull = ConvexHull(pts)
					hull_pts = pts[hull.vertices]
					poly = plt.Polygon(
						hull_pts,
						closed=True,
						facecolor="crimson",
						edgecolor="crimson",
						lw=0.5,
						alpha=0.15,
						zorder=1,
					)
					ax.add_patch(poly)
				except Exception:
					pass  # Skip collinear points

		# Draw remaining 3-cliques (2-Simplices / Triangles) - Background Z-order 2
		for c3 in independent_3_cliques:
			if all(n in pos for n in c3):
				pts = np.array([pos[n] for n in c3])
				try:
					hull = ConvexHull(pts)
					hull_pts = pts[hull.vertices]
					poly = plt.Polygon(
						hull_pts,
						closed=True,
						facecolor="deepskyblue",
						edgecolor="deepskyblue",
						lw=0.5,
						alpha=0.25,
						zorder=2,
					)
					ax.add_patch(poly)
				except Exception:
					pass

		# STANDARD NETWORK ELEMENTS (Nodes & Edges)

		# Edges - Z-order 3
		edges = filtered_G.edges(data=True)
		if edges:
			weights = [d.get("weight", 1.0) for u, v, d in edges]
			max_w = max(weights) if weights else 1.0
			edge_widths = [0.5 + 2.5 * (w / max_w) for w in weights]

			edge_collection = nx.draw_networkx_edges(
				filtered_G, pos, ax=ax, width=edge_widths, alpha=0.6, edge_color="gray"
			)
			if edge_collection is not None:
				edge_collection.set_zorder(3)

		# Nodes - Z-order 4
		node_colors = [color_map.get(n, "steelblue") for n in current_nodelist]
		if current_nodelist:
			node_collection = nx.draw_networkx_nodes(
				filtered_G,
				pos,
				ax=ax,
				nodelist=current_nodelist,
				node_size=120,
				node_color=node_colors,
				edgecolors="black",
				linewidths=0.5,
			)
			if node_collection is not None:
				node_collection.set_zorder(4)

		# Labeled highlighting ONLY the new nodes (or all nodes if alpha is small enough)
		current_labels = {n: utils.original_id(int(labels_map.get(n, str(n))), class_index=snakemake.config[class_].get('partition', 1), max_caes_id=snakemake.config.get('max_caes_id')) for n in new_nodes}
		if current_labels:
			text_dict = nx.draw_networkx_labels(
				filtered_G,
				pos,
				labels=current_labels,
				ax=ax,
				font_size=7,
				font_color="black",
				bbox=dict(
					facecolor="white",
					edgecolor="black",
					boxstyle="round,pad=0.2",
					alpha=0.6,
				),
			)
			for text in text_dict.values():
				text.set_zorder(5)

		ax.set_title(title, fontsize=14)
		ax.axis("off")
		prev_nodes = current_nodes_set

	# Add a custom legend for the simplices at the bottom center of the figure
	triangle_patch = mpatches.Patch(
		color="deepskyblue", alpha=0.35, label="2-Simplices (Triangles)"
	)
	tetra_patch = mpatches.Patch(
		color="crimson", alpha=0.25, label="3-Simplices (Tetrahedra / 4-Cliques)"
	)
	fig.legend(
		handles=[triangle_patch, tetra_patch],
		loc="lower center",
		ncol=2,
		fontsize=12,
		bbox_to_anchor=(0.5, -0.05),
		frameon=False,
	)

	plt.tight_layout()

	# Ensure the output directory exists and save
	print(f"[DEBUG] Saving plot to {output_path}")
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, bbox_inches="tight")
	plt.close()
	print("[DEBUG] Script finished successfully.")


if __name__ == "__main__":
	main()
