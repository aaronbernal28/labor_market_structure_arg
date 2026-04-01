"""
Plotting utilities extracted from the exploratory notebook.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping
import textwrap

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import src.utils as ut

LIGTHGRAY = "#a8a8a8"

# plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 100})


def fceyn_mean_edge_color(color_u, color_v):
	"""Return the mean RGB color between two node colors."""
	rgb_u = np.asarray(mcolors.to_rgb(color_u), dtype=float)
	rgb_v = np.asarray(mcolors.to_rgb(color_v), dtype=float)
	return tuple(((rgb_u + rgb_v) / 2.0).tolist())


def fceyn_edge_alpha_from_weight(
	weight: float,
	max_weight: float,
	*,
	alpha_max: float = 0.4,
	alpha_min: float = 0.05,
) -> float:
	"""Map an edge weight to an alpha in [alpha_min, min(alpha_max, 0.6)]."""
	alpha_cap = min(float(alpha_max), 0.6)
	alpha_floor = max(0.0, float(alpha_min))
	if alpha_floor > alpha_cap:
		alpha_floor = alpha_cap

	try:
		w = float(weight)
	except (TypeError, ValueError):
		w = 0.0
	if not np.isfinite(w) or w <= 0:
		return alpha_floor

	try:
		mw = float(max_weight)
	except (TypeError, ValueError):
		mw = 0.0
	if not np.isfinite(mw) or mw <= 0:
		return alpha_floor

	r = max(0.0, min(1.0, w / mw))
	return alpha_floor + (alpha_cap - alpha_floor) * r


def fceyn_edge_rgba_from_node_colors(color_u, color_v, alpha: float):
	rgb = fceyn_mean_edge_color(color_u, color_v)
	return mcolors.to_rgba(rgb, alpha=alpha)


def fceyn_draw_bipartite_by_color(
	graph: nx.Graph,
	color_map: Dict[int, str],
	label_map: Dict[int, str] = None,
	output_path: Path = None,
	seed: int = 28,
	top_n: int = 6,
	shift_x: float = 0.5,
	title: str = "",
	save: bool = True,
	figsize: tuple = (12, 8),
	edge_alpha: float = 0.9,
	font_size: int = 9,
	legend_marker_size: float = 11.0,
	factor_node_size: float = 3.0,
	node_size_map: Mapping[int, float] = None,
	node_size_exponent: float = 1.0,
	show: bool | None = None,
) -> Dict[str, tuple]:
	"""Draw the bipartite network with custom layout and return the positions.

	Parameters:
	- factor_node_size: Multiplier for node sizes.
	- node_size_map: Optional mapping from node -> scalar (e.g. worker counts) used for sizing.
	  Falls back to degree when not provided.
	- node_size_exponent: Power transform applied before the factor.
	- node_size_exponent: Power transform applied before the factor.
	- legend_marker_size: Marker size (points) used in the legend examples (triangle/circle).
	"""
	np.random.seed(seed)

	assert set(color_map.keys()) >= set(graph.nodes()), (
		"Graph contains nodes not present in color map."
	)
	assert label_map is None or set(label_map.keys()) >= set(graph.nodes()), (
		"Graph contains nodes not present in label map."
	)

	# Compute initial layout
	pos = nx.spring_layout(graph, seed=seed, k=0.5, iterations=1000, method="force")
	pos_caes_y = [
		pos[node][1]
		for node in graph.nodes()
		if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("caes")
	]
	pos_ciuo_y = [
		pos[node][1]
		for node in graph.nodes()
		if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("ciuo")
	]

	# Sigmoidal normalization functions
	def fceyn_sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def fceyn_normalize_caes_y(y):
		return fceyn_sigmoid(2.5 * (y - np.mean(pos_caes_y)) / np.std(pos_caes_y))

	def fceyn_normalize_ciuo_y(y):
		return fceyn_sigmoid(2.5 * (y - np.mean(pos_ciuo_y)) / np.std(pos_ciuo_y))

	# Adjust positions (shift axis x and normalize y)
	for node in graph.nodes():
		if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("caes"):
			pos[node][0] -= shift_x
			pos[node][1] = fceyn_normalize_caes_y(pos[node][1])
		else:
			pos[node][0] += shift_x
			pos[node][1] = fceyn_normalize_ciuo_y(pos[node][1])

	# Defining color and size maps
	if node_size_map is not None:
		raw_sizes = {
			node: float(node_size_map.get(node, node_size_map.get(int(node), 1.0)))
			for node in graph.nodes()
		}
	else:
		raw_sizes = {node: float(degree) for node, degree in graph.degree()}

	size_map = {
		node: float(
			np.power(max(raw_sizes.get(node, 1.0), 0.0), node_size_exponent)
			* factor_node_size
		)
		for node in graph.nodes()
	}
	caes_nodes = [
		node
		for node in graph.nodes()
		if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("caes")
	]
	ciuo_nodes = [
		node
		for node in graph.nodes()
		if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("ciuo")
	]

	# Plotting
	plt.figure(figsize=figsize)
	# nx.draw_networkx_edges(graph, pos, edge_color="white", width=0.1, alpha=0.7)
	nx.draw_networkx_nodes(
		graph,
		pos,
		nodelist=caes_nodes,
		node_color=[color_map.get(int(node), LIGTHGRAY) for node in caes_nodes],
		node_size=[size_map[node] for node in caes_nodes],
		node_shape="^",
		alpha=0.7,
		# edgecolors="white",
	)
	nx.draw_networkx_nodes(
		graph,
		pos,
		nodelist=ciuo_nodes,
		node_color=[color_map.get(int(node), LIGTHGRAY) for node in ciuo_nodes],
		node_size=[size_map[node] for node in ciuo_nodes],
		node_shape="o",
		alpha=0.7,
		# edgecolors="white",
	)

	# Draw spline edges
	path_cls = mpath.Path
	for u, v in graph.edges():
		# Get initial and final positions
		start = pos[u]
		end = pos[v]

		# Define Bezier curve control points
		control1 = (np.mean([start[0], end[0]]), start[1])
		control2 = (np.mean([start[0], end[0]]), start[1])
		verts = [start, control1, control2, end]
		codes = [path_cls.MOVETO, path_cls.CURVE4, path_cls.CURVE4, path_cls.CURVE4]
		path = path_cls(verts, codes)

		# Assign color based on starting node
		color = color_map.get(int(u), LIGTHGRAY)

		# Draw the edge
		patch = patches.PathPatch(
			path, facecolor="none", edgecolor=color, lw=0.1, alpha=edge_alpha
		)
		plt.gca().add_patch(patch)

	if top_n is not None and len(label_map.values()) > top_n:
		# Label top N nodes by degree and class
		degrees = dict(graph.degree())
		sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
		top_nodes_caes = [
			node
			for node, _ in sorted_degrees
			if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("caes")
		][:top_n]
		top_nodes_ciuo = [
			node
			for node, _ in sorted_degrees
			if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("ciuo")
		][:top_n]
		label_map = {
			node: label_map[node] if label_map and node in label_map else node
			for node in top_nodes_caes + top_nodes_ciuo
		}

		# Draw labels
		nx.draw_networkx_labels(
			graph,
			pos,
			labels=label_map,
			font_size=max(font_size - 1, 6),
			font_color="black",
			font_weight="bold",
			horizontalalignment="left",
			verticalalignment="center",
			bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.3),
		)

	# Legends
	caes_groups: Dict[str, str] = {}
	ciuo_groups: Dict[str, str] = {}
	if label_map is not None:
		for node in graph.nodes():
			node_id = int(node)
			lbl = label_map.get(node_id, str(node_id))
			color = color_map.get(node_id, LIGTHGRAY)
			if graph.nodes[node].get("bipartite") == ut.fceyn_get_class_index("caes"):
				caes_groups[lbl] = color
			else:
				ciuo_groups[lbl] = color

	def fceyn_make_handles(groups, marker_shape="o"):
		ms = float(legend_marker_size) if legend_marker_size is not None else 11.0
		return [
			plt.Line2D(
				[0],
				[0],
				marker=marker_shape,
				color="w",
				markerfacecolor=c,
				markersize=ms,
				label=textwrap.fill(lbl, 28),
				linestyle="",
			)
			for lbl, c in sorted(groups.items())
		]

	if caes_groups:
		leg_l = plt.legend(
			handles=fceyn_make_handles(caes_groups, marker_shape="^"),
			title="Ramas de actividad (CAES)",
			loc="upper left",
			bbox_to_anchor=(0.0, 1.0),
			fontsize=max(font_size - 1, 6),
			title_fontsize=font_size,
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)
		plt.gca().add_artist(leg_l)
	if ciuo_groups:
		plt.legend(
			handles=fceyn_make_handles(ciuo_groups, marker_shape="o"),
			title="Ocupaciones (CIUO)",
			loc="upper right",
			bbox_to_anchor=(1.0, 1.0),
			fontsize=max(font_size - 1, 6),
			title_fontsize=font_size,
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)

	# Finalize plot
	plt.title(title, fontsize=font_size + 1)
	plt.axis("off")
	plt.xlim(-1.4, 1.6)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()
	return pos


def fceyn_draw_bipartite_normal_layout_by_color(
	graph: nx.Graph,
	color_map: Dict[int, str],
	label_map: Dict[int, str] = None,
	output_path: Path = None,
	seed: int = 28,
	top_n: int = 6,
	title: str = "",
	save: bool = True,
	figsize: tuple = (16, 12),
	node_scale: float = 8.0,
	edge_alpha: float = 0.35,
	edge_lw: float = 0.15,
	show: bool | None = None,
) -> Dict[str, tuple]:
	"""Draw the bipartite network with nodes sorted by group to minimise edge crossings."""
	import matplotlib.patches as mpatches

	assert set(color_map.keys()) >= set(graph.nodes()), (
		"Graph contains nodes not present in color map."
	)
	assert label_map is None or set(label_map.keys()) >= set(graph.nodes()), (
		"Graph contains nodes not present in label map."
	)

	caes_idx = ut.fceyn_get_class_index("caes")
	ciuo_idx = ut.fceyn_get_class_index("ciuo")

	caes_nodes = [
		n for n in graph.nodes() if graph.nodes[n].get("bipartite") == caes_idx
	]
	ciuo_nodes = [
		n for n in graph.nodes() if graph.nodes[n].get("bipartite") == ciuo_idx
	]

	# Sort each partition by group label so same-colored nodes are contiguous,
	# which dramatically reduces edge crossings.
	def fceyn_group_key(n):
		return label_map.get(int(n), str(n)) if label_map else str(n)

	caes_nodes = sorted(caes_nodes, key=fceyn_group_key)
	ciuo_nodes = sorted(ciuo_nodes, key=fceyn_group_key)

	def fceyn_linear_positions(nodes, x):
		n = len(nodes)
		ys = np.linspace(1.0, -1.0, n) if n > 1 else [0.0]
		return {node: np.array([x, y]) for node, y in zip(nodes, ys)}

	pos = {}
	pos.update(fceyn_linear_positions(caes_nodes, -1.0))
	pos.update(fceyn_linear_positions(ciuo_nodes, 1.0))

	fig, ax = plt.subplots(figsize=figsize)
	ax.set_aspect("auto")
	ax.axis("off")

	# ── Bezier edges (sorted by CIUO y-position for visual grouping) ─────────
	path_cls = mpath.Path
	edges_sorted = sorted(
		graph.edges(),
		key=lambda e: pos[e[1]][1],  # sort by target y → reduces colour mixing
	)
	weights = [graph[u][v].get("weight", 1.0) for u, v in edges_sorted]
	max_weight = max(weights) if weights else 0.0
	for u, v in edges_sorted:
		start, end = pos[u], pos[v]
		mx = 0.0  # midpoint x → straight-line control gives S-curve
		ctrl1 = (mx, start[1])
		ctrl2 = (mx, end[1])
		path = path_cls(
			[start, ctrl1, ctrl2, end],
			[path_cls.MOVETO, path_cls.CURVE4, path_cls.CURVE4, path_cls.CURVE4],
		)
		color_u = color_map.get(int(u), LIGTHGRAY)
		color_v = color_map.get(int(v), LIGTHGRAY)
		w = graph[u][v].get("weight", 1.0)
		alpha = fceyn_edge_alpha_from_weight(w, max_weight, alpha_max=edge_alpha)
		edge_color = fceyn_edge_rgba_from_node_colors(color_u, color_v, alpha)
		ax.add_patch(
			patches.PathPatch(
				path,
				facecolor="none",
				edgecolor=edge_color,
				lw=edge_lw,
				alpha=1.0,
			)
		)

	# Nodes
	degrees = dict(graph.degree())
	for node in graph.nodes():
		x, y = pos[node]
		color = color_map.get(int(node), LIGTHGRAY)
		size = np.sqrt(degrees[node] + 1) * node_scale
		ax.scatter(x, y, s=size**2 * 0.15, c=[color], zorder=3, linewidths=0)

	# Legends
	caes_groups: Dict[str, tuple] = {}
	ciuo_groups: Dict[str, tuple] = {}
	if label_map is not None:
		for node in graph.nodes():
			node_id = int(node)
			lbl = label_map.get(node_id, str(node_id))
			c = color_map.get(node_id, LIGTHGRAY)
			if graph.nodes[node].get("bipartite") == caes_idx:
				caes_groups[lbl] = c
			else:
				ciuo_groups[lbl] = c

	def fceyn_make_handles(groups):
		return [
			mpatches.Patch(facecolor=c, label=textwrap.fill(lbl, 28), linewidth=0)
			for lbl, c in sorted(groups.items())
		]

	if caes_groups:
		leg_l = ax.legend(
			handles=fceyn_make_handles(caes_groups),
			title="Ramas economicas\n(CAES)",
			loc="upper left",
			bbox_to_anchor=(0.0, 1.0),
			fontsize=8.5,
			title_fontsize=8.5,
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)
		ax.add_artist(leg_l)
	if ciuo_groups:
		ax.legend(
			handles=fceyn_make_handles(ciuo_groups),
			title="Ocupaciones\n(CIUO)",
			loc="upper right",
			bbox_to_anchor=(1.0, 1.0),
			fontsize=8.5,
			title_fontsize=8.5,
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)

	ax.set_xlim(-1.25, 1.25)
	ax.set_title(title, pad=12, fontsize=12)
	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()
	return pos


def fceyn_plot_projection_by_group(
	graph: nx.Graph,
	group_map: Mapping[str, str],
	group_color_map: Mapping[str, str],
	title: str,
	legend_title: str,
	output_path: Path = None,
	figsize: tuple = (10, 10),
	font_size: int = 11,
	seed: int = 42,
	save: bool = True,
	legend_label_fmt=None,
	spring_layout_iterations: int = 1000,
	spring_layout_k: float = None,
	factor_node_size: int = 0.5,
	node_size_map: Mapping[int, float] = None,
	node_size_exponent: float = 1.0,
	pos: dict = None,
	rotate: bool = False,
	method: str = "auto",
	edge_alpha: float = 0.1,
	node_alpha: float = 0.5,
	show: bool | None = None,
) -> dict:
	"""
	Plot the graph with nodes colored by their group.
	Parameters:
	- graph: The NetworkX graph to plot.
	- group_map: A mapping from node to its group.
	- group_color_map: A mapping from group to its color.
	- title: Title of the plot.
	- legend_title: Title for the legend.
	- output_path: Path = None to save the output image.
	- legend_label_fmt: formatter for legend labels.
	- spring_layout_iterations: Number of iterations for spring layout.
	- factor_node_size: Multiplier for node sizes based on degree.
	- node_size_map: Optional mapping from node to a scalar value (e.g. n_obs) used for sizing.
	  Falls back to degree when not provided.
	- pos: Optional precomputed positions for nodes (if None, will compute using spring layout).
	- rotate: Whether to rotate the layout 90 degrees anticlockwise.
	"""
	assert set(group_color_map.keys()) <= set(group_map.values()), (
		"Color map contains groups not present in group map."
	)
	assert set(graph.nodes()) <= set(group_map.keys()), (
		"Graph contains nodes not present in group map."
	)

	# Get the largest connected component for layout
	if not nx.is_connected(graph):
		largest_cc = max(nx.connected_components(graph), key=len)
		graph = graph.subgraph(largest_cc)
		if pos is not None:
			pos = {node: pos[node] for node in graph.nodes() if node in pos}

	if pos is None:
		if method == "kamada_kawai":
			pos = nx.kamada_kawai_layout(graph)
		else:
			pos = nx.spring_layout(
				graph,
				seed=seed,
				k=spring_layout_k,
				iterations=spring_layout_iterations,
				threshold=1e-3,
				method=method,
			)

	if rotate:
		pos = {node: (-y, x) for node, (x, y) in pos.items()}

	# Prepare node colors and sizes
	node_colors = [
		group_color_map.get(group_map.get(node), LIGTHGRAY) for node in graph.nodes()
	]
	node_color_by_node = {
		node: group_color_map.get(group_map.get(node), LIGTHGRAY)
		for node in graph.nodes()
	}
	if node_size_map is not None:
		raw_sizes = [node_size_map.get(node, 1) for node in graph.nodes()]
	else:
		raw_sizes = [graph.degree(node) + 1 for node in graph.nodes()]

	node_sizes = [np.power(s, node_size_exponent) * factor_node_size for s in raw_sizes]

	# Prepare edge widths and alphas
	edges = list(graph.edges())
	if len(edges) > 0:
		edge_data = next(iter(graph.edges(data=True)))[-1]
		if "weight" in edge_data:
			weights = [graph[u][v].get("weight", 1.0) for u, v in edges]
			max_weight = max(weights) if max(weights) > 0 else 1.0
			edge_widths = [0.1 + 1.9 * (w / max_weight) for w in weights]
			edge_alphas = [
				fceyn_edge_alpha_from_weight(w, max_weight, alpha_max=edge_alpha)
				for w in weights
			]
		else:
			edge_widths = 0.3
	else:
		edge_widths = 0.3

	# Plotting
	plt.figure(figsize=figsize)
	nx.draw_networkx_nodes(
		graph, pos, node_color=node_colors, node_size=node_sizes, alpha=node_alpha
	)
	if len(edges) > 0:
		if isinstance(edge_alphas, list):
			edge_colors = [
				fceyn_edge_rgba_from_node_colors(
					node_color_by_node[u], node_color_by_node[v], a
				)
				for (u, v), a in zip(edges, edge_alphas)
			]
		else:
			edge_colors = [
				fceyn_edge_rgba_from_node_colors(
					node_color_by_node[u], node_color_by_node[v], edge_alphas
				)
				for u, v in edges
			]
		nx.draw_networkx_edges(
			graph,
			pos,
			edgelist=edges,
			edge_color=edge_colors,
			width=edge_widths,
			alpha=edge_alpha,
		)

	# Create legend
	label_fn = legend_label_fmt or (lambda g: g)
	for group, color in group_color_map.items():
		plt.scatter([], [], color=color, label=label_fn(group))

	plt.legend(
		title=legend_title,
		fontsize=max(font_size - 2, 6),
		title_fontsize=font_size,
		loc="best",
		borderaxespad=4.0,
		framealpha=0.7,
	)
	plt.title(title, fontsize=font_size + 1)
	plt.axis("off")
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()
	return pos


def fceyn_plot_projection_gradient(
	graph: nx.Graph,
	pos: dict,
	node_values: dict,
	title: str,
	colorbar_label: str = "Valor",
	cmap: str = "viridis",
	figsize: tuple = (10, 10),
	font_size: int = 11,
	output_path: Path = None,
	save: bool = True,
	factor_node_size: float = 0.5,
	node_size_map: Mapping[int, float] = None,
	vmin: float = None,
	vmax: float = None,
	node_size_exponent: float = 1.0,
	edge_alpha: float = 0.1,
	node_alpha: float = 0.5,
	show: bool | None = None,
):
	"""Plot the projection network with nodes colored by a continuous scalar gradient.

	Parameters:
	- graph: NetworkX graph to plot.
	- pos: Precomputed node positions (from fceyn_plot_projection_by_group).
	- node_values: Mapping from node id to float scalar (nodes missing from the map get gray).
	- title: Plot title.
	- colorbar_label: Label shown on the colorbar.
	- cmap: Matplotlib colormap name.
	- output_path: Path to save the image.
	- save: Save to file when True, show interactively otherwise.
	- factor_node_size: Multiplier for node sizes based on degree.
	- node_size_map: Optional mapping from node to a scalar value (e.g. n_obs) used for sizing.
	  Falls back to degree when not provided.
	- vmin / vmax: Explicit colormap range; defaults to the min/max of known values.
	"""
	# Restrict to nodes that have a position (largest-cc layout)
	nodes = [n for n in graph.nodes() if n in pos]
	values = np.array([node_values.get(n, np.nan) for n in nodes], dtype=float)

	finite = values[np.isfinite(values)]
	if vmin is None:
		vmin = float(finite.min()) if len(finite) else 0.0
	if vmax is None:
		vmax = float(finite.max()) if len(finite) else 1.0

	colormap = plt.get_cmap(cmap)
	norm = plt.Normalize(vmin=vmin, vmax=vmax)

	node_colors = [colormap(norm(v)) if np.isfinite(v) else "lightgray" for v in values]
	node_color_by_node = {n: c for n, c in zip(nodes, node_colors)}
	if node_size_map is not None:
		raw_sizes = [node_size_map.get(n, 1) for n in nodes]
	else:
		raw_sizes = [graph.degree(n) + 1 for n in nodes]

	node_sizes = [np.power(s, node_size_exponent) * factor_node_size for s in raw_sizes]

	subgraph = graph.subgraph(nodes)
	subpos = {n: pos[n] for n in nodes}

	# Prepare edge widths and alphas for subgraph
	edges = list(subgraph.edges())
	if len(edges) > 0:
		edge_data = next(iter(subgraph.edges(data=True)))[-1]
		if "weight" in edge_data:
			weights = [subgraph[u][v].get("weight", 1.0) for u, v in edges]
			max_weight = max(weights) if max(weights) > 0 else 1.0
			edge_widths = [0.1 + 1.9 * (w / max_weight) for w in weights]
			edge_alphas = [
				fceyn_edge_alpha_from_weight(w, max_weight, alpha_max=edge_alpha)
				for w in weights
			]
		else:
			edge_widths = 0.3
	else:
		edge_widths = 0.3

	fig, ax = plt.subplots(figsize=figsize)
	nx.draw_networkx_nodes(
		subgraph,
		subpos,
		ax=ax,
		node_color=node_colors,
		node_size=node_sizes,
		alpha=node_alpha,
	)
	if len(edges) > 0:
		if isinstance(edge_alphas, list):
			edge_colors = [
				fceyn_edge_rgba_from_node_colors(
					node_color_by_node.get(u, "lightgray"),
					node_color_by_node.get(v, "lightgray"),
					a,
				)
				for (u, v), a in zip(edges, edge_alphas)
			]
		else:
			edge_colors = [
				fceyn_edge_rgba_from_node_colors(
					node_color_by_node.get(u, "lightgray"),
					node_color_by_node.get(v, "lightgray"),
					edge_alphas,
				)
				for u, v in edges
			]
		nx.draw_networkx_edges(
			subgraph,
			subpos,
			ax=ax,
			edgelist=edges,
			edge_color=edge_colors,
			width=edge_widths,
			alpha=edge_alpha,
		)

	sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=-0.15)
	cbar.set_label(colorbar_label, fontsize=font_size)
	cbar.ax.tick_params(labelsize=max(font_size - 1, 6))

	ax.set_title(title, fontsize=font_size + 1)
	ax.axis("off")
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()


def fceyn_color_map_caes(caes_nodes: Iterable[str]) -> Dict[int, str]:
	"""Create a color map for CAES nodes based on their group."""
	cmap = mpl.colormaps["Accent"]
	x = np.linspace(
		0, 1, max(caes_nodes) + 1
	)  # hypothesis: len(caes_nodes) << max(caes_nodes)
	caes_nodes = sorted(caes_nodes, reverse=False)
	return {node: cmap(x[node]) for node in caes_nodes}


def fceyn_color_map_ciuo(ciuo_nodes: Iterable[str], max_caes_id: int) -> Dict[int, str]:
	"""Create a color map for CIUO nodes based on their group."""
	cmap = mpl.colormaps["inferno"]
	# Mapping from input node (desambiated) to original ID
	node_to_original = {
		node: ut.fceyn_original_ciuo_id(int(node), max_caes_id=max_caes_id)
		for node in ciuo_nodes
	}
	original_ids = list(node_to_original.values())

	if not original_ids:
		return {}

	x = np.linspace(0, 1, max(original_ids) + 1)
	# Return map from input node to color
	return {node: cmap(x[orig_id]) for node, orig_id in node_to_original.items()}


def fceyn_mean_color(colors):
	colors_array = np.array([list(c) for c in colors])
	return tuple(colors_array.mean(axis=0))


def fceyn_color_letra_map_caes(
	caes_df: pd.DataFrame, letra_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CAES letra based on their group."""
	return caes_df.groupby(letra_col)[base_color_col].apply(fceyn_mean_color).to_dict()


def fceyn_color_1digit_map_ciuo(
	ciuo_df: pd.DataFrame, letra_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CIUO letra based on their group."""
	return ciuo_df.groupby(letra_col)[base_color_col].apply(fceyn_mean_color).to_dict()


def fceyn_color_agrupation_map_caes(
	caes_df: pd.DataFrame, ag_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CAES agrupation based on their group."""
	return caes_df.groupby(ag_col)[base_color_col].apply(fceyn_mean_color).to_dict()


def fceyn_color_ciuo3cat_map_ciuo(
	ciuo_df: pd.DataFrame, cat_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CIUO 3-category based on their group."""
	return ciuo_df.groupby(cat_col)[base_color_col].apply(fceyn_mean_color).to_dict()


def fceyn_plot_alpha_sensitivity(
	alphas: np.ndarray,
	nodes_with_edges: np.ndarray,
	edge_counts: np.ndarray,
	clustering_coefficients: np.ndarray,
	title: str,
	output_path: Path,
	modularities: np.ndarray = None,
	nodes_largest_cc: np.ndarray = None,
	reference_alpha: float = 0.05,
	save: bool = True,
	logscale: bool = True,
	show: bool | None = None,
) -> None:
	"""Plot backbone sensitivity to alpha: relative nodes-with-edges and edge fraction on
	the left y-axis; clustering coefficient and (optionally) modularity on the right y-axis.

	Parameters
	----------
	alphas: 1-D array of alpha values in (0, 1).
	nodes_with_edges: fraction of nodes that have at least one edge at each alpha.
	edge_counts: fraction of edges retained at each alpha.
	clustering_coefficients: average clustering coefficient at each alpha.
	title: plot title (typically the network name).
	output_path: destination file path.
	modularities: Louvain modularity at each alpha (optional).
	nodes_largest_cc: fraction of nodes in the largest connected component (optional).
	reference_alpha: vertical reference line (default 0.05).
	save: if True save to file, else show interactively.
	"""
	fig, ax = plt.subplots(figsize=(7, 6))

	color_nodes = "steelblue"
	color_edges = "coral"
	color_clust = "seagreen"
	color_mod = "darkorchid"
	color_lcc = "firebrick"

	(l1,) = ax.plot(
		alphas, nodes_with_edges, color=color_nodes, linewidth=2, label="Nodos"
	)
	(l2,) = ax.plot(
		alphas,
		edge_counts,
		color=color_edges,
		linewidth=2,
		linestyle="--",
		label="Aristas",
	)
	(l3,) = ax.plot(
		alphas,
		clustering_coefficients,
		color=color_clust,
		linewidth=2,
		linestyle=":",
		label="Coef. de clustering prom.",
	)

	lines = [l1, l2, l3]
	if modularities is not None:
		(l4,) = ax.plot(
			alphas,
			modularities,
			color=color_mod,
			linewidth=2,
			label="Modularidad (Louvain - relativa)",
		)
		lines.append(l4)

	if nodes_largest_cc is not None:
		(l5,) = ax.plot(
			alphas,
			nodes_largest_cc,
			color=color_lcc,
			linewidth=2,
			linestyle="-.",
			label="Nodos (mayor CC)",
		)
		lines.append(l5)

	# Reference vertical line
	vline = ax.axvline(
		x=reference_alpha,
		color="grey",
		linestyle="--",
		linewidth=1.2,
		alpha=0.7,
		label=f"alpha = {reference_alpha}",
	)
	lines.append(vline)

	ax.legend(handles=lines, fontsize=10)
	ax.set_title(title, fontsize=13)
	ax.set_xlabel("Alfa", fontsize=12)
	ax.tick_params(axis="y")
	ax.set_ylim(0, 1.05)
	if logscale:
		ax.set_xlim(min(alphas), 1.0)
		ax.set_xscale("log")
	else:
		ax.set_xlim(0, 1)

	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()


def fceyn_compute_and_plot_edge_correlation(
	G: nx.Graph,
	feature_map: dict,
	color_map: dict,
	title: str,
	output_path: Path,
	community_map: dict = None,
	highlight_communities: Iterable[int] = None,
	node_size_map: dict = None,
	legend_label_fmt=None,
	factor_node_size: float = 1.0,
	node_size_exponent: float = 1.0,
	save: bool = True,
	perfect_line: bool = True,
	figsize: tuple = (9, 8),
	font_size: int = 11,
	show: bool | None = None,
) -> None:
	# Only keep nodes that have the feature
	valid_nodes = set(feature_map.keys())

	x_vals_map = {}
	y_vals_map = {u: [] for u in valid_nodes}
	edge_weights = {u: [] for u in valid_nodes}

	for u, v, data in G.edges(data=True):
		if u in valid_nodes and v in valid_nodes:
			# Undirected, we add both (u,v) and (v,u) to make it symmetric
			w = data.get("weight", 0.0)

			x_vals_map[u] = feature_map[u]
			y_vals_map[u].append(feature_map[v])
			edge_weights[u].append(w)

			x_vals_map[v] = feature_map[v]
			y_vals_map[v].append(feature_map[u])
			edge_weights[v].append(w)

	x_vals = []
	y_vals = []
	plotted_nodes = []
	for u in x_vals_map:
		# For each node, we take its feature value and the average of its neighbors' feature values (weighted if desired)
		if not y_vals_map[u]:
			continue
		x_vals.append(x_vals_map[u])
		y_vals.append(np.average(y_vals_map[u], weights=edge_weights[u]))
		plotted_nodes.append(u)

	if len(x_vals) < 2:
		print(
			f"Advertencia: no hay suficientes puntos validos para calcular la correlacion de {title}."
		)
		return

	highlight_set = set(highlight_communities) if highlight_communities else None
	raw_node_sizes = node_size_map or {}

	def fceyn_node_color(node_id: int) -> str:
		if highlight_set and community_map is not None:
			community = community_map.get(node_id)
			if community in highlight_set:
				return color_map.get(node_id, LIGTHGRAY)
			return LIGTHGRAY  # light gray for non-highlighted nodes
		return color_map.get(node_id, LIGTHGRAY)

	# Plot
	plt.figure(figsize=figsize)
	node_sizes = [
		float(
			np.power(max(raw_node_sizes.get(u, 1.0), 0.0), node_size_exponent)
			* factor_node_size
		)
		for u in plotted_nodes
	]

	# Scatter plot of node feature vs average neighbor feature, colored by community
	sns.scatterplot(
		x=x_vals,
		y=y_vals,
		s=node_sizes,
		alpha=0.7,
		c=[fceyn_node_color(u) for u in plotted_nodes],
		edgecolor="white",
		linewidth=0.5,
	)
	if perfect_line:
		plt.plot(
			[0, 100], [0, 100], "k--", label="y=x (Asortatividad perfecta)", alpha=0.5
		)

	# Add a legend for the communities
	if highlight_set and community_map is not None:
		label_fn = legend_label_fmt or (lambda c: f"C{c}")
		for community in sorted(highlight_set):
			color = None
			for node_id, node_community in community_map.items():
				if node_community == community:
					color = color_map.get(node_id, LIGTHGRAY)
					break
			if color:
				plt.scatter([], [], c=color, label=label_fn(community))
	else:
		unique_colors = sorted(
			set(color_map.get(u, LIGTHGRAY) for u in plotted_nodes) - {LIGTHGRAY}
		)
		for color in unique_colors:
			plt.scatter([], [], c=color, label=color)

	# Add regression line on top to show trend
	sns.regplot(
		x=x_vals,
		y=y_vals,
		scatter=False,
		color="red",
		line_kws={"linestyle": "--", "alpha": 0.5},
		label="Trend",
	)

	# Compute correlation
	pearson_r, p_value = stats.pearsonr(x_vals, y_vals)

	print(f"--- {title} ---")
	print(
		f"Coeficiente de asortatividad (Pearson r): {pearson_r:.4f} (p-valor: {p_value:.4e})"
	)
	if pearson_r > 0 and p_value < 0.05:
		print(
			"Correlacion positiva: existe homofilia de genero. Ocupaciones con composiciones similares se agrupan."
		)
	elif pearson_r < 0 and p_value < 0.05:
		print(
			"Correlacion negativa: existe heterofilia. Ocupaciones conectan principalmente con composiciones opuestas."
		)
	else:
		print("Correlacion nula: el genero se distribuye aleatoriamente en la red.")

	plt.title(
		f"{title}\nAsortatividad (Pearson r): {pearson_r:.4f} (p={p_value:.4e})"
		if title
		else None,
		fontsize=font_size + 1,
	)
	plt.xlabel("X_i", fontsize=font_size)
	plt.ylabel("Y_i", fontsize=font_size)
	plt.xticks(fontsize=font_size - 1)
	plt.yticks(fontsize=font_size - 1)
	plt.xlim(-3, 103)
	plt.ylim(-3, 103)
	sns.despine()
	plt.legend(fontsize=font_size - 1)

	# Generate Assortativity Scatter Plot (Node value vs Average Neighbor Value)
	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()
