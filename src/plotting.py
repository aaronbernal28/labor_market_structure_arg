"""
Plotting utilities extracted from the exploratory notebook.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping
import src.utils as ut

from persim import plot_diagrams as persim_plot_diagrams
import gudhi
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
from scipy import stats

LIGTHGRAY = "#a8a8a8"


def sort_labels():
	handles, labels = plt.gca().get_legend_handles_labels()
	sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
	sorted_labels, sorted_handles = zip(*sorted_pairs)

	plt.legend(sorted_handles, sorted_labels)


def _mean_edge_color(color_u, color_v):
	"""Return the mean RGB color between two node colors."""
	rgb_u = np.asarray(mcolors.to_rgb(color_u), dtype=float)
	rgb_v = np.asarray(mcolors.to_rgb(color_v), dtype=float)
	return tuple(((rgb_u + rgb_v) / 2.0).tolist())


def _edge_alpha_from_weight(
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


def _edge_rgba_from_node_colors(color_u, color_v, alpha: float):
	rgb = _mean_edge_color(color_u, color_v)
	return mcolors.to_rgba(rgb, alpha=alpha)


def plot_heatmap(
	biadjacency: pd.DataFrame,
	output_path: Path = None,
	save: bool = True,
	figsize: tuple | None = None,
) -> None:
	"""Plot heatmap of the bipartite adjacency matrix."""
	plt.figure(figsize=figsize)

	# Text Wrapping Configuration
	col_wrap_width = 15
	idx_wrap_width = 35

	wrapped_columns = [
		textwrap.fill(str(col), width=col_wrap_width) for col in biadjacency.columns
	]
	wrapped_index = [
		textwrap.fill(str(idx), width=idx_wrap_width) for idx in biadjacency.index
	]

	biadjacency.columns = wrapped_columns
	biadjacency.index = wrapped_index

	values = biadjacency.to_numpy()
	is_integer = np.issubdtype(values.dtype, np.integer)
	fmt = "d" if is_integer else ".2f"
	ax = sns.heatmap(
		biadjacency,
		annot=True,
		fmt=fmt,
		cmap="Greens",
		cbar=False,
	)

	ax.xaxis.tick_top()  # Move ticks to top
	ax.xaxis.set_label_position("top")
	plt.xticks(rotation=0)

	# Row labels: align LEFT
	ax.set_yticklabels(ax.get_yticklabels(), ha="left")
	ax.tick_params(axis="y", pad=190)
	ax.tick_params(left=False, top=False)

	plt.xlabel("")
	plt.ylabel("")
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def _wrap_labels(df: pd.DataFrame) -> pd.DataFrame:
	wrapped = df.copy()
	col_wrap_width = 15
	idx_wrap_width = 35
	wrapped.columns = [
		textwrap.fill(str(c), width=col_wrap_width) for c in wrapped.columns
	]
	wrapped.index = [textwrap.fill(str(i), width=idx_wrap_width) for i in wrapped.index]
	return wrapped


def _style_heatmap_axes(ax, title: str) -> None:
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position("top")
	plt.xticks(rotation=0)
	plt.yticks(rotation=0)
	ax.set_yticklabels(ax.get_yticklabels(), ha="left")
	ax.tick_params(axis="y", pad=12)
	ax.tick_params(left=False, top=False)
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.set_title(title, pad=12)


def plot_rejection_heatmap(
	p_values: np.ndarray,
	rejected: np.ndarray,
	rownames: Iterable[str],
	colnames: Iterable[str],
	bonferroni_threshold: float,
	output_path: Path,
	save: bool = True,
	figsize: tuple | None = None,
) -> None:
	"""Heatmap of p-values: black below Bonferroni threshold, YlOrRd above it."""
	from matplotlib.colors import ListedColormap

	# Smart formatting: hide extreme values, automatic scientific notation for others
	def format_pvalue(v):
		if v > 0.99:  # Hide extreme high p-values (essentially 1)
			return ""
		else:
			return f"{v:.3g}"

	annot = np.vectorize(format_pvalue)(p_values)
	df = pd.DataFrame(p_values, index=rownames, columns=colnames)
	df = _wrap_labels(df)
	annot_df = pd.DataFrame(annot, index=df.index, columns=df.columns)
	n_rejected = int(rejected.sum())
	title = (
		f"Prueba de Wald - p-valores (Bonferroni alpha/d = {bonferroni_threshold:.3g})\n"
		f"n rechazadas = {n_rejected} / {rejected.size} ({100 * n_rejected / rejected.size:.1f}%)"
	)

	n_black = 25
	n_total = max(256, int(round(n_black / bonferroni_threshold)))
	n_rest = n_total - n_black
	base_cmap = plt.get_cmap("plasma", n_rest)
	black_colours = np.tile([0.0, 0.0, 0.0, 1.0], (n_black, 1))
	rest_colours = base_cmap(np.linspace(0, 1, n_rest))
	all_colours = np.vstack([black_colours, rest_colours])
	cmap = ListedColormap(all_colours)

	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(
		df,
		ax=ax,
		annot=annot_df,
		fmt="",
		cmap=cmap,
		vmin=0,
		vmax=1,
		cbar=True,
		cbar_kws={"shrink": 0.35, "label": "p-valor"},
		alpha=0.9,
	)
	# Mark the Bonferroni threshold on the colorbar
	cbar = ax.collections[0].colorbar
	cbar.ax.axhline(
		y=bonferroni_threshold, color="white", linewidth=1.5, linestyle="--"
	)
	cbar.ax.text(
		1.05,
		bonferroni_threshold,
		f"{bonferroni_threshold:.2e}",
		va="center",
		ha="left",
		transform=cbar.ax.transData,
		color="dimgray",
	)
	_style_heatmap_axes(ax, title)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_delta_heatmap(
	delta_hat: np.ndarray,
	rownames: Iterable[str],
	colnames: Iterable[str],
	output_path: Path,
	save: bool = True,
	figsize: tuple | None = None,
) -> None:
	"""Diverging heatmap showing delta_hat annotations in scientific notation."""

	# Smart formatting: hide extreme values, automatic scientific notation for others
	def format_delta(v):
		if abs(v) < 0.0001:  # Hide essentially zero differences
			return ""
		else:
			return f"{v:.3g}"

	annot = np.vectorize(format_delta)(delta_hat)
	df = pd.DataFrame(delta_hat, index=rownames, columns=colnames)
	df = _wrap_labels(df)
	annot_df = pd.DataFrame(annot, index=df.index, columns=df.columns)
	abs_max = np.max(np.abs(delta_hat))
	title = "Diferencia estimada delta = p(ENES 2019) - p(ESAyPP 2021)"
	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(
		df,
		ax=ax,
		annot=annot_df,
		fmt="",
		cmap="twilight_shifted",
		vmin=-abs_max,
		vmax=abs_max,
		cbar=True,
		cbar_kws={"shrink": 0.35, "label": "delta"},
	)
	_style_heatmap_axes(ax, title)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_bootstrap_se_heatmap(
	se_boot: np.ndarray,
	rownames: Iterable[str],
	colnames: Iterable[str],
	output_path: Path,
	save: bool = True,
	figsize: tuple | None = None,
) -> None:
	"""Heatmap of bootstrap SE estimates (B=1000) for the delta proportions."""

	# Smart formatting: hide extreme values, automatic scientific notation for others
	def format_se(v):
		if v < 0.00001:  # Hide essentially zero SE
			return ""
		else:
			return f"{v:.3g}"

	annot = np.vectorize(format_se)(se_boot)
	df = pd.DataFrame(se_boot, index=rownames, columns=colnames)
	df = _wrap_labels(df)
	annot_df = pd.DataFrame(annot, index=df.index, columns=df.columns)
	title = "Bootstrap SE de delta (B=1000)"
	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(
		df,
		ax=ax,
		annot=annot_df,
		fmt="",
		cmap="YlOrRd",
		cbar=True,
		cbar_kws={"shrink": 0.35, "label": "SE bootstrap"},
	)
	_style_heatmap_axes(ax, title)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def draw_bipartite_by_color(
	graph: nx.Graph,
	color_map: Dict[int, str],
	label_map: Dict[int, str] = None,
	output_path: Path = None,
	seed: int = 28,
	top_n: int = 6,
	shift_x: float = 0.5,
	title: str = "",
	save: bool = True,
	figsize: tuple | None = None,
	edge_alpha: float = 0.9,
	legend_marker_size: float = 11.0,
	factor_node_size: float = 3.0,
	node_size_map: Mapping[int, float] = None,
	node_size_exponent: float = 1.0,
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

	if set(color_map.keys()) >= set(graph.nodes()):
		print("Graph contains nodes not present in color map.")

	if label_map is None or set(label_map.keys()) >= set(graph.nodes()):
		print("Graph contains nodes not present in label map.")

	# Compute initial layout
	pos = nx.spring_layout(graph, seed=seed, k=0.5, iterations=1000, method="force")
	pos_caes_y = [
		pos[node][1]
		for node in graph.nodes()
		if graph.nodes[node].get("bipartite") == ut.get_class_index("caes")
	]
	pos_ciuo_y = [
		pos[node][1]
		for node in graph.nodes()
		if graph.nodes[node].get("bipartite") == ut.get_class_index("ciuo")
	]

	# Sigmoidal normalization functions
	sigmoid_ = lambda x: 1 / (1 + np.exp(-x))
	normalize_caes_y = lambda y: sigmoid_(
		2.5 * (y - np.mean(pos_caes_y)) / np.std(pos_caes_y)
	)
	normalize_ciuo_y = lambda y: sigmoid_(
		2.5 * (y - np.mean(pos_ciuo_y)) / np.std(pos_ciuo_y)
	)

	# Adjust positions (shift axis x and normalize y)
	for node in graph.nodes():
		if graph.nodes[node].get("bipartite") == ut.get_class_index("caes"):
			pos[node][0] -= shift_x
			pos[node][1] = normalize_caes_y(pos[node][1])
		else:
			pos[node][0] += shift_x
			pos[node][1] = normalize_ciuo_y(pos[node][1])

	# Defining color and size maps
	# node_colors = [color_map.get(int(node), LIGTHGRAY) for node in graph.nodes()]
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
		if graph.nodes[node].get("bipartite") == ut.get_class_index("caes")
	]
	ciuo_nodes = [
		node
		for node in graph.nodes()
		if graph.nodes[node].get("bipartite") == ut.get_class_index("ciuo")
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
			if graph.nodes[node].get("bipartite") == ut.get_class_index("caes")
		][:top_n]
		top_nodes_ciuo = [
			node
			for node, _ in sorted_degrees
			if graph.nodes[node].get("bipartite") == ut.get_class_index("ciuo")
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
			if node_id in label_map:
				lbl = label_map[node_id]
				color = color_map.get(node_id, LIGTHGRAY)
				if graph.nodes[node].get("bipartite") == ut.get_class_index("caes"):
					caes_groups[lbl] = color
				else:
					ciuo_groups[lbl] = color

	def _make_handles(groups, marker_shape="o"):
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
			handles=_make_handles(caes_groups, marker_shape="^"),
			title="Ramas de actividad (CAES)",
			loc="upper left",
			bbox_to_anchor=(0.0, 1.0),
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)
		plt.gca().add_artist(leg_l)
	if ciuo_groups:
		plt.legend(
			handles=_make_handles(ciuo_groups, marker_shape="o"),
			title="Ocupaciones (CIUO)",
			loc="upper right",
			bbox_to_anchor=(1.0, 1.0),
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)

	# Finalize plot
	plt.title(title)
	plt.axis("off")
	plt.xlim(-1.4, 1.6)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()
	return pos


def draw_bipartite_normal_layout_by_color(
	graph: nx.Graph,
	color_map: Dict[int, str],
	label_map: Dict[int, str] = None,
	output_path: Path = None,
	seed: int = 28,
	top_n: int = 6,
	title: str = "",
	save: bool = True,
	figsize: tuple | None = None,
	node_scale: float = 8.0,
	edge_alpha: float = 0.35,
	edge_lw: float = 0.15,
) -> Dict[str, tuple]:
	"""Draw the bipartite network with nodes sorted by group to minimise edge crossings."""
	import matplotlib.patches as mpatches

	assert set(color_map.keys()) >= set(graph.nodes()), (
		"Graph contains nodes not present in color map."
	)
	assert label_map is None or set(label_map.keys()) >= set(graph.nodes()), (
		"Graph contains nodes not present in label map."
	)

	caes_idx = ut.get_class_index("caes")
	ciuo_idx = ut.get_class_index("ciuo")

	caes_nodes = [
		n for n in graph.nodes() if graph.nodes[n].get("bipartite") == caes_idx
	]
	ciuo_nodes = [
		n for n in graph.nodes() if graph.nodes[n].get("bipartite") == ciuo_idx
	]

	# Sort each partition by group label so same-colored nodes are contiguous,
	# which dramatically reduces edge crossings.
	group_key = lambda n: label_map.get(int(n), str(n)) if label_map else str(n)
	caes_nodes = sorted(caes_nodes, key=group_key)
	ciuo_nodes = sorted(ciuo_nodes, key=group_key)

	def _linear_positions(nodes, x):
		n = len(nodes)
		ys = np.linspace(1.0, -1.0, n) if n > 1 else [0.0]
		return {node: np.array([x, y]) for node, y in zip(nodes, ys)}

	pos = {}
	pos.update(_linear_positions(caes_nodes, -1.0))
	pos.update(_linear_positions(ciuo_nodes, 1.0))

	fig, ax = plt.subplots(figsize=figsize)
	ax.set_aspect("auto")
	ax.axis("off")

	# ── Bezier edges (sorted by CIUO y-position for visual grouping) ─────────
	path_cls = mpath.Path
	edges_sorted = sorted(
		graph.edges(),
		key=lambda e: pos[e[1]][1],  # sort by target y → reduces colour mixing
	)
	weights = [graph[u][v].get("weight", 0.0) for u, v in edges_sorted]
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
		w = graph[u][v].get("weight", 0.0)
		alpha = _edge_alpha_from_weight(w, max_weight, alpha_max=edge_alpha)
		edge_color = _edge_rgba_from_node_colors(color_u, color_v, alpha)
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

	def _make_handles(groups):
		return [
			mpatches.Patch(facecolor=c, label=textwrap.fill(lbl, 28), linewidth=0)
			for lbl, c in sorted(groups.items())
		]

	if caes_groups:
		leg_l = ax.legend(
			handles=_make_handles(caes_groups),
			title="Ramas economicas\n(CAES)",
			loc="upper left",
			bbox_to_anchor=(0.0, 1.0),
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)
		ax.add_artist(leg_l)
	if ciuo_groups:
		ax.legend(
			handles=_make_handles(ciuo_groups),
			title="Ocupaciones\n(CIUO)",
			loc="upper right",
			bbox_to_anchor=(1.0, 1.0),
			framealpha=0.85,
			edgecolor="lightgray",
			handlelength=1.2,
			handleheight=1.0,
			labelspacing=1.2,
		)

	ax.set_xlim(-1.25, 1.25)
	ax.set_title(title, pad=12)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()
	return pos


def plot_degree_histogram(
	degrees: Iterable,
	color: str,
	title: str,
	output_path: Path = None,
	save: bool = False,
	ax: plt.Axes = None,
	logscale: bool = False,
	is_discrete: bool = True,
) -> None:
	"""Plot degree distribution as scatter plot for discrete data or histogram for continuous data."""
	created_fig = ax is None
	if created_fig:
		fig, ax = plt.subplots()

	degrees_array = np.array(list(degrees))

	# Use scatter plot for discrete data (few unique values), histogram for continuous
	if logscale:
		ax.set_xscale("log")
		ax.set_yscale("log")

	if is_discrete:
		# Count frequency of each degree value and plot as scatter
		degree_counts = pd.Series(degrees_array).value_counts().sort_index()
		sns.scatterplot(
			x=degree_counts.index,
			y=degree_counts.values,
			color=color,
			s=50,
			alpha=0.7,
			ax=ax,
		)
		if not logscale:
			ax.set_ylim(-0.05 * degree_counts.max(), degree_counts.max() * 1.05)
	else:
		# Use seaborn histogram for continuous data
		sns.histplot(degrees_array, color=color, alpha=0.5, ax=ax)
		if not logscale:
			ylim_max = max(ax.get_ylim()[1], 1) * 1.05
			ax.set_ylim(-0.05 * ylim_max, ylim_max)

	ax.set_xlabel("k")
	ax.set_ylabel("Frecuencia")
	ax.set_title(f"{title}\n<k> = {np.mean(degrees_array):.2f}")
	ax.grid(True, alpha=0.3)

	if created_fig:
		if save and output_path is not None:
			plt.savefig(output_path, bbox_inches="tight")
			plt.close()
		else:
			plt.show()


def plot_degree_histograms(
	degrees: Dict[str, list],
	colors: Dict[str, str],
	output_path: Path = None,
	save: bool = True,
	logscale: bool = False,
) -> None:
	"""Plot degree histograms for all nodes, CAES nodes, and CIUO nodes."""
	fig, axes = plt.subplots(1, 3)
	configs = [
		(degrees["all"], colors["all"], "Grados (todos los nodos)"),
		(degrees["caes"], colors["caes"], "Grados CAES"),
		(degrees["ciuo"], colors["ciuo"], "Grados CIUO"),
	]
	for i, (values, color, title) in enumerate(configs):
		plot_degree_histogram(values, color, title, ax=axes[i], logscale=logscale)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_projection_by_group(
	graph: nx.Graph,
	group_map: Mapping[str, str],
	group_color_map: Mapping[str, str],
	title: str,
	legend_title: str,
	output_path: Path = None,
	figsize: tuple | None = None,
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
	if not set(group_color_map.keys()) >= set(group_map.values()):
		print("Color map is missing groups present in group map.")
	if not set(graph.nodes()) <= set(group_map.keys()):
		print("Graph contains nodes not present in group map.")

	# Get the largest connected component for layout
	if graph.number_of_nodes() == 0:
		print("Warning: Projection graph is empty; skipping plot.")
		return {}
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
			weights = [graph[u][v].get("weight", 0.0) for u, v in edges]
			max_weight = max(weights) if max(weights) > 0 else 1.0
			edge_widths = [0.1 + 1.9 * (w / max_weight) for w in weights]
			edge_alphas = [
				_edge_alpha_from_weight(w, max_weight, alpha_max=edge_alpha)
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
				_edge_rgba_from_node_colors(
					node_color_by_node[u], node_color_by_node[v], a
				)
				for (u, v), a in zip(edges, edge_alphas)
			]
		else:
			edge_colors = [
				_edge_rgba_from_node_colors(
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

	groups = list(group_color_map.keys())
	ordered_groups = sorted(groups, key=lambda value: str(value).lower())

	for group in ordered_groups:
		plt.scatter([], [], color=group_color_map[group], label=group)

	plt.legend(
		title=legend_title,
		loc="best",
		borderaxespad=2.0,
		framealpha=0.7,
	)
	plt.title(title)
	plt.axis("off")
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()
	return pos


def plot_projection_gradient(
	graph: nx.Graph,
	pos: dict,
	node_values: dict,
	title: str,
	colorbar_label: str = "Valor",
	cmap: str = "viridis",
	figsize: tuple | None = None,
	output_path: Path = None,
	save: bool = True,
	factor_node_size: float = 0.5,
	node_size_map: Mapping[int, float] = None,
	vmin: float = None,
	vmax: float = None,
	node_size_exponent: float = 1.0,
	edge_alpha: float = 0.1,
	node_alpha: float = 0.5,
):
	"""Plot the projection network with nodes colored by a continuous scalar gradient.

	Parameters:
	- graph: NetworkX graph to plot.
	- pos: Precomputed node positions (from plot_projection_by_group).
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
			weights = [subgraph[u][v].get("weight", 0.0) for u, v in edges]
			max_weight = max(weights) if max(weights) > 0 else 1.0
			edge_widths = [0.1 + 1.9 * (w / max_weight) for w in weights]
			edge_alphas = [
				_edge_alpha_from_weight(w, max_weight, alpha_max=edge_alpha)
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
				_edge_rgba_from_node_colors(
					node_color_by_node.get(u, "lightgray"),
					node_color_by_node.get(v, "lightgray"),
					a,
				)
				for (u, v), a in zip(edges, edge_alphas)
			]
		else:
			edge_colors = [
				_edge_rgba_from_node_colors(
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
	cbar.set_label(colorbar_label)

	ax.set_title(title)
	ax.axis("off")
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_stacked_by_group(
	df_index: pd.DataFrame,
	group_col: str,
	community_map: Dict[int, int],
	title: str,
	output_path: Path,
	group_color_map: Dict[str, tuple] = None,
	legend_title: str = None,
	figsize: tuple | None = None,
	save: bool = True,
	percentage: bool = True,
) -> None:
	"""Plot stacked bar chart showing distribution of groups within communities."""
	df_index_copy = df_index.copy()
	df_index_copy["community"] = df_index_copy.index.map(community_map)
	df_index_copy = df_index_copy.dropna(subset=["community"])

	# Create crosstab and normalize if needed
	if percentage:
		ct = (
			pd.crosstab(
				df_index_copy["community"], df_index_copy[group_col], normalize="index"
			)
			* 100
		)
	else:
		ct = pd.crosstab(df_index_copy["community"], df_index_copy[group_col])

	# Build color list matching the column order
	if group_color_map:
		colors = [group_color_map.get(col, "gray") for col in ct.columns]
		plot_kwargs = {"kind": "barh", "stacked": True, "width": 0.8, "color": colors}
		if figsize is not None:
			plot_kwargs["figsize"] = figsize
		ax = ct.plot(**plot_kwargs)
	else:
		plot_kwargs = {"kind": "barh", "stacked": True, "width": 0.8}
		if figsize is not None:
			plot_kwargs["figsize"] = figsize
		ax = ct.plot(**plot_kwargs)

	ax.set_xlabel("Porcentaje (%)" if percentage else "Conteo")
	ax.set_title(title)
	ax.set_xlim(0, 100 if percentage else None)
	legend_title = legend_title or group_col
	ax.legend(
		title=legend_title,
		bbox_to_anchor=(1.05, 1),
		loc="upper left",
	)

	# Format y-axis labels as C0, C1, ...
	yticks = ax.get_yticks()
	ax.set_yticklabels([f"C{int(y)}" for y in yticks])

	# Remove axis borders
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["bottom"].set_visible(False)

	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_distance_histogram(
	distance_matrix: np.ndarray,
	output_path: Path = None,
	bins: int = 30,
	title: str = "Histograma de distancias",
	include_infinite: bool = True,
	ax=None,
	save: bool = True,
) -> None:
	"""Plot histogram of finite distances from a distance matrix.

	If `ax` is provided, the plot is drawn onto that axis and the function will not
	save/show/close figures.
	"""
	values = np.asarray(distance_matrix, dtype=float).ravel()
	finite = values[np.isfinite(values) & (values > 0)]
	inf_count = np.isinf(values).sum()

	created_fig = False
	if ax is None:
		fig, ax = plt.subplots()
		created_fig = True
	counts, bin_edges, _ = ax.hist(finite, bins=bins, alpha=0.8, color="steelblue")
	if include_infinite and inf_count > 0:
		bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
		inf_x = bin_edges[-1] + bin_width
		ax.bar(
			[inf_x],
			[inf_count],
			width=bin_width * 0.8,
			color="tomato",
			alpha=0.8,
		)
		ticks, tick_labels = ax.get_xticks(), [str(x) for x in ax.get_xticks()]
		ax.set_xticks(list(ticks) + [inf_x])
		ax.set_xticklabels(tick_labels + ["inf"])
	ax.set_xlabel("Distancia")
	ax.set_ylabel("Frecuencia")
	ax.set_title(f"{title}\nfinito={len(finite)} | inf={inf_count}")
	ax.grid(True, alpha=0.3)
	if created_fig:
		if save:
			fig.savefig(output_path, bbox_inches="tight")
			plt.close(fig)
		else:
			plt.show()


def plot_distance_heatmap(
	distance_matrix: np.ndarray,
	output_path: Path = None,
	title: str = "Matriz de distancias",
	labels: Iterable[str] = None,
	ax=None,
	save: bool = True,
) -> None:
	"""Plot heatmap for a distance matrix (infinite distances are masked).

	If `ax` is provided, the plot is drawn onto that axis and the function will not
	save/show/close figures.
	"""
	data = np.asarray(distance_matrix, dtype=float).copy()
	data[np.isinf(data)] = np.nan
	mask = np.isnan(data)

	created_fig = False
	if ax is None:
		fig, ax = plt.subplots()
		created_fig = True
	ax = sns.heatmap(data, cmap="mako", mask=mask, cbar=True, ax=ax)
	if labels:
		ax.set_xticks(np.arange(len(labels)) + 0.5)
		ax.set_yticks(np.arange(len(labels)) + 0.5)
		ax.set_xticklabels(
			[f"{i:02d}" for i in range(len(labels))],
			rotation=45,
			ha="right",
		)
		ax.set_yticklabels([f"{i:02d}" for i in range(len(labels))], rotation=0)
		ax.set_xticklabels(labels, rotation=45, ha="right")
		ax.set_yticklabels(labels, rotation=0)
	else:
		ax.set_xticks([])
		ax.set_yticks([])
	ax.set_title(title)
	if created_fig:
		if save:
			fig.savefig(output_path, bbox_inches="tight")
			plt.close(fig)
		else:
			plt.show()


def plot_backbone_weight_histogram(
	original_weights: list,
	backbone_weights: list,
	alpha: float,
	title_prefix: str,
	output_path: Path,
	save: bool = True,
	figsize: tuple | None = None,
) -> None:
	"""Plot overlapped histograms comparing original and backbone edge weights."""
	fig, ax = plt.subplots(figsize=figsize)

	sns.histplot(
		original_weights,
		bins=50,
		kde=True,
		ax=ax,
		color="steelblue",
		alpha=0.3,
		label=f"Original ({len(original_weights)} aristas)",
	)
	sns.histplot(
		backbone_weights,
		bins=50,
		kde=True,
		ax=ax,
		color="coral",
		alpha=0.3,
		label=f"Esqueleto ({len(backbone_weights)} aristas)",
	)

	if title_prefix:
		ax.set_title(
			f"{title_prefix} Distribucion de pesos de aristas: Original vs Esqueleto (alpha={alpha})"
		)

	ax.set_xlabel("Peso de arista")
	ax.set_ylabel("Frecuencia")
	ax.set_yscale("log")
	# ax.set_ylim(bottom=1e-1)
	ax.legend()

	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_persistence_diagrams(
	diagrams,
	title: str = "Persistence Diagrams",
	output_path: Path = None,
	ax=None,
	save: bool = True,
):
	"""Plot persistence diagrams.

	If `ax` is provided, the plot is drawn onto that axis and the function will not
	save/show/close figures.
	"""
	if persim_plot_diagrams is None:
		raise ImportError("persim is required for plot_persistence_diagrams.")

	created_fig = False
	if ax is None:
		fig, ax = plt.subplots()
		created_fig = True
	plt.sca(ax)
	persim_plot_diagrams(diagrams, show=False)
	ax.set_title(title)
	if created_fig:
		if save:
			fig.savefig(output_path)
			plt.close(fig)
		else:
			plt.show()


def plot_persistence_barcodes(
	diagrams,
	title: str = "Persistence Barcodes",
	output_path: Path = None,
	ax=None,
	save: bool = True,
) -> None:
	"""Plot persistence barcodes using GUDHI.

	If `ax` is provided, the plot is drawn onto that axis and the function will not
	save/show/close figures.
	"""
	if gudhi is None:
		raise ImportError("gudhi is required for plot_persistence_barcodes.")

	finite_deaths: list[float] = []
	for dgm in diagrams:
		for _, death in dgm:
			if np.isfinite(death):
				finite_deaths.append(float(death))
	clip_value = (max(finite_deaths) if finite_deaths else 1.0) * 1.05

	gudhi_diagrams = []
	for dim, dgm in enumerate(diagrams):
		if len(dgm) == 0:
			continue
		for birth, death in dgm:
			b = float(birth)
			d = float(death)
			if not np.isfinite(b):
				continue
			if not np.isfinite(d):
				d = float(clip_value)
			gudhi_diagrams.append((dim, (b, d)))

	created_fig = False
	if ax is None:
		fig, ax = plt.subplots()
		created_fig = True
	# Convert colormap to a list that can be indexed by dimension
	cmap = mpl.colormaps["tab10"]
	colormap_list = [cmap(i) for i in range(len(cmap.colors))]
	gudhi.plot_persistence_barcode(
		persistence=gudhi_diagrams, axes=ax, colormap=colormap_list
	)
	ax.set_title(title)
	if created_fig:
		if save:
			fig.savefig(output_path)
			plt.close(fig)
		else:
			plt.show()


def color_map_caes(caes_nodes: Iterable[str]) -> Dict[int, str]:
	"""Create a color map for CAES nodes based on their group."""
	cmap = mpl.colormaps["Accent"]
	x = np.linspace(
		0, 1, max(caes_nodes) + 1
	)  # hypothesis: len(caes_nodes) << max(caes_nodes)
	caes_nodes = sorted(caes_nodes, reverse=False)
	return {node: cmap(x[node]) for node in caes_nodes}


def color_map_ciuo(ciuo_nodes: Iterable[str], max_caes_id: int) -> Dict[int, str]:
	"""Create a color map for CIUO nodes based on their group."""
	cmap = mpl.colormaps["inferno"]
	# Mapping from input node (desambiated) to original ID
	node_to_original = {
		node: ut.original_ciuo_id(int(node), max_caes_id=max_caes_id)
		for node in ciuo_nodes
	}
	original_ids = list(node_to_original.values())

	if not original_ids:
		return {}

	x = np.linspace(0, 1, max(original_ids) + 1)
	# Return map from input node to color
	return {node: cmap(x[orig_id]) for node, orig_id in node_to_original.items()}


def mean_color(colors):
	colors_array = np.array([list(c) for c in colors])
	return tuple(colors_array.mean(axis=0))


def color_letra_map_caes(
	caes_df: pd.DataFrame, letra_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CAES letra based on their group."""
	return caes_df.groupby(letra_col)[base_color_col].apply(mean_color).to_dict()


def color_1digit_map_ciuo(
	ciuo_df: pd.DataFrame, letra_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CIUO letra based on their group."""
	return ciuo_df.groupby(letra_col)[base_color_col].apply(mean_color).to_dict()


def color_agrupation_map_caes(
	caes_df: pd.DataFrame, ag_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CAES agrupation based on their group."""
	return caes_df.groupby(ag_col)[base_color_col].apply(mean_color).to_dict()


def color_ciuo3cat_map_ciuo(
	ciuo_df: pd.DataFrame, cat_col: str, base_color_col: str
) -> Dict[str, str]:
	"""Create a color map for CIUO 3-category based on their group."""
	return ciuo_df.groupby(cat_col)[base_color_col].apply(mean_color).to_dict()


def plot_top_n_bar(
	df: pd.DataFrame,
	label_col: str,
	val_col: str,
	color_col: str,
	title: str,
	xlabel: str,
	top_n: int = 15,
	figsize: tuple | None = None,
	output_path: Path = None,
	save: bool = True,
) -> None:
	if val_col not in df.columns or label_col not in df.columns:
		return
	top_df = df.nlargest(top_n, val_col)

	palette_by_label = {}
	for _, row in top_df.iterrows():
		label = row[label_col]
		if color_col in top_df.columns:
			palette_by_label[label] = ut.parse_color(row[color_col])
		else:
			palette_by_label[label] = "steelblue"

	plt.figure(figsize=figsize)
	ax = sns.barplot(
		data=top_df,
		x=val_col,
		y=label_col,
		hue=label_col,
		dodge=False,
		palette=palette_by_label,
		legend=False,
	)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel("")

	ax.xaxis.set_major_formatter(
		plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
	)

	# Remove axis borders
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	plt.tight_layout()

	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_exploratory_analysis(
	caes_nodes: pd.DataFrame, ciuo_nodes: pd.DataFrame, output_dir: Path
) -> None:
	"""Generate exploratory data analysis plots for node characteristics."""
	output_dir.mkdir(parents=True, exist_ok=True)

	def plot_scatter(
		x_col, y_col, xlabel, ylabel, filename, hue_col=None, hue_label=None
	):
		fig, axes = plt.subplots(1, 2)

		if x_col in caes_nodes.columns and y_col in caes_nodes.columns:
			kwargs = {
				"data": caes_nodes,
				"x": x_col,
				"y": y_col,
				"ax": axes[0],
				"alpha": 0.7,
			}
			if "n_obs" in caes_nodes.columns:
				caes_nodes["n_obs"] = caes_nodes["n_obs"].astype(float)
				kwargs["size"] = "n_obs"
				kwargs["sizes"] = (20, 500)
			if hue_col and hue_col in caes_nodes.columns:
				kwargs["hue"] = hue_col
				kwargs["palette"] = "viridis"

			sns.scatterplot(**kwargs)
			axes[0].set_title("Economic Branches (CAES)")
			axes[0].set_xlabel(xlabel)
			axes[0].set_ylabel(ylabel)
			if hue_col and hue_col in caes_nodes.columns and axes[0].get_legend():
				axes[0].legend(
					title=hue_label, bbox_to_anchor=(1.05, 1), loc="upper left"
				)
			elif "n_obs" in caes_nodes.columns and axes[0].get_legend():
				axes[0].legend(
					title="Total Workers", bbox_to_anchor=(1.05, 1), loc="upper left"
				)

		if x_col in ciuo_nodes.columns and y_col in ciuo_nodes.columns:
			kwargs = {
				"data": ciuo_nodes,
				"x": x_col,
				"y": y_col,
				"ax": axes[1],
				"alpha": 0.7,
			}
			if "n_obs" in ciuo_nodes.columns:
				ciuo_nodes["n_obs"] = ciuo_nodes["n_obs"].astype(float)
				kwargs["size"] = "n_obs"
				kwargs["sizes"] = (20, 500)
			if hue_col and hue_col in ciuo_nodes.columns:
				kwargs["hue"] = hue_col
				kwargs["palette"] = "viridis"

			sns.scatterplot(**kwargs)
			axes[1].set_title("Occupations (CIUO)")
			axes[1].set_xlabel(xlabel)
			axes[1].set_ylabel(ylabel)
			if hue_col and hue_col in ciuo_nodes.columns and axes[1].get_legend():
				axes[1].legend(
					title=hue_label, bbox_to_anchor=(1.05, 1), loc="upper left"
				)
			elif "n_obs" in ciuo_nodes.columns and axes[1].get_legend():
				axes[1].legend(
					title="Total Workers", bbox_to_anchor=(1.05, 1), loc="upper left"
				)

		plt.savefig(output_dir / filename, bbox_inches="tight")
		plt.close()

	plot_scatter(
		"higher_education_pct",
		"income_median",
		"% Higher Education",
		"Median Income",
		"00_income_vs_education.png",
		hue_col="female_pct",
		hue_label="% Female",
	)
	plot_scatter(
		"female_pct",
		"income_median",
		"% Female Workers",
		"Median Income",
		"00_income_vs_gender.png",
		hue_col="higher_education_pct",
		hue_label="% Higher Ed",
	)
	plot_scatter(
		"age_median",
		"income_median",
		"Median Age",
		"Median Income",
		"00_age_vs_income.png",
		hue_col="salaried_pct",
		hue_label="% Salaried",
	)

	for df in [caes_nodes, ciuo_nodes]:
		if "income_q3" in df.columns and "income_q1" in df.columns:
			df["income_iqr"] = df["income_q3"] - df["income_q1"]

	plot_scatter(
		"income_median",
		"income_iqr",
		"Median Income",
		"Income IQR (Q3 - Q1)",
		"00_income_inequality.png",
		hue_col="higher_education_pct",
		hue_label="% Higher Ed",
	)


def plot_alpha_sensitivity(
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
) -> None:
	"""Plot backbone sensitivity to alpha: relative nodes-with-edges and edge fraction on
	the left y-axis; clustering coefficient and (optionally) modularity on the right y-axis.

	Parameters
	----------
	alphas: 1-D array of alpha values in (0, 1).
	nodes_with_edges: fraction of nodes that have at least one edge at each alpha.
	edge_counts: fraction of edges retained at each alpha.
	clustering_coefficients: Average clustering coefficient ponderado at each alpha.
	title: plot title (typically the network name).
	output_path: destination file path.
	modularities: Louvain modularity at each alpha (optional).
	nodes_largest_cc: fraction of nodes in the largest connected component (optional).
	reference_alpha: vertical reference line (default 0.05).
	save: if True save to file, else show interactively.
	"""
	fig, ax = plt.subplots()

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
		label="Coef. de clustering prom. ponderado",
	)

	lines = [l1, l2, l3]
	if modularities is not None:
		(l4,) = ax.plot(
			alphas,
			modularities,
			color=color_mod,
			linewidth=2,
			label="Modularidad",
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

	ax.legend(handles=lines)
	ax.set_title(title)
	ax.set_xlabel("Alfa")
	ax.tick_params(axis="y")
	ax.set_ylim(0, 1.05)
	if logscale:
		ax.set_xlim(min(alphas), 1.0)
		ax.set_xscale("log")
	else:
		ax.set_xlim(0, 1)

	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def plot_alpha_sensitivity_multi_series(
	alphas: np.ndarray,
	series_labels: list[str],
	*,
	nodes_with_edges: np.ndarray,
	edge_counts: np.ndarray,
	clustering_coefficients: np.ndarray,
	nodes_largest_cc: np.ndarray,
	title: str,
	output_path: Path,
	reference_alphas: np.ndarray | None = None,
	save: bool = True,
	logscale: bool = True,
	max_cbar_ticks: int = 10,
) -> None:
	"""Plot alpha-sensitivity curves for multiple series.

	This overlays the same 5 metrics as :func:`plot_alpha_sensitivity` for many
	input graphs (e.g. multiple EPH files). Each metric uses a colormap family,
	where shade encodes chronological order of the series.
	"""
	fig, ax = plt.subplots()

	if len(series_labels) == 0:
		raise ValueError("series_labels must be non-empty")

	n_series = len(series_labels)
	for arr_name, arr in {
		"nodes_with_edges": nodes_with_edges,
		"edge_counts": edge_counts,
		"clustering_coefficients": clustering_coefficients,
		"nodes_largest_cc": nodes_largest_cc,
	}.items():
		if arr.shape != (n_series, len(alphas)):
			raise ValueError(
				f"{arr_name} must have shape (n_series, n_alphas) = ({n_series}, {len(alphas)}); got {arr.shape}"
			)

	if reference_alphas is not None and len(reference_alphas) != n_series:
		raise ValueError(
			"reference_alphas must be None or have length equal to series_labels"
		)

	norm = mpl.colors.Normalize(vmin=0, vmax=max(1, n_series - 1))

	metric_specs = [
		{
			"label": "Nodos",
			"data": nodes_with_edges,
			"cmap": mpl.cm.Blues,
			"linestyle": "-",
		},
		{
			"label": "Aristas",
			"data": edge_counts,
			"cmap": mpl.cm.Oranges,
			"linestyle": "--",
		},
		{
			"label": "Coef. de clustering prom. ponderado",
			"data": clustering_coefficients,
			"cmap": mpl.cm.Greens,
			"linestyle": ":",
		},
		{
			"label": "Nodos (mayor CC)",
			"data": nodes_largest_cc,
			"cmap": mpl.cm.Reds,
			"linestyle": "-.",
		},
	]

	proxy_lines = []
	for spec in metric_specs:
		cmap = spec["cmap"]
		for i in range(n_series):
			color = cmap(norm(i))
			ax.plot(
				alphas,
				spec["data"][i, :],
				color=color,
				linewidth=1.4,
				linestyle=spec["linestyle"],
				alpha=0.9,
			)
		# Proxy for metric legend (use darkest shade)
		proxy_lines.append(
			mpl.lines.Line2D(
				[0],
				[0],
				color=cmap(norm(n_series - 1)),
				linewidth=2,
				linestyle=spec["linestyle"],
				label=spec["label"],
			)
		)

	# Optional: per-series reference alpha as light vertical lines
	if reference_alphas is not None:
		ref_cmap = mpl.cm.Greys
		for i, ref in enumerate(reference_alphas):
			if not np.isfinite(ref):
				continue
			ax.axvline(
				x=float(ref),
				color=ref_cmap(norm(i)),
				linestyle="--",
				linewidth=0.8,
				alpha=0.22,
			)

	# Metric legend only (avoid giant eph_file legend)
	ax.legend(handles=proxy_lines, loc="best")

	# Colorbar encodes series order (chronological index)
	sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greys)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax, pad=0.02)
	cbar.set_label("EPH (orden cronológico)")

	max_ticks = max(2, int(max_cbar_ticks))
	n_ticks = min(n_series, max_ticks)
	if n_series == 1:
		tick_idx = np.array([0], dtype=int)
	else:
		tick_idx = np.unique(
			np.round(np.linspace(0, n_series - 1, n_ticks)).astype(int)
		)
	cbar.set_ticks(tick_idx)
	cbar.set_ticklabels([series_labels[i] for i in tick_idx])

	ax.set_title(title)
	ax.set_xlabel("Alfa")
	ax.set_ylim(0, 1.05)
	if logscale:
		ax.set_xlim(float(np.min(alphas)), 1.0)
		ax.set_xscale("log")
	else:
		ax.set_xlim(0, 1)

	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def compute_and_plot_edge_correlation(
	G: nx.Graph,
	feature_map: dict,
	color_map: dict,
	title: str,
	output_path: Path,
	community_map: dict = None,
	highlight_communities: Iterable[int] = None,
	node_size_map: dict = None,
	factor_node_size: float = 1.0,
	node_size_exponent: float = 1.0,
	save: bool = True,
	perfect_line: bool = True,
	figsize: tuple | None = None,
) -> None:
	"""Scatter of node feature vs weighted average-neighbor feature.

	This function also prints and embeds the Pearson correlation (assortativity)
	computed on the plotted points.
	"""
	valid_nodes = {node for node, val in feature_map.items() if np.isfinite(val)}

	x_vals_map = {}
	y_vals_map = {u: [] for u in valid_nodes}
	edge_weights = {u: [] for u in valid_nodes}

	for u, v, data in G.edges(data=True):
		if u in valid_nodes and v in valid_nodes:
			# Undirected, we add both (u,v) and (v,u) to make it symmetric
			w = data.get("weight", 0.0)

			if np.isfinite(feature_map[u]) and np.isfinite(feature_map[v]):
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
		# Safely handle zero-sum weights: fall back to unweighted mean
		try:
			weights_arr = np.asarray(edge_weights[u], dtype=float)
			if weights_arr.size == 0:
				# No neighbor values (should be caught by earlier check), but guard anyway
				y_vals.append(np.mean(y_vals_map[u]))
			elif np.isclose(weights_arr.sum(), 0.0):
				# All weights sum to zero -> cannot normalize, use unweighted mean
				y_vals.append(np.mean(y_vals_map[u]))
			else:
				y_vals.append(np.average(y_vals_map[u], weights=weights_arr))
		except Exception:
			# As a last resort, use unweighted mean to avoid crashing the pipeline
			y_vals.append(np.mean(y_vals_map[u]))
		plotted_nodes.append(u)

	if len(x_vals) < 2:
		print(
			f"Advertencia: no hay suficientes puntos validos para calcular la correlacion de {title}."
		)
		return

	highlight_set = set(highlight_communities) if highlight_communities else None
	raw_node_sizes = node_size_map or {}

	def _node_color(node_id: int) -> str:
		if highlight_set and community_map is not None:
			community = community_map.get(node_id)
			if community in highlight_set:
				return color_map.get(node_id, LIGTHGRAY)
			return LIGTHGRAY  # light gray for non-highlighted nodes
		return color_map.get(node_id, LIGTHGRAY)

	# Plot
	plt.figure(figsize=figsize)
	raw_node_sizes_plot = [
		float(
			np.power(max(raw_node_sizes.get(u, 1.0), 0.0), node_size_exponent)
			* factor_node_size
		)
		for u in plotted_nodes
	]
	node_colors = [_node_color(u) for u in plotted_nodes]

	# Keep x/y/size/color arrays aligned after removing non-finite values.
	filtered_points = [
		(x, y, size, color)
		for x, y, size, color in zip(x_vals, y_vals, raw_node_sizes_plot, node_colors)
		if np.isfinite(x) and np.isfinite(y)
	]
	if len(filtered_points) < 2:
		print(
			f"Advertencia: no hay suficientes puntos finitos para calcular la correlacion de {title}."
		)
		return

	x_vals, y_vals, node_sizes, node_colors = map(list, zip(*filtered_points))

	# Scatter plot of node feature vs average neighbor feature, colored by community
	plt.scatter(
		x=x_vals,
		y=y_vals,
		s=node_sizes,
		alpha=0.7,
		c=node_colors,
		edgecolor="white",
		linewidth=0.5,
	)
	axis_min = float(min(min(x_vals), min(y_vals)))
	axis_max = float(max(max(x_vals), max(y_vals)))
	axis_range = axis_max - axis_min
	axis_padding = 0.05 * axis_range if axis_range > 0 else 1.0
	axis_min -= axis_padding
	axis_max += axis_padding

	axis_min = min(axis_min, -3)  # Ensure we have a negative range
	axis_max = max(axis_max, 103)  # Ensure we have a positive range

	if perfect_line:
		plt.plot(
			[axis_min, axis_max],
			[axis_min, axis_max],
			"k--",
			label="y=x (Asortatividad perfecta)",
			alpha=0.5,
		)
	# Add a legend for the communities
	if community_map is not None:
		for node_id, node_community in community_map.items():
			if node_community in sorted(highlight_set):
				plt.scatter(
					[],
					[],
					color=color_map.get(node_id, LIGTHGRAY),
					label=node_community,
				)
				highlight_set.remove(node_community)  # Avoid duplicate legend entries

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
	)
	plt.xlabel("X_i")
	plt.ylabel("Y_i")
	# Set y-limits to be the same as x-limits to maintain a square aspect ratio
	print(
		f"Axis limits: x=({axis_min:.2f}, {axis_max:.2f}), y=({axis_min:.2f}, {axis_max:.2f})"
	)
	plt.ylim(plt.xlim())
	sort_labels()

	# Generate Assortativity Scatter Plot (Node value vs Average Neighbor Value)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def compute_edge_assortativity_pearson(
	G: nx.Graph,
	feature_map: Mapping,
	*,
	weight_attr: str = "weight",
) -> tuple[float, float, int]:
	"""Return Pearson assortativity between $x_i$ and weighted neighbor mean $y_i$.

	Definition (matches compute_and_plot_edge_correlation):
	- $x_i$ is the node feature value.
	- $\bar y_i$ is the weighted average of neighbor feature values, where weights
	  are taken from edge attribute `weight_attr`.

	Returns (pearson_r, p_value, n_points). If fewer than 2 valid points exist,
	returns (nan, nan, n_points).
	"""
	# Only keep nodes that have finite feature values
	valid_nodes = {node for node, val in feature_map.items() if np.isfinite(val)}

	x_vals_map: dict = {}
	y_vals_map: dict = {u: [] for u in valid_nodes}
	edge_weights: dict = {u: [] for u in valid_nodes}

	for u, v, data in G.edges(data=True):
		if u in valid_nodes and v in valid_nodes:
			w = data.get(weight_attr, 0.0)
			if np.isfinite(feature_map[u]) and np.isfinite(feature_map[v]):
				x_vals_map[u] = feature_map[u]
				y_vals_map[u].append(feature_map[v])
				edge_weights[u].append(w)

				x_vals_map[v] = feature_map[v]
				y_vals_map[v].append(feature_map[u])
				edge_weights[v].append(w)

	x_vals: list[float] = []
	y_vals: list[float] = []
	for u in x_vals_map:
		if not y_vals_map[u]:
			continue
		x = float(x_vals_map[u])
		y = float(np.average(y_vals_map[u], weights=edge_weights[u]))
		if np.isfinite(x) and np.isfinite(y):
			x_vals.append(x)
			y_vals.append(y)

	n_points = int(len(x_vals))
	if n_points < 2:
		return (float("nan"), float("nan"), n_points)

	try:
		pearson_r, p_value = stats.pearsonr(x_vals, y_vals)
		return (float(pearson_r), float(p_value), n_points)
	except Exception:
		# scipy can fail on constant inputs or other numerical edge cases
		return (float("nan"), float("nan"), n_points)


def plot_community_boxplots(
	df_nodes: pd.DataFrame,
	metrics_dict: dict,
	class_: str,
	algorithm: str,
	output_path: str,
):
	"""Create a stacked horizontal boxplot for community metrics."""
	plot_cols = {
		col: title for col, title in metrics_dict.items() if col in df_nodes.columns
	}
	if not plot_cols:
		return

	n_cols = len(plot_cols)
	fig, axes = plt.subplots(n_cols, 1, figsize=(10, 4 * n_cols), sharey=True)
	if n_cols == 1:
		axes = [axes]

	for ax, (col, title) in zip(axes, plot_cols.items()):
		sns.boxplot(
			data=df_nodes,
			x=col,
			y=df_nodes["community"].astype(str),
			orient="h",
			ax=ax,
			hue=df_nodes["community"].astype(str),
			palette="dark:steelblue",
			legend=False,
			order=sorted(df_nodes["community"].unique(), reverse=True),
		)
		ax.set_title(
			f"{class_.upper()} - {title} Distribution by Community ({algorithm})"
		)
		ax.set_xlabel(title)
		ax.set_ylabel("Community")

	plt.tight_layout()
	plt.savefig(output_path, bbox_inches="tight")
	plt.close()


def plot_correlation_matrix(
	df, features, calib_col=None, title="Correlation Matrix", output_path=None
):
	cols = features.copy()
	if calib_col and calib_col in df.columns:
		cols.append(calib_col)

	# Keep only columns that exist
	cols = [c for c in cols if c in df.columns]
	corr = df[cols].corr()

	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, fmt=".2f")
	ax.set_title(title)

	if output_path:
		plt.tight_layout()
		plt.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def plot_weighted_histograms(
	df, features, calib_col=None, title="Feature Distributions", output_path=None
):
	cols = [c for c in features if c in df.columns]
	n_cols = len(cols)
	if n_cols == 0:
		return

	fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
	if n_cols == 1:
		axes = [axes]

	for ax, col in zip(axes, cols):
		if calib_col and calib_col in df.columns and df[calib_col].notna().any():
			weights = df[calib_col]
		else:
			weights = None

		# Drop NaNs for the histogram
		valid = df[[col]].dropna()
		if weights is not None:
			valid_weights = weights.loc[valid.index]
		else:
			valid_weights = None

		ax.hist(
			valid[col],
			bins=30,
			weights=valid_weights,
			alpha=0.7,
			color="steelblue",
			edgecolor="black",
		)
		ax.set_title(f"Distribution of {col}")
		ax.set_xlabel(col)
		if weights is not None:
			ax.set_ylabel("Weighted Count")
		else:
			ax.set_ylabel("Count")

	fig.suptitle(title)
	if output_path:
		plt.tight_layout()
		plt.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
