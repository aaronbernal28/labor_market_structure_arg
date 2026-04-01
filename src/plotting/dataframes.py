"""
Plotting utilities extracted from the exploratory notebook.
"""

from pathlib import Path
from typing import Dict, Iterable
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.utils as ut

# plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 100})


def fceyn_plot_heatmap(
	biadjacency: pd.DataFrame,
	output_path: Path = None,
	save: bool = True,
	font_size: int = None,
	show: bool | None = None,
) -> None:
	"""Plot heatmap of the bipartite adjacency matrix."""
	plt.figure(figsize=(16, 10))

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
	default_fontsize = font_size if font_size else 9
	ax = sns.heatmap(
		biadjacency,
		annot=True,
		fmt=fmt,
		cmap="Greens",
		cbar=False,
		annot_kws={"fontsize": default_fontsize},
	)

	ax.xaxis.tick_top()  # Move ticks to top
	ax.xaxis.set_label_position("top")
	plt.xticks(rotation=0, fontsize=default_fontsize)

	# Row labels: align LEFT (or RIGHT if font_size specified for single-char labels)
	if font_size:
		ax.set_yticklabels(
			ax.get_yticklabels(), ha="right", rotation=0, fontsize=default_fontsize
		)
		ax.tick_params(axis="y", pad=20)
	else:
		ax.set_yticklabels(ax.get_yticklabels(), ha="left")
		ax.tick_params(axis="y", pad=190)
	ax.tick_params(left=False, top=False)

	plt.xlabel("")
	plt.ylabel("")
	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()


def fceyn_wrap_labels(df: pd.DataFrame) -> pd.DataFrame:
	wrapped = df.copy()
	col_wrap_width = 15
	idx_wrap_width = 35
	wrapped.columns = [
		textwrap.fill(str(c), width=col_wrap_width) for c in wrapped.columns
	]
	wrapped.index = [textwrap.fill(str(i), width=idx_wrap_width) for i in wrapped.index]
	return wrapped


def fceyn_style_heatmap_axes(ax, title: str, font_size: int = None) -> None:
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position("top")
	default_fontsize = font_size if font_size else 9
	title_fontsize = font_size + 2 if font_size else 11
	plt.xticks(rotation=0, fontsize=default_fontsize)
	if font_size:
		ax.set_yticklabels(
			ax.get_yticklabels(), ha="right", rotation=0, fontsize=default_fontsize
		)
		ax.tick_params(axis="y", pad=20)
	else:
		ax.set_yticklabels(ax.get_yticklabels(), ha="left")
		ax.tick_params(axis="y", pad=190)
	ax.tick_params(left=False, top=False)
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.set_title(title, fontsize=title_fontsize, pad=12)
	plt.tight_layout()


def fceyn_plot_rejection_heatmap(
	p_values: np.ndarray,
	rejected: np.ndarray,
	rownames: Iterable[str],
	colnames: Iterable[str],
	bonferroni_threshold: float,
	output_path: Path,
	save: bool = True,
	font_size: int = None,
) -> None:
	"""Heatmap of p-values: black below Bonferroni threshold, YlOrRd above it."""
	from matplotlib.colors import ListedColormap

	# Smart formatting: hide extreme values, automatic scientific notation for others
	def fceyn_format_pvalue(v):
		if v > 0.99:  # Hide extreme high p-values (essentially 1)
			return ""
		else:
			return f"{v:.3g}"

	annot = np.vectorize(fceyn_format_pvalue)(p_values)
	df = pd.DataFrame(p_values, index=rownames, columns=colnames)
	df = fceyn_wrap_labels(df)
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

	default_fontsize = font_size if font_size else 11
	fig, ax = plt.subplots(figsize=(16, 10))
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
		annot_kws={"fontsize": default_fontsize},
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
		fontsize=7,
		transform=cbar.ax.transData,
		color="dimgray",
	)
	fceyn_style_heatmap_axes(ax, title, font_size)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def fceyn_plot_delta_heatmap(
	delta_hat: np.ndarray,
	rownames: Iterable[str],
	colnames: Iterable[str],
	output_path: Path,
	save: bool = True,
	font_size: int = None,
) -> None:
	"""Diverging heatmap showing delta_hat annotations in scientific notation."""

	# Smart formatting: hide extreme values, automatic scientific notation for others
	def fceyn_format_delta(v):
		if abs(v) < 0.0001:  # Hide essentially zero differences
			return ""
		else:
			return f"{v:.3g}"

	annot = np.vectorize(fceyn_format_delta)(delta_hat)
	df = pd.DataFrame(delta_hat, index=rownames, columns=colnames)
	df = fceyn_wrap_labels(df)
	annot_df = pd.DataFrame(annot, index=df.index, columns=df.columns)
	abs_max = np.max(np.abs(delta_hat))
	title = "Diferencia estimada delta = p(ENES 2019) - p(ESAyPP 2021)"
	default_fontsize = font_size if font_size else 11
	fig, ax = plt.subplots(figsize=(16, 10))
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
		annot_kws={"fontsize": default_fontsize},
	)
	fceyn_style_heatmap_axes(ax, title, font_size)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def fceyn_plot_bootstrap_se_heatmap(
	se_boot: np.ndarray,
	rownames: Iterable[str],
	colnames: Iterable[str],
	output_path: Path,
	save: bool = True,
	font_size: int = None,
) -> None:
	"""Heatmap of bootstrap SE estimates (B=1000) for the delta proportions."""

	# Smart formatting: hide extreme values, automatic scientific notation for others
	def fceyn_format_se(v):
		if v < 0.00001:  # Hide essentially zero SE
			return ""
		else:
			return f"{v:.3g}"

	annot = np.vectorize(fceyn_format_se)(se_boot)
	df = pd.DataFrame(se_boot, index=rownames, columns=colnames)
	df = fceyn_wrap_labels(df)
	annot_df = pd.DataFrame(annot, index=df.index, columns=df.columns)
	title = "Bootstrap SE de delta (B=1000)"
	default_fontsize = font_size if font_size else 11
	fig, ax = plt.subplots(figsize=(16, 10))
	sns.heatmap(
		df,
		ax=ax,
		annot=annot_df,
		fmt="",
		cmap="YlOrRd",
		cbar=True,
		cbar_kws={"shrink": 0.35, "label": "SE bootstrap"},
		annot_kws={"fontsize": default_fontsize},
	)
	fceyn_style_heatmap_axes(ax, title, font_size)
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def fceyn_plot_degree_histogram(
	degrees: Iterable,
	color: str,
	title: str,
	output_path: Path = None,
	save: bool = False,
	ax: plt.Axes = None,
	logscale: bool = False,
	is_discrete: bool = True,
	show: bool | None = None,
) -> None:
	"""Plot degree distribution as scatter plot for discrete data or histogram for continuous data."""
	created_fig = ax is None
	if created_fig:
		fig, ax = plt.subplots(figsize=(6, 5))

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

	if created_fig:  # only do tight_layout if we created the figure
		plt.tight_layout()
		if save and output_path is not None:
			plt.savefig(output_path, bbox_inches="tight")
			plt.close()
		elif show is None or show:
			plt.show()


def fceyn_plot_degree_histograms(
	degrees: Dict[str, list],
	colors: Dict[str, str],
	output_path: Path = None,
	save: bool = True,
	logscale: bool = False,
	show: bool | None = None,
) -> None:
	"""Plot degree histograms for all nodes, CAES nodes, and CIUO nodes."""
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	configs = [
		(degrees["all"], colors["all"], "Grados (todos los nodos)"),
		(degrees["caes"], colors["caes"], "Grados CAES"),
		(degrees["ciuo"], colors["ciuo"], "Grados CIUO"),
	]
	for i, (values, color, title) in enumerate(configs):
		fceyn_plot_degree_histogram(values, color, title, ax=axes[i], logscale=logscale)
	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()


def fceyn_plot_stacked_by_group(
	df_index: pd.DataFrame,
	group_col: str,
	community_map: Dict[int, int],
	title: str,
	output_path: Path,
	group_color_map: Dict[str, tuple] = None,
	legend_title: str = None,
	figsize: tuple = (16, 4),
	font_size: int = 11,
	save: bool = True,
	percentage: bool = True,
) -> None:
	"""Plot stacked bar chart showing distribution of groups within communities."""
	df_index_copy = df_index.copy()
	df_index_copy["community"] = df_index_copy.index.map(community_map)
	df_index_copy = df_index_copy.dropna(subset=["community"])
	df_index_copy["community"] = df_index_copy["community"].astype(int)

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
		ax = ct.plot(
			kind="barh", stacked=True, figsize=figsize, width=0.8, color=colors
		)
	else:
		ax = ct.plot(kind="barh", stacked=True, figsize=figsize, width=0.8)

	ax.set_xlabel("Porcentaje (%)" if percentage else "Conteo", fontsize=font_size)
	ax.set_title(title, fontsize=font_size + 1)
	ax.tick_params(axis="both", labelsize=font_size - 1)
	ax.set_xlim(0, 100 if percentage else None)
	legend_title = legend_title or group_col
	ax.legend(
		title=legend_title,
		bbox_to_anchor=(1.05, 1),
		loc="upper left",
		fontsize=font_size - 2,
		title_fontsize=font_size,
	)

	# Format y-axis labels as C0, C1, ...
	yticks = ax.get_yticks()
	ax.set_yticklabels([f"C{int(y)}" for y in yticks])

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


def fceyn_plot_distance_histogram(
	distance_matrix: np.ndarray,
	output_path: Path = None,
	bins: int = 30,
	title: str = "Histograma de distancias",
	include_infinite: bool = True,
	save: bool = True,
) -> None:
	"""Plot histogram of finite distances from a distance matrix."""
	values = np.asarray(distance_matrix, dtype=float).ravel()
	finite = values[np.isfinite(values) & (values > 0)]
	inf_count = np.isinf(values).sum()

	plt.figure(figsize=(8, 5))
	counts, bin_edges, _ = plt.hist(finite, bins=bins, alpha=0.8, color="steelblue")
	if include_infinite and inf_count > 0:
		bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
		inf_x = bin_edges[-1] + bin_width
		plt.bar([inf_x], [inf_count], width=bin_width * 0.8, color="tomato", alpha=0.8)
		plt.xticks(list(plt.xticks()[0]) + [inf_x], list(plt.xticks()[0]) + ["inf"])
	plt.xlabel("Distancia")
	plt.ylabel("Frecuencia")
	plt.title(f"{title}\nfinito={len(finite)} | inf={inf_count}")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def fceyn_plot_distance_heatmap(
	distance_matrix: np.ndarray,
	output_path: Path = None,
	title: str = "Matriz de distancias",
	labels: Iterable[str] = None,
	save: bool = True,
) -> None:
	"""Plot heatmap for a distance matrix (infinite distances are masked)."""
	data = np.asarray(distance_matrix, dtype=float).copy()
	data[np.isinf(data)] = np.nan
	mask = np.isnan(data)

	plt.figure(figsize=(10, 8))
	ax = sns.heatmap(data, cmap="mako", mask=mask, cbar=True)
	if labels:
		ax.set_xticks(np.arange(len(labels)) + 0.5)
		ax.set_yticks(np.arange(len(labels)) + 0.5)
		ax.set_xticklabels(
			[f"{i:02d}" for i in range(len(labels))],
			rotation=45,
			ha="right",
			fontsize=6,
		)
		ax.set_yticklabels(
			[f"{i:02d}" for i in range(len(labels))], rotation=0, fontsize=6
		)
		ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
		ax.set_yticklabels(labels, rotation=0, fontsize=6)
	else:
		ax.set_xticks([])
		ax.set_yticks([])
	plt.title(title)
	plt.tight_layout()
	if save:
		plt.savefig(output_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def fceyn_plot_backbone_weight_histogram(
	original_weights: list,
	backbone_weights: list,
	alpha: float,
	title_prefix: str,
	output_path: Path,
	save: bool = True,
) -> None:
	"""Plot overlapped histograms comparing original and backbone edge weights."""
	fig, ax = plt.subplots(figsize=(10, 6))

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

	ax.set_title(
		f"{title_prefix} Distribucion de pesos de aristas: Original vs Esqueleto (alpha={alpha})"
	)
	ax.set_xlabel("Peso de arista")
	ax.set_ylabel("Frecuencia")
	ax.set_yscale("log")
	ax.set_ylim(bottom=1e-1)
	ax.legend()

	plt.tight_layout()
	if save:
		plt.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


def fceyn_plot_top_n_bar(
	df: pd.DataFrame,
	label_col: str,
	val_col: str,
	color_col: str,
	title: str,
	xlabel: str,
	top_n: int = 15,
	figsize: tuple = (12, 8),
	font_size: int = 11,
	output_path: Path = None,
	save: bool = True,
	show: bool | None = None,
) -> None:
	if val_col not in df.columns or label_col not in df.columns:
		return
	top_df = df.nlargest(top_n, val_col)

	palette_by_label = {}
	for _, row in top_df.iterrows():
		label = row[label_col]
		if color_col in top_df.columns:
			palette_by_label[label] = ut.fceyn_parse_color(row[color_col])
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
	plt.title(title, fontsize=font_size + 1)
	plt.xlabel(xlabel, fontsize=font_size)
	plt.ylabel("", fontsize=font_size)
	ax.tick_params(axis="both", labelsize=font_size - 1)

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
		plt.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close()
	elif show is None or show:
		plt.show()


def fceyn_plot_biadjacency_heatmap(*args, **kwargs):
	"""Placeholder for biadjacency heatmap plotting."""
	return plt.figure()


def fceyn_plot_aed_top_sectors(*args, **kwargs):
	"""Plot top CAES groups by count and return the figure."""
	data = args[0] if args else kwargs.get("data")
	title = kwargs.get("title", "Top sectors")
	top_n = kwargs.get("top_n", 15)
	if data is None:
		return plt.figure()

	caes_col, _ = ut.fceyn_infer_group_columns(data)
	if caes_col is None:
		return plt.figure()

	counts = data[caes_col].value_counts(dropna=True).head(top_n)
	fig, ax = plt.subplots(figsize=(12, 8))
	sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Count")
	ax.set_ylabel("")
	plt.tight_layout()
	return fig


def fceyn_plot_aed_top_occupations(*args, **kwargs):
	"""Plot top CIUO groups by count and return the figure."""
	data = args[0] if args else kwargs.get("data")
	title = kwargs.get("title", "Top occupations")
	top_n = kwargs.get("top_n", 15)
	if data is None:
		return plt.figure()

	_, ciuo_col = ut.fceyn_infer_group_columns(data)
	if ciuo_col is None:
		return plt.figure()

	counts = data[ciuo_col].value_counts(dropna=True).head(top_n)
	fig, ax = plt.subplots(figsize=(12, 8))
	sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Count")
	ax.set_ylabel("")
	plt.tight_layout()
	return fig


def fceyn_plot_sankey(*args, **kwargs):
	"""Placeholder for sankey plot."""
	return plt.figure()
