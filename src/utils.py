import ast
from functools import lru_cache

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@lru_cache(maxsize=1000)
def fceyn_parse_color_from_string(color_str: str):
	parsed = ast.literal_eval(color_str)
	if hasattr(parsed, "__iter__") and not isinstance(parsed, str):
		return tuple(float(x) for x in parsed)
	return parsed


def fceyn_parse_color(color_value):
	"""Parse color value from CSV (can be string tuple or already a tuple)."""
	if isinstance(color_value, str):
		return fceyn_parse_color_from_string(color_value)
	# If it's already a tuple/list/array, ensure it's a regular tuple
	if hasattr(color_value, "__iter__") and not isinstance(color_value, str):
		return tuple(float(x) for x in color_value)
	return color_value


def fceyn_get_class_index(col_name: str) -> int:
	"""Determine if the column corresponds to CAES or CIUO IDs for bipartite graph construction."""
	return int("caes" in col_name.lower())


def fceyn_original_ciuo_id(id: int, max_caes_id: int) -> int:
	"""
	Recover original CIUO ID from disambiguated ID.
	"""
	return id - max_caes_id


def fceyn_desambiated_caes_id(id: int) -> int:
	return id


def fceyn_desambiated_ciuo_id(id: int, max_caes_id: int) -> int:
	"""
	Recover original CIUO ID from disambiguated ID.
	"""
	return id + max_caes_id


def fceyn_find_column(df: pd.DataFrame, hint: str | None = None) -> str | None:
	if df is None or df.empty:
		return None
	if hint is None:
		return df.columns[0] if len(df.columns) else None
	hint = str(hint).lower()
	matches = [col for col in df.columns if hint in str(col).lower()]
	return matches[0] if matches else None


def fceyn_is_finite_number(value) -> bool:
	try:
		return np.isfinite(float(value))
	except (TypeError, ValueError):
		return False


def fceyn_node_attr_map(graph, attr: str) -> dict:
	return {node: data.get(attr) for node, data in graph.nodes(data=True)}


def fceyn_infer_node_attribute(
	graph, *, prefer_numeric: bool | None = None
) -> str | None:
	attr_values = {}
	for _, data in graph.nodes(data=True):
		for key, value in data.items():
			if value is None:
				continue
			attr_values.setdefault(key, []).append(value)

	if not attr_values:
		return None

	if prefer_numeric is not None:
		filtered = {
			key: values
			for key, values in attr_values.items()
			if all(fun_is_finite_number(v) for v in values) == prefer_numeric
		}
		if filtered:
			attr_values = filtered

	return max(attr_values, key=lambda k: len(attr_values[k]))


def fceyn_categorical_palette(values) -> dict:
	unique_vals = [v for v in dict.fromkeys(values) if v is not None]
	if not unique_vals:
		unique_vals = ["All"]

	cycle = plt.rcParams.get("axes.prop_cycle")
	cycle_colors = cycle.by_key().get("color", []) if cycle else []
	if len(cycle_colors) < len(unique_vals):
		cmap = plt.get_cmap("tab20", len(unique_vals))
		cycle_colors = [mcolors.to_hex(cmap(i)) for i in range(len(unique_vals))]

	return {v: cycle_colors[i % len(cycle_colors)] for i, v in enumerate(unique_vals)}


def fceyn_infer_layout_method(layout: str | None) -> dict:
	if not layout:
		return {"method": "auto"}
	key = str(layout).lower().strip()
	if key in {"kamada_kawai", "kamada-kawai"}:
		return {"method": "kamada_kawai"}
	return {"method": "auto"}


def fceyn_metadata_index_values(
	df: pd.DataFrame | None, id_col: str | None
) -> list | None:
	if df is None:
		return None
	if id_col is None:
		return df.index.tolist()
	if id_col in df.columns:
		return df[id_col].astype(int, errors="ignore").tolist()
	return df.index.tolist()


def fceyn_infer_enes_columns(
	enes_df: pd.DataFrame,
	*,
	caes_df: pd.DataFrame | None = None,
	ciuo_df: pd.DataFrame | None = None,
) -> tuple[str, str]:
	def fceyn_match_from_meta(meta_df: pd.DataFrame | None, hint: str) -> str | None:
		if meta_df is not None:
			index_name = meta_df.index.name
			if index_name and index_name in enes_df.columns:
				return index_name
			for col in meta_df.columns:
				if col in enes_df.columns:
					return col
		return fceyn_find_column(enes_df, hint)

	caes_col = fceyn_match_from_meta(caes_df, "caes")
	ciuo_col = fceyn_match_from_meta(ciuo_df, "ciuo")
	if caes_col is None or ciuo_col is None:
		raise KeyError(
			"Unable to infer CAES/CIUO columns from ENES dataframe. "
			"Provide columns or ensure defaults exist."
		)
	return caes_col, ciuo_col


def fceyn_infer_group_column(df: pd.DataFrame, hint: str) -> str | None:
	if df is None or df.empty:
		return None
	hint = str(hint).lower()
	matches = [col for col in df.columns if hint in str(col).lower()]
	if not matches:
		return None
	for col in matches:
		if not pd.api.types.is_numeric_dtype(df[col]):
			return col
	return matches[0]


def fceyn_infer_group_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
	return (fun_infer_group_column(df, "caes"), fceyn_infer_group_column(df, "ciuo"))


def fceyn_ensure_label_column(
	enes_df: pd.DataFrame,
	meta_df: pd.DataFrame | None,
	*,
	id_col: str | None,
	label_col: str,
) -> pd.DataFrame:
	if label_col in enes_df.columns:
		return enes_df
	if meta_df is None or id_col is None or id_col not in enes_df.columns:
		return enes_df
	meta = meta_df.copy()
	if id_col not in meta.columns:
		meta = meta.reset_index()
		if id_col not in meta.columns:
			return enes_df
	cols = [id_col, label_col]
	if label_col not in meta.columns:
		return enes_df
	return enes_df.merge(meta[cols], on=id_col, how="left")
