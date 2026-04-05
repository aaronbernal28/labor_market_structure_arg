"""
Data loading and cleaning utilities for the ENES occupational network analysis.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import src.utils as ut
import pandas as pd
import numpy as np
from scipy import stats

from src.plotting import (
	color_map_caes,
	color_map_ciuo,
	color_letra_map_caes,
	color_1digit_map_ciuo,
	color_agrupation_map_caes,
	color_ciuo3cat_map_ciuo,
)


class Feature:
	"""Class to hold feature column names for the ENES dataset."""

	def __init__(self, name: str, type: str, default: any):
		self.name = name
		self.type = type
		self.default = default


def clean_enes_base_features(df: pd.DataFrame, features: List[Feature]) -> pd.DataFrame:
	"""Clean and standardize feature columns in the ENES dataframe."""
	for feature in features:
		if feature.name in df.columns:
			df[feature.name] = pd.to_numeric(df[feature.name], errors="coerce").fillna(
				feature.default
			)
		else:
			df[feature.name] = feature.default
		if feature.default is not None:
			df[feature.name] = df[feature.name].astype(feature.type, errors="ignore")
	return df


def lcd_load_enes_base(
	df_enes: pd.DataFrame,
	id_1: str,
	id_2: str,
	id_caes: str,
	id_ciuo: str,
	features: List[Feature],
	max_caes_id: int = 10000,
) -> pd.DataFrame:

	# Copy after filtering to avoid chained-assignment warnings in pandas.
	df_enes = df_enes.dropna(subset=[id_ciuo, id_caes]).copy()
	df_enes.loc[:, id_caes] = df_enes[id_caes].astype(int)
	df_enes.loc[:, id_ciuo] = df_enes[id_ciuo].astype(int)
	df_enes = df_enes[df_enes[id_caes] < 9999]  # Invalid nodes
	df_enes = df_enes[df_enes[id_ciuo] < 9999]

	df_enes.loc[:, id_caes] = df_enes[id_caes].apply(
		lambda x: ut.desambiated_caes_id(x)
	)
	df_enes.loc[:, id_ciuo] = df_enes[id_ciuo].apply(
		lambda x: ut.desambiated_ciuo_id(x, max_caes_id)
	)

	return df_enes


def load_enes_base(
	enes_path: Path, caes_id: str, ciuo_id: str, max_caes_id: int
) -> pd.DataFrame:
	"""Load and clean the base ENES data (person-level responses)."""
	try:
		# Try with semicolon first (original format)
		enes_df = pd.read_csv(enes_path, sep=";")
		if caes_id not in enes_df.columns:
			# If fails, try with comma (2021 format)
			enes_df = pd.read_csv(enes_path, sep=",")
	except Exception:
		enes_df = pd.read_csv(enes_path, sep=",")

	enes_df = enes_df.dropna(subset=[ciuo_id, caes_id])
	enes_df[caes_id] = enes_df[caes_id].astype(int)
	enes_df[ciuo_id] = enes_df[ciuo_id].astype(int)

	enes_df[caes_id] = enes_df[caes_id].apply(lambda x: ut.desambiated_caes_id(x))
	enes_df[ciuo_id] = enes_df[ciuo_id].apply(
		lambda x: ut.desambiated_ciuo_id(x, max_caes_id)
	)
	return enes_df


def load_nodelist_caes(
	caes_path: Path,
	caes_id: str,
	caes_letra_col: str,
	caes_ag_col: str,
	caes_label_color_col: str,
	caes_letra_color_col: str,
	caes_ag_color_col: str,
) -> pd.DataFrame:
	"""Load CAES node metadata and normalize labels."""
	caes_df = pd.read_csv(caes_path)
	caes_df[caes_id] = caes_df[caes_id].astype(int)
	caes_df[caes_id] = caes_df[caes_id].apply(lambda x: ut.desambiated_caes_id(x))
	caes_df = caes_df.set_index(caes_id)
	caes_df[caes_letra_col] = caes_df[caes_letra_col].apply(lambda x: x.split(";")[0])

	color_map = color_map_caes(caes_df.index.to_list())
	caes_df[caes_label_color_col] = caes_df.index.map(color_map)

	color_map = color_letra_map_caes(
		caes_df, letra_col=caes_letra_col, base_color_col=caes_label_color_col
	)
	caes_df[caes_letra_color_col] = caes_df[caes_letra_col].map(color_map)

	color_map = color_agrupation_map_caes(
		caes_df, ag_col=caes_ag_col, base_color_col=caes_letra_color_col
	)
	caes_df[caes_ag_color_col] = caes_df[caes_ag_col].map(color_map)
	return caes_df


def load_nodelist_ciuo(
	ciuo_path: Path,
	ciuo_id: str,
	max_caes_id: int,
	ciuo_letra_col: str,
	ciuo_3cat_col: str,
	ciuo_label_color_col: str,
	ciuo_letra_color_col: str,
	ciuo_3cat_color_col: str,
) -> pd.DataFrame:
	"""Load CIUO node metadata and normalize labels."""
	ciuo_df = pd.read_csv(ciuo_path)
	ciuo_df[ciuo_id] = ciuo_df[ciuo_id].astype(int)
	ciuo_df[ciuo_id] = ciuo_df[ciuo_id].apply(
		lambda x: ut.desambiated_ciuo_id(x, max_caes_id)
	)
	ciuo_df = ciuo_df.set_index(ciuo_id)

	color_map = color_map_ciuo(ciuo_df.index.to_list(), max_caes_id=max_caes_id)
	ciuo_df[ciuo_label_color_col] = ciuo_df.index.map(color_map)

	color_map = color_1digit_map_ciuo(
		ciuo_df, letra_col=ciuo_letra_col, base_color_col=ciuo_label_color_col
	)
	ciuo_df[ciuo_letra_color_col] = ciuo_df[ciuo_letra_col].map(color_map)

	color_map = color_ciuo3cat_map_ciuo(
		ciuo_df, cat_col=ciuo_3cat_col, base_color_col=ciuo_letra_color_col
	)
	ciuo_df[ciuo_3cat_color_col] = ciuo_df[ciuo_3cat_col].map(color_map)
	return ciuo_df


def merge_enes_with_metadata(
	enes_df: pd.DataFrame,
	caes_df: pd.DataFrame,
	ciuo_df: pd.DataFrame,
	caes_id: str,
	ciuo_id: str,
) -> pd.DataFrame:
	"""Attach CAES and CIUO labels to the ENES responses."""
	missing_caes = sorted(set(enes_df[caes_id].unique()) - set(caes_df.index))
	missing_ciuo = sorted(set(enes_df[ciuo_id].unique()) - set(ciuo_df.index))
	if missing_caes or missing_ciuo:
		dropped_mask = enes_df[caes_id].isin(missing_caes) | enes_df[ciuo_id].isin(
			missing_ciuo
		)
		dropped_rows = int(dropped_mask.sum())
		if missing_caes:
			preview = ", ".join(str(code) for code in missing_caes[:20])
			print(
				"Warning: CAES codes missing from node list. "
				f"Missing count={len(missing_caes)}, preview=[{preview}]"
			)
		if missing_ciuo:
			preview = ", ".join(str(code) for code in missing_ciuo[:20])
			print(
				"Warning: CIUO codes missing from node list. "
				f"Missing count={len(missing_ciuo)}, preview=[{preview}]"
			)
		print(f"Warning: Rows dropped by metadata merge. Dropped rows={dropped_rows}")
	merged = enes_df.merge(caes_df, left_on=caes_id, right_index=True, how="inner")
	merged = merged.merge(ciuo_df, left_on=ciuo_id, right_index=True, how="inner")
	return merged


def get_dataset(data_config: dict) -> pd.DataFrame:
	"""Extract the dataset using the provided configuration."""

	def _read_csv_auto(path_or_url):
		# Infer delimiter to support both ';' and ',' ENES sources.
		return pd.read_csv(path_or_url, sep=None, engine="python")

	if data_config["source"] is not None:
		return _read_csv_auto(data_config["source"])
	elif data_config["url"] is not None:
		return _read_csv_auto(data_config["url"])
	else:
		raise ValueError("Data configuration must include either 'source' or 'url'.")


def load_dataset(
	enes_config: dict,
	caes_config: dict,
	ciuo_config: dict,
	extra_enes_config: list[dict] = None,
) -> Dict[str, pd.DataFrame]:
	"""
	Load ENES, CAES, and CIUO datasets from config dictionaries.
	Merges with metadata and optionally appends extra datasets using column renaming.
	"""
	# Extract column names from base ENES config
	caes_id = enes_config["col_caes_id"]
	ciuo_id = enes_config["col_ciuo_id"]
	sex_col = enes_config["col_sex_id"]
	public_col = enes_config["col_public_worker"]
	income_col = enes_config["col_total_income"]

	# Extract CAES column names
	caes_letra = caes_config["col_letra"]
	caes_ag = caes_config["col_ag"]
	caes_label_color = caes_config["col_label_color"]
	caes_letra_color = caes_config["col_letra_color"]
	caes_ag_color = caes_config["col_ag_color"]

	# Extract CIUO column names
	ciuo_letra = ciuo_config["col_letra"]
	ciuo_3cat = ciuo_config["col_3cat"]
	ciuo_label_color = ciuo_config["col_label_color"]
	ciuo_letra_color = ciuo_config["col_letra_color"]
	ciuo_3cat_color = ciuo_config["col_3cat_color"]

	max_caes_id = 10000

	# Load base ENES dataset
	enes_df = get_dataset(enes_config)

	# Append extra datasets (e.g., 2021 survey) by renaming columns
	if extra_enes_config:
		for extra_enes_data in extra_enes_config:
			if (
				extra_enes_data.get("source") is None
				and extra_enes_data.get("url") is None
			):
				continue

			try:
				extra_df = get_dataset(extra_enes_data)
			except Exception as e:
				print(
					f"Failed to load extra ENES dataset from {extra_enes_data.get('source') or extra_enes_data.get('url')}: {e}"
				)
				continue

			# Build column rename mapping
			rename_mapping = {
				extra_enes_data["col_caes_id"]: caes_id,
				extra_enes_data["col_ciuo_id"]: ciuo_id,
			}

			# Add optional columns if they exist
			for extra_col_key in [
				"col_sex_id",
				"col_public_worker",
				"col_total_income",
			]:
				extra_col = extra_enes_data.get(extra_col_key)
				if extra_col and extra_col in extra_df.columns:
					rename_mapping[extra_col] = enes_config[extra_col_key]
				else:
					print(
						f"Warning: Extra ENES dataset is missing expected column '{extra_enes_data.get(extra_col_key)}' for '{extra_col_key}'."
					)

			extra_df = extra_df.rename(columns=rename_mapping)
			extra_df["encuesta"] = extra_enes_data.get("year", "extra")
			enes_df = pd.concat([enes_df, extra_df], ignore_index=True)

	# Process ENES IDs: drop missing, convert to int, and apply disambiguation
	enes_df = enes_df.dropna(subset=[ciuo_id, caes_id])
	enes_df[caes_id] = enes_df[caes_id].astype(int)
	enes_df[ciuo_id] = enes_df[ciuo_id].astype(int)
	enes_df[caes_id] = enes_df[caes_id].apply(lambda x: ut.desambiated_caes_id(x))
	enes_df[ciuo_id] = enes_df[ciuo_id].apply(
		lambda x: ut.desambiated_ciuo_id(x, max_caes_id)
	)

	# Load and process node lists
	caes_df = load_nodelist_caes(
		caes_config["source"],
		caes_id,
		caes_letra_col=caes_letra,
		caes_ag_col=caes_ag,
		caes_label_color_col=caes_label_color,
		caes_letra_color_col=caes_letra_color,
		caes_ag_color_col=caes_ag_color,
	)

	ciuo_df = load_nodelist_ciuo(
		ciuo_config["source"],
		ciuo_id,
		max_caes_id=max_caes_id,
		ciuo_letra_col=ciuo_letra,
		ciuo_3cat_col=ciuo_3cat,
		ciuo_label_color_col=ciuo_label_color,
		ciuo_letra_color_col=ciuo_letra_color,
		ciuo_3cat_color_col=ciuo_3cat_color,
	)

	# Merge ENES with node metadata
	enes = merge_enes_with_metadata(enes_df, caes_df, ciuo_df, caes_id, ciuo_id)

	if enes.empty:
		raise ValueError(
			"Merged ENES dataset is empty after joining with CAES and CIUO metadata."
		)

	return {"enes": enes, "caes_nodes": caes_df, "ciuo_nodes": ciuo_df}


def export_processed(enes_df: pd.DataFrame, processed_path: Path, name: str) -> Path:
	"""
	Persist the merged dataset to CSV for reuse by scripts.
	"""
	processed_path.mkdir(parents=True, exist_ok=True)
	enes_df.to_csv(processed_path / f"{name}.csv", index=True)
	return processed_path / f"{name}.csv"


def insert_positions(
	nodelist_df: pd.DataFrame,
	positions: dict[int | str, list[float]],
	id_col: str | None = None,
) -> pd.DataFrame:
	"""
	Insert precomputed positions into the node list dataframe for consistent plotting.
	"""
	pos_df = pd.DataFrame.from_dict(positions, orient="index", columns=["x", "y"])

	result = nodelist_df.drop(columns=["x", "y"], errors="ignore")
	if id_col:
		pos_df.index.name = id_col
		if id_col in result.columns:
			pos_df.index = pos_df.index.astype(result[id_col].dtype)
		return result.merge(pos_df, on=id_col, how="left")

	return result.join(pos_df[["x", "y"]], how="left")


def load_positions(
	nodelist_df: pd.DataFrame, id_col: str | None = None
) -> dict[int | str, Tuple[float, float]]:
	"""Extract positions from a node list dataframe."""
	if "x" not in nodelist_df.columns or "y" not in nodelist_df.columns:
		return {}

	if id_col:
		if id_col not in nodelist_df.columns:
			raise KeyError(f"ID column '{id_col}' not found in node list dataframe.")
		valid_positions = nodelist_df.dropna(subset=["x", "y"]).set_index(id_col)
		return {
			idx: (float(row["x"]), float(row["y"]))
			for idx, row in valid_positions.iterrows()
		}

	valid_positions = nodelist_df[["x", "y"]].dropna(subset=["x", "y"])
	return {
		idx: (float(row["x"]), float(row["y"]))
		for idx, row in valid_positions.iterrows()
	}


def wald_test_comparison_proportions(
	df1, df2, caes_col, ciuo_col, rownames, colnames, alpha=0.05
):
	"""
	Performs Wald test on proportions comparing two datasets with Bonferroni correction.

	Args:
		df1: DataFrame for dataset 1
		df2: DataFrame for dataset 2
		caes_col: Column name for CAES (industry)
		ciuo_col: Column name for CIUO (occupation)
		rownames: Row labels (CAES categories)
		colnames: Column labels (CIUO categories)
		alpha: Significance level (default 0.05)

	Returns:
		dict containing test results
	"""
	n1 = len(df1)
	n2 = len(df2)

	print(f"Sample sizes: n1={n1}, n2={n2}")

	# Create contingency tables (counts for each cell)
	# First, create a crosstab for each dataset
	crosstab1 = pd.crosstab(df1[caes_col], df1[ciuo_col])
	crosstab2 = pd.crosstab(df2[caes_col], df2[ciuo_col])

	# Ensure both have the same index and columns (fill missing with 0)
	all_caes = sorted(set(crosstab1.index) | set(crosstab2.index))
	all_ciuo = sorted(set(crosstab1.columns) | set(crosstab2.columns))

	crosstab1 = crosstab1.reindex(index=all_caes, columns=all_ciuo, fill_value=0)
	crosstab2 = crosstab2.reindex(index=all_caes, columns=all_ciuo, fill_value=0)

	# Filter to match the rownames and colnames provided
	crosstab1 = crosstab1.reindex(index=rownames, columns=colnames, fill_value=0)
	crosstab2 = crosstab2.reindex(index=rownames, columns=colnames, fill_value=0)

	# Convert to numpy arrays
	counts1 = crosstab1.to_numpy().astype(float)
	counts2 = crosstab2.to_numpy().astype(float)

	# Calculate proportions
	p1 = counts1 / n1
	p2 = counts2 / n2

	# Estimate delta: delta_hat = p1_hat - p2_hat
	delta_hat = p1 - p2

	# Calculate variances (binomial variance for proportions)
	# var(p_hat) = p(1-p) / n
	var1 = p1 * (1 - p1) / n1
	var2 = p2 * (1 - p2) / n2

	# Standard error: SE(delta_hat) = sqrt(var1 + var2)
	se = np.sqrt(var1 + var2)

	# Avoid division by zero
	se = np.where(se > 0, se, 1e-10)

	# Wald statistic: W = delta_hat / SE(delta_hat)
	W = delta_hat / se

	# P-values (two-tailed test using normal approximation)
	p_values = 2 * (1 - stats.norm.cdf(np.abs(W)))

	# Bonferroni correction
	d = delta_hat.size  # Total number of tests
	bonferroni_threshold = alpha / d

	# Reject H_0 if p_i < alpha/d
	rejected = p_values < bonferroni_threshold

	print("\n=== Wald Test Results (Proportions) ===")
	print(f"Total number of tests (d): {d}")
	print(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
	print(
		f"Number of rejections: {np.sum(rejected)} ({100 * np.sum(rejected) / d:.2f}%)"
	)
	print(f"Mean |delta_hat|: {np.mean(np.abs(delta_hat)):.6f}")
	print(f"Max |delta_hat|: {np.max(np.abs(delta_hat)):.6f}")
	print(f"Mean p-value: {np.mean(p_values):.4f}")
	print(f"Min p-value: {np.min(p_values):.2e}")

	return {
		"delta_hat": delta_hat,
		"se": se,
		"W": W,
		"p_values": p_values,
		"bonferroni_threshold": bonferroni_threshold,
		"rejected": rejected,
		"n1": n1,
		"n2": n2,
		"p1": p1,
		"p2": p2,
		"counts1": counts1,
		"counts2": counts2,
	}


def bootstrap_se(df1, df2, caes_col, ciuo_col, rownames, colnames, B=1000, seed=28):
	"""Estimate SE of delta_hat = p1 - p2 via bootstrap (B resamples)."""
	deltas = np.zeros((B, len(rownames), len(colnames)))
	for b in range(B):
		s1 = df1.sample(n=len(df1), replace=True, random_state=seed + 2 * b)
		s2 = df2.sample(n=len(df2), replace=True, random_state=seed + 2 * b + 1)
		ct1 = (
			pd.crosstab(s1[caes_col], s1[ciuo_col])
			.reindex(index=rownames, columns=colnames, fill_value=0)
			.to_numpy(dtype=float)
		)
		ct2 = (
			pd.crosstab(s2[caes_col], s2[ciuo_col])
			.reindex(index=rownames, columns=colnames, fill_value=0)
			.to_numpy(dtype=float)
		)
		deltas[b] = ct1 / len(df1) - ct2 / len(df2)
	return deltas.std(axis=0, ddof=1)
