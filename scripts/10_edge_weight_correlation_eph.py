from pathlib import Path
from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	class_ = snakemake.wildcards["class_"]
	weight_function = snakemake.wildcards["weight_function"]
	feature = snakemake.wildcards.get("feature", "female_pct")
	translation = snakemake.config.get("translation", {})

	def _t(label: str) -> str:
		return utils.translate_label(label, translation)

	# utils.setup_networkx_backend(algorithm=None)

	if class_ not in {"caes", "cno"}:
		raise ValueError("This EPH-only script supports class_ in {'caes', 'cno'}.")

	id_col = snakemake.config[class_]["id"]

	projection_paths = [Path(p) for p in snakemake.input["projections"]]
	processed_paths = [Path(p) for p in snakemake.input["processed"]]

	# Map eph_file -> paths
	eph_to_projection = {}
	for p in projection_paths:
		eph_file = utils.extract_eph_file_from_path(p)
		eph_to_projection.setdefault(eph_file, p)

	eph_to_processed = {}
	for p in processed_paths:
		eph_to_processed.setdefault(p.stem, p)

	eph_files = sorted(set(eph_to_projection.keys()) & set(eph_to_processed.keys()))
	missing_processed = sorted(
		set(eph_to_projection.keys()) - set(eph_to_processed.keys())
	)
	missing_projection = sorted(
		set(eph_to_processed.keys()) - set(eph_to_projection.keys())
	)

	if not eph_files:
		raise ValueError(
			"No overlapping EPH series between projection inputs and processed CSV inputs."
		)

	eph_files_sorted = utils.sort_eph_files(eph_files)

	# Keep only third-quarter (Q3) waves (period == 3)
	_filtered = []
	for _f in eph_files_sorted:
		_k = utils.parse_eph_file_key(_f)
		if _k is not None and getattr(_k, "period", None) == 3:
			_filtered.append(_f)
	eph_files_sorted = _filtered
	if not eph_files_sorted:
		raise ValueError(
			"No Q3 waves found after filtering. Check EPH filenames or adjust filter."
		)

	wave_time_meta: dict[str, tuple[float, pd.Timestamp, str]] = {}
	fallback_idx = 0
	for eph_file in eph_files_sorted:
		key = utils.parse_eph_file_key(eph_file)
		if key is None:
			# Fallback: keep wave-level ordering for unparsed files.
			year = 2000 + fallback_idx // 4
			period = 1 + (fallback_idx % 4)
			time_date = pd.Timestamp(utils.eph_quarter_start_date(year, period))
			time_num = float(year) + float(period - 1) / 4.0
			lbl = eph_file
			fallback_idx += 1
		else:
			time_num = key.time_numeric
			lbl = key.label
			time_date = pd.Timestamp(key.time_date)

		wave_time_meta[eph_file] = (time_num, time_date, lbl)

	alphas = [1.0, 0.1, 0.03, 1e-2, 1e-3]
	x_dates = []

	rows = []
	projection_metrics = {}

	for eph_file in eph_files_sorted:
		time_num, time_date, lbl = wave_time_meta[eph_file]

		x_dates.append(time_date)

		projection_path = eph_to_projection[eph_file]
		processed_path = eph_to_processed[eph_file]

		projection = nx.read_gexf(projection_path, node_type=int)
		# projection_metrics[eph_file] = metrics.summarize_graph(projection)

		df_wave = pd.read_csv(processed_path)
		if id_col not in df_wave.columns:
			raise KeyError(f"Missing group id column '{id_col}' in {processed_path}.")

		# Use ponderation weights for computing group characteristics per wave
		features_df = nc.compute_group_characteristics(
			df_wave, col_group=id_col, calib_col="ponderation"
		)
		if feature not in features_df.columns:
			raise KeyError(
				f"Feature '{feature}' not found in computed characteristics for wave {eph_file}. "
				f"Available columns include: {', '.join(features_df.columns[:10])}..."
			)

		feature_map = features_df[feature].to_dict()

		# Compute assortativity (unfiltered)
		r_unf, p_unf, n_unf = pl.compute_edge_assortativity_pearson(
			projection, feature_map
		)
		rows.append(
			{
				"eph_file": eph_file,
				"time_numeric": time_num,
				"time_date": time_date,
				"time_label": lbl,
				"alpha": _t("unfiltered"),
				"pearson_r": r_unf,
				"p_value": p_unf,
				"n_points": n_unf,
				"n_nodes": projection.number_of_nodes(),
				"n_edges": projection.number_of_edges(),
			}
		)

		# Disparity once per projection; filter for each alpha
		disp = gc.get_disparity_graph(projection)
		for a in alphas:
			bb = gc.disparity_filter_backbone(
				disparity_graph=disp, alpha=float(a), keep_isolates=True
			)
			r, p, n = pl.compute_edge_assortativity_pearson(bb, feature_map)
			rows.append(
				{
					"eph_file": eph_file,
					"time_numeric": time_num,
					"time_date": time_date,
					"time_label": lbl,
					"alpha": float(a),
					"pearson_r": r,
					"p_value": p,
					"n_points": n,
					"n_nodes": bb.number_of_nodes(),
					"n_edges": bb.number_of_edges(),
				}
			)

	results = pd.DataFrame(rows)
	results["time_date"] = pd.to_datetime(results["time_date"], errors="coerce")
	results = results.sort_values(["time_date", "alpha"], kind="mergesort")

	# Prepare a tidy summary dataframe with date, assortativity and alpha for downstream work
	_summary_cols = [
		"time_date",
		"pearson_r",
		"alpha",
		"eph_file",
		"time_label",
		"time_numeric",
	]
	_summary_df = results.loc[
		:, [c for c in _summary_cols if c in results.columns]
	].copy()
	_summary_df = _summary_df.sort_values(["time_date", "alpha"], kind="mergesort")

	print("Example of results dataframe:")
	print(_summary_df.head())

	sns.lineplot(
		data=_summary_df,
		x="time_date",
		y="pearson_r",
		hue="alpha",
		palette="viridis",
	)
	ax = plt.gca()
	ax.set_xlabel(_t("time_date"))
	ax.set_ylabel(_t("pearson_r"))
	legend = ax.get_legend()
	if legend is not None:
		legend.set_title(_t("alpha"))

	out_path = Path(snakemake.output[0])
	plt.savefig(out_path, bbox_inches="tight")
	plt.close("all")

	# Logging
	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("EDGE WEIGHT CORRELATION (EPH - TIME SERIES)")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"SETTINGS",
		[
			f"Class: {class_}",
			f"Weight function: {weight_function}",
			f"Feature: {feature}",
			f"Alphas: {', '.join([str(a) for a in alphas])}",
			f"Series count: {len(eph_files_sorted)}",
		],
	)

	if missing_processed:
		log_lines.append("")
		log_lines.append("WARNING: missing processed CSV for these EPH projections:")
		for s in missing_processed:
			log_lines.append(f"  - {s}")
	if missing_projection:
		log_lines.append("")
		log_lines.append("WARNING: missing projection graph for these processed CSVs:")
		for s in missing_projection:
			log_lines.append(f"  - {s}")

	log_lines.append("")
	log_lines.append("EPH FILES (chronological order):")
	for j, eph_file in enumerate(eph_files_sorted):
		k = utils.parse_eph_file_key(eph_file)
		lbl = k.label if k else eph_file
		log_lines.append(f"  {j:>3d}. {eph_file} -> {lbl}")

	log_path = None
	if hasattr(snakemake, "log") and snakemake.log:
		log_path = snakemake.log[0]
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
