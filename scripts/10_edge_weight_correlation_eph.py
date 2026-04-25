from pathlib import Path
from typing import Any

from scripts import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from src.seeding import initialize_seeds, get_seed_from_config

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	seed = get_seed_from_config(snakemake.config)
	initialize_seeds(seed)

	class_ = snakemake.wildcards["class_"]
	weight_function = snakemake.wildcards["weight_function"]
	feature = snakemake.wildcards.get("feature", "female_pct")

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

	alphas = [1.0, 0.1, 0.03, 1e-2, 1e-3]
	x_dates = []

	rows = []
	projection_metrics = {}
	fallback_idx = 0

	for eph_file in eph_files_sorted:
		key = utils.parse_eph_file_key(eph_file)
		if key is None:
			# Fallback: create synthetic quarterly slots after a base date.
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

		x_dates.append(time_date)

		projection_path = eph_to_projection[eph_file]
		processed_path = eph_to_processed[eph_file]

		projection = nx.read_gexf(projection_path, node_type=int)
		projection_metrics[eph_file] = metrics.summarize_graph(projection)

		df = pd.read_csv(processed_path)
		if id_col not in df.columns:
			raise KeyError(f"Missing group id column '{id_col}' in {processed_path}.")

		features_df = nc.compute_group_characteristics(df, col_group=id_col)
		if feature not in features_df.columns:
			raise KeyError(
				f"Feature '{feature}' not found in computed characteristics for {processed_path}. "
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
				"alpha": "unfiltered",
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

	# Plot
	figsize = snakemake.config.get("figsizes", {}).get("edge_correlation", (7, 4))
	plt.figure(figsize=figsize)

	def _alpha_label(a_val) -> str:
		if isinstance(a_val, str):
			return a_val
		return f"alpha={a_val:g}"

	for a_val in alphas:
		df_a = results[results["alpha"].astype(str) == str(a_val)].copy()
		if df_a.empty:
			continue
		x = pd.to_datetime(df_a["time_date"]).to_numpy()
		y = df_a["pearson_r"].to_numpy(dtype=float)
		plt.plot(
			x, y, marker="o", linewidth=1.3, markersize=3.5, label=_alpha_label(a_val)
		)

	ax = plt.gca()
	plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
	plt.ylim(bottom=min(0, plt.ylim()[0]), top=1.0)

	# Continuous calendar axis with regular date ticks (not only observed EPH points).
	t_min = pd.to_datetime(min(x_dates)) - pd.Timedelta(days=45)
	t_max = pd.to_datetime(max(x_dates)) + pd.Timedelta(days=45)
	xmin = float(mdates.date2num(t_min.to_pydatetime()))
	xmax = float(mdates.date2num(t_max.to_pydatetime()))
	ax.set_xlim(xmin, xmax)
	major_locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
	ax.xaxis.set_major_locator(major_locator)
	ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major_locator))
	ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
	plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

	title = f"EPH - {class_.upper()} - Asortatividad (Pearson r)\n{weight_function} | feature={feature}"
	plt.title(title)
	plt.xlabel("Fecha (EPH)")
	plt.ylabel("Asortatividad (Pearson r)")
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()

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

	log_lines.append("")
	log_lines.append("PER-FILE PROJECTION METRICS (unfiltered):")
	for eph_file in eph_files_sorted:
		gm = projection_metrics.get(eph_file, {})
		items = []
		for k in ["n_nodes", "n_edges", "density", "avg_degree"]:
			if k in gm:
				items.append(f"{k}={gm[k]}")
		log_lines.append(
			f"  - {eph_file}: " + (", ".join(items) if items else "(no metrics)")
		)

	log_lines.append("")
	log_lines.append("PER-ALPHA SUMMARY (across time):")
	for a_val in alphas:
		df_a = results[results["alpha"].astype(str) == str(a_val)].copy()
		if df_a.empty:
			continue
		r_vals = pd.to_numeric(df_a["pearson_r"], errors="coerce")
		r_vals = r_vals[np.isfinite(r_vals)]
		if len(r_vals) == 0:
			log_lines.append(f"  - {_alpha_label(a_val)}: no finite r values")
			continue
		log_lines.append(
			f"  - {_alpha_label(a_val)}: mean={r_vals.mean():.4f}, std={r_vals.std():.4f}, min={r_vals.min():.4f}, max={r_vals.max():.4f}"
		)

	log_path = None
	if hasattr(snakemake, "log") and snakemake.log:
		log_path = snakemake.log[0]
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
