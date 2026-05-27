import math
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from scripts import *
import src.logging_utils as log

snakemake: Any


def _fmt_float(value: float | None, fmt: str) -> str:
	if value is None or np.isnan(value):
		return "NA"
	return format(value, fmt)


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")

	if snakemake is None:
		raise RuntimeError("This script is designed to be executed via Snakemake.")

	# 1. Determine file paths from Snakemake
	try:
		path_2021 = Path(snakemake.input[0])
		path_base = Path(snakemake.input[1])
		output_plot_path = Path(snakemake.output[0])
	except IndexError:
		raise RuntimeError(
			"Check Snakemake inputs/outputs. Expected 2 inputs, 1 output."
		)

	# 2. Read datasets (low_memory=False to suppress DtypeWarnings)
	try:
		enes_2021 = pd.read_csv(path_2021)
	except Exception as e:
		raise RuntimeError(f"Failed to read ENES 2021 file at {path_2021}: {e}")

	try:
		enes_base = pd.read_csv(path_base, sep=";", low_memory=False)
	except Exception as e:
		raise RuntimeError(f"Failed to read base ENES file at {path_base}: {e}")

	# 3. Hardcoded columns to compare
	hardcoded_ponds = [
		"POND1_FIN",
		"POND1_FIN_n",
		"POND1R_FIN",
		"POND1R_FIN_n",
		"POND2_FIN",
		"POND2_FIN_n",
		"POND2R_FIN",
		"POND2R_FIN_n",
	]
	f_col = "f_calib3"

	# Ensure base column exists (case-insensitive and whitespace-stripped fallback)
	f_col_actual = None
	cols_base_clean = {str(c).lower().strip(): c for c in enes_base.columns}

	if f_col.lower() in cols_base_clean:
		f_col_actual = cols_base_clean[f_col.lower()]

	results = []
	skipped_cols: list[str] = []
	alpha = 0.05
	N_base = len(enes_base)
	N_2021 = len(enes_2021)

	if not f_col_actual:
		print(
			f"Warning: '{f_col}' not found in base ENES file. Headers found: {list(enes_base.columns)[:10]}..."
		)
	else:
		# Extract robust clean array for the base column
		base_vals = pd.to_numeric(enes_base[f_col_actual], errors="coerce").dropna()
		n_base_valid = len(base_vals)

		# 4. Execute Robust Test (Two-Sample Kolmogorov-Smirnov)
		for pc in hardcoded_ponds:
			# Check for column with whitespace stripped
			pc_actual = next(
				(c for c in enes_2021.columns if str(c).strip() == pc), None
			)

			if not pc_actual:
				print(f"Column {pc} missing in 2021 data. Skipping.")
				skipped_cols.append(pc)
				continue

			pc_vals = pd.to_numeric(enes_2021[pc_actual], errors="coerce").dropna()
			n_pc_valid = len(pc_vals)

			if n_pc_valid == 0 or n_base_valid == 0:
				stat, pval = np.nan, np.nan
				same_distribution = False
				different_distribution = False
			else:
				stat, pval = ks_2samp(pc_vals, base_vals)
				same_distribution = bool(pval >= alpha)
				different_distribution = bool(pval < alpha)

			results.append(
				{
					"pond_col": pc_actual,
					"n_nonmissing_2021": n_pc_valid,
					"n_nonmissing_base": n_base_valid,
					"nobs_2021": N_2021,
					"nobs_base": N_base,
					"ks_stat": float(stat) if not np.isnan(stat) else None,
					"pval": float(pval) if not np.isnan(pval) else None,
					"same_distribution": same_distribution,
					"different_distribution": different_distribution,
				}
			)

	res_df = pd.DataFrame(results)

	# 5. Build and write Log
	log_lines: list[str] = []
	try:
		log.add_snakemake_overview(log_lines, snakemake)
	except Exception:
		log_lines.append("SNAKEMAKE: overview not available")

	log.add_dataframe_info(log_lines, "ENES base", N_base, enes_base.shape[1])
	log.add_dataframe_info(log_lines, "ENES 2021", N_2021, enes_2021.shape[1])
	log.add_notes(
		log_lines,
		"Ponderations comparison (Different Size Sample KS-Test)",
		[
			f"Alpha = {alpha}",
			"H0: both samples come from the same distribution.",
			"same_distribution=True means p >= alpha (do not reject H0).",
			"different_distribution=True means p < alpha (reject H0).",
		],
	)

	if skipped_cols:
		log_lines.append(f"Skipped columns in ENES 2021: {', '.join(skipped_cols)}")

	for r in results:
		ks_txt = _fmt_float(r["ks_stat"], ".4f")
		p_txt = _fmt_float(r["pval"], ".4e")
		log_lines.append(
			f"{r['pond_col']}: n_2021={r['n_nonmissing_2021']} / {r['nobs_2021']}, "
			f"n_base={r['n_nonmissing_base']} / {r['nobs_base']}, "
			f"KS-stat={ks_txt}, p={p_txt}, "
			f"same_distribution={r['same_distribution']}, "
			f"different_distribution={r['different_distribution']}"
		)

	if not res_df.empty:
		n_same = int(res_df["same_distribution"].sum())
		n_diff = int(res_df["different_distribution"].sum())
		best_idx = res_df["ks_stat"].astype(float).idxmin()
		best_match = res_df.loc[best_idx, "pond_col"]
		log_lines.append(
			f"Summary: same_distribution={n_same}, different_distribution={n_diff}, "
			f"best_match_by_min_ks={best_match}"
		)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)

	# 6. Plot Single Multi-Panel Grid for Snakemake Output
	passed_tests = (
		res_df[res_df["same_distribution"] == True]
		if not res_df.empty
		else pd.DataFrame()
	)
	plot_df = passed_tests.copy()
	plot_mode = "same_distribution"

	if plot_df.empty and not res_df.empty:
		# Fallback: plot the best matching column (minimum KS statistic)
		plot_df = res_df.sort_values("ks_stat", ascending=True).head(1).copy()
		plot_mode = "best_match_min_ks"

	if plot_df.empty or not f_col_actual:
		# Create a blank plot with error text to satisfy Snakemake's output requirement
		fig, ax = plt.subplots(figsize=(4, 4))
		msg = (
			f"'{f_col}' missing in base file."
			if not f_col_actual
			else "No valid KS comparisons available."
		)
		ax.text(0.5, 0.5, msg, ha="center", va="center", color="red")
		ax.axis("off")
		plt.savefig(output_plot_path)
		plt.close()
		print(f"Plot saved with warning message to {output_plot_path}")
	else:
		# Generate grid
		n_plots = len(plot_df)
		cols = 2
		rows = math.ceil(n_plots / cols)
		fig, axes = plt.subplots(rows, cols, figsize=(5, 5 * rows))
		axes = np.array(axes).flatten()  # Ensure iterable for 1 row

		for i, (idx, row) in enumerate(plot_df.iterrows()):
			ax = axes[i]
			pc = row["pond_col"]
			pc_vals = pd.to_numeric(enes_2021[pc], errors="coerce").dropna()

			sns.histplot(
				pc_vals,
				color="dodgerblue",
				label=f"2021: {pc}",
				stat="density",
				common_norm=False,
				kde=True,
				alpha=0.3,
				edgecolor=None,
				ax=ax,
			)
			sns.histplot(
				base_vals,
				color="darkorange",
				label=f"Base: {f_col_actual}",
				stat="density",
				common_norm=False,
				kde=True,
				alpha=0.3,
				edgecolor=None,
				ax=ax,
			)

			label_suffix = (
				"(same distribution by p-value)"
				if plot_mode == "same_distribution"
				else "(best match by min KS statistic)"
			)
			ax.set_title(
				f"{pc} vs {f_col_actual} {label_suffix}\n"
				f"KS D={_fmt_float(row['ks_stat'], '.4f')}, p={_fmt_float(row['pval'], '.4e')}"
			)
			ax.set_xlabel("Ponderation Weight")
			ax.set_ylabel("Density")
			ax.legend()

		# Hide empty subplots
		for j in range(len(plot_df), len(axes)):
			axes[j].set_visible(False)

		plt.tight_layout()
		plt.savefig(output_plot_path)
		plt.close()


if __name__ == "__main__":
	main()
