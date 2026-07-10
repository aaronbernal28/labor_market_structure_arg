from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

snakemake: Any


def weighted_wald_test_comparison_proportions(
	df1: pd.DataFrame,
	df2: pd.DataFrame,
	caes_col: str,
	ciuo_col: str,
	weight_col: str,
	rownames: list[str],
	colnames: list[str],
	alpha: float = 0.05,
) -> dict:
	n1 = df1[weight_col].sum()
	n2 = df2[weight_col].sum()

	print(f"Weighted sample sizes: n1={n1:.2f}, n2={n2:.2f}")

	crosstab1 = pd.crosstab(
		df1[caes_col], df1[ciuo_col], values=df1[weight_col], aggfunc="sum"
	).fillna(0)
	crosstab2 = pd.crosstab(
		df2[caes_col], df2[ciuo_col], values=df2[weight_col], aggfunc="sum"
	).fillna(0)

	crosstab1 = crosstab1.reindex(index=rownames, columns=colnames, fill_value=0)
	crosstab2 = crosstab2.reindex(index=rownames, columns=colnames, fill_value=0)

	counts1 = crosstab1.to_numpy().astype(float)
	counts2 = crosstab2.to_numpy().astype(float)

	p1 = counts1 / n1
	p2 = counts2 / n2

	delta_hat = p1 - p2

	w2_1 = df1[weight_col] ** 2
	df1_w2 = df1.assign(**{weight_col: w2_1})
	crosstab1_w2 = pd.crosstab(
		df1_w2[caes_col], df1_w2[ciuo_col], values=df1_w2[weight_col], aggfunc="sum"
	).fillna(0)
	crosstab1_w2 = crosstab1_w2.reindex(index=rownames, columns=colnames, fill_value=0)
	w2_cell_1 = crosstab1_w2.to_numpy().astype(float)
	w2_tot_1 = w2_1.sum()

	var1 = ((1 - p1) ** 2 * w2_cell_1 + p1**2 * (w2_tot_1 - w2_cell_1)) / (n1**2)

	w2_2 = df2[weight_col] ** 2
	df2_w2 = df2.assign(**{weight_col: w2_2})
	crosstab2_w2 = pd.crosstab(
		df2_w2[caes_col], df2_w2[ciuo_col], values=df2_w2[weight_col], aggfunc="sum"
	).fillna(0)
	crosstab2_w2 = crosstab2_w2.reindex(index=rownames, columns=colnames, fill_value=0)
	w2_cell_2 = crosstab2_w2.to_numpy().astype(float)
	w2_tot_2 = w2_2.sum()

	var2 = ((1 - p2) ** 2 * w2_cell_2 + p2**2 * (w2_tot_2 - w2_cell_2)) / (n2**2)

	se = np.sqrt(var1 + var2)
	se = np.where(se > 0, se, 1e-10)

	W = delta_hat / se
	p_values = 2 * (1 - stats.norm.cdf(np.abs(W)))

	d = delta_hat.size
	bonferroni_threshold = alpha / d
	rejected = p_values < bonferroni_threshold

	print("\n=== Wald Test Results (Weighted Proportions) ===")
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


def weighted_bootstrap_se(
	df1: pd.DataFrame,
	df2: pd.DataFrame,
	caes_col: str,
	ciuo_col: str,
	weight_col: str,
	rownames: list[str],
	colnames: list[str],
	B: int = 1000,
	seed: int = 28,
) -> np.ndarray:
	deltas = np.zeros((B, len(rownames), len(colnames)))

	prob1 = df1[weight_col] / df1[weight_col].sum()
	prob2 = df2[weight_col] / df2[weight_col].sum()

	n1 = len(df1)
	n2 = len(df2)

	for b in range(B):
		s1 = df1.sample(n=n1, replace=True, weights=prob1, random_state=seed + 2 * b)
		s2 = df2.sample(
			n=n2, replace=True, weights=prob2, random_state=seed + 2 * b + 1
		)

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

		deltas[b] = ct1 / n1 - ct2 / n2

	return deltas.std(axis=0, ddof=1)


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	df_2019 = pd.read_csv(
		snakemake.input[0],
		dtype={
			snakemake.config["caes"]["id"]: int,
			snakemake.config["ciuo"]["id"]: int,
		},
	)
	df_2021 = pd.read_csv(
		snakemake.input[1],
		dtype={
			snakemake.config["caes"]["id"]: int,
			snakemake.config["ciuo"]["id"]: int,
		},
	)
	nodelist_caes = pd.read_csv(
		snakemake.input[2], dtype={snakemake.config["caes"]["id"]: int}
	)
	nodelist_ciuo = pd.read_csv(
		snakemake.input[3], dtype={snakemake.config["ciuo"]["id"]: int}
	)

	df_2019 = df_2019.merge(
		nodelist_caes,
		left_on=snakemake.config["caes"]["id"],
		right_on=snakemake.config["caes"]["id"],
		how="left",
	)
	df_2019 = df_2019.merge(
		nodelist_ciuo,
		left_on=snakemake.config["ciuo"]["id"],
		right_on=snakemake.config["ciuo"]["id"],
		how="left",
	)
	df_2021 = df_2021.merge(
		nodelist_caes,
		left_on=snakemake.config["caes"]["id"],
		right_on=snakemake.config["caes"]["id"],
		how="left",
	)
	df_2021 = df_2021.merge(
		nodelist_ciuo,
		left_on=snakemake.config["ciuo"]["id"],
		right_on=snakemake.config["ciuo"]["id"],
		how="left",
	)

	caes_col = "caesletra"
	ciuo_col = "ciuo1diglabel"

	required_cols = [caes_col, ciuo_col]
	for col in required_cols:
		if col not in df_2019.columns:
			raise KeyError(f"Column '{col}' not found in {snakemake.input[0]}.")
		if col not in df_2021.columns:
			raise KeyError(f"Column '{col}' not found in {snakemake.input[1]}.")

	rownames = sorted(
		set(df_2019[caes_col].dropna().astype(str))
		| set(df_2021[caes_col].dropna().astype(str))
	)

	colnames = sorted(
		set(df_2019[ciuo_col].dropna().astype(str))
		| set(df_2021[ciuo_col].dropna().astype(str))
	)

	alpha = 0.05
	bootstrap_B = 1000

	if "ponderation" in df_2019.columns and "ponderation" in df_2021.columns:
		print("Using weighted proportions and weighted bootstrap (ponderation found).")
		test_results = weighted_wald_test_comparison_proportions(
			df_2019,
			df_2021,
			caes_col,
			ciuo_col,
			"ponderation",
			rownames,
			colnames,
			alpha=alpha,
		)
		se_boot = weighted_bootstrap_se(
			df_2019,
			df_2021,
			caes_col,
			ciuo_col,
			"ponderation",
			rownames,
			colnames,
			B=bootstrap_B,
		)
	else:
		print(
			"Using unweighted proportions and unweighted bootstrap (ponderation missing)."
		)
		test_results = dl.wald_test_comparison_proportions(
			df_2019,
			df_2021,
			caes_col,
			ciuo_col,
			rownames,
			colnames,
			alpha=alpha,
		)
		se_boot = dl.bootstrap_se(
			df_2019,
			df_2021,
			caes_col,
			ciuo_col,
			rownames,
			colnames,
			B=bootstrap_B,
		)

	# Extract first digit labels for better readability in plots
	rownames_plot = rownames  # [label.split(".")[0] for label in rownames]
	colnames_plot = colnames  # [label.split(".")[0] for label in colnames]

	pl.plot_bootstrap_se_heatmap(
		se_boot,
		rownames_plot,
		colnames_plot,
		snakemake.output[0],
		save=True,
		figsize=tuple(snakemake.config["figsizes"]["heatmap"]),
	)
	pl.plot_delta_heatmap(
		test_results["delta_hat"],
		rownames_plot,
		colnames_plot,
		snakemake.output[1],
		save=True,
		figsize=tuple(snakemake.config["figsizes"]["heatmap"]),
		logscale=True,
		logscale_vmin=1e-3,
	)
	pl.plot_rejection_heatmap(
		test_results["p_values"],
		test_results["rejected"],
		rownames_plot,
		colnames_plot,
		test_results["bonferroni_threshold"],
		snakemake.output[2],
		save=True,
		figsize=tuple(snakemake.config["figsizes"]["heatmap"]),
	)

	rows = []
	for i, caes_label in enumerate(rownames):
		for j, ciuo_label in enumerate(colnames):
			rows.append(
				{
					caes_col: str(caes_label),
					ciuo_col: str(ciuo_label),
					"p_value": float(test_results["p_values"][i, j]),
					"rejected": bool(test_results["rejected"][i, j]),
					"delta_hat": float(test_results["delta_hat"][i, j]),
					"se": float(test_results["se"][i, j]),
					"se_boot": float(se_boot[i, j]),
					"wald_stat": float(test_results["W"][i, j]),
					"p1": float(test_results["p1"][i, j]),
					"p2": float(test_results["p2"][i, j]),
					"count_2019": float(test_results["counts1"][i, j]),
					"count_2021": float(test_results["counts2"][i, j]),
					"bonferroni_threshold": float(test_results["bonferroni_threshold"]),
				}
			)

	pvalue_detailed = pd.DataFrame(rows).sort_values("p_value", ascending=True)

	total_tests = len(rownames) * len(colnames)

	def _sig_stars(p_value: float) -> str:
		if p_value < 0.001 / total_tests:
			return "***"
		if p_value < 0.01 / total_tests:
			return "**"
		if p_value < 0.05 / total_tests:
			return "*"
		return ""

	pvalue_detailed["sig"] = pvalue_detailed["p_value"].map(_sig_stars)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("WALD TEST SUMMARY")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_notes(
		log_lines,
		"DATA OVERVIEW",
		[
			f"Input 2019: {snakemake.input[0]}",
			f"Input 2021: {snakemake.input[1]}",
			f"Rows 2019: {len(df_2019)}",
			f"Rows 2021: {len(df_2021)}",
			f"CAES categories: {len(rownames)}",
			f"CIUO categories: {len(colnames)}",
			f"Total tests: {total_tests}",
		],
	)
	log.add_notes(
		log_lines,
		"PARAMETERS",
		[
			f"Alpha: {alpha}",
			f"Bonferroni threshold: {test_results['bonferroni_threshold']:.2e}",
			f"Rejected pairs: {int(test_results['rejected'].sum())} ({100 * test_results['rejected'].mean():.2f}%)",
			"Significance stars (Bonferroni-style): *** p < 0.001/d, ** p < 0.01/d, * p < 0.05/d",
			f"Bootstrap B: {bootstrap_B}",
		],
	)

	top5_df = pvalue_detailed.head(5).sort_values("p_value", ascending=True)

	latex_lines = [
		r"\begin{tabular}{llrrrl}",
		r"\toprule",
		r"\textbf{\gls{caes}} & \textbf{\gls{ciuo}} & \textbf{$\hat{\delta}$} &",
		r"\textbf{SE boot} & \textbf{p-valor} & \textbf{sig.} \\",
		r"\midrule",
	]

	for _, row in top5_df.iterrows():
		latex_lines.append(f"{row[caes_col]} &")
		latex_lines.append(
			f"{row[ciuo_col]} & {row['delta_hat']:.6f} & {row['se_boot']:.6f} & {row['p_value']:e} & {row['sig']} \\\\"
		)

	latex_lines.append(r"\bottomrule")
	latex_lines.append(r"\end{tabular}%")

	log_lines.append("")
	log_lines.append("TOP 5 LOWEST P-VALUES (LATEX):")
	log_lines.extend(latex_lines)
	log_lines.append("")
	log_lines.append(
		"Note: Significance stars indicate levels of significance after Bonferroni correction."
	)

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
