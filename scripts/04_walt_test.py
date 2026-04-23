from scripts import *
import matplotlib.pyplot as plt
import pandas as pd

snakemake: any


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
	test_results = dl.wald_test_comparison_proportions(
		df_2019,
		df_2021,
		caes_col,
		ciuo_col,
		rownames,
		colnames,
		alpha=alpha,
	)

	bootstrap_B = 1000
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
	rownames_plot = [label.split(".")[0] for label in rownames]
	colnames_plot = [label.split(".")[0] for label in colnames]

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

	rejected_count = int(test_results["rejected"].sum())
	if rejected_count > 0:
		rejected_df = pvalue_detailed.loc[
			pvalue_detailed["rejected"],
			[caes_col, ciuo_col, "delta_hat", "se_boot", "p_value", "sig"],
		].sort_values([caes_col, ciuo_col], ascending=True, kind="mergesort")

		log_lines.append("")
		log_lines.append("REJECTED PAIRS (NULL HYPOTHESIS REJECTED):")
		log_lines.append(rejected_df.to_string(index=False))
		log_lines.append("")
		log_lines.append(
			"Note: Significance stars indicate levels of significance after Bonferroni correction."
		)
	else:
		log_lines.append("")
		log_lines.append("No pairs rejected (all p-values above Bonferroni threshold).")

	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
