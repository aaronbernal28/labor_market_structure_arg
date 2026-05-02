from typing import Any
from scripts import *
import matplotlib.pyplot as plt
import pandas as pd

snakemake: Any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	meta_caes = snakemake.config["caes"]
	meta_ciuo = snakemake.config["ciuo"]
	df_nodelist_caes = pd.read_csv(snakemake.input[1], dtype={meta_caes["id"]: int})
	df_nodelist_ciuo = pd.read_csv(snakemake.input[2], dtype={meta_ciuo["id"]: int})
	df_individual = pd.read_csv(snakemake.input[0])

	pl.plot_top_n_bar(
		df=df_nodelist_caes,
		label_col=meta_caes["label"],
		val_col="n_obs",
		color_col="",
		title="",
		xlabel="Workers",
		figsize=snakemake.config["figsizes"]["top_n_bar"],
		output_path=snakemake.output[0],
		save=True,
	)

	pl.plot_top_n_bar(
		df=df_nodelist_ciuo,
		label_col=meta_ciuo["label"],
		val_col="n_obs",
		color_col="",
		title="",
		xlabel="Workers",
		figsize=snakemake.config["figsizes"]["top_n_bar"],
		output_path=snakemake.output[1],
		save=True,
	)

	dataset_name = snakemake.wildcards.dataset
	features_to_plot = ["total_income", "age", "nivel_ed"]
	features_for_corr = features_to_plot + ["sex_id", "public_worker"]
	calib_col = "ponderation" if "ponderation" in df_individual.columns else None

	pl.plot_weighted_histograms(
		df=df_individual,
		features=features_to_plot,
		calib_col=calib_col,
		title=f"Feature Distributions ({dataset_name})",
		output_path=snakemake.output[2],
	)

	pl.plot_correlation_matrix(
		df=df_individual,
		features=features_for_corr,
		calib_col=calib_col,
		title=f"Correlation Matrix ({dataset_name})",
		output_path=snakemake.output[3],
	)

	log_lines: list[str] = []
	log_lines.append("=" * 60)
	log_lines.append("AED REPORT")
	log_lines.append("=" * 60)
	log.add_snakemake_overview(log_lines, snakemake)
	log.add_dataframe_info(
		log_lines,
		"CAES NODELIST",
		row_count=len(df_nodelist_caes),
		column_count=len(df_nodelist_caes.columns),
	)
	log.add_dataframe_info(
		log_lines,
		"CIUO NODELIST",
		row_count=len(df_nodelist_ciuo),
		column_count=len(df_nodelist_ciuo.columns),
	)
	log_path = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None
	log.write_log(log_lines, log_path)


if __name__ == "__main__":
	main()
