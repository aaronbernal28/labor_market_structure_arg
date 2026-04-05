from scripts import *
import matplotlib.pyplot as plt
import pandas as pd

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	meta_caes = snakemake.config["caes"]
	meta_ciuo = snakemake.config["ciuo"]
	df_nodelist_caes = pd.read_csv(snakemake.input[1], dtype={meta_caes["id"]: int})
	df_nodelist_ciuo = pd.read_csv(snakemake.input[2], dtype={meta_ciuo["id"]: int})

	pl.plot_top_n_bar(
		df=df_nodelist_caes,
		label_col=meta_caes["label"],
		val_col="n_obs",
		color_col=None,
		title=None,
		xlabel="Workers",
		figsize=snakemake.config["figsizes"]["top_n_bar"],
		output_path=snakemake.output[0],
		save=True,
	)

	pl.plot_top_n_bar(
		df=df_nodelist_ciuo,
		label_col=meta_ciuo["label"],
		val_col="n_obs",
		color_col=None,
		title=None,
		xlabel="Workers",
		figsize=snakemake.config["figsizes"]["top_n_bar"],
		output_path=snakemake.output[1],
		save=True,
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
