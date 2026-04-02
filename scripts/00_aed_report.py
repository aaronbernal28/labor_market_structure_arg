from scripts import *
import pandas as pd

snakemake: any


def main() -> None:
	df_nodelist_caes = pd.read_csv(snakemake.input[1])
	df_nodelist_ciuo = pd.read_csv(snakemake.input[2])
	meta_caes = snakemake.config["nodelist_caes"]
	meta_ciuo = snakemake.config["nodelist_ciuo"]

	pl.plot_top_n_bar(
		df=df_nodelist_caes,
		label_col=meta_caes["label"],
		val_col="n_obs",
		color_col=meta_caes["label_color"],
		title="Top sectors",
		xlabel="Workers",
		output_path=snakemake.output[0],
		save=True,
	)

	pl.plot_top_n_bar(
		df=df_nodelist_ciuo,
		label_col=meta_ciuo["label"],
		val_col="n_obs",
		color_col=meta_ciuo["label_color"],
		title="Top occupations",
		xlabel="Workers",
		output_path=snakemake.output[1],
		save=True,
	)


if __name__ == "__main__":
	main()
