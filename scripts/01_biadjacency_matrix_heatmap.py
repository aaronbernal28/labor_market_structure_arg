from scripts import *
import pandas as pd

snakemake: any


def main() -> None:
	enes_df = pd.read_csv(snakemake.input[0])
	caes_df = pd.read_csv(snakemake.input[1])
	ciuo_df = pd.read_csv(snakemake.input[2])
	caes_id = snakemake.config["nodelist_caes"]["id"]
	ciuo_id = snakemake.config["nodelist_ciuo"]["id"]
	letra_caes = snakemake.config["nodelist_caes"]["letra"]
	letra_ciuo = snakemake.config["nodelist_ciuo"]["letra"]

	enes_df = pd.merge(
		enes_df,
		caes_df[[caes_id, letra_caes]],
		left_on=snakemake.config["id_caes"],
		right_on=caes_id,
		how="left",
	)
	enes_df = pd.merge(
		enes_df,
		ciuo_df[[ciuo_id, letra_ciuo]],
		left_on=snakemake.config["id_ciuo"],
		right_on=ciuo_id,
		how="left",
	)

	biadjacency = gc.build_biadjacency(
		enes_df,
		letra_caes,
		letra_ciuo,
		logscale=snakemake.config["logscale"],
	)

	pl.plot_heatmap(biadjacency, output_path=snakemake.output[0], save=True)


if __name__ == "__main__":
	main()
