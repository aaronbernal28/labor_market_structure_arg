from scripts import *
import matplotlib.pyplot as plt
import pandas as pd

snakemake: any


def main() -> None:
	plt.style.use("src/styles/publication.mplstyle")
	enes_df = pd.read_csv(
		snakemake.input[0],
		dtype={snakemake.config["id_caes"]: int, snakemake.config["id_ciuo"]: int},
	)
	caes_id = snakemake.config["caes"]["id"]
	ciuo_id = snakemake.config["ciuo"]["id"]
	caes_df = pd.read_csv(snakemake.input[1], dtype={caes_id: int})
	ciuo_df = pd.read_csv(snakemake.input[2], dtype={ciuo_id: int})
	letra_caes = snakemake.config["caes"]["letra"]
	letra_ciuo = snakemake.config["ciuo"]["letra"]

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
		logscale=False,
	)

	pl.plot_heatmap(
		biadjacency,
		output_path=snakemake.output[0],
		save=True,
		figsize=tuple(snakemake.config["figsizes"]["heatmap"]),
	)


if __name__ == "__main__":
	main()
