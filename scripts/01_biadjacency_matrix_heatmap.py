from scripts import *
import pandas as pd

snakemake: any


def main() -> None:
	enes_df = pd.read_csv(snakemake.input[0])
	caes_df = pd.read_csv(snakemake.input[1])
	ciuo_df = pd.read_csv(snakemake.input[2])

	fig = fceyn_plot_biadjacency_heatmap(
		(enes_df, caes_df, ciuo_df), title="Biadjacency heatmap"
	)
	fig.savefig(snakemake.output[0], bbox_inches="tight")


if __name__ == "__main__":
	main()
