from scripts import *
import pandas as pd


def main() -> None:
	enes_df = pd.read_csv(snakemake.input[0])
	caes_df = pd.read_csv(snakemake.input[1])
	ciuo_df = pd.read_csv(snakemake.input[2])

	fig = fceyn_plot_sankey((enes_df, caes_df, ciuo_df), title="Sankey plot")
	fig.savefig(snakemake.output[0], bbox_inches="tight")


if __name__ == "__main__":
	main()
