from scripts import *
import pandas as pd


def main() -> None:
	df = pd.read_csv(snakemake.input[0])

	fig = plot_aed_top_sectors(df, title="Top sectors")
	fig.savefig(snakemake.output[0], bbox_inches="tight")

	fig = plot_aed_top_occupations(df, title="Top occupations")
	fig.savefig(snakemake.output[1], bbox_inches="tight")

	pd.DataFrame().to_csv(snakemake.output[2], index=False)


if __name__ == "__main__":
	main()
