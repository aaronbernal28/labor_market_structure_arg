from snakemake.script import snakemake
from src import fceyn_plot_aed_top_occupations, fceyn_plot_aed_top_sectors
import pandas as pd


def main() -> None:
	df = pd.read_csv(snakemake.input[0])

	fig = fceyn_plot_aed_top_sectors(df, title="Top sectors")
	fig.savefig(snakemake.output[0], bbox_inches="tight")

	fig = fceyn_plot_aed_top_occupations(df, title="Top occupations")
	fig.savefig(snakemake.output[1], bbox_inches="tight")

	pd.DataFrame().to_csv(snakemake.output[2], index=False)


if __name__ == "__main__":
	main()
