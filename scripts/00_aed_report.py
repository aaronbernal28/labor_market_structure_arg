import os
import sys

sys.path.insert(0, os.getcwd())

from src import fceyn_plot_aed_top_occupations, fceyn_plot_aed_top_sectors
import pandas as pd


def main() -> None:
	df_enes = pd.read_csv(snakemake.input[0])
	df_nodelist_caes = pd.read_csv(snakemake.input[1])
	df_nodelist_ciuo = pd.read_csv(snakemake.input[2])

	caes_id = snakemake.config["datasets"]["enes_2019"]["caes_id"]
	ciuo_id = snakemake.config["datasets"]["enes_2019"]["ciuo_id"]
	caes_meta_id = snakemake.config["metadata"]["nodelist_caes"]["id"]
	ciuo_meta_id = snakemake.config["metadata"]["nodelist_ciuo"]["id"]

	caes_counts = (
		df_enes[caes_id].value_counts().rename("count").reset_index()
	)
	caes_counts.columns = [caes_id, "count"]

	df_nodelist_caes = pd.merge(
		df_nodelist_caes,
		caes_counts,
		left_on=caes_meta_id,
		right_on=caes_id,
		how="left",
	).fillna(0)

	fig = fceyn_plot_aed_top_sectors(df_nodelist_caes, title="Top sectors")
	fig.savefig(snakemake.output[0], bbox_inches="tight")

	ciuo_counts = (
		df_enes[ciuo_id].value_counts().rename("count").reset_index()
	)
	ciuo_counts.columns = [ciuo_id, "count"]
	df_nodelist_ciuo = pd.merge(
		df_nodelist_ciuo,
		ciuo_counts,
		left_on=ciuo_meta_id,
		right_on=ciuo_id,
		how="left",
	).fillna(0)

	fig = fceyn_plot_aed_top_occupations(df_nodelist_ciuo, title="Top occupations")
	fig.savefig(snakemake.output[1], bbox_inches="tight")


if __name__ == "__main__":
	main()
