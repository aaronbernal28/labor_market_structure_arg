from scripts import *
import pandas as pd


def main() -> None:
	df_enes = pd.read_csv(snakemake.input[0])
	nodelist = snakemake.params[0]
	df_nodelist = pd.read_csv(snakemake.config["datasets"][nodelist]["source"])
	id = snakemake.config["datasets"][nodelist]["id"]

	new_features = fceyn_compute_group_characteristics(df_enes, id)
	df_nodelist = fceyn_attach_group_characteristics(df_nodelist, new_features)

	df_nodelist.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
