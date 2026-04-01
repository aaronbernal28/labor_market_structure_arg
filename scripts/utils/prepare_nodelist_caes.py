import os
import sys

sys.path.insert(0, os.getcwd())

from src import fceyn_attach_group_characteristics, fceyn_compute_group_characteristics
import pandas as pd


def main() -> None:
	df_enes = pd.read_csv(snakemake.input[0])
	nodelist = snakemake.params[0]
	metadata = snakemake.config["metadata"][nodelist]
	df_nodelist = pd.read_csv(metadata["source"])
	id = metadata["id"]

	new_features = fceyn_compute_group_characteristics(df_enes, id)
	df_nodelist = fceyn_attach_group_characteristics(df_nodelist, new_features)

	df_nodelist.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
