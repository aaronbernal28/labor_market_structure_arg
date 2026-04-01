from snakemake.script import snakemake
from src import fceyn_clean_data_enes
import pandas as pd


def main() -> None:
	dataset = snakemake.params[0]

	SOURCE = snakemake.config["datasets"][dataset]["source"]
	URL = snakemake.config["datasets"][dataset]["url"]
	FEATURES = snakemake.config["datasets"][dataset]["features"]

	df_enes = pd.read_csv(SOURCE if SOURCE else URL)
	df_enes = fceyn_clean_data_enes(
		df_enes=df_enes,
		id_1=snakemake.config["datasets"][dataset]["id_1"],
		id_2=snakemake.config["datasets"][dataset]["id_2"],
		id_caes=snakemake.config["datasets"][dataset]["caes_id"],
		id_ciuo=snakemake.config["datasets"][dataset]["ciuo_id"],
		features=FEATURES,
	)

	if dataset == "enes_2021":
		# For the 2021 survey, we need to rename the columns to match the 2019 survey for CAES and CIUO IDs, as well as the features.
		df_enes = df_enes.rename(
			columns={
				snakemake.config["datasets"][dataset]["caes_id"]: snakemake.config[
					"datasets"
				]["enes_2019"]["caes_id"],
				snakemake.config["datasets"][dataset]["ciuo_id"]: snakemake.config[
					"datasets"
				]["enes_2019"]["ciuo_id"],
				**{
					feature: FEATURES[i]
					for i, feature in enumerate(FEATURES)
					if feature in df_enes.columns
				},
			}
		)

	df_enes.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
