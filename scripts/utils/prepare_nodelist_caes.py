from scripts import *
import pandas as pd

snakemake: any


CHARACTERISTIC_COLUMNS = [
	"n_obs",
	"total_workers_weighted",
	"income_mean",
	"income_min",
	"income_q1",
	"income_median",
	"income_q3",
	"income_max",
	"income_std",
	"female_pct",
	"male_pct",
	"public_sector_pct",
]


def main() -> None:
	df_enes = pd.read_csv(snakemake.input[0])
	nodelist = snakemake.params[0]
	metadata = snakemake.config[nodelist]
	id = metadata["id"]
	max_caes_id = snakemake.config["max_caes_id"]

	if nodelist == "nodelist_caes":
		df_nodelist = dl.load_nodelist_caes(
			caes_path=metadata["source"],
			caes_id=metadata["id"],
			caes_letra_col=metadata["letra"],
			caes_ag_col=metadata["grupo"],
			caes_label_color_col=metadata["label_color"],
			caes_letra_color_col=metadata["letra_color"],
			caes_ag_color_col=metadata["grupo_color"],
		)
	elif nodelist == "nodelist_ciuo":
		df_nodelist = dl.load_nodelist_ciuo(
			ciuo_path=metadata["source"],
			ciuo_id=metadata["id"],
			max_caes_id=max_caes_id,
			ciuo_letra_col=metadata["letra"],
			ciuo_3cat_col=metadata["grupo"],
			ciuo_label_color_col=metadata["label_color"],
			ciuo_letra_color_col=metadata["letra_color"],
			ciuo_3cat_color_col=metadata["grupo_color"],
		)
	else:
		raise ValueError(f"Unsupported nodelist type: {nodelist}")

	new_features = nc.compute_group_characteristics(enes_df=df_enes, col_group=id)
	for col in CHARACTERISTIC_COLUMNS:
		if col not in new_features.columns:
			new_features[col] = pd.NA

	df_nodelist = nc.attach_group_characteristics(
		nodelist_df=df_nodelist,
		features_df=new_features,
		keep_columns=CHARACTERISTIC_COLUMNS,
	)
	df_nodelist = df_nodelist.reset_index()

	df_nodelist.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
	main()
